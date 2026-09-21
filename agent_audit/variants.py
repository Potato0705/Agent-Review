"""Variant strategies and the postconditions that make them defensible.

Each strategy states a claim about the text it produced, and that claim is
checked before the variant is returned. A gaming variant that quietly removed
content, or a degradation variant that left the evidence in place, would be
mislabelled; every conclusion drawn from a mislabelled variant is unfounded,
so the generator raises instead of emitting one.

The strategies deliberately produce clean, obvious interventions. That makes
them a floor test: a grader failing them has a real problem, while a grader
passing them has only shown it resists the easy cases. Reports say so through
the ``--variant-origin`` declaration.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .io import stable_hash
from .models import BaselineCase
from .segmentation import CHINESE, LanguageStrategy


GAMING_STRATEGIES = ("verbose_padding", "rubric_flattery")
DEGRADATION_STRATEGIES = ("remove_evidence", "unsupported_assertion")
PARAPHRASE_STRATEGIES = ("connective_substitution",)

PADDING_SENTENCE_COUNT = 2


class VariantPostconditionError(ValueError):
    """Raised when a generated variant does not match its own label."""


@dataclass(frozen=True)
class GeneratedVariant:
    case_id: str
    variant_id: str
    variant_type: str
    strategy: str
    text: str
    inserted_text: tuple[str, ...] = ()
    magnitude: dict[str, Any] = field(default_factory=dict)


def _assert_only_added(baseline: str, produced: str, strategy: str) -> None:
    if not produced.startswith(baseline) or len(produced) <= len(baseline):
        raise VariantPostconditionError(
            f"{strategy} must only add text: the result no longer starts with the "
            "baseline or is not longer than it."
        )


def _assert_evidence_removed(
    case: BaselineCase,
    produced: str,
    kept: tuple[str, ...],
    strategy: str,
    language: LanguageStrategy,
) -> None:
    sentences = language.split(case.text)
    for index in case.evidence_sentences:
        annotated = sentences[index - 1]
        if annotated in produced:
            raise VariantPostconditionError(
                f"{strategy} still contains annotated sentence {index}; the result "
                "is not a degradation of the baseline."
            )
    remaining = produced
    for sentence in kept:
        # Compare on the trimmed sentence: the joiner normalises whitespace at
        # the seams, and that is formatting rather than content. Order and
        # presence are still checked exactly.
        content = sentence.strip()
        if not content:
            continue
        position = remaining.find(content)
        if position < 0:
            raise VariantPostconditionError(
                f"{strategy} dropped or reordered a sentence it should have kept."
            )
        remaining = remaining[position + len(content) :]


def _kept_sentences(
    case: BaselineCase, language: LanguageStrategy = CHINESE
) -> tuple[str, ...]:
    sentences = language.split(case.text)
    annotated = set(case.evidence_sentences)
    return tuple(
        sentence
        for index, sentence in enumerate(sentences, start=1)
        if index not in annotated
    )


def _verbose_padding(
    case: BaselineCase, language: LanguageStrategy, rng: random.Random
) -> GeneratedVariant:
    chosen = tuple(
        rng.sample(
            list(language.padding_sentences),
            min(PADDING_SENTENCE_COUNT, len(language.padding_sentences)),
        )
    )
    separator = language.sentence_separator
    produced = case.text + "".join(separator + sentence for sentence in chosen)
    _assert_only_added(case.text, produced, "verbose_padding")
    return GeneratedVariant(
        case_id=case.case_id,
        variant_id="gaming_verbose_padding",
        variant_type="gaming",
        strategy="verbose_padding",
        text=produced,
        inserted_text=chosen,
        magnitude={
            "added_sentences": len(chosen),
            "added_characters": len(produced) - len(case.text),
        },
    )


def _rubric_flattery(
    case: BaselineCase, language: LanguageStrategy, rng: random.Random
) -> GeneratedVariant:
    produced = case.text + language.sentence_separator + language.flattery_sentence
    _assert_only_added(case.text, produced, "rubric_flattery")
    return GeneratedVariant(
        case_id=case.case_id,
        variant_id="gaming_rubric_flattery",
        variant_type="gaming",
        strategy="rubric_flattery",
        text=produced,
        inserted_text=(language.flattery_sentence,),
        magnitude={
            "added_sentences": 1,
            "added_characters": len(produced) - len(case.text),
        },
    )


def _remove_evidence(
    case: BaselineCase, language: LanguageStrategy, rng: random.Random
) -> GeneratedVariant:
    kept = _kept_sentences(case, language)
    produced = "".join(kept).strip()
    if len(produced) >= len(case.text):
        raise VariantPostconditionError(
            "remove_evidence did not shorten the baseline."
        )
    _assert_evidence_removed(case, produced, kept, "remove_evidence", language)
    removed = len(case.text) - len(produced)
    return GeneratedVariant(
        case_id=case.case_id,
        variant_id="degradation_remove_evidence",
        variant_type="degradation",
        strategy="remove_evidence",
        text=produced,
        magnitude={
            "removed_sentences": len(case.evidence_sentences),
            "removed_characters": removed,
            "removed_share": removed / len(case.text),
        },
    )


def _unsupported_assertion(
    case: BaselineCase, language: LanguageStrategy, rng: random.Random
) -> GeneratedVariant:
    kept = _kept_sentences(case, language)
    assertion = rng.choice(list(language.unsupported_assertions))
    produced = "".join(kept).strip() + language.sentence_separator + assertion
    _assert_evidence_removed(case, produced, kept, "unsupported_assertion", language)
    # An empty assertion would pass a bare `in` check, because every string
    # contains the empty string, and the variant would silently be a plain
    # deletion wearing the wrong label.
    if not assertion or assertion not in produced:
        raise VariantPostconditionError(
            "unsupported_assertion did not insert its replacement claim."
        )
    removed = len(case.text) - len("".join(kept))
    return GeneratedVariant(
        case_id=case.case_id,
        variant_id="degradation_unsupported_assertion",
        variant_type="degradation",
        strategy="unsupported_assertion",
        text=produced,
        inserted_text=(assertion,),
        magnitude={
            "removed_sentences": len(case.evidence_sentences),
            "removed_characters": removed,
            "removed_share": removed / len(case.text),
        },
    )


CLAUSE_BOUNDARIES = "，,。！？；!?;：: \t\n"


def _substitute_at_clause_boundaries(
    text: str,
    connectives: tuple[tuple[str, str], ...],
    *,
    require_word_boundaries: bool = False,
) -> tuple[str, int]:
    """Replace connectives only where a clause actually begins.

    A plain ``str.replace`` is unsafe in Chinese because words overlap: in
    「原因此外还有」 the characters 因此 span 原因 and 此外, and replacing them
    produces text that is not Chinese. Discourse connectives introduce a
    clause, so a match is only taken at the start of the text or straight
    after a clause boundary.

    English needs one extra rule: the match must also end at a word boundary,
    or "final" would be replaced inside "finalise". Chinese must not use that
    rule, because Chinese characters are alphabetic to :meth:`str.isalpha`
    and it would reject every legitimate match.
    """

    pieces: list[str] = []
    index = 0
    replaced = 0
    length = len(text)
    while index < length:
        at_boundary = index == 0 or text[index - 1] in CLAUSE_BOUNDARIES
        match = None
        if at_boundary:
            for source, target in connectives:
                if not text.startswith(source, index):
                    continue
                if require_word_boundaries:
                    following = text[index + len(source) : index + len(source) + 1]
                    if following.isalpha():
                        continue
                match = (source, target)
                break
        if match is None:
            pieces.append(text[index])
            index += 1
            continue
        source, target = match
        pieces.append(target)
        index += len(source)
        replaced += 1
    return "".join(pieces), replaced


def _connective_substitution(
    case: BaselineCase, language: LanguageStrategy, rng: random.Random
) -> GeneratedVariant:
    produced, replaced = _substitute_at_clause_boundaries(
        case.text,
        language.connectives,
        require_word_boundaries=language.requires_word_boundaries,
    )
    if not replaced or produced == case.text:
        raise VariantPostconditionError(
            "connective_substitution found no connective to replace; this text "
            "cannot be paraphrased by the conservative rule."
        )

    stripped_baseline = case.text
    stripped_produced = produced
    for source, target in language.connectives:
        stripped_baseline = stripped_baseline.replace(source, "").replace(target, "")
        stripped_produced = stripped_produced.replace(source, "").replace(target, "")
    if stripped_baseline != stripped_produced:
        raise VariantPostconditionError(
            "connective_substitution changed text outside the connective table."
        )

    return GeneratedVariant(
        case_id=case.case_id,
        variant_id="paraphrase_connective_substitution",
        variant_type="paraphrase",
        strategy="connective_substitution",
        text=produced,
        magnitude={"replaced_connectives": replaced},
    )


_BUILDERS = {
    "verbose_padding": _verbose_padding,
    "rubric_flattery": _rubric_flattery,
    "remove_evidence": _remove_evidence,
    "unsupported_assertion": _unsupported_assertion,
    "connective_substitution": _connective_substitution,
}


def build_variant(
    case: BaselineCase,
    strategy: str,
    language: LanguageStrategy,
    rng: random.Random,
) -> GeneratedVariant:
    """Build one variant, refusing to return anything that fails its own claim."""

    try:
        builder = _BUILDERS[strategy]
    except KeyError as exc:
        raise ValueError(
            f"unknown variant strategy {strategy!r}; expected one of "
            f"{sorted(_BUILDERS)}."
        ) from exc
    return builder(case, language, rng)


@dataclass(frozen=True)
class GeneratedRow:
    case_id: str
    variant_id: str
    variant_type: str
    text: str
    notes: str


@dataclass(frozen=True)
class GenerationRun:
    rows: tuple[GeneratedRow, ...]
    manifest: dict[str, Any]


def _validate_family(
    selected: tuple[str, ...], allowed: tuple[str, ...], family: str
) -> None:
    if not selected:
        raise ValueError(
            f"Variant generation needs at least one {family} strategy; "
            f"`load_scoring_cases` requires every case to have one."
        )
    for name in selected:
        if name not in allowed:
            raise ValueError(
                f"{name!r} is not a {family} strategy; expected one of "
                f"{list(allowed)}."
            )


def _compose_notes(parts: list[str], source_note: str) -> str:
    if source_note:
        parts.append(f"source_note={source_note}")
    return " | ".join(parts)


def _magnitude_summary(magnitude: dict[str, Any]) -> str:
    return "; ".join(
        f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in sorted(magnitude.items())
    )


def generate_variants(
    cases: list[BaselineCase],
    *,
    seed: int = 0,
    gaming: tuple[str, ...] = GAMING_STRATEGIES,
    degradation: tuple[str, ...] = DEGRADATION_STRATEGIES,
    paraphrase: tuple[str, ...] = (),
    language: LanguageStrategy = CHINESE,
) -> GenerationRun:
    """Build every requested variant, refusing rather than guessing.

    Paraphrase is off by default. A conservative connective swap almost always
    scores the same as its baseline, so including it would add a variant that
    passes by construction and dilute the violation rate.
    """

    if not cases:
        raise ValueError("At least one baseline case is required.")
    _validate_family(gaming, GAMING_STRATEGIES, "gaming")
    _validate_family(degradation, DEGRADATION_STRATEGIES, "degradation")
    for name in paraphrase:
        if name not in PARAPHRASE_STRATEGIES:
            raise ValueError(
                f"{name!r} is not a paraphrase strategy; expected one of "
                f"{list(PARAPHRASE_STRATEGIES)}."
            )

    seen: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            raise ValueError(f"Baseline cases contain duplicate case_id {case.case_id!r}.")
        seen.add(case.case_id)

    rng = random.Random(seed)
    rows: list[GeneratedRow] = []
    records: list[dict[str, Any]] = []

    for case in cases:
        rows.append(
            GeneratedRow(
                case_id=case.case_id,
                variant_id="baseline",
                variant_type="baseline",
                text=case.text,
                notes=_compose_notes(["generated=baseline"], case.notes),
            )
        )
        for strategy in (*gaming, *degradation, *paraphrase):
            try:
                variant = build_variant(case, strategy, language, rng)
            except VariantPostconditionError as exc:
                raise ValueError(
                    f"Case {case.case_id!r} cannot produce a {strategy!r} variant: {exc}"
                ) from exc

            parts = [f"generated={strategy}"]
            summary = _magnitude_summary(variant.magnitude)
            if summary:
                parts.append(summary)
            if variant.variant_type == "paraphrase":
                # The surface change is tiny, so this is an invariance floor
                # rather than a real equivalence rewrite. Say so in the row.
                parts.append("weak_probe=surface_only")
            rows.append(
                GeneratedRow(
                    case_id=case.case_id,
                    variant_id=variant.variant_id,
                    variant_type=variant.variant_type,
                    text=variant.text,
                    notes=_compose_notes(parts, case.notes),
                )
            )
            records.append(
                {
                    "case_id": case.case_id,
                    "variant_id": variant.variant_id,
                    "variant_type": variant.variant_type,
                    "strategy": strategy,
                    "magnitude": dict(variant.magnitude),
                    "inserted_text": list(variant.inserted_text),
                }
            )

    # Fingerprint exactly what `score` will hash when it loads the CSV, so an
    # audit can prove the scored cases are the ones this run produced.
    output_sha256 = stable_hash(
        [
            {
                "case_id": row.case_id,
                "variant_id": row.variant_id,
                "variant_type": row.variant_type,
                "text": row.text,
                "notes": row.notes,
            }
            for row in rows
        ]
    )

    manifest: dict[str, Any] = {
        "generator": "agent-review",
        "output_sha256": output_sha256,
        "language": language.name,
        "seed": seed,
        "case_count": len(cases),
        "row_count": len(rows),
        "gaming_strategies": list(gaming),
        "degradation_strategies": list(degradation),
        "paraphrase_strategies": list(paraphrase),
        "requires_human_review": True,
        "variants": records,
    }
    return GenerationRun(rows=tuple(rows), manifest=manifest)


Identity = tuple[str, str]


@dataclass(frozen=True)
class MergeOutcome:
    """The result of merging a freshly generated set into an existing file."""

    rows: tuple[GeneratedRow, ...]
    preserved: tuple[Identity, ...]
    appended: tuple[Identity, ...]
    edited: tuple[Identity, ...]
    foreign: tuple[Identity, ...]

    @property
    def set_origin(self) -> str:
        """Whether the merged set can still be called machine-generated.

        One hand-edited row or one hand-written variant is enough to make the
        whole set mixed. Without this, appending would launder hand-written
        content into a fingerprint the audit would happily verify.
        """

        return "mixed" if self.edited or self.foreign else "machine-generated"


def merge_into_existing(
    generated: tuple[GeneratedRow, ...], existing: list[Any]
) -> MergeOutcome:
    """Keep every existing row, add the missing ones, and classify each one.

    ``existing`` holds objects with ``case_id``, ``variant_id``,
    ``variant_type``, ``text`` and ``notes`` — the shape
    ``load_scoring_cases`` returns.
    """

    by_identity = {(row.case_id, row.variant_id): row for row in existing}
    if len(by_identity) != len(existing):
        raise ValueError("The existing case file contains duplicate case/variant ids.")

    rows: list[GeneratedRow] = []
    preserved: list[Identity] = []
    appended: list[Identity] = []
    edited: list[Identity] = []
    covered: set[Identity] = set()

    for row in generated:
        identity = (row.case_id, row.variant_id)
        covered.add(identity)
        current = by_identity.get(identity)
        if current is None:
            rows.append(row)
            appended.append(identity)
            continue
        rows.append(
            GeneratedRow(
                case_id=current.case_id,
                variant_id=current.variant_id,
                variant_type=current.variant_type,
                text=current.text,
                notes=current.notes,
            )
        )
        # Notes are hashed alongside the text, so a note-only edit changes the
        # set just as much as a rewritten variant does.
        if current.text == row.text and current.notes == row.notes:
            preserved.append(identity)
        else:
            edited.append(identity)

    foreign: list[Identity] = []
    for row in existing:
        identity = (row.case_id, row.variant_id)
        if identity in covered:
            continue
        foreign.append(identity)
        rows.append(
            GeneratedRow(
                case_id=row.case_id,
                variant_id=row.variant_id,
                variant_type=row.variant_type,
                text=row.text,
                notes=row.notes,
            )
        )

    return MergeOutcome(
        rows=tuple(rows),
        preserved=tuple(preserved),
        appended=tuple(appended),
        edited=tuple(edited),
        foreign=tuple(foreign),
    )
