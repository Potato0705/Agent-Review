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

from .models import BaselineCase
from .segmentation import LanguageStrategy, split_sentences


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
    case: BaselineCase, produced: str, kept: tuple[str, ...], strategy: str
) -> None:
    sentences = split_sentences(case.text)
    for index in case.evidence_sentences:
        annotated = sentences[index - 1]
        if annotated in produced:
            raise VariantPostconditionError(
                f"{strategy} still contains annotated sentence {index}; the result "
                "is not a degradation of the baseline."
            )
    remaining = produced
    for sentence in kept:
        position = remaining.find(sentence)
        if position < 0:
            raise VariantPostconditionError(
                f"{strategy} dropped or reordered a sentence it should have kept."
            )
        remaining = remaining[position + len(sentence) :]


def _kept_sentences(case: BaselineCase) -> tuple[str, ...]:
    sentences = split_sentences(case.text)
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
    produced = case.text + "".join(chosen)
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
    produced = case.text + language.flattery_sentence
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
    kept = _kept_sentences(case)
    produced = "".join(kept)
    if len(produced) >= len(case.text):
        raise VariantPostconditionError(
            "remove_evidence did not shorten the baseline."
        )
    _assert_evidence_removed(case, produced, kept, "remove_evidence")
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
    kept = _kept_sentences(case)
    assertion = rng.choice(list(language.unsupported_assertions))
    produced = "".join(kept) + assertion
    _assert_evidence_removed(case, produced, kept, "unsupported_assertion")
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


def _connective_substitution(
    case: BaselineCase, language: LanguageStrategy, rng: random.Random
) -> GeneratedVariant:
    produced = case.text
    replaced = 0
    for source, target in language.connectives:
        occurrences = produced.count(source)
        if occurrences:
            produced = produced.replace(source, target)
            replaced += occurrences
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
