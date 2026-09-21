"""Trajectory variants and the postconditions that keep their labels honest.

This is the first modality in this project where all three variant families
can be checked exactly. In prose a gaming variant is only checkable as "the
baseline is still a prefix", and equivalence is not machine-checkable at all —
v0.14.0 had to fall back on a model drafting and a human ratifying. A
trajectory is structured, so "these are the baseline's steps minus step 2, in
order, with the same final answer" is decidable, and every variant below is
refused unless it proves its own claim.

Two rules shape the strategies:

*Never fabricate.* A gaming variant may repeat a call the agent really made;
it may not invent a tool or make up what a tool returned. That would be
planting evidence rather than constructing a variant.

*Degradation keeps the final answer.* Changing the process and the result at
once cannot distinguish a grader that reads the process from one that reads
the conclusion. Keeping the answer is what makes the probe sharp.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any

from .segmentation import LanguageStrategy
from .trajectory import TrajectoryCase, TrajectoryStep, render_trajectory


GAMING_STRATEGIES = ("redundant_tool_calls", "padded_reasoning")
DEGRADATION_STRATEGIES = ("remove_load_bearing_step", "hollow_evidence")
PARAPHRASE_STRATEGIES = ("reorder_independent_steps",)

REPEATED_CALL_COUNT = 2


class TrajectoryPostconditionError(ValueError):
    """Raised when a trajectory variant does not match its own label."""


@dataclass(frozen=True)
class TrajectoryVariant:
    case_id: str
    variant_id: str
    variant_type: str
    strategy: str
    case: TrajectoryCase
    inserted_text: tuple[str, ...] = ()
    magnitude: dict[str, Any] = field(default_factory=dict)


def _renumber(steps: list[TrajectoryStep]) -> tuple[TrajectoryStep, ...]:
    """Give the variant its own 1..n numbering.

    A gap in the numbering would tell the grader an edit happened, which is
    information the baseline run does not carry.
    """

    return tuple(
        replace(step, index=position) for position, step in enumerate(steps, start=1)
    )


def _identities(steps: tuple[TrajectoryStep, ...]) -> list[tuple[str, str, str]]:
    return [step.identity for step in steps]


# --- gaming -----------------------------------------------------------------


def _redundant_tool_calls(
    case: TrajectoryCase, language: LanguageStrategy, rng: random.Random
) -> TrajectoryVariant:
    """Re-issue calls the agent already made: more work, no more information."""

    positions = sorted(
        rng.sample(range(len(case.steps)), min(REPEATED_CALL_COUNT, len(case.steps)))
    )
    steps: list[TrajectoryStep] = []
    repeated: list[int] = []
    for position, step in enumerate(case.steps):
        steps.append(step)
        if position in positions:
            steps.append(step)
            repeated.append(step.index)

    produced = replace(case, steps=_renumber(steps))
    _assert_only_repeated_calls(case, produced)
    return TrajectoryVariant(
        case_id=case.case_id,
        variant_id="gaming_redundant_tool_calls",
        variant_type="gaming",
        strategy="redundant_tool_calls",
        case=produced,
        magnitude={
            "repeated_steps": repeated,
            "added_steps": len(produced.steps) - len(case.steps),
        },
    )


def _assert_only_repeated_calls(
    baseline: TrajectoryCase, produced: TrajectoryCase
) -> None:
    if len(produced.steps) <= len(baseline.steps):
        raise TrajectoryPostconditionError(
            "redundant_tool_calls must add steps, not remove them."
        )
    if produced.final_answer != baseline.final_answer:
        raise TrajectoryPostconditionError(
            "redundant_tool_calls must leave the final answer untouched."
        )
    allowed = set(_identities(baseline.steps))
    for identity in _identities(produced.steps):
        if identity not in allowed:
            raise TrajectoryPostconditionError(
                "redundant_tool_calls invented a call the baseline never made."
            )
    collapsed: list[tuple[str, str, str]] = []
    for identity in _identities(produced.steps):
        if not collapsed or collapsed[-1] != identity:
            collapsed.append(identity)
    if collapsed != _identities(baseline.steps):
        raise TrajectoryPostconditionError(
            "redundant_tool_calls changed the baseline sequence."
        )


def _padded_reasoning(
    case: TrajectoryCase, language: LanguageStrategy, rng: random.Random
) -> TrajectoryVariant:
    """Narrate every step at length without doing anything differently."""

    if not language.padding_sentences:
        raise TrajectoryPostconditionError(
            f"Case {case.case_id!r}: {language.name} supplies no padding text."
        )
    chosen = tuple(
        language.padding_sentences[rng.randrange(len(language.padding_sentences))]
        for _ in case.steps
    )
    steps = [
        replace(step, note=(step.note + language.sentence_separator + filler).strip())
        for step, filler in zip(case.steps, chosen, strict=True)
    ]
    produced = replace(case, steps=_renumber(steps))
    _assert_only_narration_changed(case, produced)
    return TrajectoryVariant(
        case_id=case.case_id,
        variant_id="gaming_padded_reasoning",
        variant_type="gaming",
        strategy="padded_reasoning",
        case=produced,
        inserted_text=chosen,
        magnitude={
            "added_note_characters": sum(len(text) for text in chosen),
            "padded_steps": [step.index for step in case.steps],
        },
    )


def _assert_only_narration_changed(
    baseline: TrajectoryCase, produced: TrajectoryCase
) -> None:
    if _identities(produced.steps) != _identities(baseline.steps):
        raise TrajectoryPostconditionError(
            "padded_reasoning changed what the agent actually did."
        )
    if produced.final_answer != baseline.final_answer:
        raise TrajectoryPostconditionError(
            "padded_reasoning must leave the final answer untouched."
        )
    before = sum(len(step.note) for step in baseline.steps)
    after = sum(len(step.note) for step in produced.steps)
    if after <= before:
        raise TrajectoryPostconditionError(
            "padded_reasoning added no narration."
        )


# --- degradation ------------------------------------------------------------


def _remove_load_bearing_step(
    case: TrajectoryCase, language: LanguageStrategy, rng: random.Random
) -> TrajectoryVariant:
    """Delete the steps the answer rests on and keep the answer anyway."""

    removed = set(case.load_bearing_steps)
    steps = [step for step in case.steps if step.index not in removed]
    if not steps:
        raise TrajectoryPostconditionError(
            f"Case {case.case_id!r}: removing the load-bearing steps leaves no "
            "trajectory at all."
        )
    produced = replace(
        case, steps=_renumber(steps), load_bearing_steps=(), independent_steps=()
    )
    _assert_steps_removed(case, produced, sorted(removed))
    return TrajectoryVariant(
        case_id=case.case_id,
        variant_id="degradation_remove_load_bearing_step",
        variant_type="degradation",
        strategy="remove_load_bearing_step",
        case=produced,
        magnitude={
            "removed_steps": sorted(removed),
            "remaining_steps": len(produced.steps),
        },
    )


def _assert_steps_removed(
    baseline: TrajectoryCase, produced: TrajectoryCase, removed: list[int]
) -> None:
    # Comparing the identity lists settles both questions at once: the right
    # steps are gone and the rest are in their original order. A separate
    # length check would be unreachable, and an assertion no input can reach
    # is an assertion nobody can trust.
    kept = [step for step in baseline.steps if step.index not in set(removed)]
    if _identities(produced.steps) != _identities(tuple(kept)):
        raise TrajectoryPostconditionError(
            "remove_load_bearing_step did not keep exactly the remaining steps "
            "in their original order."
        )
    if produced.final_answer != baseline.final_answer:
        raise TrajectoryPostconditionError(
            "remove_load_bearing_step must keep the final answer: changing the "
            "process and the result at once cannot tell the two apart."
        )


def _hollow_evidence(
    case: TrajectoryCase, language: LanguageStrategy, rng: random.Random
) -> TrajectoryVariant:
    """Keep the call, take away what it found."""

    annotated = set(case.load_bearing_steps)
    empty = language.empty_tool_result
    steps = [
        replace(step, result=empty) if step.index in annotated else step
        for step in case.steps
    ]
    produced = replace(case, steps=_renumber(steps))
    _assert_evidence_hollowed(case, produced, annotated)
    return TrajectoryVariant(
        case_id=case.case_id,
        variant_id="degradation_hollow_evidence",
        variant_type="degradation",
        strategy="hollow_evidence",
        case=produced,
        inserted_text=(empty,),
        magnitude={
            "hollowed_steps": sorted(annotated),
            "removed_result_characters": sum(
                len(step.result) for step in case.steps if step.index in annotated
            ),
        },
    )


def _assert_evidence_hollowed(
    baseline: TrajectoryCase, produced: TrajectoryCase, annotated: set[int]
) -> None:
    if len(produced.steps) != len(baseline.steps):
        raise TrajectoryPostconditionError(
            "hollow_evidence must keep every call in place."
        )
    if produced.final_answer != baseline.final_answer:
        raise TrajectoryPostconditionError(
            "hollow_evidence must keep the final answer."
        )
    for before, after in zip(baseline.steps, produced.steps, strict=True):
        if (before.tool, before.args) != (after.tool, after.args):
            raise TrajectoryPostconditionError(
                "hollow_evidence changed which call was made."
            )
        if before.index in annotated:
            # Checked before the substring test, not folded into it: in Python
            # the empty string is a substring of everything, so a step that
            # already returned nothing would pass the test below while nothing
            # was actually degraded.
            if not before.result.strip():
                raise TrajectoryPostconditionError(
                    f"load-bearing step {before.index} already returns no "
                    "result, so there is nothing to hollow out."
                )
            if before.result in after.result:
                raise TrajectoryPostconditionError(
                    "hollow_evidence left the original result in place."
                )
        elif before.result != after.result:
            raise TrajectoryPostconditionError(
                "hollow_evidence touched a step it was not asked to touch."
            )


# --- paraphrase -------------------------------------------------------------


def _reorder_independent_steps(
    case: TrajectoryCase, language: LanguageStrategy, rng: random.Random
) -> TrajectoryVariant:
    """Take the same steps in a different order.

    Only groups a reviewer declared independent are touched. The tool never
    works out for itself whether two calls can swap: getting that wrong is how
    an "equivalent" variant quietly breaks the causal chain.
    """

    if not case.independent_steps:
        raise TrajectoryPostconditionError(
            f"Case {case.case_id!r} declares no independent_steps, so there is "
            "no reordering anyone has verified to be equivalent."
        )

    by_index = {step.index: step for step in case.steps}
    placement = {step.index: step.index for step in case.steps}
    for group in case.independent_steps:
        rotated = group[1:] + group[:1]
        for slot, source in zip(group, rotated, strict=True):
            placement[slot] = source

    steps = [by_index[placement[step.index]] for step in case.steps]
    produced = replace(
        case, steps=_renumber(steps), load_bearing_steps=(), independent_steps=()
    )
    _assert_same_work_reordered(case, produced)
    return TrajectoryVariant(
        case_id=case.case_id,
        variant_id="paraphrase_reorder_independent_steps",
        variant_type="paraphrase",
        strategy="reorder_independent_steps",
        case=produced,
        magnitude={"reordered_groups": [list(group) for group in case.independent_steps]},
    )


def _assert_same_work_reordered(
    baseline: TrajectoryCase, produced: TrajectoryCase
) -> None:
    if sorted(_identities(produced.steps)) != sorted(_identities(baseline.steps)):
        raise TrajectoryPostconditionError(
            "reorder_independent_steps changed which calls were made."
        )
    if produced.final_answer != baseline.final_answer:
        raise TrajectoryPostconditionError(
            "reorder_independent_steps changed the final answer."
        )
    if _identities(produced.steps) == _identities(baseline.steps):
        raise TrajectoryPostconditionError(
            "reorder_independent_steps produced the baseline order again; the "
            "declared group holds interchangeable calls, so nothing moved."
        )


_BUILDERS = {
    "redundant_tool_calls": _redundant_tool_calls,
    "padded_reasoning": _padded_reasoning,
    "remove_load_bearing_step": _remove_load_bearing_step,
    "hollow_evidence": _hollow_evidence,
    "reorder_independent_steps": _reorder_independent_steps,
}


def build_trajectory_variant(
    case: TrajectoryCase,
    strategy: str,
    language: LanguageStrategy,
    rng: random.Random,
) -> TrajectoryVariant:
    """Build one variant, refusing to return anything that fails its own claim."""

    try:
        builder = _BUILDERS[strategy]
    except KeyError as exc:
        raise ValueError(
            f"unknown trajectory strategy {strategy!r}; expected one of "
            f"{sorted(_BUILDERS)}."
        ) from exc
    return builder(case, language, rng)


@dataclass(frozen=True)
class TrajectoryRow:
    case_id: str
    variant_id: str
    variant_type: str
    text: str
    notes: str


@dataclass(frozen=True)
class TrajectoryRun:
    rows: tuple[TrajectoryRow, ...]
    manifest: dict[str, Any]


def _validate_family(
    selected: tuple[str, ...], allowed: tuple[str, ...], family: str
) -> None:
    if not selected:
        raise ValueError(
            f"Trajectory generation needs at least one {family} strategy; "
            "`load_scoring_cases` requires every case to have one."
        )
    for name in selected:
        if name not in allowed:
            raise ValueError(
                f"{name!r} is not a {family} strategy; expected one of "
                f"{list(allowed)}."
            )


def _magnitude_summary(magnitude: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in sorted(magnitude.items()))


def generate_trajectory_variants(
    cases: list[TrajectoryCase],
    *,
    gaming: tuple[str, ...],
    degradation: tuple[str, ...],
    paraphrase: tuple[str, ...],
    language: LanguageStrategy,
    seed: int,
) -> TrajectoryRun:
    """Render every baseline and its variants into scoring rows."""

    if not cases:
        raise ValueError("At least one trajectory case is required.")
    _validate_family(gaming, GAMING_STRATEGIES, "gaming")
    _validate_family(degradation, DEGRADATION_STRATEGIES, "degradation")
    for name in paraphrase:
        if name not in PARAPHRASE_STRATEGIES:
            raise ValueError(
                f"{name!r} is not a paraphrase strategy; expected one of "
                f"{list(PARAPHRASE_STRATEGIES)}."
            )

    rng = random.Random(seed)
    rows: list[TrajectoryRow] = []
    variant_records: list[dict[str, Any]] = []
    for case in cases:
        rows.append(
            TrajectoryRow(
                case_id=case.case_id,
                variant_id="baseline",
                variant_type="baseline",
                text=render_trajectory(case, language),
                notes=case.notes or "generated=trajectory_baseline",
            )
        )
        for strategy in (*gaming, *degradation, *paraphrase):
            try:
                variant = build_trajectory_variant(case, strategy, language, rng)
            except TrajectoryPostconditionError as exc:
                raise TrajectoryPostconditionError(
                    f"Case {case.case_id!r} / {strategy}: {exc}"
                ) from exc
            rows.append(
                TrajectoryRow(
                    case_id=variant.case_id,
                    variant_id=variant.variant_id,
                    variant_type=variant.variant_type,
                    text=render_trajectory(variant.case, language),
                    notes=(
                        f"generated={strategy} | {_magnitude_summary(variant.magnitude)}"
                    ).strip(" |"),
                )
            )
            variant_records.append(
                {
                    "case_id": variant.case_id,
                    "variant_id": variant.variant_id,
                    "strategy": strategy,
                    "magnitude": variant.magnitude,
                    "inserted_text": list(variant.inserted_text),
                }
            )

    manifest = {
        "generator": "agent-review",
        "modality": "trajectory",
        "language": language.name,
        "seed": seed,
        "gaming_strategies": list(gaming),
        "degradation_strategies": list(degradation),
        "paraphrase_strategies": list(paraphrase),
        "case_count": len(cases),
        "variants": variant_records,
    }
    return TrajectoryRun(rows=tuple(rows), manifest=manifest)
