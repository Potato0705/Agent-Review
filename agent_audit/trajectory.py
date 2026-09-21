"""Agent trajectories: the data model, the loader, and the rendered transcript.

What gets audited here is the grader that scores agent runs, not the agent.
The question is the same one this project always asks, moved to a new
modality: is the grader reading the process, or only the last paragraph?

Trajectories are structured, which is why they suit this method better than
prose. A text degradation variant can only be checked loosely — "the gaming
variant must still contain the baseline as a prefix". A trajectory variant can
be checked exactly: this step list is the baseline's minus the annotated step,
in order, with the same final answer.

Two things are annotated by a person and never inferred:

``load_bearing_steps``
    which steps the final answer actually rests on. Guessing wrong produces a
    variant that claims to be a degradation without degrading anything.

``independent_steps``
    which steps may be reordered without changing what happened. Guessing
    wrong produces an "equivalent" rewrite that silently broke the causal
    chain, which is the one thing a paraphrase variant must never do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .segmentation import LanguageStrategy


@dataclass(frozen=True)
class TrajectoryStep:
    """One tool call and what it returned.

    ``index`` is 1-based and assigned by position at load time, so annotations
    given by number always mean what the reviewer saw. ``args`` is already
    serialised with sorted keys: the same input has to render byte-for-byte
    identically or the generation fingerprint means nothing.
    """

    index: int
    tool: str
    args: str
    result: str
    note: str = ""

    @property
    def identity(self) -> tuple[str, str, str]:
        """What the step did, ignoring where it sits and how it is narrated."""

        return (self.tool, self.args, self.result)


@dataclass(frozen=True)
class TrajectoryCase:
    case_id: str
    task: str
    steps: tuple[TrajectoryStep, ...]
    final_answer: str
    load_bearing_steps: tuple[int, ...]
    independent_steps: tuple[tuple[int, ...], ...] = ()
    notes: str = ""


def _text(payload: dict[str, object], key: str, line: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line}: {key} must be a non-empty string.")
    return value.strip()


def _indices(raw: object, key: str, line: int) -> tuple[int, ...]:
    if not isinstance(raw, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw
    ):
        raise ValueError(f"Line {line}: {key} must be a list of integers.")
    return tuple(raw)


def _parse_steps(raw: object, line: int) -> tuple[TrajectoryStep, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"Line {line}: steps must be a list.")
    if not raw:
        raise ValueError(f"Line {line}: a trajectory needs at least one step.")

    steps: list[TrajectoryStep] = []
    for position, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Line {line}: step {position} must be an object.")
        tool = item.get("tool")
        if not isinstance(tool, str) or not tool.strip():
            raise ValueError(
                f"Line {line}: step {position} must name a tool."
            )
        result = item.get("result", "")
        if not isinstance(result, str):
            raise ValueError(
                f"Line {line}: step {position} result must be a string."
            )
        note = item.get("note", "")
        if not isinstance(note, str):
            raise ValueError(f"Line {line}: step {position} note must be a string.")
        steps.append(
            TrajectoryStep(
                index=position,
                tool=tool.strip(),
                args=_serialise_args(item.get("args", {}), position, line),
                result=result,
                note=note,
            )
        )
    return tuple(steps)


def _serialise_args(raw: object, position: int, line: int) -> str:
    """Render call arguments so the same call always produces the same bytes."""

    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False, sort_keys=True)
    except TypeError as exc:
        raise ValueError(
            f"Line {line}: step {position} args are not serialisable."
        ) from exc


def _validate_load_bearing(
    indices: tuple[int, ...], step_count: int, case_id: str
) -> None:
    if not indices:
        raise ValueError(
            f"Case {case_id!r} has no load_bearing_steps. The tool does not "
            "infer which steps the answer rests on; annotate them."
        )
    if len(set(indices)) != len(indices):
        raise ValueError(f"Case {case_id!r} repeats a load_bearing_steps index.")
    if any(index < 1 or index > step_count for index in indices):
        raise ValueError(
            f"Case {case_id!r} has a load_bearing_steps index out of range "
            f"1..{step_count}."
        )
    if len(indices) == step_count:
        raise ValueError(
            f"Case {case_id!r} annotates every step as load-bearing, which "
            "leaves nothing to keep in a degradation variant."
        )


def _validate_independent(
    groups: tuple[tuple[int, ...], ...], step_count: int, case_id: str
) -> None:
    seen: set[int] = set()
    for group in groups:
        if len(group) < 2:
            raise ValueError(
                f"Case {case_id!r} has an independent_steps group with fewer "
                "than at least two members; a group of one reorders nothing."
            )
        if len(set(group)) != len(group):
            raise ValueError(
                f"Case {case_id!r} repeats an index inside one "
                "independent_steps group."
            )
        if any(index < 1 or index > step_count for index in group):
            raise ValueError(
                f"Case {case_id!r} has an independent_steps index out of range "
                f"1..{step_count}."
            )
        overlap = seen.intersection(group)
        if overlap:
            raise ValueError(
                f"Case {case_id!r} puts step {sorted(overlap)[0]} in more than "
                "one group, which makes the reordering ambiguous."
            )
        seen.update(group)


def load_trajectory_cases(path: str | Path) -> list[TrajectoryCase]:
    """Load annotated trajectories, refusing anything that could be misread."""

    source = Path(path)
    if not source.exists():
        raise ValueError(f"Trajectory file does not exist: {source}")

    cases: list[TrajectoryCase] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: malformed JSON.") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Line {line_number}: each line must be an object.")

        case_id = _text(payload, "case_id", line_number)
        if case_id in seen:
            raise ValueError(f"Duplicate case_id: {case_id!r}.")
        seen.add(case_id)

        steps = _parse_steps(payload.get("steps"), line_number)
        load_bearing = _indices(
            payload.get("load_bearing_steps", []), "load_bearing_steps", line_number
        )
        _validate_load_bearing(load_bearing, len(steps), case_id)

        raw_groups = payload.get("independent_steps", [])
        if not isinstance(raw_groups, list):
            raise ValueError(
                f"Line {line_number}: independent_steps must be a list of lists."
            )
        groups = tuple(
            _indices(group, "independent_steps", line_number) for group in raw_groups
        )
        _validate_independent(groups, len(steps), case_id)

        notes = payload.get("notes", "")
        cases.append(
            TrajectoryCase(
                case_id=case_id,
                task=_text(payload, "task", line_number),
                steps=steps,
                final_answer=_text(payload, "final_answer", line_number),
                load_bearing_steps=load_bearing,
                independent_steps=groups,
                notes=notes if isinstance(notes, str) else "",
            )
        )

    if not cases:
        raise ValueError(f"Trajectory file contains no trajectories: {source}")
    return cases


def render_trajectory(case: TrajectoryCase, language: LanguageStrategy) -> str:
    """Render the transcript a grader sees.

    Steps are numbered by position rather than by ``TrajectoryStep.index``, so
    a variant that removed or reordered steps reads as a coherent run instead
    of advertising the edit through a gap in the numbering.
    """

    labels = language.trajectory_labels
    mark = _separator(language)
    lines = [f"{labels.task}{mark}{case.task}", ""]
    for position, step in enumerate(case.steps, start=1):
        lines.append(f"{labels.step} {position}")
        lines.append(f"  {labels.tool}{mark}{step.tool}")
        lines.append(f"  {labels.args}{mark}{step.args}")
        lines.append(f"  {labels.result}{mark}{step.result}")
        if step.note:
            lines.append(f"  {labels.note}{mark}{step.note}")
        lines.append("")
    lines.append(f"{labels.answer}{mark}{case.final_answer}")
    return "\n".join(lines)


def _separator(language: LanguageStrategy) -> str:
    """Chinese labels take a full-width colon; English takes a colon and space."""

    return "：" if language.name == "chinese" else ": "
