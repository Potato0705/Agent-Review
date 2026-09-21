"""Helpers shared by more than one subcommand.

Each subcommand lives in its own module so that adding one means adding a
file rather than growing a thousand-line switchboard. What genuinely belongs
to two of them lives here, once: a refusal copied into two places drifts, and
a drifted refusal stops refusing in one of them without anyone noticing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def refuse_to_overwrite_cases(output_path: Path, append: bool) -> None:
    """A case file carries hand edits, so it is never silently replaced."""

    if output_path.exists() and not append:
        raise ValueError(
            f"Case file already exists: {output_path}. Hand edits live in this "
            "file, so it is never overwritten; use --append to merge or choose "
            "a new path."
        )


def strategy_list(raw: str | None) -> tuple[str, ...]:
    """Split a comma-separated strategy option, dropping empty entries."""

    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def row_fingerprint_payload(rows: Any) -> list[dict[str, str]]:
    """Describe generated rows for hashing.

    Notes are included on purpose: editing only a row's note still changes the
    case set, and a fingerprint that ignored notes would call the edited set
    machine-generated.
    """

    return [
        {
            "case_id": row.case_id,
            "variant_id": row.variant_id,
            "variant_type": row.variant_type,
            "text": row.text,
            "notes": row.notes,
        }
        for row in rows
    ]


def merge_summary_of(rows: Any) -> dict[str, Any]:
    """The merge report for a run that did not merge into anything."""

    return {
        "appended": [],
        "preserved": [[row.case_id, row.variant_id] for row in rows],
        "edited": [],
        "foreign": [],
    }


def describe_merge(outcome: Any) -> dict[str, Any]:
    """The merge report for a run that did merge."""

    return {
        "appended": [list(item) for item in outcome.appended],
        "preserved": [list(item) for item in outcome.preserved],
        "edited": [list(item) for item in outcome.edited],
        "foreign": [list(item) for item in outcome.foreign],
    }
