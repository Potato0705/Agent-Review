from __future__ import annotations

import csv
import math
from pathlib import Path

from .models import ScoreRecord, ScoringCase, VariantType


REQUIRED_COLUMNS = {
    "system_name",
    "case_id",
    "variant_id",
    "variant_type",
    "score",
}
ALLOWED_VARIANTS: set[str] = {"baseline", "gaming", "degradation", "paraphrase"}
CASE_REQUIRED_COLUMNS = {"case_id", "variant_id", "variant_type", "text"}


def load_score_records(path: str | Path) -> list[ScoreRecord]:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"Input file does not exist: {source}")

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")

        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Input CSV is missing columns: {', '.join(sorted(missing))}")

        records: list[ScoreRecord] = []
        for row_number, row in enumerate(reader, start=2):
            variant_type = row["variant_type"].strip().lower()
            if variant_type not in ALLOWED_VARIANTS:
                raise ValueError(
                    f"Row {row_number}: unsupported variant_type {variant_type!r}; "
                    f"expected one of {sorted(ALLOWED_VARIANTS)}."
                )
            try:
                score = float(row["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Row {row_number}: score must be numeric.") from exc
            if not math.isfinite(score):
                raise ValueError(f"Row {row_number}: score must be finite.")

            stddev_text = (row.get("score_stddev") or "").strip()
            score_stddev: float | None = None
            if stddev_text:
                try:
                    score_stddev = float(stddev_text)
                except ValueError as exc:
                    raise ValueError(
                        f"Row {row_number}: score_stddev must be numeric."
                    ) from exc
                if not math.isfinite(score_stddev) or score_stddev < 0:
                    raise ValueError(
                        f"Row {row_number}: score_stddev must be finite and non-negative."
                    )

            sample_count_text = (row.get("sample_count") or "1").strip()
            try:
                sample_count = int(sample_count_text)
            except ValueError as exc:
                raise ValueError(f"Row {row_number}: sample_count must be an integer.") from exc
            if sample_count < 1:
                raise ValueError(f"Row {row_number}: sample_count must be at least 1.")
            if sample_count > 1 and score_stddev is None:
                raise ValueError(
                    f"Row {row_number}: score_stddev is required when sample_count > 1."
                )
            if sample_count == 1 and score_stddev not in {None, 0.0}:
                raise ValueError(
                    f"Row {row_number}: score_stddev must be empty or 0 when sample_count is 1."
                )

            required_values = {
                key: (row.get(key) or "").strip()
                for key in ("system_name", "case_id", "variant_id")
            }
            empty = [key for key, value in required_values.items() if not value]
            if empty:
                raise ValueError(
                    f"Row {row_number}: empty required values: {', '.join(empty)}."
                )

            records.append(
                ScoreRecord(
                    system_name=required_values["system_name"],
                    case_id=required_values["case_id"],
                    variant_id=required_values["variant_id"],
                    variant_type=variant_type,  # type: ignore[arg-type]
                    score=score,
                    notes=(row.get("notes") or "").strip(),
                    score_stddev=score_stddev,
                    sample_count=sample_count,
                )
            )

    if not records:
        raise ValueError("Input CSV contains no score records.")
    return records


def load_scoring_cases(path: str | Path) -> list[ScoringCase]:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"Input file does not exist: {source}")

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")
        missing = CASE_REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Input CSV is missing columns: {', '.join(sorted(missing))}")

        cases: list[ScoringCase] = []
        seen: set[tuple[str, str]] = set()
        for row_number, row in enumerate(reader, start=2):
            values = {
                key: (row.get(key) or "").strip()
                for key in ("case_id", "variant_id", "variant_type", "text")
            }
            empty = [key for key, value in values.items() if not value]
            if empty:
                raise ValueError(
                    f"Row {row_number}: empty required values: {', '.join(empty)}."
                )
            variant_type = values["variant_type"].lower()
            if variant_type not in ALLOWED_VARIANTS:
                raise ValueError(
                    f"Row {row_number}: unsupported variant_type {variant_type!r}; "
                    f"expected one of {sorted(ALLOWED_VARIANTS)}."
                )
            identity = (values["case_id"], values["variant_id"])
            if identity in seen:
                raise ValueError(
                    f"Row {row_number}: duplicate case_id/variant_id pair {identity!r}."
                )
            seen.add(identity)
            cases.append(
                ScoringCase(
                    case_id=values["case_id"],
                    variant_id=values["variant_id"],
                    variant_type=variant_type,  # type: ignore[arg-type]
                    text=values["text"],
                    notes=(row.get("notes") or "").strip(),
                )
            )

    if not cases:
        raise ValueError("Input CSV contains no scoring cases.")

    grouped: dict[str, list[ScoringCase]] = {}
    for case in cases:
        grouped.setdefault(case.case_id, []).append(case)
    for case_id, group in grouped.items():
        baseline_count = sum(item.variant_type == "baseline" for item in group)
        if baseline_count != 1:
            raise ValueError(
                f"Case {case_id!r} must have exactly one baseline; found {baseline_count}."
            )
        present_types = {item.variant_type for item in group}
        missing_paired_types = {"gaming", "degradation"}.difference(present_types)
        if missing_paired_types:
            raise ValueError(
                f"Case {case_id!r} is missing paired validity variants: "
                f"{', '.join(sorted(missing_paired_types))}."
            )
    return cases


def write_score_records(path: str | Path, records: list[ScoreRecord]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "system_name",
                "case_id",
                "variant_id",
                "variant_type",
                "score",
                "score_stddev",
                "sample_count",
                "notes",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "system_name": record.system_name,
                    "case_id": record.case_id,
                    "variant_id": record.variant_id,
                    "variant_type": record.variant_type,
                    "score": f"{record.score:.10g}",
                    "score_stddev": (
                        ""
                        if record.score_stddev is None
                        else f"{record.score_stddev:.10g}"
                    ),
                    "sample_count": record.sample_count,
                    "notes": record.notes,
                }
            )
    return destination


def write_text(path: str | Path, content: str) -> Path:
    """Write text with LF endings so artifacts are byte-identical per platform."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8", newline="\n")
    return destination
