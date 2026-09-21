from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from .models import BaselineCase, ScoreRecord, ScoringCase, VariantType
from .segmentation import CHINESE, LanguageStrategy


REQUIRED_COLUMNS = {
    "system_name",
    "case_id",
    "variant_id",
    "variant_type",
    "score",
}
ALLOWED_VARIANTS: set[str] = {"baseline", "gaming", "degradation", "paraphrase"}
CASE_REQUIRED_COLUMNS = {"case_id", "variant_id", "variant_type", "text"}
BASELINE_REQUIRED_COLUMNS = {"case_id", "text", "evidence_sentences"}


def stable_hash(value: object) -> str:
    """Hash a payload so the same content always yields the same digest.

    Generation and scoring both fingerprint the case set with this, and the
    audit proves the scored cases are the generated ones by comparing the two
    digests. They must therefore share one definition, not two that happen to
    agree today.
    """

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _parse_evidence_indices(raw: str, row_number: int) -> tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError(f"Row {row_number}: evidence_sentences must not be empty.")
    indices: list[int] = []
    for part in parts:
        if not part.isdigit() or int(part) < 1:
            raise ValueError(
                f"Row {row_number}: evidence_sentences must be positive integers."
            )
        indices.append(int(part))
    if len(set(indices)) != len(indices):
        raise ValueError(f"Row {row_number}: duplicate evidence_sentences index.")
    return tuple(sorted(indices))


def load_baseline_cases(
    path: str | Path, *, language: LanguageStrategy | None = None
) -> list[BaselineCase]:
    """Load annotated baselines, rejecting any annotation we cannot act on."""

    strategy = language or CHINESE

    source = Path(path)
    if not source.exists():
        raise ValueError(f"Input file does not exist: {source}")

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")
        missing = BASELINE_REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Input CSV is missing columns: {', '.join(sorted(missing))}"
            )

        cases: list[BaselineCase] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            case_id = (row.get("case_id") or "").strip()
            text = (row.get("text") or "").strip()
            empty = [
                name
                for name, value in (("case_id", case_id), ("text", text))
                if not value
            ]
            if empty:
                raise ValueError(
                    f"Row {row_number}: empty required values: {', '.join(empty)}."
                )
            if case_id in seen:
                raise ValueError(f"Row {row_number}: duplicate case_id {case_id!r}.")
            seen.add(case_id)

            sentences = strategy.split(text)
            declared = (row.get("sentence_count") or "").strip()
            if declared:
                # The annotation is given by index, so a splitter that
                # disagrees with the reviewer silently points every later
                # index at the wrong sentence. This turns that into a refusal.
                if not declared.isdigit() or int(declared) < 1:
                    raise ValueError(
                        f"Row {row_number}: sentence_count must be a positive integer."
                    )
                if int(declared) != len(sentences):
                    numbered = "; ".join(
                        f"{index}. {sentence.strip()}"
                        for index, sentence in enumerate(sentences, start=1)
                    )
                    plural = "" if len(sentences) == 1 else "s"
                    raise ValueError(
                        f"Row {row_number}: sentence_count says {declared} but the "
                        f"{strategy.name} splitter found {len(sentences)} "
                        f"sentence{plural}: {numbered}"
                    )
            if len(sentences) < 2:
                raise ValueError(
                    f"Row {row_number}: a baseline needs at least two sentences so "
                    "that removing the evidence still leaves content."
                )

            indices = _parse_evidence_indices(
                row.get("evidence_sentences") or "", row_number
            )
            if indices[-1] > len(sentences):
                raise ValueError(
                    f"Row {row_number}: evidence_sentences names sentence "
                    f"{indices[-1]} but the text has only {len(sentences)} sentences."
                )
            if len(indices) == len(sentences):
                raise ValueError(
                    f"Row {row_number}: cannot annotate every sentence; the "
                    "degradation variant would be empty."
                )

            cases.append(
                BaselineCase(
                    case_id=case_id,
                    text=text,
                    evidence_sentences=indices,
                    notes=(row.get("notes") or "").strip(),
                )
            )

    if not cases:
        raise ValueError("Input CSV contains no baseline cases.")
    return cases


def write_scoring_cases(path: str | Path, rows) -> Path:
    """Write generated rows in the exact shape ``load_scoring_cases`` expects."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "variant_id", "variant_type", "text", "notes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row.case_id,
                    "variant_id": row.variant_id,
                    "variant_type": row.variant_type,
                    "text": row.text,
                    "notes": row.notes,
                }
            )
    return destination


def write_text(path: str | Path, content: str) -> Path:
    """Write text with LF endings so artifacts are byte-identical per platform."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8", newline="\n")
    return destination
