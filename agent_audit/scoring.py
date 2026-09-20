from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Protocol
from uuid import uuid4

from .models import ProviderScore, ScoreRecord, ScoringCase


class Scorer(Protocol):
    def score(self, text: str, rubric: str) -> ProviderScore: ...


@dataclass(frozen=True)
class ScoringRun:
    records: tuple[ScoreRecord, ...]
    traces: tuple[dict[str, object], ...]
    manifest: dict[str, object]


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_scoring(
    cases: list[ScoringCase],
    scorer: Scorer,
    rubric: str,
    *,
    system_name: str,
    provider_name: str,
    model: str,
    base_url: str,
    score_min: float,
    score_max: float,
    temperature: float,
    repeats: int = 1,
) -> ScoringRun:
    if not cases:
        raise ValueError("At least one scoring case is required.")
    if not system_name.strip():
        raise ValueError("system_name must not be empty.")
    if not rubric.strip():
        raise ValueError("rubric must not be empty.")
    if not math.isfinite(score_min) or not math.isfinite(score_max):
        raise ValueError("score_min and score_max must be finite.")
    if score_max <= score_min:
        raise ValueError("score_max must be greater than score_min.")
    if not 1 <= repeats <= 100:
        raise ValueError("repeats must be between 1 and 100.")

    identities = [(case.case_id, case.variant_id) for case in cases]
    if len(identities) != len(set(identities)):
        raise ValueError("Scoring cases contain duplicate case_id/variant_id pairs.")
    grouped_types: dict[str, list[str]] = {}
    for case in cases:
        grouped_types.setdefault(case.case_id, []).append(case.variant_type)
    for case_id, variant_types in grouped_types.items():
        if variant_types.count("baseline") != 1:
            raise ValueError(f"Case {case_id!r} must have exactly one baseline.")
        missing = {"gaming", "degradation"}.difference(variant_types)
        if missing:
            raise ValueError(
                f"Case {case_id!r} is missing paired validity variants: "
                f"{', '.join(sorted(missing))}."
            )

    records: list[ScoreRecord] = []
    traces: list[dict[str, object]] = []
    for case in cases:
        results = [scorer.score(case.text, rubric) for _ in range(repeats)]
        scores = [result.score for result in results]
        mean_score = fmean(scores)
        score_stddev = stdev(scores) if repeats > 1 else None
        if any(
            not math.isfinite(score) or not score_min <= score <= score_max
            for score in scores
        ):
            raise ValueError(
                f"Scorer returned an out-of-range value for {case.case_id}/{case.variant_id}, "
                f"outside [{score_min}, {score_max}]."
            )
        note_parts = []
        first_reason = results[0].reason.replace("\r", " ").replace("\n", " ")
        if first_reason:
            note_parts.append(
                first_reason if repeats == 1 else f"sample_1_reason={first_reason}"
            )
        if case.notes:
            note_parts.append(f"source_note={case.notes}")
        records.append(
            ScoreRecord(
                system_name=system_name.strip(),
                case_id=case.case_id,
                variant_id=case.variant_id,
                variant_type=case.variant_type,
                score=mean_score,
                notes=" | ".join(note_parts),
                score_stddev=score_stddev,
                sample_count=repeats,
            )
        )
        for sample_index, sample in enumerate(results, start=1):
            traces.append(
                {
                    "case_id": case.case_id,
                    "variant_id": case.variant_id,
                    "variant_type": case.variant_type,
                    "sample_index": sample_index,
                    "input_sha256": hashlib.sha256(case.text.encode("utf-8")).hexdigest(),
                    "score": sample.score,
                    "reason": sample.reason,
                    "raw_content": sample.raw_content,
                    "response_id": sample.response_id,
                    "prompt_tokens": sample.prompt_tokens,
                    "completion_tokens": sample.completion_tokens,
                }
            )

    case_payload = [asdict(case) for case in cases]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "run_id": str(uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system_name": system_name.strip(),
        "provider": provider_name,
        "model": model,
        "base_url": base_url,
        "score_range": [score_min, score_max],
        "temperature": temperature,
        "repeats": repeats,
        "record_count": len(records),
        "input_sha256": _stable_hash(case_payload),
        "rubric_sha256": hashlib.sha256(rubric.encode("utf-8")).hexdigest(),
    }
    return ScoringRun(tuple(records), tuple(traces), manifest)


def write_json(path: str | Path, value: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def write_jsonl(path: str | Path, rows: tuple[dict[str, object], ...]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return destination
