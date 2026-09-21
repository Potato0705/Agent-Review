from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from .checkpoint import (
    CheckpointSample,
    append_checkpoint_sample,
    create_checkpoint,
    load_checkpoint,
    repair_checkpoint_truncated_tail,
)
from .io import stable_hash
from .models import ProviderScore, ScoreRecord, ScoringCase


class Scorer(Protocol):
    def score(self, text: str, rubric: str) -> ProviderScore: ...


@dataclass(frozen=True)
class ScoringRun:
    records: tuple[ScoreRecord, ...]
    traces: tuple[dict[str, object], ...]
    manifest: dict[str, object]


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
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
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
    for name, value in {
        "provider_name": provider_name,
        "model": model,
        "base_url": base_url,
    }.items():
        if not value.strip():
            raise ValueError(f"{name} must not be empty.")
    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite.")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 100:
        raise ValueError("repeats must be between 1 and 100.")
    if resume and checkpoint_path is None:
        raise ValueError("resume requires checkpoint_path.")

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

    case_payload = [asdict(case) for case in cases]
    input_sha256 = stable_hash(case_payload)
    rubric_sha256 = hashlib.sha256(rubric.encode("utf-8")).hexdigest()
    checkpoint_context: dict[str, object] = {
        "system_name": system_name.strip(),
        "provider": provider_name.strip(),
        "model": model.strip(),
        "base_url": base_url.strip(),
        "score_range": [score_min, score_max],
        "temperature": temperature,
        "repeats": repeats,
        "input_sha256": input_sha256,
        "rubric_sha256": rubric_sha256,
    }
    run_started = perf_counter()
    resumed_samples: dict[tuple[str, str, int], CheckpointSample] = {}
    repaired_truncated_tail = False
    truncated_tail_offset: int | None = None
    checkpoint_original_size = 0
    if checkpoint_path is not None:
        if resume:
            loaded = load_checkpoint(checkpoint_path, checkpoint_context)
            run_id = loaded.run_id
            created_at_utc = loaded.created_at_utc
            resumed_samples = loaded.samples
            truncated_tail_offset = loaded.truncated_tail_offset
            checkpoint_original_size = loaded.original_size
        else:
            run_id, created_at_utc = create_checkpoint(
                checkpoint_path, checkpoint_context
            )
    else:
        run_id = str(uuid4())
        created_at_utc = datetime.now(timezone.utc).isoformat()

    expected_samples: dict[
        tuple[str, str, int], tuple[str, str]
    ] = {}
    for case in cases:
        case_input_hash = hashlib.sha256(case.text.encode("utf-8")).hexdigest()
        for sample_index in range(1, repeats + 1):
            expected_samples[(case.case_id, case.variant_id, sample_index)] = (
                case.variant_type,
                case_input_hash,
            )
    unexpected_samples = sorted(set(resumed_samples).difference(expected_samples))
    if unexpected_samples:
        raise ValueError(
            f"Checkpoint contains samples outside this run: {unexpected_samples}."
        )
    for identity, sample in resumed_samples.items():
        expected_variant_type, expected_input_hash = expected_samples[identity]
        if (
            sample.variant_type != expected_variant_type
            or sample.input_sha256 != expected_input_hash
        ):
            raise ValueError(
                f"Checkpoint sample metadata does not match this run: {identity!r}."
            )
        _validate_provider_score(
            sample.result,
            score_min,
            score_max,
            case_id=sample.case_id,
            variant_id=sample.variant_id,
        )
    if checkpoint_path is not None and truncated_tail_offset is not None:
        repair_checkpoint_truncated_tail(
            checkpoint_path,
            truncated_tail_offset=truncated_tail_offset,
            expected_size=checkpoint_original_size,
        )
        repaired_truncated_tail = True

    records: list[ScoreRecord] = []
    traces: list[dict[str, object]] = []
    resumed_sample_count = 0
    new_sample_count = 0
    new_sample_latency_seconds = 0.0
    for case in cases:
        samples: list[CheckpointSample] = []
        case_input_hash = hashlib.sha256(case.text.encode("utf-8")).hexdigest()
        for sample_index in range(1, repeats + 1):
            identity = (case.case_id, case.variant_id, sample_index)
            sample = resumed_samples.get(identity)
            if sample is not None:
                resumed_sample_count += 1
            else:
                sample_started = perf_counter()
                result = scorer.score(case.text, rubric)
                latency_seconds = perf_counter() - sample_started
                _validate_provider_score(
                    result,
                    score_min,
                    score_max,
                    case_id=case.case_id,
                    variant_id=case.variant_id,
                )
                sample = CheckpointSample(
                    case_id=case.case_id,
                    variant_id=case.variant_id,
                    variant_type=case.variant_type,
                    sample_index=sample_index,
                    input_sha256=case_input_hash,
                    result=result,
                    latency_seconds=latency_seconds,
                )
                if checkpoint_path is not None:
                    append_checkpoint_sample(checkpoint_path, sample)
                new_sample_count += 1
                new_sample_latency_seconds += latency_seconds
            samples.append(sample)

        results = [sample.result for sample in samples]
        scores = [sample.score for sample in results]
        mean_score = fmean(scores)
        score_stddev = stdev(scores) if repeats > 1 else None
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
        for sample in samples:
            result = sample.result
            traces.append(
                {
                    "case_id": case.case_id,
                    "variant_id": case.variant_id,
                    "variant_type": case.variant_type,
                    "sample_index": sample.sample_index,
                    "input_sha256": sample.input_sha256,
                    "score": result.score,
                    "reason": result.reason,
                    "raw_content": result.raw_content,
                    "response_id": result.response_id,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "latency_seconds": sample.latency_seconds,
                    "resumed_from_checkpoint": sample.identity in resumed_samples,
                }
            )

    completed_at_utc = datetime.now(timezone.utc).isoformat()
    prompt_tokens = [trace["prompt_tokens"] for trace in traces]
    completion_tokens = [trace["completion_tokens"] for trace in traces]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "completed_at_utc": completed_at_utc,
        "system_name": system_name.strip(),
        "provider": provider_name,
        "model": model,
        "base_url": base_url,
        "score_range": [score_min, score_max],
        "temperature": temperature,
        "repeats": repeats,
        "record_count": len(records),
        "sample_count": len(traces),
        "input_sha256": input_sha256,
        "rubric_sha256": rubric_sha256,
        "checkpoint_used": checkpoint_path is not None,
        "checkpoint_resumed": resume,
        "checkpoint_repaired_truncated_tail": repaired_truncated_tail,
        "resumed_sample_count": resumed_sample_count,
        "new_sample_count": new_sample_count,
        "total_sample_latency_seconds": sum(
            float(trace["latency_seconds"]) for trace in traces
        ),
        "new_sample_latency_seconds": new_sample_latency_seconds,
        "invocation_wall_time_seconds": perf_counter() - run_started,
        "prompt_tokens_total": _complete_token_total(prompt_tokens),
        "completion_tokens_total": _complete_token_total(completion_tokens),
    }
    return ScoringRun(tuple(records), tuple(traces), manifest)


def _validate_provider_score(
    result: ProviderScore,
    score_min: float,
    score_max: float,
    *,
    case_id: str,
    variant_id: str,
) -> None:
    if isinstance(result.score, bool) or not isinstance(result.score, (int, float)):
        raise ValueError("Scorer score must be numeric.")
    if not math.isfinite(result.score) or not score_min <= result.score <= score_max:
        raise ValueError(
            f"Scorer returned an out-of-range value for {case_id}/{variant_id}, "
            f"outside [{score_min}, {score_max}]."
        )
    if not isinstance(result.reason, str) or not isinstance(result.raw_content, str):
        raise ValueError("Scorer reason and raw_content must be strings.")
    if result.response_id is not None and not isinstance(result.response_id, str):
        raise ValueError("Scorer response_id must be a string or None.")
    for name, value in {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }.items():
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ValueError(f"Scorer {name} must be a non-negative integer or None.")


def _complete_token_total(values: list[object]) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def write_json(path: str | Path, value: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    return destination


def write_jsonl(path: str | Path, rows: tuple[dict[str, object], ...]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return destination
