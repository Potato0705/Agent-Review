from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from uuid import uuid4

from .models import ProviderScore, VariantType


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckpointSample:
    case_id: str
    variant_id: str
    variant_type: VariantType
    sample_index: int
    input_sha256: str
    result: ProviderScore
    latency_seconds: float

    @property
    def identity(self) -> tuple[str, str, int]:
        return (self.case_id, self.variant_id, self.sample_index)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "sample",
            "case_id": self.case_id,
            "variant_id": self.variant_id,
            "variant_type": self.variant_type,
            "sample_index": self.sample_index,
            "input_sha256": self.input_sha256,
            "score": self.result.score,
            "reason": self.result.reason,
            "raw_content": self.result.raw_content,
            "response_id": self.result.response_id,
            "prompt_tokens": self.result.prompt_tokens,
            "completion_tokens": self.result.completion_tokens,
            "latency_seconds": self.latency_seconds,
        }


@dataclass(frozen=True)
class LoadedCheckpoint:
    run_id: str
    created_at_utc: str
    samples: dict[tuple[str, str, int], CheckpointSample]
    truncated_tail_offset: int | None
    original_size: int


def create_checkpoint(
    path: str | Path, context: dict[str, object]
) -> tuple[str, str]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid4())
    created_at_utc = datetime.now(timezone.utc).isoformat()
    header = {
        "type": "header",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": created_at_utc,
        "context": context,
    }
    encoded = (json.dumps(header, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        with destination.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise ValueError(
            f"Checkpoint already exists: {destination}. Use --resume or choose a new path."
        ) from exc
    return run_id, created_at_utc


def append_checkpoint_sample(path: str | Path, sample: CheckpointSample) -> None:
    destination = Path(path)
    if not destination.exists():
        raise ValueError(f"Checkpoint does not exist: {destination}")
    encoded = (json.dumps(sample.to_dict(), ensure_ascii=False) + "\n").encode("utf-8")
    with destination.open("r+b") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) not in {b"\n", b"\r"}:
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
        handle.seek(0, os.SEEK_END)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(
    path: str | Path,
    expected_context: dict[str, object],
    *,
    allow_truncated_tail: bool = True,
) -> LoadedCheckpoint:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"Checkpoint does not exist: {source}")
    raw = source.read_bytes()
    if not raw:
        raise ValueError(f"Checkpoint is empty: {source}")

    rows: list[dict[str, object]] = []
    valid_offset = 0
    truncated_tail_offset: int | None = None
    lines = raw.splitlines(keepends=True)
    for index, raw_line in enumerate(lines):
        is_last = index == len(lines) - 1
        has_line_ending = raw_line.endswith((b"\n", b"\r"))
        try:
            decoded = raw_line.decode("utf-8").strip()
            if not decoded:
                raise ValueError("Checkpoint contains a blank line.")
            row = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if is_last and not has_line_ending and allow_truncated_tail:
                truncated_tail_offset = valid_offset
                break
            raise ValueError(
                f"Checkpoint contains malformed JSON on line {index + 1}."
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"Checkpoint line {index + 1} must contain a JSON object."
            )
        rows.append(row)
        valid_offset += len(raw_line)

    if not rows:
        raise ValueError("Checkpoint has no valid header.")
    header = rows[0]
    if header.get("type") != "header":
        raise ValueError("Checkpoint first line must be a header.")
    schema_version = header.get("checkpoint_schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported checkpoint schema version.")
    run_id = _required_text(header, "run_id", "checkpoint header")
    created_at_utc = _required_text(header, "created_at_utc", "checkpoint header")
    context = header.get("context")
    if not isinstance(context, dict):
        raise ValueError("Checkpoint header context must be an object.")
    if context != expected_context:
        mismatches = sorted(
            key
            for key in set(context).union(expected_context)
            if context.get(key) != expected_context.get(key)
        )
        raise ValueError(
            "Checkpoint context does not match this scoring run: "
            + ", ".join(mismatches)
            + "."
        )

    samples: dict[tuple[str, str, int], CheckpointSample] = {}
    for line_number, row in enumerate(rows[1:], start=2):
        sample = _parse_sample(row, line_number)
        if sample.identity in samples:
            raise ValueError(
                f"Checkpoint contains duplicate sample {sample.identity!r}."
            )
        samples[sample.identity] = sample
    return LoadedCheckpoint(
        run_id,
        created_at_utc,
        samples,
        truncated_tail_offset,
        len(raw),
    )


def repair_checkpoint_truncated_tail(
    path: str | Path, *, truncated_tail_offset: int, expected_size: int
) -> None:
    destination = Path(path)
    with destination.open("r+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() != expected_size:
            raise ValueError(
                "Checkpoint changed while it was being validated; refusing to repair it."
            )
        handle.truncate(truncated_tail_offset)
        handle.flush()
        os.fsync(handle.fileno())


def _parse_sample(row: dict[str, object], line_number: int) -> CheckpointSample:
    if row.get("type") != "sample":
        raise ValueError(f"Checkpoint line {line_number} must be a sample row.")
    case_id = _required_text(row, "case_id", f"checkpoint line {line_number}")
    variant_id = _required_text(row, "variant_id", f"checkpoint line {line_number}")
    variant_type = _required_text(
        row, "variant_type", f"checkpoint line {line_number}"
    )
    if variant_type not in {"baseline", "gaming", "degradation", "paraphrase"}:
        raise ValueError(
            f"Checkpoint line {line_number} has an unsupported variant_type."
        )
    sample_index = row.get("sample_index")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise ValueError(
            f"Checkpoint line {line_number} sample_index must be an integer."
        )
    if sample_index < 1:
        raise ValueError(
            f"Checkpoint line {line_number} sample_index must be positive."
        )
    input_sha256 = _required_text(
        row, "input_sha256", f"checkpoint line {line_number}"
    )
    if len(input_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in input_sha256
    ):
        raise ValueError(
            f"Checkpoint line {line_number} input_sha256 must be a SHA-256 hash."
        )

    score = _finite_number(row.get("score"), "score", line_number)
    latency_seconds = _finite_number(
        row.get("latency_seconds"), "latency_seconds", line_number
    )
    if latency_seconds < 0:
        raise ValueError(
            f"Checkpoint line {line_number} latency_seconds must be non-negative."
        )
    reason = row.get("reason")
    raw_content = row.get("raw_content")
    if not isinstance(reason, str) or not isinstance(raw_content, str):
        raise ValueError(
            f"Checkpoint line {line_number} reason and raw_content must be strings."
        )
    response_id = row.get("response_id")
    if response_id is not None and not isinstance(response_id, str):
        raise ValueError(
            f"Checkpoint line {line_number} response_id must be a string or null."
        )
    prompt_tokens = _optional_non_negative_int(
        row.get("prompt_tokens"), "prompt_tokens", line_number
    )
    completion_tokens = _optional_non_negative_int(
        row.get("completion_tokens"), "completion_tokens", line_number
    )
    return CheckpointSample(
        case_id=case_id,
        variant_id=variant_id,
        variant_type=variant_type,  # type: ignore[arg-type]
        sample_index=sample_index,
        input_sha256=input_sha256.lower(),
        result=ProviderScore(
            score=score,
            reason=reason,
            raw_content=raw_content,
            response_id=response_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        latency_seconds=latency_seconds,
    )


def _required_text(row: dict[str, object], key: str, location: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} field {key!r} must be a non-empty string.")
    return value.strip()


def _finite_number(value: object, key: str, line_number: int) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Checkpoint line {line_number} {key} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Checkpoint line {line_number} {key} must be numeric."
        ) from exc
    if not math.isfinite(number):
        raise ValueError(f"Checkpoint line {line_number} {key} must be finite.")
    return number


def _optional_non_negative_int(
    value: object, key: str, line_number: int
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"Checkpoint line {line_number} {key} must be a non-negative integer or null."
        )
    return value
