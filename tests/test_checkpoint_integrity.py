"""Gate 7 tests: a checkpoint is resumed only when it is provably intact.

A checkpoint stands in for model calls that were already paid for, so resuming
a damaged or foreign one would quietly mix fabricated or mismatched samples
into a client's score file. Every refusal below is exercised directly rather
than through ``run_scoring``, so a specific corruption maps to a specific
message.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agent_audit.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointSample,
    append_checkpoint_sample,
    create_checkpoint,
    load_checkpoint,
    repair_checkpoint_truncated_tail,
)
from agent_audit.models import ProviderScore


CONTEXT: dict[str, object] = {
    "system_name": "Grader",
    "model": "demo-model",
    "base_url": "http://localhost:11434/v1",
    "score_min": 0.0,
    "score_max": 10.0,
    "temperature": 0.0,
    "repeats": 2,
    "input_sha256": "a" * 64,
    "rubric_sha256": "b" * 64,
}


def _sample_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "sample",
        "case_id": "c1",
        "variant_id": "base",
        "variant_type": "baseline",
        "sample_index": 1,
        "input_sha256": "c" * 64,
        "score": 7.0,
        "reason": "clear thesis",
        "raw_content": '{"score": 7.0}',
        "response_id": "resp-1",
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "latency_seconds": 0.42,
    }
    row.update(overrides)
    return row


def _header_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "type": "header",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_id": "run-1",
        "created_at_utc": "2026-09-21T00:00:00+00:00",
        "context": dict(CONTEXT),
    }
    row.update(overrides)
    return row


def _write_checkpoint(*rows: dict[str, Any], trailing_newline: bool = True) -> Path:
    path = Path(tempfile.mkdtemp()) / "checkpoint.jsonl"
    body = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if trailing_newline:
        body += "\n"
    path.write_bytes(body.encode("utf-8"))
    return path


class CheckpointCreationTests(unittest.TestCase):
    def test_creates_a_header_and_returns_the_run_identity(self) -> None:
        path = Path(tempfile.mkdtemp()) / "checkpoint.jsonl"

        run_id, created_at = create_checkpoint(path, dict(CONTEXT))

        header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(header["run_id"], run_id)
        self.assertEqual(header["created_at_utc"], created_at)
        self.assertEqual(header["context"], CONTEXT)

    def test_refuses_to_overwrite_an_existing_checkpoint(self) -> None:
        path = Path(tempfile.mkdtemp()) / "checkpoint.jsonl"
        create_checkpoint(path, dict(CONTEXT))

        with self.assertRaisesRegex(ValueError, "already exists"):
            create_checkpoint(path, dict(CONTEXT))

    def test_appending_requires_an_existing_checkpoint(self) -> None:
        missing = Path(tempfile.mkdtemp()) / "absent.jsonl"
        sample = CheckpointSample(
            case_id="c1",
            variant_id="base",
            variant_type="baseline",
            sample_index=1,
            input_sha256="c" * 64,
            result=ProviderScore(7.0, "ok", "{}"),
            latency_seconds=0.1,
        )

        with self.assertRaisesRegex(ValueError, "does not exist"):
            append_checkpoint_sample(missing, sample)

    def test_appending_repairs_a_missing_trailing_newline(self) -> None:
        """A crash can leave the last record without its newline."""

        path = _write_checkpoint(_header_row(), trailing_newline=False)
        sample = CheckpointSample(
            case_id="c1",
            variant_id="base",
            variant_type="baseline",
            sample_index=1,
            input_sha256="c" * 64,
            result=ProviderScore(7.0, "ok", "{}"),
            latency_seconds=0.1,
        )

        append_checkpoint_sample(path, sample)

        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[1])["case_id"], "c1")


class CheckpointLoadingTests(unittest.TestCase):
    def test_loads_a_well_formed_checkpoint(self) -> None:
        path = _write_checkpoint(_header_row(), _sample_row())

        loaded = load_checkpoint(path, dict(CONTEXT))

        self.assertEqual(loaded.run_id, "run-1")
        self.assertIsNone(loaded.truncated_tail_offset)
        self.assertEqual(list(loaded.samples), [("c1", "base", 1)])
        self.assertAlmostEqual(loaded.samples[("c1", "base", 1)].result.score, 7.0)

    def test_refuses_a_missing_file(self) -> None:
        missing = Path(tempfile.mkdtemp()) / "absent.jsonl"
        with self.assertRaisesRegex(ValueError, "does not exist"):
            load_checkpoint(missing, dict(CONTEXT))

    def test_refuses_an_empty_file(self) -> None:
        path = Path(tempfile.mkdtemp()) / "checkpoint.jsonl"
        path.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "is empty"):
            load_checkpoint(path, dict(CONTEXT))

    def test_refuses_a_blank_line(self) -> None:
        path = Path(tempfile.mkdtemp()) / "checkpoint.jsonl"
        path.write_bytes(
            (json.dumps(_header_row()) + "\n\n" + json.dumps(_sample_row()) + "\n").encode(
                "utf-8"
            )
        )
        with self.assertRaisesRegex(ValueError, "blank line"):
            load_checkpoint(path, dict(CONTEXT))

    def test_refuses_a_line_that_is_not_an_object(self) -> None:
        path = Path(tempfile.mkdtemp()) / "checkpoint.jsonl"
        path.write_bytes((json.dumps(_header_row()) + "\n[1, 2]\n").encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "must contain a JSON object"):
            load_checkpoint(path, dict(CONTEXT))

    def test_refuses_a_first_line_that_is_not_a_header(self) -> None:
        path = _write_checkpoint(_sample_row())
        with self.assertRaisesRegex(ValueError, "first line must be a header"):
            load_checkpoint(path, dict(CONTEXT))

    def test_refuses_an_unsupported_schema_version(self) -> None:
        for version in (2, "1", True):
            with self.subTest(version=version):
                path = _write_checkpoint(
                    _header_row(checkpoint_schema_version=version)
                )
                with self.assertRaisesRegex(ValueError, "Unsupported checkpoint schema"):
                    load_checkpoint(path, dict(CONTEXT))

    def test_refuses_a_header_without_a_run_id(self) -> None:
        path = _write_checkpoint(_header_row(run_id="  "))
        with self.assertRaisesRegex(ValueError, "must be a non-empty string"):
            load_checkpoint(path, dict(CONTEXT))

    def test_refuses_a_context_that_is_not_an_object(self) -> None:
        path = _write_checkpoint(_header_row(context="demo-model"))
        with self.assertRaisesRegex(ValueError, "context must be an object"):
            load_checkpoint(path, dict(CONTEXT))

    def test_refuses_a_context_from_a_different_run(self) -> None:
        """The message must name what differs, so the user can tell why."""

        foreign = dict(CONTEXT)
        foreign["temperature"] = 0.7
        path = _write_checkpoint(_header_row(context=foreign))

        with self.assertRaisesRegex(ValueError, "temperature"):
            load_checkpoint(path, dict(CONTEXT))

    def test_refuses_duplicate_samples(self) -> None:
        path = _write_checkpoint(_header_row(), _sample_row(), _sample_row())
        with self.assertRaisesRegex(ValueError, "duplicate sample"):
            load_checkpoint(path, dict(CONTEXT))


class CheckpointSampleValidationTests(unittest.TestCase):
    def _assert_refused(self, pattern: str, **overrides: Any) -> None:
        path = _write_checkpoint(_header_row(), _sample_row(**overrides))
        with self.assertRaisesRegex(ValueError, pattern):
            load_checkpoint(path, dict(CONTEXT))

    def test_refuses_a_row_that_is_not_a_sample(self) -> None:
        self._assert_refused("must be a sample row", type="note")

    def test_refuses_an_empty_identifier(self) -> None:
        for field in ("case_id", "variant_id"):
            with self.subTest(field=field):
                self._assert_refused("must be a non-empty string", **{field: "  "})

    def test_refuses_an_unsupported_variant_type(self) -> None:
        self._assert_refused("unsupported variant_type", variant_type="rewrite")

    def test_refuses_a_non_integer_sample_index(self) -> None:
        for value in ("1", 1.0, True):
            with self.subTest(value=value):
                self._assert_refused(
                    "sample_index must be an integer", sample_index=value
                )

    def test_refuses_a_sample_index_below_one(self) -> None:
        self._assert_refused("sample_index must be positive", sample_index=0)

    def test_refuses_a_malformed_input_hash(self) -> None:
        for value in ("z" * 64, "abc"):
            with self.subTest(value=value):
                self._assert_refused(
                    "input_sha256 must be a SHA-256 hash", input_sha256=value
                )

    def test_refuses_a_non_numeric_score(self) -> None:
        for value in ("high", None, True):
            with self.subTest(value=value):
                self._assert_refused("score must be numeric", score=value)

    def test_refuses_a_non_finite_score(self) -> None:
        path = Path(tempfile.mkdtemp()) / "checkpoint.jsonl"
        # json.dumps writes bare NaN, which json.loads accepts on the way back.
        body = (
            json.dumps(_header_row())
            + "\n"
            + json.dumps(_sample_row(score=float("nan")))
            + "\n"
        )
        path.write_bytes(body.encode("utf-8"))

        with self.assertRaisesRegex(ValueError, "score must be finite"):
            load_checkpoint(path, dict(CONTEXT))

    def test_refuses_a_negative_latency(self) -> None:
        self._assert_refused("latency_seconds must be non-negative", latency_seconds=-1)

    def test_refuses_non_string_text_fields(self) -> None:
        for field in ("reason", "raw_content"):
            with self.subTest(field=field):
                self._assert_refused(
                    "reason and raw_content must be strings", **{field: 7}
                )

    def test_refuses_a_non_string_response_id(self) -> None:
        self._assert_refused("response_id must be a string or null", response_id=7)

    def test_accepts_a_null_response_id(self) -> None:
        path = _write_checkpoint(_header_row(), _sample_row(response_id=None))

        loaded = load_checkpoint(path, dict(CONTEXT))

        self.assertIsNone(loaded.samples[("c1", "base", 1)].result.response_id)

    def test_refuses_malformed_token_counts(self) -> None:
        for field in ("prompt_tokens", "completion_tokens"):
            for value in (-1, 1.5, True, "120"):
                with self.subTest(field=field, value=value):
                    self._assert_refused(
                        "must be a non-negative integer or null", **{field: value}
                    )

    def test_accepts_null_token_counts(self) -> None:
        path = _write_checkpoint(
            _header_row(), _sample_row(prompt_tokens=None, completion_tokens=None)
        )

        loaded = load_checkpoint(path, dict(CONTEXT))

        self.assertIsNone(loaded.samples[("c1", "base", 1)].result.prompt_tokens)


class TruncatedTailTests(unittest.TestCase):
    """A crash mid-write leaves a partial last line; everything before it stands."""

    def _truncated(self) -> Path:
        path = _write_checkpoint(_header_row(), _sample_row())
        with path.open("ab") as handle:
            handle.write(b'{"type": "sample", "case_id": "c1", "variant_i')
        return path

    def test_reports_the_offset_of_a_partial_final_line(self) -> None:
        path = self._truncated()

        loaded = load_checkpoint(path, dict(CONTEXT))

        self.assertIsNotNone(loaded.truncated_tail_offset)
        self.assertEqual(len(loaded.samples), 1)
        self.assertEqual(loaded.original_size, path.stat().st_size)

    def test_refuses_a_partial_final_line_when_repair_is_not_allowed(self) -> None:
        path = self._truncated()

        with self.assertRaisesRegex(ValueError, "malformed JSON"):
            load_checkpoint(path, dict(CONTEXT), allow_truncated_tail=False)

    def test_refuses_a_malformed_line_that_is_not_the_last(self) -> None:
        path = Path(tempfile.mkdtemp()) / "checkpoint.jsonl"
        path.write_bytes(
            (
                json.dumps(_header_row())
                + "\n"
                + '{"type": "sample", "case_i'
                + "\n"
                + json.dumps(_sample_row())
                + "\n"
            ).encode("utf-8")
        )

        with self.assertRaisesRegex(ValueError, "malformed JSON on line 2"):
            load_checkpoint(path, dict(CONTEXT))

    def test_repair_truncates_only_the_partial_line(self) -> None:
        path = self._truncated()
        loaded = load_checkpoint(path, dict(CONTEXT))
        assert loaded.truncated_tail_offset is not None

        repair_checkpoint_truncated_tail(
            path,
            truncated_tail_offset=loaded.truncated_tail_offset,
            expected_size=loaded.original_size,
        )

        repaired = load_checkpoint(path, dict(CONTEXT))
        self.assertIsNone(repaired.truncated_tail_offset)
        self.assertEqual(len(repaired.samples), 1)

    def test_repair_refuses_a_file_that_changed_while_being_validated(self) -> None:
        path = self._truncated()
        loaded = load_checkpoint(path, dict(CONTEXT))
        assert loaded.truncated_tail_offset is not None
        with path.open("ab") as handle:
            handle.write(b"d")

        with self.assertRaisesRegex(ValueError, "changed while it was being validated"):
            repair_checkpoint_truncated_tail(
                path,
                truncated_tail_offset=loaded.truncated_tail_offset,
                expected_size=loaded.original_size,
            )


if __name__ == "__main__":
    unittest.main()
