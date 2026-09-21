from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest

from agent_audit.models import ProviderScore, ScoringCase
from agent_audit.scoring import run_scoring


class _DeterministicScorer:
    def __init__(self, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.calls = 0

    def score(self, text: str, rubric: str) -> ProviderScore:
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("injected provider failure")
        self.calls += 1
        match = re.search(r"SCORE=([0-9.]+)", text)
        score = float(match.group(1)) if match else 0.0
        return ProviderScore(
            score=score,
            reason=f"reason-{self.calls}",
            raw_content=f'{{"score": {score}}}',
            response_id=f"response-{self.calls}",
            prompt_tokens=10,
            completion_tokens=2,
        )


class _NoCallScorer:
    def score(self, text: str, rubric: str) -> ProviderScore:
        raise AssertionError("resume unexpectedly called the provider")


class _InvalidScoreScorer:
    def score(self, text: str, rubric: str) -> ProviderScore:
        return ProviderScore(
            score="7",  # type: ignore[arg-type]
            reason="invalid",
            raw_content="{}",
        )


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cases = [
            ScoringCase("c1", "base", "baseline", "SCORE=7.0"),
            ScoringCase("c1", "game", "gaming", "SCORE=6.5"),
            ScoringCase("c1", "drop", "degradation", "SCORE=5.0"),
        ]

    def _run(
        self,
        scorer: object,
        checkpoint: Path,
        *,
        resume: bool,
        rubric: str = "Test rubric",
    ):
        return run_scoring(
            self.cases,
            scorer,  # type: ignore[arg-type]
            rubric,
            system_name="Checkpoint System",
            provider_name="test-provider",
            model="test-model",
            base_url="https://provider.example/v1",
            score_min=0,
            score_max=10,
            temperature=0.2,
            repeats=2,
            checkpoint_path=checkpoint,
            resume=resume,
        )

    def test_interrupted_run_resumes_only_missing_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "run.checkpoint.jsonl"
            failing = _DeterministicScorer(fail_after=3)

            with self.assertRaisesRegex(RuntimeError, "injected provider failure"):
                self._run(failing, checkpoint, resume=False)

            self.assertEqual(failing.calls, 3)
            self.assertEqual(len(checkpoint.read_text(encoding="utf-8").splitlines()), 4)
            self.assertNotIn("api_key", checkpoint.read_text(encoding="utf-8"))

            completing = _DeterministicScorer()
            run = self._run(completing, checkpoint, resume=True)

            self.assertEqual(completing.calls, 3)
            self.assertEqual(run.manifest["resumed_sample_count"], 3)
            self.assertEqual(run.manifest["new_sample_count"], 3)
            self.assertEqual(run.manifest["sample_count"], 6)
            self.assertEqual(run.manifest["prompt_tokens_total"], 60)
            self.assertEqual(run.manifest["completion_tokens_total"], 12)
            self.assertEqual(
                sum(bool(trace["resumed_from_checkpoint"]) for trace in run.traces),
                3,
            )
            self.assertEqual([record.score for record in run.records], [7.0, 6.5, 5.0])

            replay = self._run(_NoCallScorer(), checkpoint, resume=True)
            self.assertEqual(replay.manifest["resumed_sample_count"], 6)
            self.assertEqual(replay.manifest["new_sample_count"], 0)

    def test_resume_rejects_context_mismatch_before_provider_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "run.checkpoint.jsonl"
            self._run(_DeterministicScorer(), checkpoint, resume=False)
            with checkpoint.open("ab") as handle:
                handle.write(b'{"type": "sample"')
            original = checkpoint.read_bytes()

            with self.assertRaisesRegex(ValueError, "Checkpoint context does not match"):
                self._run(
                    _NoCallScorer(), checkpoint, resume=True, rubric="Changed rubric"
                )

            self.assertEqual(checkpoint.read_bytes(), original)

    def test_fresh_run_refuses_to_overwrite_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "run.checkpoint.jsonl"
            self._run(_DeterministicScorer(), checkpoint, resume=False)
            original = checkpoint.read_bytes()

            with self.assertRaisesRegex(ValueError, "Checkpoint already exists"):
                self._run(_NoCallScorer(), checkpoint, resume=False)

            self.assertEqual(checkpoint.read_bytes(), original)

    def test_resume_repairs_only_unterminated_truncated_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "run.checkpoint.jsonl"
            self._run(_DeterministicScorer(), checkpoint, resume=False)
            with checkpoint.open("ab") as handle:
                handle.write(b'{"type": "sample"')

            run = self._run(_NoCallScorer(), checkpoint, resume=True)

            self.assertTrue(run.manifest["checkpoint_repaired_truncated_tail"])
            self.assertEqual(run.manifest["resumed_sample_count"], 6)
            self.assertTrue(checkpoint.read_bytes().endswith(b"\n"))

    def test_resume_rejects_terminated_malformed_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "run.checkpoint.jsonl"
            self._run(_DeterministicScorer(), checkpoint, resume=False)
            with checkpoint.open("ab") as handle:
                handle.write(b"not-json\n")

            with self.assertRaisesRegex(ValueError, "malformed JSON"):
                self._run(_NoCallScorer(), checkpoint, resume=True)

    def test_semantic_mismatch_does_not_repair_truncated_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "run.checkpoint.jsonl"
            self._run(_DeterministicScorer(), checkpoint, resume=False)
            lines = checkpoint.read_text(encoding="utf-8").splitlines()
            sample = json.loads(lines[1])
            sample["variant_type"] = "gaming"
            lines[1] = json.dumps(sample)
            corrupted = ("\n".join(lines) + "\n" + '{"type": "sample"').encode(
                "utf-8"
            )
            checkpoint.write_bytes(corrupted)

            with self.assertRaisesRegex(ValueError, "metadata does not match"):
                self._run(_NoCallScorer(), checkpoint, resume=True)

            self.assertEqual(checkpoint.read_bytes(), corrupted)

    def test_invalid_provider_result_is_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "run.checkpoint.jsonl"

            with self.assertRaisesRegex(ValueError, "score must be numeric"):
                self._run(_InvalidScoreScorer(), checkpoint, resume=False)

            self.assertEqual(len(checkpoint.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
