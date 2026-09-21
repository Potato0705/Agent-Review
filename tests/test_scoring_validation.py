"""Fail-closed tests for ``run_scoring`` arguments and provider results.

These guards run before and around paid model calls. A bad argument caught here
costs nothing; the same mistake caught later has already spent the user's quota
and may have written a misleading score file.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agent_audit.models import ProviderScore, ScoringCase
from agent_audit.scoring import run_scoring


def _cases() -> list[ScoringCase]:
    return [
        ScoringCase("c1", "base", "baseline", "the original essay"),
        ScoringCase("c1", "game", "gaming", "the padded essay"),
        ScoringCase("c1", "drop", "degradation", "the gutted essay"),
    ]


class _FixedScorer:
    """Returns whatever it is told to, so result validation can be probed."""

    def __init__(self, **overrides: Any) -> None:
        self.overrides = overrides
        self.calls = 0

    def score(self, text: str, rubric: str) -> ProviderScore:
        self.calls += 1
        fields: dict[str, Any] = {
            "score": 7.0,
            "reason": "ok",
            "raw_content": "{}",
            "response_id": "r1",
            "prompt_tokens": 10,
            "completion_tokens": 2,
        }
        fields.update(self.overrides)
        return ProviderScore(**fields)


def _run(scorer: Any = None, *, cases: list[ScoringCase] | None = None, **overrides: Any):
    kwargs: dict[str, Any] = {
        "system_name": "Grader",
        "provider_name": "openai-compatible",
        "model": "demo-model",
        "base_url": "http://localhost:11434/v1",
        "score_min": 0.0,
        "score_max": 10.0,
        "temperature": 0.0,
    }
    kwargs.update(overrides)
    return run_scoring(
        _cases() if cases is None else cases,
        scorer or _FixedScorer(),
        overrides.pop("rubric", "Score the essay."),
        **kwargs,
    )


class RunScoringArgumentTests(unittest.TestCase):
    def test_requires_at_least_one_case(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one scoring case"):
            _run(cases=[])

    def test_requires_a_system_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "system_name must not be empty"):
            _run(system_name="   ")

    def test_requires_a_rubric(self) -> None:
        with self.assertRaisesRegex(ValueError, "rubric must not be empty"):
            run_scoring(
                _cases(),
                _FixedScorer(),
                "   ",
                system_name="Grader",
                provider_name="openai-compatible",
                model="demo-model",
                base_url="http://localhost:11434/v1",
                score_min=0.0,
                score_max=10.0,
                temperature=0.0,
            )

    def test_requires_a_finite_score_range(self) -> None:
        for field in ("score_min", "score_max"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    _run(**{field: float("nan")})

    def test_requires_an_ordered_score_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "score_max must be greater"):
            _run(score_min=10.0, score_max=1.0)

    def test_requires_provider_identity_fields(self) -> None:
        for field in ("provider_name", "model", "base_url"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"{field} must not be empty"):
                    _run(**{field: "  "})

    def test_requires_a_finite_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature must be finite"):
            _run(temperature=float("inf"))

    def test_requires_a_repeat_count_in_range(self) -> None:
        for value in (0, -1, 101, True, 2.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "repeats must be between 1 and 100"):
                    _run(repeats=value)

    def test_resume_requires_a_checkpoint_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "resume requires checkpoint_path"):
            _run(resume=True)

    def test_rejects_duplicate_case_and_variant_pairs(self) -> None:
        cases = _cases()
        cases.append(ScoringCase("c1", "base", "paraphrase", "a duplicate id"))

        with self.assertRaisesRegex(ValueError, "duplicate case_id/variant_id"):
            _run(cases=cases)

    def test_requires_exactly_one_baseline_per_case(self) -> None:
        cases = _cases()
        cases.append(ScoringCase("c1", "base2", "baseline", "a second baseline"))

        with self.assertRaisesRegex(ValueError, "exactly one baseline"):
            _run(cases=cases)

    def test_requires_the_paired_validity_variants(self) -> None:
        cases = [
            ScoringCase("c1", "base", "baseline", "the original essay"),
            ScoringCase("c1", "game", "gaming", "the padded essay"),
        ]

        with self.assertRaisesRegex(ValueError, "missing paired validity variants"):
            _run(cases=cases)

    def test_a_valid_run_scores_every_case(self) -> None:
        scorer = _FixedScorer()

        run = _run(scorer)

        self.assertEqual(scorer.calls, 3)
        self.assertEqual(len(run.records), 3)
        self.assertEqual(run.manifest["record_count"], 3)
        self.assertEqual(run.manifest["repeats"], 1)


class ProviderResultValidationTests(unittest.TestCase):
    """A scorer is untrusted, even when it is a local object."""

    def test_rejects_a_score_outside_the_requested_range(self) -> None:
        for value in (10.5, -0.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "out-of-range value"):
                    _run(_FixedScorer(score=value))

    def test_rejects_a_non_numeric_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            _run(_FixedScorer(score="7"))

    def test_rejects_a_boolean_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            _run(_FixedScorer(score=True))

    def test_rejects_a_non_finite_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "out-of-range value"):
            _run(_FixedScorer(score=float("nan")))

    def test_rejects_non_string_text_fields(self) -> None:
        for field in ("reason", "raw_content"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "must be strings"):
                    _run(_FixedScorer(**{field: 7}))

    def test_rejects_a_non_string_response_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "response_id must be a string or None"):
            _run(_FixedScorer(response_id=7))

    def test_rejects_malformed_token_counts(self) -> None:
        for field in ("prompt_tokens", "completion_tokens"):
            for value in (-1, 1.5, True, "10"):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(
                        ValueError, "must be a non-negative integer or None"
                    ):
                        _run(_FixedScorer(**{field: value}))

    def test_missing_token_counts_leave_the_manifest_total_unset(self) -> None:
        run = _run(_FixedScorer(prompt_tokens=None))

        self.assertIsNone(run.manifest["prompt_tokens_total"])
        self.assertEqual(run.manifest["completion_tokens_total"], 6)


class CheckpointScopeTests(unittest.TestCase):
    def test_refuses_a_checkpoint_written_for_different_cases(self) -> None:
        """A different case set changes the context hash, so resume is refused."""

        work = Path(tempfile.mkdtemp())
        checkpoint = work / "checkpoint.jsonl"
        _run(checkpoint_path=checkpoint)

        extra_cases = [
            ScoringCase("c2", "base", "baseline", "a different essay"),
            ScoringCase("c2", "game", "gaming", "a different padded essay"),
            ScoringCase("c2", "drop", "degradation", "a different gutted essay"),
        ]

        with self.assertRaisesRegex(ValueError, "context does not match"):
            _run(cases=extra_cases, checkpoint_path=checkpoint, resume=True)

    def test_refuses_a_checkpoint_holding_a_sample_outside_this_run(self) -> None:
        """A tampered checkpoint keeps its context but gains a foreign sample.

        The header hash still matches, so only the per-sample scope check can
        catch it. Without that check the extra sample would be resumed as if
        this run had paid for it.
        """

        work = Path(tempfile.mkdtemp())
        checkpoint = work / "checkpoint.jsonl"
        _run(checkpoint_path=checkpoint)

        lines = checkpoint.read_text(encoding="utf-8").splitlines()
        smuggled = json.loads(lines[1])
        smuggled["case_id"] = "c9"
        smuggled["variant_id"] = "smuggled"
        checkpoint.write_text(
            "\n".join(lines + [json.dumps(smuggled, ensure_ascii=False)]) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaisesRegex(ValueError, "samples outside this run"):
            _run(checkpoint_path=checkpoint, resume=True)


if __name__ == "__main__":
    unittest.main()
