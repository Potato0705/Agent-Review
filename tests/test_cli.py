"""CLI tests: argument wiring, dispatch, and scoring-manifest validation.

The manifest is the evidence that two audits were produced under the same
conditions. ``compare`` refuses a pair whose fingerprints differ, but that
defence only works if ``audit`` refuses to copy an unverified manifest into its
JSON in the first place, so those refusals are exercised here.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from agent_audit.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[1]
DEMO_CSV = ROOT / "examples" / "demo_scores.csv"


def _manifest(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_sha256": "a" * 64,
        "rubric_sha256": "b" * 64,
        "record_count": 12,
        "repeats": 3,
        "temperature": 0.2,
        "model": "demo-model",
        "score_range": [0.0, 10.0],
    }
    payload.update(overrides)
    return payload


class ParserTests(unittest.TestCase):
    def test_every_subcommand_is_registered(self) -> None:
        parser = build_parser()

        for argv, expected in (
            (["audit", "--input", "a.csv", "--report", "r.md"], "audit"),
            (
                [
                    "score",
                    "--input", "a.csv",
                    "--output", "o.csv",
                    "--rubric-file", "r.md",
                    "--model", "m",
                ],
                "score",
            ),
            (
                ["compare", "--reference", "a.json", "--candidate", "b.json", "--report", "r.md"],
                "compare",
            ),
        ):
            with self.subTest(command=expected):
                self.assertEqual(parser.parse_args(argv).command, expected)

    def test_audit_defaults_match_the_documented_thresholds(self) -> None:
        args = build_parser().parse_args(["audit", "--input", "a.csv", "--report", "r.md"])

        self.assertEqual(args.invariance_tolerance, 0.5)
        self.assertEqual(args.min_degradation_drop, 1.0)
        self.assertEqual(args.gaming_tolerance, 0.0)
        self.assertEqual(args.data_provenance, "unspecified")
        self.assertIsNone(args.score_min)
        self.assertIsNone(args.score_max)

    def test_score_defaults_do_not_repeat_or_keep_raw_output(self) -> None:
        args = build_parser().parse_args(
            ["score", "--input", "a.csv", "--output", "o.csv", "--rubric-file", "r.md", "--model", "m"]
        )

        self.assertEqual(args.repeats, 1)
        self.assertIsNone(args.raw_output)
        self.assertIsNone(args.checkpoint)
        self.assertFalse(args.resume)
        self.assertEqual(args.api_key_env, "OPENAI_API_KEY")

    def test_an_undeclared_provenance_choice_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["audit", "--input", "a.csv", "--report", "r.md",
                 "--data-provenance", "client-private"]
            )

    def test_a_missing_subcommand_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])


class MainDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = Path(tempfile.mkdtemp())

    def test_audit_writes_every_requested_artefact(self) -> None:
        report = self.work / "report.md"
        result = self.work / "result.json"
        page = self.work / "report.html"

        status = main(
            [
                "audit",
                "--input", str(DEMO_CSV),
                "--report", str(report),
                "--json", str(result),
                "--html", str(page),
                "--score-min", "0",
                "--score-max", "10",
                "--data-provenance", "synthetic",
            ]
        )

        self.assertEqual(status, 0)
        self.assertIn("效度余量", report.read_text(encoding="utf-8"))
        self.assertEqual(json.loads(result.read_text(encoding="utf-8"))["case_count"], 3)
        self.assertIn("<!doctype html>", page.read_text(encoding="utf-8").lower())

    def test_audit_reports_a_bad_input_as_a_usage_error(self) -> None:
        """A malformed input must exit with a message, not a traceback."""

        missing = self.work / "absent.csv"

        with self.assertRaises(SystemExit) as caught:
            main(["audit", "--input", str(missing), "--report", str(self.work / "r.md")])

        self.assertEqual(caught.exception.code, 2)

    def test_compare_reports_an_unverified_pair_as_a_usage_error(self) -> None:
        reference = self.work / "reference.json"
        reference.write_text("{}", encoding="utf-8")

        with self.assertRaises(SystemExit) as caught:
            main(
                [
                    "compare",
                    "--reference", str(reference),
                    "--candidate", str(reference),
                    "--report", str(self.work / "r.md"),
                ]
            )

        self.assertEqual(caught.exception.code, 2)

    def test_score_reports_a_missing_api_key_as_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(
                [
                    "score",
                    "--input", str(DEMO_CSV),
                    "--output", str(self.work / "scores.csv"),
                    "--rubric-file", str(self.work / "rubric.md"),
                    "--model", "m",
                    "--api-key-env", "AGENT_REVIEW_UNSET_KEY_FOR_TESTS",
                ]
            )

        self.assertEqual(caught.exception.code, 2)


class ScoreCommandGuardTests(unittest.TestCase):
    """Guards that run before a single paid model call is made."""

    def setUp(self) -> None:
        self.work = Path(tempfile.mkdtemp())
        self.rubric = self.work / "rubric.md"
        self.rubric.write_text("Score the essay.", encoding="utf-8")

    def _args(self, **overrides: Any) -> list[str]:
        argv = [
            "score",
            "--input", str(DEMO_CSV),
            "--output", str(self.work / "scores.csv"),
            "--rubric-file", str(self.rubric),
            "--model", "m",
            # A local endpoint needs no API key, so these tests reach the
            # guards rather than stopping at the credential check.
            "--base-url", "http://127.0.0.1:9/v1",
        ]
        for flag, value in overrides.items():
            argv += [f"--{flag.replace('_', '-')}"] + ([] if value is None else [str(value)])
        return argv

    def test_resume_without_a_checkpoint_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(self._args(resume=None))

        self.assertEqual(caught.exception.code, 2)

    def test_a_missing_rubric_file_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            main(
                [
                    "score",
                    "--input", str(DEMO_CSV),
                    "--output", str(self.work / "scores.csv"),
                    "--rubric-file", str(self.work / "absent.md"),
                    "--model", "m",
                    "--base-url", "http://127.0.0.1:9/v1",
                ]
            )

        self.assertEqual(caught.exception.code, 2)

    def test_repeated_scoring_without_raw_output_warns_about_lost_reasons(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(self._args(repeats=3))

        self.assertIn("keeps only the first", stderr.getvalue())

    def test_using_a_checkpoint_warns_that_it_holds_model_replies(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(self._args(checkpoint=str(self.work / "ck.jsonl")))

        self.assertIn("handled as sensitive data", stderr.getvalue())


class ManifestValidationTests(unittest.TestCase):
    """An audit only records a scoring context it has actually checked."""

    def setUp(self) -> None:
        self.work = Path(tempfile.mkdtemp())
        self.input_csv = self.work / "scores.csv"
        self.input_csv.write_bytes(DEMO_CSV.read_bytes())
        self.manifest_path = self.work / "scores.manifest.json"
        self.report = self.work / "report.md"
        self.result = self.work / "result.json"

    def _run(self, manifest: dict[str, Any] | None, *, explicit: bool = False) -> int:
        if manifest is not None:
            self.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )
        argv = [
            "audit",
            "--input", str(self.input_csv),
            "--report", str(self.report),
            "--json", str(self.result),
            "--score-min", "0",
            "--score-max", "10",
            "--data-provenance", "synthetic",
        ]
        if explicit:
            argv += ["--manifest", str(self.manifest_path)]
        return main(argv)

    def _assert_refused(self, **overrides: Any) -> None:
        with self.assertRaises(SystemExit) as caught:
            self._run(_manifest(**overrides))
        self.assertEqual(caught.exception.code, 2)

    def test_a_valid_manifest_is_copied_into_the_audit_json(self) -> None:
        self.assertEqual(self._run(_manifest()), 0)

        context = json.loads(self.result.read_text(encoding="utf-8"))["comparison_context"]
        self.assertEqual(context["input_sha256"], "a" * 64)
        self.assertEqual(context["model"], "demo-model")
        self.assertEqual(context["repeats"], 3)
        self.assertAlmostEqual(context["temperature"], 0.2)

    def test_an_adjacent_manifest_is_picked_up_without_being_named(self) -> None:
        self.assertEqual(self._run(_manifest()), 0)
        self.assertIn(
            "comparison_context", json.loads(self.result.read_text(encoding="utf-8"))
        )

    def test_no_manifest_means_no_comparison_context(self) -> None:
        self.assertEqual(self._run(None), 0)

        payload = json.loads(self.result.read_text(encoding="utf-8"))
        self.assertIsNone(payload["comparison_context"])

    def test_an_explicitly_named_manifest_must_exist(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self._run(None, explicit=True)
        self.assertEqual(caught.exception.code, 2)

    def test_refuses_a_malformed_manifest(self) -> None:
        self.manifest_path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            self._run(None)
        self.assertEqual(caught.exception.code, 2)

    def test_refuses_a_manifest_that_is_not_an_object(self) -> None:
        self.manifest_path.write_text("[1, 2]", encoding="utf-8")
        with self.assertRaises(SystemExit) as caught:
            self._run(None)
        self.assertEqual(caught.exception.code, 2)

    def test_refuses_malformed_hashes(self) -> None:
        for field in ("input_sha256", "rubric_sha256"):
            for value in ("short", "z" * 64, 12345, None):
                with self.subTest(field=field, value=value):
                    self._assert_refused(**{field: value})

    def test_refuses_a_record_count_that_is_not_an_integer(self) -> None:
        for value in ("12", 12.0, True, None):
            with self.subTest(value=value):
                self._assert_refused(record_count=value)

    def test_refuses_a_record_count_that_contradicts_the_csv(self) -> None:
        self._assert_refused(record_count=11)

    def test_refuses_an_unnamed_model(self) -> None:
        for value in ("  ", 7, None):
            with self.subTest(value=value):
                self._assert_refused(model=value)

    def test_refuses_a_non_positive_repeat_count(self) -> None:
        for value in (0, -1, 1.5, True, "3"):
            with self.subTest(value=value):
                self._assert_refused(repeats=value)

    def test_refuses_a_non_numeric_temperature(self) -> None:
        for value in ("warm", None, True):
            with self.subTest(value=value):
                self._assert_refused(temperature=value)

    def test_refuses_a_non_finite_temperature(self) -> None:
        self._assert_refused(temperature=float("inf"))

    def test_refuses_a_malformed_score_range(self) -> None:
        for value in ([0.0], "0-10", [0.0, 10.0, 20.0], [True, False], ["0", "ten"]):
            with self.subTest(value=value):
                self._assert_refused(score_range=value)

    def test_refuses_a_score_range_that_contradicts_the_audit_config(self) -> None:
        self._assert_refused(score_range=[0.0, 100.0])

    def test_accepts_a_manifest_without_a_repeat_count(self) -> None:
        manifest = _manifest()
        del manifest["repeats"]

        self.assertEqual(self._run(manifest), 0)

        context = json.loads(self.result.read_text(encoding="utf-8"))["comparison_context"]
        self.assertEqual(context["repeats"], 1)


class ModuleEntryPointTests(unittest.TestCase):
    """``python -m agent_audit`` is documented in the README, so it must work.

    Every other test calls ``main`` in-process, so the four-line ``__main__``
    shim that the README tells users to run is otherwise never executed.
    """

    def _run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "agent_audit", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUTF8": "1"},
        )

    def test_the_module_entry_point_runs_an_audit(self) -> None:
        work = Path(tempfile.mkdtemp())
        report = work / "report.md"
        result = work / "result.json"

        completed = self._run(
            "audit",
            "--input", str(DEMO_CSV),
            "--report", str(report),
            "--json", str(result),
            "--score-min", "0",
            "--score-max", "10",
            "--data-provenance", "synthetic",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            json.loads(result.read_text(encoding="utf-8"))["case_count"], 3
        )
        self.assertIn("效度余量", report.read_text(encoding="utf-8"))

    def test_the_module_entry_point_reports_a_bad_input(self) -> None:
        completed = self._run(
            "audit",
            "--input", str(ROOT / "examples" / "does_not_exist.csv"),
            "--report", str(Path(tempfile.mkdtemp()) / "report.md"),
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("does not exist", completed.stderr)


if __name__ == "__main__":
    unittest.main()
