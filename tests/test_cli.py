"""CLI tests: argument wiring, dispatch, and scoring-manifest validation.

The manifest is the evidence that two audits were produced under the same
conditions. ``compare`` refuses a pair whose fingerprints differ, but that
defence only works if ``audit`` refuses to copy an unverified manifest into its
JSON in the first place, so those refusals are exercised here.
"""

from __future__ import annotations

import contextlib
import csv
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
from agent_audit.io import load_scoring_cases

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

    def test_variant_origin_defaults_to_unspecified(self) -> None:
        args = build_parser().parse_args(["audit", "--input", "a.csv", "--report", "r.md"])

        self.assertEqual(args.variant_origin, "unspecified")

    def test_an_undeclared_variant_origin_choice_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["audit", "--input", "a.csv", "--report", "r.md",
                 "--variant-origin", "auto"]
            )

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

    def test_audit_records_the_declared_variant_origin(self) -> None:
        """`machine-generated` needs proof, so `mixed` is used here.

        The verified path lives in GenerationProvenanceTests.
        """

        result = self.work / "result.json"

        status = main(
            [
                "audit",
                "--input", str(DEMO_CSV),
                "--report", str(self.work / "report.md"),
                "--json", str(result),
                "--variant-origin", "mixed",
            ]
        )

        self.assertEqual(status, 0)
        payload = json.loads(result.read_text(encoding="utf-8"))
        self.assertEqual(payload["config"]["variant_origin"], "mixed")

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


class GenerateCommandTests(unittest.TestCase):
    BASELINES = ROOT / "examples" / "essay_baselines.csv"

    def setUp(self) -> None:
        self.work = Path(tempfile.mkdtemp())
        self.output = self.work / "cases.csv"

    def _run(self, *extra: str) -> int:
        return main(
            [
                "generate",
                "--input", str(self.BASELINES),
                "--output", str(self.output),
                *extra,
            ]
        )

    def test_parser_defaults_cover_both_required_families(self) -> None:
        args = build_parser().parse_args(
            ["generate", "--input", "a.csv", "--output", "b.csv"]
        )

        self.assertEqual(args.gaming, "verbose_padding,rubric_flattery")
        self.assertEqual(args.degradation, "remove_evidence,unsupported_assertion")
        self.assertIsNone(args.paraphrase)
        self.assertEqual(args.seed, 0)

    def test_generates_a_csv_the_scoring_loader_accepts(self) -> None:
        self.assertEqual(self._run(), 0)

        cases = load_scoring_cases(self.output)
        self.assertEqual(len(cases), 25)
        self.assertEqual(len({case.case_id for case in cases}), 5)

    def test_writes_a_manifest_next_to_the_output_by_default(self) -> None:
        self._run()

        manifest_path = self.output.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["requires_human_review"])
        self.assertEqual(manifest["row_count"], 25)
        self.assertEqual(manifest["input_sha256"], manifest["input_sha256"].lower())
        self.assertEqual(len(manifest["input_sha256"]), 64)

    def test_the_manifest_path_can_be_named(self) -> None:
        target = self.work / "named.json"

        self._run("--manifest", str(target))

        self.assertTrue(target.exists())

    def test_the_same_seed_reproduces_the_output_byte_for_byte(self) -> None:
        self._run("--seed", "5")
        first = self.output.read_bytes()
        self.output.unlink()
        self._run("--seed", "5")

        self.assertEqual(self.output.read_bytes(), first)

    def test_paraphrase_is_not_generated_by_default(self) -> None:
        self._run()

        cases = load_scoring_cases(self.output)

        self.assertNotIn("paraphrase", {case.variant_type for case in cases})

    def test_paraphrase_is_generated_when_the_text_allows_it(self) -> None:
        source = self.work / "baselines.csv"
        source.write_text(
            "case_id,text,evidence_sentences,notes\n"
            "c1,学校应推迟上课。一项调查显示睡眠充足更稳。因此可以先试行。,2,\n",
            encoding="utf-8",
            newline="\n",
        )

        status = main(
            [
                "generate",
                "--input", str(source),
                "--output", str(self.output),
                "--paraphrase", "connective_substitution",
            ]
        )

        self.assertEqual(status, 0)
        cases = load_scoring_cases(self.output)
        self.assertIn("paraphrase", {case.variant_type for case in cases})

    def test_a_baseline_with_no_listed_connective_refuses_and_names_itself(self) -> None:
        """The conservative rewrite cannot handle every text, and says which."""

        with self.assertRaises(SystemExit) as caught:
            self._run("--paraphrase", "connective_substitution")

        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_refuses_to_overwrite_an_existing_case_file(self) -> None:
        """The README invites hand-editing these rows, so overwriting loses work."""

        self._run()
        rows = list(csv.DictReader(self.output.open(encoding="utf-8")))
        rows[1]["text"] = "【手工改写的变体】"
        with self.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaises(SystemExit) as caught:
            self._run()

        self.assertEqual(caught.exception.code, 2)
        preserved = list(csv.DictReader(self.output.open(encoding="utf-8")))
        self.assertEqual(preserved[1]["text"], "【手工改写的变体】")

    def test_an_unknown_strategy_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self._run("--gaming", "keyword_stuffing")

        self.assertEqual(caught.exception.code, 2)

    def test_an_unannotated_baseline_is_a_usage_error(self) -> None:
        source = self.work / "baselines.csv"
        source.write_text(
            "case_id,text,evidence_sentences,notes\nc1,甲。乙。,,\n",
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaises(SystemExit) as caught:
            main(
                [
                    "generate",
                    "--input", str(source),
                    "--output", str(self.output),
                ]
            )

        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(self.output.exists())


class AppendModeTests(unittest.TestCase):
    BASELINES = ROOT / "examples" / "essay_baselines.csv"

    def setUp(self) -> None:
        self.work = Path(tempfile.mkdtemp())
        self.output = self.work / "cases.csv"
        self.manifest = self.output.with_suffix(".manifest.json")

    def _run(self, *extra: str) -> int:
        return main(
            [
                "generate",
                "--input", str(self.BASELINES),
                "--output", str(self.output),
                *extra,
            ]
        )

    def _rows(self) -> list[dict[str, str]]:
        return list(csv.DictReader(self.output.open(encoding="utf-8")))

    def _edit_first_gaming_row(self) -> str:
        rows = self._rows()
        target = next(r for r in rows if r["variant_id"] == "gaming_verbose_padding")
        target["text"] = "【手工改写的变体】"
        with self.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return target["text"]

    def test_append_adds_a_newly_selected_strategy(self) -> None:
        self._run("--gaming", "verbose_padding")
        before = {(r["case_id"], r["variant_id"]) for r in self._rows()}

        self.assertEqual(self._run("--append"), 0)

        after = {(r["case_id"], r["variant_id"]) for r in self._rows()}
        self.assertTrue(before < after)
        self.assertIn(("school_start", "gaming_rubric_flattery"), after)

    def test_append_keeps_hand_edits_untouched(self) -> None:
        self._run("--gaming", "verbose_padding")
        edited = self._edit_first_gaming_row()

        self._run("--append")

        kept = next(
            r for r in self._rows() if r["variant_id"] == "gaming_verbose_padding"
        )
        self.assertEqual(kept["text"], edited)

    def test_a_clean_append_keeps_the_set_machine_generated(self) -> None:
        self._run("--gaming", "verbose_padding")

        self._run("--append")

        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["set_origin"], "machine-generated")
        self.assertGreater(len(manifest["merge"]["appended"]), 0)

    def test_an_edited_row_marks_the_set_mixed(self) -> None:
        self._run("--gaming", "verbose_padding")
        self._edit_first_gaming_row()

        self._run("--append")

        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["set_origin"], "mixed")
        self.assertEqual(len(manifest["merge"]["edited"]), 1)

    def test_a_first_run_records_a_machine_generated_set(self) -> None:
        self._run()

        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["set_origin"], "machine-generated")
        self.assertEqual(manifest["merge"]["edited"], [])
        self.assertEqual(manifest["merge"]["foreign"], [])

    def test_append_without_an_existing_file_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self._run("--append")

        self.assertEqual(caught.exception.code, 2)

    def test_the_fingerprint_follows_the_merged_file(self) -> None:
        self._run("--gaming", "verbose_padding")
        first = json.loads(self.manifest.read_text(encoding="utf-8"))["output_sha256"]

        self._run("--append")
        second = json.loads(self.manifest.read_text(encoding="utf-8"))["output_sha256"]

        self.assertNotEqual(first, second)
        cases = load_scoring_cases(self.output)
        self.assertEqual(len(cases), len(self._rows()))


class GenerationProvenanceTests(unittest.TestCase):
    """A machine-generated claim must be provable, not merely typed.

    The README invites users to hand-edit generated rows, so `machine-generated`
    is only honest when the scored cases are byte-for-byte what the generator
    produced. The audit proves that by matching the two fingerprints.
    """

    def setUp(self) -> None:
        self.work = Path(tempfile.mkdtemp())
        self.cases = self.work / "cases.csv"
        self.generation_manifest = self.work / "cases.manifest.json"
        main(
            [
                "generate",
                "--input", str(ROOT / "examples" / "essay_baselines.csv"),
                "--output", str(self.cases),
            ]
        )
        self.scores = self.work / "scores.csv"
        self.scoring_manifest = self.work / "scores.manifest.json"
        self._write_scores()

    def _write_scores(self) -> None:
        """Score the generated cases without calling a model."""

        from dataclasses import asdict

        from agent_audit.io import load_scoring_cases, stable_hash, write_score_records
        from agent_audit.models import ScoreRecord

        cases = load_scoring_cases(self.cases)
        records = [
            ScoreRecord(
                "Grader",
                case.case_id,
                case.variant_id,
                case.variant_type,
                7.0 if case.variant_type != "degradation" else 5.0,
            )
            for case in cases
        ]
        write_score_records(self.scores, records)
        self.scoring_manifest.write_text(
            json.dumps(
                {
                    "input_sha256": stable_hash([asdict(case) for case in cases]),
                    "rubric_sha256": "b" * 64,
                    "record_count": len(records),
                    "repeats": 1,
                    "temperature": 0.0,
                    "model": "demo-model",
                    "score_range": [0.0, 10.0],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _audit(self, *extra: str) -> int:
        return main(
            [
                "audit",
                "--input", str(self.scores),
                "--report", str(self.work / "report.md"),
                "--json", str(self.work / "result.json"),
                "--score-min", "0",
                "--score-max", "10",
                *extra,
            ]
        )

    def _result(self) -> dict[str, Any]:
        return json.loads((self.work / "result.json").read_text(encoding="utf-8"))

    def test_a_matching_pair_records_the_generation_fingerprint(self) -> None:
        status = self._audit(
            "--variant-origin", "machine-generated",
            "--generation-manifest", str(self.generation_manifest),
        )

        self.assertEqual(status, 0)
        context = self._result()["comparison_context"]
        self.assertEqual(len(context["generation_sha256"]), 64)

    def test_an_edited_case_set_is_refused(self) -> None:
        """Exactly the hole this closes: edit a row, keep the label."""

        rows = list(csv.DictReader(self.cases.open(encoding="utf-8")))
        rows[1]["text"] = rows[1]["text"] + "人工补充的一句话。"
        with self.cases.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self._write_scores()

        with self.assertRaises(SystemExit) as caught:
            self._audit(
                "--variant-origin", "machine-generated",
                "--generation-manifest", str(self.generation_manifest),
            )

        self.assertEqual(caught.exception.code, 2)

    def test_declaring_machine_generated_without_proof_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self._audit("--variant-origin", "machine-generated")

        self.assertEqual(caught.exception.code, 2)

    def test_mixed_needs_no_generation_manifest(self) -> None:
        """An edited set is honestly `mixed`, and that claim needs no proof."""

        self.assertEqual(self._audit("--variant-origin", "mixed"), 0)
        self.assertIsNone(self._result()["comparison_context"]["generation_sha256"])

    def test_a_missing_generation_manifest_file_is_refused(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self._audit(
                "--variant-origin", "machine-generated",
                "--generation-manifest", str(self.work / "absent.json"),
            )

        self.assertEqual(caught.exception.code, 2)

    def test_a_malformed_generation_manifest_is_refused(self) -> None:
        broken = self.work / "broken.json"
        broken.write_text("{not json", encoding="utf-8")

        with self.assertRaises(SystemExit) as caught:
            self._audit(
                "--variant-origin", "machine-generated",
                "--generation-manifest", str(broken),
            )

        self.assertEqual(caught.exception.code, 2)

    def test_a_generation_manifest_without_a_fingerprint_is_refused(self) -> None:
        manifest = json.loads(self.generation_manifest.read_text(encoding="utf-8"))
        del manifest["output_sha256"]
        broken = self.work / "no_fingerprint.json"
        broken.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

        with self.assertRaises(SystemExit) as caught:
            self._audit(
                "--variant-origin", "machine-generated",
                "--generation-manifest", str(broken),
            )

        self.assertEqual(caught.exception.code, 2)

    def test_a_mixed_set_cannot_be_declared_machine_generated(self) -> None:
        """Appending onto hand edits must not launder them into a proof."""

        self._refuse_with_manifest(
            self._mutated_manifest(set_origin="mixed"),
            "contains hand-written or hand-edited rows",
        )

    def test_a_manifest_predating_the_origin_field_still_verifies(self) -> None:
        manifest = self._mutated_manifest()
        manifest.pop("set_origin", None)
        manifest.pop("merge", None)
        path = self.work / "legacy.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

        status = self._audit(
            "--variant-origin", "machine-generated",
            "--generation-manifest", str(path),
        )

        self.assertEqual(status, 0)

    def _refuse_with_manifest(self, payload: Any, pattern: str) -> None:
        """Assert which refusal fired, not merely that something failed."""

        broken = self.work / "broken.json"
        broken.write_text(
            payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as caught:
            self._audit(
                "--variant-origin", "machine-generated",
                "--generation-manifest", str(broken),
            )
        self.assertEqual(caught.exception.code, 2)
        self.assertIn(pattern, stderr.getvalue())

    def _mutated_manifest(self, **overrides: Any) -> dict[str, Any]:
        manifest = json.loads(self.generation_manifest.read_text(encoding="utf-8"))
        manifest.update(overrides)
        return manifest

    def test_a_manifest_that_is_not_an_object_is_refused(self) -> None:
        self._refuse_with_manifest("[1, 2]", "must contain a JSON object")

    def test_a_non_integer_seed_is_refused(self) -> None:
        for value in ("0", 1.5, True, None):
            with self.subTest(value=value):
                self._refuse_with_manifest(
                    self._mutated_manifest(seed=value), "seed must be an integer"
                )

    def test_an_unnamed_generator_or_language_is_refused(self) -> None:
        for field in ("generator", "language"):
            for value in ("  ", 7, None):
                with self.subTest(field=field, value=value):
                    self._refuse_with_manifest(
                        self._mutated_manifest(**{field: value}),
                        "must be a non-empty string",
                    )

    def test_a_malformed_strategy_list_is_refused(self) -> None:
        for field in (
            "gaming_strategies",
            "degradation_strategies",
            "paraphrase_strategies",
        ):
            for value in ("verbose_padding", [1], None):
                with self.subTest(field=field, value=value):
                    self._refuse_with_manifest(
                        self._mutated_manifest(**{field: value}),
                        "must be a list of strings",
                    )

    def test_verification_needs_a_scoring_manifest_to_compare_against(self) -> None:
        self.scoring_manifest.unlink()

        with self.assertRaises(SystemExit) as caught:
            self._audit(
                "--variant-origin", "machine-generated",
                "--generation-manifest", str(self.generation_manifest),
            )

        self.assertEqual(caught.exception.code, 2)

    def test_the_fingerprint_changes_with_the_generation_seed(self) -> None:
        first = self.work / "seeded.json"
        main(
            [
                "generate",
                "--input", str(ROOT / "examples" / "essay_baselines.csv"),
                "--output", str(self.work / "seeded.csv"),
                "--manifest", str(first),
                "--seed", "9",
            ]
        )

        baseline = json.loads(self.generation_manifest.read_text(encoding="utf-8"))
        seeded = json.loads(first.read_text(encoding="utf-8"))

        self.assertNotEqual(baseline["output_sha256"], seeded["output_sha256"])


if __name__ == "__main__":
    unittest.main()
