"""End-to-end tests for the `trajectory` subcommand.

The command reuses everything the text generator established — refuse to
overwrite, merge on `--append`, fingerprint the output, declare the set
origin — so these tests concentrate on what is new: the JSONL input, the
rendered transcript that reaches the scorer, and the refusals that keep an
unannotated trajectory out of a variant family nobody verified.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from agent_audit.cli import build_parser, main
from agent_audit.io import load_scoring_cases


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "examples" / "trajectory_baselines.jsonl"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class ParserTests(unittest.TestCase):
    def test_both_required_families_default_to_every_strategy(self) -> None:
        args = build_parser().parse_args(
            ["trajectory", "--input", "a.jsonl", "--output", "b.csv"]
        )

        self.assertEqual(args.gaming, "redundant_tool_calls,padded_reasoning")
        self.assertEqual(
            args.degradation, "remove_load_bearing_step,hollow_evidence"
        )

    def test_paraphrase_is_off_by_default(self) -> None:
        """Turning it on for an unannotated case would stop the whole run."""

        args = build_parser().parse_args(
            ["trajectory", "--input", "a.jsonl", "--output", "b.csv"]
        )

        self.assertIsNone(args.paraphrase)


class GenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = Path(tempfile.mkdtemp())
        self.output = self.work / "cases.csv"

    def _run(self, *extra: str) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            return main(
                [
                    "trajectory",
                    "--input", str(BASELINES),
                    "--output", str(self.output),
                    *extra,
                ]
            )

    def _manifest(self) -> dict:
        return json.loads(
            self.output.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )

    def test_the_output_is_a_csv_the_scoring_loader_accepts(self) -> None:
        self.assertEqual(self._run(), 0)

        cases = load_scoring_cases(self.output)

        self.assertEqual(len(cases), 25)
        self.assertEqual(len({case.case_id for case in cases}), 5)

    def test_every_case_gets_both_required_families(self) -> None:
        self._run()
        families: dict[str, set[str]] = {}
        for case in load_scoring_cases(self.output):
            families.setdefault(case.case_id, set()).add(case.variant_type)

        for case_id, seen in families.items():
            with self.subTest(case=case_id):
                self.assertIn("gaming", seen)
                self.assertIn("degradation", seen)

    def test_the_scored_text_is_the_rendered_transcript(self) -> None:
        self._run()
        baseline = next(
            case
            for case in load_scoring_cases(self.output)
            if case.case_id == "flight_refund" and case.variant_type == "baseline"
        )

        self.assertIn("任务：", baseline.text)
        self.assertIn("步骤 1", baseline.text)
        self.assertIn("工具：search_orders", baseline.text)
        self.assertIn("最终答案：", baseline.text)

    def test_the_degradation_keeps_the_answer_and_drops_the_evidence(self) -> None:
        """The whole point: same conclusion, no longer supported."""

        self._run()
        rows = {
            case.variant_id: case.text
            for case in load_scoring_cases(self.output)
            if case.case_id == "flight_refund"
        }
        degraded = rows["degradation_remove_load_bearing_step"]

        self.assertNotIn("read_policy", degraded)
        self.assertIn("符合全额退款条件", degraded)
        self.assertIn("符合全额退款条件", rows["baseline"])

    def test_the_hollowed_variant_keeps_the_call_and_empties_the_result(self) -> None:
        self._run()
        hollowed = next(
            case.text
            for case in load_scoring_cases(self.output)
            if case.variant_id == "degradation_hollow_evidence"
            and case.case_id == "flight_refund"
        )

        self.assertIn("read_policy", hollowed)
        self.assertIn("（无结果）", hollowed)
        self.assertNotIn("24 小时内可申请全额退款", hollowed)

    def test_the_manifest_records_the_modality_and_the_fingerprint(self) -> None:
        self._run()
        manifest = self._manifest()

        self.assertEqual(manifest["modality"], "trajectory")
        self.assertEqual(manifest["set_origin"], "machine-generated")
        self.assertEqual(len(manifest["output_sha256"]), 64)
        self.assertEqual(manifest["row_count"], 25)

    def test_the_manifest_records_each_intervention(self) -> None:
        """A reviewer has to see which step was removed or repeated."""

        self._run()
        removals = [
            record
            for record in self._manifest()["variants"]
            if record["strategy"] == "remove_load_bearing_step"
        ]

        self.assertEqual(len(removals), 5)
        self.assertTrue(all(record["magnitude"]["removed_steps"] for record in removals))

    def test_the_same_seed_reproduces_the_file_byte_for_byte(self) -> None:
        self._run()
        first = self.output.read_bytes()
        self.output.unlink()
        self.output.with_suffix(".manifest.json").unlink()

        self._run()

        self.assertEqual(self.output.read_bytes(), first)

    def test_the_manifest_path_can_be_named(self) -> None:
        elsewhere = self.work / "run.json"

        self._run("--manifest", str(elsewhere))

        self.assertTrue(elsewhere.exists())

    def test_english_labels_are_used_for_english_runs(self) -> None:
        self._run("--language", "english")
        baseline = next(
            case.text
            for case in load_scoring_cases(self.output)
            if case.variant_type == "baseline"
        )

        self.assertIn("Task: ", baseline)
        self.assertIn("Step 1", baseline)


class RefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = Path(tempfile.mkdtemp())
        self.output = self.work / "cases.csv"

    def _run(self, *extra: str, stderr: io.StringIO | None = None) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            stderr or io.StringIO()
        ):
            return main(
                [
                    "trajectory",
                    "--input", str(BASELINES),
                    "--output", str(self.output),
                    *extra,
                ]
            )

    def test_an_existing_case_file_is_never_overwritten(self) -> None:
        self.output.write_text("手工编辑过的内容", encoding="utf-8")

        with self.assertRaises(SystemExit):
            self._run()

        self.assertEqual(self.output.read_text(encoding="utf-8"), "手工编辑过的内容")

    def test_an_unknown_strategy_is_a_usage_error(self) -> None:
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            self._run("--gaming", "teleport", stderr=stderr)

        self.assertIn("gaming", stderr.getvalue())

    def test_a_trajectory_without_annotations_is_refused_by_name(self) -> None:
        unannotated = self.work / "bad.jsonl"
        unannotated.write_text(
            json.dumps(
                {
                    "case_id": "no_annotation",
                    "task": "t",
                    "steps": [
                        {"tool": "a", "args": {}, "result": "r1"},
                        {"tool": "b", "args": {}, "result": "r2"},
                    ],
                    "final_answer": "done",
                    "load_bearing_steps": [],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                main(
                    [
                        "trajectory",
                        "--input", str(unannotated),
                        "--output", str(self.output),
                    ]
                )

        self.assertIn("no_annotation", stderr.getvalue())
        self.assertFalse(self.output.exists())

    def test_paraphrase_without_an_independent_group_names_the_case(self) -> None:
        without = self.work / "no_groups.jsonl"
        without.write_text(
            json.dumps(
                {
                    "case_id": "strictly_sequential",
                    "task": "t",
                    "steps": [
                        {"tool": "a", "args": {}, "result": "r1"},
                        {"tool": "b", "args": {}, "result": "r2"},
                    ],
                    "final_answer": "done",
                    "load_bearing_steps": [2],
                    "independent_steps": [],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                main(
                    [
                        "trajectory",
                        "--input", str(without),
                        "--output", str(self.output),
                        "--paraphrase", "reorder_independent_steps",
                    ]
                )

        self.assertIn("strictly_sequential", stderr.getvalue())

    def test_append_without_an_existing_file_is_a_usage_error(self) -> None:
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            self._run("--append", stderr=stderr)

        self.assertIn("--append", stderr.getvalue())


class AppendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = Path(tempfile.mkdtemp())
        self.output = self.work / "cases.csv"

    def _run(self, *extra: str) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            return main(
                [
                    "trajectory",
                    "--input", str(BASELINES),
                    "--output", str(self.output),
                    *extra,
                ]
            )

    def _manifest(self) -> dict:
        return json.loads(
            self.output.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )

    def test_append_adds_a_newly_selected_family(self) -> None:
        self._run("--gaming", "redundant_tool_calls")
        before = {(r["case_id"], r["variant_id"]) for r in _read_csv(self.output)}

        self.assertEqual(
            self._run("--append", "--paraphrase", "reorder_independent_steps"), 0
        )

        after = {(r["case_id"], r["variant_id"]) for r in _read_csv(self.output)}
        self.assertTrue(before < after)

    def test_a_hand_edited_row_marks_the_set_mixed(self) -> None:
        self._run()
        rows = _read_csv(self.output)
        rows[0]["text"] = "【手工改写的轨迹】"
        with self.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        self._run("--append")

        self.assertEqual(self._manifest()["set_origin"], "mixed")
        self.assertEqual(_read_csv(self.output)[0]["text"], "【手工改写的轨迹】")


class ShowStepsTests(unittest.TestCase):
    def test_printing_the_steps_writes_no_files(self) -> None:
        work = Path(tempfile.mkdtemp())
        output = work / "cases.csv"
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            status = main(
                [
                    "trajectory",
                    "--input", str(BASELINES),
                    "--output", str(output),
                    "--show-steps",
                ]
            )

        self.assertEqual(status, 0)
        self.assertFalse(output.exists())
        printed = stdout.getvalue()
        self.assertIn("[flight_refund]", printed)
        self.assertIn("load-bearing: 2", printed)
        self.assertIn("independent: 1, 3", printed)

    def test_a_case_without_groups_prints_none(self) -> None:
        work = Path(tempfile.mkdtemp())
        source = work / "one.jsonl"
        source.write_text(
            json.dumps(
                {
                    "case_id": "solo",
                    "task": "t",
                    "steps": [
                        {"tool": "a", "args": {}, "result": "r1"},
                        {"tool": "b", "args": {}, "result": "r2"},
                    ],
                    "final_answer": "done",
                    "load_bearing_steps": [2],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            main(
                [
                    "trajectory",
                    "--input", str(source),
                    "--output", str(work / "cases.csv"),
                    "--show-steps",
                ]
            )

        self.assertIn("independent: none", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
