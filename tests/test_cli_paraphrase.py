"""End-to-end tests for drafting paraphrases and merging the ratified ones.

Two commands meet here. `paraphrase` calls a model and writes a file nobody
has approved yet; `generate --append --paraphrase-review` takes the rows a
human approved and puts them in the case set. The boundary between them is the
whole point of the design, so most of what is tested below is what must *not*
cross it: unapproved drafts, stale rewrites, and the claim that the resulting
set is machine-generated.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent_audit.cli import build_parser, main
from agent_audit.io import (
    REVIEW_COLUMNS,
    load_baseline_cases,
    load_scoring_cases,
    stable_hash,
)
from agent_audit.segmentation import CHINESE


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "examples" / "essay_baselines.csv"

DRAFT_TEXT = (
    "中学的上课时间可以推后到八点半。睡得不够会让人难以集中，"
    "校内一次调查也发现，能睡满八小时的学生上午测验的分数更稳。"
    "所以学校不妨先挑一个年级试一试，再看看迟到率和成绩有没有变化。"
)


def _baselines() -> dict[str, str]:
    return {
        case.case_id: case.text
        for case in load_baseline_cases(BASELINES, language=CHINESE)
    }


def _review_row(case_id: str, text: str, status: str, **overrides: str) -> dict[str, str]:
    row = {
        "case_id": case_id,
        "baseline_sha256": stable_hash(text),
        "baseline_text": text,
        "draft_text": DRAFT_TEXT,
        "status": status,
        "blocking_checks": "",
        "review_notes": "",
        "reviewer_note": "",
    }
    row.update(overrides)
    return row


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_review(
    path: Path, rows: list[dict[str, str]], *, model: str | None = "test-rewriter"
) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    if model is not None:
        path.with_suffix(".manifest.json").write_text(
            json.dumps({"paraphrase_model": model}), encoding="utf-8"
        )
    return path


class _RewriteHandler(BaseHTTPRequestHandler):
    """Answers every chat completion with one fixed rewrite."""

    prompts: list[list[dict[str, str]]] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.prompts.append(request["messages"])
        body = json.dumps(
            {"id": "rw-1", "choices": [{"message": {"content": DRAFT_TEXT}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class ParaphraseCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _RewriteHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self) -> None:
        _RewriteHandler.prompts = []
        self.work = Path(tempfile.mkdtemp())
        self.output = self.work / "review.csv"

    def _run(self, *extra: str) -> int:
        return main(
            [
                "paraphrase",
                "--input", str(BASELINES),
                "--output", str(self.output),
                "--model", "test-rewriter",
                "--base-url", self.base_url,
                *extra,
            ]
        )

    def test_defaults_keep_the_rewrite_deterministic_and_chinese(self) -> None:
        args = build_parser().parse_args(
            ["paraphrase", "--input", "a.csv", "--output", "b.csv", "--model", "m"]
        )

        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.language, "chinese")
        self.assertEqual(args.api_key_env, "OPENAI_API_KEY")

    def test_a_run_writes_a_review_file_and_a_manifest(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(self._run(), 0)

        rows = _read_csv(self.output)
        self.assertEqual(len(rows), 5)
        self.assertEqual(list(rows[0]), list(REVIEW_COLUMNS))
        manifest = json.loads(
            self.output.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["paraphrase_model"], "test-rewriter")
        self.assertTrue(manifest["requires_human_review"])

    def test_nothing_is_approved_by_the_tool_itself(self) -> None:
        """The command's whole contract: it drafts, it never ratifies."""

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self._run()

        statuses = {row["status"] for row in _read_csv(self.output)}
        self.assertNotIn("approved", statuses)

    def test_the_rubric_never_reaches_the_provider(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self._run()

        sent = " ".join(
            message["content"]
            for prompt in _RewriteHandler.prompts
            for message in prompt
        )
        for word in ("评分", "打分", "rubric"):
            with self.subTest(word=word):
                self.assertNotIn(word, sent)

    def test_the_operator_is_warned_that_client_text_is_leaving(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            stderr
        ):
            self._run()

        self.assertIn("model provider", stderr.getvalue())

    def test_an_existing_review_file_is_never_overwritten(self) -> None:
        self.output.write_text("reviewer decisions live here", encoding="utf-8")

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                self._run()

        self.assertEqual(
            self.output.read_text(encoding="utf-8"), "reviewer decisions live here"
        )

    def test_a_local_endpoint_needs_no_key(self) -> None:
        """Running a model on your own machine must not require a key."""

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            status = self._run("--api-key-env", "AGENT_REVIEW_ABSENT_KEY")

        self.assertEqual(status, 0)
        self.assertTrue(self.output.exists())

    def test_the_manifest_path_can_be_named(self) -> None:
        elsewhere = self.work / "drafting.json"

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self._run("--manifest", str(elsewhere))

        self.assertTrue(elsewhere.exists())
        self.assertFalse(self.output.with_suffix(".manifest.json").exists())

    def test_a_remote_endpoint_without_a_key_is_a_usage_error(self) -> None:
        stderr = io.StringIO()
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(stderr):
                main(
                    [
                        "paraphrase",
                        "--input", str(BASELINES),
                        "--output", str(self.output),
                        "--model", "m",
                        "--base-url", "https://api.example.com/v1",
                        "--api-key-env", "AGENT_REVIEW_ABSENT_KEY",
                    ]
                )

        self.assertIn("AGENT_REVIEW_ABSENT_KEY", stderr.getvalue())
        self.assertFalse(self.output.exists())


class RatifiedMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = Path(tempfile.mkdtemp())
        self.output = self.work / "cases.csv"
        self.review = self.work / "review.csv"
        self.texts = _baselines()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self._generate()

    def _generate(self, *extra: str) -> int:
        return main(
            [
                "generate",
                "--input", str(BASELINES),
                "--output", str(self.output),
                *extra,
            ]
        )

    def _merge(self, *extra: str) -> int:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            return self._generate(
                "--append", "--paraphrase-review", str(self.review), *extra
            )

    def _approved_review(self) -> Path:
        return _write_review(
            self.review,
            [
                _review_row("school_start", self.texts["school_start"], "approved"),
                _review_row("public_transit", self.texts["public_transit"], "pending"),
                _review_row("campus_phones", self.texts["campus_phones"], "rejected"),
            ],
        )

    def _rows(self) -> list[dict[str, str]]:
        return _read_csv(self.output)

    def _manifest(self) -> dict:
        return json.loads(
            self.output.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )

    def test_an_approved_row_reaches_the_case_file(self) -> None:
        self._approved_review()

        self.assertEqual(self._merge(), 0)

        ratified = [
            row for row in self._rows() if row["variant_id"] == "paraphrase_ratified"
        ]
        self.assertEqual([row["case_id"] for row in ratified], ["school_start"])
        self.assertEqual(ratified[0]["variant_type"], "paraphrase")
        self.assertEqual(ratified[0]["text"], DRAFT_TEXT)
        self.assertIn("human_ratified_paraphrase", ratified[0]["notes"])

    def test_the_merged_file_still_loads_as_a_case_set(self) -> None:
        self._approved_review()
        self._merge()

        cases = load_scoring_cases(self.output)

        self.assertIn(
            ("school_start", "paraphrase_ratified"),
            {(case.case_id, case.variant_id) for case in cases},
        )

    def test_unapproved_rows_stay_out(self) -> None:
        self._approved_review()
        self._merge()

        ids = {row["case_id"] for row in self._rows() if row["variant_id"] == "paraphrase_ratified"}

        self.assertNotIn("public_transit", ids)
        self.assertNotIn("campus_phones", ids)

    def test_a_ratified_row_makes_the_set_mixed(self) -> None:
        """Equivalence was judged by a human, so the set is not tool-verified."""

        self._approved_review()
        self._merge()

        self.assertEqual(self._manifest()["set_origin"], "mixed")

    def test_merging_twice_does_not_duplicate_the_row(self) -> None:
        self._approved_review()
        self._merge()
        first = self._rows()

        self.assertEqual(self._merge(), 0)

        self.assertEqual(len(self._rows()), len(first))

    def test_the_review_tally_is_recorded_and_printed(self) -> None:
        self._approved_review()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self._generate("--append", "--paraphrase-review", str(self.review))

        self.assertEqual(
            self._manifest()["paraphrase_review"],
            {"pending": 1, "blocked": 0, "approved": 1, "rejected": 1},
        )
        self.assertIn("approved=1", stdout.getvalue())

    def test_the_manifest_names_the_ratified_rows(self) -> None:
        self._approved_review()
        self._merge()

        self.assertEqual(
            self._manifest()["merge"]["ratified_paraphrase"],
            [["school_start", "paraphrase_ratified"]],
        )

    def test_a_run_without_a_review_records_no_tally(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self._generate("--append")

        self.assertIsNone(self._manifest()["paraphrase_review"])

    def test_a_stale_baseline_hash_is_refused_and_names_the_case(self) -> None:
        """An edited baseline leaves the old draft rewriting text that is gone."""

        _write_review(
            self.review,
            [
                _review_row(
                    "school_start",
                    self.texts["school_start"],
                    "approved",
                    baseline_sha256=stable_hash("被编辑过的旧基准文本"),
                )
            ],
        )
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                self._generate("--append", "--paraphrase-review", str(self.review))

        self.assertIn("school_start", stderr.getvalue())
        self.assertNotIn(
            "paraphrase_ratified", {row["variant_id"] for row in self._rows()}
        )

    def test_a_case_missing_from_the_baselines_is_refused(self) -> None:
        _write_review(
            self.review,
            [_review_row("ghost_case", self.texts["school_start"], "approved")],
        )
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                self._generate("--append", "--paraphrase-review", str(self.review))

        self.assertIn("ghost_case", stderr.getvalue())

    def test_a_misspelled_status_is_refused(self) -> None:
        _write_review(
            self.review,
            [_review_row("school_start", self.texts["school_start"], "aproved")],
        )
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                self._generate("--append", "--paraphrase-review", str(self.review))

        self.assertIn("status must be one of", stderr.getvalue())

    def test_a_missing_review_file_is_refused(self) -> None:
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                self._generate(
                    "--append", "--paraphrase-review", str(self.work / "absent.csv")
                )

        self.assertIn("Review file does not exist", stderr.getvalue())

    def test_the_manifest_names_the_drafting_model(self) -> None:
        self._approved_review()
        self._merge()

        self.assertEqual(self._manifest()["paraphrase_model"], "test-rewriter")

    def test_a_run_without_a_review_names_no_drafting_model(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self._generate("--append")

        self.assertIsNone(self._manifest()["paraphrase_model"])

    def test_a_missing_drafting_manifest_is_refused(self) -> None:
        """Without the model's name the audit cannot rule out circularity."""

        _write_review(
            self.review,
            [_review_row("school_start", self.texts["school_start"], "approved")],
            model=None,
        )
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                self._generate("--append", "--paraphrase-review", str(self.review))

        self.assertIn("Paraphrase manifest does not exist", stderr.getvalue())

    def test_a_drafting_manifest_without_a_model_is_refused(self) -> None:
        self._approved_review()
        self.review.with_suffix(".manifest.json").write_text(
            json.dumps({"case_count": 3}), encoding="utf-8"
        )
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                self._generate("--append", "--paraphrase-review", str(self.review))

        self.assertIn("paraphrase_model", stderr.getvalue())

    def test_a_drafting_manifest_that_is_not_an_object_is_refused(self) -> None:
        self._approved_review()
        self.review.with_suffix(".manifest.json").write_text(
            json.dumps(["test-rewriter"]), encoding="utf-8"
        )
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                self._generate("--append", "--paraphrase-review", str(self.review))

        self.assertIn("must contain a JSON object", stderr.getvalue())

    def test_a_reviewer_note_travels_with_the_ratified_row(self) -> None:
        """The reviewer's own reasoning is why the row was admitted."""

        _write_review(
            self.review,
            [
                _review_row(
                    "school_start",
                    self.texts["school_start"],
                    "approved",
                    reviewer_note="核对过证据句，含义一致",
                )
            ],
        )
        self._merge()

        ratified = next(
            row for row in self._rows() if row["variant_id"] == "paraphrase_ratified"
        )

        self.assertIn("reviewer_note=核对过证据句，含义一致", ratified["notes"])

    def test_a_malformed_drafting_manifest_is_refused(self) -> None:
        self._approved_review()
        self.review.with_suffix(".manifest.json").write_text(
            "{not json", encoding="utf-8"
        )
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                self._generate("--append", "--paraphrase-review", str(self.review))

        self.assertIn("malformed", stderr.getvalue())

    def test_the_drafting_manifest_can_be_named(self) -> None:
        _write_review(
            self.review,
            [_review_row("school_start", self.texts["school_start"], "approved")],
            model=None,
        )
        elsewhere = self.work / "elsewhere.json"
        elsewhere.write_text(
            json.dumps({"paraphrase_model": "named-rewriter"}), encoding="utf-8"
        )

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            status = self._generate(
                "--append",
                "--paraphrase-review", str(self.review),
                "--paraphrase-manifest", str(elsewhere),
            )

        self.assertEqual(status, 0)
        self.assertEqual(self._manifest()["paraphrase_model"], "named-rewriter")

    def test_a_review_without_append_is_refused(self) -> None:
        """Merging into a fresh file would silently drop the existing rows."""

        self._approved_review()
        fresh = self.work / "fresh.csv"
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                main(
                    [
                        "generate",
                        "--input", str(BASELINES),
                        "--output", str(fresh),
                        "--paraphrase-review", str(self.review),
                    ]
                )

        self.assertIn("--append", stderr.getvalue())
        self.assertFalse(fresh.exists())


class CircularityTests(unittest.TestCase):
    """The rewriter and the grader must not be the same model.

    If they are, the system under test also supplied the definition of "same
    meaning", and a paraphrase-invariance result says nothing. An exact name
    match is refused; same family at different versions cannot be detected, so
    the report prints both names and leaves that judgement to the reader.
    """

    def setUp(self) -> None:
        self.work = Path(tempfile.mkdtemp())
        self.cases = self.work / "cases.csv"
        self.generation_manifest = self.work / "cases.manifest.json"
        self.review = self.work / "review.csv"
        texts = _baselines()
        _write_review(
            self.review,
            [_review_row("school_start", texts["school_start"], "approved")],
            model="rewriter-model",
        )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            main(
                [
                    "generate",
                    "--input", str(BASELINES),
                    "--output", str(self.cases),
                ]
            )
            main(
                [
                    "generate",
                    "--input", str(BASELINES),
                    "--output", str(self.cases),
                    "--append",
                    "--paraphrase-review", str(self.review),
                ]
            )
        self.scores = self.work / "scores.csv"
        self.scoring_manifest = self.work / "scores.manifest.json"

    def _write_scores(self, model: str) -> None:
        from dataclasses import asdict

        from agent_audit.io import write_score_records
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
                    "model": model,
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
                "--html", str(self.work / "report.html"),
                "--score-min", "0",
                "--score-max", "10",
                "--variant-origin", "mixed",
                "--generation-manifest", str(self.generation_manifest),
                *extra,
            ]
        )

    def test_the_same_model_on_both_sides_is_refused(self) -> None:
        self._write_scores("rewriter-model")
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                self._audit()

        self.assertIn("rewriter-model", stderr.getvalue())
        self.assertFalse((self.work / "result.json").exists())

    def test_the_match_ignores_case(self) -> None:
        self._write_scores("Rewriter-Model")

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self._audit()

    def test_different_models_pass_and_are_both_recorded(self) -> None:
        self._write_scores("grader-model")

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            status = self._audit()

        self.assertEqual(status, 0)
        context = json.loads(
            (self.work / "result.json").read_text(encoding="utf-8")
        )["comparison_context"]
        self.assertEqual(context["paraphrase_model"], "rewriter-model")
        self.assertEqual(context["model"], "grader-model")

    def test_both_model_names_appear_in_the_report(self) -> None:
        self._write_scores("grader-model")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self._audit()

        for rendered in ("report.md", "report.html"):
            with self.subTest(file=rendered):
                text = (self.work / rendered).read_text(encoding="utf-8")

                self.assertIn("rewriter-model", text)
                self.assertIn("grader-model", text)

    def test_a_mixed_set_may_now_present_a_generation_manifest(self) -> None:
        """Before this, a mixed set could not prove which run produced it."""

        self._write_scores("grader-model")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self._audit()

        context = json.loads(
            (self.work / "result.json").read_text(encoding="utf-8")
        )["comparison_context"]
        self.assertIsNotNone(context["generation_sha256"])

    def test_declaring_machine_generated_over_a_mixed_set_is_still_refused(self) -> None:
        self._write_scores("grader-model")
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                main(
                    [
                        "audit",
                        "--input", str(self.scores),
                        "--report", str(self.work / "report.md"),
                        "--score-min", "0",
                        "--score-max", "10",
                        "--variant-origin", "machine-generated",
                        "--generation-manifest", str(self.generation_manifest),
                    ]
                )

        self.assertIn("mixed", stderr.getvalue())

    def test_declaring_human_authored_over_a_generated_set_is_refused(self) -> None:
        """A matching fingerprint proves a generator wrote these rows."""

        self._write_scores("grader-model")
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                main(
                    [
                        "audit",
                        "--input", str(self.scores),
                        "--report", str(self.work / "report.md"),
                        "--score-min", "0",
                        "--score-max", "10",
                        "--variant-origin", "human-authored",
                        "--generation-manifest", str(self.generation_manifest),
                    ]
                )

        self.assertIn("human-authored is false", stderr.getvalue())

    def test_an_unnamed_drafting_model_in_the_generation_manifest_is_refused(
        self,
    ) -> None:
        """A blank name would pass the circularity check by saying nothing."""

        self._write_scores("grader-model")
        manifest = json.loads(
            self.generation_manifest.read_text(encoding="utf-8")
        )
        manifest["paraphrase_model"] = "   "
        self.generation_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        stderr = io.StringIO()

        with self.assertRaises(SystemExit):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                stderr
            ):
                self._audit()

        self.assertIn("paraphrase_model", stderr.getvalue())

    def test_an_audit_without_a_drafting_model_names_no_models(self) -> None:
        """A plain generated set must not gain a models line in the report."""

        plain = self.work / "plain.csv"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            main(
                [
                    "generate",
                    "--input", str(BASELINES),
                    "--output", str(plain),
                ]
            )
        self.cases = plain
        self.generation_manifest = self.work / "plain.manifest.json"
        self._write_scores("grader-model")

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self._audit()

        report = (self.work / "report.md").read_text(encoding="utf-8")
        self.assertNotIn("改写模型", report)
        context = json.loads(
            (self.work / "result.json").read_text(encoding="utf-8")
        )["comparison_context"]
        self.assertIsNone(context["paraphrase_model"])


if __name__ == "__main__":
    unittest.main()
