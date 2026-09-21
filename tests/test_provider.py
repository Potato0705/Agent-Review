from __future__ import annotations

import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import threading
import tempfile
from types import SimpleNamespace
import unittest

from agent_audit.audit import AuditConfig, audit_records
from agent_audit.cli_audit import run as run_audit
from agent_audit.cli_score import run as run_score
from agent_audit.io import load_score_records
from agent_audit.models import ScoringCase
from agent_audit.provider import (
    OpenAICompatibleConfig,
    OpenAICompatibleScorer,
    ProviderError,
    _parse_score_content,
)
from agent_audit.scoring import run_scoring


class _MockHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, object]] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        content_length = int(self.headers.get("Content-Length", "0"))
        request_payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        self.__class__.requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "payload": request_payload,
            }
        )
        user_text = request_payload["messages"][1]["content"]
        match = re.search(r"SCORE=([0-9.]+)", user_text)
        score = float(match.group(1)) if match else 6.0
        response_payload = {
            "id": "mock-response-1",
            "choices": [
                {
                    "message": {
                        "content": (
                            "```json\n"
                            + json.dumps({"score": score, "reason": "mock reason"})
                            + "\n```"
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        }
        encoded = json.dumps(response_payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


class ProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self) -> None:
        _MockHandler.requests.clear()

    def _scorer(self) -> OpenAICompatibleScorer:
        return OpenAICompatibleScorer(
            OpenAICompatibleConfig(
                base_url=self.base_url,
                model="mock-model",
                api_key="test-secret",
                score_min=0,
                score_max=10,
                max_retries=0,
            )
        )

    def test_rejects_a_boolean_score(self) -> None:
        """A boolean is not a score, and float(True) would silently mean 1.0.

        A grader that answers with true/false has failed to produce a rating.
        Converting that to 1.0 or 0.0 would put a fabricated number into the
        client score file with no trace that the model never scored the text.
        """

        for literal in ("true", "false"):
            with self.subTest(literal=literal):
                content = '{"score": ' + literal + ', "reason": "no rating"}'
                with self.assertRaisesRegex(ProviderError, "numeric score"):
                    _parse_score_content(content, 0.0, 10.0)

    def test_extracts_a_score_from_a_fenced_code_block(self) -> None:
        content = '```json\n{"score": 7.5, "reason": "clear thesis"}\n```'

        score, reason = _parse_score_content(content, 0.0, 10.0)

        self.assertAlmostEqual(score, 7.5)
        self.assertEqual(reason, "clear thesis")

    def test_extracts_a_score_from_surrounding_prose(self) -> None:
        content = 'Here is my rating:\n{"score": 4, "reason": "weak evidence"} Hope that helps.'

        score, reason = _parse_score_content(content, 0.0, 10.0)

        self.assertAlmostEqual(score, 4.0)
        self.assertEqual(reason, "weak evidence")

    def test_rejects_braces_that_do_not_parse_as_json(self) -> None:
        """Prose containing braces must not be mistaken for a scoring object."""

        with self.assertRaisesRegex(ProviderError, "malformed JSON scoring output"):
            _parse_score_content("My rating: {score: seven, reason: good}", 0.0, 10.0)

    def test_rejects_output_without_any_json_object(self) -> None:
        with self.assertRaisesRegex(ProviderError, "did not return a JSON scoring object"):
            _parse_score_content("I would rate this an eight out of ten.", 0.0, 10.0)

    def test_rejects_a_json_array(self) -> None:
        with self.assertRaisesRegex(ProviderError, "must be a JSON object"):
            _parse_score_content("[7.5]", 0.0, 10.0)

    def test_rejects_a_missing_score_field(self) -> None:
        with self.assertRaisesRegex(ProviderError, "numeric score"):
            _parse_score_content('{"rating": 7.5}', 0.0, 10.0)

    def test_rejects_a_non_finite_score(self) -> None:
        with self.assertRaisesRegex(ProviderError, "must be finite"):
            _parse_score_content('{"score": NaN}', 0.0, 10.0)

    def test_rejects_a_score_outside_the_requested_range(self) -> None:
        for value in ("11", "-1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ProviderError, "outside the requested range"):
                    _parse_score_content('{"score": ' + value + "}", 0.0, 10.0)

    def test_rejects_a_non_string_reason(self) -> None:
        with self.assertRaisesRegex(ProviderError, "reason must be a string"):
            _parse_score_content('{"score": 7.5, "reason": 42}', 0.0, 10.0)

    def test_rejects_non_local_http_endpoints(self) -> None:
        """Gate 3 forbids sending prompts to a remote endpoint in clear text."""

        config = OpenAICompatibleConfig(
            base_url="http://scores.example.com/v1", model="m", api_key="k"
        )
        with self.assertRaisesRegex(ValueError, "must use HTTPS"):
            config.validate()

    def test_rejects_credentials_embedded_in_the_base_url(self) -> None:
        config = OpenAICompatibleConfig(
            base_url="https://user:secret@api.example.com/v1", model="m", api_key="k"
        )
        with self.assertRaisesRegex(ValueError, "must not contain credentials"):
            config.validate()

    def test_allows_a_local_http_endpoint(self) -> None:
        for host in ("localhost", "127.0.0.1"):
            with self.subTest(host=host):
                OpenAICompatibleConfig(
                    base_url=f"http://{host}:11434/v1", model="m", api_key="k"
                ).validate()

    def test_scores_with_openai_compatible_response(self) -> None:
        result = self._scorer().score("SCORE=7.5", "Test rubric")

        self.assertEqual(result.score, 7.5)
        self.assertEqual(result.reason, "mock reason")
        self.assertEqual(result.response_id, "mock-response-1")
        self.assertEqual(result.prompt_tokens, 20)
        self.assertEqual(_MockHandler.requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(
            _MockHandler.requests[0]["authorization"], "Bearer test-secret"
        )

    def test_prompt_does_not_reveal_variant_type(self) -> None:
        self._scorer().score("SCORE=6.0", "Test rubric")
        messages = _MockHandler.requests[0]["payload"]["messages"]
        serialized = json.dumps(messages)
        self.assertNotIn("gaming", serialized)
        self.assertNotIn("degradation", serialized)

    def test_scoring_run_is_traceable_without_api_key(self) -> None:
        cases = [
            ScoringCase("c1", "base", "baseline", "SCORE=7.0"),
            ScoringCase("c1", "game", "gaming", "SCORE=7.8"),
            ScoringCase("c1", "drop", "degradation", "SCORE=5.0"),
        ]
        run = run_scoring(
            cases,
            self._scorer(),
            "Test rubric",
            system_name="Mock System",
            provider_name="openai-compatible",
            model="mock-model",
            base_url=self.base_url,
            score_min=0,
            score_max=10,
            temperature=0,
        )

        self.assertEqual(len(run.records), 3)
        self.assertEqual(run.records[1].score, 7.8)
        self.assertEqual(run.manifest["record_count"], 3)
        self.assertIn("input_sha256", run.manifest)
        self.assertIn("rubric_sha256", run.manifest)
        self.assertNotIn("api_key", run.manifest)
        self.assertEqual(run.traces[0]["response_id"], "mock-response-1")

    def test_repeated_run_labels_first_sample_reason(self) -> None:
        cases = [
            ScoringCase("c1", "base", "baseline", "SCORE=7.0"),
            ScoringCase("c1", "game", "gaming", "SCORE=7.8"),
            ScoringCase("c1", "drop", "degradation", "SCORE=5.0"),
        ]
        run = run_scoring(
            cases,
            self._scorer(),
            "Test rubric",
            system_name="Mock System",
            provider_name="openai-compatible",
            model="mock-model",
            base_url=self.base_url,
            score_min=0,
            score_max=10,
            temperature=0,
            repeats=2,
        )

        self.assertEqual(run.records[0].notes, "sample_1_reason=mock reason")
        self.assertEqual(len(run.traces), 6)

    def test_rejects_insecure_remote_http(self) -> None:
        with self.assertRaisesRegex(ValueError, "must use HTTPS"):
            OpenAICompatibleScorer(
                OpenAICompatibleConfig(
                    base_url="http://example.com/v1",
                    model="model",
                    api_key="secret",
                )
            )

    def test_the_manifest_lands_next_to_the_scores_when_unnamed(self) -> None:
        """The default path and the checkpoint tally are what an operator sees."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases_path = root / "cases.csv"
            rubric_path = root / "rubric.md"
            scores_path = root / "scores.csv"
            cases_path.write_text(
                "case_id,variant_id,variant_type,text,notes\n"
                'c1,base,baseline,"SCORE=7.0",base\n'
                'c1,game,gaming,"SCORE=7.8",game\n'
                'c1,drop,degradation,"SCORE=5.0",drop\n',
                encoding="utf-8",
            )
            rubric_path.write_text("Test rubric", encoding="utf-8")
            args = SimpleNamespace(
                api_key_env="MISSING_TEST_KEY",
                base_url=self.base_url,
                rubric_file=str(rubric_path),
                input=str(cases_path),
                output=str(scores_path),
                model="mock-model",
                system_name="Mock System",
                score_min=0.0,
                score_max=10.0,
                temperature=0.0,
                repeats=1,
                timeout=5.0,
                max_retries=0,
                manifest=None,
                raw_output=None,
                checkpoint=str(root / "checkpoint.jsonl"),
                resume=False,
            )
            printed = io.StringIO()

            with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self.assertEqual(run_score(args), 0)

            self.assertTrue((root / "scores.manifest.json").exists())
            self.assertIn("Checkpoint samples:", printed.getvalue())
            self.assertIn("3 new", printed.getvalue())

    def test_cli_scoring_files_feed_audit_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cases_path = root / "cases.csv"
            rubric_path = root / "rubric.md"
            scores_path = root / "scores.csv"
            manifest_path = root / "manifest.json"
            raw_path = root / "raw.jsonl"
            report_path = root / "audit.md"
            audit_json_path = root / "audit.json"
            audit_html_path = root / "audit.html"
            cases_path.write_text(
                "case_id,variant_id,variant_type,text,notes\n"
                'c1,base,baseline,"SCORE=7.0",base\n'
                'c1,game,gaming,"SCORE=7.8",game\n'
                'c1,drop,degradation,"SCORE=5.0",drop\n',
                encoding="utf-8",
            )
            rubric_path.write_text("Test rubric", encoding="utf-8")
            args = SimpleNamespace(
                api_key_env="MISSING_TEST_KEY",
                base_url=self.base_url,
                rubric_file=str(rubric_path),
                input=str(cases_path),
                output=str(scores_path),
                model="mock-model",
                system_name="Mock System",
                score_min=0.0,
                score_max=10.0,
                temperature=0.0,
                repeats=2,
                timeout=5.0,
                max_retries=0,
                manifest=str(manifest_path),
                raw_output=str(raw_path),
            )

            self.assertEqual(run_score(args), 0)
            records = load_score_records(scores_path)
            audit = audit_records(
                records,
                AuditConfig(score_min=0, score_max=10),
            )

            self.assertEqual(audit.risk_level, "HIGH")
            self.assertFalse(audit.risk_is_provisional)
            self.assertEqual(audit.uncertainty_evaluable_count, 2)
            self.assertTrue(all(record.sample_count == 2 for record in records))
            self.assertTrue(all(record.score_stddev == 0 for record in records))
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["repeats"], 2)
            self.assertEqual(len(raw_path.read_text(encoding="utf-8").splitlines()), 6)

            audit_args = SimpleNamespace(
                input=str(scores_path),
                report=str(report_path),
                json_output=str(audit_json_path),
                invariance_tolerance=0.5,
                min_degradation_drop=1.0,
                gaming_tolerance=0.0,
                score_min=0.0,
                score_max=10.0,
                data_provenance="public-demo",
                variant_origin="human-authored",
                manifest=str(manifest_path),
                html_output=str(audit_html_path),
            )
            self.assertEqual(run_audit(audit_args), 0)
            audit_payload = json.loads(audit_json_path.read_text(encoding="utf-8"))
            self.assertEqual(audit_payload["comparison_context"]["repeats"], 2)
            self.assertEqual(
                audit_payload["comparison_context"]["model"], "mock-model"
            )
            self.assertEqual(
                audit_payload["comparison_context"]["input_sha256"],
                manifest["input_sha256"],
            )
            self.assertTrue(
                audit_html_path.read_text(encoding="utf-8").startswith(
                    "<!doctype html>"
                )
            )


if __name__ == "__main__":
    unittest.main()
