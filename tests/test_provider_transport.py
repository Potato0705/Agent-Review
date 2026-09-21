"""Transport-level tests for the OpenAI-compatible scorer.

These cover the paths that only run when the network or the provider
misbehaves: retries, terminal HTTP errors, malformed envelopes, and the
configuration guards that keep prompts off unencrypted remote endpoints. They
are separated from ``test_provider.py`` so the scripted-failure server does not
complicate the happy-path harness.
"""

from __future__ import annotations

import json
import socket
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from agent_audit.provider import (
    OpenAICompatibleConfig,
    OpenAICompatibleScorer,
    ProviderError,
    _chat_completions_url,
    _content_to_text,
)


def _score_body(score: float = 7.0) -> bytes:
    return json.dumps(
        {
            "id": "mock-1",
            "choices": [
                {"message": {"content": json.dumps({"score": score, "reason": "ok"})}}
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 5},
        }
    ).encode("utf-8")


class _ScriptedHandler(BaseHTTPRequestHandler):
    """Replays a queue of (status, body) pairs, one per request."""

    script: list[tuple[int, bytes]] = []
    attempts: int = 0

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        cls = self.__class__
        status, body = cls.script[min(cls.attempts, len(cls.script) - 1)]
        cls.attempts += 1
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self) -> None:
        _ScriptedHandler.script = [(200, _score_body())]
        _ScriptedHandler.attempts = 0
        # Backoff is real time, and these tests only care about the control
        # flow around it.
        sleep_patcher = patch("agent_audit.provider.time.sleep")
        self.sleep = sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def _scorer(self, *, max_retries: int = 2) -> OpenAICompatibleScorer:
        return OpenAICompatibleScorer(
            OpenAICompatibleConfig(
                base_url=self.base_url,
                model="mock-model",
                api_key="test-secret",
                score_min=0,
                score_max=10,
                max_retries=max_retries,
                timeout_seconds=5.0,
            )
        )

    def test_retries_a_transient_server_error_then_succeeds(self) -> None:
        _ScriptedHandler.script = [
            (503, b'{"error": "overloaded"}'),
            (200, _score_body(8.0)),
        ]

        result = self._scorer().score("essay text", "rubric text")

        self.assertAlmostEqual(result.score, 8.0)
        self.assertEqual(_ScriptedHandler.attempts, 2)
        self.assertEqual(self.sleep.call_count, 1)

    def test_retries_each_retryable_status(self) -> None:
        for status in (408, 409, 429, 500, 502, 503, 504):
            with self.subTest(status=status):
                _ScriptedHandler.script = [(status, b"{}"), (200, _score_body())]
                _ScriptedHandler.attempts = 0

                self._scorer().score("essay text", "rubric text")

                self.assertEqual(_ScriptedHandler.attempts, 2)

    def test_gives_up_after_the_configured_retries(self) -> None:
        _ScriptedHandler.script = [(503, b'{"error": "still down"}')]

        with self.assertRaisesRegex(ProviderError, "HTTP error 503"):
            self._scorer(max_retries=2).score("essay text", "rubric text")

        self.assertEqual(_ScriptedHandler.attempts, 3)

    def test_does_not_retry_a_client_error(self) -> None:
        """A 401 will not fix itself, and retrying wastes the user's quota."""

        _ScriptedHandler.script = [(401, b'{"error": "bad key"}')]

        with self.assertRaisesRegex(ProviderError, "HTTP error 401"):
            self._scorer(max_retries=2).score("essay text", "rubric text")

        self.assertEqual(_ScriptedHandler.attempts, 1)
        self.sleep.assert_not_called()

    def test_reports_the_provider_error_body(self) -> None:
        _ScriptedHandler.script = [(400, b'{"error": "context length exceeded"}')]

        with self.assertRaisesRegex(ProviderError, "context length exceeded"):
            self._scorer(max_retries=0).score("essay text", "rubric text")

    def test_refuses_a_non_json_http_response(self) -> None:
        _ScriptedHandler.script = [(200, b"<html>gateway</html>")]

        with self.assertRaisesRegex(ProviderError, "non-JSON HTTP response"):
            self._scorer(max_retries=0).score("essay text", "rubric text")

    def test_refuses_a_response_without_a_message(self) -> None:
        for body in (b'{"choices": []}', b'{"choices": [{}]}', b"{}"):
            with self.subTest(body=body):
                _ScriptedHandler.script = [(200, body)]
                _ScriptedHandler.attempts = 0

                with self.assertRaisesRegex(
                    ProviderError, "missing choices\\[0\\].message.content"
                ):
                    self._scorer(max_retries=0).score("essay text", "rubric text")

    def test_reads_usage_totals_when_present(self) -> None:
        result = self._scorer(max_retries=0).score("essay text", "rubric text")

        self.assertEqual(result.prompt_tokens, 11)
        self.assertEqual(result.completion_tokens, 5)
        self.assertEqual(result.response_id, "mock-1")

    def test_tolerates_a_response_without_usage_totals(self) -> None:
        _ScriptedHandler.script = [
            (
                200,
                json.dumps(
                    {
                        "choices": [
                            {"message": {"content": json.dumps({"score": 5, "reason": "x"})}}
                        ],
                        "usage": {"prompt_tokens": "many"},
                    }
                ).encode("utf-8"),
            )
        ]

        result = self._scorer(max_retries=0).score("essay text", "rubric text")

        self.assertIsNone(result.prompt_tokens)
        self.assertIsNone(result.completion_tokens)
        self.assertIsNone(result.response_id)

    def test_refuses_empty_text_or_rubric(self) -> None:
        scorer = self._scorer(max_retries=0)

        with self.assertRaisesRegex(ValueError, "text must not be empty"):
            scorer.score("   ", "rubric text")
        with self.assertRaisesRegex(ValueError, "rubric must not be empty"):
            scorer.score("essay text", "   ")


class UnreachableEndpointTests(unittest.TestCase):
    """A refused connection is retried, then reported as a request failure."""

    def setUp(self) -> None:
        sleep_patcher = patch("agent_audit.provider.time.sleep")
        self.sleep = sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

        # Bind then release a port so nothing is listening on it.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.dead_port = probe.getsockname()[1]

    def test_retries_then_reports_a_failed_request(self) -> None:
        scorer = OpenAICompatibleScorer(
            OpenAICompatibleConfig(
                base_url=f"http://127.0.0.1:{self.dead_port}/v1",
                model="m",
                api_key="k",
                max_retries=2,
                # Short, because each attempt waits out this timeout: the OS
                # does not always refuse an unbound loopback port outright.
                timeout_seconds=0.25,
            )
        )

        with self.assertRaisesRegex(ProviderError, "Provider request failed"):
            scorer.score("essay text", "rubric text")

        self.assertEqual(self.sleep.call_count, 2)


class ConfigValidationTests(unittest.TestCase):
    BASE = {
        "base_url": "https://api.example.com/v1",
        "model": "m",
        "api_key": "k",
    }

    def _assert_refused(self, pattern: str, **overrides: object) -> None:
        config = OpenAICompatibleConfig(**{**self.BASE, **overrides})  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, pattern):
            config.validate()

    def test_refuses_a_relative_or_non_http_url(self) -> None:
        for value in ("api.example.com/v1", "ftp://api.example.com/v1", "https://"):
            with self.subTest(value=value):
                self._assert_refused("absolute HTTP\\(S\\) URL", base_url=value)

    def test_refuses_a_url_with_a_query_or_fragment(self) -> None:
        for value in (
            "https://api.example.com/v1?key=secret",
            "https://api.example.com/v1#frag",
        ):
            with self.subTest(value=value):
                self._assert_refused("query string or fragment", base_url=value)

    def test_refuses_an_empty_model_or_key(self) -> None:
        self._assert_refused("model must not be empty", model="  ")
        self._assert_refused("api_key must not be empty", api_key="  ")

    def test_refuses_non_finite_numbers(self) -> None:
        for field in ("score_min", "score_max", "temperature", "timeout_seconds"):
            with self.subTest(field=field):
                self._assert_refused("must be finite", **{field: float("nan")})

    def test_refuses_an_inverted_score_range(self) -> None:
        self._assert_refused(
            "score_max must be greater", score_min=10.0, score_max=1.0
        )

    def test_refuses_a_temperature_outside_the_api_range(self) -> None:
        for value in (-0.1, 2.5):
            with self.subTest(value=value):
                self._assert_refused("temperature must be between 0 and 2", temperature=value)

    def test_refuses_a_non_positive_timeout(self) -> None:
        self._assert_refused("timeout_seconds must be positive", timeout_seconds=0.0)

    def test_refuses_negative_retries(self) -> None:
        self._assert_refused("max_retries must be non-negative", max_retries=-1)

    def test_accepts_a_valid_configuration(self) -> None:
        OpenAICompatibleConfig(**self.BASE).validate()  # type: ignore[arg-type]


class UrlAndContentHelperTests(unittest.TestCase):
    def test_appends_the_chat_completions_path_once(self) -> None:
        for base in (
            "https://api.example.com/v1",
            "https://api.example.com/v1/",
            "https://api.example.com/v1/chat/completions",
        ):
            with self.subTest(base=base):
                self.assertEqual(
                    _chat_completions_url(base),
                    "https://api.example.com/v1/chat/completions",
                )

    def test_joins_multi_part_message_content(self) -> None:
        content = [
            {"type": "text", "text": '{"score": 7,'},
            {"type": "text", "text": ' "reason": "ok"}'},
        ]

        self.assertEqual(_content_to_text(content), '{"score": 7, "reason": "ok"}')

    def test_ignores_non_text_parts_but_keeps_text_ones(self) -> None:
        content = [{"type": "image", "url": "x"}, {"type": "text", "text": "kept"}]

        self.assertEqual(_content_to_text(content), "kept")

    def test_refuses_content_that_holds_no_text(self) -> None:
        for content in ([], [{"type": "image", "url": "x"}], 7, None, {}):
            with self.subTest(content=content):
                with self.assertRaisesRegex(ProviderError, "content is not text"):
                    _content_to_text(content)


if __name__ == "__main__":
    unittest.main()
