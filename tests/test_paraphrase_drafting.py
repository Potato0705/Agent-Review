"""Tests for drafting paraphrases and round-tripping the review file.

The review file is what a human actually works in, so its contract matters as
much as the drafting: a typo in `status` must not silently drop an approved
row, and a draft must not survive an edit to the baseline it was written for.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from agent_audit.io import (
    load_paraphrase_review,
    stable_hash,
    write_paraphrase_review,
)
from agent_audit.models import BaselineCase
from agent_audit.paraphrase import draft_paraphrases
from agent_audit.provider import ProviderError
from agent_audit.segmentation import CHINESE


BASELINE = "学校应推迟上课时间。一项调查显示睡眠充足的学生成绩更稳定。"
GOOD_DRAFT = "中学可以把上课时间往后调。校内调查发现睡眠充足的学生成绩更稳。"


def _cases() -> list[BaselineCase]:
    return [
        BaselineCase("c1", BASELINE, (2,), "原始短文"),
        BaselineCase("c2", BASELINE, (2,), ""),
    ]


class _StubRewriter:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        self.calls.append(messages)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply, {"id": f"resp-{len(self.calls)}"}


class _FailingRewriter:
    def __init__(self, fail_on: int) -> None:
        self.fail_on = fail_on
        self.calls = 0

    def complete(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        self.calls += 1
        if self.calls == self.fail_on:
            raise ProviderError("provider exploded")
        return GOOD_DRAFT, {}


class DraftingTests(unittest.TestCase):
    def test_one_draft_per_case_with_the_baseline_fingerprint(self) -> None:
        rewriter = _StubRewriter(GOOD_DRAFT)

        drafts = draft_paraphrases(_cases(), rewriter, CHINESE)

        self.assertEqual([d.case_id for d in drafts], ["c1", "c2"])
        self.assertEqual(drafts[0].draft_text, GOOD_DRAFT)
        self.assertEqual(drafts[0].baseline_sha256, stable_hash(BASELINE))

    def test_the_model_only_sees_the_baseline(self) -> None:
        rewriter = _StubRewriter(GOOD_DRAFT)

        draft_paraphrases(_cases()[:1], rewriter, CHINESE)

        self.assertEqual(rewriter.calls[0][1]["content"], BASELINE)

    def test_a_clean_draft_starts_as_pending(self) -> None:
        drafts = draft_paraphrases(_cases()[:1], _StubRewriter(GOOD_DRAFT), CHINESE)

        self.assertEqual(drafts[0].status, "pending")
        self.assertEqual(drafts[0].blocking_checks, ())

    def test_a_failing_check_starts_as_blocked(self) -> None:
        drafts = draft_paraphrases(_cases()[:1], _StubRewriter(BASELINE), CHINESE)

        self.assertEqual(drafts[0].status, "blocked")
        self.assertIn("identical", drafts[0].blocking_checks)

    def test_the_raw_reply_and_latency_are_recorded(self) -> None:
        drafts = draft_paraphrases(_cases()[:1], _StubRewriter(GOOD_DRAFT), CHINESE)

        self.assertEqual(drafts[0].raw_content, GOOD_DRAFT)
        self.assertGreaterEqual(drafts[0].latency_seconds, 0.0)

    def test_a_provider_failure_names_the_case_it_died_on(self) -> None:
        """A failed run writes nothing, so the message must say where to resume."""

        with self.assertRaisesRegex(ProviderError, "c2"):
            draft_paraphrases(_cases(), _FailingRewriter(fail_on=2), CHINESE)

    def test_no_cases_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one baseline case"):
            draft_paraphrases([], _StubRewriter(GOOD_DRAFT), CHINESE)


class ReviewFileTests(unittest.TestCase):
    def _written(self) -> Path:
        drafts = draft_paraphrases(_cases(), _StubRewriter(GOOD_DRAFT), CHINESE)
        path = Path(tempfile.mkdtemp()) / "review.csv"
        write_paraphrase_review(path, drafts)
        return path

    def test_a_written_file_round_trips(self) -> None:
        rows = load_paraphrase_review(self._written())

        self.assertEqual([row.case_id for row in rows], ["c1", "c2"])
        self.assertEqual(rows[0].draft_text, GOOD_DRAFT)
        self.assertEqual(rows[0].status, "pending")
        self.assertEqual(rows[0].baseline_sha256, stable_hash(BASELINE))

    def test_an_unrecognised_status_is_refused(self) -> None:
        """Silently skipping `aproved` would drop a row the reviewer approved."""

        path = self._written()
        text = path.read_text(encoding="utf-8").replace("pending", "aproved", 1)
        path.write_text(text, encoding="utf-8", newline="\n")

        with self.assertRaisesRegex(ValueError, "status must be one of"):
            load_paraphrase_review(path)

    def test_every_declared_status_is_accepted(self) -> None:
        pristine = self._written().read_text(encoding="utf-8")

        for status in ("pending", "blocked", "approved", "rejected"):
            with self.subTest(status=status):
                path = Path(tempfile.mkdtemp()) / "review.csv"
                path.write_text(
                    pristine.replace(",pending,", f",{status},"),
                    encoding="utf-8",
                    newline="\n",
                )

                rows = load_paraphrase_review(path)

                self.assertTrue(all(row.status == status for row in rows))

    def test_a_missing_file_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            load_paraphrase_review(Path(tempfile.mkdtemp()) / "absent.csv")

    def test_missing_columns_are_refused(self) -> None:
        path = Path(tempfile.mkdtemp()) / "review.csv"
        path.write_text("case_id,draft_text\nc1,x\n", encoding="utf-8", newline="\n")

        with self.assertRaisesRegex(ValueError, "missing columns"):
            load_paraphrase_review(path)

    def test_an_empty_case_id_or_draft_is_refused(self) -> None:
        path = self._written()
        text = path.read_text(encoding="utf-8").replace(GOOD_DRAFT, "", 1)
        path.write_text(text, encoding="utf-8", newline="\n")

        with self.assertRaisesRegex(ValueError, "empty required values"):
            load_paraphrase_review(path)

    def test_a_header_only_file_is_refused(self) -> None:
        path = self._written()
        header = path.read_text(encoding="utf-8").splitlines()[0]
        path.write_text(header + "\n", encoding="utf-8", newline="\n")

        with self.assertRaisesRegex(ValueError, "no review rows"):
            load_paraphrase_review(path)


if __name__ == "__main__":
    unittest.main()
