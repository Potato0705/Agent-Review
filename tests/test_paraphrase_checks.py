"""Tests for the paraphrase review checks.

The design rests on one measured fact: checks that look sensible can reject
genuinely equivalent rewrites. Number and negation checks rejected 2 of the 5
hand-written paraphrases in this repo — the "missing number" in one was the 一
inside 统一, and the "lost negation" in another was 不足 rewritten as 缺乏.

So those two are notes, not gates, and the regression test below keeps every
gate honest against the real paraphrases.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from agent_audit.io import load_scoring_cases
from dataclasses import replace

from agent_audit.paraphrase import build_messages, review_checks
from agent_audit.segmentation import CHINESE, ENGLISH


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "学校应推迟上课时间。一项调查显示睡眠充足的学生成绩更稳定。"


class BlockingCheckTests(unittest.TestCase):
    def test_a_reasonable_rewrite_is_not_blocked(self) -> None:
        draft = "中学可以把上课时间往后调。校内调查发现睡眠充足的学生成绩更稳。"

        blocking, _ = review_checks(BASELINE, draft, CHINESE)

        self.assertEqual(blocking, ())

    def test_an_empty_draft_is_blocked(self) -> None:
        for draft in ("", "   ", "\n"):
            with self.subTest(draft=repr(draft)):
                blocking, _ = review_checks(BASELINE, draft, CHINESE)
                self.assertIn("empty", blocking)

    def test_an_unchanged_draft_is_blocked(self) -> None:
        blocking, _ = review_checks(BASELINE, BASELINE, CHINESE)

        self.assertIn("identical", blocking)

    def test_a_draft_that_echoes_the_baseline_is_blocked(self) -> None:
        """A model that answers with 'Here is the text: <original>' is echoing."""

        draft = f"以下是改写后的文本：{BASELINE}希望对你有帮助。"

        blocking, _ = review_checks(BASELINE, draft, CHINESE)

        self.assertIn("echoes_baseline", blocking)

    def test_a_truncated_draft_is_blocked(self) -> None:
        blocking, _ = review_checks(BASELINE, "学校应推迟。", CHINESE)

        self.assertIn("length_out_of_band", blocking)

    def test_a_runaway_draft_is_blocked(self) -> None:
        blocking, _ = review_checks(BASELINE, "改写。" * 200, CHINESE)

        self.assertIn("length_out_of_band", blocking)


class ReviewNoteTests(unittest.TestCase):
    """Notes point the reviewer at what to check; they never gate."""

    def test_a_missing_number_is_reported_without_blocking(self) -> None:
        baseline = "睡眠达到八小时的学生成绩更稳定。调查覆盖了三个班级。"
        draft = "睡眠充足的学生成绩更稳定。调查覆盖了三个班级。"

        blocking, notes = review_checks(baseline, draft, CHINESE)

        self.assertEqual(blocking, ())
        self.assertTrue(any("八" in note for note in notes))

    def test_a_negation_change_is_reported_without_blocking(self) -> None:
        baseline = "学校不应全面禁止手机。午休时仍需联系家人。"
        draft = "学校应当允许使用手机。午休时仍需联系家人。"

        blocking, notes = review_checks(baseline, draft, CHINESE)

        self.assertEqual(blocking, ())
        self.assertTrue(any("negation" in note for note in notes))

    def test_a_faithful_rewrite_produces_no_notes(self) -> None:
        baseline = "睡眠达到八小时的学生成绩更稳定。"
        draft = "睡眠达到八小时的学生，成绩表现更为稳定。"

        _, notes = review_checks(baseline, draft, CHINESE)

        self.assertEqual(notes, ())

    def test_english_negations_are_recognised(self) -> None:
        baseline = "Schools should not ban phones. Students need to reach family."
        draft = "Schools should allow phones. Students need to reach family."

        blocking, notes = review_checks(baseline, draft, ENGLISH)

        self.assertEqual(blocking, ())
        self.assertTrue(any("negation" in note for note in notes))


class PromptTests(unittest.TestCase):
    """The prompt is where circularity would sneak in through a side door."""

    RUBRIC = "只评价文章针对题目所展现的写作质量，不因篇幅或术语数量加分。"

    def test_the_prompt_carries_only_the_instruction_and_the_text(self) -> None:
        messages = build_messages(BASELINE, CHINESE)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1], {"role": "user", "content": BASELINE})

    def test_the_rubric_never_reaches_the_rewriter(self) -> None:
        """A rewrite optimised against the criteria under test is circular."""

        joined = " ".join(message["content"] for message in build_messages(BASELINE, CHINESE))

        self.assertNotIn(self.RUBRIC, joined)
        for word in ("评分", "打分", "rubric", "score"):
            with self.subTest(word=word):
                self.assertNotIn(word, joined.lower())

    def test_the_prompt_does_not_say_what_the_rewrite_is_for(self) -> None:
        """Telling the model it probes a grader invites adversarial rewriting."""

        joined = " ".join(message["content"] for message in build_messages(BASELINE, ENGLISH))

        for word in ("grader", "test", "evaluat", "audit"):
            with self.subTest(word=word):
                self.assertNotIn(word, joined.lower())

    def test_each_language_supplies_its_own_instruction(self) -> None:
        self.assertNotEqual(
            build_messages(BASELINE, CHINESE)[0]["content"],
            build_messages(BASELINE, ENGLISH)[0]["content"],
        )
        self.assertTrue(CHINESE.paraphrase_instruction)
        self.assertTrue(ENGLISH.paraphrase_instruction)

    def test_a_language_without_negation_markers_reports_zero(self) -> None:
        silent = replace(CHINESE, negation_markers=())

        _, notes = review_checks("学校不应禁止。可以试行。", "学校应当允许。可以试行。", silent)

        self.assertEqual(notes, ())


class RealParaphraseRegressionTests(unittest.TestCase):
    """The measured fact the design is built on, kept as a standing test."""

    def _pairs(self) -> list[tuple[str, str, str]]:
        grouped: dict[str, dict[str, str]] = {}
        for case in load_scoring_cases(ROOT / "examples" / "essay_cases.csv"):
            grouped.setdefault(case.case_id, {})[case.variant_type] = case.text
        return [
            (case_id, texts["baseline"], texts["paraphrase"])
            for case_id, texts in grouped.items()
            if "paraphrase" in texts
        ]

    def test_no_hand_written_paraphrase_is_ever_blocked(self) -> None:
        pairs = self._pairs()

        self.assertGreaterEqual(len(pairs), 5, "fixture looks wrong")
        for case_id, baseline, paraphrase in pairs:
            with self.subTest(case=case_id):
                blocking, _ = review_checks(baseline, paraphrase, CHINESE)

                self.assertEqual(
                    blocking,
                    (),
                    f"{case_id} is a genuine equivalent rewrite and must not be gated",
                )

    def test_the_notes_still_fire_on_those_same_paraphrases(self) -> None:
        """Proof the two demoted checks were demoted, not deleted."""

        noted = [
            case_id
            for case_id, baseline, paraphrase in self._pairs()
            if review_checks(baseline, paraphrase, CHINESE)[1]
        ]

        self.assertGreaterEqual(len(noted), 1)


if __name__ == "__main__":
    unittest.main()
