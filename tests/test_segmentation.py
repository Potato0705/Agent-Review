"""Tests for sentence segmentation.

Everything the generator does to a degradation variant is expressed as
"delete sentence N". That is only precise if splitting loses nothing, so the
round-trip invariant is asserted against every real case text in the repo
rather than a handful of toy strings.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from agent_audit.io import load_scoring_cases
from agent_audit.segmentation import CHINESE, split_sentences


ROOT = Path(__file__).resolve().parents[1]


class SplitSentencesTests(unittest.TestCase):
    def test_rejoining_the_split_reproduces_every_real_case(self) -> None:
        cases = load_scoring_cases(ROOT / "examples" / "essay_cases.csv")

        self.assertGreater(len(cases), 10, "fixture looks wrong")
        for case in cases:
            with self.subTest(case=f"{case.case_id}/{case.variant_id}"):
                self.assertEqual("".join(split_sentences(case.text)), case.text)

    def test_keeps_the_terminator_with_its_sentence(self) -> None:
        self.assertEqual(split_sentences("甲。乙！丙？"), ("甲。", "乙！", "丙？"))

    def test_splits_on_a_semicolon(self) -> None:
        self.assertEqual(split_sentences("甲；乙。"), ("甲；", "乙。"))

    def test_absorbs_closing_marks_that_follow_the_terminator(self) -> None:
        self.assertEqual(
            split_sentences("他说“好。”然后走了。"), ("他说“好。”", "然后走了。")
        )

    def test_absorbs_a_closing_bracket(self) -> None:
        self.assertEqual(split_sentences("甲（乙。）丙。"), ("甲（乙。）", "丙。"))

    def test_consecutive_terminators_end_separate_sentences(self) -> None:
        self.assertEqual(split_sentences("甲。。乙。"), ("甲。", "。", "乙。"))

    def test_text_without_a_terminator_is_a_single_sentence(self) -> None:
        self.assertEqual(split_sentences("没有句号"), ("没有句号",))

    def test_a_trailing_fragment_is_kept(self) -> None:
        self.assertEqual(split_sentences("甲。尾巴"), ("甲。", "尾巴"))

    def test_empty_text_yields_no_sentences(self) -> None:
        self.assertEqual(split_sentences(""), ())

    def test_whitespace_between_sentences_is_never_dropped(self) -> None:
        text = "甲。 乙。"

        self.assertEqual("".join(split_sentences(text)), text)


class ChineseStrategyTests(unittest.TestCase):
    def test_the_strategy_exposes_a_non_empty_corpus(self) -> None:
        self.assertEqual(CHINESE.name, "chinese")
        self.assertGreaterEqual(len(CHINESE.padding_sentences), 4)
        self.assertGreaterEqual(len(CHINESE.unsupported_assertions), 2)
        self.assertTrue(CHINESE.flattery_sentence)
        self.assertGreaterEqual(len(CHINESE.connectives), 4)

    def test_every_corpus_sentence_is_a_single_sentence(self) -> None:
        """Corpus entries are appended verbatim, so they must be well formed."""

        entries = (
            *CHINESE.padding_sentences,
            *CHINESE.unsupported_assertions,
            CHINESE.flattery_sentence,
        )
        for entry in entries:
            with self.subTest(entry=entry):
                self.assertEqual(len(split_sentences(entry)), 1)

    def test_connective_replacements_are_distinct(self) -> None:
        for source, target in CHINESE.connectives:
            with self.subTest(source=source):
                self.assertNotEqual(source, target)


if __name__ == "__main__":
    unittest.main()
