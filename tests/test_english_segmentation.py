"""Tests for English sentence segmentation.

Chinese terminators are unambiguous; English periods are not. An abbreviation
or a decimal that splits into an extra sentence shifts every index after it,
and the annotation is given by index — so the generator would delete a
different sentence than the reviewer marked and still call the result a
degradation. These tests pin the cases that cause that.
"""

from __future__ import annotations

import random
import unittest

from agent_audit.models import BaselineCase
from agent_audit.segmentation import CHINESE, ENGLISH, split_sentences
from agent_audit.variants import (
    _assert_evidence_removed,
    _substitute_at_clause_boundaries,
    build_variant,
)


def _split(text: str) -> tuple[str, ...]:
    return split_sentences(text, ENGLISH)


class EnglishSplitTests(unittest.TestCase):
    def test_rejoining_the_split_reproduces_the_input(self) -> None:
        for text in (
            "One. Two! Three?",
            "Dr. Smith argues that sleep matters. Evidence shows a gap.",
            "The gap is 3.5 hours. That matters.",
            "",
            "No terminator here",
            'He said "stop." Then he left.',
        ):
            with self.subTest(text=text):
                self.assertEqual("".join(_split(text)), text)

    def test_splits_on_each_terminator(self) -> None:
        self.assertEqual(_split("One. Two! Three?"), ("One.", " Two!", " Three?"))

    def test_does_not_split_inside_a_title_abbreviation(self) -> None:
        text = "Dr. Smith argues that sleep matters. Evidence shows a gap."

        self.assertEqual(len(_split(text)), 2)
        self.assertTrue(_split(text)[0].startswith("Dr. Smith"))

    def test_does_not_split_inside_a_decimal_number(self) -> None:
        text = "The gap is 3.5 hours. That matters."

        self.assertEqual(len(_split(text)), 2)

    def test_does_not_split_inside_a_latin_abbreviation(self) -> None:
        for text in (
            "We used e.g. surveys. They worked.",
            "That is, i.e. the point. It holds.",
        ):
            with self.subTest(text=text):
                self.assertEqual(len(_split(text)), 2)

    def test_does_not_split_before_a_lowercase_continuation(self) -> None:
        """A lowercase word after a period is a strong signal of no boundary."""

        self.assertEqual(len(_split("Rated 8. out of ten overall.")), 1)

    def test_does_not_split_between_initials(self) -> None:
        self.assertEqual(len(_split("J. K. Rowling wrote it. It sold well.")), 2)

    def test_absorbs_a_closing_quote_after_the_terminator(self) -> None:
        self.assertEqual(
            _split('He said "stop." Then he left.'),
            ('He said "stop."', " Then he left."),
        )

    def test_a_question_mark_always_ends_a_sentence(self) -> None:
        self.assertEqual(len(_split("Why? because it works.")), 2)

    def test_a_trailing_fragment_is_kept(self) -> None:
        """A capitalised continuation without a final terminator still splits."""

        self.assertEqual(_split("One. Two"), ("One.", " Two"))

    def test_empty_text_yields_no_sentences(self) -> None:
        self.assertEqual(_split(""), ())


class StrategySelectionTests(unittest.TestCase):
    def test_the_default_language_is_still_chinese(self) -> None:
        self.assertEqual(split_sentences("甲。乙。"), ("甲。", "乙。"))

    def test_each_strategy_knows_whether_words_need_boundaries(self) -> None:
        """Chinese characters are alphabetic, so the rule must not apply there."""

        self.assertTrue(ENGLISH.requires_word_boundaries)
        self.assertFalse(CHINESE.requires_word_boundaries)

    def test_the_english_corpus_is_populated(self) -> None:
        self.assertEqual(ENGLISH.name, "english")
        self.assertGreaterEqual(len(ENGLISH.padding_sentences), 4)
        self.assertGreaterEqual(len(ENGLISH.unsupported_assertions), 2)
        self.assertTrue(ENGLISH.flattery_sentence)
        self.assertGreaterEqual(len(ENGLISH.connectives), 6)

    def test_every_english_corpus_entry_is_one_sentence(self) -> None:
        entries = (
            *ENGLISH.padding_sentences,
            *ENGLISH.unsupported_assertions,
            ENGLISH.flattery_sentence,
        )
        for entry in entries:
            with self.subTest(entry=entry):
                self.assertEqual(len(_split(entry)), 1)

    def test_english_connective_sources_and_targets_are_disjoint(self) -> None:
        sources = {source for source, _ in ENGLISH.connectives}
        targets = {target for _, target in ENGLISH.connectives}

        self.assertEqual(sources & targets, set())

    def test_english_connectives_cover_both_capitalisations(self) -> None:
        """A connective starts a sentence capitalised and a clause lowercase."""

        sources = {source for source, _ in ENGLISH.connectives}

        self.assertIn("Therefore", sources)
        self.assertIn("therefore", sources)


class EnglishVariantTextTests(unittest.TestCase):
    """Appended corpus text must not run into the essay.

    Chinese sentences abut directly; English ones need a space. Without it the
    variant is malformed English, and a grader may react to the formatting
    rather than to the intervention being tested.
    """

    TEXT = (
        "Schools should start later. "
        "A survey found an eight hour sleep gap. "
        "Therefore a trial is worth running."
    )

    def _case(self) -> BaselineCase:
        return BaselineCase("c1", self.TEXT, (2,), "")

    def _build(self, strategy: str):
        return build_variant(self._case(), strategy, ENGLISH, random.Random(0))

    def test_each_language_declares_how_sentences_join(self) -> None:
        self.assertEqual(ENGLISH.sentence_separator, " ")
        self.assertEqual(CHINESE.sentence_separator, "")

    def test_padding_is_separated_from_the_essay(self) -> None:
        variant = self._build("verbose_padding")

        self.assertTrue(variant.text.startswith(self.TEXT))
        self.assertTrue(variant.text[len(self.TEXT)].isspace())

    def test_flattery_is_separated_from_the_essay(self) -> None:
        variant = self._build("rubric_flattery")

        self.assertTrue(variant.text[len(self.TEXT)].isspace())

    def test_the_replacement_claim_is_separated_from_the_kept_text(self) -> None:
        variant = self._build("unsupported_assertion")
        assertion = variant.inserted_text[0]

        position = variant.text.index(assertion)
        self.assertTrue(variant.text[position - 1].isspace())

    def test_no_generated_variant_glues_two_sentences_together(self) -> None:
        for strategy in ("verbose_padding", "rubric_flattery", "unsupported_assertion"):
            with self.subTest(strategy=strategy):
                text = self._build(strategy).text

                self.assertNotRegex(text, r"[.!?][A-Z]")

    def test_a_degradation_does_not_start_with_whitespace(self) -> None:
        case = BaselineCase("c1", self.TEXT, (1,), "")

        variant = build_variant(case, "remove_evidence", ENGLISH, random.Random(0))

        self.assertEqual(variant.text, variant.text.lstrip())


class EnglishWordBoundaryTests(unittest.TestCase):
    """English must replace whole words only."""

    def _sub(self, text: str):
        return _substitute_at_clause_boundaries(
            text, ENGLISH.connectives, require_word_boundaries=True
        )

    def test_a_connective_inside_a_longer_word_is_left_alone(self) -> None:
        # "Finally" is a connective; "Finalised" merely starts with it.
        text = "Finalised reports are due. They matter."

        self.assertEqual(self._sub(text), (text, 0))

    def test_a_whole_word_connective_is_replaced(self) -> None:
        produced, replaced = self._sub("Finally, reports are due.")

        self.assertEqual(replaced, 1)
        self.assertTrue(produced.startswith("Lastly,"))

    def test_chinese_must_not_use_the_word_boundary_rule(self) -> None:
        """Chinese characters are alphabetic, so the rule would reject everything."""

        with_rule, replaced_with = _substitute_at_clause_boundaries(
            "因此可以先试行。", CHINESE.connectives, require_word_boundaries=True
        )
        without_rule, replaced_without = _substitute_at_clause_boundaries(
            "因此可以先试行。", CHINESE.connectives, require_word_boundaries=False
        )

        self.assertEqual(replaced_with, 0)
        self.assertEqual(replaced_without, 1)
        self.assertNotEqual(with_rule, without_rule)


class SeamWhitespaceTests(unittest.TestCase):
    def test_a_whitespace_only_kept_sentence_is_skipped(self) -> None:
        """Trailing whitespace can split off as a sentence with no content."""

        case = BaselineCase("c1", "One. Two.  ", (1,), "")

        _assert_evidence_removed(case, "Two.", ("Two.", "  "), "probe", ENGLISH)


if __name__ == "__main__":
    unittest.main()
