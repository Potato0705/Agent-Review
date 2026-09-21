"""Tests for the variant strategies and their postconditions.

Each strategy carries a machine-checkable claim about what it did. A gaming
variant that quietly dropped a sentence, or a degradation variant that kept
the evidence, would be mislabelled — and a mislabelled variant makes every
conclusion drawn from it unfounded. The postconditions are what stop that, so
they are tested directly, including the case where they must refuse.
"""

from __future__ import annotations

import random
import unittest

from agent_audit.models import BaselineCase
from agent_audit.segmentation import CHINESE, split_sentences
from agent_audit.variants import (
    _assert_evidence_removed,
    _substitute_at_clause_boundaries,
    _kept_sentences,
    GAMING_STRATEGIES,
    DEGRADATION_STRATEGIES,
    PARAPHRASE_STRATEGIES,
    VariantPostconditionError,
    build_variant,
)


TEXT = (
    "学校应推迟上课时间。"
    "一项调查显示睡眠充足的学生成绩更稳定。"
    "因此可以先试行一个学期。"
)
CASE = BaselineCase(case_id="c1", text=TEXT, evidence_sentences=(2,), notes="原始短文")


def _build(strategy: str, case: BaselineCase = CASE, seed: int = 0):
    return build_variant(case, strategy, CHINESE, random.Random(seed))


class GamingStrategyTests(unittest.TestCase):
    def test_every_gaming_strategy_only_adds_text(self) -> None:
        for strategy in GAMING_STRATEGIES:
            with self.subTest(strategy=strategy):
                variant = _build(strategy)

                self.assertEqual(variant.variant_type, "gaming")
                self.assertTrue(variant.text.startswith(CASE.text))
                self.assertGreater(len(variant.text), len(CASE.text))

    def test_padding_appends_two_corpus_sentences(self) -> None:
        variant = _build("verbose_padding")

        self.assertEqual(len(variant.inserted_text), 2)
        self.assertEqual(len(set(variant.inserted_text)), 2)
        for sentence in variant.inserted_text:
            self.assertIn(sentence, CHINESE.padding_sentences)
            self.assertIn(sentence, variant.text)

    def test_flattery_appends_the_rubric_term_sentence(self) -> None:
        variant = _build("rubric_flattery")

        self.assertEqual(variant.inserted_text, (CHINESE.flattery_sentence,))

    def test_the_magnitude_records_how_much_was_added(self) -> None:
        variant = _build("verbose_padding")

        self.assertEqual(variant.magnitude["added_sentences"], 2)
        self.assertGreater(variant.magnitude["added_characters"], 0)


class DegradationStrategyTests(unittest.TestCase):
    def test_remove_evidence_deletes_exactly_the_annotated_sentence(self) -> None:
        variant = _build("remove_evidence")

        sentences = split_sentences(CASE.text)
        self.assertEqual(variant.variant_type, "degradation")
        self.assertLess(len(variant.text), len(CASE.text))
        self.assertNotIn(sentences[1], variant.text)
        self.assertIn(sentences[0], variant.text)
        self.assertIn(sentences[2], variant.text)

    def test_unsupported_assertion_swaps_evidence_for_a_bare_claim(self) -> None:
        variant = _build("unsupported_assertion")

        sentences = split_sentences(CASE.text)
        self.assertNotIn(sentences[1], variant.text)
        self.assertEqual(len(variant.inserted_text), 1)
        self.assertIn(variant.inserted_text[0], CHINESE.unsupported_assertions)
        self.assertIn(variant.inserted_text[0], variant.text)

    def test_the_magnitude_records_the_share_of_characters_removed(self) -> None:
        variant = _build("remove_evidence")

        removed = variant.magnitude["removed_characters"]
        self.assertEqual(removed, len(split_sentences(CASE.text)[1]))
        self.assertAlmostEqual(
            variant.magnitude["removed_share"], removed / len(CASE.text), places=9
        )

    def test_removes_every_annotated_sentence(self) -> None:
        case = BaselineCase(case_id="c1", text=TEXT, evidence_sentences=(2, 3))

        variant = build_variant(case, "remove_evidence", CHINESE, random.Random(0))

        sentences = split_sentences(TEXT)
        self.assertEqual(variant.text, sentences[0])


class ParaphraseStrategyTests(unittest.TestCase):
    TEXT_WITH_CONNECTIVE = (
        "学校应推迟上课时间。因此可以先试行一个学期。但是需要记录迟到率。"
    )

    def _case(self) -> BaselineCase:
        return BaselineCase(
            case_id="c1", text=self.TEXT_WITH_CONNECTIVE, evidence_sentences=(2,)
        )

    def test_only_connectives_change(self) -> None:
        variant = _build("connective_substitution", self._case())

        self.assertEqual(variant.variant_type, "paraphrase")
        self.assertNotEqual(variant.text, self.TEXT_WITH_CONNECTIVE)

        stripped_before = self.TEXT_WITH_CONNECTIVE
        stripped_after = variant.text
        for source, target in CHINESE.connectives:
            stripped_before = stripped_before.replace(source, "").replace(target, "")
            stripped_after = stripped_after.replace(source, "").replace(target, "")
        self.assertEqual(stripped_before, stripped_after)

    def test_the_magnitude_records_how_many_connectives_moved(self) -> None:
        variant = _build("connective_substitution", self._case())

        self.assertEqual(variant.magnitude["replaced_connectives"], 2)

    def test_refuses_a_text_with_no_connective_to_replace(self) -> None:
        case = BaselineCase(
            case_id="c1", text="学校应推迟上课时间。可以先试行。", evidence_sentences=(2,)
        )

        with self.assertRaisesRegex(VariantPostconditionError, "no connective"):
            build_variant(case, "connective_substitution", CHINESE, random.Random(0))


class ClauseBoundarySubstitutionTests(unittest.TestCase):
    """Chinese words overlap, so a plain replace corrupts text."""

    def _sub(self, text: str):
        return _substitute_at_clause_boundaries(text, CHINESE.connectives)

    def test_leaves_a_connective_that_spans_two_words_alone(self) -> None:
        # 原因此外 is 原因 + 此外; a naive replace of 因此 would produce 原所以外.
        text = "原因此外还有别的理由。"

        self.assertEqual(self._sub(text), (text, 0))

    def test_replaces_a_connective_at_the_start_of_the_text(self) -> None:
        self.assertEqual(
            self._sub("因此可以先试行。"), ("所以可以先试行。", 1)
        )

    def test_replaces_a_connective_after_a_comma(self) -> None:
        self.assertEqual(
            self._sub("城市应增设，因为载客量高。"),
            ("城市应增设，由于载客量高。", 1),
        )

    def test_replaces_a_connective_after_a_full_stop(self) -> None:
        self.assertEqual(
            self._sub("甲。但是乙。"), ("甲。然而乙。", 1)
        )

    def test_counts_every_replacement(self) -> None:
        self.assertEqual(self._sub("因此甲。但是乙。")[1], 2)

    def test_sources_and_targets_are_disjoint(self) -> None:
        """A target that is also a source would undo an earlier substitution."""

        sources = {source for source, _ in CHINESE.connectives}
        targets = {target for _, target in CHINESE.connectives}

        self.assertEqual(sources & targets, set())


class PostconditionTests(unittest.TestCase):
    """The postconditions must refuse, not merely describe."""

    def test_a_gaming_variant_that_drops_text_is_refused(self) -> None:
        broken = CHINESE.__class__(
            name="broken",
            padding_sentences=("",),
            flattery_sentence="",
            unsupported_assertions=CHINESE.unsupported_assertions,
            connectives=CHINESE.connectives,
        )

        with self.assertRaisesRegex(VariantPostconditionError, "must only add"):
            build_variant(CASE, "rubric_flattery", broken, random.Random(0))

    def test_a_degradation_variant_that_keeps_the_evidence_is_refused(self) -> None:
        # Annotate a sentence that also appears verbatim elsewhere, so deleting
        # one copy leaves the text still containing it.
        repeated = "甲。乙。乙。"
        case = BaselineCase(case_id="c1", text=repeated, evidence_sentences=(2,))

        with self.assertRaisesRegex(VariantPostconditionError, "still contains"):
            build_variant(case, "remove_evidence", CHINESE, random.Random(0))

    def test_a_kept_sentence_that_went_missing_is_refused(self) -> None:
        """Guards a future strategy that reorders or drops retained text."""

        kept = _kept_sentences(CASE)

        with self.assertRaisesRegex(VariantPostconditionError, "dropped or reordered"):
            _assert_evidence_removed(CASE, "".join(reversed(kept)), kept, "probe")

    def test_an_empty_annotation_cannot_produce_a_degradation(self) -> None:
        case = BaselineCase(case_id="c1", text=TEXT, evidence_sentences=())

        with self.assertRaisesRegex(VariantPostconditionError, "did not shorten"):
            build_variant(case, "remove_evidence", CHINESE, random.Random(0))

    def test_an_empty_replacement_claim_is_refused(self) -> None:
        broken = CHINESE.__class__(
            name="broken",
            padding_sentences=CHINESE.padding_sentences,
            flattery_sentence=CHINESE.flattery_sentence,
            unsupported_assertions=("",),
            connectives=CHINESE.connectives,
        )

        with self.assertRaisesRegex(VariantPostconditionError, "did not insert"):
            build_variant(CASE, "unsupported_assertion", broken, random.Random(0))

    def test_a_connective_table_that_rewrites_more_than_connectives_is_refused(
        self,
    ) -> None:
        """A replacement containing its own source corrupts the exact invariant."""

        broken = CHINESE.__class__(
            name="broken",
            padding_sentences=CHINESE.padding_sentences,
            flattery_sentence=CHINESE.flattery_sentence,
            unsupported_assertions=CHINESE.unsupported_assertions,
            connectives=(("因此", "因此所以"),),
        )
        case = BaselineCase(
            case_id="c1",
            text="学校应推迟上课时间。因此可以先试行。",
            evidence_sentences=(2,),
        )

        with self.assertRaisesRegex(
            VariantPostconditionError, "outside the connective table"
        ):
            build_variant(case, "connective_substitution", broken, random.Random(0))

    def test_an_unknown_strategy_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown variant strategy"):
            _build("keyword_stuffing")


class DeterminismTests(unittest.TestCase):
    def test_the_same_seed_produces_the_same_variant(self) -> None:
        for strategy in (*GAMING_STRATEGIES, *DEGRADATION_STRATEGIES):
            with self.subTest(strategy=strategy):
                self.assertEqual(
                    _build(strategy, seed=7).text, _build(strategy, seed=7).text
                )

    def test_a_different_seed_changes_the_corpus_choice(self) -> None:
        texts = {_build("verbose_padding", seed=seed).text for seed in range(6)}

        self.assertGreater(len(texts), 1)

    def test_strategy_registries_are_disjoint_and_named(self) -> None:
        self.assertEqual(GAMING_STRATEGIES, ("verbose_padding", "rubric_flattery"))
        self.assertEqual(
            DEGRADATION_STRATEGIES, ("remove_evidence", "unsupported_assertion")
        )
        self.assertEqual(PARAPHRASE_STRATEGIES, ("connective_substitution",))


if __name__ == "__main__":
    unittest.main()
