"""Tests for merging generated variants into an existing case file.

Appending is only safe if it cannot launder hand-written content into a
verified `machine-generated` claim. Every preserved row is therefore compared
against what the generator would produce, and any difference downgrades the
whole set to `mixed`.
"""

from __future__ import annotations

import unittest

from agent_audit.models import BaselineCase, ScoringCase
from agent_audit.variants import (
    GAMING_STRATEGIES,
    generate_variants,
    merge_into_existing,
)


TEXT = (
    "学校应推迟上课时间。"
    "一项调查显示睡眠充足的学生成绩更稳定。"
    "因此可以先试行一个学期。"
)


def _cases() -> list[BaselineCase]:
    return [BaselineCase("c1", TEXT, (2,), "原始短文")]


def _as_scoring_cases(rows) -> list[ScoringCase]:
    return [
        ScoringCase(row.case_id, row.variant_id, row.variant_type, row.text, row.notes)
        for row in rows
    ]


class MergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = generate_variants(_cases(), seed=0).rows

    def test_merging_into_an_identical_set_changes_nothing(self) -> None:
        outcome = merge_into_existing(self.target, _as_scoring_cases(self.target))

        self.assertEqual(
            [(row.variant_id, row.text) for row in outcome.rows],
            [(row.variant_id, row.text) for row in self.target],
        )
        self.assertEqual(outcome.appended, ())
        self.assertEqual(outcome.edited, ())
        self.assertEqual(outcome.foreign, ())
        self.assertEqual(len(outcome.preserved), len(self.target))
        self.assertEqual(outcome.set_origin, "machine-generated")

    def test_missing_rows_are_appended(self) -> None:
        existing = _as_scoring_cases(
            [row for row in self.target if row.variant_id != "gaming_rubric_flattery"]
        )

        outcome = merge_into_existing(self.target, existing)

        self.assertEqual(outcome.appended, (("c1", "gaming_rubric_flattery"),))
        self.assertEqual(len(outcome.rows), len(self.target))
        self.assertEqual(outcome.set_origin, "machine-generated")

    def test_an_existing_row_is_kept_verbatim(self) -> None:
        edited = list(self.target)
        index = next(
            i for i, row in enumerate(edited) if row.variant_id == "gaming_verbose_padding"
        )
        existing = _as_scoring_cases(edited)
        existing[index] = ScoringCase(
            "c1", "gaming_verbose_padding", "gaming", "【手工改写】", "手工说明"
        )

        outcome = merge_into_existing(self.target, existing)

        kept = next(
            row for row in outcome.rows if row.variant_id == "gaming_verbose_padding"
        )
        self.assertEqual(kept.text, "【手工改写】")
        self.assertEqual(kept.notes, "手工说明")

    def test_an_edited_row_downgrades_the_whole_set(self) -> None:
        """The hazard: appending must not launder a hand edit into a proof."""

        existing = _as_scoring_cases(self.target)
        existing[1] = ScoringCase(
            existing[1].case_id,
            existing[1].variant_id,
            existing[1].variant_type,
            existing[1].text + "人工补充。",
            existing[1].notes,
        )

        outcome = merge_into_existing(self.target, existing)

        self.assertEqual(outcome.edited, ((existing[1].case_id, existing[1].variant_id),))
        self.assertEqual(outcome.set_origin, "mixed")

    def test_an_edited_note_also_downgrades_the_set(self) -> None:
        """Notes are hashed with the text, so editing one changes the set."""

        existing = _as_scoring_cases(self.target)
        existing[1] = ScoringCase(
            existing[1].case_id,
            existing[1].variant_id,
            existing[1].variant_type,
            existing[1].text,
            existing[1].notes + " | 人工批注",
        )

        outcome = merge_into_existing(self.target, existing)

        self.assertEqual(outcome.set_origin, "mixed")

    def test_a_hand_written_variant_is_kept_and_downgrades_the_set(self) -> None:
        existing = _as_scoring_cases(self.target)
        existing.append(
            ScoringCase("c1", "gaming_handwritten", "gaming", "人工写的作弊变体。", "")
        )

        outcome = merge_into_existing(self.target, existing)

        self.assertEqual(outcome.foreign, (("c1", "gaming_handwritten"),))
        self.assertIn(
            "gaming_handwritten", [row.variant_id for row in outcome.rows]
        )
        self.assertEqual(outcome.set_origin, "mixed")

    def test_rows_stay_grouped_by_case_with_foreign_rows_last(self) -> None:
        existing = _as_scoring_cases(self.target)
        existing.insert(
            0, ScoringCase("c1", "gaming_handwritten", "gaming", "人工写的。", "")
        )

        outcome = merge_into_existing(self.target, existing)

        ids = [row.variant_id for row in outcome.rows]
        self.assertEqual(ids[0], "baseline")
        self.assertEqual(ids[-1], "gaming_handwritten")

    def test_refuses_an_existing_file_with_duplicate_identities(self) -> None:
        """Two rows with one identity make "keep the existing one" ambiguous."""

        existing = _as_scoring_cases(self.target)
        existing.append(existing[1])

        with self.assertRaisesRegex(ValueError, "duplicate case/variant ids"):
            merge_into_existing(self.target, existing)

    def test_merging_is_deterministic(self) -> None:
        existing = _as_scoring_cases(
            [row for row in self.target if row.variant_id != "gaming_rubric_flattery"]
        )

        first = merge_into_existing(self.target, existing)
        second = merge_into_existing(self.target, existing)

        self.assertEqual(
            [(row.variant_id, row.text) for row in first.rows],
            [(row.variant_id, row.text) for row in second.rows],
        )

    def test_every_generated_strategy_is_present_after_merging(self) -> None:
        existing = _as_scoring_cases([self.target[0]])

        outcome = merge_into_existing(self.target, existing)

        for strategy in GAMING_STRATEGIES:
            with self.subTest(strategy=strategy):
                self.assertIn(
                    f"gaming_{strategy}", [row.variant_id for row in outcome.rows]
                )


if __name__ == "__main__":
    unittest.main()
