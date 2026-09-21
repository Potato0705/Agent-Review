"""Tests for assembling a reviewable variant set.

The output has one job beyond being correct: it must be directly usable as the
input to `score`. If the generator can emit a CSV that `load_scoring_cases`
rejects, the user finds out only after the next command fails, so that is
asserted here rather than left to the CLI.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_audit.io import load_scoring_cases, write_scoring_cases
from agent_audit.models import BaselineCase
from agent_audit.segmentation import CHINESE
from agent_audit.variants import (
    DEGRADATION_STRATEGIES,
    GAMING_STRATEGIES,
    generate_variants,
)


TEXT_A = (
    "学校应推迟上课时间。"
    "一项调查显示睡眠充足的学生成绩更稳定。"
    "因此可以先试行一个学期。"
)
TEXT_B = (
    "社区应增设夜间照明。"
    "去年的报修记录显示夜间事故集中在三条街道。"
    "但是预算需要分两年安排。"
)


def _cases() -> list[BaselineCase]:
    return [
        BaselineCase("c1", TEXT_A, (2,), "原始短文"),
        BaselineCase("c2", TEXT_B, (2,), ""),
    ]


class GenerationTests(unittest.TestCase):
    def test_produces_a_baseline_row_plus_one_row_per_strategy(self) -> None:
        run = generate_variants(_cases(), seed=0)

        self.assertEqual(len(run.rows), 2 * (1 + len(GAMING_STRATEGIES) + len(DEGRADATION_STRATEGIES)))
        first = [row for row in run.rows if row.case_id == "c1"]
        self.assertEqual(first[0].variant_type, "baseline")
        self.assertEqual(first[0].text, TEXT_A)

    def test_the_output_is_accepted_by_the_scoring_loader(self) -> None:
        run = generate_variants(_cases(), seed=0)
        path = Path(tempfile.mkdtemp()) / "cases.csv"

        write_scoring_cases(path, run.rows)
        loaded = load_scoring_cases(path)

        self.assertEqual(len(loaded), len(run.rows))
        self.assertEqual({case.case_id for case in loaded}, {"c1", "c2"})

    def test_notes_carry_the_strategy_and_the_source_note(self) -> None:
        run = generate_variants(_cases(), seed=0)

        notes = {(row.case_id, row.variant_id): row.notes for row in run.rows}
        self.assertEqual(notes[("c1", "baseline")], "generated=baseline | source_note=原始短文")
        self.assertIn("generated=verbose_padding", notes[("c1", "gaming_verbose_padding")])
        self.assertIn("source_note=原始短文", notes[("c1", "gaming_verbose_padding")])
        self.assertNotIn("source_note", notes[("c2", "baseline")])

    def test_paraphrase_is_off_unless_it_is_asked_for(self) -> None:
        without = generate_variants(_cases(), seed=0)
        with_paraphrase = generate_variants(
            _cases(), seed=0, paraphrase=("connective_substitution",)
        )

        self.assertNotIn(
            "paraphrase", {row.variant_type for row in without.rows}
        )
        self.assertIn(
            "paraphrase", {row.variant_type for row in with_paraphrase.rows}
        )

    def test_a_paraphrase_row_is_labelled_a_weak_probe(self) -> None:
        run = generate_variants(
            _cases(), seed=0, paraphrase=("connective_substitution",)
        )

        row = next(row for row in run.rows if row.variant_type == "paraphrase")
        self.assertIn("weak_probe", row.notes)

    def test_the_same_seed_produces_identical_rows(self) -> None:
        first = generate_variants(_cases(), seed=11)
        second = generate_variants(_cases(), seed=11)

        self.assertEqual(
            [(row.variant_id, row.text) for row in first.rows],
            [(row.variant_id, row.text) for row in second.rows],
        )


class ManifestTests(unittest.TestCase):
    def test_the_manifest_records_the_run_and_flags_human_review(self) -> None:
        run = generate_variants(_cases(), seed=3)

        self.assertTrue(run.manifest["requires_human_review"])
        self.assertEqual(run.manifest["seed"], 3)
        self.assertEqual(run.manifest["case_count"], 2)
        self.assertEqual(run.manifest["row_count"], len(run.rows))
        self.assertEqual(run.manifest["language"], "chinese")
        self.assertEqual(
            run.manifest["gaming_strategies"], list(GAMING_STRATEGIES)
        )

    def test_the_manifest_records_intervention_magnitude_per_variant(self) -> None:
        run = generate_variants(_cases(), seed=0)

        entry = next(
            item
            for item in run.manifest["variants"]
            if item["variant_id"] == "degradation_remove_evidence"
            and item["case_id"] == "c1"
        )
        self.assertEqual(entry["strategy"], "remove_evidence")
        self.assertGreater(entry["magnitude"]["removed_characters"], 0)
        self.assertGreater(entry["magnitude"]["removed_share"], 0.0)

    def test_the_manifest_records_the_corpus_sentences_actually_used(self) -> None:
        """A reviewer must be able to confirm the padding really is irrelevant."""

        run = generate_variants(_cases(), seed=0)

        entry = next(
            item
            for item in run.manifest["variants"]
            if item["variant_id"] == "gaming_verbose_padding"
            and item["case_id"] == "c1"
        )
        self.assertEqual(len(entry["inserted_text"]), 2)
        for sentence in entry["inserted_text"]:
            self.assertIn(sentence, CHINESE.padding_sentences)

    def test_the_manifest_is_json_serialisable(self) -> None:
        run = generate_variants(_cases(), seed=0)

        restored = json.loads(json.dumps(run.manifest, ensure_ascii=False))

        self.assertEqual(restored["row_count"], run.manifest["row_count"])


class GenerationRefusalTests(unittest.TestCase):
    def test_refuses_an_empty_case_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one baseline case"):
            generate_variants([], seed=0)

    def test_refuses_an_empty_gaming_strategy_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one gaming strategy"):
            generate_variants(_cases(), seed=0, gaming=())

    def test_refuses_an_empty_degradation_strategy_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one degradation strategy"):
            generate_variants(_cases(), seed=0, degradation=())

    def test_refuses_a_strategy_from_the_wrong_family(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a gaming strategy"):
            generate_variants(_cases(), seed=0, gaming=("remove_evidence",))

    def test_refuses_an_unknown_strategy_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a gaming strategy"):
            generate_variants(_cases(), seed=0, gaming=("keyword_stuffing",))

    def test_refuses_an_unknown_paraphrase_strategy(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a paraphrase strategy"):
            generate_variants(_cases(), seed=0, paraphrase=("back_translation",))

    def test_refuses_a_duplicate_case_id(self) -> None:
        duplicated = [*_cases(), BaselineCase("c1", TEXT_B, (2,))]

        with self.assertRaisesRegex(ValueError, "duplicate case_id"):
            generate_variants(duplicated, seed=0)

    def test_names_the_case_when_a_strategy_refuses(self) -> None:
        """A refusal must say which baseline caused it."""

        cases = [BaselineCase("no_connective", "甲乙丙。丁戊己。", (2,))]

        with self.assertRaisesRegex(ValueError, "no_connective"):
            generate_variants(
                cases, seed=0, paraphrase=("connective_substitution",)
            )


if __name__ == "__main__":
    unittest.main()
