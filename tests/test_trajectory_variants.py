"""Tests for trajectory variant generation and its postconditions.

Every postcondition here is exactly decidable, which is what makes this
modality worth the work. In prose, a gaming variant can only be checked
loosely ("still contains the baseline as a prefix") and an equivalent rewrite
cannot be checked at all. A trajectory is structured: "these are the baseline's
steps minus step 2, in order, with the same final answer" is either true or it
is not.

The degradation variants deliberately keep the final answer. That is the whole
probe: if the grader scores a run the same after the step that actually found
the answer is gone, it was only reading the last paragraph.
"""

from __future__ import annotations

import random
import unittest
from dataclasses import replace

from agent_audit import trajectory_variants as tv
from agent_audit.segmentation import CHINESE
from agent_audit.trajectory import TrajectoryCase, TrajectoryStep
from agent_audit.trajectory_variants import (
    DEGRADATION_STRATEGIES,
    GAMING_STRATEGIES,
    PARAPHRASE_STRATEGIES,
    TrajectoryPostconditionError,
    build_trajectory_variant,
    generate_trajectory_variants,
)


def _case(**overrides: object) -> TrajectoryCase:
    defaults: dict[str, object] = {
        "case_id": "flight_refund",
        "task": "查出订单 A7731 是否可全额退款。",
        "steps": (
            TrajectoryStep(1, "search_orders", '{"order_id": "A7731"}', "舱位 Y"),
            TrajectoryStep(2, "read_policy", '{"fare_class": "Y"}', "24 小时内可全退"),
            TrajectoryStep(3, "get_time", "{}", "2026-03-02T18:40Z"),
        ),
        "final_answer": "可以全额退。",
        "load_bearing_steps": (2,),
        "independent_steps": ((1, 3),),
    }
    defaults.update(overrides)
    return TrajectoryCase(**defaults)  # type: ignore[arg-type]


def _build(strategy: str, case: TrajectoryCase | None = None):
    return build_trajectory_variant(
        case or _case(), strategy, CHINESE, random.Random(0)
    )


def _identities(case: TrajectoryCase) -> list[tuple[str, str, str]]:
    return [step.identity for step in case.steps]


class StrategyCatalogueTests(unittest.TestCase):
    def test_every_family_offers_at_least_one_strategy(self) -> None:
        self.assertTrue(GAMING_STRATEGIES)
        self.assertTrue(DEGRADATION_STRATEGIES)
        self.assertTrue(PARAPHRASE_STRATEGIES)

    def test_strategy_names_do_not_collide_across_families(self) -> None:
        names = GAMING_STRATEGIES + DEGRADATION_STRATEGIES + PARAPHRASE_STRATEGIES

        self.assertEqual(len(names), len(set(names)))

    def test_an_unknown_strategy_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            _build("teleport")


class RedundantToolCallTests(unittest.TestCase):
    STRATEGY = "redundant_tool_calls"

    def test_the_variant_is_longer_than_the_baseline(self) -> None:
        variant = _build(self.STRATEGY)

        self.assertGreater(len(variant.case.steps), len(_case().steps))

    def test_the_final_answer_is_untouched(self) -> None:
        """Gaming means more apparent work for the same result."""

        self.assertEqual(_build(self.STRATEGY).case.final_answer, "可以全额退。")

    def test_every_added_call_already_existed_in_the_baseline(self) -> None:
        """Inventing a tool or a result would be fabricating evidence."""

        baseline = set(_identities(_case()))

        for identity in _identities(_build(self.STRATEGY).case):
            with self.subTest(identity=identity):
                self.assertIn(identity, baseline)

    def test_removing_the_duplicates_restores_the_baseline(self) -> None:
        variant = _build(self.STRATEGY)
        seen: list[tuple[str, str, str]] = []
        for identity in _identities(variant.case):
            if not seen or seen[-1] != identity:
                seen.append(identity)

        self.assertEqual(seen, _identities(_case()))

    def test_the_magnitude_records_what_was_repeated(self) -> None:
        variant = _build(self.STRATEGY)

        self.assertIn("repeated_steps", variant.magnitude)
        self.assertTrue(variant.magnitude["repeated_steps"])


class PaddedReasoningTests(unittest.TestCase):
    STRATEGY = "padded_reasoning"

    def test_what_the_agent_did_is_unchanged(self) -> None:
        variant = _build(self.STRATEGY)

        self.assertEqual(_identities(variant.case), _identities(_case()))
        self.assertEqual(variant.case.final_answer, "可以全额退。")

    def test_the_narration_grew(self) -> None:
        variant = _build(self.STRATEGY)
        before = sum(len(step.note) for step in _case().steps)
        after = sum(len(step.note) for step in variant.case.steps)

        self.assertGreater(after, before)

    def test_the_padding_is_recorded_for_review(self) -> None:
        """A reviewer has to confirm the padding really is irrelevant."""

        self.assertTrue(_build(self.STRATEGY).inserted_text)


class RemoveLoadBearingStepTests(unittest.TestCase):
    STRATEGY = "remove_load_bearing_step"

    def test_the_annotated_step_is_gone(self) -> None:
        variant = _build(self.STRATEGY)

        self.assertNotIn(
            _case().steps[1].identity, _identities(variant.case)
        )

    def test_the_other_steps_survive_in_order(self) -> None:
        variant = _build(self.STRATEGY)

        self.assertEqual(
            _identities(variant.case),
            [_case().steps[0].identity, _case().steps[2].identity],
        )

    def test_the_final_answer_is_kept_on_purpose(self) -> None:
        """Right answer, broken process: the sharpest probe in this modality."""

        self.assertEqual(_build(self.STRATEGY).case.final_answer, "可以全额退。")

    def test_the_magnitude_names_the_removed_step(self) -> None:
        self.assertEqual(_build(self.STRATEGY).magnitude["removed_steps"], [2])


class HollowEvidenceTests(unittest.TestCase):
    STRATEGY = "hollow_evidence"

    def test_the_call_remains_but_returns_nothing(self) -> None:
        variant = _build(self.STRATEGY)
        hollowed = variant.case.steps[1]

        self.assertEqual(hollowed.tool, "read_policy")
        self.assertEqual(hollowed.args, '{"fare_class": "Y"}')
        self.assertNotIn("24 小时内可全退", hollowed.result)

    def test_untouched_steps_keep_their_results(self) -> None:
        variant = _build(self.STRATEGY)

        self.assertEqual(variant.case.steps[0].result, "舱位 Y")
        self.assertEqual(variant.case.steps[2].result, "2026-03-02T18:40Z")

    def test_the_step_count_and_answer_are_unchanged(self) -> None:
        variant = _build(self.STRATEGY)

        self.assertEqual(len(variant.case.steps), 3)
        self.assertEqual(variant.case.final_answer, "可以全额退。")

    def test_an_already_empty_result_is_refused(self) -> None:
        """Emptying an empty result degrades nothing and would still 'pass'.

        In Python the empty string is a substring of everything, so a naive
        "the old result is gone" check is vacuously true here.
        """

        case = _case(
            steps=(
                TrajectoryStep(1, "a", "{}", "r1"),
                TrajectoryStep(2, "b", "{}", ""),
                TrajectoryStep(3, "c", "{}", "r3"),
            )
        )

        with self.assertRaisesRegex(
            TrajectoryPostconditionError, "already returns no result"
        ):
            _build(self.STRATEGY, case)


class ReorderIndependentStepsTests(unittest.TestCase):
    STRATEGY = "reorder_independent_steps"

    def test_the_same_steps_are_taken_in_a_different_order(self) -> None:
        variant = _build(self.STRATEGY)

        self.assertEqual(
            sorted(_identities(variant.case)), sorted(_identities(_case()))
        )
        self.assertNotEqual(_identities(variant.case), _identities(_case()))

    def test_the_final_answer_is_unchanged(self) -> None:
        self.assertEqual(_build(self.STRATEGY).case.final_answer, "可以全额退。")

    def test_a_case_without_an_independent_group_is_refused(self) -> None:
        """Better to refuse than to invent an equivalence nobody verified."""

        with self.assertRaisesRegex(
            TrajectoryPostconditionError, "declares no independent_steps"
        ):
            _build(self.STRATEGY, _case(independent_steps=()))

    def test_a_group_whose_steps_are_identical_is_refused(self) -> None:
        """Rotating identical steps produces the baseline back, not a variant."""

        case = _case(
            steps=(
                TrajectoryStep(1, "a", "{}", "r"),
                TrajectoryStep(2, "b", "{}", "r2"),
                TrajectoryStep(3, "a", "{}", "r"),
            ),
            load_bearing_steps=(2,),
            independent_steps=((1, 3),),
        )

        with self.assertRaisesRegex(
            TrajectoryPostconditionError, "produced the baseline order again"
        ):
            _build(self.STRATEGY, case)


class PostconditionNetTests(unittest.TestCase):
    """Fire each assertion directly.

    These bodies never run while the builders are correct, which is exactly
    why they need their own tests: an assertion that has never rejected
    anything is an assertion nobody knows works. Each case below is the defect
    that assertion exists to catch.
    """

    def _steps(self, *triples: tuple[str, str, str]) -> tuple[TrajectoryStep, ...]:
        return tuple(
            TrajectoryStep(position, tool, args, result)
            for position, (tool, args, result) in enumerate(triples, start=1)
        )

    def setUp(self) -> None:
        self.baseline = _case(
            steps=self._steps(("a", "{}", "r1"), ("b", "{}", "r2"), ("c", "{}", "r3")),
            load_bearing_steps=(2,),
            independent_steps=((1, 3),),
        )

    def _variant(self, **overrides: object) -> TrajectoryCase:
        return replace(self.baseline, **overrides)  # type: ignore[arg-type]

    def test_repeated_calls_that_shrink_the_run_are_caught(self) -> None:
        with self.assertRaisesRegex(TrajectoryPostconditionError, "add steps"):
            tv._assert_only_repeated_calls(
                self.baseline, self._variant(steps=self.baseline.steps[:1])
            )

    def test_repeated_calls_that_change_the_answer_are_caught(self) -> None:
        longer = self.baseline.steps + (self.baseline.steps[0],)

        with self.assertRaisesRegex(TrajectoryPostconditionError, "final answer"):
            tv._assert_only_repeated_calls(
                self.baseline, self._variant(steps=longer, final_answer="别的答案")
            )

    def test_an_invented_call_is_caught(self) -> None:
        invented = self.baseline.steps + (TrajectoryStep(4, "z", "{}", "made up"),)

        with self.assertRaisesRegex(TrajectoryPostconditionError, "invented"):
            tv._assert_only_repeated_calls(self.baseline, self._variant(steps=invented))

    def test_a_reordered_sequence_is_not_a_repeat(self) -> None:
        shuffled = (
            self.baseline.steps[1],
            self.baseline.steps[0],
            self.baseline.steps[2],
            self.baseline.steps[2],
        )

        with self.assertRaisesRegex(TrajectoryPostconditionError, "sequence"):
            tv._assert_only_repeated_calls(self.baseline, self._variant(steps=shuffled))

    def test_padding_that_changed_the_work_is_caught(self) -> None:
        with self.assertRaisesRegex(TrajectoryPostconditionError, "actually did"):
            tv._assert_only_narration_changed(
                self.baseline, self._variant(steps=self.baseline.steps[:2])
            )

    def test_padding_that_changed_the_answer_is_caught(self) -> None:
        with self.assertRaisesRegex(TrajectoryPostconditionError, "final answer"):
            tv._assert_only_narration_changed(
                self.baseline, self._variant(final_answer="别的答案")
            )

    def test_padding_that_added_nothing_is_caught(self) -> None:
        with self.assertRaisesRegex(TrajectoryPostconditionError, "no narration"):
            tv._assert_only_narration_changed(self.baseline, self.baseline)

    def test_a_language_without_padding_text_is_refused(self) -> None:
        silent = replace(CHINESE, padding_sentences=())

        with self.assertRaisesRegex(TrajectoryPostconditionError, "padding"):
            build_trajectory_variant(
                self.baseline, "padded_reasoning", silent, random.Random(0)
            )

    def test_removing_every_step_is_refused(self) -> None:
        """The loader blocks this annotation; the builder must not rely on it."""

        everything = replace(self.baseline, load_bearing_steps=(1, 2, 3))

        with self.assertRaisesRegex(TrajectoryPostconditionError, "no trajectory"):
            build_trajectory_variant(
                everything, "remove_load_bearing_step", CHINESE, random.Random(0)
            )

    def test_a_removal_that_lost_the_order_is_caught(self) -> None:
        scrambled = (self.baseline.steps[2], self.baseline.steps[0])

        with self.assertRaisesRegex(TrajectoryPostconditionError, "original order"):
            tv._assert_steps_removed(self.baseline, self._variant(steps=scrambled), [2])

    def test_a_removal_that_also_changed_the_answer_is_caught(self) -> None:
        kept = (self.baseline.steps[0], self.baseline.steps[2])

        with self.assertRaisesRegex(TrajectoryPostconditionError, "keep the final"):
            tv._assert_steps_removed(
                self.baseline,
                self._variant(steps=kept, final_answer="别的答案"),
                [2],
            )

    def test_hollowing_that_dropped_a_call_is_caught(self) -> None:
        with self.assertRaisesRegex(TrajectoryPostconditionError, "every call"):
            tv._assert_evidence_hollowed(
                self.baseline, self._variant(steps=self.baseline.steps[:2]), {2}
            )

    def test_hollowing_that_changed_the_answer_is_caught(self) -> None:
        with self.assertRaisesRegex(TrajectoryPostconditionError, "final answer"):
            tv._assert_evidence_hollowed(
                self.baseline, self._variant(final_answer="别的答案"), {2}
            )

    def test_hollowing_that_changed_the_call_is_caught(self) -> None:
        swapped = (
            self.baseline.steps[0],
            replace(self.baseline.steps[1], tool="other"),
            self.baseline.steps[2],
        )

        with self.assertRaisesRegex(TrajectoryPostconditionError, "which call"):
            tv._assert_evidence_hollowed(self.baseline, self._variant(steps=swapped), {2})

    def test_hollowing_that_kept_the_result_is_caught(self) -> None:
        untouched = self.baseline.steps

        with self.assertRaisesRegex(TrajectoryPostconditionError, "left the original"):
            tv._assert_evidence_hollowed(
                self.baseline, self._variant(steps=untouched), {2}
            )

    def test_hollowing_an_unannotated_step_is_caught(self) -> None:
        spilled = (
            replace(self.baseline.steps[0], result="（无结果）"),
            replace(self.baseline.steps[1], result="（无结果）"),
            self.baseline.steps[2],
        )

        with self.assertRaisesRegex(TrajectoryPostconditionError, "not asked to touch"):
            tv._assert_evidence_hollowed(self.baseline, self._variant(steps=spilled), {2})

    def test_a_reordering_that_changed_the_calls_is_caught(self) -> None:
        with self.assertRaisesRegex(TrajectoryPostconditionError, "which calls"):
            tv._assert_same_work_reordered(
                self.baseline, self._variant(steps=self.baseline.steps[:2])
            )

    def test_a_reordering_that_changed_the_answer_is_caught(self) -> None:
        rotated = (
            self.baseline.steps[2],
            self.baseline.steps[1],
            self.baseline.steps[0],
        )

        with self.assertRaisesRegex(TrajectoryPostconditionError, "final answer"):
            tv._assert_same_work_reordered(
                self.baseline, self._variant(steps=rotated, final_answer="别的答案")
            )


class GenerationRunTests(unittest.TestCase):
    def _run(self, **kwargs: object):
        defaults: dict[str, object] = {
            "gaming": GAMING_STRATEGIES,
            "degradation": DEGRADATION_STRATEGIES,
            "paraphrase": (),
            "language": CHINESE,
            "seed": 0,
        }
        defaults.update(kwargs)
        return generate_trajectory_variants([_case()], **defaults)  # type: ignore[arg-type]

    def test_a_baseline_row_is_emitted_alongside_the_variants(self) -> None:
        rows = self._run().rows

        self.assertEqual(
            [row.variant_type for row in rows][0], "baseline"
        )
        self.assertEqual(len(rows), 1 + 2 + 2)

    def test_rows_carry_the_rendered_transcript(self) -> None:
        rows = self._run().rows

        self.assertIn("步骤 1", rows[0].text)
        self.assertIn("最终答案：可以全额退。", rows[0].text)

    def test_variant_ids_are_prefixed_by_family(self) -> None:
        ids = {row.variant_id for row in self._run().rows}

        self.assertIn("gaming_redundant_tool_calls", ids)
        self.assertIn("degradation_remove_load_bearing_step", ids)

    def test_the_manifest_declares_the_modality(self) -> None:
        self.assertEqual(self._run().manifest["modality"], "trajectory")

    def test_the_manifest_keeps_the_identity_fields_the_audit_reads(self) -> None:
        """The audit computes a fingerprint from a fixed set of keys."""

        manifest = self._run().manifest

        for key in (
            "generator",
            "language",
            "seed",
            "gaming_strategies",
            "degradation_strategies",
            "paraphrase_strategies",
        ):
            with self.subTest(key=key):
                self.assertIn(key, manifest)

    def test_the_same_seed_reproduces_the_same_rows(self) -> None:
        first = [row.text for row in self._run().rows]
        second = [row.text for row in self._run().rows]

        self.assertEqual(first, second)

    def test_an_empty_family_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            self._run(gaming=())

    def test_an_unknown_strategy_names_the_family(self) -> None:
        with self.assertRaisesRegex(ValueError, "gaming"):
            self._run(gaming=("remove_load_bearing_step",))

    def test_an_unknown_paraphrase_strategy_is_refused(self) -> None:
        """Paraphrase may be empty, so it needs its own name check."""

        with self.assertRaisesRegex(ValueError, "paraphrase strategy"):
            self._run(paraphrase=("hollow_evidence",))

    def test_a_refusing_strategy_names_the_case(self) -> None:
        with self.assertRaisesRegex(TrajectoryPostconditionError, "flight_refund"):
            generate_trajectory_variants(
                [_case(independent_steps=())],
                gaming=GAMING_STRATEGIES,
                degradation=DEGRADATION_STRATEGIES,
                paraphrase=PARAPHRASE_STRATEGIES,
                language=CHINESE,
                seed=0,
            )

    def test_no_cases_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            generate_trajectory_variants(
                [],
                gaming=GAMING_STRATEGIES,
                degradation=DEGRADATION_STRATEGIES,
                paraphrase=(),
                language=CHINESE,
                seed=0,
            )


if __name__ == "__main__":
    unittest.main()
