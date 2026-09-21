"""Tests for loading and rendering agent trajectories.

Two annotations carry the whole feature: which steps the answer rests on, and
which steps may be reordered without changing what happened. The tool must
never infer either one — a wrong guess produces a variant that claims to be a
degradation without degrading anything, or an equivalent rewrite that silently
changed the causal chain. So most of what is tested here is refusal.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_audit.segmentation import CHINESE, ENGLISH
from agent_audit.trajectory import (
    TrajectoryCase,
    TrajectoryStep,
    load_trajectory_cases,
    render_trajectory,
)


def _case_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "case_id": "flight_refund",
        "task": "查出订单 A7731 是否可全额退款，并说明依据。",
        "steps": [
            {
                "tool": "search_orders",
                "args": {"order_id": "A7731"},
                "result": "舱位 Y，出票日 2026-03-02",
            },
            {
                "tool": "read_policy",
                "args": {"fare_class": "Y"},
                "result": "Y 舱出票 24 小时内可全额退款",
            },
            {"tool": "get_time", "args": {}, "result": "2026-03-02T18:40Z"},
        ],
        "final_answer": "可以全额退。订单为 Y 舱，出票于 03-02，仍在 24 小时内。",
        "load_bearing_steps": [2],
        "independent_steps": [[1, 3]],
        "notes": "原始轨迹",
    }
    payload.update(overrides)
    return payload


def _write(payloads: list[dict[str, object]]) -> Path:
    path = Path(tempfile.mkdtemp()) / "trajectories.jsonl"
    path.write_text(
        "\n".join(json.dumps(payload, ensure_ascii=False) for payload in payloads)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


class LoadingTests(unittest.TestCase):
    def test_a_well_formed_file_round_trips(self) -> None:
        cases = load_trajectory_cases(_write([_case_payload()]))

        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case.case_id, "flight_refund")
        self.assertEqual(len(case.steps), 3)
        self.assertEqual(case.steps[0].tool, "search_orders")
        self.assertEqual(case.load_bearing_steps, (2,))
        self.assertEqual(case.independent_steps, ((1, 3),))

    def test_step_indices_are_one_based_and_assigned_by_position(self) -> None:
        cases = load_trajectory_cases(_write([_case_payload()]))

        self.assertEqual([step.index for step in cases[0].steps], [1, 2, 3])

    def test_arguments_are_serialised_with_sorted_keys(self) -> None:
        """Byte-identical output for the same input needs a stable key order."""

        payload = _case_payload()
        payload["steps"] = [  # type: ignore[index]
            {"tool": "t", "args": {"b": 2, "a": 1}, "result": "r"},
            {"tool": "u", "args": {}, "result": "r"},
        ]
        payload["load_bearing_steps"] = [1]
        payload["independent_steps"] = []

        cases = load_trajectory_cases(_write([payload]))

        self.assertEqual(cases[0].steps[0].args, '{"a": 1, "b": 2}')

    def test_blank_lines_are_skipped(self) -> None:
        path = _write([_case_payload()])
        path.write_text(
            path.read_text(encoding="utf-8") + "\n   \n", encoding="utf-8", newline="\n"
        )

        self.assertEqual(len(load_trajectory_cases(path)), 1)

    def test_a_missing_file_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            load_trajectory_cases(Path(tempfile.mkdtemp()) / "absent.jsonl")

    def test_an_empty_file_is_refused(self) -> None:
        path = Path(tempfile.mkdtemp()) / "empty.jsonl"
        path.write_text("", encoding="utf-8", newline="\n")

        with self.assertRaisesRegex(ValueError, "no trajectories"):
            load_trajectory_cases(path)

    def test_a_malformed_line_names_its_line_number(self) -> None:
        path = _write([_case_payload(), _case_payload(case_id="second")])
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[1] = "{not json"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

        with self.assertRaisesRegex(ValueError, "Line 2"):
            load_trajectory_cases(path)

    def test_a_duplicate_case_id_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "flight_refund"):
            load_trajectory_cases(_write([_case_payload(), _case_payload()]))

    def test_a_case_without_steps_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one step"):
            load_trajectory_cases(_write([_case_payload(steps=[])]))

    def test_an_empty_required_field_is_refused(self) -> None:
        for field in ("case_id", "task", "final_answer"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    load_trajectory_cases(_write([_case_payload(**{field: "  "})]))

    def test_a_step_without_a_tool_name_is_refused(self) -> None:
        payload = _case_payload()
        payload["steps"][0]["tool"] = ""  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "tool"):
            load_trajectory_cases(_write([payload]))


class MalformedInputTests(unittest.TestCase):
    """Anything the loader cannot read exactly, it refuses by name."""

    def _assert_refused(self, pattern: str, payload: object) -> None:
        path = Path(tempfile.mkdtemp()) / "trajectories.jsonl"
        path.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        with self.assertRaisesRegex(ValueError, pattern):
            load_trajectory_cases(path)

    def test_a_line_that_is_not_an_object_is_refused(self) -> None:
        self._assert_refused("must be an object", ["not", "a", "case"])

    def test_non_integer_annotation_indices_are_refused(self) -> None:
        self._assert_refused(
            "list of integers", _case_payload(load_bearing_steps=["2"])
        )

    def test_a_boolean_annotation_index_is_refused(self) -> None:
        """`True` is an int in Python and would silently mean step 1."""

        self._assert_refused(
            "list of integers", _case_payload(load_bearing_steps=[True])
        )

    def test_independent_steps_that_are_not_a_list_are_refused(self) -> None:
        self._assert_refused(
            "list of lists", _case_payload(independent_steps={"a": 1})
        )

    def test_steps_that_are_not_a_list_are_refused(self) -> None:
        self._assert_refused("steps must be a list", _case_payload(steps="1,2,3"))

    def test_a_step_that_is_not_an_object_is_refused(self) -> None:
        self._assert_refused("must be an object", _case_payload(steps=["search"]))

    def test_a_non_string_result_is_refused(self) -> None:
        payload = _case_payload()
        payload["steps"][0]["result"] = {"rows": 3}  # type: ignore[index]

        self._assert_refused("result must be a string", payload)

    def test_a_non_string_note_is_refused(self) -> None:
        payload = _case_payload()
        payload["steps"][0]["note"] = 7  # type: ignore[index]

        self._assert_refused("note must be a string", payload)

    def test_arguments_already_given_as_a_string_are_kept(self) -> None:
        """Some exports store the call arguments pre-serialised."""

        payload = _case_payload()
        payload["steps"][0]["args"] = '{"order_id": "A7731"}'  # type: ignore[index]
        path = Path(tempfile.mkdtemp()) / "t.jsonl"
        path.write_text(
            json.dumps(payload, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        cases = load_trajectory_cases(path)

        self.assertEqual(cases[0].steps[0].args, '{"order_id": "A7731"}')

    def test_arguments_that_cannot_be_serialised_are_refused(self) -> None:
        """JSON cannot hold every value a converter might hand us."""

        from agent_audit.trajectory import _serialise_args

        with self.assertRaisesRegex(ValueError, "not serialisable"):
            _serialise_args({"when": {1, 2}}, position=1, line=1)


class AnnotationRefusalTests(unittest.TestCase):
    """The tool never guesses which steps matter."""

    def _assert_refused(self, pattern: str, **overrides: object) -> None:
        with self.assertRaisesRegex(ValueError, pattern):
            load_trajectory_cases(_write([_case_payload(**overrides)]))

    def test_no_load_bearing_annotation_is_refused(self) -> None:
        self._assert_refused("load_bearing_steps", load_bearing_steps=[])

    def test_an_out_of_range_load_bearing_index_is_refused(self) -> None:
        self._assert_refused("out of range", load_bearing_steps=[4])

    def test_a_zero_or_negative_index_is_refused(self) -> None:
        self._assert_refused("out of range", load_bearing_steps=[0])

    def test_annotating_every_step_is_refused(self) -> None:
        """Removing all of them leaves no trajectory to degrade."""

        self._assert_refused("every step", load_bearing_steps=[1, 2, 3])

    def test_a_repeated_load_bearing_index_is_refused(self) -> None:
        self._assert_refused("repeats", load_bearing_steps=[2, 2])

    def test_an_independent_group_of_one_is_refused(self) -> None:
        self._assert_refused("at least two", independent_steps=[[1]])

    def test_an_out_of_range_independent_index_is_refused(self) -> None:
        self._assert_refused("out of range", independent_steps=[[1, 9]])

    def test_a_step_in_two_independent_groups_is_refused(self) -> None:
        """Overlapping groups make the reordering ambiguous."""

        self._assert_refused("more than one group", independent_steps=[[1, 2], [2, 3]])

    def test_a_repeated_index_inside_one_group_is_refused(self) -> None:
        self._assert_refused("repeats", independent_steps=[[1, 1]])

    def test_a_load_bearing_step_may_still_be_reorderable(self) -> None:
        """Order and causality are different questions; both are annotated."""

        cases = load_trajectory_cases(
            _write([_case_payload(load_bearing_steps=[2], independent_steps=[[2, 3]])])
        )

        self.assertEqual(cases[0].independent_steps, ((2, 3),))

    def test_no_independent_group_is_allowed(self) -> None:
        """Only paraphrase needs it, and paraphrase is off by default."""

        cases = load_trajectory_cases(_write([_case_payload(independent_steps=[])]))

        self.assertEqual(cases[0].independent_steps, ())


class RenderingTests(unittest.TestCase):
    def _case(self, **overrides: object) -> TrajectoryCase:
        return load_trajectory_cases(_write([_case_payload(**overrides)]))[0]

    def test_the_transcript_carries_task_steps_and_answer(self) -> None:
        rendered = render_trajectory(self._case(), CHINESE)

        self.assertIn("任务：查出订单 A7731", rendered)
        self.assertIn("步骤 1", rendered)
        self.assertIn("工具：search_orders", rendered)
        self.assertIn('参数：{"order_id": "A7731"}', rendered)
        self.assertIn("最终答案：可以全额退。", rendered)

    def test_steps_are_numbered_in_order(self) -> None:
        rendered = render_trajectory(self._case(), CHINESE)

        self.assertLess(rendered.index("步骤 1"), rendered.index("步骤 2"))
        self.assertLess(rendered.index("步骤 2"), rendered.index("步骤 3"))

    def test_an_empty_note_leaves_no_empty_label(self) -> None:
        """A bare '说明：' tells the grader nothing and invites guessing."""

        self.assertNotIn("说明：", render_trajectory(self._case(), CHINESE))

    def test_a_present_note_is_rendered(self) -> None:
        payload = _case_payload()
        payload["steps"][0]["note"] = "先确认订单存在"  # type: ignore[index]
        case = load_trajectory_cases(_write([payload]))[0]

        self.assertIn("说明：先确认订单存在", render_trajectory(case, CHINESE))

    def test_english_uses_english_labels(self) -> None:
        rendered = render_trajectory(self._case(), ENGLISH)

        self.assertIn("Task:", rendered)
        self.assertIn("Step 1", rendered)
        self.assertIn("Final answer:", rendered)
        self.assertNotIn("任务", rendered)

    def test_a_colon_inside_the_content_survives(self) -> None:
        """Label separators must be written, not substituted after the fact."""

        payload = _case_payload(task="判断：订单能否退款", final_answer="结论：可以")
        case = load_trajectory_cases(_write([payload]))[0]

        rendered = render_trajectory(case, ENGLISH)

        self.assertIn("Task: 判断：订单能否退款", rendered)
        self.assertIn("Final answer: 结论：可以", rendered)

    def test_rendering_is_byte_stable(self) -> None:
        case = self._case()

        self.assertEqual(
            render_trajectory(case, CHINESE), render_trajectory(case, CHINESE)
        )

    def test_two_cases_differing_only_in_step_order_render_differently(self) -> None:
        """Order is the whole content of a reordering paraphrase."""

        original = self._case()
        swapped = TrajectoryCase(
            case_id=original.case_id,
            task=original.task,
            steps=(original.steps[2], original.steps[1], original.steps[0]),
            final_answer=original.final_answer,
            load_bearing_steps=original.load_bearing_steps,
            independent_steps=original.independent_steps,
        )

        self.assertNotEqual(
            render_trajectory(original, CHINESE), render_trajectory(swapped, CHINESE)
        )

    def test_a_renumbered_step_is_rendered_at_its_new_position(self) -> None:
        case = TrajectoryCase(
            case_id="c",
            task="t",
            steps=(
                TrajectoryStep(index=1, tool="a", args="{}", result="r1"),
                TrajectoryStep(index=2, tool="b", args="{}", result="r2"),
            ),
            final_answer="done",
            load_bearing_steps=(1,),
            independent_steps=(),
        )

        rendered = render_trajectory(case, CHINESE)

        self.assertLess(rendered.index("工具：a"), rendered.index("工具：b"))


if __name__ == "__main__":
    unittest.main()
