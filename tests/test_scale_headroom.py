"""Tests for detecting a ceiling or floor that makes a metric unreadable.

A baseline scored at the top of the scale leaves a gaming variant nowhere to
go. A zero gaming gain then means nothing: it cannot be told apart from a
grader that genuinely resists gaming. The same holds at the bottom for
degradation.

This was found the hard way. `gemma4:e4b` scored the trajectory baselines
9.97 out of 10, and "gaming does not work on it" had to be walked back by
hand in the case study prose. The tool should say it, not the author.
"""

from __future__ import annotations

import unittest

from agent_audit.audit import AuditConfig, audit_records
from agent_audit.models import ScoreRecord


def _case(case_id: str, baseline: float, gaming: float, degradation: float):
    return [
        ScoreRecord("Grader", case_id, "base", "baseline", baseline),
        ScoreRecord("Grader", case_id, "game", "gaming", gaming),
        ScoreRecord("Grader", case_id, "drop", "degradation", degradation),
    ]


def _audit(records, **overrides):
    config = AuditConfig(score_min=0.0, score_max=10.0, **overrides)
    return audit_records(records, config)


class CeilingTests(unittest.TestCase):
    def test_a_baseline_at_the_top_is_reported_as_ceiling_limited(self) -> None:
        result = _audit(_case("c1", 10.0, 10.0, 8.0))

        self.assertEqual(result.ceiling_limited_count, 1)

    def test_a_baseline_with_room_above_is_not_flagged(self) -> None:
        result = _audit(_case("c1", 6.0, 6.5, 4.0))

        self.assertEqual(result.ceiling_limited_count, 0)

    def test_the_boundary_is_the_minimum_degradation_drop(self) -> None:
        """Headroom is judged against the effect size the audit already uses.

        Inventing a second threshold would mean two numbers that must be kept
        consistent, and nobody would notice when they drifted apart.
        """

        exactly_enough = _audit(_case("c1", 9.0, 9.0, 7.0), min_degradation_drop=1.0)
        just_short = _audit(_case("c1", 9.5, 9.5, 7.0), min_degradation_drop=1.0)

        self.assertEqual(exactly_enough.ceiling_limited_count, 0)
        self.assertEqual(just_short.ceiling_limited_count, 1)

    def test_every_case_can_be_ceiling_limited(self) -> None:
        records = _case("c1", 10.0, 10.0, 8.0) + _case("c2", 9.9, 9.9, 7.0)

        self.assertEqual(_audit(records).ceiling_limited_count, 2)


class FloorTests(unittest.TestCase):
    def test_a_baseline_at_the_bottom_is_reported_as_floor_limited(self) -> None:
        """Nothing can fall further, so a small degradation drop proves nothing."""

        result = _audit(_case("c1", 0.0, 1.0, 0.0))

        self.assertEqual(result.floor_limited_count, 1)

    def test_a_baseline_with_room_below_is_not_flagged(self) -> None:
        result = _audit(_case("c1", 6.0, 6.5, 4.0))

        self.assertEqual(result.floor_limited_count, 0)


class UndeclaredScaleTests(unittest.TestCase):
    def test_without_a_declared_scale_neither_is_computed(self) -> None:
        """Headroom is meaningless when nobody said where the scale ends."""

        result = audit_records(_case("c1", 10.0, 10.0, 8.0), AuditConfig())

        self.assertIsNone(result.ceiling_limited_count)
        self.assertIsNone(result.floor_limited_count)

    def test_half_a_scale_is_refused_outright(self) -> None:
        """One end without the other was already a hard refusal; keep it that way."""

        with self.assertRaisesRegex(ValueError, "provided together"):
            audit_records(_case("c1", 10.0, 10.0, 8.0), AuditConfig(score_max=10.0))


class ReportTests(unittest.TestCase):
    def _rendered(self, records, **overrides) -> str:
        from agent_audit.report import render_markdown_report

        return render_markdown_report(_audit(records, **overrides))

    def test_a_ceiling_is_stated_in_the_report(self) -> None:
        text = self._rendered(_case("c1", 10.0, 10.0, 8.0))

        self.assertIn("作弊收益", text)
        self.assertIn("量表上限", text)

    def test_a_clean_scale_adds_no_warning(self) -> None:
        text = self._rendered(_case("c1", 6.0, 6.5, 4.0))

        self.assertNotIn("量表上限", text)
        self.assertNotIn("量表下限", text)

    def test_a_floor_is_stated_in_the_report(self) -> None:
        text = self._rendered(_case("c1", 0.0, 1.0, 0.0))

        self.assertIn("量表下限", text)
        self.assertIn("退化降分", text)

    def test_the_html_report_states_a_floor_too(self) -> None:
        from agent_audit.html_report import render_audit_html

        html = render_audit_html(_audit(_case("c1", 0.0, 1.0, 0.0)))

        self.assertIn("量表下限", html)

    def test_the_html_report_states_it_too(self) -> None:
        from agent_audit.html_report import render_audit_html

        html = render_audit_html(_audit(_case("c1", 10.0, 10.0, 8.0)))

        self.assertIn("量表上限", html)

    def test_the_json_carries_the_counts(self) -> None:
        payload = _audit(_case("c1", 10.0, 10.0, 8.0)).to_dict()

        self.assertEqual(payload["ceiling_limited_count"], 1)
        self.assertEqual(payload["floor_limited_count"], 0)


if __name__ == "__main__":
    unittest.main()
