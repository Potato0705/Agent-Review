from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from agent_audit.audit import (
    AuditConfig,
    _conservative_critical_value,
    audit_records,
)
from agent_audit.io import load_score_records, write_score_records
from agent_audit.models import ScoreRecord
from agent_audit.report import render_markdown_report


ROOT = Path(__file__).resolve().parents[1]


class AuditTests(unittest.TestCase):
    def test_demo_detects_high_risk(self) -> None:
        records = load_score_records(ROOT / "examples" / "demo_scores.csv")
        result = audit_records(records)

        self.assertEqual(result.risk_level, "HIGH")
        self.assertEqual(result.case_count, 3)
        self.assertEqual(result.variant_count, 9)
        self.assertEqual(result.violation_count, 6)
        self.assertAlmostEqual(result.violation_rate, 2 / 3)
        self.assertIsNotNone(result.mean_validity_margin)
        self.assertLessEqual(result.mean_validity_margin or 0.0, 0.0)

    def test_low_risk_system(self) -> None:
        records = [
            ScoreRecord("Stable", "c1", "base", "baseline", 7.0),
            ScoreRecord("Stable", "c1", "gaming", "gaming", 6.9),
            ScoreRecord("Stable", "c1", "degraded", "degradation", 5.5),
            ScoreRecord("Stable", "c1", "paraphrase", "paraphrase", 7.1),
        ]
        result = audit_records(records, AuditConfig())

        self.assertEqual(result.risk_level, "LOW")
        self.assertEqual(result.violation_count, 0)
        self.assertGreater(result.mean_validity_margin or 0.0, 0.0)

    def test_requires_one_baseline_per_case(self) -> None:
        records = [
            ScoreRecord("Broken", "c1", "gaming", "gaming", 6.0),
        ]
        with self.assertRaisesRegex(ValueError, "exactly one baseline"):
            audit_records(records)

    def test_rejects_duplicate_variant_ids(self) -> None:
        records = [
            ScoreRecord("Broken", "c1", "base", "baseline", 6.0),
            ScoreRecord("Broken", "c1", "same", "gaming", 6.5),
            ScoreRecord("Broken", "c1", "same", "paraphrase", 6.0),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate variant_id"):
            audit_records(records)

    def test_report_contains_findings_and_limits(self) -> None:
        result = audit_records(
            load_score_records(ROOT / "examples" / "demo_scores.csv"),
            AuditConfig(data_provenance="synthetic"),
        )
        report = render_markdown_report(result)

        self.assertIn("可靠性审计报告", report)
        self.assertIn("效度余量", report)
        self.assertIn("合成数据", report)
        self.assertIn("演示级", report)

    def test_report_does_not_mislabel_public_demo_as_synthetic(self) -> None:
        result = audit_records(
            load_score_records(ROOT / "examples" / "demo_scores.csv"),
            AuditConfig(data_provenance="public-demo"),
        )
        report = render_markdown_report(result)

        self.assertIn("公开示范数据", report)
        self.assertNotIn("合成数据，仅用于方法演示", report)

    def test_rejects_non_finite_scores_without_a_declared_range(self) -> None:
        """A NaN score must fail closed, not silently report LOW risk.

        Comparisons against NaN are always False, so an unchecked NaN makes
        every variant look non-violating. The range check only runs when
        score_min/score_max are declared, so audit_records must reject
        non-finite scores on its own.
        """

        for bad_score in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(score=bad_score):
                records = [
                    ScoreRecord("Broken", "c1", "base", "baseline", bad_score),
                    ScoreRecord("Broken", "c1", "gaming", "gaming", 9.0),
                    ScoreRecord("Broken", "c1", "degraded", "degradation", 1.0),
                ]
                with self.assertRaisesRegex(ValueError, "finite"):
                    audit_records(records, AuditConfig())

    def test_conservative_critical_value_never_falls_below_the_table(self) -> None:
        """The interval must stay conservative past the tabulated range.

        Returning the normal approximation 1.96 for large samples makes the
        interval narrower than the true t interval, which would hide genuinely
        borderline judgments instead of marking them provisional.
        """

        smallest_tabulated = 2.042
        previous = None
        for sample_count in range(2, 120):
            value = _conservative_critical_value(sample_count, sample_count)
            self.assertGreaterEqual(
                value,
                smallest_tabulated,
                f"critical value for n={sample_count} is anti-conservative",
            )
            if previous is not None:
                self.assertLessEqual(value, previous)
            previous = value

    def test_evidence_grade_follows_the_published_case_count_tiers(self) -> None:
        """Gate 2 fixes three tiers; the label drives how a client may cite it.

        Under 5 baselines is demonstration-only, 5-29 is exploratory, and 30 or
        more is still only screening grade. Both the Markdown and the HTML
        report must agree on the tier.
        """

        from agent_audit.html_report import render_audit_html

        for case_count, markdown_label, html_label in (
            (3, "演示级", "演示级"),
            (5, "探索性", "探索性"),
            (30, "筛查级", "筛查级"),
        ):
            with self.subTest(case_count=case_count):
                records: list[ScoreRecord] = []
                for index in range(case_count):
                    case_id = f"c{index}"
                    records.extend(
                        [
                            ScoreRecord("Graded", case_id, "base", "baseline", 7.0),
                            ScoreRecord("Graded", case_id, "gaming", "gaming", 6.8),
                            ScoreRecord(
                                "Graded", case_id, "degraded", "degradation", 5.5
                            ),
                        ]
                    )

                result = audit_records(records, AuditConfig())

                self.assertEqual(result.case_count, case_count)

                # The report also prints a static sentence naming all three
                # tiers, so assert on the computed line rather than the whole
                # document.
                evidence_lines = [
                    line
                    for line in render_markdown_report(result).splitlines()
                    if line.startswith("- 证据强度：")
                ]
                self.assertEqual(len(evidence_lines), 1)
                self.assertIn(markdown_label, evidence_lines[0])

                html = render_audit_html(result)
                self.assertIn(
                    f'<div class="label">证据强度</div><div class="value">{html_label}</div>',
                    html,
                )

    def test_report_states_the_variant_origin_and_its_evidence_limit(self) -> None:
        """A machine-generated set is a floor test and must say so."""

        records = [
            ScoreRecord("Graded", "c1", "base", "baseline", 7.0),
            ScoreRecord("Graded", "c1", "gaming", "gaming", 6.8),
            ScoreRecord("Graded", "c1", "degraded", "degradation", 5.5),
        ]
        floor_wording = "下限测试"

        for origin, expect_floor_note in (
            ("human-authored", False),
            ("machine-generated", True),
            ("mixed", True),
            ("unspecified", False),
        ):
            with self.subTest(origin=origin):
                result = audit_records(records, AuditConfig(variant_origin=origin))

                report = render_markdown_report(result)
                origin_lines = [
                    line
                    for line in report.splitlines()
                    if line.startswith("- 变体来源：")
                ]

                self.assertEqual(len(origin_lines), 1)
                self.assertEqual(floor_wording in origin_lines[0], expect_floor_note)

    def test_rejects_negative_thresholds(self) -> None:
        records = load_score_records(ROOT / "examples" / "demo_scores.csv")
        with self.assertRaisesRegex(ValueError, "invariance_tolerance"):
            audit_records(records, AuditConfig(invariance_tolerance=-0.1))

    def test_enforces_configured_score_range(self) -> None:
        records = load_score_records(ROOT / "examples" / "demo_scores.csv")
        with self.assertRaisesRegex(ValueError, "outside the configured range"):
            audit_records(records, AuditConfig(score_min=0, score_max=5))

    def test_requires_paired_gaming_and_degradation_variants(self) -> None:
        records = [
            ScoreRecord("Incomplete", "c1", "base", "baseline", 6.0),
            ScoreRecord("Incomplete", "c1", "game", "gaming", 6.5),
        ]
        with self.assertRaisesRegex(ValueError, "missing paired validity variants"):
            audit_records(records)

    def test_critical_threshold_scales_with_score_range(self) -> None:
        hundred_point_records = [
            ScoreRecord("Hundred", "c1", "base", "baseline", 70.0),
            ScoreRecord("Hundred", "c1", "game", "gaming", 73.0),
            ScoreRecord("Hundred", "c1", "drop", "degradation", 65.0),
        ]
        five_point_records = [
            ScoreRecord("Five", "c1", "base", "baseline", 3.0),
            ScoreRecord("Five", "c1", "game", "gaming", 4.1),
            ScoreRecord("Five", "c1", "drop", "degradation", 1.5),
        ]

        hundred_point = audit_records(
            hundred_point_records,
            AuditConfig(
                score_min=0,
                score_max=100,
                min_degradation_drop=4,
            ),
        )
        five_point = audit_records(
            five_point_records,
            AuditConfig(
                score_min=0,
                score_max=5,
                min_degradation_drop=0.5,
            ),
        )

        self.assertEqual(hundred_point.risk_level, "HIGH")
        self.assertEqual(five_point.risk_level, "CRITICAL")

    def test_repeated_scores_mark_threshold_uncertainty(self) -> None:
        records = [
            ScoreRecord(
                "Repeated", "c1", "base", "baseline", 7.0,
                score_stddev=0.2, sample_count=4,
            ),
            ScoreRecord(
                "Repeated", "c1", "game", "gaming", 7.3,
                score_stddev=0.3, sample_count=4,
            ),
            ScoreRecord(
                "Repeated", "c1", "drop", "degradation", 5.0,
                score_stddev=0.2, sample_count=4,
            ),
        ]
        result = audit_records(records, AuditConfig(score_min=0, score_max=10))
        report = render_markdown_report(result)

        self.assertEqual(result.uncertain_count, 1)
        self.assertEqual(result.uncertainty_evaluable_count, 2)
        self.assertTrue(result.risk_is_provisional)
        self.assertTrue(result.cases[0].variants[0].uncertain)
        self.assertIn("是（临界）", report)
        self.assertIn("SE", report)
        self.assertIn("HIGH，暂定", report)

    def test_small_repeated_sample_uses_conservative_t_interval(self) -> None:
        records = [
            ScoreRecord(
                "Repeated", "c1", "base", "baseline", 7.0,
                score_stddev=0.18, sample_count=3,
            ),
            ScoreRecord(
                "Repeated", "c1", "game", "gaming", 7.4,
                score_stddev=0.18, sample_count=3,
            ),
            ScoreRecord(
                "Repeated", "c1", "drop", "degradation", 5.0,
                score_stddev=0.0, sample_count=3,
            ),
        ]

        result = audit_records(records, AuditConfig(score_min=0, score_max=10))

        # SE is about 0.147: a normal 1.96 interval would not cross zero,
        # while the conservative t critical value for df=2 correctly does.
        self.assertTrue(result.cases[0].variants[0].uncertain)
        self.assertEqual(result.uncertain_count, 1)

    def test_single_scores_make_risk_provisional(self) -> None:
        records = [
            ScoreRecord("Single", "c1", "base", "baseline", 7.0),
            ScoreRecord("Single", "c1", "game", "gaming", 7.0),
            ScoreRecord("Single", "c1", "drop", "degradation", 5.0),
        ]

        result = audit_records(records, AuditConfig(score_min=0, score_max=10))
        report = render_markdown_report(result)

        self.assertEqual(result.uncertainty_evaluable_count, 0)
        self.assertTrue(result.risk_is_provisional)
        self.assertIn("0/2 项变体", report)
        self.assertIn("未估计评分随机性", report)

    def test_score_record_uncertainty_round_trip(self) -> None:
        records = [
            ScoreRecord(
                "Repeated", "c1", "base", "baseline", 7.0,
                notes="test", score_stddev=0.25, sample_count=3,
            )
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "scores.csv"
            write_score_records(path, records)
            loaded = load_score_records(path)

        self.assertEqual(loaded, records)


if __name__ == "__main__":
    unittest.main()
