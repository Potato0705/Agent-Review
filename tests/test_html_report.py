from __future__ import annotations

import unittest

from agent_audit.audit import AuditConfig, audit_records
from agent_audit.comparison import compare_audits
from agent_audit.html_report import render_audit_html, render_comparison_html
from agent_audit.models import ScoreRecord
from agent_audit.report import render_markdown_report


def _audit_payload(system_name: str, model: str, game_score: float) -> dict[str, object]:
    records = [
        ScoreRecord(system_name, "case<svg/onload=alert(1)>", "base", "baseline", 7.0),
        ScoreRecord(
            system_name,
            "case<svg/onload=alert(1)>",
            "game<img/onerror=alert(1)>",
            "gaming",
            game_score,
        ),
        ScoreRecord(
            system_name,
            "case<svg/onload=alert(1)>",
            "drop",
            "degradation",
            5.0,
        ),
    ]
    payload = audit_records(
        records,
        AuditConfig(score_min=0, score_max=10, data_provenance="public-demo"),
    ).to_dict()
    payload["comparison_context"] = {
        "input_sha256": "a" * 64,
        "rubric_sha256": "b" * 64,
        "temperature": 0.2,
        "repeats": 3,
        "model": model,
    }
    return payload


class BorderlineBadgeTests(unittest.TestCase):
    """A borderline judgement must be visible as such in the delivered page."""

    def _borderline_result(self):
        # Repeated sampling with wide spread puts the gaming judgement's
        # conservative interval across the threshold.
        records = [
            ScoreRecord(
                "Grader", "c1", "base", "baseline", 7.0,
                score_stddev=1.2, sample_count=3,
            ),
            ScoreRecord(
                "Grader", "c1", "gaming", "gaming", 7.05,
                score_stddev=1.2, sample_count=3,
            ),
            ScoreRecord(
                "Grader", "c1", "degraded", "degradation", 5.9,
                score_stddev=1.2, sample_count=3,
            ),
        ]
        return audit_records(
            records, AuditConfig(score_min=0, score_max=10, data_provenance="synthetic")
        )

    def test_borderline_variants_are_badged_in_the_html(self) -> None:
        result = self._borderline_result()
        self.assertGreater(result.uncertain_count, 0, "fixture is not borderline")

        html = render_audit_html(result)

        self.assertIn("· 临界", html)
        self.assertIn('class="badge pending"', html)

    def test_borderline_count_drives_a_repeat_sampling_recommendation(self) -> None:
        result = self._borderline_result()

        html = render_audit_html(result)

        self.assertIn(
            f"有{result.uncertain_count}项保守95%区间跨越阈值，应增加重复次数后复核。",
            html,
        )

    def test_paraphrase_drift_adds_its_own_recommendation(self) -> None:
        records = [
            ScoreRecord("Grader", "c1", "base", "baseline", 7.0),
            ScoreRecord("Grader", "c1", "gaming", "gaming", 6.9),
            ScoreRecord("Grader", "c1", "degraded", "degradation", 5.5),
            ScoreRecord("Grader", "c1", "reworded", "paraphrase", 9.0),
        ]
        result = audit_records(records, AuditConfig())

        self.assertIn(
            "使用等义改写回归集监控表达形式敏感性。", render_audit_html(result)
        )
        self.assertIn(
            "使用等义改写回归集监控表达形式敏感性，并考虑多次评分或确定性解码。",
            render_markdown_report(result),
        )

    def test_a_clean_result_still_carries_a_forward_recommendation(self) -> None:
        records = [
            ScoreRecord(
                "Grader", "c1", "base", "baseline", 7.0,
                score_stddev=0.0, sample_count=3,
            ),
            ScoreRecord(
                "Grader", "c1", "gaming", "gaming", 6.0,
                score_stddev=0.0, sample_count=3,
            ),
            ScoreRecord(
                "Grader", "c1", "degraded", "degradation", 4.0,
                score_stddev=0.0, sample_count=3,
            ),
        ]
        result = audit_records(records, AuditConfig())

        self.assertEqual(result.violation_count, 0)
        self.assertIn(
            "保留当前测试集作为版本回归门槛，并扩展真实用户切片。",
            render_audit_html(result),
        )


class HtmlReportTests(unittest.TestCase):
    def test_audit_html_escapes_untrusted_labels_and_disables_scripts(self) -> None:
        result = audit_records(
            [
                ScoreRecord(
                    "<script>alert('system')</script>",
                    "case<svg/onload=alert(1)>",
                    "base",
                    "baseline",
                    7.0,
                ),
                ScoreRecord(
                    "<script>alert('system')</script>",
                    "case<svg/onload=alert(1)>",
                    "game<img/onerror=alert(1)>",
                    "gaming",
                    7.5,
                ),
                ScoreRecord(
                    "<script>alert('system')</script>",
                    "case<svg/onload=alert(1)>",
                    "drop",
                    "degradation",
                    5.0,
                ),
            ],
            AuditConfig(score_min=0, score_max=10, data_provenance="public-demo"),
        )

        html = render_audit_html(result)

        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn("Content-Security-Policy", html)
        self.assertIn("default-src 'none'", html)
        self.assertNotIn("<script>alert('system')</script>", html)
        self.assertNotIn("<svg/onload=alert(1)>", html)
        self.assertNotIn("<img/onerror=alert(1)>", html)
        self.assertIn("&lt;script&gt;alert(&#x27;system&#x27;)&lt;/script&gt;", html)
        self.assertIn("可靠性审计", html)

    def test_comparison_html_escapes_system_and_manifest_model(self) -> None:
        reference = _audit_payload("Reference<script>", "model<script>", 7.0)
        candidate = _audit_payload("Candidate<img/onerror=x>", "model<img/onerror=x>", 7.8)
        result = compare_audits(reference, candidate)

        html = render_comparison_html(result)

        self.assertNotIn("Reference<script>", html)
        self.assertNotIn("<img/onerror=x>", html)
        self.assertIn("Reference&lt;script&gt;", html)
        self.assertIn("model&lt;img/onerror=x&gt;", html)
        self.assertIn("跨阈值回归", html)
        self.assertIn(result.input_sha256, html)


if __name__ == "__main__":
    unittest.main()
