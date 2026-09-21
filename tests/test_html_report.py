from __future__ import annotations

import unittest

from agent_audit.audit import AuditConfig, audit_records
from agent_audit.comparison import compare_audits
from agent_audit.html_report import render_audit_html, render_comparison_html
from agent_audit.models import ScoreRecord


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
