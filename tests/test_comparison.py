from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from agent_audit.audit import AuditConfig, audit_records
from agent_audit.cli_compare import run as run_compare
from agent_audit.comparison import compare_audits, render_comparison_report
from agent_audit.models import ScoreRecord


def _audit(system_name: str, game: float, drop: float, paraphrase: float) -> dict[str, object]:
    records = [
        ScoreRecord(system_name, "c1", "base", "baseline", 7.0),
        ScoreRecord(system_name, "c1", "game", "gaming", game),
        ScoreRecord(system_name, "c1", "drop", "degradation", drop),
        ScoreRecord(system_name, "c1", "para", "paraphrase", paraphrase),
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
        "model": system_name,
    }
    return payload


class ComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = _audit("Reference", game=7.0, drop=5.0, paraphrase=7.0)
        self.candidate = _audit("Candidate", game=7.8, drop=6.5, paraphrase=7.2)

    def test_detects_threshold_regressions_and_magnitude_change(self) -> None:
        result = compare_audits(self.reference, self.candidate)
        report = render_comparison_report(result)

        self.assertEqual(result.reference_risk, "LOW")
        self.assertEqual(result.candidate_risk, "HIGH")
        self.assertEqual(result.input_sha256, "a" * 64)
        self.assertEqual(result.reference_model, "Reference")
        self.assertEqual(result.candidate_model, "Candidate")
        self.assertEqual(result.temperature, 0.2)
        self.assertEqual(result.repeats, 3)
        self.assertEqual(result.risk_rank_change, 2)
        self.assertEqual(result.regression_count, 2)
        self.assertEqual(result.worsened_count, 1)
        self.assertEqual(result.improvement_count, 0)
        self.assertEqual(result.provisional_change_count, 3)
        self.assertIn("跨阈值回归：2", report)
        self.assertIn("跨阈值回归（暂定）", report)
        self.assertIn("严重度变化统一为正数表示候选系统变差", report)

    def test_reverse_comparison_detects_improvements(self) -> None:
        result = compare_audits(self.candidate, self.reference)

        self.assertEqual(result.improvement_count, 2)
        self.assertEqual(result.improved_count, 1)
        self.assertEqual(result.regression_count, 0)

    def test_rejects_incompatible_thresholds(self) -> None:
        incompatible = json.loads(json.dumps(self.candidate))
        incompatible["config"]["invariance_tolerance"] = 0.25

        with self.assertRaisesRegex(ValueError, "incompatible thresholds"):
            compare_audits(self.reference, incompatible)

    def test_rejects_different_provenance_declarations(self) -> None:
        incompatible = json.loads(json.dumps(self.candidate))
        incompatible["config"]["data_provenance"] = "authorized-private"

        with self.assertRaisesRegex(ValueError, "different data_provenance"):
            compare_audits(self.reference, incompatible)

    def test_rejects_different_variant_sets(self) -> None:
        incompatible = json.loads(json.dumps(self.candidate))
        incompatible["cases"][0]["variants"].pop()
        incompatible["variant_count"] = 2

        with self.assertRaisesRegex(ValueError, "same case/variant identities"):
            compare_audits(self.reference, incompatible)

    def test_rejects_unsupported_schema_version(self) -> None:
        incompatible = json.loads(json.dumps(self.candidate))
        incompatible["schema_version"] = 99

        with self.assertRaisesRegex(ValueError, "Unsupported audit result schema_version"):
            compare_audits(self.reference, incompatible)

    def test_rejects_inconsistent_summary_counts(self) -> None:
        incompatible = json.loads(json.dumps(self.candidate))
        incompatible["violation_count"] = 0

        with self.assertRaisesRegex(ValueError, "violation_count does not match"):
            compare_audits(self.reference, incompatible)

    def test_rejects_different_verified_inputs(self) -> None:
        incompatible = json.loads(json.dumps(self.candidate))
        incompatible["comparison_context"]["input_sha256"] = "c" * 64

        with self.assertRaisesRegex(ValueError, "same verified scoring context"):
            compare_audits(self.reference, incompatible)

    def test_rejects_missing_comparison_context(self) -> None:
        incompatible = json.loads(json.dumps(self.candidate))
        incompatible.pop("comparison_context")

        with self.assertRaisesRegex(ValueError, "missing comparison_context"):
            compare_audits(self.reference, incompatible)

    def test_compare_cli_writes_markdown_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference_path = root / "reference.json"
            candidate_path = root / "candidate.json"
            report_path = root / "comparison.md"
            json_path = root / "comparison.json"
            html_path = root / "comparison.html"
            reference_path.write_text(
                json.dumps(self.reference, ensure_ascii=False), encoding="utf-8"
            )
            candidate_path.write_text(
                json.dumps(self.candidate, ensure_ascii=False), encoding="utf-8"
            )
            args = SimpleNamespace(
                reference=str(reference_path),
                candidate=str(candidate_path),
                report=str(report_path),
                json_output=str(json_path),
                html_output=str(html_path),
            )

            self.assertEqual(run_compare(args), 0)
            self.assertTrue(report_path.exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["regression_count"], 2)
            self.assertIn(
                "评分系统版本对比",
                html_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
