"""Fail-closed tests for the validation surface.

Gate 1 and Gate 3 promise that malformed configuration, malformed records and
malformed CSV input are rejected rather than silently audited. Those refusal
branches carry the project's safety claim, so each one is exercised here with
the specific message a user would have to act on.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_audit.audit import AuditConfig, audit_records
from agent_audit.io import load_score_records, load_scoring_cases
from agent_audit.models import ScoreRecord


def _paired_case(
    *,
    baseline: float = 6.0,
    gaming: float = 6.0,
    degradation: float = 4.0,
) -> list[ScoreRecord]:
    """Return the minimum valid case: one baseline plus the required pair."""

    return [
        ScoreRecord("System", "c1", "baseline", "baseline", baseline),
        ScoreRecord("System", "c1", "gaming", "gaming", gaming),
        ScoreRecord("System", "c1", "degraded", "degradation", degradation),
    ]


class AuditConfigValidationTests(unittest.TestCase):
    def test_rejects_non_finite_thresholds(self) -> None:
        for field in (
            "invariance_tolerance",
            "min_degradation_drop",
            "gaming_tolerance",
        ):
            with self.subTest(field=field):
                config = AuditConfig(**{field: float("nan")})
                with self.assertRaisesRegex(ValueError, "finite, non-negative"):
                    config.validate()

    def test_rejects_a_half_declared_score_range(self) -> None:
        for kwargs in ({"score_min": 0.0}, {"score_max": 10.0}):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, "provided together"):
                    AuditConfig(**kwargs).validate()

    def test_rejects_a_non_finite_score_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "score_min and score_max must be finite"):
            AuditConfig(score_min=0.0, score_max=float("inf")).validate()

    def test_rejects_an_inverted_score_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than score_min"):
            AuditConfig(score_min=10.0, score_max=0.0).validate()

    def test_rejects_an_undeclared_provenance_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "data_provenance must be one of"):
            AuditConfig(data_provenance="client-private").validate()


class AuditRecordValidationTests(unittest.TestCase):
    def test_rejects_an_empty_record_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one score record"):
            audit_records([])

    def test_rejects_more_than_one_system_in_a_run(self) -> None:
        records = _paired_case()
        records.append(ScoreRecord("Other", "c1", "extra", "paraphrase", 6.0))
        with self.assertRaisesRegex(ValueError, "exactly one system_name"):
            audit_records(records)

    def test_rejects_a_non_positive_sample_count(self) -> None:
        records = _paired_case()
        records[1] = ScoreRecord(
            "System", "c1", "gaming", "gaming", 6.0, sample_count=0
        )
        with self.assertRaisesRegex(ValueError, "must be at least 1"):
            audit_records(records)

    def test_rejects_a_negative_standard_deviation(self) -> None:
        records = _paired_case()
        records[1] = ScoreRecord(
            "System", "c1", "gaming", "gaming", 6.0, score_stddev=-0.1, sample_count=3
        )
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            audit_records(records)

    def test_requires_a_standard_deviation_for_repeated_scores(self) -> None:
        records = _paired_case()
        records[1] = ScoreRecord(
            "System", "c1", "gaming", "gaming", 6.0, sample_count=3
        )
        with self.assertRaisesRegex(ValueError, "score_stddev is required"):
            audit_records(records)

    def test_rejects_a_standard_deviation_on_a_single_score(self) -> None:
        records = _paired_case()
        records[1] = ScoreRecord(
            "System", "c1", "gaming", "gaming", 6.0, score_stddev=0.4, sample_count=1
        )
        with self.assertRaisesRegex(ValueError, "must be empty or 0"):
            audit_records(records)

    def test_rejects_scores_outside_a_declared_range(self) -> None:
        records = _paired_case(gaming=11.0)
        config = AuditConfig(score_min=0.0, score_max=10.0)
        with self.assertRaisesRegex(ValueError, "outside the configured range"):
            audit_records(records, config)

    def test_rejects_a_case_without_a_baseline(self) -> None:
        records = [
            ScoreRecord("System", "c1", "gaming", "gaming", 6.0),
            ScoreRecord("System", "c1", "degraded", "degradation", 4.0),
        ]
        with self.assertRaisesRegex(ValueError, "exactly one baseline"):
            audit_records(records)

    def test_rejects_a_case_missing_its_paired_variants(self) -> None:
        records = [
            ScoreRecord("System", "c1", "baseline", "baseline", 6.0),
            ScoreRecord("System", "c1", "gaming", "gaming", 6.0),
        ]
        with self.assertRaisesRegex(ValueError, "missing paired validity variants"):
            audit_records(records)


class ScoreCsvValidationTests(unittest.TestCase):
    HEADER = "system_name,case_id,variant_id,variant_type,score,score_stddev,sample_count"
    VALID_ROWS = (
        "System,c1,baseline,baseline,6.0,,1",
        "System,c1,gaming,gaming,6.0,,1",
        "System,c1,degraded,degradation,4.0,,1",
    )

    def _write_csv(self, body: str) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "scores.csv"
        path.write_text(body, encoding="utf-8", newline="\n")
        return path

    def _csv_with(self, *rows: str) -> Path:
        return self._write_csv("\n".join((self.HEADER, *rows)) + "\n")

    def test_rejects_a_missing_file(self) -> None:
        missing = Path(tempfile.mkdtemp()) / "absent.csv"
        with self.assertRaisesRegex(ValueError, "does not exist"):
            load_score_records(missing)

    def test_rejects_a_file_without_a_header(self) -> None:
        with self.assertRaisesRegex(ValueError, "no header row"):
            load_score_records(self._write_csv(""))

    def test_rejects_missing_required_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing columns"):
            load_score_records(self._write_csv("case_id,score\nc1,6.0\n"))

    def test_rejects_an_unknown_variant_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported variant_type"):
            load_score_records(self._csv_with("System,c1,x,rewrite,6.0,,1"))

    def test_rejects_a_non_numeric_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "score must be numeric"):
            load_score_records(self._csv_with("System,c1,x,gaming,high,,1"))

    def test_rejects_a_non_finite_score(self) -> None:
        for literal in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(literal=literal):
                with self.assertRaisesRegex(ValueError, "score must be finite"):
                    load_score_records(
                        self._csv_with(f"System,c1,x,gaming,{literal},,1")
                    )

    def test_rejects_a_non_numeric_standard_deviation(self) -> None:
        with self.assertRaisesRegex(ValueError, "score_stddev must be numeric"):
            load_score_records(self._csv_with("System,c1,x,gaming,6.0,wide,3"))

    def test_rejects_a_negative_standard_deviation(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            load_score_records(self._csv_with("System,c1,x,gaming,6.0,-1,3"))

    def test_rejects_a_non_integer_sample_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_count must be an integer"):
            load_score_records(self._csv_with("System,c1,x,gaming,6.0,,three"))

    def test_rejects_a_sample_count_below_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_count must be at least 1"):
            load_score_records(self._csv_with("System,c1,x,gaming,6.0,,0"))

    def test_requires_a_standard_deviation_for_repeated_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "score_stddev is required"):
            load_score_records(self._csv_with("System,c1,x,gaming,6.0,,3"))

    def test_rejects_a_standard_deviation_on_a_single_row(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be empty or 0"):
            load_score_records(self._csv_with("System,c1,x,gaming,6.0,0.4,1"))

    def test_rejects_empty_identifier_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty required values"):
            load_score_records(self._csv_with("System,,x,gaming,6.0,,1"))

    def test_rejects_a_header_only_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "no score records"):
            load_score_records(self._write_csv(self.HEADER + "\n"))

    def test_accepts_the_minimum_valid_file(self) -> None:
        records = load_score_records(self._csv_with(*self.VALID_ROWS))

        self.assertEqual(len(records), 3)
        self.assertEqual({record.case_id for record in records}, {"c1"})


class ScoringCaseCsvValidationTests(unittest.TestCase):
    HEADER = "case_id,variant_id,variant_type,text"

    def _csv_with(self, *rows: str) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "cases.csv"
        path.write_text(
            "\n".join((self.HEADER, *rows)) + "\n", encoding="utf-8", newline="\n"
        )
        return path

    def test_rejects_duplicate_case_and_variant_pairs(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate case_id/variant_id"):
            load_scoring_cases(
                self._csv_with(
                    "c1,baseline,baseline,text one",
                    "c1,baseline,gaming,text two",
                )
            )

    def test_rejects_a_case_without_exactly_one_baseline(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one baseline"):
            load_scoring_cases(
                self._csv_with(
                    "c1,g1,gaming,text one",
                    "c1,d1,degradation,text two",
                )
            )

    def test_rejects_a_case_missing_its_paired_variants(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing paired validity variants"):
            load_scoring_cases(
                self._csv_with(
                    "c1,b1,baseline,text one",
                    "c1,g1,gaming,text two",
                )
            )

    def test_rejects_an_empty_text_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty required values"):
            load_scoring_cases(
                self._csv_with(
                    "c1,b1,baseline,",
                    "c1,g1,gaming,text two",
                    "c1,d1,degradation,text three",
                )
            )

    def test_accepts_a_complete_case_set(self) -> None:
        cases = load_scoring_cases(
            self._csv_with(
                "c1,b1,baseline,text one",
                "c1,g1,gaming,text two",
                "c1,d1,degradation,text three",
            )
        )

        self.assertEqual(len(cases), 3)
        self.assertEqual(
            {case.variant_type for case in cases},
            {"baseline", "gaming", "degradation"},
        )
        self.assertEqual([case.text for case in cases][0], "text one")


if __name__ == "__main__":
    unittest.main()
