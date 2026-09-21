"""Gate 6 tests: an incomparable or tampered pair must be refused.

Every case here starts from a pair that compares cleanly, then breaks exactly
one thing. If any of these silently produced a comparison, the tool would let a
user publish a model-versus-model claim that its own inputs do not support.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from agent_audit.audit import AuditConfig, audit_records
from agent_audit.comparison import (
    compare_audits,
    load_audit_result,
    render_comparison_report,
)
from agent_audit.models import ScoreRecord


def _audit_payload(
    system_name: str, *, game: float, drop: float, paraphrase: float
) -> dict[str, Any]:
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
    # Round-trip through JSON so the fixture has the same shape the CLI feeds
    # in: lists rather than the tuples a dataclass conversion leaves behind.
    return json.loads(json.dumps(payload, ensure_ascii=False))


class ComparisonRefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = _audit_payload("Reference", game=7.0, drop=5.0, paraphrase=7.0)
        self.candidate = _audit_payload("Candidate", game=7.8, drop=6.5, paraphrase=7.2)
        # The fixtures must compare cleanly before anything is broken, or a
        # refusal below could be passing for the wrong reason.
        compare_audits(self.reference, self.candidate)

    def _assert_refused(
        self, mutate: Callable[[dict[str, Any]], None], pattern: str
    ) -> None:
        candidate = copy.deepcopy(self.candidate)
        mutate(candidate)
        with self.assertRaisesRegex(ValueError, pattern):
            compare_audits(self.reference, candidate)

    # --- verified scoring context -------------------------------------------

    def test_refuses_a_missing_comparison_context(self) -> None:
        self._assert_refused(
            lambda payload: payload.pop("comparison_context"),
            "missing comparison_context",
        )

    def test_refuses_a_malformed_input_hash(self) -> None:
        self._assert_refused(
            lambda payload: payload["comparison_context"].update(
                input_sha256="not-a-hash"
            ),
            "is not a SHA-256 hash",
        )

    def test_refuses_a_different_input_hash(self) -> None:
        self._assert_refused(
            lambda payload: payload["comparison_context"].update(
                input_sha256="c" * 64
            ),
            "same verified scoring context",
        )

    def test_refuses_a_different_rubric_hash(self) -> None:
        self._assert_refused(
            lambda payload: payload["comparison_context"].update(
                rubric_sha256="d" * 64
            ),
            "same verified scoring context",
        )

    def test_refuses_a_different_temperature(self) -> None:
        self._assert_refused(
            lambda payload: payload["comparison_context"].update(temperature=0.9),
            "same verified scoring context",
        )

    def test_refuses_a_different_repeat_count(self) -> None:
        self._assert_refused(
            lambda payload: payload["comparison_context"].update(repeats=5),
            "same verified scoring context",
        )

    def test_refuses_a_non_positive_repeat_count(self) -> None:
        self._assert_refused(
            lambda payload: payload["comparison_context"].update(repeats=0),
            "must be a positive integer",
        )

    def test_refuses_a_different_generation_fingerprint(self) -> None:
        self._assert_refused(
            lambda payload: payload["comparison_context"].update(
                generation_sha256="e" * 64
            ),
            "same verified scoring context",
        )

    def test_refuses_a_malformed_generation_fingerprint(self) -> None:
        self._assert_refused(
            lambda payload: payload["comparison_context"].update(
                generation_sha256="not-a-hash"
            ),
            "is not a SHA-256 hash",
        )

    def test_accepts_a_pair_generated_from_the_same_run(self) -> None:
        reference = copy.deepcopy(self.reference)
        candidate = copy.deepcopy(self.candidate)
        reference["comparison_context"]["generation_sha256"] = "f" * 64
        candidate["comparison_context"]["generation_sha256"] = "f" * 64

        result = compare_audits(reference, candidate)

        self.assertEqual(result.reference_risk, "LOW")

    def test_accepts_a_pair_with_no_generation_fingerprint(self) -> None:
        """Hand-written variant sets carry no generation fingerprint."""

        reference = copy.deepcopy(self.reference)
        candidate = copy.deepcopy(self.candidate)
        reference["comparison_context"]["generation_sha256"] = None
        candidate["comparison_context"]["generation_sha256"] = None

        result = compare_audits(reference, candidate)

        self.assertEqual(result.reference_risk, "LOW")

    def test_refuses_an_unnamed_model(self) -> None:
        self._assert_refused(
            lambda payload: payload["comparison_context"].update(model="  "),
            "must be a non-empty string",
        )

    # --- declared configuration ---------------------------------------------

    def test_refuses_a_different_data_provenance(self) -> None:
        self._assert_refused(
            lambda payload: payload["config"].update(data_provenance="synthetic"),
            "different data_provenance",
        )

    def test_refuses_a_different_variant_origin(self) -> None:
        self._assert_refused(
            lambda payload: payload["config"].update(
                variant_origin="machine-generated"
            ),
            "different variant_origin",
        )

    def test_accepts_a_pair_that_predates_the_origin_field(self) -> None:
        """Older audit JSON has no field at all; absent on both sides is equal."""

        reference = copy.deepcopy(self.reference)
        candidate = copy.deepcopy(self.candidate)
        reference["config"].pop("variant_origin", None)
        candidate["config"].pop("variant_origin", None)

        result = compare_audits(reference, candidate)

        self.assertEqual(result.reference_risk, "LOW")

    def test_refuses_different_decision_thresholds(self) -> None:
        for field in (
            "invariance_tolerance",
            "min_degradation_drop",
            "gaming_tolerance",
        ):
            with self.subTest(field=field):
                self._assert_refused(
                    lambda payload, field=field: payload["config"].update(
                        {field: payload["config"][field] + 0.25}
                    ),
                    "incompatible thresholds or score ranges",
                )

    def test_refuses_a_different_score_range(self) -> None:
        self._assert_refused(
            lambda payload: payload["config"].update(score_max=100.0),
            "incompatible thresholds or score ranges",
        )

    def test_refuses_a_half_declared_score_range(self) -> None:
        self._assert_refused(
            lambda payload: payload["config"].update(score_max=None),
            "score_min and score_max together",
        )

    def test_refuses_an_inverted_score_range(self) -> None:
        self._assert_refused(
            lambda payload: payload["config"].update(score_min=20.0, score_max=10.0),
            "invalid score range",
        )

    def test_refuses_a_config_that_is_not_an_object(self) -> None:
        self._assert_refused(
            lambda payload: payload.update(config="0-10"),
            "missing a valid config object",
        )

    # --- case and variant identity ------------------------------------------

    def test_refuses_a_different_variant_set(self) -> None:
        self._assert_refused(
            lambda payload: payload["cases"][0]["variants"].pop(),
            "same case/variant identities",
        )

    def test_refuses_a_retyped_variant(self) -> None:
        self._assert_refused(
            lambda payload: payload["cases"][0]["variants"][0].update(
                variant_type="paraphrase"
            ),
            "different types to variants",
        )

    def test_refuses_a_baseline_inside_the_variant_list(self) -> None:
        self._assert_refused(
            lambda payload: payload["cases"][0]["variants"][0].update(
                variant_type="baseline"
            ),
            "unsupported type",
        )

    def test_refuses_duplicate_variant_identities(self) -> None:
        def mutate(payload: dict[str, Any]) -> None:
            variants = payload["cases"][0]["variants"]
            variants.append(copy.deepcopy(variants[0]))

        self._assert_refused(mutate, "duplicate variant")

    def test_refuses_an_empty_cases_array(self) -> None:
        self._assert_refused(
            lambda payload: payload.update(cases=[]),
            "non-empty cases array",
        )

    def test_refuses_a_case_without_a_variants_array(self) -> None:
        self._assert_refused(
            lambda payload: payload["cases"][0].update(variants=None),
            "no valid variants array",
        )

    # --- per-variant payload integrity --------------------------------------

    def test_refuses_a_non_finite_delta(self) -> None:
        self._assert_refused(
            lambda payload: payload["cases"][0]["variants"][0].update(
                delta=float("inf")
            ),
            "must be finite",
        )

    def test_refuses_a_non_boolean_violation_flag(self) -> None:
        self._assert_refused(
            lambda payload: payload["cases"][0]["variants"][0].update(violated="yes"),
            "invalid violated flag",
        )

    def test_refuses_a_non_boolean_uncertainty_flag(self) -> None:
        self._assert_refused(
            lambda payload: payload["cases"][0]["variants"][0].update(uncertain=1),
            "invalid uncertain flag",
        )

    def test_refuses_a_negative_standard_error(self) -> None:
        self._assert_refused(
            lambda payload: payload["cases"][0]["variants"][0].update(
                delta_standard_error=-0.5
            ),
            "negative delta_standard_error",
        )

    # --- summary tampering --------------------------------------------------

    def test_refuses_a_summary_that_contradicts_its_own_findings(self) -> None:
        """Editing a headline number without editing the findings is refused."""

        for field, pattern in (
            ("variant_count", "variant_count does not match"),
            ("violation_count", "violation_count does not match"),
            ("uncertain_count", "uncertain_count does not match"),
        ):
            with self.subTest(field=field):
                self._assert_refused(
                    lambda payload, field=field: payload.update(
                        {field: payload[field] + 1}
                    ),
                    pattern,
                )

    def test_refuses_a_violation_rate_that_does_not_match_the_findings(self) -> None:
        self._assert_refused(
            lambda payload: payload.update(violation_rate=0.99),
            "violation_rate does not match",
        )

    # --- envelope -----------------------------------------------------------

    def test_refuses_an_unsupported_schema_version(self) -> None:
        self._assert_refused(
            lambda payload: payload.update(schema_version=2),
            "Unsupported audit result schema_version",
        )

    def test_refuses_an_unknown_risk_level(self) -> None:
        self._assert_refused(
            lambda payload: payload.update(risk_level="SEVERE"),
            "risk_level must be LOW, MEDIUM, HIGH, or CRITICAL",
        )

    def test_refuses_a_non_boolean_provisional_flag(self) -> None:
        self._assert_refused(
            lambda payload: payload.update(risk_is_provisional="maybe"),
            "must be boolean",
        )


class ComparisonTypeGuardTests(unittest.TestCase):
    """Malformed primitives are rejected rather than coerced.

    A JSON file can be hand-edited or produced by another tool, so every
    numeric field is checked for type as well as value. ``True`` is the case
    that matters most: Python would happily read it as ``1``.
    """

    def setUp(self) -> None:
        self.reference = _audit_payload("Reference", game=7.0, drop=5.0, paraphrase=7.0)
        self.candidate = _audit_payload("Candidate", game=7.8, drop=6.5, paraphrase=7.2)

    def _assert_refused(
        self, mutate: Callable[[dict[str, Any]], None], pattern: str
    ) -> None:
        candidate = copy.deepcopy(self.candidate)
        mutate(candidate)
        with self.assertRaisesRegex(ValueError, pattern):
            compare_audits(self.reference, candidate)

    def test_refuses_a_boolean_where_a_number_belongs(self) -> None:
        self._assert_refused(
            lambda payload: payload["cases"][0]["variants"][0].update(delta=True),
            "must be numeric",
        )

    def test_refuses_a_non_numeric_string_where_a_number_belongs(self) -> None:
        self._assert_refused(
            lambda payload: payload["cases"][0]["variants"][0].update(delta="a lot"),
            "must be numeric",
        )

    def test_refuses_a_non_integer_count(self) -> None:
        self._assert_refused(
            lambda payload: payload.update(variant_count=3.0),
            "must be an integer",
        )

    def test_refuses_a_negative_count(self) -> None:
        self._assert_refused(
            lambda payload: payload.update(violation_count=-1),
            "must be non-negative",
        )

    def test_refuses_a_non_integer_schema_version(self) -> None:
        self._assert_refused(
            lambda payload: payload.update(schema_version="1"),
            "schema_version must be an integer",
        )

    def test_refuses_a_case_that_is_not_an_object(self) -> None:
        self._assert_refused(
            lambda payload: payload["cases"].append("c2"),
            "must be an object",
        )

    def test_refuses_a_variant_that_is_not_an_object(self) -> None:
        self._assert_refused(
            lambda payload: payload["cases"][0]["variants"].append("game"),
            "non-object variant",
        )

    def test_refuses_an_audit_with_no_comparable_variants(self) -> None:
        self._assert_refused(
            lambda payload: payload["cases"][0].update(variants=[]),
            "no comparable variants",
        )

    def test_refuses_a_boolean_threshold(self) -> None:
        # Caught by the numeric type check before the equality comparison, so
        # the message names the field rather than the mismatch.
        self._assert_refused(
            lambda payload: payload["config"].update(gaming_tolerance=True),
            "'gaming_tolerance' must be numeric",
        )

    def test_refuses_a_non_numeric_threshold(self) -> None:
        self._assert_refused(
            lambda payload: payload["config"].update(gaming_tolerance="strict"),
            "must be numeric",
        )

    def test_compares_two_audits_without_a_declared_score_range(self) -> None:
        """An undeclared range on both sides is comparable, not an error."""

        def undeclared(system_name: str, game: float) -> dict[str, Any]:
            records = [
                ScoreRecord(system_name, "c1", "base", "baseline", 7.0),
                ScoreRecord(system_name, "c1", "game", "gaming", game),
                ScoreRecord(system_name, "c1", "drop", "degradation", 5.0),
            ]
            payload = audit_records(
                records, AuditConfig(data_provenance="public-demo")
            ).to_dict()
            payload["comparison_context"] = {
                "input_sha256": "a" * 64,
                "rubric_sha256": "b" * 64,
                "temperature": 0.2,
                "repeats": 3,
                "model": system_name,
            }
            return json.loads(json.dumps(payload, ensure_ascii=False))

        result = compare_audits(undeclared("Reference", 7.0), undeclared("Candidate", 8.0))

        self.assertEqual(result.regression_count, 1)


class ProvisionalChangeTests(unittest.TestCase):
    """A change is provisional whenever either side is itself uncertain."""

    def _repeated(self, system_name: str, gaming: float, spread: float) -> dict[str, Any]:
        records = [
            ScoreRecord(
                system_name, "c1", "base", "baseline", 7.0,
                score_stddev=spread, sample_count=3,
            ),
            ScoreRecord(
                system_name, "c1", "game", "gaming", gaming,
                score_stddev=spread, sample_count=3,
            ),
            ScoreRecord(
                system_name, "c1", "drop", "degradation", 5.0,
                score_stddev=spread, sample_count=3,
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
            "model": system_name,
        }
        return json.loads(json.dumps(payload, ensure_ascii=False))

    def test_a_tight_pair_is_not_marked_provisional(self) -> None:
        result = compare_audits(
            self._repeated("Reference", 6.0, 0.01),
            self._repeated("Candidate", 6.1, 0.01),
        )

        self.assertEqual(result.provisional_change_count, 0)

    def test_an_uncertain_candidate_makes_the_change_provisional(self) -> None:
        candidate = self._repeated("Candidate", 7.05, 1.2)
        self.assertGreater(
            sum(
                variant["uncertain"]
                for case in candidate["cases"]
                for variant in case["variants"]
            ),
            0,
            "fixture is not uncertain",
        )

        result = compare_audits(self._repeated("Reference", 6.0, 0.01), candidate)

        self.assertGreater(result.provisional_change_count, 0)


class ComparisonRenderingTests(unittest.TestCase):
    def test_an_unmeasured_metric_renders_as_not_tested(self) -> None:
        """With no paraphrase variants, the paraphrase metric has no value."""

        def without_paraphrase(system_name: str, game: float) -> dict[str, Any]:
            records = [
                ScoreRecord(system_name, "c1", "base", "baseline", 7.0),
                ScoreRecord(system_name, "c1", "game", "gaming", game),
                ScoreRecord(system_name, "c1", "drop", "degradation", 5.0),
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
            return json.loads(json.dumps(payload, ensure_ascii=False))

        result = compare_audits(
            without_paraphrase("Reference", 7.0), without_paraphrase("Candidate", 7.2)
        )

        self.assertIn("未测试", render_comparison_report(result))


class AuditJsonLoadingTests(unittest.TestCase):
    def _json_path(self, body: str) -> Path:
        path = Path(tempfile.mkdtemp()) / "audit.json"
        path.write_text(body, encoding="utf-8", newline="\n")
        return path

    def test_refuses_a_missing_file(self) -> None:
        missing = Path(tempfile.mkdtemp()) / "absent.json"
        with self.assertRaisesRegex(ValueError, "does not exist"):
            load_audit_result(missing)

    def test_refuses_malformed_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "is malformed"):
            load_audit_result(self._json_path("{not json"))

    def test_refuses_json_that_is_not_an_object(self) -> None:
        with self.assertRaisesRegex(ValueError, "must contain an object"):
            load_audit_result(self._json_path("[1, 2, 3]"))

    def test_round_trips_a_written_audit(self) -> None:
        payload = _audit_payload("Reference", game=7.0, drop=5.0, paraphrase=7.0)

        loaded = load_audit_result(
            self._json_path(json.dumps(payload, ensure_ascii=False))
        )

        self.assertEqual(loaded["system_name"], "Reference")
        self.assertEqual(loaded["comparison_context"]["repeats"], 3)


if __name__ == "__main__":
    unittest.main()
