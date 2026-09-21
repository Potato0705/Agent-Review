"""Property tests that re-derive the published formulas independently.

The tests in this module deliberately avoid reusing helpers from
``agent_audit``. They restate the definitions exactly as ``README.md`` and
``docs/governance/quality_gates.md`` publish them, then compare that
independent derivation against the shipped implementation over randomised
inputs. A sign error or a changed definition shows up here even when the
hand-written example tests still agree with the code.
"""

from __future__ import annotations

import random
import unittest
from statistics import fmean

from agent_audit.audit import AuditConfig, audit_records
from agent_audit.comparison import _severity
from agent_audit.models import ScoreRecord


TRIALS = 400
SEED = 20260921


def _random_config(rng: random.Random) -> AuditConfig:
    return AuditConfig(
        invariance_tolerance=round(rng.uniform(0.0, 2.0), 2),
        min_degradation_drop=round(rng.uniform(0.0, 3.0), 2),
        gaming_tolerance=round(rng.uniform(0.0, 1.0), 2),
    )


def _random_case_set(
    rng: random.Random, config: AuditConfig
) -> tuple[list[ScoreRecord], list[float], int]:
    """Build a random audit input plus independently derived expectations."""

    records: list[ScoreRecord] = []
    case_margins: list[float] = []
    expected_violations = 0

    for case_index in range(rng.randint(1, 4)):
        case_id = f"case{case_index}"
        baseline_score = round(rng.uniform(0.0, 10.0), 2)
        records.append(
            ScoreRecord("System", case_id, "baseline", "baseline", baseline_score)
        )

        gaming_gains: list[float] = []
        for variant_index in range(rng.randint(1, 3)):
            score = round(rng.uniform(0.0, 10.0), 2)
            records.append(
                ScoreRecord("System", case_id, f"g{variant_index}", "gaming", score)
            )
            gain = score - baseline_score
            gaming_gains.append(gain)
            # Published rule: any gain above the tolerance is a violation.
            if gain > config.gaming_tolerance:
                expected_violations += 1

        degradation_drops: list[float] = []
        for variant_index in range(rng.randint(1, 3)):
            score = round(rng.uniform(0.0, 10.0), 2)
            records.append(
                ScoreRecord("System", case_id, f"d{variant_index}", "degradation", score)
            )
            drop = baseline_score - score
            degradation_drops.append(drop)
            # Published rule: too small a drop is a violation.
            if drop < config.min_degradation_drop:
                expected_violations += 1

        for variant_index in range(rng.randint(0, 2)):
            score = round(rng.uniform(0.0, 10.0), 2)
            records.append(
                ScoreRecord("System", case_id, f"p{variant_index}", "paraphrase", score)
            )
            # Published rule: absolute drift beyond the tolerance is a violation.
            if abs(score - baseline_score) > config.invariance_tolerance:
                expected_violations += 1

        # Published formula:
        # 效度余量 = 平均内容退化降分 - 最大正向作弊收益
        case_margins.append(
            fmean(degradation_drops) - max(0.0, max(gaming_gains))
        )

    return records, case_margins, expected_violations


class ValidityFormulaTests(unittest.TestCase):
    def test_validity_margin_matches_the_published_formula(self) -> None:
        rng = random.Random(SEED)
        for trial in range(TRIALS):
            config = _random_config(rng)
            records, case_margins, _ = _random_case_set(rng, config)

            result = audit_records(records, config)

            self.assertIsNotNone(result.mean_validity_margin)
            self.assertAlmostEqual(
                result.mean_validity_margin or 0.0,
                fmean(case_margins),
                places=9,
                msg=f"validity margin diverged on trial {trial}",
            )

    def test_violation_counts_match_the_published_rules(self) -> None:
        rng = random.Random(SEED + 1)
        for trial in range(TRIALS):
            config = _random_config(rng)
            records, _, expected_violations = _random_case_set(rng, config)

            result = audit_records(records, config)

            self.assertEqual(
                result.violation_count,
                expected_violations,
                msg=f"violation count diverged on trial {trial}",
            )
            self.assertAlmostEqual(
                result.violation_rate,
                expected_violations / result.variant_count,
                places=12,
            )

    def test_gaming_reward_is_never_counted_as_negative(self) -> None:
        """A grader that punishes gaming must not inflate the margin.

        Clamping the rewarded gain at zero keeps the margin reporting how much
        degradation sensitivity exceeds real gaming reward, rather than adding
        credit for gaming attempts the grader already rejected.
        """

        punished = [
            ScoreRecord("System", "c1", "baseline", "baseline", 8.0),
            ScoreRecord("System", "c1", "gaming", "gaming", 5.0),
            ScoreRecord("System", "c1", "degraded", "degradation", 6.0),
        ]

        result = audit_records(punished, AuditConfig())

        # Degradation sensitivity is 2.0; the -3.0 gaming gain must clamp to 0.
        self.assertAlmostEqual(result.mean_validity_margin or 0.0, 2.0, places=9)


class ThresholdBoundaryTests(unittest.TestCase):
    """Pin the exact meaning of each published threshold.

    README states the rules as strict comparisons: gaming violates when the
    score *rises*, degradation when the drop is *short of* the minimum, and
    paraphrase when drift *exceeds* the tolerance. A variant that lands exactly
    on a threshold is therefore compliant. Client reports turn on this boundary,
    so it is asserted rather than left to the implementation.
    """

    CONFIG = AuditConfig(
        gaming_tolerance=0.5,
        min_degradation_drop=1.0,
        invariance_tolerance=0.5,
    )

    def _single_variant(
        self, variant_type: str, variant_score: float
    ) -> ScoreRecord:
        return ScoreRecord("System", "c1", "probe", variant_type, variant_score)

    def _audit(self, probe: ScoreRecord) -> bool:
        """Audit one probe variant alongside the required paired variants."""

        records = [
            ScoreRecord("System", "c1", "baseline", "baseline", 6.0),
            ScoreRecord("System", "c1", "pair_g", "gaming", 6.0),
            ScoreRecord("System", "c1", "pair_d", "degradation", 4.0),
            probe,
        ]
        result = audit_records(records, self.CONFIG)
        findings = {
            variant.variant_id: variant
            for case in result.cases
            for variant in case.variants
        }
        return findings["probe"].violated

    def test_gaming_gain_exactly_at_tolerance_is_not_a_violation(self) -> None:
        # Baseline 6.0, tolerance 0.5: a gain of exactly 0.5 is compliant.
        self.assertFalse(self._audit(self._single_variant("gaming", 6.5)))

    def test_gaming_gain_just_above_tolerance_is_a_violation(self) -> None:
        self.assertTrue(self._audit(self._single_variant("gaming", 6.51)))

    def test_degradation_drop_exactly_at_minimum_is_not_a_violation(self) -> None:
        # Baseline 6.0, minimum drop 1.0: a drop of exactly 1.0 is compliant.
        self.assertFalse(self._audit(self._single_variant("degradation", 5.0)))

    def test_degradation_drop_just_below_minimum_is_a_violation(self) -> None:
        self.assertTrue(self._audit(self._single_variant("degradation", 5.01)))

    def test_paraphrase_drift_exactly_at_tolerance_is_not_a_violation(self) -> None:
        # Baseline 6.0, tolerance 0.5: drift of exactly 0.5 is compliant.
        for score in (6.5, 5.5):
            with self.subTest(score=score):
                self.assertFalse(self._audit(self._single_variant("paraphrase", score)))

    def test_paraphrase_drift_just_above_tolerance_is_a_violation(self) -> None:
        for score in (6.51, 5.49):
            with self.subTest(score=score):
                self.assertTrue(self._audit(self._single_variant("paraphrase", score)))


class SeverityDirectionTests(unittest.TestCase):
    """Gate 6 requires positive severity change to mean 'candidate is worse'."""

    CONFIG = {
        "gaming_tolerance": 0.0,
        "min_degradation_drop": 1.0,
        "invariance_tolerance": 0.5,
    }

    def test_more_gaming_reward_raises_severity(self) -> None:
        worse = _severity(1.5, "gaming", self.CONFIG)
        better = _severity(0.2, "gaming", self.CONFIG)
        self.assertGreater(worse, better)

    def test_weaker_degradation_drop_raises_severity(self) -> None:
        # delta is variant minus baseline, so a smaller drop is a larger delta.
        weak_drop = _severity(-0.2, "degradation", self.CONFIG)
        strong_drop = _severity(-3.0, "degradation", self.CONFIG)
        self.assertGreater(weak_drop, strong_drop)

    def test_larger_paraphrase_drift_raises_severity(self) -> None:
        for worse_delta, better_delta in ((1.2, 0.1), (-1.2, -0.1)):
            with self.subTest(worse=worse_delta):
                self.assertGreater(
                    _severity(worse_delta, "paraphrase", self.CONFIG),
                    _severity(better_delta, "paraphrase", self.CONFIG),
                )


if __name__ == "__main__":
    unittest.main()
