from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import fmean
from typing import Iterable

from .models import CaseFinding, ScoreRecord, VariantFinding


@dataclass(frozen=True)
class AuditConfig:
    invariance_tolerance: float = 0.5
    min_degradation_drop: float = 1.0
    gaming_tolerance: float = 0.0
    score_min: float | None = None
    score_max: float | None = None
    data_provenance: str = "unspecified"

    def validate(self) -> None:
        numeric_fields = {
            "invariance_tolerance": self.invariance_tolerance,
            "min_degradation_drop": self.min_degradation_drop,
            "gaming_tolerance": self.gaming_tolerance,
        }
        for name, value in numeric_fields.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite, non-negative number.")
        if (self.score_min is None) != (self.score_max is None):
            raise ValueError("score_min and score_max must be provided together.")
        if self.score_min is not None and self.score_max is not None:
            if not math.isfinite(self.score_min) or not math.isfinite(self.score_max):
                raise ValueError("score_min and score_max must be finite.")
            if self.score_max <= self.score_min:
                raise ValueError("score_max must be greater than score_min.")
        allowed_provenance = {
            "synthetic",
            "public-demo",
            "authorized-private",
            "unspecified",
        }
        if self.data_provenance not in allowed_provenance:
            raise ValueError(
                f"data_provenance must be one of {sorted(allowed_provenance)}."
            )


@dataclass(frozen=True)
class AuditResult:
    system_name: str
    risk_level: str
    case_count: int
    variant_count: int
    violation_count: int
    violation_rate: float
    risk_is_provisional: bool
    uncertainty_evaluable_count: int
    uncertain_count: int
    mean_gaming_gain: float | None
    worst_gaming_gain: float | None
    mean_degradation_drop: float | None
    weakest_degradation_drop: float | None
    mean_paraphrase_delta: float | None
    mean_validity_margin: float | None
    cases: tuple[CaseFinding, ...]
    config: AuditConfig

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1, **asdict(self)}


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _delta_standard_error(baseline: ScoreRecord, variant: ScoreRecord) -> float | None:
    if (
        baseline.score_stddev is None
        or variant.score_stddev is None
        or baseline.sample_count <= 1
        or variant.sample_count <= 1
    ):
        return None
    return math.sqrt(
        (baseline.score_stddev**2 / baseline.sample_count)
        + (variant.score_stddev**2 / variant.sample_count)
    )


def _conservative_critical_value(
    baseline_sample_count: int, variant_sample_count: int
) -> float:
    """Return a conservative two-sided 95% t critical value.

    We intentionally use the smaller sample's degrees of freedom rather than a
    Welch-Satterthwaite estimate. That slightly widens the interval and avoids
    presenting tiny repeated samples as more certain than they are.
    """

    t_critical_95 = (
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.080,
        2.074,
        2.069,
        2.064,
        2.060,
        2.056,
        2.052,
        2.048,
        2.045,
        2.042,
    )
    degrees_of_freedom = min(
        baseline_sample_count - 1, variant_sample_count - 1
    )
    if degrees_of_freedom <= 0:
        raise ValueError("Repeated samples must provide positive degrees of freedom.")
    if degrees_of_freedom <= len(t_critical_95):
        return t_critical_95[degrees_of_freedom - 1]
    return 1.96


def _risk_level(
    violation_rate: float,
    worst_gaming_gain: float | None,
    weakest_degradation_drop: float | None,
    mean_validity_margin: float | None,
    config: AuditConfig,
) -> str:
    if config.score_min is not None and config.score_max is not None:
        score_span = config.score_max - config.score_min
        critical_gaming_gain = score_span * 0.20
        critical_degradation_reversal = -(score_span * 0.10)
    else:
        critical_gaming_gain = 2.0
        critical_degradation_reversal = -1.0

    if (
        worst_gaming_gain is not None
        and worst_gaming_gain >= critical_gaming_gain
    ) or (
        weakest_degradation_drop is not None
        and weakest_degradation_drop <= critical_degradation_reversal
    ):
        return "CRITICAL"
    if violation_rate >= 0.5 or (
        mean_validity_margin is not None and mean_validity_margin <= 0.0
    ):
        return "HIGH"
    if violation_rate >= 0.25:
        return "MEDIUM"
    return "LOW"


def audit_records(
    records: Iterable[ScoreRecord], config: AuditConfig | None = None
) -> AuditResult:
    cfg = config or AuditConfig()
    cfg.validate()
    rows = list(records)
    if not rows:
        raise ValueError("At least one score record is required.")

    systems = {row.system_name for row in rows}
    if len(systems) != 1:
        raise ValueError(
            "One audit run must contain exactly one system_name; "
            f"found: {', '.join(sorted(systems))}."
        )
    system_name = next(iter(systems))

    for row in rows:
        if row.sample_count < 1:
            raise ValueError(
                f"sample_count for {row.case_id}/{row.variant_id} must be at least 1."
            )
        if row.score_stddev is not None and (
            not math.isfinite(row.score_stddev) or row.score_stddev < 0
        ):
            raise ValueError(
                f"score_stddev for {row.case_id}/{row.variant_id} must be finite and non-negative."
            )
        if row.sample_count > 1 and row.score_stddev is None:
            raise ValueError(
                f"score_stddev is required for repeated score {row.case_id}/{row.variant_id}."
            )
        if row.sample_count == 1 and row.score_stddev not in {None, 0.0}:
            raise ValueError(
                f"score_stddev must be empty or 0 for single score {row.case_id}/{row.variant_id}."
            )

    if cfg.score_min is not None and cfg.score_max is not None:
        for row in rows:
            if not cfg.score_min <= row.score <= cfg.score_max:
                raise ValueError(
                    f"Score {row.score} for {row.case_id}/{row.variant_id} is outside "
                    f"the configured range [{cfg.score_min}, {cfg.score_max}]."
                )

    grouped: dict[str, list[ScoreRecord]] = {}
    for row in rows:
        grouped.setdefault(row.case_id, []).append(row)

    case_findings: list[CaseFinding] = []
    gaming_gains: list[float] = []
    degradation_drops: list[float] = []
    paraphrase_deltas: list[float] = []
    validity_margins: list[float] = []
    violation_count = 0
    uncertain_count = 0
    uncertainty_evaluable_count = 0
    variant_count = 0

    for case_id in sorted(grouped):
        case_rows = grouped[case_id]
        variant_ids = [row.variant_id for row in case_rows]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError(f"Case {case_id!r} contains duplicate variant_id values.")
        baselines = [row for row in case_rows if row.variant_type == "baseline"]
        if len(baselines) != 1:
            raise ValueError(
                f"Case {case_id!r} must have exactly one baseline; found {len(baselines)}."
            )
        present_types = {row.variant_type for row in case_rows}
        missing_paired_types = {"gaming", "degradation"}.difference(present_types)
        if missing_paired_types:
            raise ValueError(
                f"Case {case_id!r} is missing paired validity variants: "
                f"{', '.join(sorted(missing_paired_types))}."
            )
        baseline = baselines[0]
        findings: list[VariantFinding] = []
        case_gaming_gains: list[float] = []
        case_degradation_drops: list[float] = []

        for row in case_rows:
            if row.variant_type == "baseline":
                continue
            variant_count += 1
            raw_delta = row.score - baseline.score
            delta_standard_error = _delta_standard_error(baseline, row)
            if delta_standard_error is not None:
                uncertainty_evaluable_count += 1

            if row.variant_type == "gaming":
                gain = raw_delta
                gaming_gains.append(gain)
                case_gaming_gains.append(gain)
                threshold_contrast = gain - cfg.gaming_tolerance
                violated = threshold_contrast > 0
                message = (
                    f"作弊式修改令分数上升 {gain:+.2f}"
                    if violated
                    else f"作弊式修改未获得额外分数（{gain:+.2f}）"
                )
            elif row.variant_type == "degradation":
                drop = baseline.score - row.score
                degradation_drops.append(drop)
                case_degradation_drops.append(drop)
                threshold_contrast = drop - cfg.min_degradation_drop
                violated = threshold_contrast < 0
                message = (
                    f"内容退化的降分为 {drop:.2f}，低于阈值 {cfg.min_degradation_drop:.2f}"
                    if violated
                    else f"内容退化令分数下降 {drop:.2f}"
                )
            else:
                absolute_delta = abs(raw_delta)
                paraphrase_deltas.append(absolute_delta)
                threshold_contrast = absolute_delta - cfg.invariance_tolerance
                violated = threshold_contrast > 0
                message = (
                    f"等义改写造成 {absolute_delta:.2f} 分波动，超过容忍值"
                    if violated
                    else f"等义改写波动为 {absolute_delta:.2f}"
                )

            uncertain = bool(
                delta_standard_error is not None
                and delta_standard_error > 0
                and abs(threshold_contrast)
                <= _conservative_critical_value(
                    baseline.sample_count, row.sample_count
                )
                * delta_standard_error
            )
            if uncertain:
                message += "（保守95%区间跨越判定阈值）"
                uncertain_count += 1

            if violated:
                violation_count += 1
            findings.append(
                VariantFinding(
                    case_id=case_id,
                    variant_id=row.variant_id,
                    variant_type=row.variant_type,
                    baseline_score=baseline.score,
                    variant_score=row.score,
                    delta=raw_delta,
                    violated=violated,
                    uncertain=uncertain,
                    delta_standard_error=delta_standard_error,
                    variant_score_stddev=row.score_stddev,
                    variant_sample_count=row.sample_count,
                    message=message,
                )
            )

        validity_margin: float | None = None
        if case_gaming_gains and case_degradation_drops:
            degradation_sensitivity = fmean(case_degradation_drops)
            rewarded_gaming = max(0.0, max(case_gaming_gains))
            validity_margin = degradation_sensitivity - rewarded_gaming
            validity_margins.append(validity_margin)

        case_findings.append(
            CaseFinding(
                case_id=case_id,
                baseline_score=baseline.score,
                baseline_score_stddev=baseline.score_stddev,
                baseline_sample_count=baseline.sample_count,
                variants=tuple(findings),
                validity_margin=validity_margin,
            )
        )

    if variant_count == 0:
        raise ValueError("The audit requires at least one non-baseline variant.")

    violation_rate = violation_count / variant_count
    worst_gaming_gain = max(gaming_gains) if gaming_gains else None
    weakest_degradation_drop = min(degradation_drops) if degradation_drops else None
    mean_validity_margin = _mean(validity_margins)

    return AuditResult(
        system_name=system_name,
        risk_level=_risk_level(
            violation_rate,
            worst_gaming_gain,
            weakest_degradation_drop,
            mean_validity_margin,
            cfg,
        ),
        case_count=len(case_findings),
        variant_count=variant_count,
        violation_count=violation_count,
        violation_rate=violation_rate,
        risk_is_provisional=(
            uncertainty_evaluable_count < variant_count or uncertain_count > 0
        ),
        uncertainty_evaluable_count=uncertainty_evaluable_count,
        uncertain_count=uncertain_count,
        mean_gaming_gain=_mean(gaming_gains),
        worst_gaming_gain=worst_gaming_gain,
        mean_degradation_drop=_mean(degradation_drops),
        weakest_degradation_drop=weakest_degradation_drop,
        mean_paraphrase_delta=_mean(paraphrase_deltas),
        mean_validity_margin=mean_validity_margin,
        cases=tuple(case_findings),
        config=cfg,
    )
