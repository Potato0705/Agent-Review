from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


RISK_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
STATUS_LABELS = {
    "regression": "跨阈值回归",
    "improvement": "跨阈值改善",
    "worsened": "阈值内恶化",
    "improved": "阈值内改善",
    "unchanged": "无变化",
}
COMPARABLE_CONFIG_FIELDS = (
    "invariance_tolerance",
    "min_degradation_drop",
    "gaming_tolerance",
    "score_min",
    "score_max",
)
METRIC_FIELDS = (
    ("mean_gaming_gain", "平均作弊收益"),
    ("worst_gaming_gain", "最大作弊收益"),
    ("mean_degradation_drop", "平均退化降分"),
    ("weakest_degradation_drop", "最弱退化降分"),
    ("mean_paraphrase_delta", "等义改写平均波动"),
    ("mean_validity_margin", "平均效度余量"),
)


@dataclass(frozen=True)
class ComparisonFinding:
    case_id: str
    variant_id: str
    variant_type: str
    reference_delta: float
    candidate_delta: float
    severity_change: float
    reference_violated: bool
    candidate_violated: bool
    reference_uncertain: bool
    candidate_uncertain: bool
    change_is_provisional: bool
    status: str


@dataclass(frozen=True)
class ComparisonResult:
    reference_system: str
    candidate_system: str
    reference_model: str
    candidate_model: str
    input_sha256: str
    rubric_sha256: str
    temperature: float
    repeats: int
    reference_risk: str
    candidate_risk: str
    reference_risk_is_provisional: bool
    candidate_risk_is_provisional: bool
    risk_rank_change: int
    reference_violation_rate: float
    candidate_violation_rate: float
    violation_rate_change: float
    reference_uncertain_count: int
    candidate_uncertain_count: int
    variant_count: int
    regression_count: int
    improvement_count: int
    worsened_count: int
    improved_count: int
    unchanged_count: int
    provisional_change_count: int
    metric_values: dict[str, dict[str, float | None]]
    findings: tuple[ComparisonFinding, ...]

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1, **asdict(self)}


def load_audit_result(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise ValueError(f"Audit JSON does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Audit JSON is malformed: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Audit JSON must contain an object: {source}")
    return payload


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Audit result field {key!r} must be a non-empty string.")
    return value.strip()


def _required_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        raise ValueError(f"Audit result field {key!r} must be numeric.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Audit result field {key!r} must be numeric.") from exc
    if not math.isfinite(number):
        raise ValueError(f"Audit result field {key!r} must be finite.")
    return number


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Audit result field {key!r} must be an integer.")
    if value < 0:
        raise ValueError(f"Audit result field {key!r} must be non-negative.")
    return value


def _provisional(payload: dict[str, Any]) -> bool:
    value = payload.get("risk_is_provisional", True)
    if not isinstance(value, bool):
        raise ValueError("Audit result field 'risk_is_provisional' must be boolean.")
    return value


def _validate_schema(payload: dict[str, Any]) -> None:
    version = payload.get("schema_version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("Audit result schema_version must be an integer.")
    if version != 1:
        raise ValueError(f"Unsupported audit result schema_version: {version}.")


def _config(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("config")
    if not isinstance(value, dict):
        raise ValueError("Audit result is missing a valid config object.")
    return value


def _comparison_context(payload: dict[str, Any]) -> dict[str, Any]:
    context = payload.get("comparison_context")
    if not isinstance(context, dict):
        raise ValueError(
            "Audit result is missing comparison_context; regenerate it from a score CSV "
            "with its scoring manifest."
        )
    for key in ("input_sha256", "rubric_sha256"):
        value = context.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in value)
        ):
            raise ValueError(f"comparison_context field {key!r} is not a SHA-256 hash.")
    _required_number(context, "temperature")
    repeats = context.get("repeats")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("comparison_context field 'repeats' must be a positive integer.")
    _required_text(context, "model")
    return context


def _variant_map(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    cases = payload.get("cases")
    if not isinstance(cases, (list, tuple)) or not cases:
        raise ValueError("Audit result must contain a non-empty cases array.")
    variants: dict[tuple[str, str], dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each case in an audit result must be an object.")
        case_id = _required_text(case, "case_id")
        case_variants = case.get("variants")
        if not isinstance(case_variants, (list, tuple)):
            raise ValueError(f"Case {case_id!r} has no valid variants array.")
        for variant in case_variants:
            if not isinstance(variant, dict):
                raise ValueError(f"Case {case_id!r} contains a non-object variant.")
            variant_id = _required_text(variant, "variant_id")
            identity = (case_id, variant_id)
            if identity in variants:
                raise ValueError(f"Audit result contains duplicate variant {identity!r}.")
            variant_type = _required_text(variant, "variant_type")
            if variant_type not in {"gaming", "degradation", "paraphrase"}:
                raise ValueError(
                    f"Variant {identity!r} has unsupported type {variant_type!r}."
                )
            _required_number(variant, "delta")
            if not isinstance(variant.get("violated"), bool):
                raise ValueError(f"Variant {identity!r} has invalid violated flag.")
            if not isinstance(variant.get("uncertain"), bool):
                raise ValueError(f"Variant {identity!r} has invalid uncertain flag.")
            standard_error = variant.get("delta_standard_error")
            if standard_error is not None:
                parsed_standard_error = _required_number(
                    variant, "delta_standard_error"
                )
                if parsed_standard_error < 0:
                    raise ValueError(
                        f"Variant {identity!r} has negative delta_standard_error."
                    )
            variants[identity] = {**variant, "variant_type": variant_type}
    if not variants:
        raise ValueError("Audit result contains no comparable variants.")
    return variants


def _same_number(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(left_number) and math.isfinite(right_number) and math.isclose(
        left_number, right_number, rel_tol=0.0, abs_tol=1e-12
    )


def _validate_comparable(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    reference_config = _config(reference)
    candidate_config = _config(candidate)
    reference_context = _comparison_context(reference)
    candidate_context = _comparison_context(candidate)
    context_mismatches: list[str] = []
    for field in ("input_sha256", "rubric_sha256"):
        if str(reference_context[field]).lower() != str(candidate_context[field]).lower():
            context_mismatches.append(field)
    for field in ("temperature", "repeats"):
        if not _same_number(reference_context[field], candidate_context[field]):
            context_mismatches.append(field)
    if context_mismatches:
        raise ValueError(
            "Audit results do not share the same verified scoring context: "
            + ", ".join(context_mismatches)
            + "."
        )
    if reference_config.get("data_provenance") != candidate_config.get(
        "data_provenance"
    ):
        raise ValueError("Audit results use different data_provenance declarations.")
    if reference_config.get("variant_origin") != candidate_config.get(
        "variant_origin"
    ):
        raise ValueError("Audit results use different variant_origin declarations.")
    for field in (
        "invariance_tolerance",
        "min_degradation_drop",
        "gaming_tolerance",
    ):
        _required_number(reference_config, field)
        _required_number(candidate_config, field)
    for config_name, config in (
        ("reference", reference_config),
        ("candidate", candidate_config),
    ):
        score_min = config.get("score_min")
        score_max = config.get("score_max")
        if (score_min is None) != (score_max is None):
            raise ValueError(
                f"{config_name} audit must provide score_min and score_max together."
            )
        if score_min is not None and score_max is not None:
            parsed_min = _required_number(config, "score_min")
            parsed_max = _required_number(config, "score_max")
            if parsed_max <= parsed_min:
                raise ValueError(f"{config_name} audit has an invalid score range.")
    mismatched_fields = [
        field
        for field in COMPARABLE_CONFIG_FIELDS
        if not _same_number(reference_config.get(field), candidate_config.get(field))
    ]
    if mismatched_fields:
        raise ValueError(
            "Audit results use incompatible thresholds or score ranges: "
            + ", ".join(mismatched_fields)
            + "."
        )

    reference_variants = _variant_map(reference)
    candidate_variants = _variant_map(candidate)
    if set(reference_variants) != set(candidate_variants):
        missing = sorted(set(reference_variants).difference(candidate_variants))
        extra = sorted(set(candidate_variants).difference(reference_variants))
        raise ValueError(
            "Audit results must contain the same case/variant identities; "
            f"missing in candidate={missing}, extra in candidate={extra}."
        )
    type_mismatches = [
        identity
        for identity in reference_variants
        if reference_variants[identity]["variant_type"]
        != candidate_variants[identity]["variant_type"]
    ]
    if type_mismatches:
        raise ValueError(
            "Audit results assign different types to variants: "
            + ", ".join(f"{case_id}/{variant_id}" for case_id, variant_id in type_mismatches)
            + "."
        )
    return reference_variants, candidate_variants


def _severity(delta: float, variant_type: str, config: dict[str, Any]) -> float:
    if variant_type == "gaming":
        return delta - float(config["gaming_tolerance"])
    if variant_type == "degradation":
        return delta + float(config["min_degradation_drop"])
    return abs(delta) - float(config["invariance_tolerance"])


def compare_audits(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> ComparisonResult:
    _validate_schema(reference)
    _validate_schema(candidate)
    reference_variants, candidate_variants = _validate_comparable(reference, candidate)
    reference_context = _comparison_context(reference)
    candidate_context = _comparison_context(candidate)
    config = _config(reference)

    reference_risk = _required_text(reference, "risk_level")
    candidate_risk = _required_text(candidate, "risk_level")
    if reference_risk not in RISK_RANK or candidate_risk not in RISK_RANK:
        raise ValueError("Audit risk_level must be LOW, MEDIUM, HIGH, or CRITICAL.")

    findings: list[ComparisonFinding] = []
    counts = {status: 0 for status in STATUS_LABELS}
    provisional_change_count = 0
    for identity in sorted(reference_variants):
        reference_variant = reference_variants[identity]
        candidate_variant = candidate_variants[identity]
        variant_type = str(reference_variant["variant_type"])
        reference_delta = _required_number(reference_variant, "delta")
        candidate_delta = _required_number(candidate_variant, "delta")
        severity_change = _severity(candidate_delta, variant_type, config) - _severity(
            reference_delta, variant_type, config
        )
        reference_violated = bool(reference_variant["violated"])
        candidate_violated = bool(candidate_variant["violated"])
        reference_uncertain = bool(reference_variant["uncertain"])
        candidate_uncertain = bool(candidate_variant["uncertain"])
        change_is_provisional = (
            reference_variant.get("delta_standard_error") is None
            or candidate_variant.get("delta_standard_error") is None
            or reference_uncertain
            or candidate_uncertain
        )
        if change_is_provisional:
            provisional_change_count += 1
        if not reference_violated and candidate_violated:
            status = "regression"
        elif reference_violated and not candidate_violated:
            status = "improvement"
        elif severity_change > 1e-9:
            status = "worsened"
        elif severity_change < -1e-9:
            status = "improved"
        else:
            status = "unchanged"
        counts[status] += 1
        findings.append(
            ComparisonFinding(
                case_id=identity[0],
                variant_id=identity[1],
                variant_type=variant_type,
                reference_delta=reference_delta,
                candidate_delta=candidate_delta,
                severity_change=severity_change,
                reference_violated=reference_violated,
                candidate_violated=candidate_violated,
                reference_uncertain=reference_uncertain,
                candidate_uncertain=candidate_uncertain,
                change_is_provisional=change_is_provisional,
                status=status,
            )
        )

    metric_values: dict[str, dict[str, float | None]] = {}
    for field, _ in METRIC_FIELDS:
        reference_value = reference.get(field)
        candidate_value = candidate.get(field)
        parsed_reference = (
            None if reference_value is None else _required_number(reference, field)
        )
        parsed_candidate = (
            None if candidate_value is None else _required_number(candidate, field)
        )
        change = (
            None
            if parsed_reference is None or parsed_candidate is None
            else parsed_candidate - parsed_reference
        )
        metric_values[field] = {
            "reference": parsed_reference,
            "candidate": parsed_candidate,
            "change": change,
        }

    variant_count = len(findings)
    validated_summaries: dict[str, tuple[float, int]] = {}
    for label, payload, variants in (
        ("reference", reference, reference_variants),
        ("candidate", candidate, candidate_variants),
    ):
        declared_variant_count = _required_int(payload, "variant_count")
        declared_violation_count = _required_int(payload, "violation_count")
        declared_uncertain_count = _required_int(payload, "uncertain_count")
        violation_rate = _required_number(payload, "violation_rate")
        actual_violation_count = sum(
            bool(variant["violated"]) for variant in variants.values()
        )
        actual_uncertain_count = sum(
            bool(variant["uncertain"]) for variant in variants.values()
        )
        expected_rate = actual_violation_count / variant_count
        if declared_variant_count != variant_count:
            raise ValueError(
                f"{label} audit variant_count does not match its case findings."
            )
        if declared_violation_count != actual_violation_count:
            raise ValueError(
                f"{label} audit violation_count does not match its case findings."
            )
        if declared_uncertain_count != actual_uncertain_count:
            raise ValueError(
                f"{label} audit uncertain_count does not match its case findings."
            )
        if not 0 <= violation_rate <= 1 or not math.isclose(
            violation_rate, expected_rate, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"{label} audit violation_rate does not match its case findings."
            )
        validated_summaries[label] = (violation_rate, declared_uncertain_count)

    reference_violation_rate, reference_uncertain_count = validated_summaries[
        "reference"
    ]
    candidate_violation_rate, candidate_uncertain_count = validated_summaries[
        "candidate"
    ]

    return ComparisonResult(
        reference_system=_required_text(reference, "system_name"),
        candidate_system=_required_text(candidate, "system_name"),
        reference_model=_required_text(reference_context, "model"),
        candidate_model=_required_text(candidate_context, "model"),
        input_sha256=str(reference_context["input_sha256"]).lower(),
        rubric_sha256=str(reference_context["rubric_sha256"]).lower(),
        temperature=float(reference_context["temperature"]),
        repeats=int(reference_context["repeats"]),
        reference_risk=reference_risk,
        candidate_risk=candidate_risk,
        reference_risk_is_provisional=_provisional(reference),
        candidate_risk_is_provisional=_provisional(candidate),
        risk_rank_change=RISK_RANK[candidate_risk] - RISK_RANK[reference_risk],
        reference_violation_rate=reference_violation_rate,
        candidate_violation_rate=candidate_violation_rate,
        violation_rate_change=candidate_violation_rate - reference_violation_rate,
        reference_uncertain_count=reference_uncertain_count,
        candidate_uncertain_count=candidate_uncertain_count,
        variant_count=variant_count,
        regression_count=counts["regression"],
        improvement_count=counts["improvement"],
        worsened_count=counts["worsened"],
        improved_count=counts["improved"],
        unchanged_count=counts["unchanged"],
        provisional_change_count=provisional_change_count,
        metric_values=metric_values,
        findings=tuple(findings),
    )


def _format_value(value: float | None, *, percentage: bool = False) -> str:
    if value is None:
        return "未测试"
    return f"{value:.1%}" if percentage else f"{value:.2f}"


def render_comparison_report(result: ComparisonResult) -> str:
    reference_suffix = "，暂定" if result.reference_risk_is_provisional else ""
    candidate_suffix = "，暂定" if result.candidate_risk_is_provisional else ""
    lines = [
        "# 评分系统版本对比报告",
        "",
        f"- 参考系统：{result.reference_system}",
        f"- 候选系统：{result.candidate_system}",
        f"- 清单模型：`{result.reference_model}` → `{result.candidate_model}`",
        f"- 输入SHA-256：`{result.input_sha256}`",
        f"- 评分标准SHA-256：`{result.rubric_sha256}`",
        f"- 采样条件：温度 {result.temperature:g}，每条输入 {result.repeats} 次",
        (
            f"- 风险等级：{result.reference_risk}{reference_suffix} → "
            f"{result.candidate_risk}{candidate_suffix}"
        ),
        (
            "- 违规率："
            f"{result.reference_violation_rate:.1%} → {result.candidate_violation_rate:.1%} "
            f"（{result.violation_rate_change:+.1%}）"
        ),
        (
            "- 临界判定："
            f"{result.reference_uncertain_count} → {result.candidate_uncertain_count}"
        ),
        f"- 可比变体：{result.variant_count}",
        "",
        "## 变化摘要",
        "",
        f"- 跨阈值回归：{result.regression_count}",
        f"- 跨阈值改善：{result.improvement_count}",
        f"- 阈值内恶化：{result.worsened_count}",
        f"- 阈值内改善：{result.improved_count}",
        f"- 无变化：{result.unchanged_count}",
        f"- 暂定变化：{result.provisional_change_count}（任一侧未估计不确定性或属于临界判定）",
        "",
        "## 核心指标",
        "",
        "| 指标 | 参考系统 | 候选系统 | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for field, label in METRIC_FIELDS:
        values = result.metric_values[field]
        change = values["change"]
        change_text = "不可比" if change is None else f"{change:+.2f}"
        lines.append(
            f"| {label} | {_format_value(values['reference'])} | "
            f"{_format_value(values['candidate'])} | {change_text} |"
        )

    lines.extend(
        [
            "",
            "## 逐变体变化",
            "",
            "严重度变化统一为正数表示候选系统变差，负数表示改善。",
            "",
            "| 案例/变体 | 类型 | 参考差值 | 候选差值 | 严重度变化 | 变化判定 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    ordered = sorted(
        result.findings,
        key=lambda finding: (
            {"regression": 0, "worsened": 1, "improvement": 2, "improved": 3, "unchanged": 4}[
                finding.status
            ],
            -abs(finding.severity_change),
            finding.case_id,
            finding.variant_id,
        ),
    )
    for finding in ordered:
        status_text = STATUS_LABELS[finding.status]
        if finding.change_is_provisional:
            status_text += "（暂定）"
        lines.append(
            f"| {finding.case_id}/{finding.variant_id} | {finding.variant_type} | "
            f"{finding.reference_delta:+.2f} | {finding.candidate_delta:+.2f} | "
            f"{finding.severity_change:+.2f} | {status_text} |"
        )

    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- 本报告只比较案例、变体类型、阈值和量表完全一致的审计结果。",
            "- 风险等级和违规率下降不等同于候选系统总体更优；仍需检查代表性、临界判定和业务代价。",
            "- 严重度变化是描述性差异，不是两个系统差异的统计显著性检验。",
            "- 标为“暂定”的变化涉及未估计的不确定性或至少一侧临界判定，不能作为稳定改善或回归。",
            "- 自动对比必须由人工逐项复核，尤其是跨阈值回归。",
            "",
        ]
    )
    return "\n".join(lines)
