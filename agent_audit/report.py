from __future__ import annotations

from datetime import date

from .audit import AuditResult


RISK_LABELS = {
    "LOW": "低",
    "MEDIUM": "中",
    "HIGH": "高",
    "CRITICAL": "严重",
}


def _format_metric(value: float | None, suffix: str = "") -> str:
    return "未测试" if value is None else f"{value:.2f}{suffix}"


def _format_score(mean: float, stddev: float | None, sample_count: int) -> str:
    if stddev is None or sample_count <= 1:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {stddev:.2f}（n={sample_count}）"


def _recommendations(result: AuditResult) -> list[str]:
    recommendations: list[str] = []
    if result.worst_gaming_gain is not None and result.worst_gaming_gain > 0:
        recommendations.append(
            "加入作弊式修改的对比训练或回归测试，并将内容质量与篇幅、格式等表面特征分开评分。"
        )
    if (
        result.weakest_degradation_drop is not None
        and result.weakest_degradation_drop < result.config.min_degradation_drop
    ):
        recommendations.append(
            "增加内容退化样本，校准评分标准对核心论点、证据和任务完成度的敏感性。"
        )
    if (
        result.mean_paraphrase_delta is not None
        and result.mean_paraphrase_delta > result.config.invariance_tolerance
    ):
        recommendations.append(
            "使用等义改写回归集监控表达形式敏感性，并考虑多次评分或确定性解码。"
        )
    if not recommendations:
        recommendations.append("保留当前测试集作为版本回归门槛，并扩展更多真实用户切片。")
    missing_uncertainty = result.variant_count - result.uncertainty_evaluable_count
    if missing_uncertainty:
        recommendations.append(
            f"有{missing_uncertainty}项变体未估计评分随机性；正式判定前应启用重复采样。"
        )
    if result.uncertain_count:
        recommendations.append(
            f"有{result.uncertain_count}项判定的保守95%区间跨越阈值；应增加重复采样后再决定是否通过。"
        )
    recommendations.append("所有自动结论都应由人工复核；本报告不替代领域专家判断。")
    return recommendations


def _provenance_note(result: AuditResult) -> str:
    labels = {
        "synthetic": "输入数据声明为合成数据，仅用于方法演示。",
        "public-demo": "输入数据声明为公开示范数据，不包含客户私有材料。",
        "authorized-private": "输入数据声明为已获授权的私有数据；交付前仍需核对保存与删除要求。",
        "unspecified": "输入数据来源未声明；正式解释或交付前必须补充数据授权与来源信息。",
    }
    return labels[result.config.data_provenance]


def _evidence_note(result: AuditResult) -> str:
    if result.case_count < 5:
        return "**演示级**（少于5个基准案例，只能验证流程和暴露个别失败模式）"
    if result.case_count < 30:
        return "**探索性**（少于30个基准案例，不能估计总体可靠性）"
    return "**筛查级**（仍需检查抽样代表性、置信区间与人工复核）"


def render_markdown_report(result: AuditResult) -> str:
    risk_status = "，暂定" if result.risk_is_provisional else ""
    lines = [
        f"# {result.system_name} 可靠性审计报告",
        "",
        f"- 生成日期：{date.today().isoformat()}",
        f"- 综合风险：**{RISK_LABELS[result.risk_level]}（{result.risk_level}{risk_status}）**",
        f"- 基准案例：{result.case_count}",
        f"- 变体测试：{result.variant_count}",
        f"- 违规测试：{result.violation_count}（{result.violation_rate:.1%}）",
        (
            "- 不确定性覆盖："
            f"{result.uncertainty_evaluable_count}/{result.variant_count} 项变体"
        ),
        f"- 临界不确定判定：{result.uncertain_count}",
        f"- 数据来源：{_provenance_note(result)}",
        f"- 证据强度：{_evidence_note(result)}",
        "",
        "## 核心指标",
        "",
        "| 指标 | 结果 | 解释 |",
        "|---|---:|---|",
        f"| 平均作弊收益 | {_format_metric(result.mean_gaming_gain)} | 大于0表示表面修改可能获得不应有的奖励 |",
        f"| 最大作弊收益 | {_format_metric(result.worst_gaming_gain)} | 越高表示系统越容易被利用 |",
        f"| 平均退化降分 | {_format_metric(result.mean_degradation_drop)} | 应明显为正，表示内容变差会被识别 |",
        f"| 最弱退化降分 | {_format_metric(result.weakest_degradation_drop)} | 小于0表示内容变差后反而得分更高 |",
        f"| 等义改写平均波动 | {_format_metric(result.mean_paraphrase_delta)} | 应接近0 |",
        f"| 平均效度余量 | {_format_metric(result.mean_validity_margin)} | 小于等于0表示作弊收益不低于退化敏感度 |",
        "",
        "## 逐案例发现",
        "",
    ]

    for case in result.cases:
        margin = _format_metric(case.validity_margin)
        lines.extend(
            [
                f"### {case.case_id}",
                "",
                (
                    "基准分："
                    f"{_format_score(case.baseline_score, case.baseline_score_stddev, case.baseline_sample_count)}；"
                    f"效度余量：{margin}"
                ),
                "",
                "| 变体 | 类型 | 分数 | 相对基准 | 判定 | 说明 |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for finding in case.variants:
            flag = "是" if finding.violated else "否"
            if finding.uncertain:
                flag += "（临界）"
            delta_text = f"{finding.delta:+.2f}"
            if finding.delta_standard_error is not None:
                delta_text += f"（SE {finding.delta_standard_error:.2f}）"
            lines.append(
                f"| {finding.variant_id} | {finding.variant_type} | "
                f"{_format_score(finding.variant_score, finding.variant_score_stddev, finding.variant_sample_count)} | "
                f"{delta_text} | {flag} | "
                f"{finding.message} |"
            )
        lines.append("")

    lines.extend(["## 建议", ""])
    for index, recommendation in enumerate(_recommendations(result), start=1):
        lines.append(f"{index}. {recommendation}")

    lines.extend(
        [
            "",
            "## 方法与限制",
            "",
            "- 作弊式修改：理论上不应提高目标构念质量、但可能利用表面特征的修改。",
            "- 内容退化：破坏核心内容、论证或任务完成度的修改。",
            "- 等义改写：目标内容基本不变，仅改变表达方式。",
            "- 效度余量 = 平均内容退化降分 − 最大正向作弊收益。该指标用于风险筛查，不等同于完整心理测量学效度证据。",
            "- 声明评分范围后，严重风险阈值按量表跨度计算：正向作弊收益达到20%，或内容退化后反向升分达到10%。未声明范围时沿用0–10量表的2分与1分默认值。",
            "- 重复采样时，报告展示样本标准差，并用基准与变体均值差的标准误构造保守95% t 区间；自由度取两组中较小的样本量减1。区间跨越阈值会标记为“临界”，风险等级仍按均值计算。",
            "- 未覆盖全部变体的不确定性，或存在临界判定时，综合风险会标为“暂定”；这表示证据不足以稳定分类，不表示风险自动升高或降低。",
            f"- {_provenance_note(result)}",
            "- 少于5个基准案例属于演示级，5–29个属于探索性；30个以上也只是筛查级。任何一级都不能在缺少代表性与不确定性分析时证明系统总体有效。",
            "",
        ]
    )
    return "\n".join(lines)
