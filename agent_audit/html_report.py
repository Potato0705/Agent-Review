from __future__ import annotations

from html import escape

from .audit import AuditResult
from .comparison import (
    METRIC_FIELDS,
    STATUS_LABELS,
    ComparisonFinding,
    ComparisonResult,
)
from .report import RISK_LABELS


_STYLE = """
:root {
  color-scheme: light;
  --ink: #172033;
  --muted: #5f6b7a;
  --line: #dce3ea;
  --surface: #ffffff;
  --canvas: #f4f7fa;
  --brand: #155e75;
  --brand-soft: #e6f5f8;
  --danger: #b42318;
  --danger-soft: #fff0ee;
  --warning: #9a6700;
  --warning-soft: #fff7d6;
  --success: #067647;
  --success-soft: #eaf8f1;
  --shadow: 0 12px 30px rgba(23, 32, 51, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--canvas);
  color: var(--ink);
  font-family: Inter, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  line-height: 1.6;
}
.shell { max-width: 1180px; margin: 0 auto; padding: 40px 24px 72px; }
.hero {
  background: linear-gradient(135deg, #0f4658, #17758c);
  color: white;
  border-radius: 22px;
  padding: 34px 38px;
  box-shadow: var(--shadow);
}
.eyebrow { margin: 0 0 8px; font-size: 13px; letter-spacing: .12em; text-transform: uppercase; opacity: .76; }
h1 { margin: 0; font-size: clamp(28px, 4vw, 44px); line-height: 1.18; }
.subtitle { margin: 12px 0 0; max-width: 820px; opacity: .86; }
.grid { display: grid; gap: 16px; }
.summary { grid-template-columns: repeat(4, minmax(0, 1fr)); margin: 22px 0; }
.card, section {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: var(--shadow);
}
.card { padding: 18px 20px; }
.card .label { color: var(--muted); font-size: 13px; }
.card .value { margin-top: 4px; font-size: 25px; font-weight: 750; }
section { margin-top: 20px; padding: 24px; overflow: hidden; }
h2 { margin: 0 0 16px; font-size: 22px; }
h3 { margin: 0; font-size: 18px; }
.meta { color: var(--muted); font-size: 14px; }
.notice { margin-top: 16px; padding: 13px 15px; border-radius: 12px; background: var(--warning-soft); color: #6b4a00; }
.notice.info { background: var(--brand-soft); color: #164e63; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
.detail-table { min-width: 760px; }
th, td { border-bottom: 1px solid var(--line); padding: 11px 10px; text-align: left; vertical-align: top; }
th { color: var(--muted); font-size: 12px; letter-spacing: .03em; text-transform: uppercase; background: #f8fafc; }
tr:last-child td { border-bottom: 0; }
.number { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.badge { display: inline-block; padding: 3px 9px; border-radius: 999px; font-size: 12px; font-weight: 700; }
.badge.pass { background: var(--success-soft); color: var(--success); }
.badge.fail { background: var(--danger-soft); color: var(--danger); }
.badge.pending { background: var(--warning-soft); color: var(--warning); }
.badge.neutral { background: #eef2f6; color: #475467; }
.case { margin-top: 18px; border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }
.case-head { display: flex; justify-content: space-between; gap: 16px; padding: 15px 17px; background: #f8fafc; }
.case .table-wrap { padding: 0 8px 6px; }
ol, ul { margin: 0; padding-left: 22px; }
li + li { margin-top: 8px; }
code { font-family: "Cascadia Code", Consolas, monospace; font-size: .9em; overflow-wrap: anywhere; }
.hashes { display: grid; gap: 8px; }
.footer { margin-top: 22px; color: var(--muted); font-size: 12px; text-align: center; }
@media (max-width: 820px) {
  .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .shell { padding: 20px 12px 48px; }
  .hero { padding: 26px 22px; border-radius: 16px; }
  section { padding: 18px 14px; }
}
@media print {
  body { background: white; }
  .shell { max-width: none; padding: 0; }
  .hero, .card, section { box-shadow: none; break-inside: avoid; }
  .hero { border-radius: 0; }
}
"""


def _document(title: str, body: str) -> str:
    safe_title = escape(title, quote=True)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
  <title>{safe_title}</title>
  <style>{_STYLE}</style>
</head>
<body>
{body}
</body>
</html>
"""


def _fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "未测试"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def _fmt_score(mean: float, stddev: float | None, count: int) -> str:
    if stddev is None or count <= 1:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {stddev:.2f} (n={count})"


def _evidence_text(case_count: int) -> tuple[str, str]:
    if case_count < 5:
        return "演示级", "少于5个基准案例，只能验证流程和暴露个别失败模式。"
    if case_count < 30:
        return "探索性", "少于30个基准案例，不能估计总体可靠性。"
    return "筛查级", "仍需检查抽样代表性、置信区间与人工复核。"


def _provenance_text(value: str) -> str:
    return {
        "synthetic": "合成数据，仅用于方法演示。",
        "public-demo": "公开示范数据，不包含客户私有材料。",
        "authorized-private": "已获授权的私有数据，仍需遵循保存与删除要求。",
        "unspecified": "数据来源未声明，正式解释或交付前必须补充。",
    }[value]


def _audit_recommendations(result: AuditResult) -> list[str]:
    recommendations: list[str] = []
    if result.worst_gaming_gain is not None and result.worst_gaming_gain > 0:
        recommendations.append("加入作弊式修改回归测试，并分离内容质量与表面特征。")
    if (
        result.weakest_degradation_drop is not None
        and result.weakest_degradation_drop < result.config.min_degradation_drop
    ):
        recommendations.append("增加内容退化样本，校准系统对核心论点、证据和任务完成度的敏感性。")
    if (
        result.mean_paraphrase_delta is not None
        and result.mean_paraphrase_delta > result.config.invariance_tolerance
    ):
        recommendations.append("使用等义改写回归集监控表达形式敏感性。")
    missing = result.variant_count - result.uncertainty_evaluable_count
    if missing:
        recommendations.append(f"有{missing}项变体未估计评分随机性，正式判定前应重复采样。")
    if result.uncertain_count:
        recommendations.append(
            f"有{result.uncertain_count}项保守95%区间跨越阈值，应增加重复次数后复核。"
        )
    if not recommendations:
        recommendations.append("保留当前测试集作为版本回归门槛，并扩展真实用户切片。")
    recommendations.append("所有自动结论都应由人工复核；本报告不替代领域专家判断。")
    return recommendations


def render_audit_html(result: AuditResult) -> str:
    system = escape(result.system_name, quote=True)
    risk_label = escape(RISK_LABELS[result.risk_level])
    risk_suffix = "，暂定" if result.risk_is_provisional else ""
    evidence_label, evidence_description = _evidence_text(result.case_count)
    metric_rows = [
        ("平均作弊收益", result.mean_gaming_gain, "大于0表示表面修改获得不应有的奖励"),
        ("最大作弊收益", result.worst_gaming_gain, "越高表示系统越容易被利用"),
        ("平均退化降分", result.mean_degradation_drop, "应明显为正"),
        ("最弱退化降分", result.weakest_degradation_drop, "小于0表示退化后反而升分"),
        ("等义改写平均波动", result.mean_paraphrase_delta, "应接近0"),
        ("平均效度余量", result.mean_validity_margin, "小于等于0表示缺少正向安全边际"),
    ]
    metrics = "".join(
        "<tr>"
        f"<td>{escape(label)}</td><td class=\"number\">{_fmt(value)}</td>"
        f"<td>{escape(explanation)}</td></tr>"
        for label, value, explanation in metric_rows
    )

    case_blocks: list[str] = []
    for case in result.cases:
        rows: list[str] = []
        for finding in case.variants:
            if finding.violated:
                badge_class, decision = "fail", "违规"
            else:
                badge_class, decision = "pass", "通过"
            if finding.uncertain:
                badge_class, decision = "pending", f"{decision} · 临界"
            delta = f"{finding.delta:+.2f}"
            if finding.delta_standard_error is not None:
                delta += f" (SE {finding.delta_standard_error:.2f})"
            rows.append(
                "<tr>"
                f"<td><code>{escape(finding.variant_id, quote=True)}</code></td>"
                f"<td>{escape(finding.variant_type)}</td>"
                f"<td class=\"number\">{escape(_fmt_score(finding.variant_score, finding.variant_score_stddev, finding.variant_sample_count))}</td>"
                f"<td class=\"number\">{escape(delta)}</td>"
                f"<td><span class=\"badge {badge_class}\">{escape(decision)}</span></td>"
                f"<td>{escape(finding.message)}</td>"
                "</tr>"
            )
        case_blocks.append(
            "<article class=\"case\">"
            "<div class=\"case-head\">"
            f"<h3><code>{escape(case.case_id, quote=True)}</code></h3>"
            f"<div class=\"meta\">基准 {_fmt_score(case.baseline_score, case.baseline_score_stddev, case.baseline_sample_count)} · 效度余量 {_fmt(case.validity_margin)}</div>"
            "</div><div class=\"table-wrap\"><table class=\"detail-table\">"
            "<thead><tr><th>变体</th><th>类型</th><th class=\"number\">分数</th><th class=\"number\">相对基准</th><th>判定</th><th>说明</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div></article>"
        )

    recommendation_items = "".join(
        f"<li>{escape(item)}</li>" for item in _audit_recommendations(result)
    )
    provisional_notice = (
        '<div class="notice">当前风险等级为暂定：存在未覆盖的不确定性或临界判定。</div>'
        if result.risk_is_provisional
        else ""
    )
    body = f"""
<main class="shell">
  <header class="hero">
    <p class="eyebrow">Agent Review · Validity Audit</p>
    <h1>{system}</h1>
    <p class="subtitle">评分可靠性审计：检查作弊式修改、内容退化和等义改写敏感性。</p>
  </header>
  <div class="grid summary">
    <div class="card"><div class="label">综合风险</div><div class="value">{risk_label}（{escape(result.risk_level)}{risk_suffix}）</div></div>
    <div class="card"><div class="label">违规测试</div><div class="value">{result.violation_count}/{result.variant_count}</div></div>
    <div class="card"><div class="label">临界判定</div><div class="value">{result.uncertain_count}</div></div>
    <div class="card"><div class="label">证据强度</div><div class="value">{escape(evidence_label)}</div></div>
  </div>
  <section>
    <h2>执行摘要</h2>
    <p>违规率 <strong>{result.violation_rate:.1%}</strong>；不确定性覆盖 {result.uncertainty_evaluable_count}/{result.variant_count} 项变体。</p>
    <p class="meta">{escape(evidence_description)} 数据来源：{escape(_provenance_text(result.config.data_provenance))}</p>
    {provisional_notice}
  </section>
  <section>
    <h2>核心指标</h2>
    <div class="table-wrap"><table><thead><tr><th>指标</th><th class="number">结果</th><th>解释</th></tr></thead><tbody>{metrics}</tbody></table></div>
  </section>
  <section>
    <h2>逐案例发现</h2>
    {''.join(case_blocks)}
  </section>
  <section><h2>建议</h2><ol>{recommendation_items}</ol></section>
  <section>
    <h2>方法与限制</h2>
    <ul>
      <li>效度余量 = 平均内容退化降分 − 最大正向作弊收益，仅用于风险筛查。</li>
      <li>重复采样使用均值差标准误与保守95% t 区间；风险等级仍按均值计算。</li>
      <li>少于30个基准案例不能估计总体可靠性；自动结论必须人工复核。</li>
    </ul>
  </section>
  <p class="footer">Agent Review · 自包含报告 · 无外部脚本或网络资源</p>
</main>"""
    return _document(f"{result.system_name} 可靠性审计报告", body)


def _comparison_order(finding: ComparisonFinding) -> tuple[int, float, str, str]:
    rank = {
        "regression": 0,
        "worsened": 1,
        "improvement": 2,
        "improved": 3,
        "unchanged": 4,
    }[finding.status]
    return rank, -abs(finding.severity_change), finding.case_id, finding.variant_id


def render_comparison_html(result: ComparisonResult) -> str:
    metric_rows: list[str] = []
    for field, label in METRIC_FIELDS:
        values = result.metric_values[field]
        change = values["change"]
        metric_rows.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td class=\"number\">{_fmt(values['reference'])}</td>"
            f"<td class=\"number\">{_fmt(values['candidate'])}</td>"
            f"<td class=\"number\">{_fmt(change, signed=True) if change is not None else '不可比'}</td>"
            "</tr>"
        )

    finding_rows: list[str] = []
    for finding in sorted(result.findings, key=_comparison_order):
        label = STATUS_LABELS[finding.status]
        badge_class = {
            "regression": "fail",
            "worsened": "pending",
            "improvement": "pass",
            "improved": "pass",
            "unchanged": "neutral",
        }[finding.status]
        if finding.change_is_provisional:
            label += " · 暂定"
            badge_class = "pending"
        finding_rows.append(
            "<tr>"
            f"<td><code>{escape(finding.case_id, quote=True)}/{escape(finding.variant_id, quote=True)}</code></td>"
            f"<td>{escape(finding.variant_type)}</td>"
            f"<td class=\"number\">{finding.reference_delta:+.2f}</td>"
            f"<td class=\"number\">{finding.candidate_delta:+.2f}</td>"
            f"<td class=\"number\">{finding.severity_change:+.2f}</td>"
            f"<td><span class=\"badge {badge_class}\">{escape(label)}</span></td>"
            "</tr>"
        )

    reference_suffix = "，暂定" if result.reference_risk_is_provisional else ""
    candidate_suffix = "，暂定" if result.candidate_risk_is_provisional else ""
    body = f"""
<main class="shell">
  <header class="hero">
    <p class="eyebrow">Agent Review · Version Comparison</p>
    <h1>评分系统版本对比</h1>
    <p class="subtitle">{escape(result.reference_system, quote=True)} → {escape(result.candidate_system, quote=True)}</p>
  </header>
  <div class="grid summary">
    <div class="card"><div class="label">风险变化</div><div class="value">{escape(result.reference_risk)}{reference_suffix} → {escape(result.candidate_risk)}{candidate_suffix}</div></div>
    <div class="card"><div class="label">违规率变化</div><div class="value">{result.violation_rate_change:+.1%}</div></div>
    <div class="card"><div class="label">跨阈值回归</div><div class="value">{result.regression_count}</div></div>
    <div class="card"><div class="label">暂定变化</div><div class="value">{result.provisional_change_count}/{result.variant_count}</div></div>
  </div>
  <section>
    <h2>已验证的比较上下文</h2>
    <div class="hashes meta">
      <div>清单模型：<code>{escape(result.reference_model, quote=True)}</code> → <code>{escape(result.candidate_model, quote=True)}</code></div>
      <div>输入 SHA-256：<code>{escape(result.input_sha256)}</code></div>
      <div>评分标准 SHA-256：<code>{escape(result.rubric_sha256)}</code></div>
      <div>温度 {result.temperature:g}；每条输入 {result.repeats} 次；可比变体 {result.variant_count} 项。</div>
    </div>
    <div class="notice">风险或违规率下降不等同于总体更优；暂定变化不能作为稳定改善或回归。</div>
  </section>
  <section>
    <h2>变化摘要</h2>
    <p>跨阈值改善 {result.improvement_count}；阈值内恶化 {result.worsened_count}；阈值内改善 {result.improved_count}；无变化 {result.unchanged_count}。</p>
    <div class="table-wrap"><table><thead><tr><th>指标</th><th class="number">参考</th><th class="number">候选</th><th class="number">变化</th></tr></thead><tbody>{''.join(metric_rows)}</tbody></table></div>
  </section>
  <section>
    <h2>逐变体变化</h2>
    <p class="meta">严重度变化为正表示候选系统变差，负数表示改善。</p>
    <div class="table-wrap"><table class="detail-table"><thead><tr><th>案例/变体</th><th>类型</th><th class="number">参考差值</th><th class="number">候选差值</th><th class="number">严重度变化</th><th>判定</th></tr></thead><tbody>{''.join(finding_rows)}</tbody></table></div>
  </section>
  <section>
    <h2>解释限制</h2>
    <ul>
      <li>只比较运行指纹、案例、变体、阈值和量表一致的结果。</li>
      <li>严重度变化是描述性差异，不是统计显著性检验。</li>
      <li>所有跨阈值回归和暂定变化必须人工逐项复核。</li>
    </ul>
  </section>
  <p class="footer">Agent Review · 自包含报告 · 无外部脚本或网络资源</p>
</main>"""
    return _document("评分系统版本对比报告", body)
