# 文档导航

文档按用途分类，避免案例、客户材料、项目治理和生成物混在同一目录。

## 示例

- [合成审计说明](examples/example_audit_report.md)
- [自包含HTML审计示例](examples/example_audit_report.html)

HTML示例只使用合成数据，可通过 `scripts/build_public_demo.ps1` 确定性重建。

## 公开案例

- [Gemma 3 4B评分可靠性案例](case_studies/gemma3_case_study.md)
- [Gemma 3 4B与Gemma 4 E4B同条件对比](case_studies/gemma_version_comparison.md)
- [轨迹评分：证据缺失与可见标记（Gemma 3 4B 与 Gemma 4 E4B）](case_studies/gemma3_trajectory_case_study.md)

三篇案例均为5个基准的探索性方法示范，不是模型排行榜或生产认证。

## 服务与交付

- [7天可靠性体检](service/service_one_pager.md)
- [客户需求收集表](service/client_intake.md)
- [人工报告模板](service/report_template.md)
- [首周执行清单](service/first_week_playbook.md)

## 作品集

- [中英文简历、项目与面试表述](portfolio/portfolio_summary.md)

## 设计说明

- [自动变体生成设计](specs/variant_generation_design.md)
- [模型辅助等义改写设计](specs/model_assisted_paraphrase_design.md)
- [Agent 轨迹评分器审计设计](specs/agent_trajectory_audit_design.md)

设计说明记录已批准、待实现或已实现功能的取舍依据，便于回溯为什么选了这条路。

## 项目治理

- [质量审查门](governance/quality_gates.md)
- [实现与回归审查日志](governance/review_log.md)
- [仓库与发布规范](governance/repository_policy.md)
- [90天执行路线图](governance/roadmap_90_days.md)
