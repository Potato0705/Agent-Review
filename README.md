# Agent Review

一个面向 LLM 评分器与 AI 评测系统的轻量可靠性审计工具。它比较基准样本、作弊式修改、内容退化和等义改写的评分结果，帮助发现普通准确率或相关性指标看不到的风险。

[English](README_EN.md)

项目主页：[Potato0705/Agent-Review](https://github.com/Potato0705/Agent-Review)

## 它回答什么问题

- 增加无关但流畅的内容，是否能错误提高分数？
- 删除核心论点或证据后，系统是否会明显降分？
- 保持含义不变的改写，是否造成不合理波动？
- 系统对真实质量下降的敏感度，是否高于它对作弊式修改的奖励？

项目使用一个便于沟通的筛查指标：

```text
效度余量 = 平均内容退化降分 - 最大正向作弊收益
```

效度余量小于等于 0 表示：系统奖励表面修改的幅度，不低于它识别内容退化的幅度。该指标用于风险筛查，不替代完整的效度论证。

## 快速开始

本项目只依赖 Python 标准库，Python 3.10 及以上即可运行。

```powershell
python -m agent_audit audit `
  --input examples/demo_scores.csv `
  --report outputs/demo_report.md `
  --json outputs/demo_result.json `
  --html outputs/demo_report.html `
  --score-min 0 `
  --score-max 10 `
  --data-provenance synthetic
```

运行测试：

```powershell
python -m unittest discover -s tests -v
```

检查逐模块行覆盖率下限（只使用标准库 `trace`）：

```powershell
python scripts/coverage_report.py
```

运行完整本地审查门（编译、测试、覆盖率、离线演示与空白检查）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/review.ps1
```

推送版本前执行包含文件、远端和单一贡献者身份检查的发布门：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/release_check.ps1 -RequireClean
```

也可以安装为命令行工具：

```powershell
python -m pip install -e .
agent-audit audit --input examples/demo_scores.csv --report outputs/demo_report.md
```

## 使用真实的 OpenAI-compatible 模型评分

项目可以调用实现 `/chat/completions` 接口的模型服务。API密钥只从环境变量读取，不接受命令行明文参数。

```powershell
$env:OPENAI_API_KEY = "你的密钥"

python -m agent_audit score `
  --input examples/essay_cases.csv `
  --output outputs/live_scores.csv `
  --rubric-file examples/essay_rubric.md `
  --model YOUR_MODEL_NAME `
  --base-url https://YOUR_PROVIDER/v1 `
  --system-name "YOUR_MODEL_NAME + rubric-v1" `
  --score-min 0 `
  --score-max 10 `
  --temperature 0.2 `
  --repeats 3 `
  --raw-output outputs/live_raw.jsonl `
  --checkpoint outputs/live_checkpoint.jsonl

python -m agent_audit audit `
  --input outputs/live_scores.csv `
  --report outputs/live_report.md `
  --json outputs/live_result.json `
  --html outputs/live_report.html `
  --score-min 0 `
  --score-max 10 `
  --data-provenance public-demo
```

评分命令会在输出CSV旁自动生成运行清单，例如 `live_scores.manifest.json`，其中保存模型、端点、参数、重复次数、输入哈希、评分标准哈希、调用延迟和可用的令牌汇总，但不保存API密钥。`--repeats 3` 会把每条输入独立评分3次，同时把调用量和费用约放大3倍。

`--raw-output` 是可选项。重复采样时，聚合CSV的 `notes` 只明确保留第一个样本的理由；需要复核全部理由时才启用原始JSONL，并按敏感数据存储。项目默认忽略 `outputs/` 下的所有文件，防止产物被误提交。

## 长任务检查点与恢复

`--checkpoint` 会在每个样本完成后同步写入JSONL。若进程、网络或供应商在长任务中断，可使用完全相同的命令并增加 `--resume`：

```powershell
python -m agent_audit score `
  --input examples/essay_cases.csv `
  --output outputs/live_scores.csv `
  --rubric-file examples/essay_rubric.md `
  --model YOUR_MODEL_NAME `
  --base-url https://YOUR_PROVIDER/v1 `
  --system-name "YOUR_MODEL_NAME + rubric-v1" `
  --score-min 0 --score-max 10 `
  --temperature 0.2 --repeats 3 `
  --checkpoint outputs/live_checkpoint.jsonl `
  --resume
```

恢复前会核对系统名、供应商、模型、端点、评分范围、温度、重复次数、输入哈希和评分标准哈希。新运行不会覆盖已有检查点；只有明确使用 `--resume` 才会读取它。检查点包含完整模型回复，敏感程度与 `--raw-output` 相同，应限制访问并按数据保留政策删除。不要让多个进程同时写入同一个检查点。

若异常退出只留下一个没有换行符的末尾JSON片段，恢复器会在验证运行上下文和所有完整样本后修剪该片段；任何已换行的畸形记录、重复样本或上下文不匹配都会直接报错，不会静默跳过。

若使用本机的OpenAI兼容服务，可以使用 `http://localhost/...` 或 `http://127.0.0.1/...`，本地端点不强制设置密钥。非本机端点必须使用HTTPS。

## 输入格式

CSV 至少包含以下字段：

| 字段 | 含义 |
|---|---|
| `system_name` | 被审计系统名称；单次运行只能包含一个系统 |
| `case_id` | 同一原始样本及其所有变体的分组ID |
| `variant_id` | 变体的唯一名称 |
| `variant_type` | `baseline`、`gaming`、`degradation` 或 `paraphrase` |
| `score` | 系统给出的数值分数 |
| `notes` | 可选说明 |
| `score_stddev` | 可选样本标准差；`sample_count > 1` 时必填 |
| `sample_count` | 聚合的独立评分次数，默认为1 |

每个 `case_id` 必须恰好有一个 `baseline`。可以从 [`examples/score_input_template.csv`](examples/score_input_template.csv) 开始填写。

为了让效度余量有可解释的成对对照，每个案例还必须至少包含一个 `gaming` 和一个 `degradation` 变体；`paraphrase` 可选但推荐。

无需运行代码也可以先查看[合成示例审计报告](docs/examples/example_audit_report.md)。

也可以下载并离线打开[自包含HTML示例](docs/examples/example_audit_report.html)。该文件由 `scripts/build_public_demo.ps1` 从合成数据生成，不需要服务器或网络连接。

首个真实本地模型示范见：[Gemma 3 4B 评分可靠性案例](docs/case_studies/gemma3_case_study.md)。该案例包含5个基准文本，明确属于探索性证据。

同条件版本对比见：[Gemma 3 4B 与 Gemma 4 E4B 对比](docs/case_studies/gemma_version_comparison.md)。两侧各包含60次评分，并明确区分按均值观察到的改善与暂定变化。

## 默认判定规则

- `gaming`：相对基准分数上升即记为违规。
- `degradation`：相对基准降分不足 1.0 即记为违规。
- `paraphrase`：绝对波动超过 0.5 即记为违规。

阈值均可通过命令行修改：

```powershell
python -m agent_audit audit `
  --input your_scores.csv `
  --report outputs/your_report.md `
  --gaming-tolerance 0.2 `
  --min-degradation-drop 1.5 `
  --invariance-tolerance 0.4
```

默认阈值按0–10分量表设计。使用0–5、0–100或其他量表时，必须根据评分单位调整阈值，并通过 `--score-min`、`--score-max` 声明合法范围。不同量表的原始效度余量不能直接比较。

声明评分范围后，严重风险阈值会随量表缩放：正向作弊收益达到量表跨度的20%，或者内容退化后反向升分达到量表跨度的10%。

`--data-provenance` 用于区分 `synthetic`、`public-demo`、`authorized-private` 和 `unspecified`。来源未声明的报告只能作为内部草稿，不能正式交付。

`--variant-origin` 用于区分 `human-authored`、`machine-generated`、`mixed` 和 `unspecified`。机器生成的变体属于下限测试：没通过是确凿证据，通过不能证明系统可靠，报告会明确写出这一点。版本对比要求两侧声明一致。

## 重复采样与临界判定

当基准和变体都经过重复评分时，工具根据两组均值差的标准误构造保守95% t 区间。为避免小样本过度自信，自由度取两组中较小样本量减1。区间跨越对应判定阈值时，该项标为“临界”。

风险等级仍由平均分计算，但以下任一情况会把等级标为“暂定”：

- 有变体未覆盖重复采样不确定性；
- 至少一项判定的保守区间跨越阈值。

“暂定”表示证据不足以稳定分类，并不表示风险自动升高或降低。3次重复适合暴露明显随机性，不足以替代更大样本或人工复核。

## 模型或版本对比

先分别生成两份审计JSON，再运行：

```powershell
python -m agent_audit compare `
  --reference outputs/reference_audit.json `
  --candidate outputs/candidate_audit.json `
  --report outputs/version_comparison.md `
  --json outputs/version_comparison.json `
  --html outputs/version_comparison.html
```

审计命令会自动读取评分CSV旁的 `.manifest.json`，也可以用 `--manifest` 显式指定，并把输入哈希、评分标准哈希、温度和重复次数写入审计JSON。比较工具会拒绝这些运行指纹，以及案例集合、变体类型、判定阈值或评分范围不一致的结果。严重度变化统一为正数表示候选系统变差、负数表示改善；任何涉及临界判定或未估计随机性的变化都会标为“暂定”。该命令用于同条件回归审查，不支持把不同数据集上的结果拼成模型排行榜。

`--html` 为可选输出。生成的审计或版本对比页面把样式内嵌在单个文件中，不需要JavaScript、外部字体或网络资源，适合直接发送、归档和打印。报告对系统名、案例ID、模型回复说明等不可信文本进行HTML转义，并设置限制性内容安全策略；窄屏下详细表格保持可读宽度并允许横向滚动。

## 当前范围

版本0.9支持三条相互分离的流程：

1. 调用OpenAI-compatible模型产生评分、理由和运行清单；
2. 对已有评分结果进行离线审计；
3. 对两份同条件审计结果进行模型或版本回归比较。

这种分离可以：

- 不绑定某一家模型供应商；
- 不在评分文件或运行清单中保存客户API密钥；
- 在无网络和无外部依赖的环境中复现；
- 清晰区分模型调用和效度分析。

后续版本可增加自动变体生成、Agent轨迹评测和服务端结果管理。

## 目录结构

```text
agent_audit/                核心审计与报告代码
agent_audit/checkpoint.py   长任务检查点与安全恢复
agent_audit/html_report.py  自包含HTML审计与对比报告
examples/                   合成示例和输入模板
tests/                      标准库 unittest 测试
scripts/                    本地审查、公开资产构建与发布门脚本
.github/                    CI工作流与公开需求模板
docs/README.md              分类文档导航
docs/examples/              可公开的合成示例
docs/case_studies/          真实模型探索性案例
docs/service/               客户需求、范围与交付模板
docs/portfolio/             简历、作品集与面试材料
docs/governance/            质量门、审查日志、路线图与仓库规范
outputs/                    本地生成结果（默认不提交）
```

## 数据与伦理

- `examples/demo_scores.csv` 是合成演示，不代表任何真实产品表现。
- 未经授权，不应将客户私有数据上传到公共模型服务。
- `--raw-output` 会保存模型原始回复，可能包含敏感文本；只有在得到授权并设置安全存储后才应启用。
- 运行清单保存输入和评分标准的SHA-256哈希，不保存原始输入或API密钥。
- 自动报告必须经过人工复核。
- 本工具提供风险筛查，不提供法律、监管或心理测量认证。

## 服务与联系

如需评估LLM评分器、自动反馈系统或评测流程，可先提交一份不含敏感数据的[审计需求](https://github.com/Potato0705/Agent-Review/issues/new?template=audit-request.md)。公开Issue中不要粘贴API密钥、私有样本、个人信息、保密提示词或客户名称；确认范围后再约定私密材料传输方式。

## License

MIT
