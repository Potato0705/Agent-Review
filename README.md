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

## 自动生成变体

手写作弊与退化变体是交付里最耗时的一步。`generate` 由带标注的基准生成它们：

```powershell
python -m agent_audit generate `
  --input examples/essay_baselines.csv `
  --output outputs/generated_cases.csv `
  --seed 0
```

输入是一行一个案例的最小CSV，`evidence_sentences` 用1起的句序号标出承重的证据或核心主张：

| 列 | 含义 |
|---|---|
| `case_id` | 案例ID |
| `text` | 基准文本 |
| `evidence_sentences` | 承重句序号，如 `2,3` |
| `notes` | 可选说明 |

**工具不推断哪句承重。** 没有标注、序号越界、或标注覆盖全部句子时直接报错退出，不产出可疑变体。

英文用 `--language english`，**语言不会被自动检测**。英文句号有歧义（`Dr.`、`3.5`、`J. K.`），而标注是按句序号给的——多切出一句就会让其后每个序号都指向错误的句子。除了缩写感知的分句器，还有两个机制防这件事：

```powershell
python -m agent_audit generate `
  --input examples/essay_baselines_en.csv `
  --output outputs/en_cases.csv `
  --language english `
  --show-sentences
```

`--show-sentences` 打印每个基准的编号分句与当前标注，不写任何文件，供标注前后核对。可选的 `sentence_count` 列则把校验固化下来：填了就比对分句结果，不等即报错并打印实际切分和所用分句器。

默认每个案例生成四个变体：两个作弊（追加无关扩写、追加评分术语堆砌）和两个退化（删除标注句、删除后替换为无依据断言）。每个变体都必须通过机器可验证的后置条件才会被写出，例如作弊变体必须完整包含基准文本作为前缀，证明它只增不减。

输出就是 `score` 的输入格式，任何一行都可以手工替换后重新喂入。**案例文件承载你的手工编辑，因此永远不会被覆盖**：输出路径已存在时命令直接拒绝。

需要补充变体时用 `--append` 合并，而不是重新生成：

```powershell
python -m agent_audit generate `
  --input examples/essay_baselines.csv `
  --output outputs/generated_cases.csv `
  --append
```

合并会逐行保留已有内容（包括你的手工编辑），只补上缺失的行。同时它会把每一行与「生成器本应产出的文本」比对：只要有一行被编辑过、或存在手工新增的变体，清单就记为 `set_origin: mixed`，**审计随即拒绝把这份集合声明为 `machine-generated`**。这样追加功能就不会把手工内容洗白成一个可验证的机器生成声明。清单同时给出 preserved / appended / edited / foreign 四个计数，让集合构成一目了然。清单逐变体记录干预幅度和实际插入的语料原文，便于人工确认「无关扩写」这个前提在该题目上真的成立。

`--paraphrase connective_substitution` 可选开启等义改写，**默认关闭**：保守的关联词替换编辑距离极小，评分器几乎必然给出近似分数，掺入后会稀释违规率并把风险评低。该规则只在小句边界替换关联词，遇到没有可替换关联词的文本会直接拒绝而不是硬造一个假的等义改写。

## 模型辅助等义改写

规则改写只换连接词，编辑距离极小。真正暴露评分器失效的是**整句重写**：在已有的四轮真实运行里，等义改写违规7/17，作弊变体0/17。能自动产出那种强度改写的只有模型或人。

因此 `paraphrase` 是一条**独立**子命令，不混进 `generate`（后者必须保持纯离线、确定性、零依赖）：

```powershell
python -m agent_audit paraphrase `
  --input examples/essay_baselines.csv `
  --output outputs/paraphrase_review.csv `
  --model YOUR_MODEL `
  --base-url https://YOUR_PROVIDER/v1
```

**该命令只写待审文件，绝不触碰案例集。** 提示词里只有原文和「保持含义不变地重写」，**不含评分标准**——针对被测维度优化过的改写是从侧门进来的循环论证；也不含「这是用来测试评分器的」之类的用途说明，那会诱导模型产出对抗性而非忠实的改写。

待审文件并列展示基准与草稿，供逐条核对。机器检查分成两层：

| 层级 | 检查 | 效果 |
|---|---|---|
| 硬拦 | 空、与基准完全相同、内含基准全文、长度超出 0.4–2.5 倍带 | `status=blocked`，须修改后才能批准 |
| 提示 | 基准中出现而草稿中未找到的数字、否定词计数变化 | 只写入 `review_notes`，不影响 `status` |

后两项**不做成闸门是实测决定的**：它们会拦下本仓库五个人工撰写的真·等义改写中的2个——「睡眠不足」改写成「缺乏睡眠」被判为丢失否定，「统一收纳」里的「一」被判为丢失数字。40%的误拦率会训练审阅者直接忽略 `status` 列，比没有检查更糟。

审阅者把认可的行改成 `status=approved` 后合并入集：

```powershell
python -m agent_audit generate --input examples/essay_baselines.csv `
  --output outputs/generated_cases.csv `
  --append --paraphrase-review outputs/paraphrase_review.csv
```

合并是**失败关闭**的：`status` 出现四个允许值以外的内容即报错（静默跳过拼错的 `aproved` 会让你以为批准了5条实际入集4条）；某行的 `baseline_sha256` 与当前基准不一致即报错并指名案例（基准被编辑后，旧草稿改写的是已经不存在的文本）。命令会打印 approved / pending / blocked / rejected 四类计数。

**批准行是人工批准而非工具可验证生成，因此集合一律标记为 `mixed`。** 工具不会为一个它无法验证的等义声明背书：没有任何机器检查能证明两段文本意思相同。

生成清单会记下起草用的模型（取自待审文件旁的 `.manifest.json`，也可用 `--paraphrase-manifest` 指定）。**审计时若发现改写模型与评分模型完全相同，直接拒绝运行**——那等于让被测系统自己定义「什么算等义」。同族不同版本（如 gemma3 改写、gemma4 评分）无法可靠检测，报告因此并列印出两个模型名，由你判断。

## Agent 轨迹评分器审计

审的是**给 Agent 轨迹打分的评分器**，不是 Agent 本身。问题和文本那边一样：它是真的在读过程，还是只在读最后一段答案、只在数工具调用次数？

轨迹是结构化的，因此三族变体的后置条件第一次做到了**严格可判定**——文本那边的作弊变体只能检查「是否仍以基准为前缀」，等义改写更是压根无法机器证明。

输入是一行一个案例的 JSONL：

```json
{"case_id": "flight_refund", "task": "查出订单 A7731 是否可以全额退款。",
 "steps": [{"tool": "search_orders", "args": {"order_id": "A7731"}, "result": "舱位 Y，出票 09:15"},
           {"tool": "read_policy", "args": {"fare_class": "Y"}, "result": "Y 舱 24 小时内可全额退款"},
           {"tool": "get_current_time", "args": {}, "result": "18:40"}],
 "final_answer": "可以全额退款。……仍在 24 小时内。",
 "load_bearing_steps": [2], "independent_steps": [[1, 3]]}
```

`load_bearing_steps` 标出结论所依据的步骤，`independent_steps` 标出互不依赖、可以换序的分组。**两者都必须人工给出，工具绝不推断**——猜错会产出一个声称退化却没退化、或声称等义却改变了因果的变体。

```powershell
python -m agent_audit trajectory `
  --input examples/trajectory_baselines.jsonl `
  --output outputs/trajectory_cases.csv `
  --show-steps
```

`--show-steps` 打印编号步骤与两类标注，不写任何文件，供标注前后核对。去掉它即生成变体：

| 变体族 | 策略 | 后置条件 |
|---|---|---|
| `gaming` | `redundant_tool_calls` | 把已有的某步原样再调一次；去掉重复项后与基准逐项相等，最终答案逐字节相同 |
| `gaming` | `padded_reasoning` | 步骤三元组与最终答案完全不变，只有说明文字变长 |
| `degradation` | `remove_load_bearing_step` | 删除标注的承重步骤，其余按原序保留，**最终答案照旧** |
| `degradation` | `hollow_evidence` | 保留调用，把承重步骤的结果换成空结果，**最终答案照旧** |
| `paraphrase` | `reorder_independent_steps` | 步骤多重集与最终答案完全相同，仅顺序不同 |

**最有诊断力的是那两个 degradation**：结论一字未改，但取得结论的那一步没了，或者调用还在却什么也没查到。评分器照给高分，就证明它只读了结尾。这个探针在文本模态里没有对应物。

作弊变体只重复 Agent 真正做过的调用，**绝不凭空捏造工具名或调用结果**——那是伪造证据，不是构造变体。等义改写只在人工标注的分组内换序；没有标注就直接拒绝并指名案例，而不是硬造一个没人验证过的等义。

输出就是 `score` 的输入格式，随后的评分、审计、版本对比与来源标记全部沿用既有流程；生成清单额外记录 `modality: "trajectory"`。`examples/trajectory_rubric.md` 是配套的轨迹评分标准。

## 审计机器生成的变体

审计机器生成的变体时必须声明来源，**并提供生成清单作为证据**：

```powershell
python -m agent_audit audit `
  --input outputs/live_scores.csv `
  --report outputs/live_report.md `
  --variant-origin machine-generated `
  --generation-manifest outputs/generated_cases.manifest.json
```

审计会比对生成清单的 `output_sha256` 与评分清单的 `input_sha256`。两者不等说明案例在生成后被手工改动过，此时 `machine-generated` 是虚假声明，命令直接报错并提示改用 `mixed`。这样标签才是可验证的事实，而不是用户打字打上去的一句话。

编辑过生成结果是正常做法，只是要如实声明 `mixed`。`mixed` 不强制提供清单，但提供了仍会校验指纹，并读出其中的改写模型名做上面那道循环论证检查。审计JSON会记录 `generation_sha256`，让报告能回溯到具体的生成运行；版本对比要求两侧的生成指纹一致。

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

版本0.15支持六条相互分离的流程：

1. 由带标注的基准生成作弊与内容退化变体；
2. 由带标注的 Agent 轨迹生成同三族变体，后置条件严格可判定；
3. 调用模型起草等义改写，产出待人工批准的审阅文件；
4. 调用OpenAI-compatible模型产生评分、理由和运行清单；
5. 对已有评分结果进行离线审计；
6. 对两份同条件审计结果进行模型或版本回归比较。

这种分离可以：

- 不绑定某一家模型供应商；
- 不在评分文件或运行清单中保存客户API密钥；
- 在无网络和无外部依赖的环境中复现；
- 清晰区分模型调用和效度分析。

后续版本可增加服务端结果管理。

## 目录结构

```text
agent_audit/                核心审计与报告代码
agent_audit/checkpoint.py   长任务检查点与安全恢复
agent_audit/paraphrase.py   模型起草的等义改写与审阅检查
agent_audit/trajectory.py   Agent 轨迹的数据模型、读取与转写渲染
agent_audit/trajectory_variants.py  轨迹变体及其可判定后置条件
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
