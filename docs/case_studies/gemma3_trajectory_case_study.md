# Gemma 3 4B 轨迹评分可靠性：证据缺失与可见标记

> 这是探索性方法示范，不是对 Gemma 系列的总体评价。五条合成轨迹不足以估计总体可靠性。

## 为什么做这个测试

给 Agent 轨迹打分的评分器，声称评价的是「解决任务的过程质量」。本案例问一个更窄、更可证伪的问题：**当结论所依赖的那一步证据消失时，评分器会不会降分？**

设计这套探针时我的预期是「评分器只读最后一段答案」。实测结果比这个说法更具体，也更有用。

## 设置

- 日期：2026-09-21；
- 服务：本机 Ollama 的 OpenAI-compatible 接口；
- 模型：`gemma3:4b`；
- 温度：0.2；重复采样：每条输入 3 次；
- 数据：5 条合成轨迹，每条含 2 个作弊、2 个退化、1 个等义改写变体，共 30 条输入、90 次评分；
- 评分标准：[trajectory_rubric.md](../../examples/trajectory_rubric.md)；
- 轨迹基准：[trajectory_baselines.jsonl](../../examples/trajectory_baselines.jsonl)；
- 生成清单产出哈希：`217e100aeebe840bc9d6537b892d15c8f353d2db4fdf99299bfbf5c47a6a3164`；
- 评分清单输入哈希：同上（两者相等即证明评分的就是生成的那批案例）；
- 评分标准哈希：`3c1a817a00863609b1647715575283e49ac986225531bc94f7f5eca848bdc8e3`。

变体标签没有发送给模型。两个退化变体都**原样保留最终答案**——只改过程不改结论，才能区分评分器是在读过程还是在读结论。

## 结果

| 指标 | 结果 |
|---|---:|
| 综合风险 | 严重（CRITICAL，暂定） |
| 变体测试 | 25 |
| 违规测试 | 7（28.0%） |
| 平均作弊收益 | −0.43 |
| 平均退化降分 | 0.53 |
| 最弱退化降分 | −1.00 |
| 等义改写平均波动 | 0.20 |

逐策略（基准均分 8.20，每格 n=5）：

| 变体 | 平均 Δ | 最小 | 最大 |
|---|---:|---:|---:|
| 退化 / `hollow_evidence`（调用在，结果为空） | **−1.07** | −2.00 | −0.33 |
| 退化 / `remove_load_bearing_step`（调用整个删除） | **0.00** | −1.00 | +1.00 |
| 作弊 / `padded_reasoning`（步骤说明注水） | −0.80 | −1.00 | 0.00 |
| 作弊 / `redundant_tool_calls`（重复已有调用） | −0.07 | −1.00 | +0.67 |
| 等义 / `reorder_independent_steps`（独立步骤换序） | +0.20 | 0.00 | +1.00 |

## 主要发现：评分器认的是可见标记，不是证据是否存在

两个退化变体拿掉的是**同一条证据**，差别只在这条证据是「整步消失」还是「调用还在但返回空」。结果相差 1.07 分。

以 `flight_refund` 为例。基准轨迹里，「Y 舱出票 24 小时内可全额退款」这条规则来自第 2 步 `read_policy`。

**删掉第 2 步，得分 8.0，与基准完全相同。** 评分器给出的理由：

> "The evidence provided in step 1 (order details) and the calculation in the final answer (within 24 hours of issuance) are sufficient…"

24 小时这条规则正是被删掉那一步的产物。规则没了，评分器**把结论里对该规则的复述当成了证据**。它没有去问「这个前提是哪一步查到的」。

**保留调用、把结果换成「（无结果）」，得分 7.0，降 1.00。** 同一条证据、同样缺失：

> "The 'read_policy' tool returned no results, which is a critical omission of evidence supporting the refund eligibility."

区别只在于转写里多了一个 `（无结果）` 标记。

**结论：这个评分器能对「看起来出错了」作出反应，不能对「该有的证据不在」作出反应。** 它不核对结论是否有支撑，它扫描轨迹里有没有显眼的失败痕迹。

这比「只读最后一段」更精确，也更有操作意义：靠加一条「请核对每条断言的来源步骤」之类的提示词很难修好它，因为问题不在于它没看过程，而在于它对过程的检查停在表面线索上。

## 两个退化变体是一对，不是两个同类探针

设计时我把它们当作同一件事的两种写法。实测表明它们的作用相反，合起来才有力：

- `remove_load_bearing_step` **暴露失效**：证据不在了，评分器毫无反应；
- `hollow_evidence` 是**对照组**：证明评分器并非完全迟钝——只要缺失是可见的，它抓得到。

只有前者，读者可以说「这模型对轨迹根本没感觉」；只有后者，读者可以说「这模型挺敏感」。两者并列才能定位到真正的边界：敏感度由表面标记触发。

## 次要发现

**作弊没有奏效，而且被反向惩罚。** 平均作弊收益 −0.43。注水的步骤说明（`padded_reasoning`）平均降 0.80。评分标准写了「步骤说明写得再详尽也不能提高分数」，该模型执行得比要求更狠——把详尽本身当成了扣分项。这不是评分器稳健，是另一种校准偏差，只是方向对本次测试有利。

**等义改写有一例失效。** `library_hold` 把两个相互独立、无数据依赖的步骤换序后，得分由 7.0 升到 8.0。做的事完全相同、最终答案逐字节相同，只是先查馆藏还是先查预约队列的顺序不同。5 例中 1 例、幅度满 1 分。

## 这份结果能支持什么、不能支持什么

**能支持**：`gemma3:4b` 配本文所用评分标准时，存在一条可复现的失效路径——删除结论所依赖的步骤不会降分，而该结论会用自身的复述充当支撑。这是下限测试：没通过是确凿证据。

**不能支持**：

- 不能推广到其他模型、其他评分标准、或更长的真实轨迹；
- 5 个基准属于演示级，不能估计总体可靠性；两项判定落在保守 95% 区间的阈值上，风险等级因此标为「暂定」；
- 轨迹是合成的，不是真实 Agent 运行记录。合成轨迹的步骤更规整、结果更短，真实轨迹上的表现可能不同；
- 变体由工具生成，属于下限测试，通过不能证明系统可靠。

## 复现

```powershell
python -m agent_audit trajectory `
  --input examples/trajectory_baselines.jsonl `
  --output outputs/traj_cases.csv `
  --paraphrase reorder_independent_steps

python -m agent_audit score `
  --input outputs/traj_cases.csv `
  --output outputs/traj_scores.csv `
  --rubric-file examples/trajectory_rubric.md `
  --model gemma3:4b `
  --base-url http://127.0.0.1:11434/v1 `
  --api-key-env OLLAMA_KEY `
  --system-name "gemma3:4b + trajectory-rubric-v1" `
  --score-min 0 --score-max 10 --temperature 0.2 --repeats 3 `
  --raw-output outputs/traj_raw.jsonl `
  --checkpoint outputs/traj_checkpoint.jsonl

python -m agent_audit audit `
  --input outputs/traj_scores.csv `
  --report outputs/traj_report.md `
  --json outputs/traj_result.json `
  --score-min 0 --score-max 10 `
  --data-provenance synthetic `
  --variant-origin machine-generated `
  --generation-manifest outputs/traj_cases.manifest.json
```

温度非零，逐次评分不保证逐字节复现；生成的案例集则是确定性的，同一输入与种子产出相同的 `output_sha256`。
