# 轨迹评分可靠性：证据缺失与可见标记（Gemma 3 4B 与 Gemma 4 E4B）

> 这是探索性方法示范，不是对 Gemma 系列的总体评价。五条合成轨迹不足以估计总体可靠性。

## 为什么做这个测试

给 Agent 轨迹打分的评分器，声称评价的是「解决任务的过程质量」。本案例问一个更窄、更可证伪的问题：**当结论所依赖的那一步证据消失时，评分器会不会降分？**

设计这套探针时我的预期是「评分器只读最后一段答案」。实测结果比这个说法更具体，也更有用。

## 设置

- 日期：2026-09-21；
- 服务：本机 Ollama 的 OpenAI-compatible 接口；
- 模型：`gemma3:4b`（主）与 `gemma4:e4b`（同条件对比）；
- 温度：0.2；重复采样：每条输入 3 次；
- 数据：5 条合成轨迹，每条含 2 个作弊、2 个退化、1 个等义改写变体，共 30 条输入；两个模型各评分 90 次；
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

## 第二个模型：这个失效是这一代的共性吗

同一批案例、同一份评分标准、同样的温度与重复次数，换 `gemma4:e4b` 再跑 90 次。

| 变体 | `gemma3:4b` | `gemma4:e4b` |
|---|---:|---:|
| 基准均分 | 8.20 | 9.97 |
| 退化 / `hollow_evidence` | −1.07 | **−4.67** |
| 退化 / `remove_load_bearing_step` | **0.00** | **−2.73** |
| 作弊 / `padded_reasoning` | −0.80 | +0.03 |
| 作弊 / `redundant_tool_calls` | −0.07 | −1.00 |
| 等义 / `reorder_independent_steps` | +0.20 | +0.03 |
| 综合风险 | 严重（28.0% 违规） | 低（8.0% 违规） |
| 平均效度余量 | 0.40 | 3.67 |

**核心失效不是这一代的共性。** 删掉承重步骤后 `gemma4:e4b` 降 2.73 分，确实注意到了证据不在；`gemma3:4b` 一分不降。前者的等义改写波动也降到 0.03，接近完全不变。

**但两个退化变体的不对称在两个模型上都成立。** `gemma4:e4b` 对「可见的空结果」仍然比对「整步消失」敏感得多——−4.67 对 −2.73，差 1.94 分。同一条证据、同样缺失，只因为一个留下了痕迹。方向一致，差别只在 `gemma3:4b` 上严重到降分归零。

这条对比也说明成对使用的必要性：只看 `remove_load_bearing_step`，会把两个模型的差距描述为「一个完全不管、一个管」；把两者并列，真正的图景是**两个模型都更依赖表面痕迹，只是较弱的那个完全依赖它**。

### 两条必须说明的限制

**天花板效应。** `gemma4:e4b` 的基准均分是 9.97/10，作弊变体几乎没有向上的空间——最大作弊收益 0.17。因此「作弊对 `gemma4:e4b` 无效」这个结论，在本次量表上无法与「基准已经顶格」区分开。要检验作弊敏感度，需要基准分不在天花板附近的案例。

> 这条限制原本只写在这里。v0.16.0 起由工具自己检测并写进报告——同一份数据重新审计时，5/5 个案例都会被标为「距量表上限不足 1.00，作弊收益不可解读」。散文会漏写，而读者看到的是数字。

**比较器读不出「停止了过度惩罚」。** `compare` 把 1 项判为跨阈值回归：`grant_eligibility` 的 `padded_reasoning` 由 −1.00 变为 +0.17。工具按定义把「作弊收益上升」记为变差。但 `gemma3:4b` 的 −1.00 本身就是误校准——评分标准只要求注水**不得加分**，没有要求扣分。`gemma4:e4b` 给出约 0 才是符合标准的行为。

参考系统自身存在误校准时，本工具的严重度方向无法区分「候选变得可利用」与「候选停止了过度惩罚」。这是方法的已知边界，读这类回归项时必须回看参考系统的绝对值，而不是只看变化方向。

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

第二个模型只需把 `--model` 与输出路径换掉再跑一遍 `score` 与 `audit`，然后：

```powershell
python -m agent_audit compare `
  --reference outputs/traj_result.json `
  --candidate outputs/traj_result_g4.json `
  --report outputs/traj_compare.md
```

`compare` 会校验两侧的输入哈希、评分标准哈希、阈值、量表、温度、重复次数与变体来源声明完全一致，不一致直接拒绝——避免把不同条件下的结果拼成对比。

温度非零，逐次评分不保证逐字节复现；生成的案例集则是确定性的，同一输入与种子产出相同的 `output_sha256`。
