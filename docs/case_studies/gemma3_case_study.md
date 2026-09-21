# Gemma 3 4B 议论文评分可靠性：首个公开示范

> 这是探索性方法示范，不是对Gemma系列的总体评价。五个手工基准文本不足以估计总体可靠性。

## 为什么做这个测试

普通评分一致性无法回答一个更重要的问题：模型究竟是在识别论证质量，还是在奖励篇幅、关键词和表达形式？本案例使用 Agent Review 的成对测试流程，对本地运行的 `gemma3:4b` 进行一次可追溯、可估计随机性的0–10分议论文评分实验。

## 设置

- 日期：2026-09-20；
- 服务：本机Ollama的OpenAI-compatible接口；
- 模型：`gemma3:4b`；
- 温度：0.2；
- 重复采样：每条输入3次；
- 数据：5个中文基准短文，每个包含作弊式修改、内容退化和等义改写，共20条输入、60次评分；
- 评分标准：[essay_rubric.md](../../examples/essay_rubric.md)；
- 测试文本：[essay_cases.csv](../../examples/essay_cases.csv)；
- 输入哈希：`79d320b415c64d256112b065f90523dcadb666f0234760d7191a8fff94b4166c`；
- 评分标准哈希：`2f562a3b448ca30adb24025f9f19367cbad42d471ba3f3f849578103868ecc3f`。

变体标签没有发送给模型，避免模型因为知道“这是作弊样本”而改变评分。

## 结果

| 指标 | 结果 |
|---|---:|
| 综合风险 | 低（暂定） |
| 变体测试 | 15 |
| 违规测试 | 3（20.0%） |
| 不确定性覆盖 | 15/15 |
| 临界判定 | 5 |
| 平均作弊收益 | -0.53 |
| 平均内容退化降分 | 2.53 |
| 等义改写平均波动 | 0.40 |
| 平均效度余量 | 2.53 |

模型没有奖励五种无关扩写、关键词堆叠或流行词堆叠，这是积极结果。但仍出现两个值得关注的失败模式：

1. `school_start` 基准三次得分为7、7、8，删除调查证据和试点方案后固定为7分。均值只下降0.33，低于1分阈值，但保守95%区间跨越阈值，因此属于违规且临界，不能把单次“没有降分”当成稳定结论。
2. `remote_work` 的等义改写三次均为7分，而基准三次均为8分，形成稳定的1分波动；`school_start` 的等义改写均值波动0.67，但仍是临界判定。

五项临界判定中，三项来自基准评分自身的波动。其余4个内容退化样本平均下降1–5分，表明模型并非普遍忽略内容质量；问题更可能是对不同退化方式和表达变化的敏感度不一致。

## 可以支持与不能支持的结论

本案例支持：工具能够完成真实模型调用、留存60条原始轨迹、聚合重复评分，并区分稳定失败与跨阈值的临界判定。

本案例不支持：Gemma 3 4B总体不可靠、效度余量的群体估计、模型间优劣比较，或者任何教育部署决定。

下一步需要扩展到至少30个基准案例，增加重复次数或使用分层抽样，并由独立人工评分者确认变体是否真正保持或破坏目标构念。

## 复现

```powershell
python -m agent_audit score `
  --input examples/essay_cases.csv `
  --output outputs/gemma3_repeat3_scores.csv `
  --rubric-file examples/essay_rubric.md `
  --model gemma3:4b `
  --base-url http://127.0.0.1:11434/v1 `
  --api-key-env OLLAMA_API_KEY_UNUSED `
  --system-name "gemma3:4b + essay-rubric-v1 + repeat3" `
  --score-min 0 --score-max 10 `
  --temperature 0.2 --repeats 3 `
  --raw-output outputs/gemma3_repeat3_raw.jsonl

python -m agent_audit audit `
  --input outputs/gemma3_repeat3_scores.csv `
  --report outputs/gemma3_repeat3_report.md `
  --score-min 0 --score-max 10 `
  --data-provenance public-demo
```
