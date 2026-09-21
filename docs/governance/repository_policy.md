# 仓库与发布规范

本规范用于保持 Agent Review 的源码、公开证据、客户产物和Git历史边界清晰。每次推送前必须执行本文件的发布检查。

## 目录职责

| 路径 | 用途 | 是否提交 |
|---|---|---|
| `agent_audit/` | 可复用Python包与CLI实现 | 是 |
| `tests/` | 单元、端到端、故障注入和公开资产一致性测试 | 是 |
| `examples/` | 合成或明确公开的最小复现输入 | 是 |
| `docs/` | 方法、案例、服务、质量门和公开HTML示例 | 是 |
| `.github/` | CI工作流与公开需求模板 | 是 |
| `scripts/` | 可重复的本地审查与公开资产构建脚本 | 是 |
| `outputs/` | 评分CSV、清单、原始回复、检查点和客户报告 | 否，仅保留 `.gitkeep` |
| `output/` | 浏览器截图等临时审查证据 | 否 |
| `.playwright-cli/` | 浏览器自动化临时状态 | 否 |

## 文件与命名

- Python模块、脚本和测试使用小写 `snake_case`；
- Markdown文件使用小写 `snake_case`，两类例外：顶层语言入口 `README.md`、`README_EN.md`，以及 `.github/` 下由GitHub约定命名的文件；
- `docs/` 下每个文件都必须出现在 `docs/README.md` 导航中，由自动测试强制；
- 公开Markdown中的 `issues/new?template=` 链接必须指向 `.github/ISSUE_TEMPLATE/` 中的现存文件，由自动测试强制；
- 文本文件统一由 `.gitattributes` 规范为LF行尾；
- 公开生成物必须有确定的构建脚本和一致性测试；
- 不提交编辑器缓存、构建目录、运行日志、模型原始回复或临时截图；
- 不使用模糊的 `final`、`new`、`latest2` 等文件名表达版本，版本由Git提交和标签管理。

## 公开与私有边界

- `docs/examples/example_audit_report.html` 只由合成数据生成，可公开提交；
- 真实模型运行产物即使使用公开文本，也默认保留在 `outputs/`，除非经过逐项脱敏和公开性审查；
- 客户数据、API密钥、私有提示词、个人信息和受限材料不得进入Git或公开Issue；
- 检查点和原始JSONL包含完整模型回复，按敏感数据处理；
- 对外案例必须明确数据来源、样本量、证据强度以及不能支持的结论。

## 提交与贡献者身份

- 本仓库的提交Author与Committer统一使用 `Potato0705` 的Git身份；
- 提交信息不得包含 `Co-authored-by` 或其他贡献者Trailer；
- 自动化工具不作为作者、共同作者或提交者写入历史；
- 推送前检查所有分支与标签历史，而不是只检查最新提交。

身份审查命令：

```powershell
git log --all --format='%an|%ae|%cn|%ce'
git log --all --format='%B' | Select-String -Pattern 'Co-authored-by|Signed-off-by'
git tag --format='%(refname:short)|%(taggername)|%(taggeremail)'
```

## 发布检查

1. `git status --short` 只包含本阶段预期修改；
2. 重新构建公开HTML，确认一致性测试通过；
3. 执行 `scripts/review.ps1`；
4. 确认相对Markdown链接全部存在；
5. 检查凭据类文件名、未忽略生成物和异常大文件；
6. 执行 `git diff --check`，暂存后再执行 `git diff --cached --check`；
7. 审查Author、Committer、提交Trailer和标签Tagger；
8. 确认远端目标仓库与分支正确，再推送分支和明确的版本标签；
9. 推送后读取远端引用并核对本地与远端提交哈希；
10. 推送后确认远端CI在全部受支持的Python版本上通过，并记录运行编号。本地门禁与CI在解释器版本、操作系统和路径处理上都不相同，本地全绿不能替代这一步。

完整发布检查可执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/release_check.ps1
```

提交与打标签后、推送前使用严格模式：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/release_check.ps1 -RequireClean
```
