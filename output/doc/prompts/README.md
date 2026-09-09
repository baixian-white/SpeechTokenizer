# SCIT-Speech 论文 Prompts 引导 README

> 这份 README 是给 **你自己** 的导航文件
> 目标：在 4 份产物文件之间快速找到当前应该用哪一份
> 目录：`output/doc/prompts/`

## 1. 三种路线一览

| 路线 | Framing | 实验依赖 | 投稿目标 | 当前状态 |
|---|---|---|---|---|
| Path A 实证 | "we present an empirical study" | 现有 exp1-5 | IEEE 通信类期刊 / 中文核心 | 已生成 [scit_speech_v2_path_a_draft.md](../paper_drafts/scit_speech_v2_path_a_draft.md), 用户不满意 |
| Path B1 方法学（现有实验）| "we propose ..." | 现有 exp1-5 | ICASSP / Interspeech / TASLP | Prompt 已落盘, 未启动 |
| Path B2 方法学（完整实验）| "we propose ... empirically validated" | 补做 exp7-12 后 | TASLP / JSAC / NeurIPS Audio | Prompt 已落盘, 实验未启动 |

## 2. 本目录下文件的用途

| 文件 | 给谁 | 用途 |
|---|---|---|
| [README.md](README.md)（本文件）| 你 | 导航 / 决策树 |
| [prompt_methodology_existing_experiments.md](prompt_methodology_existing_experiments.md) | Claude Code 新会话 | Path B1 论文写作提示词（Claude 版）|
| [prompt_methodology_existing_experiments_codex.md](prompt_methodology_existing_experiments_codex.md) | Codex CLI / ChatGPT | Path B1 Codex 适配版（与 Claude 版逐字一致，仅工具层适配）|
| [supplementary_experiments_design.md](supplementary_experiments_design.md) | 你（执行实验）| Tier 1+2+3 共 8 个补充实验设计 |
| [prompt_methodology_full_experiments.md](prompt_methodology_full_experiments.md) | Claude Code 新会话 | Path B2 论文写作提示词（补完实验后用）|

## 2.5 Claude vs Codex 对照写作（推荐双盲对比工作流）

如果你想让 Claude 和 Codex 各写一篇做对照：

1. **同时启动**两个独立会话：
   - Claude Code 新会话 → 用 [prompt_methodology_existing_experiments.md](prompt_methodology_existing_experiments.md)
   - Codex CLI / ChatGPT 新会话 → 用 [prompt_methodology_existing_experiments_codex.md](prompt_methodology_existing_experiments_codex.md)
2. **不同输出文件**避免覆盖：
   - Claude → `output/doc/paper_drafts/scit_speech_v2_methodology_draft.md`
   - Codex → `output/doc/paper_drafts/scit_speech_v2_methodology_draft_codex.md`
3. **两份 prompt 的实质性约束完全相同**：framing / contribution wording / 命名 / Rigor / Limitations / 数值溯源 / 自查清单逐字一致。差异仅限工具层指令（Claude 用 TodoWrite + skill 系统；Codex 用文本任务计划）。
4. **对照评估维度**：
   - 论证结构是否完整（9 节是否齐全）
   - 4 条 contribution 措辞是否严格遵循模板
   - 数值是否与 实验记录.md 一致（grep 关键数字）
   - 是否出现内部代号（grep v1/v2/nas_seed42/exp1-6/distill30/distill0）
   - §2 SAC、Glaris、DualCodec、SpeechTokenizer 区分句是否到位
   - §6.4 A3 反直觉发现是否使用弱措辞
   - Limitations 是否有 8 条且全部 future-work 框架

## 3. 决策树（按当前情况选下一步）

```
你现在想要写哪个版本的论文？
│
├── 想要方法学 framing，但不想再花时间补实验
│       → 用 prompt_methodology_existing_experiments.md
│       → 风险：审稿人可能要求补 NAS 对照、factorial 消融、跨语料
│       → 投稿成功率：中等
│
├── 想要方法学 framing，且愿意投入 2-3 周补实验
│       → 第 1 步：按 supplementary_experiments_design.md 跑 Tier 1+2 实验
│       → 第 2 步：更新 实验记录.md 包含 exp7-12 数据
│       → 第 3 步：用 prompt_methodology_full_experiments.md 写论文
│       → 风险：低
│       → 投稿成功率：高
│
└── 想要回到实证 framing（Path A）
        → 已有 scit_speech_v2_path_a_draft.md，按需修订
```

## 4. 启动新会话的步骤（适用于 B1 / B2 两个 prompt）

1. 在 VS Code / 终端打开新的 Claude Code 会话，使用 fresh context（不要在当前会话直接接续）
2. 推荐打开 ultracode 模式（如果环境支持）：在新会话首条消息明确说"开启 ultracode"
3. 复制对应 prompt 文件中 **从分隔线开始的整段内容**（不要复制顶部的元信息和 README 注释）到新会话首条消息
4. 让 Claude 自行调用 ml-paper-writing skill 与读取所有源文件，期间不要打断
5. 中途若 Claude 有需要确认的事项（如数字溯源冲突、framing 边界），按其 AskUserQuestion 的提示回复
6. 完成后 Claude 会输出交付总结；查看 `output/doc/paper_drafts/` 下的新文件
7. 自查清单未通过 → 回复 Claude 让其按清单修复
8. 自查通过 → 进入人工 review / LaTeX 转换 / 投稿前定稿

## 5. 命名约定

为避免混乱，论文 draft 文件名严格如下：

| 路线 | 文件名 |
|---|---|
| Path A 实证 | `scit_speech_v2_path_a_draft.md` |
| Path B1 方法学（现有实验）| `scit_speech_v2_methodology_draft.md` |
| Path B2 方法学（完整实验）| `scit_speech_v3_full_methodology_draft.md` |

不要让 Claude 随意命名输出文件。

## 6. 防错检查（启动新会话之前确认）

启动 B1 前确认：
- [ ] `output/doc/实验记录.md` 是最新状态
- [ ] `output/doc/literature_review_2026_06_02.md` 与 `citation_pool.md` 已落盘
- [ ] `output/doc/paper_outline_v2_path_a_2026_06_02.md` 已落盘
- [ ] `output/doc/paper_drafts/scit_speech_v2_path_a_draft.md` 存在（B1 需读它做文风参考）

启动 B2 前额外确认：
- [ ] exp7、exp8、exp9 全部跑完，run_id 和数据写入 实验记录.md
- [ ] exp10、exp11、exp12 至少有 reports/summary.md
- [ ] `output/doc/paper_drafts/scit_speech_v2_methodology_draft.md` 存在（B2 在此基础上升级）

## 7. 不要做的事

- 不要把 B1 / B2 的 prompt 复制到当前会话执行（会污染 context、消耗大量 tokens）
- 不要让 Claude 在没读完源文件的情况下开始写正文
- 不要跳过 prompt 中的"完成前自查清单"
- 不要让 Claude 改 prompt 文件本身

## 8. 投稿前的最终路径

无论选 B1 还是 B2，论文初稿生成后还需要：

1. 跑 `peer-review` skill 自审一遍（让另一会话扮演审稿人提质疑）
2. 按反馈修订
3. 转 LaTeX（推荐用 `latex-posters` 或手工转）
4. 校对图表数据与正文一致
5. 提交。

完。
