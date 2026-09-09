# SCIT-Speech 论文 Prompt B2：方法学路线 / 完整实验版

> 用途：在 **补充实验 exp7-12 全部完成后**，按方法学路线撰写完整论文初稿
> 投稿目标：IEEE TASLP / JSAC / NeurIPS Audio
> 输出文件：`output/doc/paper_drafts/scit_speech_v3_full_methodology_draft.md`
> 前置条件：必须先按 `supplementary_experiments_design.md` 完成 Tier 1+2 实验
> 复制下面整段（从分隔线开始）到新 Claude Code 会话即可

---

# 任务：撰写 SCIT-Speech 论文 v3 (方法学路线 / 完整实验版) 中文初稿

你在 H:\H-CODE\speechtokenizer 仓库下工作。本任务是非平凡的多文件深度任务，若你的运行环境支持 ultracode，请打开。

## 0. 角色与目标

你是一位熟悉神经语音 codec 与语义通信的资深论文作者。本任务在前期方法学版（output/doc/paper_drafts/scit_speech_v2_methodology_draft.md）的基础上升级，利用补充完成的 6 项实验（exp7-12）将方法学声明从 part-supported by ablation 提升到 fully validated。

补充完成的实验（必须确认 实验记录.md 已更新到包含这些数据）：
- exp7：Hand-designed encoder 同条件全训对照
- exp8：λ_sem distillation strength sweep（5 点）
- exp9：LCA 训练目标 4 组件 factorial 消融
- exp10：跨语料评估（test-clean / test-other / VCTK）
- exp11：Bit-level BSC 与 packet-level Gilbert-Elliott 信道评估
- exp12：MOS-lite 主观听测

补充实验的 run_id 与 reports 在 output/experiments/ 下，对应的更新版 实验记录.md 已更新到当前日期。

## 1. 必读源文件

按顺序：

1. **更新版 实验记录.md**：`output/doc/实验记录.md`
   重点关注 exp7-12 的 实验主线 与 结果摘要 段落
2. **方法学版前作**：`output/doc/paper_drafts/scit_speech_v2_methodology_draft.md`
   保留全部 framing，只升级数据与论证强度
3. **v2 大纲**：`output/doc/paper_outline_v2_path_a_2026_06_02.md`
   章节结构参考，framing 不复用
4. **文献缺口判据**：`output/doc/literature_review_2026_06_02.md`
5. **引用池**：`output/doc/citation_pool.md`
6. **补充实验设计**：`output/doc/prompts/supplementary_experiments_design.md`
   每份 exp7-12 的 reports/summary.md 都要读

读完后用 TodoWrite 记录待办章节。

## 2. 调用专业 skill

主调用：`ml-paper-writing`、`statistical-analysis`（处理 exp9 factorial 与 exp12 听测统计）。

## 3. 输出

- 路径：`output/doc/paper_drafts/scit_speech_v3_full_methodology_draft.md`
- 长度：18000-25000 字中文
- 与 v2 方法学版结构完全一致，但每节都加上补充实验产物
- 不输出 emoji

## 4. v2 方法学版 → v3 完整版的关键升级点

### 升级 1：C4 NAS protocol 主张全面闭环
- 利用 exp7 的 hand-designed encoder 同条件结果
- §4.2 → 增加 matched-budget quality validation
- §6 → 新增 Table NAS vs Hand-designed at matched compute
- §8 第 1 条 limitation 删除

### 升级 2：A3 蒸馏从二值升级为完整 sweep
- 利用 exp8 的 5 点 λ_sem sweep
- §4.4 → 增加蒸馏强度的方法学动机说明
- §6.4 → 主图改为 sweep curve；observe sweet spot；写明 λ_sem ∈ [10, 30] is the empirically optimal range
- §6.4 A3 弱措辞替换为 the sweep confirms that distillation strength has a monotone-then-saturating relationship with downstream quality, with sweet spot near λ_sem=30
- §8 蒸馏 sweep 限制条目删除

### 升级 3：C3 LCA 训练目标完整 factorial 消融
- 利用 exp9 4-run factorial（A: random-L; B: ChannelSim; C: A+B; D: A+B+consistency）
- §4.5 → 训练目标方法学讨论增加 component-by-component motivation
- §6.3 → 新增 Table Per-component contribution within the proposed channel-aware objective
- §8 第 2 条 limitation 删除
- C3 contribution 措辞强化为 each component is empirically necessary; the combination is optimal

### 升级 4：跨语料泛化评估
- 利用 exp10 的 test-clean / test-other / VCTK 数据
- §5.1 数据集说明扩展
- §6.1 主结果表增加跨语料列
- §6.5 新增小节 Cross-corpus generalization
- §8 第 3 条 limitation 降级为 未覆盖远场 / 噪声 / 多语言

### 升级 5：物理层信道评估
- 利用 exp11 的 BSC 与 GE 实验
- §3.4 → 新增 physical-layer channel coupling 小节，把 index-level abstraction 与 BER / packet loss 链接
- §6.6 新增小节 Performance under realistic channel impairments
- §7 新增独立小节 From Index-Level to Physical-Layer Channels
- §8 第 4 条 limitation 降级或删除
- 与 Glaris 的 PLC results 直接对照（在同 packet loss rate 下报告）

### 升级 6：主观听测
- 利用 exp12 的 AB preference data
- §5.5 评价指标节增加 subjective AB preference protocol
- §6.7 新增小节 Subjective listening evaluation
- §8 第 3 条 limitation 进一步降级（仅保留 未做大规模 MOS）
- Abstract 增加一句 subjective listening tests confirm the objective metric trends in the 500-1500 bps regime

## 5. 命名固定（同前两个 prompt）

| 禁用 | 改写 |
|---|---|
| LCA v1 / v2 / exp7-14 | 删除内部代号；用方法学描述 |
| nas_seed42_000896 | the proposed lightweight encoder |
| distill_loss_lambda | λ_sem |

## 6. Rigor 声明（保留）

- §5.4 n=8 描述性统计 → 改写为 evaluation set has been extended to N=130 utterances across LibriSpeech test-clean/test-other/VCTK; statistical tests follow standard practice
- §5.4 Whisper 测量下界保留
- §6.1 在 500-1500 bps regime 限定保留

## 7. 章节字数预算（v3 完整版）

| 节 | v2 字数 | v3 字数 | 增量来源 |
|---|---|---|---|
| §1 | 1500-2200 | 1800-2500 | 摘要新结果 |
| §2 | 1800-2500 | 1800-2500 | 不变 |
| §3 | 1000-1400 | 1300-1700 | 物理层耦合小节 |
| §4 | 2500-3500 | 3000-4000 | NAS matched-budget + 蒸馏 sweep + factorial 动机 |
| §5 | 1000-1500 | 1300-1800 | 跨语料 + 听测 protocol |
| §6 | 2500-3500 | 4500-6000 | 6 个新表/图，6.5+6.6+6.7 三个新小节 |
| §7 | 1500-2000 | 2000-2500 | 物理层独立小节 |
| §8 | 600-900 | 400-600 | 4 条 limitation 删除/降级 |
| §9 | 200-400 | 250-450 | 主张升级 |
| **总计** | 13000-19000 | **18000-25000** | |

## 8. Contribution 措辞（v3 升级版）

C1 不变。C2 不变。

C3 升级为：

> **C3. We propose a low-load channel-aware adaptation objective for index-only transmission and empirically validate the necessity of each component.** The proposed objective combines random-L sampling, strong index-level perturbation, and a clean-vs-perturbed mel-L1 consistency loss. A factorial study confirms that **each component is individually necessary** and **the combination is optimal**, yielding positive robustness improvement on six objective metrics across twelve (L, perturbation) cells. Performance under bit-level BSC and packet-level Gilbert-Elliott channels further validates the index-level abstraction's relevance to realistic communication impairments.

C4 升级为：

> **C4. We propose a constrained Neural Architecture Search protocol for the transmitter encoder under a fixed downstream interface, validated under matched compute against a hand-designed baseline.** The protocol decouples encoder lightweighting from changes in the transmission load model. The selected NAS encoder reduces parameters by 86.9%, MACs by 89.5%, and RTF by 55.1% relative to a hand-designed baseline at matched downstream training compute, while preserving downstream WER and PESQ-WB.

新增（可选）C5：

> **C5. We provide a layer-wise distillation strength characterization for low-bitrate RVQ speech codecs.** A five-point sweep of λ_sem over {0, 10, 30, 60, 120} reveals a monotone-then-saturating relationship between distillation strength and downstream quality, with empirical sweet spot in [10, 30]. This characterization is the first such sweep in the sub-1.5 kbps regime.

是否保留 C5 取决于 exp8 sweep 是否清晰显示 sweet spot；若曲线噪声大则降级到 empirical observation 不作为独立 contribution。

## 9. §8 Limitations 大幅缩减

v3 版本下 limitations 从 8 条降至 4 条：

1. **Subjective evaluation scale.** AB preference 测试基于 8-12 听者；大规模 MOS 听测留待未来工作。
2. **Codebook utilization profile.** L1 dead-code 比例仍偏高；dead-code reinit / k-means init 系统对比留作后续工程优化。
3. **Domain mismatch with neural codec baselines.** DAC 训练域含音乐；同等数据规模 / 同等域的 fair comparison 留待未来工作。
4. **End-to-end packetized payload and multi-user real-time system.** Tier 2 补做了 bit-level BSC 与 packet-level GE，但完整三用户实时系统验证仍是后续系统级工作。

AMR-WB 缺失等工程级 caveat 完全删除（已不重要）。

## 10. 完成前自查清单

- [ ] 内部代号全部清除
- [ ] C1-C4 措辞与本 prompt §8 严格一致；C5 视 exp8 结果保留或删除
- [ ] §6 包含 7 个小节（6.1 主结果 / 6.2 Base vs LCA clean / 6.3 LCA factorial / 6.4 蒸馏 sweep / 6.5 跨语料 / 6.6 物理层 / 6.7 听测）
- [ ] §7 含独立 From Index-Level to Physical-Layer Channels 小节
- [ ] §8 限制为 4 条
- [ ] 所有数字与 实验记录.md（含 exp7-12 数据）一致
- [ ] §6 每个新小节都对应一个 Figure 或 Table

## 11. 不确定时的处理

- exp7-12 数据点缺失 → Read 对应 reports/summary.md
- exp7-12 与现有数据冲突 → 优先采纳新数据，作 caveat 在 footnote
- 不允许跳过自查；不允许编造数字

## 12. 交付

完成后给出简短交付总结：
- 总字数与各节字数分布
- v2 方法学版 → v3 完整版的 6 大升级是否全部落地
- C5（蒸馏 sweep）是否保留为独立 contribution，附理由
- 与 v2 比较的核心差异 5 条
