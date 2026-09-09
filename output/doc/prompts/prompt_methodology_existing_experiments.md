# SCIT-Speech 论文 Prompt B1：方法学路线 / 现有实验版

> 用途：在 **不补做新实验** 的前提下，按方法学路线撰写论文初稿
> 投稿目标：ICASSP / Interspeech / IEEE TASLP
> 输出文件：`output/doc/paper_drafts/scit_speech_v2_methodology_draft.md`
> 复制下面整段（从分隔线开始）到新 Claude Code 会话即可

---

# 任务：撰写 SCIT-Speech 论文 v2-B (方法学路线 / 现有实验版) 中文初稿

你在 H:\H-CODE\speechtokenizer 仓库下工作。本任务是非平凡的多文件深度任务，若你的运行环境支持 ultracode，请打开。

## 0. 角色与目标

你是一位熟悉神经语音 codec 与语义通信的资深论文作者。本任务是撰写一篇 **方法学定位** 的论文中文初稿，目标投稿 ICASSP / Interspeech / IEEE TASLP / JSAC 这一档。这与之前生成的 Path A 实证版（output/doc/paper_drafts/scit_speech_v2_path_a_draft.md）不同 —— 该版本将主要贡献写成"我们做了一项实证研究"，作者认为过度退让，决定重写为方法学路线，明确主张本文提出新机制。

但本任务只能基于已有实验数据（exp1-5），不补做任何新实验。

## 1. 必读源文件（动笔前必须全部读完）

按顺序：

1. **v2 大纲（章节结构和数据来源参考；framing 不复用）**：
   `output/doc/paper_outline_v2_path_a_2026_06_02.md`
2. **文献缺口判据（核心，决定方法学声明的强度边界）**：
   `output/doc/literature_review_2026_06_02.md`
3. **引用池**：`output/doc/citation_pool.md`
4. **实验真实状态（数字唯一来源）**：`output/doc/实验记录.md`
5. **现有 Path A 初稿（仅供文风和结构参考；framing 必须重写）**：
   `output/doc/paper_drafts/scit_speech_v2_path_a_draft.md`
6. **实验路线手册**：`output/doc/实验手册.md`

读完后用 TodoWrite 记录待办章节。

## 2. 调用专业 skill

主调用 `ml-paper-writing`。
辅助：`scientific-writing`（句法层面）、`peer-review`（写完后自审）。

## 3. 输出

- 路径：`output/doc/paper_drafts/scit_speech_v2_methodology_draft.md`（新建文件）
- 长度：13000-19000 字中文
- 不输出 emoji，章节用 ## / ### Markdown 标题

## 4. 方法学 framing 强约束（与 Path A 完全相反）

本论文必须 **主张** 以下机制为本文的方法学贡献：

1. **Shared-codebook index-only transmission as a new paradigm for low-bitrate speech communication.** 必须使用 "we propose" / "本文提出"。
2. **Layer-controlled transport-layer load mechanism.** 把 L 作为通信参数训练 —— 必须主张这是本文的方法学贡献，区别于已有 codec 的 post-hoc truncation。
3. **A channel-aware adaptation objective combining random-L sampling, strong index perturbation, and clean-vs-perturbed mel-L1 consistency loss.** 必须主张这一组合为本文新提出的训练目标。
4. **A constrained NAS protocol decoupling encoder lightweighting from the transmission interface.** 主张该 protocol 为本文的工程方法贡献。

这些主张的可信度由 deep-research 验证过的文献空白支持（见 literature_review_2026_06_02.md 的 Q2+Q3+Q4+Q5 验证结果），允许使用强措辞。

但同时必须诚实承认证据边界（见 §7）。

## 5. 命名固定（禁止内部代号进入正文）

| 禁用 | 改写为 |
|---|---|
| LCA v1 / v2 | SCIT-Speech-LCA / SCIT-Speech-LCA(weak-perturb variant) |
| nas_seed42_000896 | "the proposed lightweight encoder" / "本文 NAS encoder" |
| distill_loss_lambda | λ_sem |
| distill30 / distill0 | λ_sem=30 / λ_sem=0 |
| exp1-6 / run_id | 不出现 |

## 6. Rigor 声明（必须出现，位置精确）

方法学 framing 不等于过度声明。必须保留以下 rigor 措辞：

- **§5.4 评价指标节末尾**两句不变：
  - "由于评估样本量限制（n=8），本文以描述性统计形式报告所有结果，不进行显著性检验。"
  - "Whisper base.en 在 PCM 无损音频上的 WER 约为 0.075，作为本评测的测量下界。"

- **§6.1 主结果**："优于"必须加 "in the 500-1500 bps regime" 限定。

- **§6.4 A3 蒸馏消融**：使用 "we observe" / "one possible interpretation" 弱措辞，并在 future work 中说明 λ_sem 细粒度 sweep 必要性。

- **§4.5 LCA 训练目标节**：必须明示 "the index-perturbation channel model considered in this work simulates index-level dropout and substitution; its relationship with realistic wireless channel impairments under coding and retransmission is left to future work."

## 7. 三类已知证据短板的方法学应对（核心差异点）

Path A 把短板下沉到 Limitations 并降低 contribution 强度。本路线 **保持 contribution 强度，但用"方法学 vs 完整实验验证"的二分法管理短板**：

### 7.1 NAS 质量等价对照尚未闭环
- §4.2 写法："we propose a constrained NAS protocol that searches the pre-latent encoder under a fixed downstream interface; in this work we empirically validate the efficiency benefit on params, MACs, and RTF reductions, while leaving a fully matched-budget hand-designed comparison for future work."
- §6 不要硬声明 NAS 与手工 encoder 质量等价；只报效率指标。

### 7.2 random-L 与 ChannelSim+consistency 在 LCA 中 confounded
- §4.5 写法：清晰描述训练目标的所有四个组件（random-L sampling、index dropout、light substitution、mel-L1 clean-vs-perturbed consistency）作为 **联合提出** 的训练目标。即把 confound 转化为"本文提出的 unified objective 包含若干组件"。
- §6.3 写法：报告整个目标的鲁棒性收益，使用 "the proposed objective" 集体语义，不单独归因 random-L。
- §8 Limitations：诚实指出"the marginal contribution of each component within the proposed objective is not isolated in this paper, and a per-component factorial study is left to future work."

### 7.3 Index-perturbation channel ≠ wireless channel
- §3.4 写法：明确定义本文 ChannelSim 为 "a synthetic index-level channel model"，不假装真实物理信道。
- §4.5 写法："we model the channel at the index level rather than at the bit or symbol level. This abstraction simplifies the analysis of how the receiver-side codebook lookup tolerates index errors, while leaving the joint design of index-level training and physical-layer channel coding for future work."
- §7 Discussion 单独一段标题 "From Index-Level to Physical-Layer Channels" 讨论这一抽象的研究价值与边界。

## 8. Contribution 措辞模板（直接采用）

§1 末尾的 4 条 contribution 必须严格按以下措辞：

> The contributions of this paper are as follows:
>
> **C1. We propose Shared-Codebook Index Transmission (SCIT) as a low-bitrate speech communication paradigm.** Unlike prior neural speech codecs that frame RVQ indices as internal artifacts of audio compression, SCIT explicitly repositions them as the only object transmitted over the channel, with the codebook and decoder shared between transmitter and receiver as deployment-time prior knowledge. We further introduce a three-tier payload accounting framework distinguishing ideal bit-packed index load, packed payload, and packetized payload.
>
> **C2. We propose layer-controlled transmission load as a transport-layer design knob jointly trained with the codec.** While prior RVQ codecs support truncation post-hoc for tokenizer flexibility, we train SCIT-Speech with random-L sampling so that a single model serves three communication operating points (500/1000/1500 bps) under explicit transport semantics.
>
> **C3. We propose a low-load channel-aware adaptation objective for index-only transmission.** The proposed objective combines random-L sampling, strong index-level perturbation (dropout and substitution at calibrated rates), and a clean-decoded versus perturbed-decoded mel-L1 consistency loss. Empirically, weak perturbation alone is shown insufficient, while the proposed combination yields positive robustness improvement on six objective metrics across twelve (L, perturbation) cells.
>
> **C4. We propose a constrained Neural Architecture Search protocol for the transmitter encoder under a fixed downstream interface.** The protocol fixes sample rate, downsample factor, latent rate, latent dimension, and the RVQ interface during search, decoupling encoder lightweighting from changes in the transmission load model. The selected NAS encoder reduces parameters by 86.9%, MACs by 89.5%, and RTF by 55.1% relative to a hand-designed baseline.

中文版按此严格意译，"propose" 翻译为"本文提出"或"本文构造"，不弱化。

## 9. §2 Related Work 区分句必须使用强措辞

对每个 prior work 必须明确指出 SCIT 在 framing / mechanism / objective 层面的差异。推荐句式：

- "Concurrent work SAC operates at comparable bitrates but adopts a single-codebook quantizer trained at fixed acoustic frame rates, lacking a layer-controlled transport knob and a payload-accounting framework."
- "Glaris addresses packet-loss concealment via mask-token training at 8-18 kbps; the proposed channel-aware adaptation objective differs in (a) the 500-1500 bps operating regime, (b) the strong-vs-weak perturbation contrast established in this paper, and (c) the introduction of clean-vs-perturbed mel-L1 consistency as a regularizer."
- "DualCodec employs RVQ dropout for downstream LM tokenizer flexibility; we reframe the same structural mechanism as a transport-layer knob, supported by a payload-accounting framework absent in prior work."
- "SpeechTokenizer demonstrates that HuBERT distillation aids RVQ training but does not quantify its necessity; we provide such a necessity ablation in the ultra-low-bitrate regime."

## 10. §3 Problem Formulation 必须显式数学化

不仅描述系统，还要给出可作为方法学贡献参照点的形式化定义：

- 共享码本通信信道的形式化：定义发送端集合 T、接收端集合 R、共享 codebook C* 与 decoder D* 作为 prior knowledge K_pre。
- 索引-only 信道的形式化：信道 W 仅传输 I_L = I_{1:L,:}, 其余 layers I_{L+1:M,:} 在发送端被丢弃。
- 三层 payload 形式化：R_ideal、R_packed、R_packetized。
- LCA 训练目标的形式化（C3 的核心方法学声明）：

```
L_LCA = α · L_full(x, x̂_3)
      + β · E_{L ~ U{1,2,3}, c ~ C_perturb} [ L_comm(x, x̂_{L,c}) ]
      + λ_cons · E_{L,c} [ ‖mel(x̂_{L,clean}) − mel(x̂_{L,c})‖_1 ]
      + λ_commit · L_commit + λ_sem · L_sem
      + λ_adv · L_adv + λ_fm · L_fm
```

并在 §4.5 给出每项的方法学动机说明。这是论文方法学声明的核心数学锚点。

## 11. 章节字数预算（目标 13000-19000 字）

| 节 | 字数 |
|---|---|
| Abstract | 500-700 |
| §1 Introduction | 1500-2200 |
| §2 Related Work | 1800-2500 |
| §3 Problem Formulation | 1000-1400 |
| §4 SCIT-Speech: Methodology | 2500-3500 |
| §5 Experimental Setup | 1000-1500 |
| §6 Results | 2500-3500 |
| §7 Discussion | 1500-2000 |
| §8 Limitations | 600-900 |
| §9 Conclusion | 200-400 |
| References | 30-50 条 |

## 12. 数值溯源（与 Path A 同等严格）

所有数字必须从 实验记录.md 反查。关键必现数字：

- NAS 效率：params -86.9%, MACs -89.5%, RTF -55.1%
- 主结果：L=3 (1.5 kbps) WER 0.142 ≈ Opus 6 kbps WER 0.147
- 鲁棒性：mel_l1 12/12 cell, PESQ-WB 10/12, STOI 9/12
- 蒸馏消融：dev/mel @ step 47500: λ_sem=0 → 3.202, λ_sem=30 → 1.776, Δ=+1.426
- WER 退化 @ L=1/2/3：+9.4 / +7.4 / +8.4 pp

## 13. §8 Limitations 措辞（方法学版本特别版）

方法学路线下的 Limitations 必须把"已知短板"重新框定为"未来扩展方向"：

1. **Hand-designed encoder matched-budget comparison is left to future work.** 参数 / MACs / RTF 优势已建立，但完整重训对照尚未闭环，因此本文 NAS protocol 的质量等价主张仅基于 short-distill proxy 与下游训练成功性，不基于 matched-budget 对照。
2. **Per-component factorial study of the proposed channel-aware objective.** random-L sampling、strong index perturbation、consistency loss 三组件作为联合方法学贡献提出，单独消融仍待补做。
3. **Cross-corpus and subjective evaluation.** n=8 train-clean 同源样本评估，未覆盖 LibriSpeech test-clean / test-other / VCTK / 真实通信语音；缺主观听测。
4. **Index-level vs physical-layer channel models.** 本文 ChannelSim 为 index-level 抽象信道，与 BER / 包丢失 / FEC 联合设计是后续工作。
5. **Codebook utilization profile.** L1 dead-code 82.5%, 总利用率 22.7%；A3 蒸馏消融暗示主结果不来自 codebook collapse，但 dead-code reinit / k-means init / usage entropy 正则化的系统对比是后续工作。
6. **Domain-mismatch with neural codec baselines.** DAC 训练域含音乐而 SCIT 仅语音，部分低码率优势可能来自 domain specialization。
7. **AMR-WB baseline absence.** ffmpeg 6.1.2 工具链缺 AMR-WB encoder，最直接的传统语音 codec 对照尚未纳入。
8. **End-to-end packetized payload and multi-user system validation.** 本文报告 ideal 与 packed payload，packetized payload 与三用户实时系统验证作为后续系统级工作。

## 14. 完成前自查清单

- [ ] 全文 grep 不到 v1/v2/nas_seed42/exp1-6/distill30/distill0/distill_loss_lambda
- [ ] 4 条 contribution 严格采用 §8 模板，使用强措辞 "we propose"
- [ ] §3 包含完整的 LCA 训练目标数学公式
- [ ] §2 对 SAC、Glaris、DualCodec、SpeechTokenizer 各有明确区分句
- [ ] §7 含独立小节讨论 index-level vs physical-layer channel abstraction
- [ ] §8 Limitations 8 条全部以"future work"框定，不破坏 contribution 强度
- [ ] §5.4 含 n=8 描述性统计 + Whisper 测量下界两句
- [ ] §6.1 "优于"全部加 "在 500-1500 bps 区间" 限定
- [ ] §6.4 A3 反直觉发现使用 "we observe" / "one possible interpretation" 弱措辞
- [ ] 所有数字与 实验记录.md 一致
- [ ] DAC 域不匹配仅在 §8 第 6 条出现，不进 §6
- [ ] Codebook 利用率仅在 §8 第 5 条与 Appendix 出现

## 15. 不确定时的处理

- 数字不确定 → Read 实验记录.md
- Framing 不确定 → 优先选 **方法学声明强度更高** 的措辞，但保留 rigor 边界
- 引用不确定 → Read citation_pool.md

不允许跳过自查。不允许编造数字。

## 16. 交付

完成后给出简短交付总结：
- 总字数与各节字数分布
- 自查清单逐项结果
- 与 Path A 版本的 framing 差异 5 条要点
