# SCIT-Speech 论文大纲 v2 (Path A: 实证系统研究)

> 生成: 2026-06-02
> 路线: Path A (实证 / 系统研究, 非方法学)
> 文献缺口依据: [literature_review_2026_06_02.md](literature_review_2026_06_02.md) (deep-research workflow w043ohpnt, 23 sources verified)
> 旧大纲: [paper_outline_low_load_semantic_speech.md](paper_outline_low_load_semantic_speech.md) (v1, 方法学 framing) 保留供历史参考, 不再活跃维护

## 0. 修订说明 (相对 v1)

| 维度 | v1 | v2 |
|---|---|---|
| Framing | 方法学 ("提出 framework + channel-aware objective") | 实证系统研究 ("重新表述被回避的区间 + 给出训练配方 + 隔离必要训练成分") |
| Contribution 1 | "共享码本索引传输框架" (隐含 new mechanism) | "索引-only 传输接口 + 三层 payload 核算" (对应 Q2 缺口) |
| Contribution 3 | "channel-aware objective" | "训练配方 (Base+LCA), 强扰动+一致性必要性消融" (对应 Q4 缺口) |
| Contribution 4 | 蒸馏作为隐含机制 | 蒸馏必要性消融 (对应 Q5 缺口) |
| Limitations | 边界声明嵌入正文 | 独立 §8, 不放正文 |
| 实验六 (三用户系统) | 列入 system feasibility | 暂不列入, 留 future work |
| LCA 命名 | v1 / v2 内部代号出现在正文 | 正式: SCIT-Speech-LCA / 消融变体: weak-perturb 变体 |

## 1. 标题与定位

### 1.1 标题候选 (按推荐顺序)

1. **SCIT-Speech: A Shared-Codebook Index Transmission System for Speech Communication at 500-1500 bps**
2. Index-Only Speech Transmission with Shared RVQ Codebooks: A System Study at 500-1500 bps
3. Toward Low-Bitrate Speech Communication via Shared-Codebook Index Transmission: System Design and Empirical Analysis

中文暂定: 基于共享码本索引传输的极低负载语音通信: 系统实例与训练配方实证研究

### 1.2 一句话核心主张

> 在 500-1500 bps 这个被现有神经语音 codec 系统性回避的极低负载区间, 把 RVQ 语音编码器重新表述为"共享码本 + 索引-only 传输接口", 并通过一套训练配方让它真正可用; 本文不提出新机制, 但通过完整负载核算、层数档位训练和三组消融, 给出该区间下第一个端到端可解释、可复现的语音通信系统实例.

### 1.3 文献缺口锚点

最强联合空白 = Q2 + Q3 (3-0 验证, 见 [literature_review_2026_06_02.md](literature_review_2026_06_02.md)):

七维度同时占据无先例:
1. RVQ (非 single-codebook -- SAC 不重叠)
2. 500-1500 bps 区间 (Glaris 在 8-18 kbps 不重叠)
3. Transmission framing (DualCodec 的 tokenizer-for-LM framing 不重叠)
4. Payload accounting (区分 ideal/packed/packetized -- 全部论文都没做)
5. L-knob 训练 (random-L 为传输, 而非为下游 LM consumer -- 全部论文都没做)
6. 强弱扰动消融 (Glaris 只做 PLC mask, 没有强弱对照)
7. HuBERT 蒸馏必要性消融 (SpeechTokenizer 只证明 helps, 未量化 necessity)

SCIT-Speech 同时占据这七维度.

### 1.4 论文边界 (写在 §1 末尾, 不下沉到 Limitations)

- 本文是实证系统研究, 不是新机制论文. 不声称提出新通信范式.
- 本文不在所有码率上击败通用 codec. 在 6 kbps+ 区间 Opus / DAC 仍是成熟选择.
- 本文不依赖大型语音模型作为接收端核心.
- 本文不是 audio compression benchmark, 不应读作"我们全面赢了 EnCodec / DAC".

---

## 2. 论证逻辑链 (六步)

整篇论文是一条 **收缩-扩张-收缩** 的论证链.

### 第 1 步 (§1 Introduction): 定义被回避的操作区间

**主张**: 500-1500 bps 是真实存在但被系统性回避的区间.

**证据 (Q1, deep-research 验证)**:
- SoundStream 最低 3 kbps; EnCodec abstract 仅 MUSHRA, 最低 1.5 kbps; StreamCodec2 仅 1.7 kbps 单点
- DualCodec 0.75-0.93 kbps **不报 WER**
- 仅 SAC (2510.16841, 2025-10) 和 Codec-SUPERB 在该区间发数据, 都不是稠密 RVQ 曲线
- 没有任何论文给出 500-1500 bps 的稠密单 codec WER+PESQ+STOI 曲线

### 第 2 步 (§3 Problem Formulation): 重新表述为索引-only 传输 + 负载核算

**主张**: 该区间应作为通信问题处理, 而非 audio compression. 共享码本与 decoder 是 prior knowledge, 信道仅传前 L 层 RVQ 索引, 区分 ideal / packed / packetized 三层负载.

**证据 (Q2, 3-0 验证, 完全空白)**: 见 [literature_review_2026_06_02.md](literature_review_2026_06_02.md) §Q2.

### 第 3 步 (§4 System Instance, §4.3): L 作为传输层负载控制变量

**主张**: RVQ truncation 大家都有, 但为传输训练的 L-knob 没人做. SCIT 用 random-L 训练让一个模型同时在 L=1/2/3 三档可用, 把 L 提升为传输参数.

**证据 (Q3, 3-0 验证, 完全空白)**: DualCodec 的 RVQ dropout 是 TTS tokenizer flexibility; SAC 是 single-codebook; SoundStream structured dropout 是 quality-bitrate scalability.

### 第 4 步 (§4.4-4.5, §6.1): 给出可用的训练配方

**主张**: 把上述接口跑通, 需要具体训练配方. 本文给出 Base + LCA 两阶段方案, 并给出 500-1500 bps 区间相对 PCM / Opus / EnCodec / DAC 的客观负载-质量曲线.

**证据 (来自 exp1-4)**:
- exp1: 受约束 NAS 得到轻量发送端 encoder (params -86.9%, MACs -89.5%, RTF -55.1%)
- exp2: SCIT-Base 训练完成, dev/mel best=1.124
- exp3 v2: LCA 微调后 6/6 鲁棒性指标全正
- exp4: 500/1000/1500 bps 三档同码率全面优于 DAC/EnCodec; L=3 (1.5 kbps) WER ≈ Opus 6 kbps

### 第 5 步 (§6.2-6.4): 通过四组消融, 隔离哪些训练成分必要

**主张**: 不是所有合理的训练设计都必要. 本文通过单因素消融区分 necessary vs optional.

**证据 (来自 exp5)**:
| 消融 | 关键发现 | 文献缺口 |
|---|---|---|
| A1 NAS vs hand-designed | params -86.9% / MACs -89.5% (质量等价对照尚未闭环) | -- |
| A2 Base vs LCA (clean) | mel_l1 -3-4%; 主要价值在鲁棒性 | -- |
| A3 distill=0 vs distill=30 | dev/mel +1.43, WER +7-9 pp; distill=0 反而用更多码字但质量更差 | **Q5 完全空白** |
| A5 弱扰动 vs 强扰动+consistency | 弱扰动 6 项 robust_imp ≈ 0; 强扰动+consistency 6/6 全正, mel_l1 12/12 cell | **Q4 接近空白** |

### 第 6 步 (§7 Discussion + §8 Limitations): 主动收紧

**主张**: 不取代通用 codec; 不创造新机制; 评估限于 8 条同源样本; 三用户系统验证留 future work.

---

## 3. Contributions (4 条, 写入 §1 末尾)

1. **Index-only transmission interface with shared RVQ codebooks for speech communication**: 把神经语音 codec 重新表述为通信问题, 给出 ideal / packed / packetized 三层 payload 核算, 在 500/1000/1500 bps 三档操作点上完整报告. (对应 Q2 联合缺口的主轴)
2. **A constrained NAS protocol for the transmitter encoder under a fixed latent interface**: 在固定 sample rate / downsample / latent rate / RVQ 接口的约束下搜索发送端 encoder, 把"轻量化收益"和"通信接口变化"解耦.
3. **A two-stage training recipe (Base + LCA) yielding a usable 500-1500 bps operating band**: 给出具体可用的训练方案, 并以 PCM / Opus / EnCodec / DAC 为对照建立客观负载-质量曲线. (对应 Q3 缺口的 L-knob 训练)
4. **Ablations isolating which training components are necessary vs optional**: 对 NAS encoder, Base vs LCA, 蒸馏强度 (30 vs 0), 索引扰动强度与一致性损失 (弱扰动 vs 强扰动+consistency) 进行单因素消融, 定位真正不可或缺的训练成分. (对应 Q4 + Q5 缺口)

---

## 4. 章节结构

### §1 Introduction (1-1.5 页)
- 1.1 场景: 6G / 边缘 / 受限链路语音通信; 500-1500 bps 极低负载需求
- 1.2 现状: 神经 codec 主要在 3+ kbps; 极低码率有 SAC / Codec-SUPERB / SemantiCodec / DualCodec 零散点, 无稠密 RVQ 曲线
- 1.3 研究问题: 共享码本前提下能否仅传索引在 500-1500 bps 维持可用性?
- 1.4 Contributions (4 条, 见 §3)
- 1.5 边界声明 (3-4 句, 见 §1.4)

### §2 Related Work (1 页)

#### 2.1 Neural audio codecs (SoundStream / EnCodec / DAC / WavTokenizer)
重建质量优先, 多在 3+ kbps. 不框定通信问题, 不做 payload 核算.

#### 2.2 离散语音 token 与分层 RVQ (SpeechTokenizer / Mimi / DualCodec)
SpeechTokenizer 引入 HuBERT 蒸馏; Mimi 蒸馏 WavLM; DualCodec 用 RVQ dropout 但 framing 是 tokenizer for LM.

#### 2.3 极低码率语音 codec (SAC, Codec-SUPERB)
SAC: single-codebook, 525/875 bps, 不同码率训不同模型, 无 L-knob, 无 payload 核算, 用 frozen glm4voice 而非 HuBERT 蒸馏.
Codec-SUPERB: 跨 codec sampling 0.68-1.40 kbps, 非单 codec 稠密曲线.

#### 2.4 语音语义通信 (DeepSC-S, DeepSC-ST, SyncSC, Glaris, LargeSC-era)
DeepSC-S/ST: 端到端 JSCC, 连续信道符号. SyncSC: 仅定性 overhead 讨论. Glaris (2512.08203): mask-token PLC 训练, 8-18 kbps, 无强弱消融, 无 mel-L1 一致性. LargeSC-era: 抽象层"2-4× 减码率"无核算.

#### 2.5 Differentiation summary (一段话)
显式列出 SCIT vs SAC vs Glaris vs DualCodec vs SpeechTokenizer 在七维度的占位差异, 引用 [citation_pool.md](citation_pool.md) 的 Tier 1 五篇.

### §3 Problem Formulation (0.5 页)

#### 3.1 预共享知识假设
码本 C* = {C_1*, ..., C_M*} 与解码器 D 在通信前部署到双方, 不计入信道负载.

#### 3.2 索引-only 信道
信道传 I_L = I_{1:L,:}, L ∈ {1, ..., M} 是负载控制变量.

#### 3.3 三层负载核算 (本文核心 framing 之一)
- ideal index bitrate: `R_index(L) = L · f_q · ⌈log₂ K⌉`
- packed payload bitrate: bit-packed indices after serialization
- packetized payload bitrate: packed + header + session metadata
- 本文主表使用 ideal; packed 与 packetized 在 §6.1 报告; 真实网络验证留 future work.

#### 3.4 评价口径
clean 重建质量 + 索引扰动鲁棒性两条评价轴.

### §4 SCIT-Speech: System Instance (1.5 页)

#### 4.1 系统总览图 (Figure 1)
x → encoder → Z → RVQ → I_{1:M} → keep I_{1:L} → channel → shared codebook lookup → decoder → x_hat

#### 4.2 受约束 NAS 轻量发送端 encoder
搜索边界: 固定 sample_rate=16 kHz, downsample=320, latent_rate=50, d=1024, M=3, K=1024, stride [8,5,4,2]. 选定 nas_seed42_000896. **不声称"质量等价已证"; 只声称"效率显著降低且后续完整训练成立".**

#### 4.3 共享 RVQ 码本与 L-knob
M=3, K=1024, f_q=50 → R(L) = 500L bps, 三档 500/1000/1500 bps. **L 是训练时随机采样的传输参数, 不是 post-hoc truncation.**

#### 4.4 SCIT-Speech-Base 训练
端到端从零训练 encoder + RVQ + decoder. 损失: waveform L1 + multi-scale mel + RVQ commitment + adversarial + feature matching + **HuBERT 语义蒸馏 (λ=30)**.

#### 4.5 SCIT-Speech-LCA (低负载信道感知适配)
- 训练时随机采样 L ∈ {1, 2, 3}
- ChannelSim: clean / index dropout (previous-index replacement) / light substitution
- **强扰动**: p_drop ∈ {0, 0.05, 0.10}, p_sub ∈ {0, 0.01, 0.03}
- **Consistency loss**: clean-decoded 与 perturbed-decoded 在 mel 空间 L1 一致, λ_cons=0.5
- 总目标: α·L_full + β·E_{L,c} L_comm + λ_commit·L_commit + λ_sem·L_sem + λ_adv·L_adv + λ_fm·L_fm + λ_cons·L_cons
- End-to-end 微调 (不冻结 decoder)

### §5 Experimental Setup (1 页)

#### 5.1 数据
LibriSpeech train-clean-100; 固定 8 条评估样本 (10-15 秒); 16 kHz 单声道.

#### 5.2 模型规模与接口 (一张表)

#### 5.3 对比方法
SCIT-Base / SCIT-LCA / **SCIT-LCA(weak-perturb)** (消融变体, 不写 v1/v2) / PCM 256 kbps / Opus 6/8/12/16/24 kbps / EnCodec 1.5/3/6/12 kbps / DAC n_q=1..12. AMR-WB 缺席原因列入 Limitations.

#### 5.4 评价指标
波形 L1, mel-L1, SI-SNR, 相关系数, STOI, PESQ-WB, Whisper base.en WER/CER.

**Rigor 声明 (两句话写入正文):**
- "Due to the limited evaluation set (n=8), we report descriptive statistics without significance testing."
- "Whisper base.en achieves WER ≈ 0.075 on PCM, which serves as a measurement floor; reported WER differences should be read relative to this floor rather than as absolute intelligibility."

#### 5.5 索引扰动鲁棒性评价口径
robust_imp = degradation(Base) - degradation(LCA), 跨 12 个 (L, condition) cell 聚合.

### §6 Results (2-2.5 页)

#### 6.1 主结果: 500-1500 bps 负载-质量曲线 (Figure 2 + Table 2)
三组同码率对照:
- 500 bps (SCIT-LCA L=1 vs DAC n_q=1): WER 0.461 vs 0.922
- 1000 bps (SCIT-LCA L=2 vs DAC n_q=2): WER 0.188 vs 0.305
- 1500 bps (SCIT-LCA L=3 vs DAC n_q=3 / EnCodec 1.5 kbps): WER 0.142 vs 0.152 / 0.302
跨码率单点对照: SCIT-LCA L=3 (1.5 kbps) WER 0.142 ≈ Opus 6 kbps WER 0.147.

**Rigor 措辞**: "in the 500-1500 bps regime"; 不写"全面优于".

#### 6.2 Base vs LCA (clean 条件)
三档 mel-L1 改善 -3.4 ~ -4.6%; SI-SNR 略降 -- LCA 把 clean 精度换给鲁棒性, 不掩饰.

#### 6.3 索引扰动鲁棒性: 弱扰动 vs 强扰动+consistency (Figure 3 热力图 + Table 4)
- weak-perturb 变体: 6 项 robust_imp ≈ 0 或负
- 强扰动+consistency: 6 项全正, mel_l1 12/12 cell, PESQ-WB 10/12, STOI 9/12
- 代表性 cell: L=3 dropout-mid PESQ 退化 +0.208→+0.161 (-22%); L=3 dropout-high mel_l1 +0.088→+0.072 (-18%)

#### 6.4 设计选择消融 (Figure 4 四联柱状图)

**A3 蒸馏必要性 (核心消融)**:
- distill=0 vs distill=30 同 step 47500 dev/mel 差 +1.426
- clean WER@L=1/2/3 退 +9.4 / +7.4 / +8.4 pp; PESQ@L=3 -0.118
- **反直觉观察 (弱措辞)**: "We observe that the distill=0 variant uses more codewords (L1: 228 vs 179) yet achieves worse quality. One possible interpretation is that distillation organizes encoder representations into content-relevant subspaces, but a precise causal account requires a sweep over distillation strength which we leave to future work."

**A5 弱扰动 vs 强扰动+consistency**: 见 §6.3.

**A1 NAS efficiency**: params -86.9% / MACs -89.5% / RTF -55.1%. 质量对照未闭环, 谨慎陈述.

**A2 Base vs LCA**: 见 §6.2.

**A4 random-L**: confounded with ChannelSim+consistency, 列 future work.

末尾一句: "Codebook usage statistics are reported in Appendix A; their interpretation is discussed in §8 (Limitations)."

### §7 Discussion (0.5-1 页)

#### 7.1 SCIT-Speech 的合理定位
500-1500 bps 极低负载操作区间的 shared-codebook transport interface; **不是通用 codec**; 不与 Opus 12 kbps+ / DAC 6 kbps+ 竞争.

#### 7.2 为什么蒸馏比码本利用率更重要
A3 反直觉发现的弱措辞解释: 低负载下"码字数量"未必是关键, "每个码字承载的信息密度"可能更关键. 但精确因果账需要 distill_lambda sweep, 留 future work.

#### 7.3 为什么强扰动 + consistency 必要
弱扰动下解码输出几乎无差异, 缺少梯度信号; 强扰动 + 显式一致性把 invariance 显式化.

#### 7.4 与大模型语义通信的关系
SCIT 索引可作为大模型 (LargeSC, Glaris) 恢复链路的低负载前端; 两条路径互补.

### §8 Limitations (独立节, ~1/3 页, 短而密)

1. n=8 train-clean 同源样本; 跨语料 (test-clean, test-other, VCTK, AISHELL, 真实通信语音) 未验证.
2. 无 MOS / AB 主观听测.
3. AMR-WB baseline 缺 (ffmpeg 6.1.2 工具链问题).
4. random-L 单独贡献被 ChannelSim+consistency confound (A4).
5. NAS 质量等价对照未闭环 (hand-designed 同条件全训未做).
6. Codebook 利用率偏低 (L1 dead 82.5%, 总 22.7%); A3 暗示并非 collapse 导致主结果, clean reinit study 留 future work.
7. DAC 训练域含音乐, SCIT 仅语音; 部分低码率优势可能来自 domain specialization.
8. Index-perturbation ≠ wireless channel model; 未与 FEC / channel coding 联合评估.
9. Packetized payload 与多用户实时系统验证 (实验六) 未做.

### §9 Conclusion (1 段)
回到主张: 用预共享 RVQ 码本把信道传输对象重新定义为分层离散索引, 在 500-1500 bps 区间形成可解释、可控、可用的语音通信操作带; 强扰动 + consistency 改善索引扰动鲁棒性; 蒸馏是必要正则化. 后续工作沿 Limitations 顺序展开.

### Appendices
- A. Codebook usage statistics (per L: usage rate, dead-code ratio, perplexity)
- B. Implementation details (batch size, lr, optimizer, training duration, hardware, seeds)
- C. Audio sample listing (固定 8 条, L=1/2/3, Base vs LCA, baseline)

---

## 5. Figures and Tables

| 编号 | 类型 | 内容 | 数据来源 |
|---|---|---|---|
| Fig 1 | 系统图 | SCIT-Speech 端到端框架 + L 控制器 | 设计图 |
| Fig 2 | 曲线 | bps vs (WER / STOI / PESQ-WB / mel-L1) | exp4 |
| Fig 3 | 热力图 | LCA robust_imp 跨 (L, condition) | exp3 v2 |
| Fig 4 | 四联柱状 | NAS efficiency / distill0 vs 30 / Base vs LCA / weak vs strong | exp1 + exp5 |
| Tab 1 | 设置表 | 模型与传输接口 | §3 |
| Tab 2 | 主结果 | 500/1000/1500 bps 同码率对照 | exp4 |
| Tab 2b | payload 核算 | ideal / packed / packetized 三层 | exp4 |
| Tab 3 | clean 表 | Base vs LCA 三档 clean | exp3 v2 |
| Tab 4 | 鲁棒性 | weak vs strong robust_imp 6 指标聚合 | exp3 v1 + v2 |
| Tab 5 | 消融 | A1/A2/A3/A5 (A4 列 future work) | exp5 |
| Tab 6 (Appendix) | codebook 诊断 | L1/L2/L3 usage / dead / perplexity | exp2 |
| Tab 7 (Appendix) | 音频样本 | 8 条样本 × 6 条件链接 | exp4 |

---

## 6. Abstract 草稿 (六句, 中文论文用版)

> 面向 500-1500 bps 极低带宽语音通信, 现有神经语音 codec 在该区间缺少稠密的负载-质量曲线和可控传输接口. 本文将 RVQ 语音编码器重新表述为"共享码本 + 索引-only 传输接口", 给出 ideal / packed / packetized 三层负载核算, 并把层数 L 作为可控的传输参数. 在系统实例 SCIT-Speech 中, 通过受约束 NAS 得到轻量发送端 encoder (params -86.9%, MACs -89.5%), 训练 SCIT-Speech-Base 与适配模型 SCIT-Speech-LCA. 在 500-1500 bps 区间, SCIT-Speech-LCA 的 mel-L1 / STOI / PESQ-WB / WER 全部优于同码率 DAC 与 EnCodec; L=3 (1.5 kbps) 的 WER 与 Opus 6 kbps 相当. 训练配方消融显示: HuBERT 语义蒸馏明确必要 (distill=0 时 dev/mel 退化 +1.43, WER 退 7-9 pp), 弱索引扰动训练不足以产生鲁棒性收益, 强扰动 + clean-vs-perturbed 一致性损失才在 6/6 客观指标上稳定改善. 当前评估限于 LibriSpeech train-clean-100 的 8 条同源样本与客观指标; 跨语料、主观听测、真实通信链路验证留待未来工作.

英文版 Abstract 在写正式 LaTeX draft 时基于此结构翻译.

---

## 7. 命名固定 (避免内部代号进入正文)

| 内部 | 论文用名 |
|---|---|
| LCA v2 | **SCIT-Speech-LCA** (正式模型) |
| LCA v1 | **SCIT-Speech-LCA(weak-perturb variant)** (消融变体) 或 "a weak-perturbation variant" |
| Base | **SCIT-Speech-Base** |
| nas_seed42_000896 | "the selected NAS encoder" |
| distill_loss_lambda | λ_sem |

正文不出现 "v1 / v2 / nas_seed42_000896 / distill_loss_lambda=30" 等内部代号.

---

## 8. 写作风险点 (写正文时主动检查)

1. **Q1 部分关闭风险**: SAC (2510.16841, 2025-10) 是直接对照, §1 / §2 / §6.1 都需要点名引用并区分.
2. **A3 反直觉发现**: deep-research 未独立验证文献新颖性, 论文用弱措辞 ("we observe"; "one possible interpretation"; 留 future work).
3. **n=8 显著性**: §5.4 主动声明描述性统计, 不做显著性检验.
4. **Whisper 测量地板**: §5.4 明示 PCM WER ≈ 0.075 是 Whisper 自身误差地板.
5. **DAC 域不匹配**: Limitations 写, 不进 §6 正文.
6. **Codebook 利用率**: 数字进 Appendix, 解释进 Limitations, §6.4 末尾一句导引.
7. **"全面优于"**: 全文限定 "in the 500-1500 bps regime"; 不写无限定的 "outperforms".
8. **Channel-aware 措辞**: 全文避免, 改用 "index-perturbation training" / "low-load adaptation".
9. **AMR-WB 缺**: Limitations 第 3 条; 投稿前确认是否补 (gyan.dev ffmpeg 完整版).
10. **三用户系统**: 不进正文; 一句 future work 提及即可.

---

## 9. 待决事项 (落正式初稿前)

1. 是否补 hand-designed encoder 同条件全训以闭环 NAS 质量等价 (decision: 暂不, future work).
2. 是否补 test-other / VCTK 跨语料评估 (decision: 暂不, Limitations 写).
3. 是否补 AMR-WB baseline (decision: 暂不, ffmpeg 工具链问题写入 Limitations).
4. 是否做 distill_lambda sweep (decision: 暂不, future work; A3 用 0 vs 30 二值对照已足够支撑必要性主张).
5. 是否做实验六三用户系统 (decision: 暂不, future work).
6. 投稿目标决定:
   - IEEE JSAC / TWC / TCOM (推荐, Path A 最匹配)
   - 《通信学报》/《电子学报》中文期刊 (中文 draft 已可用)
   - INFOCOM system track
   - Interspeech / ICASSP (会议短文)

投稿目标决定后再做最后一轮 §2 Related Work 扩写与 reproducibility 子节.

---

## 10. 文档关系

- 本文件 (v2 大纲): 当前论文写作的活跃指导
- [literature_review_2026_06_02.md](literature_review_2026_06_02.md): 文献缺口 + 引用判据 (Q1-Q5)
- [citation_pool.md](citation_pool.md): 23 篇验证源, 按 Tier 分级
- [paper_outline_low_load_semantic_speech.md](paper_outline_low_load_semantic_speech.md): v1 旧大纲, 历史参考
- [paper_drafts/scit_speech_cn_draft.md](paper_drafts/scit_speech_cn_draft.md): 当前初稿, 后续按本大纲对齐
- [实验记录.md](实验记录.md): 实验真实状态记录, 数字回填来源
- [实验手册.md](实验手册.md): 实验路线手册, 已部分覆盖本大纲
