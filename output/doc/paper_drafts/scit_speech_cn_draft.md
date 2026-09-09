# SCIT-Speech：基于共享码本索引传输的极低负载语音语义通信

> 中文论文初稿，基于截至 2026-06-02 的本地实验结果。所有数值均来自当前 `output/experiments/` 下的真实实验记录，未补写未完成实验。

## 摘要

面向 6G、边缘智能、低功耗终端和受限链路，语音通信系统需要在远低于常规宽带语音编码码率的负载下维持可懂度、自然度和任务层可用性。传统语音 codec 与多数神经音频 codec 主要面向信号级重建质量优化，而近期语音语义通信方法虽强调语义保持，却常依赖连续信道映射、大型语音模型或难以显式核算的传输接口。本文围绕 500-1500 bps 这一极低负载区间，构建并实证评估 **SCIT-Speech**，一种基于共享码本索引传输的语音语义通信系统实例。该系统将残差向量量化（RVQ）语音 codec 重述为“共享码本 + index-only transmission”接口：发送端和接收端预共享 RVQ 码本与重建模块，信道中仅传输前 `L` 层离散索引。由于当前实例采用 `M=3`、`K=1024` 和 50 Hz 潜在帧率，理想索引负载为 `R(L)=500L` bps，对应 `L=1/2/3` 时的 500/1000/1500 bps 操作点。

在 LibriSpeech train-clean-100 上的初步实验表明，SCIT-Speech-LCA 在 500-1500 bps 区间相对于 DAC、EnCodec 和低码率 Opus 等基线表现出更强的低负载可用性。1.5 kbps 下，SCIT-Speech-LCA 的 STOI 为 0.879，PESQ-WB 为 1.891，Whisper base.en WER 为 0.142；作为对照，EnCodec 1.5 kbps 的 WER 为 0.302，DAC 1.5 kbps 的 WER 为 0.152，Opus 6 kbps 的 WER 为 0.147。进一步地，低负载信道感知适配在 6 个客观指标上均获得正向鲁棒性改善，其中 mel-L1 在 12/12 个扰动单元上相对 Base 退化更小。消融实验显示，HuBERT 语义蒸馏是当前训练配方中的关键成分：去除蒸馏后，dev/mel 在同 step 比较下退化约 +1.4，WER 在 `L=1/2/3` 上退化 7-9 个百分点。受约束 NAS 搜索得到的发送端 encoder 进一步将参数量、MACs 和 RTF 分别降低约 86.9%、89.5% 和 55.1%。本文结果说明，分层离散索引可以作为极低负载语音语义通信中的可解释、可控传输单元；同时，当前评估仍限于小样本、同源语料和客观指标，跨语料库泛化、主观听测与真实网络验证仍需后续补充。

**关键词**：语义通信；语音通信；共享码本；索引传输；残差向量量化；低码率语音；语义蒸馏；信道扰动训练

## 1. 引言

### 1.1 背景与动机

未来通信系统正在从单纯追求比特级可靠传输，逐步走向面向语义和任务的高效信息传输。对于语音通信而言，这一转向具有直接工程意义：在应急通信、自组织中继、卫星窄带回传、嵌入式语音前端和多用户边缘网关等场景中，单路语音能够分配到的实际 payload 可能远低于常规宽带语音 codec 的工作区间。此时，系统目标不再是尽可能逼近无损波形，而是在极低负载下保留足以支持理解、交互和决策的语音信息。

传统语音编码和近年来的神经音频压缩方法已经在低码率语音和音频重建上取得显著进展。SoundStream、EnCodec、DAC、SpeechTokenizer 和 WavTokenizer 等工作表明，端到端神经 codec、RVQ 码本和离散 token 表示能够在压缩、重建和生成任务中取得良好效果 [1-5]。然而，这类方法多数仍以音频重建质量或通用音频 tokenization 为主要目标，500-1500 bps 区间通常不是其系统化评估的核心操作带。另一方面，DeepSC-S、DeepSC-ST 以及近期结合大型语音模型的语义通信方法开始从语义层面重新定义语音传输 [6-10]，但其中不少方法依赖高维连续隐表示、端到端 JSCC 结构或大模型恢复机制，传输接口的可解释性、轻量性和码率核算仍有进一步明确的空间。

### 1.2 研究问题

本文关注一个更窄但更可操作的问题：

> 如果发送端和接收端预先共享分层离散码本，语音通信是否可以只传输紧凑的码本索引序列，并在 500-1500 bps 的极低负载下保持可用的语音自然度、可懂度和语义信息？

为回答这一问题，本文构建 **SCIT-Speech**（Shared-Codebook Index Transmission Speech Model）。SCIT-Speech 不把自身定位为又一个通用神经语音 codec，也不把分层 RVQ token 本身作为新的机制贡献；相反，本文将离散语音索引重新定位为通信信道中的传输符号，围绕共享码本、层数可控索引传输、低负载适配和鲁棒性消融建立一个完整实验框架。该框架的核心约束是区分“预共享系统知识”和“真实信道负载”：码本、解码器和系统配置属于离线共享知识，运行时信道只承载离散索引序列。

这一设定使得 RVQ 层数 `L` 成为一个直接的通信控制旋钮。对于当前系统实例，码本大小为 `K=1024`，潜在帧率为 50 Hz，因此每增加一层 RVQ 索引，理想负载增加 500 bps。相比仅报告 nominal bitrate，本文进一步强调 bit-packed load 与 packetized payload 的区分，因为在 1 kbps 量级下，字节对齐、包头和会话元数据都可能显著影响工程负载解释。

### 1.3 贡献

本文贡献如下：

1. **共享码本索引传输接口**：将 RVQ 神经语音 codec 重述为预共享码本条件下的 index-only transmission 问题，信道中仅传输码本索引而非波形、连续特征或码本本身。
2. **可控负载建模**：给出分层索引传输负载公式 `R(L)=L f_q ceil(log2 K)`，并在当前系统中形成 500/1000/1500 bps 三个可解释操作点。
3. **低负载信道感知适配**：在 Base 模型基础上引入 random-L 操作点训练、强索引扰动和 clean/perturbed 重建一致性损失，使 LCA 模型在索引扰动下获得更小退化。
4. **系统性实验和消融**：与 DAC、EnCodec、Opus 和 PCM 进行负载-质量比较，并通过 NAS、语义蒸馏、LCA 和 ChannelSim 消融定位当前训练配方中的必要成分。

### 1.4 范围与边界

本文的结论边界需要提前明确。第一，当前工作是一项实证系统研究，而不是新的通信范式或新的 codec 损失机制。第二，本文不主张在所有码率上击败传统 codec 或神经 codec；对于 6 kbps 以上的常规通信条件，Opus、DAC 和 EnCodec 仍然是成熟且强大的选择。第三，本文当前结果主要证明 SCIT-Speech 在 500-1500 bps 低负载操作区间中的可行性和训练配方有效性，而不构成跨语料库、跨语言、真实网络或工业部署场景下的最终结论。这些边界将在实验设置和局限性中进一步说明。

## 2. 相关工作

### 2.1 传统语音编码与神经音频压缩

传统语音 codec 和音频 codec 已经长期服务于低码率通信。近年来，神经音频压缩进一步利用端到端神经网络、向量量化和生成式重建提升主观音质。SoundStream 使用端到端编码器-解码器和残差向量量化实现可变码率神经音频压缩 [1]；EnCodec 进一步提升了实时神经音频压缩质量 [2]；DAC 和改进 RVQGAN 类方法强调高保真音频压缩 [3]；WavTokenizer 则探索更高压缩率的离散声学表示 [5]。这些工作证明了离散码本和神经重建在音频压缩中的有效性。

与这些方法不同，本文的核心目标不是在所有码率上获得最高音频重建质量，而是定义一种通信负载可解释的 shared-codebook index interface。在该接口下，系统明确区分预共享知识（码本与解码器）和实际信道负载（索引序列），从而能够以 `L` 控制通信成本。

### 2.2 离散语音表示与语音 tokenizer

SpeechTokenizer 等工作表明，基于 RVQ 的语音 tokenizer 可以在不同量化层中形成语义和声学信息的分层结构 [4]。例如第一层 token 往往承担更多内容相关信息，后续层补充音色和细节。此类研究为语音语言模型、语音生成和统一 speech-token 表示提供了基础。

本文不声称重新发明分层 RVQ tokenizer。相反，本文利用这一类模型的离散化能力，把分层 token 从“模型内部表示”转化为“通信信道符号”。也就是说，本文关心的不是 token 是否可用于 speech LM，而是 token 是否可作为可控码率语音语义通信的传输单元。

### 2.3 语音语义通信

DeepSC-S、DeepSC-ST 等工作将语义通信思想引入语音传输，强调在噪声信道中保留语义信息而非逐比特恢复 [6,8]。近期 LargeSC、Glaris 等工作进一步结合大型语音模型、离散 token、丢包恢复和不等错误保护机制，推动语音语义通信向更复杂系统发展 [9,10]。

这些工作与本文目标相邻，但本文强调一个更轻量、更可解释的路径：不依赖大型语音基础模型作为恢复核心，而是在共享 RVQ 码本条件下直接研究索引层数、通信负载、客观质量和索引扰动鲁棒性之间的关系。这种设定有助于形成清晰的基线系统，也有利于后续接入纠错、packetization 和多用户路由。

## 3. 方法

### 3.1 系统总览

SCIT-Speech 由发送端表示编码器、共享 RVQ 码本、索引级信道和接收端重建模块构成。给定输入语音 `x(t)`，发送端首先通过编码器 `E_theta` 得到连续语音表示：

```text
Z = E_theta(x)
```

随后，`Z` 被分层 RVQ 码本量化为 `M` 层离散索引：

```text
I = {i_{m,t}} ,  m = 1,...,M,  t = 1,...,T_q
```

其中 `i_{m,t}` 是第 `m` 层码本在第 `t` 个潜在时间步上的索引。发送端和接收端预共享码本 `C*={C_1*,...,C_M*}` 及重建模块。通信过程中，信道只传输前 `L` 层索引：

```text
I_L = I_{1:L,:},  L <= M
```

接收端根据接收到的索引在共享码本中查表，得到量化表示并重建语音：

```text
\hat{x} = D(C*(I_L))
```

在本文系统实例中，`M=3`，`K=1024`，采样率为 16 kHz，encoder 下采样率为 320，因此潜在帧率 `f_q = 16000 / 320 = 50` steps/s。

### 3.2 通信负载建模

对于每层 RVQ 码本大小 `K`，每个索引需要：

```text
b = ceil(log2 K)
```

比特表示。若每秒潜在帧数为 `f_q`，传输前 `L` 层索引，则理想 bit-packed 索引负载为：

```text
R_index(L) = L * f_q * ceil(log2 K)
```

本文中 `K=1024`，因此 `ceil(log2 K)=10`；又因为 `f_q=50`，所以：

```text
R_index(L) = 500L bps
```

对应：

| 传输层数 L | 理想索引负载 |
|---:|---:|
| 1 | 500 bps |
| 2 | 1000 bps |
| 3 | 1500 bps |

相比 16 kHz、16-bit 单声道 PCM 的 256 kbps，`L=3` 的理想索引负载仅为约 0.59%。需要注意，本文区分 ideal bit-packed load 与实际 packetized payload；`.npy`、JSON header、TCP 包头等工程表示开销不能混同为理论索引负载。

### 3.3 轻量发送端 encoder 的 teacher-guided staged NAS

发送端计算成本对于低负载通信系统同样重要。即使信道中只传输少量 RVQ 索引，若发送端 encoder 过大，系统仍难以部署在移动端、边缘设备或实时通信场景中。因此，本文在保持后续通信接口不变的前提下，使用 teacher-guided staged NAS 搜索轻量 encoder 架构。

在这一设定下，NAS 仅改变发送端特征提取网络的结构复杂度，而不改变索引生成、索引传输与接收端重建之间的接口关系。对于任意候选 encoder，系统仍输出相同帧率、相同维度的 latent 表示，并接入相同的 RVQ 与解码接口。因此，不同候选之间的比较可以集中在参数量、MACs、RTF 和代理重建质量上，而不会混入码本大小、索引负载或接收端模型变化带来的影响。

NAS 搜索采用分阶段的代理训练与剪枝流程。首先，搜索空间围绕 SEANet encoder 的宽度、时序建模和局部算子展开，包括初始通道数、LSTM 层数、残差压缩比例、激活函数，以及各下采样阶段的卷积算子选择。候选算子包含标准卷积、深度可分离卷积、空洞卷积和跳连，同时搜索是否在对应阶段使用 SE 模块。所有候选都被约束为输出 1024 维 latent，并保持 320 倍总下采样率，从而保证其可以直接接入后续 RVQ 和 decoder。

每个候选结构首先经过计算量门控。本文使用固定长度的 16 kHz 输入估计 encoder 与代理 decoder 的 FLOPs，超过预设计算预算的候选会被直接剪枝，不进入训练阶段。通过门控的候选随后在 NAS 子集上进行短程代理训练，目标函数由波形重建损失和 mel 频谱损失组成。训练过程中，Optuna 根据中间 epoch 的代理 loss 执行早停剪枝，并使用 TPE 采样器根据历史 trial 结果继续提出新的结构组合。

完成搜索后，本文根据代理 loss、SI-SNR、FLOPs 和候选重建样例对结构进行筛选，并导出最佳 encoder 配置用于后续完整 codec 训练。最终选定的 NAS encoder 相对于手工设计 encoder 将参数量从 67.7M 降至 8.87M，MACs 从 5.87 G/s 降至 0.618 G/s，RTF mean 从 0.00819 降至 0.00368，分别减少约 86.9%、89.5% 和 55.1%。由于 hand-designed encoder 尚未完成同等 60 epoch 全量重训，本文不将该结果表述为最终质量优越性结论，而仅将其作为发送端效率收益的 profiling 证据；质量保持性仍需在等预算重训、完整客观指标和主观评测中进一步验证。

### 3.4 SCIT-Speech-Base 训练

SCIT-Speech-Base 基于 NAS 选出的 encoder 架构从头训练，包括 encoder、RVQ codebooks 和 decoder。训练数据为 LibriSpeech train-clean-100 及对应 HuBERT 表征。训练目标包括波形/频谱重建相关损失、对抗训练损失以及 HuBERT 语义蒸馏损失。语义蒸馏用于将低层索引组织到更有内容信息密度的表示空间中，从而提升低负载操作点的可懂度。

当前主模型采用 `distill_loss_lambda=30`。消融实验显示，若将 `distill_loss_lambda` 设为 0，其短期重建收敛可能在最初阶段不差，但中后期明显落后于 distill30。到 step 47500 时，distill0 的 dev/mel 为 3.202，而 distill30 为 1.776，差距达到 +1.426。

### 3.5 低负载信道感知适配

Base 模型完成后，本文进一步训练 SCIT-Speech-LCA（Low-load Channel-aware Adaptation）。LCA 的目标是让模型适应两个通信条件：

1. **多负载操作点**：训练时随机采样 `L=1/2/3`，避免模型只在 full-depth 条件下最优。
2. **轻量索引扰动**：训练时引入 index dropout 和 index substitution，以模拟轻量信道错误。

第一版弱扰动 LCA 未能带来稳定鲁棒性收益。因此，本文采用 v2 方案：将 dropout/substitution 概率提升约 3-6 倍，并加入 clean-decoded 与 perturbed-decoded 输出之间的 mel-L1 consistency loss。v2 配置中，`p_drop` 包含 0.05 和 0.10，`p_sub` 包含 0.01 和 0.03，`lambda_consistency=0.5`。

实验表明，弱扰动训练几乎不产生鲁棒性收益，而强扰动 + consistency loss 能在 6 个客观指标上获得正向 robustness improvement。这说明低负载语音索引传输不仅需要“看到扰动”，还需要足够强的扰动信号和显式一致性约束。

## 4. 实验设置

### 4.1 数据与模型

当前实验使用 LibriSpeech train-clean-100 作为训练来源。固定评估样本为 8 条 LibriSpeech train-clean-100 utterances，长度约 10-15 秒。所有音频统一为 16 kHz 单声道。虽然该设置足以完成系统闭环和初步消融，但不应被解释为跨语料泛化结论。

模型接口如下：

| 项目 | 值 |
|---|---:|
| sample rate | 16 kHz |
| encoder downsample rate | 320 |
| latent rate | 50 steps/s |
| RVQ layers `M` | 3 |
| codebook size `K` | 1024 |
| transmitted layers `L` | 1/2/3 |
| ideal bitrate | 500/1000/1500 bps |

### 4.2 对比方法

实验四中评估以下方法：

| 方法 | 配置 |
|---|---|
| SCIT-Speech-Base | L=1/2/3 |
| SCIT-Speech-LCA v2 | L=1/2/3 |
| PCM | 16-bit, 16 kHz |
| Opus | 6/8/12/16/24 kbps |
| EnCodec | 1.5/3/6/12 kbps |
| DAC | n_q=1/2/3/4/6/9/12 |

AMR-WB、Codec2 和 ViSQOL 因当前 Windows/conda 工具链限制未纳入主表。该缺口在局限性中说明。

### 4.3 评价指标

本文使用波形 L1、mel-L1、SI-SNR、相关系数、STOI、PESQ-WB 和 Whisper base.en ASR 得到的 WER/CER。需要注意，PCM lossless 条件下 Whisper 在 8 条样本上的 WER 约为 0.075，因此 WER 结果应主要理解为相对于该测量下界的退化，而不是绝对人类听懂率。

### 4.4 索引扰动鲁棒性评价

对于 LCA 鲁棒性，本文比较 Base 与 LCA 在 clean 和扰动条件下的退化差异：

```text
robustness_improvement = degradation(Base) - degradation(LCA)
```

若该值为正，表示 LCA 在相同扰动下退化更小。实验覆盖 `L=1/2/3` 与 4 个扰动条件，共 12 个 `(L, condition)` 单元。

## 5. 结果

### 5.1 500-1500 bps 低负载区间的质量表现

表 1 给出 SCIT-LCA 与同码率附近最强 baseline 的比较。

| 操作点 | 最佳 baseline | SCIT-Speech-LCA |
|---|---|---|
| 500 bps | DAC n_q=1: STOI 0.606, PESQ 1.055, WER 0.922 | L=1: STOI 0.797, PESQ 1.347, WER 0.461 |
| 1000 bps | DAC n_q=2: STOI 0.730, PESQ 1.148, WER 0.305 | L=2: STOI 0.857, PESQ 1.709, WER 0.188 |
| 1500 bps | DAC n_q=3: STOI 0.799, PESQ 1.271, WER 0.152；EnCodec 1.5 kbps: WER 0.302 | L=3: STOI 0.879, PESQ 1.891, WER 0.142 |

结果显示，SCIT-Speech 在 500-1500 bps 区间形成了清晰的可用操作带。在 1.5 kbps 下，SCIT-LCA 的 mel-L1 为 0.859，优于 EnCodec 1.5 kbps 的 1.373 和 DAC n_q=3 的 1.203；WER 为 0.142，略优于 DAC n_q=3 的 0.152，并明显优于 EnCodec 1.5 kbps 的 0.302。

一个值得注意的对照是：SCIT-LCA L=3 在 1.5 kbps 下 WER 为 0.142，Opus 6 kbps 下 WER 为 0.147。该结果不意味着 SCIT 全面优于 Opus；事实上，Opus 在 12 kbps 以上表现非常强。但它说明，在极低负载区间，SCIT 可以用约 1/4 的带宽达到接近 Opus 6 kbps 的 ASR 可懂度水平。

### 5.2 SCIT-Base 与 SCIT-LCA 的 clean 条件比较

在 clean 条件下，LCA v2 相比 Base 的提升较温和：

| L | Base mel-L1 | LCA mel-L1 | 变化 |
|---:|---:|---:|---:|
| 1 | 1.202 | 1.147 | -4.6% |
| 2 | 0.955 | 0.923 | -3.4% |
| 3 | 0.892 | 0.859 | -3.7% |

LCA 的主要价值不在 clean 条件下大幅提高重建质量，而在扰动条件下减少退化。clean 条件下 LCA 的 STOI 略有提升，但 SI-SNR 可能略低于 Base。这符合 LCA 训练目标：模型把一部分 clean 精确度换成了索引扰动下的稳定性。

### 5.3 索引扰动鲁棒性

LCA v2 在 12 个 `(L, perturbed condition)` 单元上的鲁棒性结果如下：

| 指标 | v1 mean robust_imp | v2 mean robust_imp | v2 中 LCA 更鲁棒单元数 |
|---|---:|---:|---:|
| mel-L1 | -0.0003 | +0.0055 | 12/12 |
| PESQ-WB | -0.002 | +0.015 | 10/12 |
| STOI | -0.0001 | +0.0023 | 9/12 |
| SI-SNR | -0.018 dB | +0.051 dB | 7/12 |
| corr | -0.0003 | +0.0010 | 7/12 |
| wave-L1 | 约 0 | 约 0 | 7/12 |

v1 使用弱扰动且没有 consistency loss，平均鲁棒性收益接近 0 或为负。v2 使用更强扰动和 consistency loss 后，6 个指标均为正向，说明该训练设计确实改善了索引级扰动下的稳定性。代表性地，在 `L=3`、dropout-high 条件下，Base 的 mel-L1 退化为 +0.0881，LCA v2 为 +0.0720，LCA 退化幅度少 18.3%；在 `L=3`、dropout-mid 条件下，PESQ-WB 退化幅度从 Base 的 +0.208 降至 LCA 的 +0.161，退化少 22%。

### 5.4 语义蒸馏消融

语义蒸馏是当前系统中最明确的必要设计。A3 消融将 `distill_loss_lambda` 从 30 改为 0，其余训练条件尽量保持一致。由于退化信号已经非常明显，distill0 在约 12-15 epoch 后早停。关键 dev/mel 对比如下：

| step | distill0 | distill30 | 差值 |
|---:|---:|---:|---:|
| 5000 | 4.587 | 4.447 | +0.139 |
| 17500 | 3.923 | 3.218 | +0.706 |
| 47500 | 3.202 | 1.776 | +1.426 |

在样本评估中，distill0 在所有 `L=1/2/3` 条件下均表现更差。例如 `L=3` 时，distill30 的 mel-L1 为 0.892、PESQ-WB 为 1.912、WER 为 0.143；distill0 的 mel-L1 为 0.990、PESQ-WB 为 1.795、WER 为 0.227。值得注意的是，distill0 的 codebook usage 更高，但质量更差。这说明蒸馏并非简单增加 code 使用数量，而是让有限 code 更集中地承载内容相关信息。

### 5.5 NAS 轻量化结果

NAS encoder 的主要结果是发送端效率提升：

| Encoder | Params | MACs (G/s) | RTF mean |
|---|---:|---:|---:|
| hand-designed | 67.7M | 5.87 | 0.00819 |
| NAS_seed42_000896 | 8.87M | 0.618 | 0.00368 |
| reduction | -86.9% | -89.5% | -55.1% |

由于 hand-designed encoder 尚未完成与 NAS encoder 同条件的完整重训，本文不把“NAS 不损质量”作为已完全证明的结论。更稳妥的表述是：NAS 搜索显著降低发送端复杂度，并已被用于本文主系统训练；其与手工 encoder 的质量等价性仍需未来补充全训对照。


## 6. 讨论

### 6.1 SCIT-Speech 的合理定位

本文最重要的定位是：SCIT-Speech 不是常规意义上追求所有码率最优的通用 audio codec，而是一个面向极低负载语音通信的 shared-codebook index transmission 框架。在 500-1500 bps 区间，传统 Opus 的实用宽带操作点通常不覆盖该范围，EnCodec 和 DAC 虽可在低码率运行，但在本文语音样本上可懂度和自然度退化明显。SCIT-Speech 的价值在于用层数 `L` 形成简单、可解释、离散可控的通信接口。

### 6.2 为什么语义蒸馏重要

消融结果显示，去掉 HuBERT 蒸馏后，模型并未发生严重 codebook collapse，反而使用了更多 code。然而更多 code 并未带来更高质量。这提示低负载语音通信中关键不只是“索引数量”或“码本覆盖率”，而是每个索引是否被训练成对内容、音素和可懂度更有用的信息单元。语义蒸馏起到将 encoder 表示约束到内容相关子空间的作用，因此在 `L=1/2/3` 截层传输时尤其重要。

### 6.3 为什么强扰动和 consistency loss 必要

v1 LCA 的负面结果很有价值：弱索引扰动不足以让模型学习鲁棒性。其原因可能是低概率 dropout/substitution 对解码输出影响较小，模型在训练中没有足够动机学习扰动不变性。v2 将扰动强度提高，并显式约束 clean-decoded 与 perturbed-decoded 输出一致，使模型面对更明显的索引错误并学习稳定重建。这也说明，低负载语义通信中的鲁棒性不能只靠“加一点噪声”获得，需要和通信错误模型匹配的训练目标。

### 6.4 与大模型语义通信的关系

近期基于大型语音模型的语义通信方法正在兴起。这类方法可利用强生成先验和语言建模能力进行丢包恢复或语义重构。本文选择另一条路径：不依赖大型语音基础模型作为接收端核心，而是用共享 RVQ 码本和轻量重建模块构建可解释基线。这并不否定大模型路线，反而为未来混合系统提供了接口：SCIT 索引可作为低负载前端，而大模型可作为可选后处理或错误恢复模块。

## 7. 局限性

当前结果仍有明显局限：

1. **测试样本少且同源**：主结果基于 8 条 LibriSpeech train-clean-100 样本，尚不能代表 test-clean、test-other、VCTK、AISHELL 或真实通信语音。
2. **缺少主观听测**：PESQ、STOI、WER 等指标不能完全替代 MOS 或 AB preference。
3. **AMR-WB 缺失**：由于当前 ffmpeg/conda 工具链不含 AMR-WB encoder，最直接的传统语音 codec baseline 尚未纳入。
4. **random-L 单独贡献未隔离**：当前 random-L 与 ChannelSim、consistency loss 在 exp3 v2 中存在 confound，需要专门训练 random-L-only 变体。
5. **NAS 质量对照不完整**：hand-designed encoder 的同条件完整重训尚未完成。
6. **真实系统验证待补**：三用户实时系统、packetized payload、端到端延迟和真实网络丢包仍需实验六进一步完成。

因此，本文当前结论应被理解为一个低负载共享码本索引传输框架的原型验证，而不是最终工业级 codec 或全场景语音通信解决方案。

## 8. 结论

本文提出 SCIT-Speech，一种基于共享 RVQ 码本和分层索引传输的低负载语音语义通信框架。通过将通信信道中的传输对象从波形或连续特征转为离散码本索引，SCIT-Speech 在当前系统实例中形成了 500/1000/1500 bps 三个可控操作点。实验表明，SCIT-Speech-LCA 在 500-1500 bps 区间相对于 DAC、EnCodec 和低码率 Opus 具有良好的可懂度和自然度表现；强索引扰动与 consistency loss 能改善轻量信道错误下的鲁棒性；HuBERT 语义蒸馏对低负载重建质量至关重要；NAS encoder 则显著降低发送端计算成本。

本文的核心意义不在于提出又一个通用神经音频 codec，而在于证明分层离散索引可以作为低负载语音语义通信中的可解释、可控传输单元。后续工作将扩展跨语料库评估、主观听测、AMR-WB 和真实网络基线，并完成 packetized payload 与多用户实时通信验证。

## 参考文献

[1] Zeghidour, N., Luebs, A., Omran, A., Skoglund, J., & Tagliasacchi, M. SoundStream: An End-to-End Neural Audio Codec. arXiv:2107.03312. https://arxiv.org/abs/2107.03312

[2] Défossez, A., Copet, J., Synnaeve, G., & Adi, Y. High Fidelity Neural Audio Compression. arXiv:2210.13438. https://arxiv.org/abs/2210.13438

[3] Kumar, R., Seetharaman, P., Luebs, A., Kumar, I., & Kumar, K. High-Fidelity Audio Compression with Improved RVQGAN. arXiv:2306.06546. https://arxiv.org/abs/2306.06546

[4] Zhang, X., Zhang, D., Li, S., Zhou, Y., & Qiu, X. SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models. arXiv:2308.16692. https://arxiv.org/abs/2308.16692

[5] Ji, S., et al. WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling. arXiv:2408.16532. https://arxiv.org/abs/2408.16532

[6] Weng, Z., Qin, Z., Tao, X., Pan, C., Liu, G., & Li, G. Y. DeepSC-S: Deep Semantic Communication for Speech Signals. arXiv:2012.05369. https://arxiv.org/abs/2012.05369

[7] Qin, Z., Tao, X., Lu, J., Tong, W., & Li, G. Y. Semantic Communications: Principles and Challenges. arXiv:2201.01389. https://arxiv.org/abs/2201.01389

[8] Weng, Z., Qin, Z., Tao, X., Pan, C., Liu, G., & Li, G. Y. Deep Learning Enabled Semantic Communications with Speech Recognition and Synthesis. arXiv:2205.04603. https://arxiv.org/abs/2205.04603

[9] Large Speech Model Enabled Semantic Communication. arXiv:2512.04711. https://arxiv.org/abs/2512.04711

[10] Glaris: Error-Resilient Semantic Communication for Speech Transmission over Packet-Loss Networks. arXiv:2512.08203. https://arxiv.org/abs/2512.08203

## 附：图表建议

**图 1：SCIT-Speech 系统框架图**  
输入语音 -> NAS encoder -> continuous representation -> RVQ indices -> 仅传前 L 层索引 -> shared codebook lookup -> decoder -> 重建语音。图中标注 `L=1/2/3` 对应 500/1000/1500 bps。

**图 2：负载-质量曲线**  
x 轴为 bps，y 轴分别为 WER、STOI、PESQ 或 mel-L1。突出 SCIT 在 500-1500 bps 区间的操作点，并显示 Opus 从 6 kbps 起才进入可用区间。

**图 3：LCA 鲁棒性热力图**  
行：L=1/2/3；列：dropout/substitution 条件；颜色：robustness improvement。突出 v2 在 mel-L1 上 12/12 单元为正。

**图 4：消融实验总览**  
四组柱状图：NAS efficiency、distill0 vs distill30 dev/mel、Base vs LCA clean、weak vs strong ChannelSim robustness。





