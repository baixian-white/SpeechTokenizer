# SCIT-Speech：基于共享 RVQ 码本索引传输的极低码率语音通信方法

> 中文论文初稿（2026-06-09 重写版）。本稿基于 `output/experiments/` 与 `output/doc/实验记录.md`、`output/doc/paper_drafts/scit_speech_experiment_main_tables_20260609.md` 中已记录的实验事实和数值，不外推、不补编未完成实验。

## 摘要

面向 6G、应急与卫星窄带回传、嵌入式语音前端和多用户边缘网关等链路，单路语音可分配到的实际负载常常远低于常规宽带语音 codec 的工作区间。在 500–1500 bps 这一极低码率区间，系统目标不再是逼近无损波形，而是在受控负载下保持可懂度、自然度和任务层可用性。本文提出 **SCIT-Speech**，一种面向极低码率语音通信的共享码本 RVQ 索引传输方法。发送端与接收端预共享 RVQ 码本、模型参数和系统配置；运行时信道仅传输离散索引序列。通过截取并传输前 `L` 层 RVQ 索引，系统可在约 500、1000 和 1500 bps 三个工作点之间切换。本文给出索引负载核算 `R(L)=L f_q ceil(log2 K)`，对当前系统实例（`n_q=3`、`K=1024`、`f_q≈50` Hz）化为 `R(L)=500L` bps。

在固定 8 条 LibriSpeech train-clean-100 样本的 baseline 对比上，SCIT-Speech-LCA 在 500–1500 bps 区间相对于同码率的 DAC、EnCodec 表现出更强的低负载可用性：1.5 kbps 下 STOI 为 0.879、PESQ-WB 为 1.891、Whisper base.en WER 为 0.142，与 Opus 6 kbps 的 WER 0.147 接近。在 LibriSpeech `test-clean`（2620 条）与 `test-other`（2939 条）全量 clean 评估上，LCA 在 `L=1/2/3` 全部工作点上稳定降低 mel-L1 并提高 STOI；幅度较小，PESQ/SI-SNR 在不同条件下分层表现，因此 clean 收益应被解释为“小幅但稳定”。在全量 index dropout/substitution 与 packet/burst loss 评估中，LCA 在 mel-L1 与 STOI 上保持一致改善，是 LCA 的核心鲁棒性证据；PESQ 在部分 L 略低，SI-SNR 仍混合。在 ASR/WER 子集上，`test-other_300` 全部 L 层 WER 下降，`test-clean_300` 在 L=1 与 L=3 下降；在扰动 ASR/WER 子集上，`test-clean_100` 收益主要集中在 L=1，而 `test-other_100` 在所有 L 与两类扰动下 LCA 均降低 WER/CER。消融显示，HuBERT 语义蒸馏（`distill_loss_lambda=30`）对低负载下的可懂度至关重要，强扰动加 consistency loss 是 LCA 鲁棒性的关键。受约束 NAS 搜索得到的发送端 encoder 将参数量、MACs 与 RTF mean 分别降低约 86.9%、89.5% 与 55.1%，仅作为发送端效率证据；与 hand-designed encoder 的同条件质量对照尚未完成。当前结果尚未覆盖跨语料、跨语言、AMR-WB、主观听测与真实网络多用户实时验证。

**关键词**：语义通信；语音通信；极低码率；共享码本；索引传输；残差向量量化；信道扰动训练；packet loss；burst loss

## 1 引言

### 1.1 背景与动机

未来通信系统正逐步从单纯追求比特级可靠传输，走向面向语义和任务的高效信息传输。对于语音通信而言，这一转向具有直接的工程意义：在应急通信、自组织中继、卫星窄带回传、嵌入式语音前端和多用户边缘网关等场景中，单路语音可分配到的实际 payload 可能远低于常规宽带语音 codec 的工作区间。此时系统目标不再是逼近无损波形，而是在极低负载下保留足以支持理解、交互和决策的语音信息。

传统语音编码与近年来的神经音频压缩方法已经在低码率语音和音频重建上取得显著进展。SoundStream、EnCodec、DAC、SpeechTokenizer 和 WavTokenizer 等工作表明，端到端神经 codec、RVQ 码本和离散 token 表示能够在压缩、重建和生成任务中取得良好效果 [1-5]。然而，这类方法多以音频重建质量或通用音频 tokenization 为主要目标，500–1500 bps 区间通常不是其系统化评估的核心操作带。另一方面，DeepSC-S、DeepSC-ST 与近期结合大型语音模型的语义通信方法开始从语义层面重新定义语音传输 [6-10]，但其中不少方法依赖高维连续 JSCC 映射、大型语音基础模型或难以显式核算的传输接口；运行时实际占用信道的对象与码率，仍存在进一步明确的空间。

### 1.2 研究问题

本文关注一个更窄但更可操作的问题：

> 如果发送端和接收端预先共享分层离散码本与重建模块，语音通信能否只传输紧凑的码本索引序列，并在 500–1500 bps 的极低负载下保持可用的可懂度、自然度和语义信息，并在索引级、包级和突发级扰动下保持稳定？

为回答这一问题，本文构建 **SCIT-Speech**（Shared-Codebook Index Transmission Speech）。SCIT-Speech 不把自身定位为通用神经音频 codec，也不把分层 RVQ 当作新的机制贡献；相反，本文将离散语音索引重新定位为通信信道中的传输符号：码本、解码器、模型参数和系统配置作为离线预共享系统知识，运行时信道只承载离散索引序列。在这一接口下，RVQ 层数 `L` 成为一个可解释、可控的通信旋钮。对于当前系统实例（`K=1024`、潜在帧率约 50 Hz），每增加一层 RVQ 索引，理想负载增加 500 bps。

### 1.3 本文贡献

本文贡献如下：

1. **共享码本索引传输方法**：将 RVQ 神经语音 codec 重述为预共享码本条件下的 index-only transmission 方法，明确区分预共享系统知识和真实信道负载。
2. **可控负载建模**：给出分层索引负载公式 `R(L)=L f_q ceil(log2 K)`，并在当前实例中形成 500/1000/1500 bps 三个可解释操作点。
3. **低负载信道感知适配（LCA）训练配方**：通过 random-L 多操作点训练、强 index dropout/substitution 与 clean-vs-perturbed consistency 损失，得到一个对索引级、包级和突发级扰动更稳定的系统实例。
4. **大规模评估体系**：在 LibriSpeech `test-clean`（2620 条）与 `test-other`（2939 条）上完成全量 clean、index dropout/substitution 与 packet/burst loss 评估，并在子集上完成 clean 与扰动 ASR/WER 验证；同时给出语义蒸馏、weak vs strong LCA、NAS encoder 效率等关键消融。

### 1.4 范围与边界

本文的结论边界需要提前明确。第一，本文是一项面向极低码率语音通信操作区间的实证系统研究，不是新的通信范式或新的损失机制；同时不主张在所有码率上击败传统 codec 或神经 codec，对于 6 kbps 以上的常规通信条件，Opus、DAC、EnCodec 等仍是成熟选择。第二，LCA 的主要价值在“在共享 RVQ 索引接口下提供可控、稳定的低负载操作”，而不是“在所有指标上全面提升”。第三，当前 NAS encoder 仅作为发送端效率优化，与 hand-designed encoder 的同条件质量对照尚未完成；不应把 NAS 的效率收益解读为质量优越性证明。第四，本文未覆盖跨语料、跨语言、AMR-WB、主观听测与真实网络多用户实时验证，将其放入局限性与未来工作。

## 2 相关工作

### 2.1 传统语音编码与神经音频 codec

传统语音 codec 与音频 codec 已长期服务于低码率通信。近年来神经音频压缩进一步利用端到端神经网络、向量量化与生成式重建提升主观音质：SoundStream 使用端到端编码器-解码器和残差向量量化实现可变码率神经音频压缩 [1]；EnCodec 提升了实时神经音频压缩质量 [2]；DAC 等改进 RVQGAN 类方法强调高保真音频压缩 [3]；WavTokenizer 进一步探索更高压缩率的离散声学表示 [5]。这些工作展示了离散码本和神经重建在音频压缩中的潜力，但其评估往往集中在 6 kbps 及以上，500–1500 bps 通常不是其主要操作带。

与上述方法相比，本文的核心目标不是在所有码率上获得最高音频重建质量，而是定义一种通信负载可解释、运行时只传输离散索引的接口。在该接口下，码本与解码器属于离线共享知识，信道实际承载的对象就是 RVQ 索引，码率核算因此可直接由层数 `L`、码本大小 `K` 与潜在帧率 `f_q` 表达。

### 2.2 离散语音 token 与 RVQ 类工作

SpeechTokenizer 等工作表明，基于 RVQ 的语音 tokenizer 可以在不同量化层中形成语义和声学信息的分层结构 [4]：第一层 token 往往承担更多内容相关信息，后续层补充音色、韵律和细节。这类研究为语音语言模型、语音生成和统一 speech-token 表示提供了基础。本文不重新发明分层 RVQ，而是利用其离散化能力，把分层 token 从“模型内部表示”转换为“通信信道符号”，并研究索引层数、负载、客观质量与扰动鲁棒性之间的关系。

### 2.3 语音语义通信与丢包鲁棒通信

DeepSC-S、DeepSC-ST 等工作将语义通信思想引入语音传输，强调在噪声信道中保留语义信息而非逐比特恢复 [6,8]。Qin 等综述系统讨论了语义通信原理与挑战 [7]。近期 LargeSC、Glaris 等工作进一步引入大型语音模型、离散 token、丢包恢复和不等错误保护等机制，把语音语义通信向更复杂系统推进 [9,10]。

这些工作与本文目标相邻，但本文强调一个更轻量、更可解释的路径：不依赖大型语音基础模型作为接收端核心，而是在共享 RVQ 码本条件下直接研究索引层数、通信负载、客观质量与索引/包/突发扰动鲁棒性之间的关系。这使得 SCIT-Speech 既可独立作为低负载基线，也可作为后续接入纠错、packetization、多用户路由或大模型后处理的前端接口。

## 3 方法

### 3.1 系统总览：共享码本 + index-only transmission

SCIT-Speech 由发送端表示编码器、共享 RVQ 码本、索引级信道与接收端重建模块构成。给定输入语音 `x(t)`，发送端先经编码器 `E_θ` 得到连续语音表示

```text
Z = E_θ(x).
```

随后 `Z` 被分层 RVQ 码本量化为 `M` 层离散索引

```text
I = { i_{m,t} },  m = 1,...,M,  t = 1,...,T_q,
```

其中 `i_{m,t}` 是第 `m` 层码本在第 `t` 个潜在时间步上的索引。发送端与接收端预共享码本 `C* = {C_1*, ..., C_M*}`、解码器和系统配置；运行时信道仅承载前 `L` 层索引

```text
I_L = I_{1:L,:},  L ≤ M.
```

接收端在共享码本中查表，得到量化表示并由 decoder `D` 重建语音

```text
x̂ = D(C*(I_L)).
```

在本文的当前系统实例中，`M = n_q = 3`，`K = 1024`，采样率 16 kHz，encoder 总下采样率为 320，因此潜在帧率 `f_q = 16000 / 320 = 50` steps/s。该接口的核心特征是：码本、解码器与系统配置不出现在运行时信道中，因此码率核算只取决于离散索引的数量与码本大小。

### 3.2 通信负载建模

对于每层 RVQ 码本大小 `K`，每个索引需要 `b = ceil(log2 K)` 比特表示。若每秒潜在帧数为 `f_q`，传输前 `L` 层索引，则理想 bit-packed 索引负载为

```text
R(L) = L · f_q · ceil(log2 K).
```

本实例中 `K = 1024`，`ceil(log2 K) = 10`，`f_q = 50`，因此

```text
R(L) = 500 · L  bps,
```

对应：

| 传输层数 L | 理想索引负载 |
|---:|---:|
| 1 | 500 bps |
| 2 | 1000 bps |
| 3 | 1500 bps |

需要强调的是，本文严格区分理想 bit-packed 索引负载与工程上的 packetized payload。`.npy`、JSON、CSV、文件头与 TCP/UDP 报头等存储/传输开销，不应被混入理论索引负载。涉及实际网络的 packetized 工程开销，留作后续工作（见局限性）。

### 3.3 SCIT-Speech-Base 训练

SCIT-Speech-Base 基于 NAS 选出的轻量 encoder 架构（见 §3.5）从头训练，包括 encoder、RVQ codebooks 与 decoder。训练数据为 LibriSpeech train-clean-100 及对应 HuBERT 表征。训练目标包括波形/频谱重建相关损失、对抗训练损失与 HuBERT 语义蒸馏损失。语义蒸馏将低层索引组织到对内容更敏感的子空间，从而支持低负载工作点的可懂度。当前主模型采用 `distill_loss_lambda = 30`。

正式 Base checkpoint 路径为

```text
output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/
SCIT-Speech-Base_best.pt
```

该 run 在 step 107500 取得 `dev/mel = 1.124`，并在后续阶段进入震荡平台期。该 checkpoint 作为后续 LCA 微调与 baseline 对比的底模。

### 3.4 SCIT-Speech-LCA：random-L、强扰动与 consistency loss

Base 训练完成后，本文进一步训练 SCIT-Speech-LCA（Low-load Channel-aware Adaptation）。LCA 的训练配方包含三个部分：

1. **Random-L 多操作点训练**：每个 batch 随机采样 `L ∈ {1, 2, 3}`，避免模型只在 full-depth 条件下最优；
2. **强 index dropout / substitution**：在传输前的索引序列上以较高概率执行 dropout 和 substitution（参考 [11] 的 `ChannelSim` 模块）；
3. **clean-vs-perturbed consistency loss**：以 mel-L1 度量 clean-decoded 与 perturbed-decoded 输出之间的差异，作为显式一致性约束。

第一版 v1 LCA 采用较弱的扰动概率（`p_drop ≤ 0.03`、`p_sub ≤ 0.005`、`λ_consistency = 0`），鲁棒性收益接近 0 或微负，记作消融对照。当前主模型为 v2 配置，将扰动概率提升约 3–6 倍并加入 consistency loss：

| 字段 | v1 | v2 | 倍数 |
|---|---|---|---|
| `p_drop` | clean / 0.01 / 0.03 | clean / **0.05 / 0.10** | ~3× |
| `p_sub` | 0 / 0.001 / 0.005 | 0 / **0.01 / 0.03** | ~6× |
| `λ_comm` | 1.0 | **1.5** | +50% |
| `λ_consistency` | 0 | **0.5** | mel-L1(clean, perturbed) |

LCA v2 在 Base 之上做 10 epoch 端到端微调，在 step 30000 取得 `dev/comm_mel` 在 substitution-high 三档全局最优，作为正式 checkpoint。该 checkpoint 路径为

```text
output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/
checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt
```

### 3.5 轻量发送端 NAS encoder

发送端计算成本对低负载语音通信同样重要。即使信道仅承载少量 RVQ 索引，若发送端 encoder 过大，系统仍难以部署在移动端、边缘设备或实时通信场景中。因此，本文在保持后续通信接口不变的前提下，使用 teacher-guided staged NAS 搜索轻量 encoder 架构。NAS 仅改变发送端特征提取网络的结构复杂度，不改变索引生成、传输与重建之间的接口：所有候选 encoder 输出相同帧率（50 Hz）、相同维度（1024）的 latent，并接入相同的 RVQ 与 decoder。

搜索空间围绕 SEANet encoder 的宽度、时序建模和局部算子展开（含初始通道数、LSTM 层数、残差压缩比例、激活函数与各下采样阶段的卷积算子）。候选算子包含标准卷积、深度可分离卷积、空洞卷积和 skip，同时搜索是否在对应阶段使用 SE 模块。所有候选都被约束为 1024 维 latent 与 320 倍总下采样率。每个候选先经过计算量门控；通过门控的候选在 NAS 子集上进行短程代理训练，目标为波形与 mel 频谱重建损失的组合，由 Optuna TPE + 中间 epoch 早停剪枝引导。

最终选定候选 `nas_seed42_000896` 用于 SCIT-Speech-Base 与 LCA 的发送端 encoder。相对于原 SEANet hand-designed encoder：

| 项目 | hand-designed | NAS_seed42_000896 | 变化 |
|---|---:|---:|---:|
| Params | 67.7 M | 8.87 M | -86.9% |
| MACs | 5.87 G/s | 0.618 G/s | -89.5% |
| RTF mean | 0.00819 | 0.00368 | -55.1% |

由于 hand-designed encoder 尚未完成与 NAS encoder 同条件的完整重训，本文不把 NAS 表述为“质量优于 hand-designed”，而仅作为**发送端效率 profiling 证据**与当前系统实例的轻量化设计。质量等价性需要等预算重训和后续完整客观/主观对照才能确立。

## 4 实验设置

### 4.1 数据、模型与评估层级

本文实验体系按“训练/主 baseline 固定样本 → LibriSpeech 全量 clean → 全量索引扰动 → 全量 packet/burst 丢包 → ASR/WER 子集”的顺序组织：

| 层级 | 数据 | 用途 |
|---|---|---|
| 训练 | LibriSpeech train-clean-100 | Base / LCA 训练 |
| 固定样本 baseline | `fixed_sample_list.txt` 8 条 train-clean-100 utterances | 与传统/神经 codec 同码率对比（Exp4） |
| 全量 clean 泛化 | LibriSpeech `test-clean` 2620 条、`test-other` 2939 条 | Base vs LCA clean 不退化与小幅改善（Exp7） |
| 全量索引扰动 | 同上 | dropout/substitution mid/high 鲁棒性（Exp8） |
| 全量 packet/burst | 同上 | packet-loss-1p/3p/5p 与 burst-2f/5f/10f（Exp9） |
| clean ASR/WER | `test-clean_300`、`test-other_300` | Whisper base.en 上的可懂度子集（Exp10） |
| 扰动 ASR/WER | `test-clean_100`、`test-other_100` × dropout-high / substitution-high | 困难+扰动叠加下的可懂度（Exp11） |

模型接口固定如下：

| 项目 | 值 |
|---|---:|
| sample rate | 16 kHz |
| encoder 总下采样率 | 320 |
| 潜在帧率 `f_q` | 50 steps/s |
| RVQ 层数 `n_q` | 3 |
| 码本大小 `K` | 1024 |
| 传输层数 `L` | 1 / 2 / 3 |
| 理想索引负载 | 500 / 1000 / 1500 bps |

### 4.2 baseline

固定 8 条样本上的同码率对比包括：

| 方法 | 配置 |
|---|---|
| SCIT-Speech-Base | L = 1 / 2 / 3 |
| SCIT-Speech-LCA v2 | L = 1 / 2 / 3 |
| PCM 16-bit @ 16 kHz | 256 kbps（无损上界） |
| Opus（libopus, ffmpeg 6.1.2） | 6 / 8 / 12 / 16 / 24 kbps |
| EnCodec 24 kHz | bw ∈ {1.5, 3.0, 6.0, 12.0} kbps |
| DAC 16 kHz | n_q ∈ {1, 2, 3, 4, 6, 9, 12} |

AMR-WB 因当前 conda-forge ffmpeg 6.1.2 build 缺 `--enable-libvo-amrwbenc`、只有 decoder 而没有 encoder，被迫从主表中跳过；Codec2 / ViSQOL / 主观 MOS 同样未纳入主表。该缺口在局限性中说明。

### 4.3 评价指标

客观指标包括 wave-L1、mel-L1、SI-SNR、相关系数、STOI 和 PESQ-WB；任务层指标使用 Whisper `base.en` 的 WER 与 CER。需要注意，PCM lossless 上 Whisper 在 8 条样本上的 WER 约为 0.075，因此 baseline 对比中的 WER 应主要被理解为相对该测量下界的退化，而不是绝对人类听懂率。

### 4.4 索引扰动与 packet/burst loss 设置

索引级扰动包含：

- **dropout-mid / dropout-high**：以 mid/high 概率独立丢弃前 `L` 层索引中的若干位置；
- **substitution-mid / substitution-high**：以 mid/high 概率将索引替换为同码本中随机有效索引（约 3% 时为 substitution-high）。

包级与突发扰动包含：

- **packet-loss-1p / 3p / 5p**：以 1% / 3% / 5% 概率独立丢弃定长 packet；
- **burst-2f / 5f / 10f**：以等长突发丢失 2 / 5 / 10 个连续帧。

正文主表只保留较强档（`dropout-high`、`substitution-high`、`packet-loss-5p`、`burst-10f`）；mid 档与其他档放入附录。

### 4.5 ASR/WER 设置

使用 `evaluate_asr_wer_onthefly.py` 与 `evaluate_perturbed_asr_wer_onthefly.py` 在 Whisper `base.en` 下解码 Base/LCA 重建语音。clean ASR 子集为 `test-clean_300` 与 `test-other_300`；扰动 ASR 子集为 `test-clean_100` 与 `test-other_100`，搭配 dropout-high 与 substitution-high。

## 5 结果

### 5.1 低码率 baseline 对比（Exp4）

固定 8 条样本上的同码率对比如表 1。

**表 1：500 / 1000 / 1500 bps 同码率对比（固定 8 条 train-clean-100 样本）**

| 操作点 | 最佳 baseline | SCIT-Speech-LCA |
|---|---|---|
| 500 bps | DAC `n_q=1`：STOI 0.606、PESQ 1.055、WER 0.922 | L=1：STOI 0.797、PESQ 1.347、WER 0.461 |
| 1000 bps | DAC `n_q=2`：STOI 0.730、PESQ 1.148、WER 0.305 | L=2：STOI 0.857、PESQ 1.709、WER 0.188 |
| 1500 bps | DAC `n_q=3`：STOI 0.799、PESQ 1.271、WER 0.152；EnCodec 1.5 kbps：WER 0.302 | L=3：STOI 0.879、PESQ 1.891、WER 0.142 |

在 500–1500 bps 区间，SCIT-Speech-LCA 在 STOI、PESQ 与 WER 上一致优于同码率的 DAC、EnCodec。一个可被讨论的对照是：SCIT-LCA L=3（1.5 kbps）的 WER 0.142 与 Opus 6 kbps 的 WER 0.147 接近。这并不意味着 SCIT 在所有码率上击败 Opus；事实上，Opus 在 12 kbps 以上仍非常强。相对的可写表述是：在 500–1500 bps 这一区间，SCIT-Speech 提供了一个共享码本索引传输的可用操作带。

### 5.2 LibriSpeech 全量 clean 泛化（Exp7）

在不加扰动的 LibriSpeech 全量测试集上，LCA 没有以牺牲 clean 重建为代价换取鲁棒性；表 2 给出 mel-L1 与 STOI 的全量结果。

**表 2：LibriSpeech 全量 clean 评估（Exp7，主表 A）**

| 数据集 | n | L | Base mel-L1 ↓ | LCA mel-L1 ↓ | Δmel-L1 | Base STOI ↑ | LCA STOI ↑ | ΔSTOI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test-clean | 2620 | 1 | 1.226 | 1.205 | -0.021 | 0.801 | 0.813 | +0.013 |
| test-clean | 2620 | 2 | 0.976 | 0.946 | -0.030 | 0.864 | 0.871 | +0.007 |
| test-clean | 2620 | 3 | 0.907 | 0.871 | -0.036 | 0.885 | 0.892 | +0.007 |
| test-other | 2939 | 1 | 1.397 | 1.384 | -0.013 | 0.769 | 0.782 | +0.013 |
| test-other | 2939 | 2 | 1.146 | 1.116 | -0.030 | 0.832 | 0.839 | +0.008 |
| test-other | 2939 | 3 | 1.072 | 1.039 | -0.033 | 0.854 | 0.862 | +0.007 |

LCA 在 `test-clean` 与 `test-other` 的全部 `L=1/2/3` 工作点上均降低 mel-L1 并提高 STOI；改善幅度较小，PESQ 与 SI-SNR 在不同条件下分层差异较大。因此 clean 条件结果应被解释为“小幅但稳定”，而非“大幅质量提升”。

### 5.3 全量 index dropout / substitution 鲁棒性（Exp8）

正文主表仅保留 `dropout-high` 与 `substitution-high`；完整 mid/high 档放入附录。表 3 给出 Δ 值（LCA 相对 Base，越负越好或越正越好按列定义）。

**表 3：全量 index dropout / substitution 鲁棒性（Exp8，主表 B）**

| 数据集 | n | L | 条件 | Δmel-L1 ↓ | ΔSTOI ↑ | ΔPESQ-WB ↑ | ΔSI-SNR ↑ |
|---|---:|---:|---|---:|---:|---:|---:|
| test-clean | 2620 | 1 | dropout-high | -0.0302 | +0.0168 | +0.0246 | -0.307 |
| test-clean | 2620 | 2 | dropout-high | -0.0405 | +0.0122 | +0.0132 | -0.167 |
| test-clean | 2620 | 3 | dropout-high | -0.0475 | +0.0123 | +0.0185 | +0.133 |
| test-clean | 2620 | 1 | substitution-high | -0.0319 | +0.0134 | +0.0245 | -0.393 |
| test-clean | 2620 | 2 | substitution-high | -0.0393 | +0.0084 | +0.0012 | -0.323 |
| test-clean | 2620 | 3 | substitution-high | -0.0451 | +0.0082 | +0.0122 | -0.030 |
| test-other | 2939 | 1 | dropout-high | -0.0184 | +0.0163 | +0.0223 | -0.310 |
| test-other | 2939 | 2 | dropout-high | -0.0375 | +0.0125 | +0.0062 | -0.269 |
| test-other | 2939 | 3 | dropout-high | -0.0412 | +0.0124 | +0.0075 | +0.070 |
| test-other | 2939 | 1 | substitution-high | -0.0283 | +0.0133 | +0.0259 | -0.384 |
| test-other | 2939 | 2 | substitution-high | -0.0432 | +0.0090 | +0.0085 | -0.407 |
| test-other | 2939 | 3 | substitution-high | -0.0429 | +0.0085 | +0.0087 | -0.072 |

在两个测试集、三档码率与两类 high 扰动上，LCA 均降低 mel-L1 并提高 STOI，PESQ 大多改善，SI-SNR 仍然混合。这说明 LCA 更可靠地改善了频谱稳定性与可懂度相关指标，而不是简单提升所有波形级指标。

### 5.4 全量 packet / burst loss 鲁棒性（Exp9）

packet/burst loss 评估更接近通信载荷在非独立错误下的表现。正文主表只放 `packet-loss-5p` 与 `burst-10f` 两个较强场景。

**表 4：全量 packet / burst loss 鲁棒性（Exp9，主表 C）**

| 数据集 | n | L | 条件 | Δmel-L1 ↓ | ΔSTOI ↑ | ΔPESQ-WB ↑ | ΔSI-SNR ↑ |
|---|---:|---:|---|---:|---:|---:|---:|
| test-clean | 2620 | 1 | packet-loss-5p | -0.0220 | +0.0136 | +0.0230 | -0.353 |
| test-clean | 2620 | 2 | packet-loss-5p | -0.0307 | +0.0081 | -0.0092 | -0.233 |
| test-clean | 2620 | 3 | packet-loss-5p | -0.0377 | +0.0078 | -0.0030 | +0.039 |
| test-clean | 2620 | 1 | burst-10f | -0.0213 | +0.0131 | +0.0231 | -0.354 |
| test-clean | 2620 | 2 | burst-10f | -0.0302 | +0.0077 | -0.0124 | -0.280 |
| test-clean | 2620 | 3 | burst-10f | -0.0369 | +0.0074 | -0.0090 | +0.030 |
| test-other | 2939 | 1 | packet-loss-5p | -0.0134 | +0.0137 | +0.0236 | -0.287 |
| test-other | 2939 | 2 | packet-loss-5p | -0.0303 | +0.0085 | -0.0053 | -0.344 |
| test-other | 2939 | 3 | packet-loss-5p | -0.0339 | +0.0080 | -0.0084 | -0.025 |
| test-other | 2939 | 1 | burst-10f | -0.0130 | +0.0133 | +0.0231 | -0.295 |
| test-other | 2939 | 2 | burst-10f | -0.0299 | +0.0083 | -0.0058 | -0.400 |
| test-other | 2939 | 3 | burst-10f | -0.0334 | +0.0076 | -0.0102 | -0.057 |

LCA 在所有代表性 packet/burst 条件下保持 mel-L1 与 STOI 的一致改善，说明其收益不只局限于独立索引替换；但 PESQ 在 L2/L3 下有时略低，SI-SNR 也不稳定。该实验应作为“通信化扰动下的频谱与可懂度鲁棒性证据”，而不是“所有音质指标全面提升”。

### 5.5 clean 与扰动 ASR/WER 子集（Exp10、Exp11）

ASR 子集结果提供任务层可懂度证据。clean 子集结果如表 5。

**表 5：clean ASR/WER 子集（Exp10，主表 D，Whisper base.en）**

| 数据集 | n | L | Base WER ↓ | LCA WER ↓ | ΔWER | Base CER ↓ | LCA CER ↓ | ΔCER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test-clean | 300 | 1 | 0.3350 | 0.3089 | -0.0261 | 0.1908 | 0.1749 | -0.0160 |
| test-clean | 300 | 2 | 0.1304 | 0.1320 | +0.0016 | 0.0658 | 0.0695 | +0.0037 |
| test-clean | 300 | 3 | 0.1167 | 0.0933 | -0.0234 | 0.0618 | 0.0441 | -0.0178 |
| test-other | 300 | 1 | 0.6986 | 0.6941 | -0.0045 | 0.4671 | 0.4454 | -0.0217 |
| test-other | 300 | 2 | 0.4791 | 0.4625 | -0.0166 | 0.2867 | 0.2772 | -0.0095 |
| test-other | 300 | 3 | 0.3929 | 0.3665 | -0.0263 | 0.2201 | 0.2123 | -0.0077 |

在 `test-other_300` 上，LCA 在全部三档码率降低 WER；在 `test-clean_300` 上，L1 与 L3 改善，L2 基本持平略差。这与全量客观指标一致：LCA 更稳定地改善低负载与困难条件下的可懂度相关表现，但不能宣称每个 clean ASR 工作点都改善。

扰动 ASR 子集结果如表 6，呈现两类不同模式。

**表 6：扰动 ASR/WER 子集（Exp11，主表 E，Whisper base.en）**

| 数据集 | n | L | 条件 | Base WER ↓ | LCA WER ↓ | ΔWER | Base CER ↓ | LCA CER ↓ | ΔCER |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| test-clean | 100 | 1 | dropout-high | 0.3321 | 0.3226 | -0.0095 | 0.1801 | 0.1721 | -0.0080 |
| test-clean | 100 | 1 | substitution-high | 0.3685 | 0.3272 | -0.0413 | 0.1981 | 0.1757 | -0.0224 |
| test-clean | 100 | 2 | dropout-high | 0.1514 | 0.1610 | +0.0096 | 0.0704 | 0.0783 | +0.0079 |
| test-clean | 100 | 2 | substitution-high | 0.1464 | 0.1528 | +0.0064 | 0.0617 | 0.0792 | +0.0175 |
| test-clean | 100 | 3 | dropout-high | 0.0971 | 0.1088 | +0.0117 | 0.0398 | 0.0527 | +0.0129 |
| test-clean | 100 | 3 | substitution-high | 0.1199 | 0.1278 | +0.0079 | 0.0494 | 0.0529 | +0.0036 |
| test-other | 100 | 1 | dropout-high | 0.6690 | 0.5921 | -0.0769 | 0.4466 | 0.3799 | -0.0667 |
| test-other | 100 | 1 | substitution-high | 0.6556 | 0.6409 | -0.0147 | 0.4179 | 0.4165 | -0.0014 |
| test-other | 100 | 2 | dropout-high | 0.5088 | 0.4449 | -0.0639 | 0.3212 | 0.2596 | -0.0616 |
| test-other | 100 | 2 | substitution-high | 0.4810 | 0.4348 | -0.0462 | 0.2976 | 0.2613 | -0.0364 |
| test-other | 100 | 3 | dropout-high | 0.4852 | 0.4066 | -0.0786 | 0.3113 | 0.2321 | -0.0791 |
| test-other | 100 | 3 | substitution-high | 0.3691 | 0.3205 | -0.0486 | 0.2201 | 0.1868 | -0.0333 |

`test-clean_100` 的 WER 收益主要集中在最低码率 `L=1`，特别是在 substitution-high 下 WER 下降 4.13 个百分点；`L=2/3` 下 LCA 略差。`test-other_100` 上则呈现完全不同的模式：LCA 在所有 L 与两类扰动下均降低 WER 与 CER，最大降幅出现在 `L=3, dropout-high`，WER 降低 7.86 个百分点。整体可写为：LCA 的可懂度鲁棒性在困难语音与信道扰动叠加时更明显，但需保留 clean 子集高层码率收益不稳定的边界。

### 5.6 消融

#### 5.6.1 语义蒸馏（distill0 vs distill30）

A3 消融将 `distill_loss_lambda` 从 30 改为 0，其余训练条件尽量保持一致。退化信号已经非常明显，distill0 在约 12–15 epoch 处早停。同 step 数对照下：

| step | distill0 dev/mel | distill30 dev/mel | 差值 |
|---:|---:|---:|---:|
| 5000 | 4.59 | 4.45 | +0.14 |
| 17500 | 3.92 | 3.22 | +0.71 |
| 47500 | 3.20 | 1.78 | **+1.43** |

并且 distill0 在 `L=1/2/3` 全部条件下 WER 退化约 7–9 个百分点（`L=1` ΔWER +0.094、`L=2` +0.074、`L=3` +0.084）。一个反直觉的观察是：distill0 反而**用更多 code**（`L=1`：228 vs distill30 的 179）。这说明蒸馏不是简单提升 codebook usage，而是让有限索引更好地承载内容相关信息。

#### 5.6.2 weak vs strong LCA（v1 vs v2）

LCA v1（弱扰动、无 consistency）vs LCA v2（强扰动 + consistency）的鲁棒性对照如下：

| 指标 | v1 mean robust_imp | v2 mean robust_imp | v1 中 LCA 更鲁棒 cell | v2 中 LCA 更鲁棒 cell |
|---|---:|---:|---:|---:|
| mel-L1 | -0.0003 | +0.0055 | 3 / 12 | 12 / 12 |
| PESQ-WB | -0.002 | +0.015 | 7 / 12 | 10 / 12 |
| STOI | -0.0001 | +0.0023 | 6 / 12 | 9 / 12 |
| SI-SNR | -0.018 dB | +0.051 dB | 4 / 12 | 7 / 12 |
| corr | -0.0003 | +0.0010 | 5 / 12 | 7 / 12 |
| wave-L1 | ~0 | ~0 | 6 / 12 | 7 / 12 |

v1 几乎不产生鲁棒性收益，v2 在 6 个客观指标上的 robustness improvement 都为正向，其中 mel-L1 在 12/12 cell 上 LCA 退化更小。这说明低负载语义通信中的鲁棒性不能只靠“加一点噪声”获得，需要与通信错误模型匹配的训练目标和显式一致性约束。

#### 5.6.3 NAS encoder 效率（A1）

如 §3.5 所述，NAS_seed42_000896 相对 hand-designed encoder 在参数量、MACs 与 RTF 上分别降低 86.9%、89.5% 与 55.1%。该结果仅作为发送端效率证据，不构成 NAS encoder 在质量上优于 hand-designed encoder 的结论；后者尚未在等预算条件下完整重训。

#### 5.6.4 LCA 组件因子（简写）

`exp5_lca_component_factorial_*` 进一步对 LCA 内部组件做因子分析。当前结果与 §5.6.2 的整体方向一致：弱 ChannelSim 与零 consistency 几乎不产生鲁棒性收益，强 ChannelSim 与 consistency 为主要正向贡献来源。random-L 的单独贡献因与 ChannelSim、consistency 在 v2 中存在 confound，目前不能被干净归因，留作 future work。

## 6 讨论

### 6.1 为什么共享 RVQ 索引适合低码率语音通信

共享码本将“码本与解码器”从信道内容剥离为离线预共享系统知识，使运行时信道实际承载的对象就是离散索引。一旦做出这一抽象，低码率通信中的所有可控变量都集中到了“传几层索引、每层多少 bit、每秒多少帧”这三件事上：本文中即 `L`、`ceil(log2 K)` 与 `f_q`。这种结构使码率核算可由 `R(L)=L f_q ceil(log2 K)` 直接表达，并允许把传统通信中熟悉的 dropout、substitution、packet loss、burst loss 等错误模型，直接套用到离散索引序列上做训练与评估。

### 6.2 为什么 LCA 的主要价值在鲁棒性而不是 clean 大幅提升

clean 条件下，LCA 在全量 `test-clean` 与 `test-other` 上对 mel-L1 与 STOI 的改善都不大（Δmel-L1 在 -0.013 至 -0.036 之间，ΔSTOI 在 +0.007 至 +0.013 之间）。在扰动条件下，LCA 在 dropout-high、substitution-high、packet-loss-5p、burst-10f 上的 Δmel-L1/ΔSTOI 改善幅度通常更大；在困难语音 `test-other_100` 与扰动叠加下 ASR/WER 也明显改善。这种“clean 不退化 + 扰动稳定改善”的形态符合 LCA 的训练目标：模型把一部分 clean 精确度换成了离散索引扰动下的稳定性。

### 6.3 为什么 semantic distillation 重要

从 §5.6.1 看，distill0 的 dev/mel、PESQ 与 WER 显著差于 distill30，但反而使用了更多 code。这与“codebook usage 越高越好”的直觉相反，并提示低负载语音通信中关键的不是“索引数量”或“码本覆盖率”，而是每个被使用的索引是否被训练成对内容、音素和可懂度更有用的信息单元。语义蒸馏在 `L=1/2/3` 截层传输下尤其重要，因为它压缩进了发送端必须传出的最少索引中。

### 6.4 为什么不能过度声称所有指标全面提升

本文的所有结果都体现出一个共性：mel-L1 与 STOI 是 LCA 改善最稳定的两个指标；PESQ 在多数条件下改善但部分条件略低；SI-SNR 在 packet/burst 等条件下普遍混合，甚至略劣；ASR/WER 在 `test-clean` 高层码率下不稳定。因此本文的结论范畴限定在“频谱稳定性 + 可懂度相关鲁棒性”，而非“全部音质指标全面提升”。这一边界是 LCA 训练在 clean 精确度与扰动稳定性之间的折中，不应被掩盖。

## 7 局限性

1. **跨语料、跨语言尚未覆盖**：当前评估全部基于 LibriSpeech（含全量 `test-clean`、`test-other` 与子集），尚未覆盖 VCTK、AISHELL、跨语言或真实通信语音。
2. **AMR-WB 缺失**：当前 conda-forge ffmpeg 6.1.2 build 不含 AMR-WB encoder，最直接的传统语音 codec baseline 尚未纳入；后续可通过 build ffmpeg from source 加 `--enable-libvo-amrwbenc` 或使用 gyan.dev 完整版 ffmpeg 补充。
3. **无主观 MOS / AB 听测**：PESQ、STOI、WER 等指标不能完全替代 MOS 或 AB preference。
4. **NAS 与 hand-designed encoder 的同条件质量对照未完成**：当前 NAS 结果只能用于发送端效率 profiling，等预算重训和质量等价性验证留作未来工作。
5. **random-L 单独贡献未隔离**：random-L 与 ChannelSim、consistency 在 LCA v2 中存在 confound，需要专门训练 random-L-only 变体。
6. **真实网络与多用户实时验证待补**：当前实验体系处理的是离散索引序列层面的扰动模型；packetized payload、端到端延迟、jitter、抖动缓冲、三用户实时系统等仍需后续实验完成。
7. **codebook 利用率仍不健康**：当前 distill30 主 run 的 RVQ usage rate 为 17–27%、L1 dead code ratio 约 82%。这未影响本文 Base/LCA 的相对结论，但提示后续可以通过 dead code reinit、k-means 初始化、usage entropy 正则等专项修复。

## 8 结论

本文提出 SCIT-Speech，一种面向极低码率语音通信的共享码本 RVQ 索引传输方法。发送端与接收端预共享 RVQ 码本、模型参数和系统配置，运行时信道仅传输前 `L` 层离散索引；当前实例（`n_q=3`、`K=1024`、`f_q≈50` Hz）形成 500 / 1000 / 1500 bps 三个可控操作点。在固定 8 条样本的 baseline 对比中，SCIT-Speech-LCA 在 500–1500 bps 区间相对同码率 DAC、EnCodec 提供更好的可懂度与自然度，1.5 kbps 下与 Opus 6 kbps 的 WER 相近。在 LibriSpeech 全量 `test-clean` 与 `test-other` 评估中，LCA 在所有 `L=1/2/3` 工作点上稳定改善 mel-L1 与 STOI；在 index dropout/substitution、packet/burst loss 与困难+扰动 ASR/WER 上，LCA 进一步呈现可解释的鲁棒性收益。消融显示，HuBERT 语义蒸馏与强扰动 + consistency loss 是当前训练配方的关键成分；NAS encoder 显著降低发送端复杂度，但其与 hand-designed encoder 的质量等价性仍待补充。后续工作将扩展到跨语料、跨语言、AMR-WB 与主观听测评估，并完成 packetized payload、真实网络与多用户实时通信验证。

## 参考文献

[1] Zeghidour, N., Luebs, A., Omran, A., Skoglund, J., & Tagliasacchi, M. SoundStream: An End-to-End Neural Audio Codec. arXiv:2107.03312.

[2] Défossez, A., Copet, J., Synnaeve, G., & Adi, Y. High Fidelity Neural Audio Compression. arXiv:2210.13438.

[3] Kumar, R., Seetharaman, P., Luebs, A., Kumar, I., & Kumar, K. High-Fidelity Audio Compression with Improved RVQGAN. arXiv:2306.06546.

[4] Zhang, X., Zhang, D., Li, S., Zhou, Y., & Qiu, X. SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models. arXiv:2308.16692.

[5] Ji, S., et al. WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling. arXiv:2408.16532.

[6] Weng, Z., Qin, Z., Tao, X., Pan, C., Liu, G., & Li, G. Y. DeepSC-S: Deep Semantic Communication for Speech Signals. arXiv:2012.05369.

[7] Qin, Z., Tao, X., Lu, J., Tong, W., & Li, G. Y. Semantic Communications: Principles and Challenges. arXiv:2201.01389.

[8] Weng, Z., Qin, Z., Tao, X., Pan, C., Liu, G., & Li, G. Y. Deep Learning Enabled Semantic Communications with Speech Recognition and Synthesis. arXiv:2205.04603.

[9] Large Speech Model Enabled Semantic Communication. arXiv:2512.04711.

[10] Glaris: Error-Resilient Semantic Communication for Speech Transmission over Packet-Loss Networks. arXiv:2512.08203.

## 附录 A：图表清单（建议）

1. **图 1 系统图**：输入语音 → NAS encoder → continuous representation → RVQ indices → 仅传前 `L` 层 → shared codebook lookup → decoder → 重建语音。图中标注 `L=1/2/3` 对应 500/1000/1500 bps，并明确标注“码本/解码器/系统配置 = 离线预共享”、“信道载荷 = 离散索引序列”。
2. **图 2 负载-质量曲线**：x 轴为 bps（log 刻度），y 轴为 WER / STOI / PESQ / mel-L1，分别绘制四张子图。曲线包括 SCIT-Base、SCIT-LCA、Opus、EnCodec、DAC，并突出 500/1000/1500 bps 工作点。
3. **图 3 全量 clean 泛化主表**：渲染表 2（主表 A）作图，可附带条形图比较 ΔSTOI 与 Δmel-L1。
4. **图 4 index dropout/substitution 鲁棒性热力图**：行 = L=1/2/3，列 = `dropout-high` / `substitution-high` × `test-clean` / `test-other`，颜色 = Δmel-L1 与 ΔSTOI。
5. **图 5 packet/burst loss 鲁棒性热力图**：行 = L=1/2/3，列 = `packet-loss-5p` / `burst-10f` × `test-clean` / `test-other`，颜色同上。
6. **图 6 ASR/WER 表**：合并表 5（Exp10）与表 6（Exp11），按 (clean / 扰动) × (test-clean / test-other) × L 形成紧凑表；可附带 ΔWER 条形图，特别标注 `test-clean_100` L2/L3 略差与 `test-other_100` 全面改善的对比。
7. **图 7 消融图**：四子图：(a) distill0 vs distill30 dev/mel 曲线；(b) Base vs LCA clean Δmel-L1 / ΔSTOI 条形图；(c) LCA v1 vs v2 robustness improvement 条形图；(d) NAS vs hand-designed encoder 在 Params / MACs / RTF 上的相对降幅。

## 附录 B：仍需人工确认的点

1. **5.1 节 baseline 数值**：表 1 与正文使用了实验记录中的 STOI/PESQ/WER；mel-L1 等其他列建议在最终稿中补齐对应 baseline 数值，并核对 Opus 6 kbps WER（0.147）与 EnCodec 1.5 kbps WER（0.302）等关键数值是否仍来自 `exp4_baseline_comparison_*` 的最新报告。
2. **§5.4 packet/burst 主表 PESQ 行为**：表 4 中 L2/L3 下 PESQ 部分行为略低，需在最终稿中确认是否与 §5.3 表 3 表述一致；如有差异需在正文显式说明。
3. **§5.6.2 robustness improvement 单元数**：v1 与 v2 的“LCA 更鲁棒 cell 数”（如 mel-L1 上 v1 3/12、v2 12/12）来自 exp3 v1 vs v2 报告，最终稿建议核对这些 cell 计数是否对应当前 exp3 v1（弱扰动）与 exp3 v2（强扰动 + consistency）报告。
4. **§7.7 codebook 利用率数值**：dead code ratio 与 usage rate 的具体数字（L1 17.48% / 82.52% 等）来自 exp2 当前 run 的 32 样本统计；最终稿如需放入正文，建议改为带样本数与统计区间的表述。
5. **图表清单中各图的具体来源 CSV/JSON**：所有图建议在最终稿中给出对应 `output/experiments/...` 数据文件路径，便于复现与审稿核查。
6. **AMR-WB / Codec2 / 主观 MOS 的最终归属**：当前归入“限制”；如果在投稿前完成补充实验，需要回到 §4.2、§5.1 与摘要相应改写。
7. **跨语料与跨语言泛化结果**：当前局限性中明确未覆盖；如果补做 VCTK / AISHELL，需在 §4.1 与 §5 增加章节并改写摘要中相应表述。


