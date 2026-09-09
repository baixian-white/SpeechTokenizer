# Low-Load Semantic Speech Communication via Shared Codebook Index Transmission

中文暂定题目：基于共享码本索引传输的低负载语音语义通信

本文档用于沉淀当前论文框架。它不是正式初稿，而是后续写作、实验组织和图表设计的总大纲。

## 1. 论文定位

### 1.1 核心问题

语音通信在未来 6G、边缘智能、空天地一体化、保密通信和量子安全通信等场景中，可能面临带宽、能耗、密钥资源或链路容量受限的问题。传统语音通信和音频压缩通常以信号级保真为目标，但在资源受限通信中，系统真正需要的是以更低负载传输足以支持理解和感知的关键信息。

本文关注的问题是：

> 语音通信是否必须传输完整波形或高维连续特征？如果发送端和接收端共享码本，是否可以仅传输紧凑的离散索引序列，从而在显著降低通信负载的同时保持可用的语音自然度、可懂度和语义信息？

### 1.2 论文主张

本文提出一种基于共享码本索引传输的低负载语音语义通信框架。该框架将连续语音表示映射为离散码本索引，并在发送端和接收端部署共享码本。通信信道中只传输索引序列，而非波形或高维连续特征；接收端再基于接收到的索引和共享码本恢复语音。

本文不以最高波形重建保真度为唯一目标，而是面向未来受限通信场景，探索离散索引作为低负载语音语义传输单元的可行性。

### 1.3 论文边界

- 本文不是纯语音压缩论文，不应将主要卖点写成“恢复质量全面优于传统或神经音频压缩方法”。
- 本文不是语音 tokenizer 论文，不应将贡献写成“提出一种全新的语音离散化模型”。
- 本文的核心贡献应落在通信框架、共享码本索引传输机制、通信负载建模和压缩负载与语义可用性之间的权衡分析。
- 6G 语义通信和量子安全通信窄带宽/资源受限场景应作为应用牵引背景，而不是把论文硬绑定成量子通信论文。

## 2. 核心研究问题

### 2.1 主研究问题

Can a shared-codebook index transmission framework reduce the communication load of speech transmission while maintaining usable speech naturalness, intelligibility, and semantic adequacy?

中文表述：

> 基于共享码本的索引传输框架能否降低语音通信负载，并在低负载条件下维持可用的语音自然度、可懂度和语义信息？

### 2.2 子问题

1. 离散码本索引能否作为语音语义通信中的有效传输单元，替代波形或高维连续特征？
2. 在不同传输负载下，索引级传输对语音自然度、可懂度和语义保持的影响如何变化？
3. 与传统语音编码、神经音频压缩和现有语义通信方案相比，该框架能否在更低负载下维持可接受的通信可用性？
4. 在未来 6G、边缘设备、窄带保密链路或量子安全通信辅助场景中，该低负载机制具有怎样的潜在适用性？

## 3. 预期贡献

### 3.1 贡献一：通信负载导向的语音语义通信框架

从语音通信负载瓶颈出发，将语音传输重新建模为共享码本索引传输问题，而非波形传输或高维连续特征传输问题。

推荐表述：

> We formulate speech semantic communication as a shared-codebook index transmission problem, shifting the communication objective from waveform-level reconstruction to low-load transmission of communication-relevant speech information.

### 3.2 贡献二：共享码本索引传输机制

设计发送端-接收端共享码本机制。发送端将连续语音表示量化为离散索引，信道中仅传输索引序列；接收端基于共享码本和重建模块恢复语音。

推荐表述：

> We introduce an index-only transmission mechanism in which compact codebook indices, rather than waveforms or continuous acoustic features, are transmitted through the communication channel.

### 3.3 贡献三：传输负载与语音可用性的权衡分析

通过系统实验分析传输负载、压缩比和语音可用性之间的关系，证明该框架在显著降低负载时仍能维持可接受的语音通信质量。

推荐表述：

> We analyze the trade-off between transmission load and speech usability, showing that discrete index transmission can maintain usable speech communication under substantially reduced load.

## 4. Introduction 写作框架

### 4.1 第一段：未来通信背景

要点：

- 6G 正在从比特级可靠传输走向语义级高效传输。
- 中国 6G/IMT-2030 相关布局中，语义通信是未来网络的重要方向之一。
- 未来通信场景包括空天地一体、边缘智能、低功耗终端、保密通信和量子安全通信辅助链路。
- 这些场景并不总是拥有充足带宽或传输资源，因此“少传但可用”成为重要目标。

可写成：

> Future communication systems are moving beyond bit-level reliable transmission toward semantic-level efficient information delivery. In emerging 6G scenarios, including edge intelligence, space-air-ground integrated networks, secure communications, and quantum-secured communication links, transmission resources may remain highly constrained. Under such conditions, speech communication systems should not only transmit accurately, but also transmit compactly.

### 4.2 第二段：现有语音通信方法不足

要点：

- 传统语音编码和神经音频压缩主要优化信号重建保真度。
- 这些方法在音质上可能很强，但目标函数仍然接近“压缩后高保真重建”。
- 语义通信更关注意义、任务相关信息和通信可用性。
- 当前语音语义通信仍缺少一种紧凑、可解释、低负载的语音传输单元。

可写成：

> Existing speech coding and neural audio compression methods have achieved strong reconstruction fidelity, but they remain primarily optimized for signal-level restoration. Semantic communication, in contrast, emphasizes the transmission of meaning and task-relevant information under limited channel resources. This difference motivates a new question: whether speech can be transmitted through compact semantic units rather than waveform samples or continuous acoustic features.

### 4.3 第三段：本文方法

要点：

- 提出共享码本索引传输框架。
- 连续语音表示映射到离散索引。
- 信道中只传索引。
- 接收端基于共享码本恢复语音。
- 不追求最高波形保真，而追求低负载下的通信可用性。

可写成：

> To address this problem, we propose a low-load semantic speech communication framework based on shared codebook index transmission. The proposed framework maps continuous speech representations into discrete codebook indices, transmits only the compact index sequence through the channel, and reconstructs speech at the receiver using the shared codebook and a reconstruction module. Rather than maximizing waveform fidelity alone, the framework is designed to preserve communication-relevant speech information under reduced transmission load.

### 4.4 第四段：贡献总结

要点：

1. 提出通信负载导向的语音语义通信框架。
2. 设计共享码本索引传输机制。
3. 分析传输负载与语音可用性的权衡关系。

## 5. 论文整体结构

### 5.1 Abstract

目标：

- 简洁说明背景：未来受限通信场景需要低负载语音传输。
- 说明问题：传统语音压缩以波形保真为主，不完全符合语义通信目标。
- 说明方法：共享码本索引传输。
- 说明结果类型：显著降低传输负载，同时维持可用语音自然度、可懂度和语义信息。
- 说明意义：离散索引可作为低负载语音语义传输单元。

### 5.2 Introduction

建议小节或段落逻辑：

1. 6G、语义通信和受限链路背景。
2. 语音通信负载问题。
3. 传统语音编码、神经音频压缩和现有语义通信方法的不足。
4. 本文提出共享码本索引传输框架。
5. 贡献总结。

### 5.3 Related Work

建议分四条线写，不围绕单一具体模型展开。

#### 5.3.1 Speech Coding and Neural Audio Compression

写作重点：

- 传统语音编码和神经音频压缩在低码率高保真重建上表现强。
- 它们主要服务于音频压缩或重建保真，而不是语义通信中的最小负载传输。
- 本文不否认其重建质量优势，而是强调通信目标不同。

#### 5.3.2 Discrete Speech Representation and Codebook Quantization

写作重点：

- 离散表征学习说明连续语音或音频特征可以被映射到离散码本空间。
- 相关研究通常服务于表示学习、语音生成、语音理解或音频建模。
- 本文将离散索引重新定位为通信信道中的传输符号。

#### 5.3.3 Semantic Communication

写作重点：

- 语义通信从传输比特转向传输意义。
- 现有语义通信方法常见路径包括文本中间表示、高维语义特征或端到端编码。
- 对语音而言，仍需要一种低负载、结构化、可解释的传输单元。

#### 5.3.4 Gap Summary

这一节用于收束 Related Work，核心任务不是简单罗列已有工作，而是说明“现有研究已经做到哪里、还没有覆盖什么问题、因此本文为什么有必要存在”。写作时需要完成三件事：

1. 承认已有工作的贡献。
   - 语音编码和神经音频压缩已经在低码率语音/音频重建方面取得重要进展。
   - 离散语音表示和码本量化研究证明连续语音表示可以被映射为紧凑的离散单元。
   - 语义通信研究已经开始从比特级传输转向意义和任务相关信息传输。

2. 指出它们尚未覆盖本文的问题。
   - 语音编码和神经音频压缩主要优化重建保真，而不是以语义通信负载最小化为核心目标。
   - 离散语音表示方法多服务于语音建模、生成、理解或下游大模型，而不是面向通信信道中的低负载传输。
   - 现有语音语义通信方法尚未充分探索“发送端和接收端共享码本、信道中仅传输索引序列”的传输范式。

3. 自然引出本文方法。
   - 因此，本文从语音通信负载问题出发，提出基于共享码本索引传输的低负载语音语义通信框架。
   - 该框架将离散索引重新定位为通信信道中的紧凑传输单元，而不是单纯的语音建模中间表示。

可形成 gap 表述：

> Existing speech coding methods optimize reconstruction fidelity, discrete speech representation methods are mainly designed for modeling or generation, and semantic communication methods have not fully explored shared-codebook index transmission for low-load speech delivery. This leaves open the question of whether compact discrete speech indices can serve as practical transmission units for semantic speech communication.

### 5.4 Proposed Framework

建议小节：

#### 5.4.1 System Overview

本小节在论文正文中用于概述整体框架，并插入 Figure 2。这里仅保留章节结构提示，详细正文草稿见第 10 节，Figure 2 素材与图片生成提示词见第 11 节。

建议正文覆盖：
- 系统由发送端、共享码本、低负载通信信道和接收端组成。
- 发送端将语音映射为连续表示，并进一步量化为离散索引。
- 信道中仅传输离散索引序列，不传输波形、连续特征或码本本身。
- 接收端基于预共享码本查找和重建模块恢复语音。
- 本文目标是低负载下的语音通信可用性，而不是最高波形级重建保真。

Figure 2 插入位置：
> [插入 Figure 2：Overall architecture of the proposed shared-codebook index transmission framework]

核心链路：
输入语音 -> 连续语音表示 -> 码本量化 -> 索引序列 -> 信道传输 -> 共享码本查表 -> 接收端重建 -> 输出语音。

#### 5.4.2 Transmitter-Side Encoding

说明发送端完成：

- 语音特征提取。
- 连续表示生成。
- 码本量化。
- 索引序列输出。

#### 5.4.3 Shared Codebook Mechanism

说明：

- 码本在发送端和接收端共享。
- 通信过程中不传输码本内容，只传输索引。
- 码本可视为双方预部署的语音语义知识结构。

#### 5.4.4 Index-Only Channel Transmission

说明：

- 信道负载由索引长度、码本大小和索引比特数决定。
- 传输对象从高维连续信号变为紧凑 index stream。
- 可以进一步讨论信道噪声、索引错误和纠错机制。

#### 5.4.5 Receiver-Side Reconstruction

说明：

- 接收端根据索引从共享码本中恢复离散或连续表示。
- 重建模块生成语音。
- 输出目标是通信可用语音，而非逐样本最高保真重建。

### 5.5 Communication Load Formulation

这是论文中需要严谨化的一节。

建议定义：

- 原始波形负载。
- 连续特征传输负载。
- 索引传输负载。
- 压缩比。
- 每秒索引数。
- 每个索引所需比特数。
- 总传输负载。

可用公式方向：

> If the codebook size is K, each transmitted index requires log2(K) bits. Given an index rate of R indices per second, the transmission load is approximately R log2(K) bits per second, excluding optional channel coding overhead.

需要注意：

- 如果存在多码本、多层索引或残差量化，需要分别计算。
- 如果加入纠错码，需要报告 raw index load 和 channel-coded load 两种版本。
- 不要只报告压缩比，也要报告实际 bitrate 或 symbol rate。

### 5.6 Experimental Setup

#### 5.6.1 Datasets

待定。需要覆盖：

- 语音自然度。
- 可懂度。
- 语义内容。
- 多说话人。
- 可选：情感语音数据。

#### 5.6.2 Baselines

建议包含三类：

1. 传统语音编码方法：作为通信系统常见基线。
2. 神经音频压缩方法：作为低码率高保真重建强基线。
3. 语义通信方法：作为同类任务基线。

表述策略：

- 对传统和神经压缩方法，不要预设“全面胜出”。
- 对比重点是不同负载下的可用性曲线，而不是单点音质排名。

#### 5.6.3 Metrics

核心指标：

- Transmission load / bitrate。
- Compression ratio。
- Intelligibility：例如 STOI 或基于 ASR 的 WER/CER。
- Naturalness：例如 MOS、PESQ、ViSQOL 或主观评价。
- Semantic adequacy：例如转写文本相似度、语义相似度或下游理解任务表现。

可选诊断指标：

- Speaker similarity。
- Emotion preservation。
- Robustness under channel noise。

注意：这些指标用于证明通信可用性，不包装为“建立全新的多属性评价体系”。

### 5.7 Results and Analysis

建议结果组织：

#### 5.7.1 Transmission Load Reduction

展示：

- 相比波形或连续特征传输，索引传输的负载降低。
- 不同码本大小、索引率、量化层数下的负载变化。

#### 5.7.2 Speech Usability Under Low Load

展示：

- 可懂度、自然度、语义保持随负载下降的变化。
- 标出可用阈值或主观可接受区间。

#### 5.7.3 Load-Usability Trade-off

核心图：

> transmission load vs. speech usability curve

目标：

- 证明本文方法在极低负载区仍保持通信可用。
- 不要求在所有质量指标上超过纯压缩方法。

#### 5.7.4 Ablation Study

可能消融：

- 不同码本大小。
- 不同索引率。
- 不同量化层数。
- 是否使用共享码本。
- 信道噪声或索引错误条件。

#### 5.7.5 Robustness Analysis

如果实验允许，可加入：

- 丢包。
- 比特错误。
- 索引错误。
- 窄带链路约束。

### 5.8 Discussion

建议讨论重点：

1. 为什么本文方法适合语义通信，而不是被简单理解为音频压缩替代品。
2. 在 6G 语义通信中的潜在价值：低负载、端侧部署、语义级传输。
3. 在量子安全通信或窄带保密链路中的潜在价值：减少需要保护或传输的数据负载。
4. 局限性：音质可能不如专门优化的神经音频压缩方法，重建质量依赖码本和接收端重建模块。
5. 未来工作：信道编码、端到端优化、鲁棒索引传输、多模态语义通信。

### 5.9 Conclusion

结论应强调：

- 本文从语音通信负载问题出发。
- 提出共享码本索引传输框架。
- 证明离散索引可作为低负载语音语义通信单元。
- 该方向为未来 6G 和资源受限安全通信场景提供一种可行路径。

## 6. 图表规划

### Figure 1: Motivation and System Context

展示：

- 6G/边缘/保密/量子安全受限链路。
- 高负载语音传输的问题。
- 低负载语义索引传输的必要性。

### Figure 2: Proposed Shared-Codebook Index Transmission Framework

展示：

发送端、共享码本、索引信道、接收端重建。

### Figure 3: Transmitter-Side Continuous Representation Encoder

展示：

- 输入语音 `x(t)`。
- 初始一维卷积层。
- 多级 residual block + strided convolution 下采样编码器。
- 有效时间下采样倍率 `S = 2 × 4 × 5 × 8 = 320`。
- 时序建模模块。
- 连续潜在表示 `Z ∈ R^{d × T_q}`。
- 从 `Z` 指向码本量化模块的虚线箭头。

### Figure 4: K-means Initialized RVQ Codebook Iteration

展示：

- 训练表示样本或当前 RVQ 层残差样本 `R_m = {r_t^{(m)}}`。
- 从样本中通过 K-means 得到初始码本 `C_m^{(0)}`。
- 最近邻分配 `i_t^{(n)} = argmin_k ||r_t^{(m,n)} - c_{m,k}^{(n)}||^2`。
- 根据分配结果更新 codeword / centroid，形成 `C_m^{(n+1)}`。
- “分配 → 更新”循环迭代至收敛，得到 `C_m^*`。
- 当前层收敛后计算残差 `r_t^{(m+1)} = r_t^{(m)} - c_{m,i_t}`，继续学习下一层码本。
- 最终输出固定共享 RVQ 码本 `C^* = {C_1^*, ..., C_M^*}`。

### Figure 5: Fixed-Codebook RVQ Lookup for Discrete Index Generation

展示：

- 连续潜在表示 `Z`。
- 离线训练后固定的多层码本 `C^* = {C_1^*, ..., C_M^*}`。
- 每层 RVQ 对当前残差执行 nearest-neighbor lookup。
- 每层输出一个量化向量 `q_{m,t}` 和一个码本索引 `i_{m,t}`。
- 残差递推过程 `r_t^{(m+1)} = r_t^{(m)} - q_{m,t}`。
- 每个时间步和每个 RVQ 层对应的索引 `i_{m,t}`。
- 离散索引张量 `I ∈ {1,...,K}^{M × T_q}`。
- 可选前 `L` 层截断机制 `I_L = I_{1:L,:}`。

### Figure 6: Transmitter-Side Speech-to-Index Overview

展示：

- 输入语音 `x(t)`。
- 发送端编码器产生连续表示 `Z`。
- 连续表示进入压缩表示的固定共享 RVQ 量化模块 `Q(Z; C^*)`。
- 固定共享码本 `C^*` 作为在线量化工具参与处理，但码本向量本身不传输。
- RVQ 模块输出离散索引矩阵 `I`，但不在图中展开三层残差机制。
- 层选择 / 负载控制模块保留前 `L` 层索引。
- 最终待传输对象为 `I_L = I_{1:L,:}`。

### Figure 7: Receiver-Side Codebook Lookup and Speech Reconstruction

展示：

- 接收索引矩阵 `\hat{I}_L`。
- 接收端本地固定共享码本 `C^*`。
- 根据索引进行 codeword lookup。
- 多层 codeword 求和恢复潜在表示 `\hat{Z}`。
- 语音重建解码器 `D_\psi`。
- 输出重建语音 `\hat{x}(t)`。
- 图的结构应与 Figure 3 对称：Figure 3 是 `x(t) -> Z`，Figure 7 是 `\hat{Z} -> \hat{x}(t)`，但 Figure 7 前面需要补充索引查表得到 `\hat{Z}` 的步骤。

### Figure 8: Communication Load Comparison

展示：

- 波形传输。
- 连续特征传输。
- 传统编码。
- 神经压缩。
- 本文索引传输。

### Figure 9: Load-Usability Trade-off Curve

核心图。

横轴：transmission load / bitrate / compression ratio。

纵轴：speech usability，例如可懂度、语义保持、自然度。

### Figure 10: Ablation Analysis

展示码本大小、索引率、量化层数等因素对负载和质量的影响。

### Table 1: Comparison with Existing Paradigms

比较维度：

- 传输对象。
- 优化目标。
- 是否共享码本。
- 是否 index-only。
- 主要适用场景。

### Table 2: Experimental Metrics

列出：

- 指标名称。
- 衡量对象。
- 越高越好或越低越好。
- 解释意义。

### Table 3: Main Results

报告：

- 负载。
- 压缩比。
- 可懂度。
- 自然度。
- 语义保持。

## 7. 论证风险与规避策略

### 7.1 风险一：被认为只是语音离散化方法的通信应用

规避：

- 主线始终从通信负载问题出发。
- Related Work 中将离散表征作为技术背景，而不是本文问题来源。
- Contribution 不写“提出新的 tokenizer”，写“提出共享码本索引传输框架”。

### 7.2 风险二：音质不如纯压缩方法

规避：

- 不以最高重建保真为目标。
- 承认纯压缩方法在音质上可能更强。
- 强调本文关注低负载下的通信可用性和语义保持。

### 7.3 风险三：6G 和量子通信背景显得过大

规避：

- 只把 6G 语义通信和量子安全通信作为资源受限场景背景。
- 不宣称本文解决量子通信核心物理问题。
- 重点仍然落在低负载语音语义传输。

### 7.4 风险四：语义保持指标不够有说服力

规避：

- 使用 ASR 转写相似度、WER/CER、语义相似度或下游理解任务来支持。
- 如果条件允许，加入主观听测。
- 明确区分 naturalness、intelligibility 和 semantic adequacy。

## 8. 推荐摘要草稿

Speech communication systems face a persistent tension between transmission load and speech usability, especially in emerging 6G, edge-intelligent, secure, and bandwidth-constrained communication scenarios. Conventional speech coding and neural audio compression methods primarily optimize signal-level reconstruction fidelity, which may remain costly when the communication objective is to deliver meaning under limited resources. In this work, we propose a low-load semantic speech communication framework based on shared codebook index transmission. The framework maps continuous speech representations into compact discrete codebook indices, transmits only the index sequence through the channel, and reconstructs speech at the receiver using the shared codebook and a reconstruction module. Rather than pursuing maximum waveform fidelity, the proposed framework focuses on preserving communication-relevant speech information under substantially reduced transmission load. Experiments are designed to analyze the trade-off between transmission load and speech usability, including intelligibility, naturalness, and semantic adequacy. The results are expected to demonstrate that discrete codebook indices can serve as effective low-load transmission units for semantic speech communication.

## 9. 下一步待完成事项

1. 明确实验数据集。
2. 明确当前可用模型和重建流程。
3. 计算索引传输 bitrate 与压缩比。
4. 确定 baseline 列表。
5. 设计 Figure 2 系统框架图。
6. 补充并核验 6G 语义通信、语义通信、神经音频压缩、离散语音表示和量子安全通信相关文献。
7. 将本大纲扩展为 Introduction 初稿。

## 10. 正文章节写作说明

### 10.1 5.4.1 System Overview

5.4.1 正文说明草稿：

这一小节不能只放 Figure 2，还需要用正文明确说明系统组成、数据流、信道传输内容和本文框架与传统语音压缩/普通 encoder-decoder 的区别。建议正文至少覆盖以下五点：

1. 系统由发送端、共享码本、低负载通信信道和接收端组成。
2. 发送端将输入语音编码为连续语音表示，并通过码本量化得到离散索引序列。
3. 通信信道中真正传输的不是波形，也不是高维连续特征，而是紧凑的索引序列。
4. 共享码本在发送端和接收端预先部署，因此每次通信不需要传输码本内容。
5. 本文关注低负载下的通信可用性，而不是追求最高波形重建保真度。

中文正文草稿：

> 如 Figure 2 所示，本文提出的低负载语音语义通信框架由发送端、共享码本、低负载通信信道和接收端四个部分组成。该框架的核心思想是将语音通信中的传输对象从波形或高维连续特征转换为紧凑的离散码本索引，从而降低信道中实际需要承载的信息量。
>
> 在发送端，输入语音首先被映射为连续语音表示，该表示包含语音内容、声学结构以及与感知相关的关键信息。随后，连续表示通过共享码本进行量化，得到离散索引序列。与直接传输波形或连续特征不同，本文框架仅将这些离散索引作为通信符号发送到信道中。
>
> 共享码本是该框架降低传输负载的关键。码本在发送端和接收端预先部署，通信过程中不随每段语音重复传输。若码本大小为 K，索引序列长度为 T，则在不考虑可选信道编码开销的情况下，原始索引传输负载约为 T log2(K) bits。由此，系统可以将高维语音传输问题转化为低负载索引传输问题。
>
> 在接收端，系统根据接收到的索引序列进行共享码本查找，恢复相应的潜在语音表示，并通过语音重建解码器生成输出语音。需要强调的是，本文框架并不以最高波形级重建保真度为唯一目标，而是关注在通信资源受限条件下，如何以较低负载维持可用的语音自然度、可懂度和语义信息。

英文正文草稿：

> As illustrated in Fig. 2, the proposed low-load semantic speech communication framework consists of four main components: a transmitter, a shared codebook, a low-load communication channel, and a receiver. The central idea is to replace waveform or high-dimensional continuous-feature transmission with compact discrete codebook indices, thereby reducing the amount of information that must be carried by the communication channel.
>
> At the transmitter, the input speech signal is first mapped into a continuous speech representation that captures speech content, acoustic structure, and perceptually relevant information. This continuous representation is then quantized using a shared codebook, producing a sequence of discrete indices. Unlike waveform-level or continuous-feature transmission, the proposed framework sends only these discrete indices as communication symbols.
>
> The shared codebook is the key mechanism that enables low-load transmission. It is pre-deployed at both the transmitter and receiver and is not repeatedly transmitted for each utterance. Given a codebook size K and an index sequence length T, the raw transmission load is approximately T log2(K) bits, excluding optional channel coding overhead. In this way, speech communication is transformed from high-dimensional signal transmission into compact index transmission.
>
> At the receiver, the received index sequence is mapped back to codebook entries through shared codebook lookup, producing a recovered latent speech representation. A speech reconstruction decoder then synthesizes the output speech. Importantly, the objective of the proposed framework is not to maximize waveform-level reconstruction fidelity alone, but to maintain usable speech naturalness, intelligibility, and semantic adequacy under constrained communication resources.

### 10.2 5.4.2 Transmitter-Side Encoding

本小节建议改写为“发送端连续表示编码”。它只回答一个问题：原始语音 `x(t)` 如何被发送端编码器映射为连续潜在表示 `Z`。不要在本小节展开码本构建、RVQ 逐层量化、索引率、bitrate 或压缩比；这些内容分别放入 5.4.3 和 5.5。

建议覆盖：

- 输入语音的形式、采样率和预处理。
- 发送端编码器的功能：从波形中提取连续语音表示。
- 编码器的时域下采样结构如何决定潜在序列长度。
- 输出连续表示 `Z` 的数学定义、维度和角色。
- 强调 `Z` 是后续码本量化的输入，但不是通信信道中的直接传输对象。
- 配 Figure 3：Transmitter-Side Continuous Representation Encoder。

建议公式：

```text
x = [x_1, x_2, ..., x_N] ∈ R^N
```

```text
Z = E_\theta(x),    Z = [z_1, z_2, ..., z_{T_q}],    z_t ∈ R^d
```

```text
S = \prod_{j=1}^{J} s_j
```

```text
T_q ≈ \left\lceil \frac{N}{S} \right\rceil
```

其中 `S` 是编码器的总时间下采样倍率。当前实现中，stride 配置为 `[8, 5, 4, 2]`，因此：

```text
S = 8 × 5 × 4 × 2 = 320
```

中文示例表述：

> 在发送端，输入语音波形表示为 `x = [x_1, x_2, ..., x_N] ∈ R^N`。为保证编码器输入的时间尺度一致，语音首先被重采样到模型采样率 `f_s`，并转换为单声道波形。在当前实现中，模型采样率为 `f_s = 16 kHz`。
>
> 发送端编码器 `E_\theta(·)` 的作用是将原始波形映射为连续潜在表示序列：
>
> ```text
> Z = E_\theta(x),    Z = [z_1, z_2, ..., z_{T_q}],    z_t ∈ R^d
> ```
>
> 其中，`z_t` 表示第 `t` 个潜在时间步的连续语音表示，`d` 为潜在表示维度。当前实现中，潜在表示维度为 `d = 1024`。与原始波形相比，`Z` 具有更紧凑的时间结构，并保留语音内容、声学结构以及与感知相关的关键信息。
>
> 编码器通过多级一维卷积、残差模块和时域下采样操作逐步压缩输入波形的时间长度。设编码器共有 `J` 个下采样阶段，第 `j` 个阶段的 stride 为 `s_j`，则总时间下采样倍率为：
>
> ```text
> S = \prod_{j=1}^{J} s_j
> ```
>
> 当前实现中的 stride 配置为 `[8, 5, 4, 2]`，因此：
>
> ```text
> S = 8 × 5 × 4 × 2 = 320
> ```
>
> 对于长度为 `N` 的输入波形，连续潜在表示的时间步数可近似表示为：
>
> ```text
> T_q ≈ \left\lceil \frac{N}{S} \right\rceil
> ```
>
> 这说明发送端编码器首先在时间维度上形成低帧率的连续语音表示，而不是直接产生最终待传输的索引。需要注意的是，配置中的 `hop_size = 240` 主要用于 Mel/STFT 损失相关计算，并不决定编码器输出 `Z` 的时间步数。实际输出长度应以编码器或量化模块输出张量的时间维度为准。
>
> 因此，5.4.2 的重点是建立从输入波形到连续表示的映射关系。连续表示 `Z` 是后续共享码本量化的输入，但在本文框架中并不作为通信信道的直接传输对象。

English example:

> At the transmitter, the input speech waveform is denoted as `x = [x_1, x_2, ..., x_N] ∈ R^N`. To ensure a consistent temporal scale for encoding, the waveform is first resampled to the model sampling rate `f_s` and converted into a monaural signal. In the current implementation, the model operates at `f_s = 16 kHz`.
>
> The transmitter-side encoder `E_\theta(·)` maps the raw waveform into a sequence of continuous latent representations:
>
> ```text
> Z = E_\theta(x),    Z = [z_1, z_2, ..., z_{T_q}],    z_t ∈ R^d
> ```
>
> where `z_t` denotes the continuous speech representation at the `t`-th latent time step, and `d` is the latent dimensionality. In the current implementation, `d = 1024`. Compared with the raw waveform, `Z` provides a more compact temporal representation while preserving speech content, acoustic structure, and perceptually relevant information.
>
> The encoder progressively reduces the temporal length of the input waveform through cascaded one-dimensional convolutional layers, residual blocks, and temporal downsampling operations. Let `s_j` denote the stride of the `j`-th downsampling stage. The overall temporal downsampling factor is:
>
> ```text
> S = \prod_{j=1}^{J} s_j
> ```
>
> In the current implementation, the stride configuration is `[8, 5, 4, 2]`, yielding:
>
> ```text
> S = 8 × 5 × 4 × 2 = 320
> ```
>
> For an input waveform of length `N`, the temporal length of the continuous latent representation can be approximated as:
>
> ```text
> T_q ≈ \left\lceil \frac{N}{S} \right\rceil
> ```
>
> This step produces a low-frame-rate continuous speech representation rather than the final transmitted indices. It should be noted that the `hop_size = 240` parameter in the configuration is mainly associated with Mel/STFT loss computation and does not determine the temporal length of `Z`. The actual temporal length should be read from the output tensor of the encoder or quantization module.
>
> Therefore, Section 5.4.2 focuses on the mapping from waveform input to continuous latent representation. The resulting representation `Z` serves as the input to the shared-codebook quantization stage, but it is not directly transmitted over the communication channel in the proposed framework.

#### 10.2.8 备选文字示例：编码与量化合并版

说明：当前大纲倾向于将 `5.4.2` 写成“发送端连续表示编码”，并将码本量化细节放入 `5.4.3 Shared Codebook Mechanism`。如果后续决定让 `5.4.2` 同时覆盖“连续表示编码 + 索引生成”，可使用下面这版表述；若保持当前拆分，则其中的码本量化段落应移动到 `10.3`。

中文备选表述：

> 在发送端，输入语音波形记为 `x ∈ R^N`。发送端编码器 `E_\theta(·)` 首先将语音映射为连续潜在表示序列 `Z = E_\theta(x) = {z_1, z_2, ..., z_T}`，其中 `z_t ∈ R^d` 表示第 `t` 个时间步的连续语音表示。该表示用于保留与语音内容和感知质量相关的信息，但并不直接通过信道传输。
>
> 随后，发送端利用共享码本 `C = {c_1, c_2, ..., c_K}` 对连续表示进行量化。对于每个潜在向量 `z_t`，量化器 `Q(·)` 将其映射到码本中的一个离散索引 `i_t`。在最近邻量化形式下，该过程可表示为 `i_t = argmin_k ||z_t - c_k||_2^2`。最终，发送端得到离散索引序列 `i = {i_1, i_2, ..., i_T}`。
>
> 需要强调的是，发送端送入通信信道的仅为离散索引序列，而不是原始语音波形、连续潜在表示或码本内容。若编码器每隔 `H` 个采样点产生一个索引，则索引率约为 `R_index = f_s / H`，其中 `f_s` 为采样率。该索引率将作为后续通信负载建模的基础。

English alternative draft:

> At the transmitter, the input speech waveform is denoted as `x ∈ R^N`. A speech representation encoder `E_\theta(·)` first maps the waveform into a sequence of continuous latent representations, `Z = E_\theta(x) = {z_1, z_2, ..., z_T}`, where `z_t ∈ R^d` denotes the latent vector at time step `t`. These representations preserve communication-relevant speech information, but they are not directly transmitted through the channel.
>
> The continuous representations are then quantized using a shared codebook `C = {c_1, c_2, ..., c_K}`. For each latent vector `z_t`, the quantizer maps it to a discrete codebook index `i_t`. Under nearest-neighbor quantization, this operation can be written as `i_t = argmin_k ||z_t - c_k||_2^2`. The transmitter therefore outputs a discrete index sequence `i = {i_1, i_2, ..., i_T}`.
>
> Importantly, only the index sequence is passed to the communication channel; the waveform, continuous latent representations, and codebook entries are not transmitted. If the encoder produces one index every `H` samples, the resulting index rate is approximately `R_index = f_s / H`, where `f_s` is the sampling rate. This index rate provides the basis for the communication-load formulation in the following section.

### 10.3 5.4.3 Shared Codebook Mechanism

本小节建议改写为“共享码本构建与残差量化机制”。它需要回答三个问题：码本如何由 K-means 初始化并迭代收敛，连续表示如何通过固定码本逐层量化，最终索引如何得到。为了避免把论文写成“提出一个新的语音 tokenizer”，本节应将码本构建表述为通信系统部署前的离线码本学习过程，而不是强调从零训练一个新模型。

建议覆盖：

- 离线阶段：从训练表示样本或残差样本中构造第 `m` 层训练集合 `R_m = {r_t^{(m)}}`。
- 初始化阶段：对 `R_m` 执行 K-means，得到初始码本 `C_m^{(0)}`。
- 迭代阶段：交替执行最近邻分配和 codeword / centroid 更新，直到第 `m` 层码本收敛为 `C_m^*`。
- 层间递推：当前层收敛后计算残差 `r_t^{(m+1)}`，并用该残差集合继续学习下一层码本。
- 部署阶段：同一组码本固定并预部署在发送端和接收端。
- 在线发送阶段：连续表示 `Z` 通过多层 RVQ 被转换为索引矩阵 `I`。
- 每一层 RVQ 对上一层残差进行量化，并输出一个码本索引。
- 最终索引矩阵 `I` 是信道传输前的离散表示。
- 配 Figure 4：K-means Initialized RVQ Codebook Iteration。
- 配 Figure 5：Fixed-Codebook RVQ Lookup for Discrete Index Generation。
- 配 Figure 6：Transmitter-Side Speech-to-Index Overview。

建议公式：

```text
C = {C_1, C_2, ..., C_M}
```

```text
C_m = {c_{m,1}, c_{m,2}, ..., c_{m,K}},    c_{m,k} ∈ R^d
```

第 `m` 层离线码本训练样本：

```text
R_m = {r_t^{(m)}},    r_t^{(1)} = z_t
```

K-means 初始码本：

```text
C_m^{(0)} = KMeans(R_m, K)
```

第 `n` 次迭代的最近邻分配：

```text
i_t^{(n)} = \arg\min_{k ∈ {1,...,K}}
\left\| r_t^{(m,n)} - c_{m,k}^{(n)} \right\|_2^2
```

Codeword / centroid 更新：

```text
c_{m,k}^{(n+1)}
= mean\{r_t^{(m,n)}: i_t^{(n)} = k\}
```

当前层收敛后的残差递推：

```text
r_t^{(m+1)} = r_t^{(m)} - c_{m,i_t}^{*}
```

概念性的离线码本学习目标可以写为：

```text
\min_C \sum_{z ∈ \mathcal{Z}_{train}}
\left\| z - \sum_{m=1}^{M} c_{m,i_m(z)} \right\|_2^2
```

逐层 RVQ 量化过程：

```text
r_{0,t} = z_t
```

```text
i_{m,t} = \arg\min_{k ∈ {1,...,K}} \left\| r_{m-1,t} - c_{m,k} \right\|_2^2
```

```text
q_{m,t} = c_{m,i_{m,t}}
```

```text
r_{m,t} = r_{m-1,t} - q_{m,t}
```

经过 `L` 层后的近似表示：

```text
\tilde{z}^{(L)}_t = \sum_{m=1}^{L} q_{m,t}
= \sum_{m=1}^{L} c_{m,i_{m,t}}
```

最终索引矩阵：

```text
I = {i_{m,t}} ∈ {1, ..., K}^{M × T_q}
```

如果只保留前 `L` 层：

```text
I_L = I_{1:L, :},    1 ≤ L ≤ M
```

中文示例表述：

> 共享码本是本文框架实现索引级通信的关键。与通信过程中随语音样本动态传输的特征不同，码本在系统部署前通过离线码本学习过程获得，并在在线通信阶段保持固定。本文不将码本本身作为信道传输对象，而是将其视为发送端和接收端预先共享的离散语音表示字典。
>
> 设共享码本集合为：
>
> ```text
> C = {C_1, C_2, ..., C_M}
> ```
>
> 其中，第 `m` 层码本表示为：
>
> ```text
> C_m = {c_{m,1}, c_{m,2}, ..., c_{m,K}},    c_{m,k} ∈ R^d
> ```
>
> 这里，`M` 表示 RVQ 层数，`K` 表示每层码本的条目数，`c_{m,k}` 是第 `m` 层码本中的第 `k` 个 codeword。当前实现中，默认 `M = 3`，`K = 1024`。从通信角度看，码本可以被理解为发送端和接收端预先共享的离散语音表示字典。
>
> 如 Figure 4 所示，离线码本构建从训练表示样本或当前 RVQ 层的残差样本开始。对第 `m` 层，记该层训练样本集合为：
>
> ```text
> R_m = {r_t^{(m)}},    r_t^{(1)} = z_t
> ```
>
> 首先对 `R_m` 执行 K-means 聚类，以得到初始码本：
>
> ```text
> C_m^{(0)} = KMeans(R_m, K)
> ```
>
> 该初始码本并不是最终共享码本，而是后续迭代优化的起点。在第 `n` 次迭代中，每个训练样本被分配给距离最近的 codeword：
>
> ```text
> i_t^{(n)} = \arg\min_{k ∈ {1,...,K}}
> \left\| r_t^{(m,n)} - c_{m,k}^{(n)} \right\|_2^2
> ```
>
> 随后，根据分配到第 `k` 个 codeword 的样本均值更新该 codeword：
>
> ```text
> c_{m,k}^{(n+1)}
> = mean\{r_t^{(m,n)}: i_t^{(n)} = k\}
> ```
>
> 最近邻分配和 codeword 更新交替执行，直到第 `m` 层码本收敛为 `C_m^*`。当前层收敛后，训练过程计算剩余残差：
>
> ```text
> r_t^{(m+1)} = r_t^{(m)} - c_{m,i_t}^{*}
> ```
>
> 并将该残差集合用于下一层码本学习。因此，多层 RVQ 码本不是一次性得到的单个字典，而是按照“初始码本、迭代更新、残差递推”的顺序逐层构建。最终得到的固定共享码本可表示为 `C^* = {C_1^*, C_2^*, ..., C_M^*}`。
>
> Figure 4 可在正文中承担这样的说明作用：K-means 提供初始聚类中心，迭代优化使 codeword 更好地覆盖训练表示空间，而残差递推使后续码本专注于补偿前一层尚未表示的细节。
>
> 码本构建可被概念化为在训练语音表示集合 `\mathcal{Z}_{train}` 上最小化连续表示与量化表示之间的误差：
>
> ```text
> \min_C \sum_{z ∈ \mathcal{Z}_{train}}
> \left\| z - \sum_{m=1}^{M} c_{m,i_m(z)} \right\|_2^2
> ```
>
> 该公式用于说明码本学习的目标，即用有限数量的 codeword 近似连续语音表示空间。论文正文中需要注意表述边界：本文关注的是将已获得的分层码本引入低负载通信框架，而不是主张提出一种全新的码本训练算法。
>
> 在线通信时，发送端并不传输码本本身，而是利用本地码本对连续表示 `Z` 进行逐层残差量化。对于第 `t` 个潜在时间步，首先令：
>
> ```text
> r_{0,t} = z_t
> ```
>
> 第 `m` 层 RVQ 从码本 `C_m` 中选择与当前残差最接近的 codeword：
>
> ```text
> i_{m,t} = \arg\min_{k ∈ {1,...,K}} \left\| r_{m-1,t} - c_{m,k} \right\|_2^2
> ```
>
> 由此得到该层的量化向量：
>
> ```text
> q_{m,t} = c_{m,i_{m,t}}
> ```
>
> 并更新残差：
>
> ```text
> r_{m,t} = r_{m-1,t} - q_{m,t}
> ```
>
> 经过 `L` 层量化后，第 `t` 个连续表示可被近似为：
>
> ```text
> \tilde{z}^{(L)}_t = \sum_{m=1}^{L} q_{m,t}
> = \sum_{m=1}^{L} c_{m,i_{m,t}}
> ```
>
> 因此，每个潜在时间步会在每一层 RVQ 中产生一个索引 `i_{m,t}`。将所有时间步和所有 RVQ 层的索引排列起来，即得到离散索引矩阵：
>
> ```text
> I = {i_{m,t}} ∈ {1, ..., K}^{M × T_q}
> ```
>
> 在实际传输前，系统也可以只保留前 `L` 层索引：
>
> ```text
> I_L = I_{1:L, :},    1 ≤ L ≤ M
> ```
>
> 这样，发送端最终输出的不是波形、连续潜在表示或码本向量，而是一组指向共享码本条目的离散索引。由于接收端保存了完全相同的码本，这些索引在接收端可以被重新解释为对应的 codeword，从而为后续潜在表示恢复和语音重建提供基础。

English example:

> The shared codebook is the key component that enables index-level speech communication in the proposed framework. Unlike features that are dynamically transmitted with each utterance, the codebooks are obtained before system deployment through an offline codebook learning process and remain fixed during online communication. The codebooks themselves are not transmitted through the channel; instead, they serve as a pre-shared discrete speech representation dictionary available at both the transmitter and the receiver.
>
> Let the shared codebook set be:
>
> ```text
> C = {C_1, C_2, ..., C_M}
> ```
>
> where the `m`-th codebook is defined as:
>
> ```text
> C_m = {c_{m,1}, c_{m,2}, ..., c_{m,K}},    c_{m,k} ∈ R^d
> ```
>
> Here, `M` denotes the number of RVQ layers, `K` is the number of entries in each codebook, and `c_{m,k}` is the `k`-th codeword in the `m`-th layer. In the current implementation, `M = 3` and `K = 1024` by default. From a communication perspective, the codebooks can be interpreted as a pre-shared discrete speech representation dictionary available at both ends.
>
> As illustrated in Figure 4, offline codebook construction starts from training representation samples or residual samples at the current RVQ layer. For the `m`-th layer, let the training sample set be:
>
> ```text
> R_m = {r_t^{(m)}},    r_t^{(1)} = z_t
> ```
>
> K-means clustering is first applied to `R_m` to obtain an initial codebook:
>
> ```text
> C_m^{(0)} = KMeans(R_m, K)
> ```
>
> This initial codebook is not the final shared codebook. It provides the starting point for iterative refinement. At iteration `n`, each training sample is assigned to its nearest codeword:
>
> ```text
> i_t^{(n)} = \arg\min_{k ∈ {1,...,K}}
> \left\| r_t^{(m,n)} - c_{m,k}^{(n)} \right\|_2^2
> ```
>
> The `k`-th codeword is then updated by the mean of the samples assigned to it:
>
> ```text
> c_{m,k}^{(n+1)}
> = mean\{r_t^{(m,n)}: i_t^{(n)} = k\}
> ```
>
> Nearest-codeword assignment and codeword update are repeated until the `m`-th layer codebook converges to `C_m^*`. After convergence, the remaining residual is computed as:
>
> ```text
> r_t^{(m+1)} = r_t^{(m)} - c_{m,i_t}^{*}
> ```
>
> and the resulting residual samples are used to learn the next RVQ layer. Therefore, the multi-layer RVQ codebook is not obtained as a single monolithic dictionary; it is constructed layer by layer through initialization, iterative update, and residual recursion. The final fixed shared codebook set is denoted as `C^* = {C_1^*, C_2^*, ..., C_M^*}`.
>
> In the manuscript, Figure 4 can be used to emphasize that K-means provides the initial cluster centers, iterative refinement adapts the codewords to the training representation space, and residual recursion allows later codebooks to capture details not represented by earlier layers.
>
> Conceptually, codebook construction can be described as minimizing the approximation error between continuous latent representations and their quantized counterparts over a training representation set `\mathcal{Z}_{train}`:
>
> ```text
> \min_C \sum_{z ∈ \mathcal{Z}_{train}}
> \left\| z - \sum_{m=1}^{M} c_{m,i_m(z)} \right\|_2^2
> ```
>
> This objective illustrates the role of the codebooks: representing a continuous speech representation space using a finite set of codewords. The manuscript should keep this claim carefully bounded. The contribution of this paper is to incorporate such hierarchical codebooks into a low-load communication framework, not to claim a new codebook training algorithm.
>
> During online communication, the transmitter does not send the codebooks themselves. Instead, it uses the local copy of the codebooks to quantize the continuous representation `Z` layer by layer. For the `t`-th latent time step, the initial residual is:
>
> ```text
> r_{0,t} = z_t
> ```
>
> The `m`-th RVQ layer selects the nearest codeword from codebook `C_m`:
>
> ```text
> i_{m,t} = \arg\min_{k ∈ {1,...,K}} \left\| r_{m-1,t} - c_{m,k} \right\|_2^2
> ```
>
> The selected codeword gives the quantized vector of that layer:
>
> ```text
> q_{m,t} = c_{m,i_{m,t}}
> ```
>
> and the residual is updated as:
>
> ```text
> r_{m,t} = r_{m-1,t} - q_{m,t}
> ```
>
> After `L` quantization layers, the `t`-th continuous representation can be approximated by:
>
> ```text
> \tilde{z}^{(L)}_t = \sum_{m=1}^{L} q_{m,t}
> = \sum_{m=1}^{L} c_{m,i_{m,t}}
> ```
>
> Thus, each latent time step produces one codebook index at each RVQ layer. Collecting the indices across all time steps and all quantization layers yields the discrete index matrix:
>
> ```text
> I = {i_{m,t}} ∈ {1, ..., K}^{M × T_q}
> ```
>
> Before transmission, the system may retain only the first `L` RVQ layers:
>
> ```text
> I_L = I_{1:L, :},    1 ≤ L ≤ M
> ```
>
> Consequently, the transmitter outputs neither the waveform, nor the continuous latent representation, nor the codebook vectors. It outputs a set of discrete indices pointing to entries in the shared codebooks. Since the receiver holds the same codebooks, these indices can be interpreted locally as codewords, providing the basis for latent representation recovery and speech reconstruction.

### 10.4 5.4.4 Index-Only Channel Transmission

本小节用于说明共享码本量化之后，离散索引如何作为通信信道中的唯一传输对象。它的重点不是计算完整 bitrate 或压缩比，而是定义信道中“传什么”和“如何传”。完整通信负载建模应放在 5.5。

建议覆盖：

- 信道中只传输离散码本索引，不传输原始波形、连续潜在表示或码本向量。
- 将索引矩阵 `I_L` 按时间步组织为索引流。
- 将索引流映射为比特流。
- 通过抽象信道模型得到接收端比特流和接收索引。
- 说明信道错误会体现为索引错误，为后续鲁棒性分析和接收端重建埋下接口。
- 不在本节展开完整负载公式、压缩率对比和实验结果。

建议公式：

```text
I_L = I_{1:L, :} ∈ {1, ..., K}^{L × T_q}
```

```text
s_t = [i_{1,t}, i_{2,t}, ..., i_{L,t}]
```

```text
S_I = [s_1, s_2, ..., s_{T_q}]
```

```text
b = \mathcal{B}(S_I)
```

```text
\hat{b} = \mathcal{H}(b),    \hat{I}_L = \mathcal{B}^{-1}(\hat{b})
```

中文示例表述：

> 经过共享码本量化后，发送端得到待传输的离散索引矩阵：
>
> ```text
> I_L = I_{1:L, :} ∈ {1, ..., K}^{L × T_q}
> ```
>
> 其中，`L` 表示实际保留并传输的 RVQ 层数，`T_q` 表示连续潜在表示的时间步数。与波形级语音通信或连续特征传输不同，本文框架中的通信信道仅承载离散码本索引。原始语音波形、连续潜在表示以及码本向量本身均不随单段语音通过信道传输。
>
> 为适配语音信号的时间顺序传输，本文将每个潜在时间步的前 `L` 层索引组织为一个索引组：
>
> ```text
> s_t = [i_{1,t}, i_{2,t}, ..., i_{L,t}]
> ```
>
> 所有时间步的索引组按时间顺序排列，形成待传输的索引流：
>
> ```text
> S_I = [s_1, s_2, ..., s_{T_q}]
> ```
>
> 随后，索引流通过比特映射函数转换为信道输入比特流：
>
> ```text
> b = \mathcal{B}(S_I)
> ```
>
> 其中，`\mathcal{B}(·)` 表示索引到比特流的映射过程。例如，当每层码本大小为 `K = 1024` 时，每个索引在理想定长编码下可由 `\lceil \log_2 K \rceil = 10` bits 表示。这里的描述仅用于说明索引到比特的映射关系，具体传输负载将在 5.5 中统一建模。
>
> 通信信道可抽象为映射 `\mathcal{H}(·)`。发送端比特流 `b` 经过信道后得到接收端比特流 `\hat{b}`，再通过解映射恢复接收索引矩阵：
>
> ```text
> \hat{b} = \mathcal{H}(b),    \hat{I}_L = \mathcal{B}^{-1}(\hat{b})
> ```
>
> 其中，`\hat{I}_L` 表示接收端获得的索引矩阵。在理想信道中，`\hat{I}_L = I_L`；在存在信道噪声或比特错误时，`\hat{I}_L` 可能包含索引错误。由于每个索引都对应共享码本中的一个离散 codeword，信道错误会表现为错误的码本地址选择，并进一步影响接收端的潜在表示恢复。
>
> 通过上述 index-only 传输机制，语音通信中的信道承载对象由高维连续语音信号转换为紧凑的离散索引流。该设计为后续通信负载建模、信道鲁棒性分析和接收端语音重建提供了统一接口。

English example:

> After shared-codebook quantization, the transmitter obtains the discrete index matrix to be transmitted:
>
> ```text
> I_L = I_{1:L, :} ∈ {1, ..., K}^{L × T_q}
> ```
>
> where `L` denotes the number of RVQ layers retained for transmission, and `T_q` is the number of latent time steps. Unlike waveform-level speech transmission or continuous-feature transmission, the communication channel in the proposed framework carries only discrete codebook indices. The raw speech waveform, continuous latent representations, and codebook vectors themselves are not transmitted with each utterance.
>
> To preserve the temporal order of speech, the indices at each latent time step are grouped as:
>
> ```text
> s_t = [i_{1,t}, i_{2,t}, ..., i_{L,t}]
> ```
>
> The index groups across all latent time steps are then arranged into an index stream:
>
> ```text
> S_I = [s_1, s_2, ..., s_{T_q}]
> ```
>
> The index stream is mapped into a channel input bitstream through a bit-mapping function:
>
> ```text
> b = \mathcal{B}(S_I)
> ```
>
> where `\mathcal{B}(·)` denotes the mapping from discrete indices to bits. For example, when the codebook size of each layer is `K = 1024`, each index can be represented by `\lceil \log_2 K \rceil = 10` bits under ideal fixed-length coding. This statement only describes the index-to-bit mapping; the full transmission-load formulation is provided in Section 5.5.
>
> The communication channel is represented by an abstract mapping `\mathcal{H}(·)`. The transmitted bitstream `b` is transformed into the received bitstream `\hat{b}`, which is then demapped into the received index matrix:
>
> ```text
> \hat{b} = \mathcal{H}(b),    \hat{I}_L = \mathcal{B}^{-1}(\hat{b})
> ```
>
> Here, `\hat{I}_L` denotes the index matrix available at the receiver. Under an ideal channel, `\hat{I}_L = I_L`. Under noisy or error-prone channel conditions, `\hat{I}_L` may contain index errors. Since each index points to a codeword in the shared codebooks, channel errors appear as incorrect codebook-address selections and may affect latent representation recovery at the receiver.
>
> Through this index-only transmission mechanism, the channel payload of speech communication is transformed from high-dimensional continuous speech signals into a compact discrete index stream. This design provides a unified interface for subsequent communication-load formulation, channel robustness analysis, and receiver-side speech reconstruction.

### 10.5 5.4.5 Receiver-Side Reconstruction

本小节用于说明接收端如何根据接收到的索引矩阵和本地共享码本恢复潜在表示，并进一步重建语音波形。它应与 5.4.2 形成方法结构上的对称：5.4.2 说明 `x -> Z`，5.4.5 说明 `\hat{I}_L -> \hat{Z} -> \hat{x}`。需要注意的是，5.4.5 不是 5.4.2 的简单反向过程，因为接收端输入不是连续表示 `Z`，而是经过信道后的索引矩阵 `\hat{I}_L`。

建议覆盖：

- 接收端输入为接收索引矩阵 `\hat{I}_L`。
- 接收端使用本地固定共享码本 `C^*` 进行 codeword lookup。
- 将前 `L` 层 codeword 相加，恢复近似潜在表示 `\hat{Z}`。
- 将 `\hat{Z}` 输入语音重建解码器，生成重建语音 `\hat{x}(t)`。
- 说明理想信道和非理想信道下的区别。
- 强调目标是通信可用语音，而不是逐采样点波形完全一致。
- 配 Figure 7：Receiver-Side Codebook Lookup and Speech Reconstruction，与 Figure 3 在视觉结构上对称。

建议公式：

```text
\hat{I}_L = {\hat{i}_{m,t}} ∈ {1, ..., K}^{L × T_q}
```

```text
\hat{q}_{m,t} = c^*_{m,\hat{i}_{m,t}}
```

```text
\hat{z}_t = \sum_{m=1}^{L} \hat{q}_{m,t}
= \sum_{m=1}^{L} c^*_{m,\hat{i}_{m,t}}
```

```text
\hat{Z} = [\hat{z}_1, \hat{z}_2, ..., \hat{z}_{T_q}]
```

```text
\hat{x} = D_\psi(\hat{Z})
```

中文示例表述：

> 在接收端，系统首先获得由信道输出和比特解映射得到的接收索引矩阵：
>
> ```text
> \hat{I}_L = {\hat{i}_{m,t}} ∈ {1, ..., K}^{L × T_q}
> ```
>
> 其中，`L` 为接收端用于重建的 RVQ 层数，`T_q` 为潜在表示的时间步数。在理想信道条件下，`\hat{I}_L = I_L`；当信道存在比特错误或索引错误时，`\hat{I}_L` 可能与发送端索引矩阵不同。
>
> 接收端保存与发送端一致的固定共享码本 `C^* = {C^*_1, C^*_2, ..., C^*_M}`。对于第 `t` 个潜在时间步和第 `m` 个 RVQ 层，接收端根据索引 `\hat{i}_{m,t}` 在本地码本中查找对应 codeword：
>
> ```text
> \hat{q}_{m,t} = c^*_{m,\hat{i}_{m,t}}
> ```
>
> 随后，接收端将前 `L` 层查找到的 codeword 相加，得到第 `t` 个时间步的恢复潜在表示：
>
> ```text
> \hat{z}_t = \sum_{m=1}^{L} \hat{q}_{m,t}
> = \sum_{m=1}^{L} c^*_{m,\hat{i}_{m,t}}
> ```
>
> 对所有时间步执行上述查表与求和操作，即可得到恢复后的潜在表示序列：
>
> ```text
> \hat{Z} = [\hat{z}_1, \hat{z}_2, ..., \hat{z}_{T_q}]
> ```
>
> 最后，语音重建解码器 `D_\psi(·)` 将恢复后的潜在表示映射回语音波形：
>
> ```text
> \hat{x} = D_\psi(\hat{Z})
> ```
>
> 该过程与发送端连续表示编码在功能上形成对称关系：发送端将语音波形编码为连续潜在表示，接收端则根据接收索引和共享码本恢复潜在表示，并进一步生成语音波形。不同之处在于，接收端并不直接接收原始连续表示 `Z`，而是通过共享码本查表从索引中恢复近似表示 `\hat{Z}`。
>
> 本文的接收端重建目标不是保证逐采样点波形完全一致，而是在较低传输负载下恢复通信可用的语音。具体而言，重建语音应尽可能保持语音可懂度、语义内容、说话人相关线索和情感信息。若信道错误导致部分索引被错误解码，则对应的 codeword 查找也会发生偏移，从而可能影响恢复潜在表示和最终语音质量。这一问题将在后续鲁棒性分析中进一步讨论。

English example:

> At the receiver, the system first obtains the received index matrix after channel output and bitstream demapping:
>
> ```text
> \hat{I}_L = {\hat{i}_{m,t}} ∈ {1, ..., K}^{L × T_q}
> ```
>
> where `L` denotes the number of RVQ layers used for reconstruction, and `T_q` is the number of latent time steps. Under an ideal channel, `\hat{I}_L = I_L`; when bit or index errors occur, the received index matrix may differ from the transmitted one.
>
> The receiver stores the same fixed shared codebooks as the transmitter, denoted as `C^* = {C^*_1, C^*_2, ..., C^*_M}`. For the `t`-th latent time step and the `m`-th RVQ layer, the receiver retrieves the corresponding codeword from the local codebook according to the received index `\hat{i}_{m,t}`:
>
> ```text
> \hat{q}_{m,t} = c^*_{m,\hat{i}_{m,t}}
> ```
>
> The codewords retrieved from the first `L` RVQ layers are then summed to recover the latent representation at time step `t`:
>
> ```text
> \hat{z}_t = \sum_{m=1}^{L} \hat{q}_{m,t}
> = \sum_{m=1}^{L} c^*_{m,\hat{i}_{m,t}}
> ```
>
> Applying this lookup-and-summation process to all latent time steps yields the recovered latent sequence:
>
> ```text
> \hat{Z} = [\hat{z}_1, \hat{z}_2, ..., \hat{z}_{T_q}]
> ```
>
> Finally, the speech reconstruction decoder `D_\psi(·)` maps the recovered latent representation back to a speech waveform:
>
> ```text
> \hat{x} = D_\psi(\hat{Z})
> ```
>
> This process is functionally symmetric to transmitter-side continuous representation encoding. The transmitter maps the speech waveform into a continuous latent representation, whereas the receiver reconstructs a latent representation from received indices and shared codebooks before synthesizing the waveform. The difference is that the receiver does not obtain the original continuous representation `Z` directly; it recovers an approximation `\hat{Z}` through codebook lookup.
>
> The objective of receiver-side reconstruction is not to guarantee sample-level waveform identity, but to recover communication-usable speech under reduced transmission load. In particular, the reconstructed speech should preserve intelligibility, semantic content, speaker-related cues, and emotional information as much as possible. If channel errors cause some indices to be decoded incorrectly, the corresponding codeword lookup will also be shifted, which may affect both the recovered latent representation and the final speech quality. This issue is further examined in the robustness analysis.
