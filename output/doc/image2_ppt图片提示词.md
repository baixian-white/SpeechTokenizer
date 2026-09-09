# image2 PPT 图片提示词：语音语义通信实验汇报

本文档用于逐页生成汇报 PPT 图片。建议每次只复制一个页面的提示词到 image2，生成 16:9 横版信息图。

## 全局风格要求

- 画面比例：16:9，横版 PPT 主视觉，推荐 1920x1080 或 3840x2160。
- 视觉风格：中文科研汇报、高级学术信息图、统一白底或极浅灰底、清晰网格、轻量阴影、矢量图质感，整体克制，不做科幻海报。
- 主色建议：深蓝 `#1f3b57`、青蓝 `#18a6a6`、绿色 `#49a66a`、琥珀色 `#f2a93b`、中性灰 `#7b8491`。
- 页面文字：中文为主，只保留必要术语，如 `SpeechTokenizer`、`RVQ`、`STOI`、`PESQ`、`bitpack10`、`payload`、`checkpoint`、`CPU`。
- 图表文字必须清楚、端正、不要艺术字。数字要准确，不要虚构新指标。
- 禁止风格：不要真实照片、不要人物、不要卡通、不要复杂 3D、不要大面积深色背景、不要赛博朋克、不要夸张渐变、不要强烈霓虹光效、不要堆叠太多装饰图标。
- 如果 image2 对中文文字渲染不稳定，优先保持图形结构、数字位置和留白，后续可在 PPT 中手动覆盖文字。

通用负面提示词：

```text
不要生成真实人物、不要生成实验室照片、不要生成随机英文段落、不要生成错误拼写、不要生成无关图标、不要把文字挤在一起、不要让图表标签重叠、不要使用低清晰度、小字糊成一团、不要使用花哨装饰背景、不要出现水印、不要出现品牌 logo。
```

---

## 背景页 1：中国 6G 背景 / 从比特通信走向语义通信

建议标题：

```text
中国 6G 背景：从“传更多比特”走向“传递语义”
```

建议副标题：

```text
第六代通信的重要方向，是把通信目标从波形与比特传输推进到语义理解、任务协同与智能重建
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 背景信息图，主题是“中国 6G 背景：从比特通信走向语义通信”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的背景页，不要做深色科幻海报，不要浮夸光效，不要宣传海报风格。

画面采用从左到右的演进结构，左侧是传统通信范式，中间是 6G 智能网络过渡，右侧是语义通信范式。三段之间用细线箭头连接，逻辑清楚。

左侧标题：1G-5G: bit-centric communication
画出基站、移动终端、宽带数据流和二进制 bit 流。
标注：
waveform / bits
rate, latency, reliability
focus on signal delivery
视觉上用浅灰和深蓝表示传统通信，数据流可以较粗，表示主要目标是承载更多比特。

中间标题：中国 6G research context
画一个克制的中国轮廓或中国区域网络示意，不要使用国旗、官方徽标或真实机构 logo。
在中国轮廓上画少量 6G 节点、卫星、边缘智能节点和 AI 计算节点，用细线连接。
标注：
6G
AI-native network
integrated sensing and communication
space-air-ground integrated network
注意文字要少，不要堆叠太多 buzzword。

右侧标题：semantic communication
画出语音 waveform 被送入语义编码模块，输出 compact semantic indices / tokens，再在接收端重建语音或完成理解任务。
流程可以写成：
speech waveform
→ semantic encoder
→ compact semantic indices
→ semantic decoder / task understanding
右侧用青蓝和绿色表示语义表征，不要画成自然语言词典，不要画成聊天机器人。

画面中心放一个强调箭头：
from bit transmission to meaning-oriented transmission

画面底部做一个三格对比条：
传统通信：尽可能完整传输信号或比特
语义通信：传递足以理解和重建的关键信息
本项目切入点：用 SpeechTokenizer + RVQ indices 传输语音语义索引

底部结论条：
6G 语义通信关注的不是“传得更多”，而是在有限链路资源下传递对理解、重建和任务完成最有用的信息。

视觉风格：
高端中文科研汇报风格，白底或极浅灰背景，深蓝主色，青蓝表示 6G 网络，绿色表示语义索引，少量琥珀色强调“语义”。线条清晰、留白充足、像论文背景图。不要真实照片，不要人物，不要复杂 3D，不要赛博朋克，不要大面积深色宇宙背景。

注意事项：
不要把语义通信画成简单的文字聊天或大语言模型界面。
不要宣称 6G 已经商用落地。
不要出现真实运营商、机构或国家标志 logo。
不要把“中国 6G”画成宣传口号页；要保持学术汇报、技术路线背景的克制风格。
```

---

## 背景页 2：量子信道窄带宽 / 安全链路中的在线载荷约束

建议标题：

```text
量子/安全信道窄带宽：语音通信必须压缩在线载荷
```

建议副标题：

```text
当安全链路可承载的数据极其有限时，直接传波形不现实，传输紧凑语义索引更匹配链路约束
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 背景信息图，主题是“量子/安全信道窄带宽：语音通信的在线载荷约束”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的问题背景页，不要做深色科幻海报，不要夸张量子光效，不要把量子信道画成高速科幻通道。

画面采用左右对比 + 中间瓶颈结构：
左侧是传统语音波形或 PCM 载荷，很粗的数据流；
中间是极窄的 quantum / secure narrowband channel 瓶颈；
右侧是可以通过瓶颈的小型语义索引包。

左侧标题：Conventional speech payload
画出较大的语音 waveform、PCM bytes、大数据包和粗箭头。
标注：
waveform stream
high-rate payload
too large for narrow secure link
用浅灰和少量红色表示“载荷过大”，可以让粗箭头在窄通道前被压缩或阻塞，但不要画得夸张。

中间标题：Quantum / secure narrowband channel
画一条非常窄的安全链路，可以用细线、锁形小图标、少量量子点或 photon dots 表示。
标注：
bandwidth is precious
small packets only
strict online payload budget
链路视觉要克制，像工程示意图，不要宇宙星云、强霓虹、复杂粒子特效。

右侧标题：Semantic index transmission
画出小型数据包顺利通过窄带链路，包内显示：
RVQ indices
[i1, i2, i3]
metadata
kbps-level stream
右侧再画接收端查表重建：
indices → local codebook lookup → SpeechTokenizer decoder → reconstructed speech

画面上方放一条清晰的问题陈述：
The bottleneck is not only computation, but online transport capacity.

画面下方做一个对比条：
不适合：直接发送 waveform / continuous latent / model parameters
适合：发送 compact RVQ indices + minimal metadata
前提：checkpoint 和 codebook 在两端预部署

底部结论条：
量子/安全窄带场景倒逼语音通信改变传输对象：从高码率波形流，转向低载荷、可查表重建的语义索引流。

视觉风格：
高端中文科研汇报风格，白底或极浅灰背景，深蓝文字，青蓝表示安全链路，绿色表示可通过的小索引包，琥珀色强调窄带瓶颈。画面要像“问题动机”页，清楚、克制、有技术感。不要真实照片，不要人物，不要复杂 3D，不要赛博朋克，不要大面积深色背景。

注意事项：
不要宣称已经完成真实量子通信链路实验。
不要说 RVQ indices 天然加密；安全性来自外部安全链路或加密机制，本页关注低载荷传输对象。
不要把 checkpoint、codebook 或模型参数画进每个在线 packet。
不要虚构具体量子信道带宽数值，除非后续 PPT 中另有真实实验数据。
不要使用军事化、商业广告或科幻战争视觉。
```

---

## 第 1 页：整体通信框架设计 / 预部署 tokenizer + 在线索引传输

建议标题：

```text
整体通信框架：预部署 SpeechTokenizer，在线只传 RVQ 离散索引
```

建议副标题：

```text
发送端编码成索引，窄带安全链路传输索引，接收端本地查表解码重建语音
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 信息图，主题是“整体通信框架设计：预部署 SpeechTokenizer，在线只传 RVQ 离散索引”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的系统架构图，不要做深色科幻海报，不要浮夸光效。

画面采用从左到右的通信链路框架，分为三大区域：发送端、极窄带宽安全链路、接收端。三大区域之间用细线箭头连接，逻辑清晰。

左侧区域标题：发送端 Sender
左侧画出流程：
麦克风 / 输入语音 waveform
SpeechTokenizer Encoder
RVQ Quantizer
输出紧凑离散索引 indices
在 RVQ Quantizer 旁边画出 3 层小型 codebook 网格，每层只高亮一个 index，表示发送的是 code id，不是连续向量。
左侧小注释：语音波形在本地被编码为 RVQ 离散索引。

中间区域标题：Online Secure / Narrowband Link
中间画一条很窄的传输通道，可以是细光纤、安全网关或窄带链路瓶颈。
链路中只流过很小的数据包，包内显示：
[i1, i2, i3] + metadata
旁边标注：
online payload only
RVQ indices
packet metadata
不要画大波形穿过链路，要明确波形不在线传输。

右侧区域标题：接收端 Receiver
右侧画出流程：
接收 indices
共享 codebook 查表
SpeechTokenizer Decoder
重建语音 waveform
扬声器 / 输出语音
右侧小注释：接收端使用本地预部署的同一 tokenizer checkpoint 重建语音。

画面顶部或底部增加一个横向虚线框，表示离线预部署阶段 Offline Provisioning：
同一个 SpeechTokenizer checkpoint
同一个 config
同一组 RVQ codebooks
箭头分别指向发送端和接收端，说明模型参数、配置和码本提前部署在两端，不作为每次在线传输载荷。

画面底部放一句核心结论：
核心改变：在线阶段不传语音波形，不传连续 latent，不传模型参数，只传可重建语音的离散索引。

视觉风格：
高端科研论文/IEEE/Nature 风格，白底或极浅灰背景，深蓝和青蓝色细线，少量红色只用于强调窄带瓶颈。画面干净、专业、有层次，像系统架构图，不要卡通，不要杂乱，不要大面积发光，不要赛博朋克。顶部左侧预留标题空间，底部预留一句总结空间。文字尽量少，所有文字必须简短清晰。

注意事项：
不要写“没有模型成本”或“没有码本成本”；要体现 checkpoint 和 codebook 是离线预部署成本。
不要画成传统语音压缩 codec 流程；重点是改变在线传输对象，从 waveform/latent 改为 RVQ indices。
不要出现真实品牌 logo、人物、过度科幻背景。
```

---

## 第 2 页：RVQ 索引构建过程 / 连续 latent 到分层离散索引

建议标题：

```text
RVQ 索引构建：连续语音表征逐层量化为离散 code id
```

建议副标题：

```text
第 1 层量化主体表征，后续层量化残差；最终发送的是每层 codebook 的索引编号
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 信息图，主题是“RVQ 索引构建过程：连续 latent 到分层离散索引”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的算法流程图，不要做深色科幻海报，不要浮夸光效。

画面采用从左到右的算法流程：
输入语音 waveform → SpeechTokenizer Encoder → 连续 latent z → RVQ 分层量化 → 输出 indices [i1, i2, i3]

左侧模块标题：1. 连续语音表征
画出输入语音 waveform，被送入 SpeechTokenizer Encoder。
Encoder 输出一个连续 latent z，用一条连续的高维向量色带或矩阵表示，标注：
z ∈ R^(D × T)
continuous latent
旁边小注释：直接传连续 latent 载荷很高，因此需要离散化。

中间主模块标题：2. Residual Vector Quantization
用三层纵向阶梯或三行流程表示 3 层 RVQ，每层都有一个 codebook 小网格和一个被选中的高亮 code。

第 1 层：
输入 z
查找 Codebook 1 中最近的 code 向量
输出索引 i1
得到量化向量 q1
计算残差 r1 = z - q1
视觉上用青蓝色表示第 1 层。

第 2 层：
输入残差 r1
查找 Codebook 2 中最近的 code 向量
输出索引 i2
得到量化向量 q2
计算残差 r2 = r1 - q2
视觉上用绿色表示第 2 层。

第 3 层：
输入残差 r2
查找 Codebook 3 中最近的 code 向量
输出索引 i3
得到量化向量 q3
最终重构 latent 近似：
z_hat = q1 + q2 + q3
视觉上用琥珀色表示第 3 层。

每一层 codebook 旁边标注：
codebook size = 1024
index = 10 bit
只高亮一个格子，强调发送的是索引编号，而不是整条 code 向量。

右侧模块标题：3. 在线发送的索引序列
画一个很小的数据包或表格：
time step t1: i1, i2, i3
time step t2: i1, i2, i3
time step t3: i1, i2, i3
旁边标注：
RVQ indices only
no waveform
no continuous latent
no codebook in packet

右下角画接收端查表解码的小流程：
indices → lookup q1,q2,q3 → z_hat → decoder → reconstructed speech

底部结论条：
RVQ 的关键是“逐层量化残差”：多发送一层 residual index，就多补充一部分语音细节，质量提升但 payload 增加很小。

视觉风格：
白底或极浅灰背景，深蓝文字，青蓝/绿色/琥珀色区分三层 RVQ。线条清楚、模块少、逻辑强，像论文方法图。不要卡通，不要复杂 3D，不要大面积发光。

注意事项：
不要把 RVQ 三层写成确定的“语义层、音色层、情感层”，避免过度解释。
不要说 codebook 在线生成；codebook 是训练后随 checkpoint 预部署的共享查表空间。
不要说完全无损压缩；这里是离散量化近似。
不要展示 STOI、PESQ、payload 柱状图，本页只解释索引构建机制。
```

---

## 第 3 页：单层量化机制 / 最近邻查表生成 code id

建议标题：

```text
单层量化机制：在 codebook 中寻找最近的向量并输出索引
```

建议副标题：

```text
每个时间步的连续向量被替换为最近的 code 向量；发送端只发送该 code 的编号
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 信息图，主题是“单层 RVQ/VQ 量化机制：最近邻查表生成 code id”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的算法细节图，不要做深色科幻海报，不要浮夸光效。

画面聚焦一个单独时间步的量化过程，采用从左到右的 5 步流程：
1. 输入连续向量
2. 与 codebook 中所有 code 向量计算距离
3. 选择最近的 code
4. 输出 code id
5. 用 code 向量近似原向量，并把残差传给下一层

左侧模块标题：1. 输入向量
画出一个连续 latent 或 residual 向量 x_t，用一列或一条高维向量表示。
标注：
x_t ∈ R^1024
来自 encoder latent 或上一层 residual
旁边小注释：这是单个时间步的连续表征。

中间左侧模块标题：2. Codebook 查找
画一个 codebook 矩阵或网格：
Codebook = 1024 codes × 1024 dims
行编号从 0 到 1023，表示 code id。
每一行是一条 code 向量 e_k。
用细线从 x_t 指向多个 code 行，表示计算距离。

中间主模块标题：3. 最近邻选择
画出距离计算公式：
i = argmin_k || x_t - e_k ||²
或用简洁文字：
choose nearest code vector
在 codebook 网格中高亮一行，例如 code id = 512。
旁边画一个小放大框：
selected index: i = 512
index cost: 10 bit

中间右侧模块标题：4. 量化替代
画出：
x_t → e_512
说明连续向量 x_t 被最近的 code 向量 e_512 近似。
标注：
q_t = e_i
only i is transmitted
code vector stays in local codebook

右侧模块标题：5. 残差传递
画出残差公式：
r_t = x_t - q_t
残差 r_t 用细箭头传给下一层 quantizer。
旁边说明：
下一层继续量化剩余误差
多层 RVQ = 逐层修正近似误差

底部放一个小的“发送内容 vs 本地共享内容”对比条：
在线发送：code id i
本地预部署：codebook vectors e_0 ... e_1023
不发送：x_t、e_i、完整 codebook、模型参数

底部结论条：
单层量化的本质是最近邻查表：用 10 bit 的 code id 指向本地共享 codebook 中的一个 1024 维向量。

视觉风格：
白底或极浅灰背景，深蓝文字，青蓝色高亮被选中的 code 行，绿色表示残差传递，少量琥珀色强调 10 bit index。线条简洁、模块清楚、像论文算法图。不要卡通，不要真实照片，不要复杂 3D。

注意事项：
不要画成分类器输出语义标签；code id 不是文字语义标签，而是 codebook 向量编号。
不要说发送端发送 code 向量；发送端只发送 index。
不要说量化完全无误差；必须表现出 residual/error 传给下一层。
不要出现 payload 实验结果、STOI、PESQ，本页只讲单层量化原理。
```

---

## 第 4 页：码本构建过程 / 训练阶段学习共享 codebook

建议标题：

```text
码本构建：从训练语音 latent 分布中学习共享 codebook
```

建议副标题：

```text
codebook 在训练阶段通过初始化、最近邻分配和 EMA 更新形成，在线阶段只用于查表
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 信息图，主题是“码本构建过程：训练阶段学习共享 codebook”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的训练机制流程图，不要做深色科幻海报，不要浮夸光效。

画面分成左右两个阶段：
左侧 70% 表示离线训练阶段 codebook 如何学习；
右侧 25% 表示训练完成后 codebook 如何预部署并用于在线通信；
底部 5% 放一句结论边界。

左侧主标题：Offline Training: learn codebook from speech latent

左侧流程从左到右：
训练语音数据
SpeechTokenizer Encoder
大量连续 latent 向量集合
K-Means 初始化 codebook
最近邻分配
EMA 更新 code 向量
低使用 code 替换
训练完成后的 codebook

第 1 个模块：训练语音数据
画出多个语音 waveform 小条，标注：
LibriSpeech training speech
audio segments

第 2 个模块：Encoder latent 分布
画出许多散点或高维向量云，标注：
encoder latent samples
z_t ∈ R^1024
旁边小注释：codebook 学习的是训练语音 latent 的分布。

第 3 个模块：K-Means 初始化
画出从 latent 云中选出多个中心点，形成初始 codebook。
标注：
initial code vectors
K = 1024
每层 RVQ 一个独立 codebook

第 4 个模块：最近邻分配
画出 latent 点被分配到最近的 code 中心，使用不同浅色小簇。
标注：
assign each latent to nearest code
i = argmin_k ||z_t - e_k||²

第 5 个模块：EMA 更新
画出 code 向量中心缓慢移动到所属样本均值附近，使用箭头表示更新。
标注：
EMA update code vectors
cluster usage statistics

第 6 个模块：低使用 code 替换
画出少数灰色 inactive code 被当前 batch 中的新样本替换。
标注：
replace low-usage / dead codes
avoid unused codebook entries

第 7 个模块：训练完成后的 codebook
画出一个整齐矩阵：
Codebook layer k
1024 code vectors
each vector has 1024 dimensions
高亮几行 code，表示训练后成为稳定查表空间。

右侧主标题：Deployment: shared lookup table
画出同一个 checkpoint 被复制到发送端和接收端：
SpeechTokenizer checkpoint
config
RVQ codebooks
箭头分别指向 Sender 和 Receiver。
再画在线阶段小流程：
Sender: latent → code id
Link: send index only
Receiver: code id → lookup code vector

右侧强调框：
codebook is pre-deployed
not transmitted per packet
index is meaningful only if both sides share the same checkpoint

底部结论条：
码本不是人工语义标签表，也不是在线生成的；它是训练阶段从语音 latent 分布中学习出的共享向量表。

视觉风格：
白底或极浅灰背景，深蓝文字，训练流程用青蓝和绿色，低使用 code 替换用灰色到琥珀色的变化，部署阶段用深蓝虚线框。整体清晰克制，像论文方法图。不要卡通，不要真实照片，不要复杂 3D。

注意事项：
不要把 codebook 画成文字词典或语义标签词表；它是向量表。
不要说 codebook 是人工构造或规则生成。
不要说在线阶段重新训练或重新构建 codebook。
不要说没有 codebook 成本；要体现 codebook 随 checkpoint 预部署。
不要展示 payload、STOI、PESQ，本页只解释 codebook 构建与部署。
```

---

## 第 5 页：理论在线载荷推导 / 为什么 3 层 RVQ 可以降到 kbps 量级

建议标题：

```text
理论在线载荷：50 steps/s × 3 层 × 10 bit ≈ 1500 bps
```

建议副标题：

```text
SpeechTokenizer 将 16 kHz 语音下采样为 50 个时间步/秒，每步每层只需一个 10 bit code id
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 信息图，主题是“理论在线载荷推导：为什么 3 层 RVQ 可以降到 kbps 量级”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的公式推导页和简洁数据图，不要做深色科幻海报，不要浮夸光效。

画面中心用一条从左到右的推导链路表示：
16 kHz waveform → stride product 320 → 50 token steps/s → 3 RVQ layers → 1024 codes/layer → 10 bit/index → 1500 bps body payload

左侧画一个简洁的语音波形和采样率标注：
sample rate = 16000 Hz
PCM baseline = 16000 × 16 = 256000 bps
用浅灰色大条表示传统 PCM 16-bit mono 的在线 body payload。

中间画一个下采样模块：
SpeechTokenizer encoder stride = 8 × 5 × 4 × 2 = 320
token rate = 16000 / 320 = 50 steps/s
用一个时间轴显示每秒只有 50 个 token step。

右侧画 RVQ 索引计数：
每个 token step 有 3 层索引：i1, i2, i3
每层 codebook size = 1024
每个 index = log2(1024) = 10 bit
用三个小方块表示三层索引，每个方块写 10 bit。

画面右下角给出主公式：
Online body payload
= 50 steps/s × 3 layers × 10 bit
= 1500 bps

在公式下方放一个克制的对比条：
PCM body: 256000 bps
3-layer RVQ theoretical body: ≈1500 bps
标注：about 170× smaller body payload

底部结论条：
低载荷来自三个因素：时间步变少、每层只传索引、码本在两端预部署。

视觉风格：
白底或极浅灰背景，深蓝文字，青蓝色表示 token rate，绿色表示 RVQ layers，琥珀色强调 10 bit/index。公式要清楚，数字要准确，排版要像论文中的方法推导图。不要卡通，不要复杂 3D，不要强烈渐变。

注意事项：
不要使用用户已否定的成本边界类表述。
不要说总系统成本消失；这里只推导在线 body payload。
不要把 1500 bps 画成最终所有场景的总传输量；它是理论 body payload，真实报文还包含 metadata/header。
不要加入 1 层、2 层质量比较，本页只解释 3 层主配置的理论载荷来源。
```

---

## 第 6 页：实验验证总路线 / 从机制到可用性的四类验证

建议标题：

```text
实验验证路线：载荷、质量、打包正确性与实时可行性
```

建议副标题：

```text
先证明“能低载荷传”，再证明“能重建、能打包、能实时跑”
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 信息图，主题是“实验验证总路线：从机制到可用性的四类验证”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的实验路线图，不要做深色科幻海报，不要浮夸光效。

画面采用横向路线图，从左到右 5 个节点：
1. 固定模型与数据
2. 在线载荷统计
3. 3 层重建质量
4. 索引序列化回环
5. CPU 实时性与部署接口

第 1 个节点：固定模型与数据
画 checkpoint、config、test set 三个小图标。
标注：
checkpoint: SpeechTokenizer_best_dev.pt
config: n_q = 3
sample rate: 16 kHz
test eval: 200 utterances

第 2 个节点：在线载荷统计
画窄带链路和小数据包。
标注：
PCM vs RVQ indices
bitpack10
body payload + metadata

第 3 个节点：3 层重建质量
画输入波形与重建波形的上下对齐图。
标注：
STOI
PESQ
Mel error
SI-SNR
只突出 3-layer RVQ main setting。

第 4 个节点：索引序列化回环
画 indices → pack → bytes → unpack → indices 的闭环。
标注：
roundtrip all OK
49 B / 0.25s chunk
pack/unpack ms-level small

第 5 个节点：CPU 实时性与部署接口
画发送端、链路、接收端和 CPU 小图标。
标注：
0.25s chunks
RTF < 1
loopback prototype
security-link ready interface

画面底部做一个“验证目标”四格条：
Payload reduction
Reconstruction quality
Lossless packet serialization
Realtime feasibility

底部结论条：
本项目不是只展示一个算法图，而是把“索引传输”放到可计量、可打包、可实时运行的通信流程中验证。

视觉风格：
白底或极浅灰背景，深蓝主线，青蓝色节点，少量绿色表示验证通过。整体像严谨的实验流程图，模块间距均匀，文字少而清楚。不要卡通，不要真实照片，不要复杂 3D。

注意事项：
不要出现成本边界类表述。
不要暗示已经完成真实量子链路传输；这里只是窄带/安全链路接口和通信负载验证。
不要把 NAS 放到主结论前面；NAS 只是后续部署优化探索。
```

---

## 第 7 页：实验设置与模型口径 / 汇报中的结果从哪里来

建议标题：

```text
实验设置：固定 n_q=3 SpeechTokenizer 与 0.25s 低延迟 chunk
```

建议副标题：

```text
所有正式结果基于同一 checkpoint、同一 config、同一测试集口径
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 信息图，主题是“实验设置与模型口径：汇报中的结果从哪里来”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的实验设置页，不要做深色科幻海报，不要浮夸光效。

画面采用三栏布局：模型口径、数据与切分、评估输出。

左栏标题：Model Scope
画 checkpoint 文件、config 文件、RVQ 层数三个简洁图标。
列出：
SpeechTokenizer checkpoint
Log/spt_base/SpeechTokenizer_best_dev.pt
config: Log/spt_base/config.json
local n_q = 3
sample rate = 16 kHz
stride product = 320
token rate = 50 steps/s

中栏标题：Data & Chunking
画测试集清单和时间切片示意图。
列出：
test_eval_200
200 utterances
0.25s low-latency chunks
10,248 chunks for payload/timing
每个 chunk 被切成短时索引包。

右栏标题：Measured Outputs
画四个小卡片式结果入口，但不要做成浮夸卡片。
卡片 1：payload statistics
body bps / metadata bps / total bps
卡片 2：quality metrics
STOI / PESQ / Mel error / SI-SNR
卡片 3：serialization
bitpack10 / roundtrip
卡片 4：runtime
encode ms / decode ms / RTF

画面右下角加一个小的“结果口径”提示框：
CPU timing only
GPU timing not used in conclusion
public n_q=8 model not used in formal result

底部结论条：
统一模型、统一数据和统一 chunk 口径，是后续载荷、质量与实时性结果可对齐比较的前提。

视觉风格：
白底或极浅灰背景，深蓝文字，青蓝色用于模型，绿色用于数据，琥珀色用于评估输出。排版规整，像论文实验设置图。不要卡通，不要复杂 3D，不要加入无关服务器照片。

注意事项：
不要写公开 n_q=8 模型结果。
不要写 GPU 速度结论。
不要新增未知数据集或训练集规模。
不要把 200 utterances 和 10,248 chunks 混淆：200 是质量评估语音条数，10,248 是 chunk 统计规模。
```

---

## 第 8 页：在线载荷实验结果 / 3 层 RVQ 的真实报文负载

建议标题：

```text
在线载荷结果：3 层 RVQ bitpack10 总载荷 4531.62 bps
```

建议副标题：

```text
0.25s 低延迟 chunk 下，body payload 为 1657.08 bps，metadata/header 占比较高
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 数据图，主题是“在线载荷实验结果：3 层 RVQ 的真实报文负载”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的对比柱状图和指标图，不要做深色科幻海报，不要浮夸光效。

画面左侧 60% 做横向对比柱状图，比较 3 个对象：
PCM 16-bit mono total online payload: 258398.10 bps
3-layer RVQ bitpack10 total online payload: 4531.62 bps
3-layer RVQ bitpack10 body payload: 1657.08 bps

柱状图要求：
PCM 用浅灰色很长柱。
RVQ total 用青蓝色短柱。
RVQ body 用绿色更短柱。
在每根柱末尾标出准确数值。
图注写：
0.25s chunk, n = 10,248 chunks

右侧 40% 做一个报文拆分示意：
3-layer RVQ packet
body indices: 1657.08 bps
metadata/header: 2874.53 bps
total: 4531.62 bps
用堆叠条显示 body 与 metadata 两部分，强调低延迟短包下 header 占比较明显。

右下角放一个小公式框：
body ≈ 3 layers × 50 steps/s × 10 bit
measured body close to theoretical 1500 bps

底部结论条：
实测证明在线传输对象已经从 256 kbps 级 PCM 波形转为 kbps 级 RVQ 索引包；0.25s 低延迟设置下 metadata 是主要额外开销。

视觉风格：
白底或极浅灰背景，深蓝标题，青蓝和绿色数据柱，浅灰作为基线。图表清楚，数字准确，留白充足。不要卡通，不要复杂 3D，不要使用对数坐标造成误读，除非明确标注。

注意事项：
不要说 4531.62 bps 是纯索引 body；它是含 metadata/header 的 total online payload。
不要说 1657.08 bps 是所有报文总量；它是 body payload。
不要加入 1 层、2 层质量比较。
不要出现成本边界类表述。
```

---

## 第 9 页：短包头部开销问题 / 为什么 0.25s total 高于理论 body

建议标题：

```text
短包头部开销：0.25s 低延迟模式下 metadata 占比明显
```

建议副标题：

```text
索引本体已经很小，固定格式头部在短包中会被放大
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 信息图，主题是“短包头部开销：为什么 0.25s total 高于理论 body”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的问题解释图，不要做深色科幻海报，不要浮夸光效。

画面采用左右对照结构：
左侧标题：Index Body is Tiny
右侧标题：Header Becomes Visible

左侧画 0.25s chunk 的索引本体：
3-layer RVQ indices
bitpack10
body: 1657.08 bps
可以画成很短的绿色数据块，里面有 i1/i2/i3 的小格。

右侧画同一个 0.25s packet 的 metadata/header：
shape
sample rate
chunk length
format tag
packet fields
metadata/header: 2874.53 bps
画成青蓝色或琥珀色头部块，占比大于 body。

中间画一个堆叠数据包：
HEADER + BODY
total: 4531.62 bps
header share ≈ 63.4%
标注：
low-latency packetization
many small packets per second

底部放一个小推理流程：
chunk shorter → packets more frequent → fixed header repeated more often → total bps increases

画面右下角放下一页过渡提示：
Can packetization reduce overhead?
longer aggregation / full utterance

底部结论条：
0.25s 低延迟设置展示的是保守在线 total payload；瓶颈不在索引本体，而在短包重复 metadata。

视觉风格：
白底或极浅灰背景，深蓝文字，绿色表示 index body，琥珀色表示 header overhead，整体清楚克制。不要卡通，不要复杂 3D，不要夸张警示图。

注意事项：
不要说 header 是错误或实验失败；它是短包实时传输中的格式开销。
不要把 header overhead 混成模型参数或 codebook 成本。
不要新增未知协议名称。
不要使用成本边界类措辞。
```

---

## 第 10 页：Packetization 补充实验 / 聚合后接近理论载荷

建议标题：

```text
Packetization 补充：聚合包可把 total payload 拉近 1500 bps
```

建议副标题：

```text
full utterance 模式下，3 层 RVQ body 1501.26 bps，JSON total 1569.47 bps
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 数据图，主题是“Packetization 补充实验：聚合后接近理论载荷”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的补充实验图，不要做深色科幻海报，不要浮夸光效。

画面左侧做一张简洁折线或阶梯图，横轴为 packetization mode：
0.25s chunk
0.5s chunk
1.0s chunk
full utterance
纵轴为 3-layer JSON total payload bps。
标出关键数值：
0.25s chunk: 4531.62 bps
1.0s chunk: 2844.30 bps
full utterance: 1569.47 bps
趋势线从左到右下降，颜色用青蓝。

画面右侧做 3-layer full utterance 的重点结果框：
body payload: 1501.26 bps
JSON total payload: 1569.47 bps
STOI: 0.8807
PESQ: 1.965
SI-SNR: 0.03 dB
Mel error: 0.843
在数值旁边画一个小的“header share decreases”堆叠条。

中间或底部画一个小机制示意：
many short packets → repeated header
aggregated packet → header amortized

底部结论条：
当 packetization 从低延迟短包转向聚合包，metadata 被摊薄，3 层 RVQ 的 total payload 接近理论 1500 bps。

视觉风格：
白底或极浅灰背景，深蓝文字，青蓝折线，绿色强调 full utterance 重点结果。图表要干净、数字要清晰，像论文补充实验结果页。不要卡通，不要复杂 3D。

注意事项：
不要把 full utterance 说成实时低延迟设置；它是带宽优化或离线聚合参考。
不要说 1569.47 bps 是所有低延迟场景都能达到。
不要做 1/2/3 层比较；只展示 3 层主配置。
不要新增不存在的质量指标。
```

---

## 第 11 页：3 层 RVQ 重建质量 / 主配置质量结果

建议标题：

```text
3 层 RVQ 重建质量：在 kbps 级载荷下保持可理解语音
```

建议副标题：

```text
0.25s 低延迟 chunk：STOI 0.8608，PESQ 1.627，Mel error 0.920，SI-SNR -0.63 dB
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 数据图，主题是“3 层 RVQ 重建质量：主配置质量结果”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的质量评估页，不要做深色科幻海报，不要浮夸光效。

重要要求：本页只展示 3 层 RVQ 在 0.25s 低延迟 chunk 设置下的主配置结果，不展示 1 层和 2 层对比，不展示 full utterance / 整句聚合结果。

画面上半部分做输入语音与重建语音的视觉对齐：
上方：原始语音 waveform / Mel spectrogram
下方：重建语音 waveform / Mel spectrogram
用两条简洁波形和两条热力图条带表示，不需要真实音频细节，但要体现上下对齐和相似结构。
中间用箭头标注：
3 层 RVQ 索引 → 解码器重建

画面下半部分左侧做一个主指标块，标题为：
0.25s 低延迟 chunk 重建质量
STOI = 0.8608
PESQ = 1.627
Mel error = 0.920
SI-SNR = -0.63 dB
旁边加小注释：
3 层 RVQ
0.25s chunk
低延迟正式设置

画面右下角预留一个清晰的试听对比框，标题为：
现场试听对比
框内画两个音频按钮或波形占位条：
原始语音
重建语音
在两个占位条旁边留出播放图标位置，方便后续在 PPT 中手动放入前后语音链接或音频按钮。
试听框只做占位，不要虚构主观评分，不要写 MOS。

画面下方可以放一个小公式/说明框：
质量结果与 0.25s 低延迟在线载荷对应
3 层 RVQ bitpack10 total payload = 4.53 kbps

底部结论条：
3 层 RVQ 在 0.25s 低延迟设置下，将在线载荷压到 kbps 量级，同时仍能重建可理解的语音内容；听感对比将在现场播放原始语音与重建语音。

视觉风格：
白底或极浅灰背景，深蓝文字，青蓝表示 original，绿色表示 reconstructed，琥珀色突出 STOI/PESQ。图表要专业、简洁，像论文结果图。不要卡通，不要复杂 3D，不要真人照片。

注意事项：
不要展示 1 层、2 层重建质量。
不要称为高保真音频 codec；PESQ 数值要如实呈现。
不要出现 full utterance、整句聚合、1.57 kbps、STOI 0.8807 或 PESQ 1.965；本页只讲 0.25s 低延迟结果。
不要虚构主观听感实验或 MOS。
不要把右下试听框画成已经完成的正式听感实验；它只是现场播放原始语音和重建语音的占位区域。
```

---

## 第 12 页：3 层 RVQ 主结论 / 载荷与质量放在同一张图里

建议标题：

```text
主结论：3 层 RVQ 在低载荷与可重建质量之间取得工作点
```

建议副标题：

```text
0.25s 用于低延迟在线验证，full utterance 用于观察头部摊薄后的载荷上限
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 综合结果图，主题是“3 层 RVQ 主结论：载荷与质量放在同一张图里”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的关键结论页，不要做深色科幻海报，不要浮夸光效。

画面中心做一个二维坐标图：
横轴：Online total payload bps，方向从左到右增大
纵轴：Reconstruction quality，使用 STOI 或 PESQ 标注

图中放三个清楚的点：
PCM baseline
total payload: 258398.10 bps
作为右侧灰色基线点，不需要质量值。

3-layer RVQ 0.25s chunk
total payload: 4531.62 bps
STOI: 0.8608
PESQ: 1.627
用青蓝色点表示，标注 low latency。

3-layer RVQ full utterance
total payload: 1569.47 bps
STOI: 0.8807
PESQ: 1.965
用绿色点表示，标注 aggregated packetization。

坐标图旁边放一个简洁信息框：
Main setting: 3 RVQ layers
index: 10 bit
token rate: 50 steps/s
pre-deployed checkpoint and codebooks

图下方做一个两段式解释：
Low-latency mode: real-time oriented, metadata overhead visible
Aggregated mode: bandwidth optimized, total payload near theoretical body

底部结论条：
本项目的核心不是追求传统音频 codec 的最高音质，而是在极低在线载荷下完成语音语义索引传输与可理解重建。

视觉风格：
白底或极浅灰背景，坐标轴细线，深蓝标题，青蓝和绿色重点点位，灰色表示 PCM baseline。画面要像论文核心结果图，数字准确，标注不要重叠。不要卡通，不要复杂 3D。

注意事项：
不要展示 1/2/3 层比较。
不要说 full utterance 是实时低延迟结果。
不要把 PCM 质量指标虚构出来。
不要说该方法已经超过所有传统 codec；本页只说明本项目工作点。
```

---

## 第 13 页：索引序列化与无损回环 / bitpack10 打包验证

建议标题：

```text
索引序列化：bitpack10 将 RVQ code id 无损打包成字节流
```

建议副标题：

```text
3 层 0.25s chunk 平均 49 B，pack 0.032 ms，unpack 0.030 ms，roundtrip 全部通过
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 信息图，主题是“索引序列化与无损回环：bitpack10 打包验证”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的工程验证页，不要做深色科幻海报，不要浮夸光效。

画面中央做一个闭环流程：
RVQ indices
→ bitpack10 pack
→ byte stream
→ bitpack10 unpack
→ recovered indices
→ compare with original
最后回到 check mark：roundtrip all OK

每个模块用清楚的矢量框表示：
indices: [i1, i2, i3], each 0...1023
pack: 10-bit fields
bytes: compact payload
unpack: restore integer code ids
compare: exact equality

画面右侧放 3 层主配置的指标表：
scheme: bitpack10
RVQ layers: 3
chunks: 10,248
body bytes/chunk: 49 B
pack mean: 0.032 ms
unpack mean: 0.030 ms
roundtrip: all OK

画面左下角画一个小的 bit-level 示意：
10 bit index | 10 bit index | 10 bit index | ...
跨字节对齐，不浪费成 16 bit。
旁边标注：
1024 codes → 10 bits per index

底部结论条：
bitpack10 证明 RVQ 索引可以被稳定、紧凑、无损地转成在线传输字节流，工程开销很小。

视觉风格：
白底或极浅灰背景，深蓝文字，青蓝表示 pack/unpack，绿色表示 roundtrip OK，琥珀色强调 10-bit packing。整体像系统工程验证图，不要卡通，不要复杂 3D。

注意事项：
不要说语音重建无损；无损的是索引序列化回环，不是音频量化本身。
不要把 49 B 写成每秒字节数；它是 0.25s chunk 的 3 层 bitpack10 body bytes。
不要新增加密、纠错或压缩算法。
```

---

## 第 14 页：实时通信原型设计 / 0.25s chunk loopback 流程

建议标题：

```text
实时通信原型：0.25s chunk 的 encode-pack-transmit-unpack-decode 闭环
```

建议副标题：

```text
把算法结果放进接近在线通信的流式处理链路中验证
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 系统图，主题是“实时通信原型设计：0.25s chunk loopback 流程”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的实时系统架构图，不要做深色科幻海报，不要浮夸光效。

画面采用从左到右的流式处理管线，分成 6 个模块：
1. Audio input buffer
2. 0.25s chunking
3. SpeechTokenizer encode + RVQ indices
4. bitpack10 serialization
5. loopback / network interface
6. unpack + decode + audio output

模块 1：
画麦克风或音频缓冲条，不画真人。
标注：16 kHz mono speech

模块 2：
画连续波形被切成 0.25s 小块。
标注：0.25s chunks
low-latency setting

模块 3：
画 encoder 和 RVQ 三层小网格。
标注：3-layer RVQ indices

模块 4：
画 pack 成字节流的小数据包。
标注：bitpack10 bytes

模块 5：
画一个中性网络接口框，可以是 loopback 或 narrowband link。
标注：
loopback test
replaceable by secure link

模块 6：
画接收端 unpack、lookup、decoder、speaker。
标注：reconstructed speech

画面上方用一条时间轴表示：
chunk arrives → encode → packet → decode → output

画面右下角放一个小的目标框：
Realtime target: compute time < chunk duration
chunk duration = 250 ms
RTF < 1 means feasible

底部结论条：
原型验证关注的是通信流程能否按 chunk 连续运行，而不只是离线计算一次重建结果。

视觉风格：
白底或极浅灰背景，深蓝线条，青蓝表示发送端，绿色表示接收端，琥珀色表示网络接口。整体克制、规整，像系统原型图。不要卡通，不要复杂 3D，不要真实电话或人物照片。

注意事项：
不要宣称真实公网、真实量子链路或真实加密设备已经完成端到端部署。
不要把 loopback 画成最终产品。
不要写 GPU 实时结果。
```

---

## 第 15 页：CPU 实时性结果 / 计算时间低于 chunk 时长

建议标题：

```text
CPU 实时性：3 层 RVQ encode+decode 平均 64.57 ms/chunk
```

建议副标题：

```text
0.25s chunk 下平均 RTF = 0.273，CPU 计算代理满足实时处理条件
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 数据图，主题是“CPU 实时性结果：计算时间低于 chunk 时长”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的性能评估页，不要做深色科幻海报，不要浮夸光效。

画面左侧做一个时间预算条：
chunk duration = 250 ms
在 250 ms 长条内部画出 3-layer compute time:
mean encode+decode = 64.57 ms
P95 = 68.70 ms
剩余部分标注：headroom
用绿色显示计算时间远小于 250 ms。

画面中间做一个分解条或小表：
Encode mean: 44.05 ms
Decode mean: 20.52 ms
End-to-end compute mean: 64.57 ms
End-to-end P95: 68.70 ms
RTF mean: 0.273
chunks: 10,248

画面右侧做一个速度仪表或 RTF 指标图：
RTF < 1 = realtime feasible
Measured RTF = 0.273
指针位于安全区域，颜色克制。

底部结论条：
在 CPU 上，3 层 RVQ 的编码与解码计算时间低于 0.25s chunk 时长，为低延迟在线通信原型提供可行性支撑。

视觉风格：
白底或极浅灰背景，深蓝文字，绿色表示 realtime feasible，青蓝表示计算时间，浅灰表示 chunk duration。数字要准确、图表简洁。不要卡通，不要复杂 3D。

注意事项：
不要写 GPU 速度。
不要说已经完成端到端网络延迟评估；这里只是 CPU compute-time proxy / loopback feasibility。
不要说所有硬件都能实时运行。
不要展示 1/2 层对比，主结果聚焦 3 层。
```

---

## 第 16 页：双端通信与安全链路接口 / 从 loopback 到真实链路

建议标题：

```text
双端接口：RVQ 索引包可以接入极窄带宽安全链路
```

建议副标题：

```text
发送端与接收端共享 checkpoint，链路只承载小型索引包和必要 metadata
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 系统接口图，主题是“双端通信与安全链路接口：从 loopback 到真实链路”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的部署接口架构图，不要做深色科幻海报，不要浮夸光效。

画面分为三段：发送端设备、极窄带宽安全链路、接收端设备。

左侧发送端设备内部画四个模块：
Audio capture
SpeechTokenizer Encoder
RVQ index generation
Packetizer bitpack10
输出一个小数据包：
header + RVQ indices

中间链路画得很窄，类似安全通道或低带宽管道。
标题：
Narrowband / secure link interface
链路上只画少量小数据包，不画大波形。
包内字段标注：
timestamp
shape
format
bitpacked indices

右侧接收端设备内部画四个模块：
Depacketizer
RVQ codebook lookup
SpeechTokenizer Decoder
Audio playback

画面顶部画一个离线同步层：
Offline endpoint provisioning
same checkpoint
same config
same RVQ codebooks
虚线箭头分别连到发送端和接收端。

画面底部画一个接口边界框：
Replace loopback with physical secure link
Keep packet format unchanged
Only transport layer changes

底部结论条：
系统设计把“语音模型侧”和“安全链路侧”解耦：链路只需要传输标准化索引包，模型和码本在端侧预部署。

视觉风格：
白底或极浅灰背景，深蓝文字，青蓝表示发送端，绿色表示接收端，琥珀色表示安全链路。整体像工程架构图，干净克制。不要卡通，不要复杂 3D，不要真实设备照片。

注意事项：
不要宣称已经完成真实量子通信链路实验。
不要说不需要离线部署成本。
不要把 checkpoint 或 codebook 画进每个在线 packet。
不要出现无关军事或商业品牌元素。
```

---

## 第 17 页：面向安全窄带场景的应用设想 / 为什么这个方向有意义

建议标题：

```text
应用设想：面向安全窄带链路的语音语义索引传输
```

建议副标题：

```text
量子密钥分发、卫星安全链路、应急低带宽通道等场景，都需要极低在线载荷
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 场景信息图，主题是“面向安全窄带场景的语音语义索引传输应用设想”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的应用场景页，不要做深色科幻海报，不要浮夸光效。

画面中心放一个核心模块：
Speech semantic index packets
3-layer RVQ indices
kbps-level online payload

围绕中心放三个克制的应用场景分支：

分支 1：Quantum-key-protected secure link
画简洁的量子密钥图标、锁形图标和细链路。
文字：
secure channel bandwidth is precious
send compact speech indices

分支 2：Satellite / remote secure communication
画卫星到地面站的细线，不要科幻大场景。
文字：
remote link
low-rate packet stream

分支 3：Emergency / constrained network
画应急通信节点和窄带网关。
文字：
bandwidth constrained
speech still needs intelligibility

画面底部放一个“为什么适合”的三点总结：
1. 在线只传 RVQ indices
2. checkpoint/codebook 端侧预部署
3. packet format 可接入不同安全链路

右下角放一个小边界说明：
Current work: payload + reconstruction + serialization + realtime proxy
Future work: real physical secure-link validation

底部结论条：
该框架的价值在于把语音从高码率波形流，转化为可在安全窄带链路上传输的小型语义索引流。

视觉风格：
白底或极浅灰背景，深蓝文字，青蓝安全链路，绿色索引包，少量琥珀色用于窄带瓶颈。风格克制、学术、信息图化。不要深色宇宙海报，不要夸张卫星光束，不要复杂 3D。

注意事项：
不要宣称已经完成真实量子通信实验。
不要画成科幻战争通信或商业产品广告。
不要说索引天然加密；安全性来自外部安全链路或加密机制，本项目关注低载荷索引传输。
不要新增法律、军事或医疗场景结论。
```

---

## 补充页：NAS 神经架构搜索机制 / 如何自动寻找轻量 tokenizer

建议标题：

```text
NAS 神经架构搜索：自动寻找更轻的 SpeechTokenizer 结构
```

建议副标题：

```text
在给定搜索空间内反复采样、训练、评估和筛选候选结构，目标是降低端侧模型复杂度
```

image2 提示词：

```text
    制作一张 16:9 学术汇报 PPT 方法图，主题是“NAS 神经架构搜索：自动寻找更轻的 SpeechTokenizer 结构”。

    整体背景必须是白色或极浅灰色，正式科研汇报方法图风格，深蓝标题，绿色表示被选中的候选结构，青蓝表示语音重建评估，浅灰表示搜索空间和未选候选，琥珀色只用于提示计算约束或待验证边界。画面清晰、克制、信息密度适中，不要科幻风，不要真实人物，不要复杂 3D，不要大面积深色背景。

    画面顶部放一条清晰前提说明：
    NAS 只优化端侧 tokenizer 的结构复杂度；它不改变在线传输对象，在线阶段仍然传 RVQ 离散索引。

    画面主体采用从左到右的流程图，分成 5 个模块：

    第 1 模块：定义搜索空间
    画一个浅灰色结构积木库，标题为：
    搜索空间
    列出可搜索维度：
    n_filters: 24 / 32
    LSTM layers: 1 / 2
    compress: 2 / 4
    activation: ELU / Snake
    layer op: std_k3 / sep_k7 / dil_k9 / skip
    SE block: on / off

    第 2 模块：采样候选结构
    画一个搜索控制器或 Optuna/TPE 采样器，从搜索空间中抽取多个候选网络。
    标注：
    Optuna TPE sampler
    candidate architectures
    trial 1, trial 2, trial 3 ...

    第 3 模块：快速训练与重建评估
    画一个小型训练循环：
    candidate tokenizer -> encode/decode -> reconstructed waveform
    旁边放评估损失：
    loss = reconstruction loss + mel loss
    用青蓝色表示语音重建对比，不要画成人工标签分类。

    第 4 模块：计算约束筛选
    画一个琥珀色门控框：
    FLOPs gate
    超过 4G FLOPs 的候选直接淘汰
    保留计算量更低且 loss 更小的结构

    第 5 模块：导出最佳结构
    画一个绿色候选结构卡片，标题为：
    best_seanet_config.json
    列出当前导出的结构摘要：
    n_filters = 32
    lstm = 1
    compress = 4
    layer ops = sep_k7, dil_k9, skip, skip
    目标：作为端侧轻量化候选

    画面右侧放一个小边界说明框：
    当前 NAS 页展示的是结构搜索机制
    searched structure 仍需重新训练 checkpoint
    正式质量结论必须经过同一测试协议验证

    画面底部放一条结论条：
    NAS 的作用是把“人工设计 tokenizer”变成“自动搜索轻量候选结构”；它服务于端侧部署优化，但不直接改变 RVQ index-only online transmission 的载荷口径。

    视觉风格：
    白底或极浅灰背景，深蓝文字，流程箭头清楚，搜索空间用浅灰，候选评估用青蓝，最佳结构用绿色，FLOPs gate 和待验证边界用琥珀色。整体像论文方法补充图或答辩解释图，信息密度适中，数字和关键词清楚。不要卡通，不要真实照片，不要赛博朋克，不要把 NAS 画成魔法自动生成完整模型。

    注意事项：
    不要说 NAS 已经得到同等质量的最终模型，除非重训练质量评估完成。
    不要把 NAS 结果当作本文正式 payload / quality / realtime 主结论。
    不要暗示在线 payload 会因为 NAS 自动下降；在线 payload 主要由 token rate、RVQ 层数和 index bit 决定。
    不要把 NAS 搜索画成语义标签搜索、文本词典搜索或人工情绪分类。
    不要把 codebook、checkpoint 或模型参数画进在线数据包。
```

---

## 第 18 页：NAS 轻量化探索 / 端侧部署优化方向

建议标题：

```text
NAS 轻量化探索：把 tokenizer 做小，为端侧部署做准备
```

建议副标题：

```text
搜索候选结构参数量约 16.63M，但尚未完成重训练质量验证，因此只作为部署优化方向
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 信息图，主题是“NAS 轻量化探索：端侧部署优化方向”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的模型效率探索页，不要做深色科幻海报，不要浮夸光效。

画面左侧做 baseline 模型信息：
Baseline SpeechTokenizer
params: 103.68M
1s FLOPs: 17.05G
checkpoint: official project model used in main experiments
用较大的灰蓝色模型块表示。

画面右侧做 NAS searched candidate：
NAS searched structure
params: 16.63M
config: nas/best_seanet_config.json
33 search records
用较小的绿色模型块表示。

中间用箭头连接：
architecture search
deployment optimization
edge endpoint friendly

箭头下方放一个明确的边界提示框，文字要醒目但不夸张：
Quality not yet validated by retrained checkpoint
Not used as main communication result

画面底部放三步后续路线：
1. retrain NAS candidate
2. evaluate 3-layer RVQ reconstruction quality
3. compare runtime / payload / quality under same protocol

底部结论条：
NAS 的作用是端侧 tokenizer 轻量化，不改变本文主线：在线阶段传输 RVQ 索引。

视觉风格：
白底或极浅灰背景，深蓝文字，baseline 用灰蓝，NAS candidate 用绿色，边界提示用琥珀色。图形简洁，像论文未来工作或部署优化页。不要卡通，不要复杂 3D。

注意事项：
不要说 NAS 模型已经保持质量。
不要把 NAS 结果作为当前正式 payload/quality 结论。
不要暗示 payload 因 NAS 直接改变；payload 主要由 token rate、RVQ 层数和 index bit 决定。
不要把 NAS 放成第一贡献。
```

---

## 第 19 页：不足与思考 / 当前工作的边界

建议标题：

```text
不足与思考：从原型验证走向真实安全链路还需要补齐什么
```

建议副标题：

```text
当前结果说明方向可行，但仍需在链路、主观质量、鲁棒性和轻量化上继续推进
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 信息图，主题是“不足与思考：当前工作的边界与后续问题”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的讨论页，不要做深色科幻海报，不要浮夸光效。

画面采用四象限结构，每个象限一个问题与思考方向。

左上象限标题：1. 真实链路验证
图标：安全链路、卫星或量子密钥小图标。
内容：
当前主要是 payload、packet、loopback 验证
后续需要接入真实窄带/安全链路
观察丢包、时延、抖动与同步问题

右上象限标题：2. 语音质量评估
图标：波形、听感评分、小耳机图标。
内容：
已有 STOI / PESQ / Mel error / SI-SNR
后续可补充主观听感或任务级理解评估
避免只用单一客观指标判断可用性

左下象限标题：3. 鲁棒性与泛化
图标：噪声、不同说话人、不同场景的小图标。
内容：
需要测试噪声、口音、不同语料与信道扰动
研究 packet loss / bit error 下的恢复策略

右下象限标题：4. 端侧轻量化
图标：小芯片、压缩模型。
内容：
NAS candidate 尚需重训练与质量验证
端侧 CPU/嵌入式部署还需进一步优化

画面底部放一个中心总结框：
当前工作回答了“索引能否低载荷传并重建”
下一步要回答“真实链路中是否稳定、自然、可部署”

视觉风格：
白底或极浅灰背景，深蓝文字，四象限用细线分隔，青蓝/绿色/琥珀色小图标。整体诚实、克制、学术，不要沮丧或夸大。不要卡通，不要复杂 3D。

注意事项：
标题必须使用“不足与思考”。
不要写成失败总结；要写成研究边界和下一步。
不要宣称已经解决真实链路、主观听感、鲁棒性和 NAS 质量验证。
不要使用“limitations”作为主标题。
```

---

## 第 20 页：总结与贡献 / 汇报收束页

建议标题：

```text
总结：用 RVQ 离散索引实现低载荷语音语义通信原型
```

建议副标题：

```text
从预部署 tokenizer、在线索引传输，到载荷、质量、序列化和实时性验证
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 总结页，主题是“总结与贡献：用 RVQ 离散索引实现低载荷语音语义通信原型”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报的最后总结页，不要做深色科幻海报，不要浮夸光效。

画面中心放一个简洁的主流程图：
Speech waveform
→ SpeechTokenizer encoder
→ 3-layer RVQ indices
→ narrowband secure link
→ decoder reconstruction

流程图下方放四个贡献点，每个贡献点用一个简洁图标和一句短句：

贡献 1：通信框架
预部署 checkpoint/codebook，在线只传 RVQ indices

贡献 2：载荷验证
3-layer theoretical body ≈ 1500 bps
0.25s total = 4531.62 bps
full utterance total = 1569.47 bps

贡献 3：重建质量
3-layer STOI = 0.8608 / PESQ = 1.627
full utterance STOI = 0.8807 / PESQ = 1.965

贡献 4：工程可行性
bitpack10 roundtrip all OK
CPU RTF = 0.273
NAS lightweight candidate: 16.63M params

画面右下角放一句收束语：
From waveform transmission to semantic index transmission

底部结论条：
本项目展示了一条面向 6G 与安全窄带链路的语音语义通信路线：端侧共享模型，在线传索引，低载荷重建语音。

视觉风格：
白底或极浅灰背景，深蓝文字，青蓝表示通信框架，绿色表示验证通过，琥珀色表示后续优化。画面要干净、有总结感、适合汇报结束页。不要卡通，不要复杂 3D，不要大面积深色背景。

注意事项：
不要把 NAS 轻量化写成已经完成质量验证的正式贡献。
不要说没有模型或码本成本；要保持“预部署”表述。
不要宣称真实 6G/量子通信链路已经部署验证。
不要加入新的未经验证指标。
```

---

## 第 21 页：K-Means 初始化过程 / 从 encoder latent 到 codebook 初值

建议标题：

```text
K-Means 初始化：从语音 latent 分布生成 codebook 初始向量
```

建议副标题：

```text
训练语音先被编码为连续 latent 样本，再聚成 K=1024 个簇；每个簇中心作为 RVQ codebook 的初始 code vector
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 方法图，主题是“K-Means 初始化过程：从 encoder latent 到 codebook 初值”。

整体背景必须是白色或极浅灰色，画面像正式科研汇报中的算法流程图，干净、克制、逻辑清楚。不要做深色科幻海报，不要浮夸光效，不要真实人物或实验室照片。

画面采用从左到右的 5 步流程结构，每一步放在一个窄竖向模块中，用细线箭头连接：

步骤 1：训练语音输入
画出少量 speech waveform 或 audio segments，输入到 SpeechTokenizer Encoder。
标注：
training speech
SpeechTokenizer Encoder

步骤 2：收集 encoder latent 样本
画出一个二维投影的蓝色点云，表示大量高维 latent samples。
标注：
latent samples z_t
z_t ∈ R^1024
sample from training set
旁边用小注释说明：码本学习的是语音 latent 分布，不是人工文本标签。

步骤 3：K-Means 聚类
画出同一批点云被分成多个彩色簇，簇中心用星形或实心圆表示。
在模块中放出核心公式：
i_t = argmin_k ||z_t - c_k||²
c_k = mean({z_t | i_t = k})
标注：
nearest-center assignment
update cluster centers
K = 1024
注意公式要清晰、简洁，不要塞太多文字。

步骤 4：得到初始 code vectors
把 K-Means 的 cluster centers 复制到一个 codebook 表格中。
画一个矩阵/查找表：
E = {e_1, e_2, ..., e_K}
K = 1024 codes
each e_k ∈ R^1024
表格中每一行代表一个 code vector，强调“聚类中心成为 codebook 初值”。

步骤 5：RVQ 多层残差初始化
画出 3 层纵向小流程或阶梯结构：
Layer 1: K-Means on z
Layer 2: K-Means on residual r_2 = z - q_1
Layer 3: K-Means on residual r_3 = z - q_1 - q_2
每层旁边画一个小 codebook 表格，颜色分别用青蓝、绿色、琥珀色区分。
强调：每层 RVQ 有独立 codebook，后续层学习前面量化后剩余的 residual。

画面底部放一条结论条：
K-Means 初始化把 codebook 放到语音 latent / residual 分布的高密度区域，后续训练再通过最近邻分配、EMA 更新和低使用 code 替换继续细化。

视觉风格：
白底或极浅灰背景，深蓝标题文字，青蓝表示 latent 样本，绿色表示聚类中心和有效 code，琥珀色表示 residual 层。线条清晰，模块边框细，留白充足，像论文方法图或答辩 PPT。可以使用二维点云投影来表示高维 latent，但不要画成真实语义词云。

注意事项：
不要把 codebook 画成人工语义标签表，不要写“词典”“单词”“情绪标签”等文本语义标签。
不要说 codebook 在线生成；本页讲的是训练阶段的初始化。
不要把 codebook 画进在线 packet，不要暗示每次通信都传 codebook。
不要宣称 K-Means 后就训练完成；它只是初始化，后续还会继续 EMA 更新和替换低使用 code。
不要加入 STOI、PESQ、payload 或 CPU 实时性数据，本页只解释初始化机制。
```

---

## 第 22 页：K-Means 初始化之后 / 前向重建、反向传播与 EMA 更新闭环

建议标题：

```text
K-Means 初始化之后：在重建训练闭环中继续学习 codebook
```

建议副标题：

```text
初始化只给出 codebook 起点；正式训练通过前向量化、重建损失、反向传播和 EMA 统计更新逐步收敛
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 方法图，主题是“码本初始化之后：前向重建、反向传播与 EMA 更新闭环”。

整体背景为白色或极浅灰色，科研论文方法图风格，深蓝标题，青蓝/绿色/琥珀色作为强调色。画面要清晰、克制、流程感强，不要科幻海报，不要真实人物，不要复杂 3D，不要大面积深色背景。

画面主体采用一个闭环训练流程图，重点说明：K-Means 初始化 codebook 之后，码本会进入完整的 encoder-RVQ-decoder 训练循环，而不是直接训练完成。

顶部标题：
K-Means 初始化之后：在重建训练闭环中继续学习 codebook

副标题：
初始化只给出 codebook 起点；正式训练通过前向量化、重建损失、反向传播和 EMA 统计更新逐步收敛

画面左侧放一个小模块：
0. K-Means initialized codebook
画一个 codebook 表格，里面有若干 code vectors，标注：
initial code vectors
K = 1024
e_k ∈ R^1024
说明：来自 encoder latent / residual 聚类中心

从左到右画出一次训练 iteration 的前向传播：

1. Training speech x
画语音 waveform 输入。

2. Encoder
画 SpeechTokenizer Encoder 模块。
输出标注：
latent z_t ∈ R^1024

3. RVQ nearest-neighbor quantization
画多层 RVQ codebook lookup，突出每层只选中一个 code id。
标注公式：
i_t = argmin_k ||z_t - e_k||²
q_t = e_{i_t}
如果表现多层 RVQ，写：
z_hat = q_1 + q_2 + q_3 + ...
旁边标注：
forward pass uses discrete code vectors

4. Decoder reconstruction
画 SpeechTokenizer Decoder 模块。
输出重建语音 waveform：
reconstructed speech x_hat

5. Reconstruction loss
在原始 speech x 和 reconstructed speech x_hat 之间画对比框。
标注：
L_rec = distance(x, x_hat)
可以用 waveform / spectrogram loss 的小图标表示，但不要写具体未验证指标。
说明：
compare original and reconstructed speech

从 loss 模块画一条向后的反向传播箭头，回到 Encoder 和 Decoder：

6. Backpropagation
标注：
update encoder / decoder parameters
straight-through gradient for quantization
用虚线穿过 RVQ quantizer，表示离散 argmin 不可导，梯度近似传回 encoder。
注意：不要把普通梯度箭头直接画成更新 codebook 的主要方式。

在 RVQ quantization 模块下方单独画一个 EMA 更新分支：

7. EMA codebook update
从“nearest-neighbor assignments”引出箭头到 EMA 模块。
画两个统计量：
N_k ← ρN_k + (1-ρ)n_k
S_k ← ρS_k + (1-ρ)s_k
e_k ← S_k / N_k
旁边画 code vectors 平滑移动到 assigned latent cluster center 的示意。
标注：
smooth update from assigned latent statistics
stable codebook learning

8. Low-usage / dead code replacement
在 EMA 模块旁边画一个小分支：
usage statistics
low-usage code
replace with current batch latent / residual sample
用灰色 dead code 变成彩色 active code 的小示意。
说明：
avoid unused codebook entries

最后从 EMA 更新和 dead code replacement 画箭头回到 RVQ codebook，形成闭环：
updated codebook for next iteration

画面底部放一条结论条：
K-Means 只是 codebook 的初始位置；真正的共享码本是在训练闭环中，通过最近邻分配、重建损失反传、EMA 统计更新和低使用 code 替换逐步学习出来的。

视觉风格：
白底或极浅灰背景，深蓝文字，青蓝表示 encoder/decoder 前向路径，绿色表示 EMA 更新，琥珀色表示 loss/backprop，灰色表示 low-usage/dead code。流程箭头清晰，闭环结构明确，像论文方法图或答辩 PPT。文字尽量短，公式清楚，不要让标签重叠。

注意事项：
不要把 codebook 画成人工语义标签表。
不要说 K-Means 后 codebook 已经训练完成。
不要把 codebook 画进在线 packet。
不要暗示在线通信阶段会更新 codebook；本页只描述离线训练阶段。
不要写“无模型成本”或“无码本成本”。
不要加入 STOI、PESQ、payload、CPU RTF 等实验结果。
不要把 EMA 画成普通反向传播梯度直接更新 codebook；EMA 是基于最近邻分配统计的平滑更新。
```

---

## 第 23 页：码本维护细节 / EMA 更新与低使用 code 替换

建议标题：

```text
码本如何继续更新：EMA 平滑移动活跃 code，替换低使用 code
```

建议副标题：

```text
重建损失反传主要更新 encoder / decoder；codebook 由最近邻分配统计驱动更新，并通过 dead-code replacement 保持容量有效
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 方法图，主题是“码本维护细节：EMA 更新与低使用 code 替换”。

整体背景必须是白色或极浅灰色，正式科研汇报方法图风格，深蓝标题，绿色强调 EMA 更新，灰色表示低使用或 dead code，青蓝表示 latent 样本。画面清晰、克制、信息密度适中，不要科幻风，不要真实人物，不要复杂 3D，不要大面积深色背景。

画面主体分成左右两大栏，中间用细竖线分隔：

左栏标题：
7. EMA codebook update
副标题：
active codes move toward assigned latent statistics

左栏从上到下画 4 个小步骤：

7.1 最近邻分配结果
画一组青蓝色 latent 点云，几个绿色 code vector 点，箭头表示 latent 被分配给最近的 code。
标注：
from RVQ nearest-neighbor assignment
i_t = argmin_k ||z_t - e_k||²

7.2 统计当前 batch
对每个 code k 画一个小统计表：
n_k = number of assigned latents
s_k = sum of assigned latents
B_k = {z_t | i_t = k}
用颜色强调：n_k 是计数，s_k 是向量和。

7.3 EMA 平滑累计
画两个长期统计量的更新公式，公式必须清楚：
N_k ← ρN_k + (1-ρ)n_k
S_k ← ρS_k + (1-ρ)s_k
说明：
N_k: smoothed usage count
S_k: smoothed latent sum
ρ close to 1, e.g. 0.99

7.4 更新 code vector
画 code vector 从旧位置缓慢移动到分配 latent 的中心附近，不要画成瞬间跳跃。
公式：
e_k ← S_k / N_k
标注：
smooth movement, stable codebook learning
不要画成 loss gradient 直接更新 codebook。

右栏标题：
8. Low-usage / dead code replacement
副标题：
unused entries are reactivated near current data distribution

右栏从上到下画 4 个小步骤：

8.1 使用频率统计
画一个 code usage histogram，横轴 code id，纵轴 usage count。大多数柱子正常，有几个灰色柱子非常低。
标注：
usage statistics from assignments
low-usage threshold

8.2 识别低使用 / 死码
把低使用 code 画成灰色圆点或灰色表格行。
标注：
low-usage code
dead code
n_k ≈ 0 for many iterations

8.3 用当前 batch 样本替换
从当前 batch 的 latent / residual 点云中选几个彩色样本，箭头指向灰色 dead code。
标注：
replace with current batch latent / residual sample
move unused code back near data distribution

8.4 重新参与下一轮竞争
灰色 code 变成彩色 active code，重新进入 codebook 表格。
标注：
reactivated code
competes in next nearest-neighbor assignment

画面顶部加一个小型上下文条：
Input to both mechanisms: nearest-neighbor assignments from the current training batch
Output: updated codebook for the next batch

画面底部放一条结论条：
EMA 更新负责让活跃 code 稳定贴近 latent / residual 分布；低使用 code 替换负责避免码本容量被长期闲置条目浪费。

视觉风格：
白底或极浅灰背景，深蓝文字，青蓝 latent 点，绿色 active code 和 EMA 箭头，灰色 dead code，琥珀色只用于提示 threshold 或 warning。左右两栏对称，公式清晰，留白充足，像论文方法补充图或答辩解释图。

注意事项：
不要把 codebook 画成人工语义标签表，不要出现“单词”“情绪标签”“文本词典”等概念。
不要把 EMA 画成由重建损失梯度直接更新 codebook；EMA 来自最近邻分配统计。
不要说 dead code replacement 每次都大量替换；它是低使用情况下的维护机制。
不要暗示在线通信阶段会更新 codebook；本页只描述离线训练。
不要把 codebook 画进在线 packet。
不要加入 STOI、PESQ、payload、CPU RTF 等实验结果。
```

---

## 补充页：在线载荷对比 / 连续 latent 与 RVQ index 的差距

建议标题：

```text
在线载荷对比：连续 latent 很重，RVQ 索引才能进入 kbps 量级
```

建议副标题：

```text
在线阶段的关键不是传波形或连续向量，而是只传可查表重建的离散索引
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 数据图，主题是“在线载荷对比：连续 latent 与 RVQ 索引的差距”。

整体背景必须是白色或极浅灰色，正式科研汇报图风格，深蓝标题，青蓝表示连续 latent 表征，绿色表示 RVQ 索引，浅灰表示 PCM 参考基线，琥珀色只用于提示头部/元数据开销。画面清晰、克制、信息密度适中，不要科幻风，不要真实人物，不要复杂 3D，不要大面积深色背景。

画面顶部放一条清晰的前提说明：
通信两端预部署同一个 SpeechTokenizer 检查点；在线阶段只统计实际经过链路发送的数据。

画面主体采用单栏大图结构，去掉右侧机制解释框，让主图占据页面约 85% 宽度和 70% 高度。主图标题为：
同一评估协议下的在线载荷对比

画一个横向堆叠条形图，比较 3 个传输方案，全部基于 0.25s 低延迟设置。每一行都拆成“有效载荷”和“元数据/头部”两段，并在右侧列出总在线载荷。数值必须准确：

1. 连续 latent（float32）
有效载荷：1800743.75 bps
元数据/头部：2975.99 bps
总在线载荷：1803719.74 bps，标注 1.80 Mbps

2. PCM 16-bit 单声道参考
有效载荷：256000.00 bps
元数据/头部：2398.10 bps
总在线载荷：258398.10 bps，标注 258.40 kbps

3. RVQ 3 层 bitpack10 索引包
有效载荷：1657.08 bps
元数据/头部：2874.53 bps
总在线载荷：4531.62 bps，标注 4.53 kbps

条形图建议使用对数尺度或断轴设计，但必须明确标注：
载荷坐标轴为对数尺度
不要让短柱完全看不见。

视觉要求：
每个条形都由两段组成：有效载荷段 + 元数据/头部段。
连续 latent 的有效载荷段用青蓝色，元数据段用浅琥珀色。
PCM 的有效载荷段用浅灰色，元数据段用浅琥珀色，旁边写“参考基线”。
RVQ 索引的有效载荷段用深绿色，元数据段用琥珀色，整行用绿色边框强调。

在主图右侧增加一组清晰的大号压缩倍数标注，压缩倍数按“在线总载荷”计算：
RVQ 索引包 vs PCM：约 57× 更低
RVQ 索引包 vs 连续 latent：约 398× 更低

在 RVQ 行旁边加小注释：
在线只传索引和少量元数据
不传波形
不传连续 latent 表征
不把码本放进在线包

在图下方放一个小公式框：
总在线载荷 = 有效载荷 + 元数据/头部
RVQ 3 层索引主体 ≈ 50 步/秒 × 3 层 × 10 bit ≈ 1500 bps
0.25s 实测索引主体 = 1657.08 bps；0.25s 实测总载荷 = 4531.62 bps

画面底部放一条结论条：
实验证明，直接传连续 latent 比 PCM 还重；0.25s 低延迟设置下，RVQ 索引包的在线总载荷相对 PCM 约降低 57 倍，相对连续 latent 约降低 398 倍。

视觉风格：
白底或极浅灰背景，深蓝文字，青蓝表示连续 latent 表征，绿色表示 RVQ 索引，浅灰表示 PCM，琥珀色表示元数据。图表要像论文主结果图或答辩核心数据页，数字大而清楚，标签不拥挤，左右两栏对齐。不要卡通，不要夸张表情，不要真实照片，不要赛博朋克，不要把数据包画成在线模型文件。

注意事项：
不要把核心比较写成只有 PCM 和 RVQ；必须突出“连续 latent 与 RVQ 索引”的差距。
不要再画右侧机制解释框；本页重点是全宽载荷对比数据图。
不要只写总载荷；每个方案都必须同时显示有效载荷、元数据/头部和总在线载荷。
压缩倍数必须说明按“在线总载荷”计算。
不要说在线阶段会传码本、checkpoint 或模型参数；这些属于端侧预部署。
不要把 1500 bps 说成所有低延迟场景的总载荷；它是理论索引主体载荷。
不要把 4531.62 bps 说成纯索引主体载荷；它是 0.25s 低延迟 JSON 报文的在线总载荷。
不要加入整句分包、full utterance 或 1.57 kbps 结果；本页只展示 0.25s 低延迟正式设置。
不要加入 STOI、PESQ、CPU RTF 或 NAS 结果；本页只讲在线载荷对比。
不要把码本画成人工语义标签表、文本词典、单词表或情绪标签表。
```

---

## 补充页：载荷计算口径 / 有效载荷与解析头部如何得到

建议标题：

```text
载荷计算口径：有效载荷、解析头部与总在线载荷
```

建议副标题：

```text
有效载荷是实际数据主体；解析头部用于接收端恢复 payload 的类型、形状和长度
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 计算说明图，主题是“载荷计算口径：有效载荷、解析头部与总在线载荷”。

整体背景必须是白色或极浅灰色，正式科研汇报计算图风格，深蓝标题，青蓝表示连续 latent，浅灰表示 PCM，绿色表示 RVQ 索引，琥珀色表示解析头部。画面清晰、克制、信息密度适中，不要科幻风，不要真实人物，不要复杂 3D，不要大面积深色背景。

画面顶部放一条核心公式：
总在线载荷 = 有效载荷 + 解析头部
bps = 字节数 × 8 / 音频有效时长

在公式旁边放一个小注释：
解析头部用于接收端恢复 payload：kind、dtype、shape、body_len，并包含 4 B 长度前缀。

画面主体采用三栏并列结构，每栏解释一个传输方案，全部基于 0.25s 低延迟 chunk。三栏标题分别是：
1. PCM 16-bit 单声道
2. 连续 latent（float32）
3. RVQ 3 层 bitpack10 索引

左栏：PCM 16-bit 单声道
画一个短波形和字节块。
列出计算：
0.25s × 16000 Hz = 4000 samples
有效载荷 = 4000 × 2 B = 8000 B
典型有效载荷 bps = 8000 × 8 / 0.25 = 256000 bps
解析头部示意：
kind = pcm_s16le
dtype = int16
shape = [4000]
body_len = 8000 B
典型解析头部 = 67 B JSON + 4 B 长度前缀 = 71 B
典型解析头部 bps = 71 × 8 / 0.25 = 2272 bps
实验均值总在线载荷 = 258398.10 bps

中栏：连续 latent（float32）
画一个青蓝色 latent 张量矩阵。
列出计算：
encoder 输出 shape = [1, 1024, 13]
元素数 = 1 × 1024 × 13 = 13312
有效载荷 = 13312 × 4 B = 53248 B
典型有效载荷 bps = 53248 × 8 / 0.25 = 1703936 bps
解析头部示意：
kind = encoder_latent_f32
dtype = float32
shape = [1, 1024, 13]
body_len = 53248 B
典型解析头部 = 84 B JSON + 4 B 长度前缀 = 88 B
典型解析头部 bps = 88 × 8 / 0.25 = 2816 bps
实验均值总在线载荷 = 1803719.74 bps

右栏：RVQ 3 层 bitpack10 索引
画 3 层绿色索引序列和一个小数据包。
列出计算：
0.25s chunk 中约 13 个索引时间步
索引个数 = 3 层 × 13 步 = 39
未打包 bit 数 = 39 × 10 bit = 390 bit
bitpack10 有效载荷 = ceil(390 / 8) = 49 B
典型有效载荷 bps = 49 × 8 / 0.25 = 1568 bps
解析头部示意：
kind = codes_bitpack10
dtype = uint10_packed
shape = [3, 1, 13]
body_len = 49 B
典型解析头部 = 81 B JSON + 4 B 长度前缀 = 85 B
典型解析头部 bps = 85 × 8 / 0.25 = 2720 bps
实验均值总在线载荷 = 4531.62 bps

在三栏下方放一个横向说明条：
表中的“典型 bps”按完整 0.25s chunk 计算；实验汇总均值会略高，因为句尾 chunk 的有效时长可能小于 0.25s，但报文格式仍需携带固定解析信息。

画面底部放一条结论条：
有效载荷决定真正发送的数据主体；解析头部用于接收端正确恢复字节流。RVQ 的优势在于主体从大量 float32 / PCM 采样变成少量 10-bit 索引。

视觉风格：
白底或极浅灰背景，深蓝文字，三栏对齐，公式清楚。PCM 用浅灰，latent 用青蓝，RVQ 用绿色，解析头部统一用琥珀色小块。数字和公式要大而清楚，避免小字密集。像论文方法补充图或答辩口径解释页。

注意事项：
不要把解析头部画成语音内容或额外语义信息；它只是用于恢复 payload 格式的说明。
不要说解析头部包含 codebook、checkpoint 或模型参数。
不要把 1500 bps 写成总在线载荷；它只是理论索引主体载荷。
不要加入 STOI、PESQ、CPU RTF、NAS 或整句分包结果。
不要把码本画成人工语义标签表、文本词典、单词表或情绪标签表。
```

---

## 补充页：双端实时通信 Demo 验证 / 真实麦克风到扬声器闭环

建议标题：

```text
双端实时通信 Demo：从麦克风到扬声器的 RVQ 索引闭环
```

建议副标题：

```text
真实进程、真实设备、真实 TCP 链路验证：在线传输对象是 RVQ indices，而不是波形、连续 latent 或 codebook
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 系统验证图，主题是“双端实时通信 Demo：从麦克风到扬声器的 RVQ 索引闭环”。

整体背景必须是白色或极浅灰色，正式科研汇报系统图风格，深蓝标题，绿色表示 RVQ 索引与在线数据包，青蓝表示语音流或音频 chunk，浅灰表示本地设备，琥珀色只用于提示解析头部或工程监视信息。画面清晰、克制、信息密度适中，不要科幻风，不要真实人物，不要复杂 3D，不要大面积深色背景。

画面顶部放一条清晰前提说明：
本页展示真实双端通信 demo 的工程闭环；正式指标仍以前面的 payload、quality 和 CPU timing 实验为准。

画面主体采用左右双主机结构，中间用一条网络链路连接：

左侧 45%：主机 A 发送端
画一台简洁电脑和麦克风图标，标题为：
主机 A：发送端

从上到下画 6 个步骤，用箭头串起来：
1. 麦克风输入
2. 0.25s audio chunk
3. SpeechTokenizer encode
4. RVQ indices
5. 可选只保留前 3 层
6. 打包并发送 TCP packet

在 RVQ indices 旁边用绿色小块表示离散索引序列，标注：
3 层 RVQ 索引
在线传索引，不传波形
不传连续 latent
不传 codebook

中间 10%：TCP 链路
画一条从主机 A 指向主机 B 的绿色网络箭头，箭头上放一个小数据包，标注：
TCP packet
[4B header_len][header JSON][body bytes]

在数据包下方用琥珀色小注释：
header 用于恢复 dtype、shape、body_len
body 是 RVQ code ndarray 字节流

右侧 45%：主机 B 接收端
画一台简洁电脑和扬声器图标，标题为：
主机 B：接收端

从上到下画 6 个步骤，用箭头串起来：
1. 接收 TCP packet
2. 解析 header
3. 恢复 RVQ indices
4. SpeechTokenizer decode
5. 播放缓冲
6. 扬声器输出

在接收端旁边放一个小型监视面板，标题为：
运行监视
列出：
输入 RMS
播放 RMS
缓冲状态
丢包 / 过载提示

画面右上角放一个小信息框，标题为：
Demo 组件
内容：
目录：两用户通信demo
发送端：demo_send_A_double.py
接收端：demo_recv_B_double.py
默认端口：12346
核心设置：chunk = 0.25s，RVQ layers = 3

画面底部放一条结论条：
双端 demo 证明系统流程已经从离线回环推进到真实麦克风采集、TCP 传输、接收端解码与扬声器播放闭环；它是工程可运行性验证，不替代严格实验指标。

视觉风格：
白底或极浅灰背景，深蓝文字，绿色突出 RVQ indices 和 TCP packet，青蓝表示音频 chunk，浅灰表示电脑、麦克风和扬声器设备，琥珀色表示 header 和 monitor。左右两端要对称，中间链路清晰，箭头方向明确，像论文系统验证图或答辩 demo 说明图。不要卡通人物，不要真实照片，不要赛博朋克，不要把数据包画成模型文件或 checkpoint。

注意事项：
不要说 demo 是严格端到端延迟 benchmark。
不要把 demo 运行体验替代 STOI、PESQ、payload 或 CPU timing 主实验。
不要说在线阶段会传 codebook、checkpoint、模型参数或连续 latent。
不要说完成真实量子链路或真实安全链路验证；这里只是普通 TCP 双端通信 demo。
不要把公开 n_q=8 模型前三层结果混成正式 n_q=3 checkpoint 的质量结论。
不要在画面里堆满完整命令，命令可以放讲者备注或附录。
```

---

## 实验路线总览页：六组实验如何支撑低负载语音语义通信

建议标题：

```text
实验路线总览：从轻量编码器搜索到三用户实时系统验证
```

建议副标题：

```text
六组实验依次回答架构是否轻量、模型是否可训练、低负载是否可用、对比是否公平、设计是否必要、系统是否能在线运行
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 实验路线总览图，主题是“六组实验如何支撑低负载语音语义通信”。

整体背景必须是白色或极浅灰色，正式中文科研汇报风格，深蓝标题，青蓝表示模型训练流程，绿色表示索引传输与低负载，琥珀色表示评估和诊断，灰色表示工程系统验证。画面清晰、克制、像论文 Experimental Plan 总览图，不要科幻海报，不要真实照片，不要人物，不要复杂 3D。

画面主体使用横向时间线或 2 行 3 列流程卡片，展示六组实验，每个实验卡片都要有编号、实验名、核心问题、主要产物。六张卡片之间用细线箭头连接，表示前后依赖关系：

实验一：Encoder-side NAS
核心问题：在固定 index-only transmission 接口下，能否搜索更轻量的发送端 encoder
主要产物：best architecture、Pareto frontier、encoder params / MACs / RTF、接口检查

实验二：SCIT-Speech-Base 训练
核心问题：能否训练共享码本索引语音模型
主要产物：Base checkpoint、L=1/2/3 截层样本、codebook usage sanity check

实验三：低负载 / 轻量信道感知适配
核心问题：random L 和 mild index perturbation 是否提升 L=1/2 可用性
主要产物：SCIT-Speech-LCA、Base vs LCA、clean / dropout / substitution 评估

实验四：Baseline 对比
核心问题：index-only transmission 在负载-可用性曲线中的位置
主要产物：ideal / packed / packetized payload、PCM / Opus / AMR-WB 等对比、质量指标

实验五：消融与诊断
核心问题：NAS、LCA、semantic distillation、random L sampling 是否真的必要
主要产物：ablation matrix、variant metrics、compute profile、codebook diagnostics

实验六：三用户实时系统验证
核心问题：共享码本索引流能否进入 A/B/C 三用户在线通信流程
主要产物：session logs、routing logs、payload、latency、reconstructed demo audio

在时间线上方放一条总目标说明：
shared RVQ codebooks are pre-deployed; online link transmits discrete indices only

在时间线下方放一条依赖关系说明：
Exp1 选择 encoder → Exp2 训练 Base → Exp3 得到 LCA → Exp4 做公平对比 → Exp5 验证设计必要性 → Exp6 验证系统可运行性

右下角放一个小型固定配置框：
sample rate = 16 kHz
M = 3, K = 1024
latent rate = 50 steps/s
ideal load = 500L bps
L ∈ {1, 2, 3}

底部结论条：
整套实验不是单点展示，而是一条从“方法可训练”到“通信可评估”再到“系统可运行”的证据链。

视觉风格：
白底或极浅灰背景，六张实验卡片不要做成厚重大卡片，使用细边框、轻阴影、统一图标和清晰编号即可。编号圆点用深蓝，模型相关用青蓝，低负载索引用绿色，评估诊断用琥珀色，系统验证用灰色。文字不要拥挤，每个实验只保留 2-3 行关键词。

注意事项：
不要填入尚未真实产生的实验结果数值。
不要把六组实验画成彼此独立的散点，要体现前后依赖。
不要把实验六画成主质量证明；它是系统可行性验证。
不要把具体开源 tokenizer 名称作为方法主语。
不要出现真实机构 logo、商业产品 logo 或真实人物照片。
```

---

## 实验描述页 1：实验一 Encoder-side NAS / 固定接口下搜索轻量语义提取器

建议标题：

```text
实验一：固定通信接口下的 Encoder-side NAS
```

建议副标题：

```text
只搜索发送端 x(t) → Z 的语义/声学 encoder，在不改变 RVQ 索引接口和 500L bps 负载公式的前提下降低计算成本
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 实验设计图，主题是“实验一：固定通信接口下的 Encoder-side NAS”。

整体背景必须是白色或极浅灰色，正式科研汇报方法图风格，深蓝标题，青蓝表示 encoder 搜索，绿色表示固定通信接口，琥珀色表示 Pareto 选择，灰色表示被固定或不搜索的模块。不要科幻风，不要真实人物，不要复杂 3D。

画面采用左中右三栏结构：

左栏标题：固定通信接口
画出一个锁定的接口框，列出固定条件：
sample rate = 16 kHz
prod(strides) = 320
latent rate = 50 steps/s
latent dimension = 1024
M = 3, K = 1024
L ∈ {1, 2, 3}
R_index(L) = 500L bps
用绿色锁形小图标表示这些条件不可被 NAS 改变。

中栏标题：NAS 搜索对象
画出输入语音 x(t) 到连续表示 Z 的 encoder 结构搜索：
x(t) → searchable encoder → Z
在 searchable encoder 内部画四层 stride schedule、residual ops、channel width、LSTM、SE switch、activation 等候选开关。
标注：
search encoder only
not RVQ
not packetization
not ChannelSim
not three-user routing
如果展示 decoder，只能画成灰色的 geometry-matched decoder，并标注：
decoder = reverse(strides), not NAS contribution

右栏标题：Proxy evaluation and selection
画出候选架构列表进入评估器，输出四类指标：
semantic proxy loss
encoder params
encoder MACs
encoder RTF
再画一个小型 Pareto frontier 散点图，横轴 compute cost，纵轴 semantic proxy loss，用琥珀色高亮 selected architecture。

画面底部放一个 run 产物条：
candidate JSON → interface check → nas_records.csv → pareto_frontier.csv → best_seanet_config.json

底部结论条：
实验一只回答“发送端 encoder 能否更轻量且保持固定索引接口”，不直接给出最终通信质量结论。

视觉风格：
白底或极浅灰背景，线条清晰，三栏对齐。固定项用绿色锁，搜索空间用青蓝模块，评估和 Pareto 用琥珀色散点，非搜索模块用灰色。像论文方法实验设计图，不要卡通，不要真实照片，不要过多装饰。

注意事项：
不要把少量手写候选画成完整 NAS 搜索。
不要暗示 NAS 搜索了 RVQ 层数 M、码本大小 K、传输层数 L 或通信协议。
不要把 decoder 变化写成 encoder NAS 贡献。
不要填入未经真实 run 产生的 best 指标、候选数或 Pareto 数值。
不要改变 latent rate，否则会污染 500L bps 负载口径。
```

---

## 实验描述页 2：实验二 SCIT-Speech-Base 训练 / 共享码本索引模型实例化

建议标题：

```text
实验二：训练 SCIT-Speech-Base，实例化共享码本索引传输模型
```

建议副标题：

```text
训练 encoder、shared RVQ codebooks 和 decoder，得到后续低负载适配、baseline 对比、消融和系统验证的基础 checkpoint
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 实验流程图，主题是“实验二：SCIT-Speech-Base 训练”。

整体背景必须是白色或极浅灰色，正式科研汇报训练流程图风格，深蓝标题，青蓝表示前向训练路径，绿色表示共享 RVQ codebooks，琥珀色表示损失函数和诊断，灰色表示数据和产物。不要真实照片，不要人物，不要复杂 3D，不要赛博朋克。

画面主体从左到右展示基础训练闭环：

1. Training data
画 LibriSpeech train-clean-100 的语音 waveform 小条和 HuBERT / semantic teacher feature 小块。
标注：
16 kHz mono speech
train / valid manifest

2. Encoder E_theta
画发送端表示编码器，把语音映射为连续表示：
x(t) → Z
标注：
Z shape = [1024, T_q]
latent rate = 50 steps/s

3. Shared RVQ codebooks C*
画 3 层 codebook 矩阵，标注：
M = 3
K = 1024
indices I
用绿色突出共享码本是模型训练对象。

4. Decoder D_psi
画接收端重建解码器：
I_{1:3} + C* → reconstructed speech x_hat
旁边小注释：
full-depth reconstruction for Base training

5. Training losses
画一个琥珀色损失模块，列出：
waveform reconstruction loss
multi-scale mel loss
RVQ commitment loss
semantic distillation loss
adversarial / feature matching loss
注意文字简短，不要写具体未验证权重。

画面下方放一条诊断支线：
Base checkpoint → L=1 / L=2 / L=3 decode samples → codebook usage sanity check
诊断框中列出：
usage rate
perplexity
dead code ratio
layer-wise distribution

右侧放产物框：
SCIT-Speech-Base checkpoint
loss curves
fixed reconstruction samples
codebook_usage.csv
layer_reconstruction.csv

底部结论条：
实验二的目标是获得可加载、可截层解码、码本未明显 collapse 的 Base 模型；完整通信质量结论留给实验三和实验四。

视觉风格：
白底或极浅灰背景，训练主路径用青蓝箭头，RVQ 码本用绿色，loss 和 sanity check 用琥珀色，产物用灰色。模块间留白充分，像论文训练 pipeline 图。

注意事项：
不要把历史 public checkpoint 直接画成本文实验结果。
不要说实验二已经完成最终 WER、PESQ、STOI 主结论；这些属于后续统一评估。
不要把 ASR 评价模型画进训练路径。
不要把 codebook 画成人工语义标签表、文本词典或单词表。
不要编造 loss 数值、checkpoint 名称或 codebook usage 数字。
```

---

## 补充页：SCIT-Speech-Base 训练机制 / 重建、蒸馏与对抗训练

建议标题：

```text
SCIT-Speech-Base 是如何训练出来的：重建损失、语义蒸馏与对抗训练联合优化
```

建议副标题：

```text
本项目把 SpeechTokenizer 作为生成器，联合 SEANet encoder、3 层 RVQ 共享码本和 SEANet decoder，并用 HiFi-GAN 风格多判别器提升重建语音质量
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 训练机制图，主题是“SCIT-Speech-Base 是如何训练出来的：重建、蒸馏与对抗训练”。

整体背景必须是白色或极浅灰色，正式科研汇报训练机制图风格，深蓝标题，青蓝表示生成器前向路径，绿色表示 RVQ codebooks / indices，琥珀色表示损失函数和反向传播，红色少量用于判别器对抗信号，灰色表示固定语义教师特征。不要真实照片，不要人物，不要复杂 3D，不要赛博朋克。

画面主体采用“上方生成器前向路径 + 下方对抗训练与损失闭环”的结构。

上方主路径标题：Generator: SpeechTokenizer
从左到右画出完整生成器：
输入 speech x(t)
→ SEANet Encoder E_theta
→ continuous latent Z
→ Residual Vector Quantizer
→ 3-layer shared RVQ codebooks C*
→ quantized representation
→ SEANet Decoder D_psi
→ reconstructed speech x_hat(t)

在 Encoder 旁边标注：
strides = [8,5,4,2]
downsample rate = 320
latent rate = 50 steps/s
latent dimension = 1024

在 RVQ 模块内部画 3 层绿色 codebook：
M = 3 RVQ layers
K = 1024 codes per layer
full-depth training uses all 3 layers
commitment loss stabilizes quantization
强调：训练阶段学习 shared RVQ codebooks；在线阶段只发送 code indices。

在生成器主路径下方画“训练数据与语义教师”支线：
LibriSpeech 16 kHz speech segment
paired HuBERT semantic feature
semantic teacher is fixed, not trained
HuBERT feature → semantic distillation loss
注意不要把 ASR 评价模型画进训练链路。

画面下半部分左侧标题：Step A. Update discriminators
画三个判别器并列，使用红色和深灰色：
MPD: Multi-Period Discriminator
MSD: Multi-Scale Discriminator
MSTFTD: Multi-Scale STFT Discriminator
箭头输入：
real speech x(t)
generated speech x_hat(t) with detach
判别器损失框写：
L_D = real should be 1 + generated should be 0
LSGAN discriminator loss
只更新 D，不更新生成器
在 x_hat 箭头上明确标注 detach。

画面下半部分右侧标题：Step B. Update generator
画同一组判别器再次接收 real speech x(t) 和 generated speech x_hat(t)，这次 x_hat 不 detach。
生成器损失面板列出 6 项：
1. waveform L1 reconstruction loss
2. multi-scale Mel loss
3. RVQ commitment loss
4. d-axis semantic distillation loss with HuBERT features
5. discriminator feature matching loss
6. adversarial loss: generated should be judged real

在损失面板旁边写当前项目配置中的权重口径：
recon_loss_lambda = 500
commitment_loss_lambda = 10
distill_loss_lambda = 120
mel_loss_lambdas = [45, 1, 1, 1]
不要写未验证的新权重。

画一个总损失公式框：
L_G =
feature matching
+ adversarial
+ multi-scale Mel
+ 10 × commitment
+ 500 × waveform L1
+ 120 × semantic distillation

在公式下方放两条说明：
semantic distillation aligns first RVQ-layer feature with fixed HuBERT representation
adversarial + feature matching improves perceptual realism of reconstructed speech

画面右侧放“训练循环”小流程：
for each batch:
1. x, semantic_feature
2. generator forward: x_hat, loss_q, feature
3. update discriminators with x and detached x_hat
4. update generator with reconstruction + distillation + GAN losses
5. log TensorBoard and validate by dev Mel error

右下角放“Base 训练边界”框：
Base training = full-depth reconstruction with n_q = 3
not random L training
not ChannelSim training
not baseline comparison
L=1/2/3 are checked after training by truncating codes

底部放产物和检查条：
SCIT-Speech-Base_best.pt
loss curves
validation audio samples
L=1/2/3 reconstruction samples
codebook usage sanity check

底部结论条：
SCIT-Speech-Base 的训练本质是一个带 RVQ 离散瓶颈的神经声码器式重建训练：用波形/Mel/语义蒸馏保证内容和谱形，用 RVQ commitment 稳定码本，用多判别器对抗训练提升重建语音的真实感。

视觉风格：
白底或极浅灰背景，上方生成器路径简洁横向展开；下方左右两块分别表示判别器更新和生成器更新。生成器用青蓝，RVQ 码本用绿色，loss 用琥珀色，判别器用深灰加少量红色。箭头方向必须清晰，尤其是 detach 和不 detach 的区别。像论文方法训练图，不要画成工程脚本流程图。

注意事项：
不要把这页画成运行命令或文件目录说明；重点是训练目标和交替优化机制。
不要说 Base 训练已经包含 random L sampling 或 ChannelSim；这些属于实验三 LCA。
不要把 HuBERT semantic teacher 画成可训练模块；它是固定教师特征来源。
不要把 ASR、WER、PESQ、STOI 放进训练损失，它们属于后续评价。
不要说在线阶段会传 codebook、checkpoint 或模型参数。
不要编造实际 loss 数值、训练步数、checkpoint 时间或最终质量指标。
```

---

## 补充页：SCIT-Speech-Base 项目训练落地 / 从配置到 checkpoint

建议标题：

```text
SCIT-Speech-Base 的项目训练落地：配置、数据清单、训练器与产物留存
```

建议副标题：

```text
这一页只解释工程留存链路：如何把上述训练机制落到本项目 run 目录、日志、checkpoint、样本和 sanity check
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 工程留存流程图，主题是“SCIT-Speech-Base 项目训练落地：从配置到 checkpoint”。

整体背景必须是白色或极浅灰色，正式科研汇报工程流程图风格，深蓝标题，青蓝表示训练入口和 trainer，绿色表示 checkpoint 与截层样本，琥珀色表示日志和 diagnostics，灰色表示配置、数据清单和文件产物。不要真实照片，不要人物，不要复杂 3D，不要赛博朋克。

画面采用从左到右的 4 阶段流水线结构，每个阶段用清晰模块表示，并用箭头串联：

阶段 1：冻结训练配置
画一个配置文件模块，标题：
config/spt_base_cfg.json
旁边列出固定关键参数：
sample_rate = 16000
strides = [8, 5, 4, 2]
dimension = 1024
n_q = 3
codebook_size = 1024
results_folder → current run directory

阶段 2：准备训练输入
画两个输入清单文件：
train_files.txt
valid_files.txt
每一行显示简化格式：
audio.wav + hubert.npy
旁边画数据读取模块：
speechtokenizer/trainer/dataset.py
标注：
load 16 kHz speech segments
load semantic teacher features
align audio and HuBERT representation
不要把 ASR 评价模型画进训练输入。

阶段 3：启动训练入口
画命令入口模块：
scripts/train_example.py
scripts/train_example.sh
Accelerate launch
箭头指向训练器：
SpeechTokenizerTrainer
speechtokenizer/trainer/trainer.py
在训练器旁边标注：
records loss
TensorBoard logs
validation samples
checkpoints

阶段 4：训练产物与 Base 检查
画输出目录树：
checkpoints/SCIT-Speech-Base_best.pt
logs/stdout.log
logs/stderr.log
logs/tensorboard/
metrics/loss_curves.csv
samples/fixed/recon_L1/
samples/fixed/recon_L2/
samples/fixed/recon_L3/
metrics/codebook_usage.csv
reports/summary.md

在输出目录旁边画两个 sanity check 小面板：
Layer reconstruction check:
L=1 decode
L=2 decode
L=3 decode
non-silent, length aligned

Codebook usage check:
usage rate
perplexity
dead code ratio
layer-wise distribution

画面顶部放一条简洁说明：
SCIT-Speech-Base is not a public checkpoint copy; it is the project-trained Base model saved with config, logs, data split and provenance.

画面底部放一条结论条：
工程留存的关键是让 Base checkpoint 能追溯到配置、数据清单、训练日志、验证样本和码本诊断，后续实验三到实验六都引用同一个可复现 Base。

视觉风格：
白底或极浅灰背景，4 个阶段横向排列，阶段编号清晰。配置和文件用灰色文档图标，训练入口和 trainer 用青蓝模块，checkpoint 与 L=1/2/3 样本用绿色，日志和诊断用琥珀色。整体像工程训练留存图，不要做成训练原理图。

注意事项：
不要说 SCIT-Speech-Base 来自 public checkpoint；如果复用历史 checkpoint，也必须表现为带 provenance 的项目 run 产物。
不要把 `L=1/2/3` 画成三组独立训练模型；它们是同一个 Base checkpoint 的三种截层解码操作点。
不要把 ASR、WER、PESQ、STOI 画进训练落地链路；这些属于后续评估。
不要暗示在线阶段会传 codebook、checkpoint 或模型参数。
不要编造实际 loss、训练步数、checkpoint 时间、usage rate 或重建质量数值。
```

---

## 实验描述页 3：实验三低负载/轻量信道感知适配 / Base 到 LCA

建议标题：

```text
实验三：低负载与轻量信道感知适配，从 Base 到 LCA
```

建议副标题：

```text
通过 random L sampling 和 mild index-level perturbation，让模型更适应只传前 L 层索引的通信条件
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 实验设计图，主题是“实验三：低负载/轻量信道感知适配”。

整体背景必须是白色或极浅灰色，正式科研汇报训练与评估图风格，深蓝标题，绿色表示 RVQ indices 和低负载，青蓝表示 Base/LCA 模型路径，琥珀色表示轻量信道扰动和评价，灰色表示固定条件。不要真实照片，不要人物，不要复杂 3D。

画面左侧标题：Initialize from Base
画一个 SCIT-Speech-Base checkpoint 输入到 fine-tuning 模块。
标注：
encoder + RVQ codebooks + decoder
end-to-end fine-tuning
not decoder-only main route

画面中间标题：Low-load and channel-aware training
画一个 batch 训练流程：
speech x → encode → RVQ indices I → sample L from {1,2,3} → retain I_L → ChannelSim → reconstruct x_hat_{L,c}

在 sample L 模块旁边画三个小按钮：
L=1, 500 bps ideal
L=2, 1000 bps ideal
L=3, 1500 bps ideal
用绿色表示它们是负载操作点。

在 ChannelSim 模块中画三种轻量条件：
clean
index dropout with previous-index replacement
light index substitution
用琥珀色小标识强调这是 mild index-level perturbation，不是强信道鲁棒实验。

画面右侧标题：Base vs LCA evaluation
画一个矩阵表格，行是模型：
SCIT-Speech-Base
SCIT-Speech-LCA
列是：
L=1
L=2
L=3
clean
dropout-low
substitution-low
表格中只画占位勾选和空白指标格，不填真实结果。

右下角放指标框：
WER / CER
STOI
PESQ / ViSQOL
semantic similarity
RTF
degradation under perturbation

底部结论条：
实验三重点验证 LCA 是否改善 L=1 和 L=2 的可用性，并观察轻量索引扰动下的退化幅度；不宣称强信道鲁棒。

视觉风格：
白底或极浅灰背景，Base 到 LCA 的主箭头清楚，random L 用绿色分支，ChannelSim 用琥珀色轻量扰动图标，评估矩阵简洁对齐。文字要短，不要让表格拥挤。

注意事项：
不要把 dropout 画成波形丢失；扰动作用在离散索引矩阵 I_L 上。
不要引入 mask token，第一版 dropout 使用 previous-index replacement。
不要把 substitution 画成非法索引；替换后仍是合法 codebook address。
不要写强信道鲁棒、高 BER、复杂 FEC 等主张。
不要编造 Base/LCA 指标数值。
```

---

## 实验描述页 4：实验四 Baseline 对比 / 负载-可用性曲线

建议标题：

```text
实验四：Baseline 对比，定位 index-only transmission 的低负载工作区间
```

建议副标题：

```text
同时报告 ideal、packed 和 packetized payload，用同一测试集比较 SCIT-Speech 与 PCM、传统 codec 和可选 neural codec
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 实验对比设计图，主题是“实验四：Baseline 对比与负载-可用性曲线”。

整体背景必须是白色或极浅灰色，正式科研汇报数据对比计划图风格，深蓝标题，绿色表示 SCIT index-only transmission，浅灰表示 PCM，青蓝表示传统或神经 codec，琥珀色表示 payload accounting。不要真实照片，不要人物，不要复杂 3D。

画面左侧标题：Methods under the same test set
画一个方法列表进入统一评估管线：
SCIT-Speech-Base, L=1/2/3
SCIT-Speech-LCA, L=1/2/3
PCM 16-bit 16 kHz
Opus / AMR-WB
optional Codec2
optional neural codec
每个方法旁边用小标签显示“executed / optional / skipped if unavailable”，但不要填真实状态。

画面中间标题：Payload accounting
画一个三层 payload 分解图，重点突出 SCIT 的三种口径：
ideal index bitrate = L × 50 × 10
packed payload = bit-packed indices
packetized payload = packed bytes + header + user/session id + timestamp + length fields
再画一个 overhead ratio 公式：
overhead ratio = (packetized - ideal) / ideal
用琥珀色突出 header / metadata，不要把 `.npy` 或 int64 文件画成 packed payload。

画面右侧标题：Unified quality evaluation
画一个统一评价器，输入所有重建/解码 wav，输出指标：
WER / CER
STOI
PESQ / ViSQOL
semantic similarity
RTF
compression ratio

右下角画两张小型占位图：
payload vs. method
quality vs. bitrate
图上只放轴和占位曲线，不写虚构数字。

底部结论条：
实验四不追求宣称全面超过传统 codec，而是展示共享码本索引传输在极低负载区间形成的不同工作点。

视觉风格：
白底或极浅灰背景，左中右三段对齐。SCIT 用绿色索引包，PCM 用浅灰波形，传统 codec 用青蓝压缩模块，payload 元数据用琥珀色。图表像论文 Results 计划页，克制清楚。

注意事项：
不要只写 500L bps，必须体现 packed 和 packetized payload。
不要把 `.npy`、int64 array storage 或模型文件大小当作传输 payload。
不要强行让 Opus / AMR-WB 工作在不支持的 500-1500 bps 同码率点；应画 bitrate-quality curve。
不要说 SCIT 在所有音质指标上全面优于传统 codec。
不要编造 baseline 编码结果、PESQ、STOI、WER 或 payload 数字。
```

---

## 实验描述页 5：实验五消融与诊断 / 关键设计是否必要

建议标题：

```text
实验五：消融与诊断，验证关键设计是否真的有用
```

建议副标题：

```text
用最小必要 ablation matrix 检查 NAS、LCA、semantic distillation、random L sampling 和可选 ChannelSim 的贡献边界
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 消融实验设计图，主题是“实验五：消融与诊断”。

整体背景必须是白色或极浅灰色，正式科研汇报实验矩阵图风格，深蓝标题，琥珀色表示消融变量，绿色表示低负载索引操作点，青蓝表示模型条件，灰色表示固定条件。不要真实照片，不要人物，不要复杂 3D。

画面主体使用一个清晰的 ablation matrix，左侧是消融组，右侧是对照组 / 变量组 / 只允许改变的因素。

必做消融 1：
hand-designed encoder vs NAS encoder
只允许改变：encoder architecture
提示：decoder must be fixed or explicitly marked

必做消融 2：
SCIT-Speech-Base vs SCIT-Speech-LCA
只允许改变：low-load and channel-aware adaptation
重点观察：L=1/2 clean and mild perturbation

必做消融 3：
with vs without semantic distillation
只允许改变：distillation loss weight
重点观察：低层索引内容保持

必做消融 4：
with vs without random L sampling
只允许改变：L sampling strategy
重点观察：截层传输 L=1/2/3

可选消融：
ChannelSim on/off
条件：只有实验三扰动评估稳定时执行

画面右侧放一个“统一评价口径”面板：
same train / valid / test split
same M=3, K=1024
same L=1/2/3
same ASR and metric scripts
same payload accounting
same fixed samples

右下角放指标面板：
WER / CER
STOI
PESQ / ViSQOL
semantic similarity
RTF
encoder params / MACs
codebook usage / perplexity / dead code ratio

画面底部放一个警示条：
每个 variant 只能改变一个目标变量；失败 variant 也要保留日志和 failure report。

底部结论条：
实验五的目标不是扩大主实验矩阵，而是把每个关键设计的收益、代价和失败风险拆清楚。

视觉风格：
白底或极浅灰背景，消融矩阵使用细线表格，变量列用琥珀色小标签，固定条件用灰色锁，低负载 L=1/2/3 用绿色小标签。文字清楚，不要拥挤。

注意事项：
不要加入 M/K 大规模搜索、高 BER、复杂 FEC 或第一层 probing 大实验。
不要让一个 variant 同时改变多个关键因素。
NAS 消融必须控制 decoder 影响；如果 decoder 同时变化，要标为 implementation variant。
不要把 semantic distillation 写成新 tokenizer 贡献，它只是训练辅助。
不要编造消融结果、checkpoint 或指标。
```

---

## 实验描述页 6：实验六三用户实时系统验证 / 多用户在线索引流闭环

建议标题：

```text
实验六：三用户实时系统验证，index-only transmission 进入在线通信流程
```

建议副标题：

```text
A/B/C 三端预部署同一共享码本，在线传输带 user/session id 的 RVQ 索引包，记录路由、payload、延迟和异常
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 系统验证图，主题是“实验六：三用户实时系统验证”。

整体背景必须是白色或极浅灰色，正式科研汇报系统图风格，深蓝标题，绿色表示 RVQ index packets，青蓝表示三用户音频流，琥珀色表示 packet schema / payload / latency logs，灰色表示本地设备和 replay 模式。不要真实照片，不要人物，不要复杂 3D，不要赛博朋克。

画面主体采用三端网络结构，三个端点分别是：
User A
User B
User C
每个端点都画简洁电脑、麦克风、扬声器图标，不要画真实人物。

在三个端点内部都放一个相同的预部署框：
same SCIT-Speech checkpoint
same shared RVQ codebooks
same config
用绿色锁形图标表示三端共享同一组 codebook。

画面中心放一个 Session Router / Relay 模块，连接 A/B/C 三端。箭头表示轮流发言模式：
A sends indices → router → target receiver
B sends indices → router → target receiver
C sends indices → router → target receiver
可以用虚线表示 optional near-concurrent mode，但不要把它画成复杂会议混音系统。

在每条在线链路上画小数据包，标注：
source_user_id
target_user_id
session_id
seq_id
timestamp
L
packed_indices
payload_length

画面下方放一条执行流程：
microphone or replay wav → chunking → encode + RVQ → retain I_L → packetize → route → receiver lookup → decode → playback / reconstructed wav

右侧放日志和指标面板：
session_events.jsonl
routing.jsonl
payload.jsonl
latency.jsonl
exceptions.jsonl
session_payload.csv
session_latency.csv
routing_accuracy.csv

右下角放指标说明：
routing accuracy
single-stream payload
session aggregate payload
avg / p95 / max latency
RTF
exception count

底部结论条：
实验六验证系统可行性：三用户会话中能否正确路由索引包、重建语音并记录真实工程开销；质量结论仍以实验三和实验四为准。

视觉风格：
白底或极浅灰背景，三端对称分布，中心 router 清晰，绿色小数据包沿箭头流动，琥珀色日志面板放在右侧，设备图标简洁灰色。整体像论文系统可行性图或答辩 demo 架构图。

注意事项：
不要说实验六替代 WER/PESQ/STOI 主评估。
不要把三用户 demo 画成复杂会议混音或商业通信产品。
不要说完成真实量子链路或真实安全链路验证。
不要只画音频，不画 user/session id、packet schema、routing log、payload 和 latency log。
不要编造路由正确率、延迟、payload 或 demo 音频结果。
```

---
