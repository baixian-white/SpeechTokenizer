# 给 AI 的论文写作提示词：基于当前实验结果重写 SCIT-Speech 论文

你将扮演一名严谨的机器学习/语音通信论文写作者。请基于我提供的本地材料，重写一篇中文论文初稿。论文主题是 **SCIT-Speech：基于共享 RVQ 码本索引传输的极低码率语音通信方法**。

## 1. 必须阅读和使用的材料

请按优先级阅读以下文件，并只使用这些文件中已经记录的实验事实和数值：

1. `output/doc/paper_drafts/scit_speech_experiment_main_tables_20260609.md`
   - 这是最新实验主表整理稿，包含 Exp7–Exp11 的全量 clean、index 扰动、packet/burst 和 ASR/WER 结果。
   - 写结果章节时优先使用这里的表格和解释边界。

2. `output/doc/实验记录.md`
   - 这是完整实验事实记录，包含 Exp1–Exp11 的运行状态、模型路径、结果解释边界。
   - 如果需要确认某个实验是否完成或某个数字来源，以此为准。

3. `output/doc/paper_drafts/scit_speech_cn_draft.md`
   - 这是旧版中文论文草稿，结构和部分方法描述可以复用，但其中实验部分仍偏旧。
   - 不要原样沿用其中“当前评估仍限于小样本、同源语料”等旧表述。

4. `output/doc/paper_outline_v2_path_a_2026_06_02.md`
   - 这是论文大纲和论证链，可用于组织章节结构。
   - 注意其中部分“待决事项”已经被后续实验完成，需以最新实验记录为准。

## 2. 论文主题和定位

请把本文写成一篇**极低码率语音通信方法论文**，不要写成“方法学论文”，也不要使用“方法学框架”这类表述。

推荐核心表述：

> 本文提出 SCIT-Speech，一种面向极低码率语音通信的共享码本 RVQ 索引传输方法。发送端与接收端预共享 RVQ 码本、模型参数和系统配置；运行时信道仅传输离散索引序列。通过截取并传输前 L 层 RVQ 索引，系统可在约 500、1000 和 1500 bps 三个工作点之间切换。

可以使用“实验体系”“系统实例”“训练配方”，但不要把 SCIT-Speech 称为“方法学框架”。如果需要用 “framework”，只用于非常宽泛的系统图说明，正文主张仍写成“方法”或“系统实例”。

## 3. 应写出的论文结构

请生成一篇完整中文论文初稿，建议结构如下：

1. 标题
2. 摘要
3. 关键词
4. 引言
   - 背景与动机
   - 研究问题
   - 本文贡献
   - 范围与边界
5. 相关工作
   - 传统语音编码与神经音频 codec
   - 离散语音 token / RVQ / SpeechTokenizer 类工作
   - 语音语义通信与丢包鲁棒通信
6. 方法
   - 系统总览：shared codebook + index-only transmission
   - 负载核算：`R(L)=L f_q ceil(log2 K)`，当前 `K=1024`、`f_q=50 Hz`，所以 `R(L)=500L bps`
   - SCIT-Speech-Base 训练
   - SCIT-Speech-LCA：random-L、强 index dropout/substitution、clean-vs-perturbed consistency
   - 轻量发送端 NAS encoder，只作为效率优化，不夸大为质量优越性证明
7. 实验设置
   - 数据、模型和评估层级
   - baseline
   - 指标
   - index dropout/substitution 与 packet/burst loss 设置
   - ASR/WER 设置
8. 结果
   - 低码率 baseline 对比（Exp4）
   - LibriSpeech 全量 clean 泛化（Exp7）
   - 全量 index dropout/substitution 鲁棒性（Exp8）
   - 全量 packet/burst loss 鲁棒性（Exp9）
   - clean 与扰动 ASR/WER 子集（Exp10/Exp11）
   - 消融：distillation、weak vs strong LCA、NAS efficiency、LCA component factorial（如有必要可简写）
9. 讨论
   - 为什么共享 RVQ 索引适合低码率语音通信
   - 为什么 LCA 的主要价值在鲁棒性而不是 clean 大幅提升
   - 为什么 semantic distillation 重要
   - 为什么不能过度声称所有指标全面提升
10. 局限性
11. 结论
12. 图表建议或附录建议

## 4. 必须使用的核心事实

### 4.1 系统配置

- 当前 SCIT-Speech 实例为 3 层 RVQ：`n_q=3`。
- 每层码本大小：`K=1024`。
- latent frame rate：约 `50 Hz`。
- 理想索引负载：`500L bps`。
- 三个操作点：`L=1/2/3` 对应约 `500/1000/1500 bps`。
- Base checkpoint：`output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`。
- LCA checkpoint：`output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`。

### 4.2 已完成实验

必须明确当前核心实验已经完成：

- Exp1：NAS encoder 搜索与 profiling。
- Exp2：SCIT-Speech-Base 训练。
- Exp3 v2：LCA 强扰动 + consistency 微调。
- Exp4：baseline 对比，包含 DAC、EnCodec、Opus、PCM 和 WER/CER。
- Exp5：消融与诊断，包括 semantic distillation、LCA 对比等。
- Exp7：LibriSpeech `test-clean/test-other` 全量 clean 评估。
- Exp8：全量 index dropout/substitution 扰动评估。
- Exp9：全量 packet/burst loss 评估。
- Exp10：clean ASR/WER 子集，`test-clean_300` 和 `test-other_300`。
- Exp11：扰动 ASR/WER 子集，`test-clean_100` 和 `test-other_100`。

### 4.3 低码率 baseline 主结果（Exp4）

固定 8 条样本的 baseline 对比中：

| 操作点 | 最佳 baseline | SCIT-Speech-LCA |
|---|---|---|
| 500 bps | DAC n_q=1: STOI 0.606, PESQ 1.055, WER 0.922 | L=1: STOI 0.797, PESQ 1.347, WER 0.461 |
| 1000 bps | DAC n_q=2: STOI 0.730, PESQ 1.148, WER 0.305 | L=2: STOI 0.857, PESQ 1.709, WER 0.188 |
| 1500 bps | DAC n_q=3: STOI 0.799, PESQ 1.271, WER 0.152；EnCodec 1.5 kbps: WER 0.302 | L=3: STOI 0.879, PESQ 1.891, WER 0.142 |

可以写：SCIT-LCA L=3 在 1.5 kbps 下 WER 0.142，与 Opus 6 kbps 的 WER 0.147 接近。

不能写：SCIT 全面优于 Opus 或全面优于所有传统 codec。

### 4.4 全量 clean 泛化（Exp7）

使用 `scit_speech_experiment_main_tables_20260609.md` 的主表 A。

核心结论：

- `test-clean` 2620 条、`test-other` 2939 条全量完成。
- LCA 在两个测试集、全部 `L=1/2/3` 上均降低 mel-L1 并提高 STOI。
- clean 收益是“小幅但稳定”，不能写成大幅提升。
- PESQ/SI-SNR 存在分层和条件差异，不能写成全部指标全面提升。

### 4.5 index dropout/substitution 鲁棒性（Exp8）

使用主表 B。

核心结论：

- `test-clean/test-other × L=1/2/3 × dropout-high/substitution-high` 上，LCA 均降低 mel-L1 并提高 STOI。
- 这说明 LCA 改善频谱稳定性和可懂度相关鲁棒性。
- SI-SNR 仍混合，不要写成所有指标全面改善。

### 4.6 packet/burst loss 鲁棒性（Exp9）

使用主表 C。

核心结论：

- `packet-loss-5p` 与 `burst-10f` 代表更通信化的包级/突发丢失场景。
- LCA 在两个测试集、三档码率上均保持 mel-L1/STOI 改善。
- PESQ 在 L2/L3 有时略低，SI-SNR 不稳定。
- 这部分是“通信化扰动下的频谱与可懂度鲁棒性证据”。

### 4.7 ASR/WER（Exp10/Exp11）

使用主表 D 和 E。

clean ASR/WER：

- `test-clean_300`：LCA 在 L1 和 L3 降低 WER，L2 基本持平略差。
- `test-other_300`：LCA 在全部 L=1/2/3 降低 WER。

扰动 ASR/WER：

- `test-clean_100`：收益主要集中在最低码率 L=1；L2/L3 下 LCA 略差。
- `test-other_100`：LCA 在所有 L 和两类扰动下均降低 WER/CER。
- 结论要写成：LCA 的可懂度鲁棒性在困难语音和信道扰动叠加时更明显，但 clean 子集高层码率 ASR 收益不稳定。

### 4.8 semantic distillation 消融

必须写：

- `distill_loss_lambda=30` 是当前主模型配置。
- distill0 在 step 47500 时 dev/mel 为 3.202，distill30 为 1.776，差距 +1.426。
- distill0 在 WER 上退化约 7–9 个百分点。
- distill0 反而可能使用更多 code，但质量更差，因此蒸馏不是简单提升 codebook usage，而是让有限索引更好承载内容相关信息。

### 4.9 NAS encoder

必须写：

- NAS encoder 参数量从 67.7M 降到 8.87M，降低约 86.9%。
- MACs 从 5.87 G/s 降到 0.618 G/s，降低约 89.5%。
- RTF mean 从 0.00819 降到 0.00368，降低约 55.1%。

必须保留边界：

- hand-designed encoder 还没有同条件完整重训。
- 所以不能把 NAS 写成“质量优于 hand-designed encoder”。
- 只能写成发送端效率 profiling 证据和当前系统实例的轻量化设计。

## 5. 必须避免的过度表述

请严格避免以下说法：

1. 不要说“方法学框架”。
2. 不要说“SCIT-Speech 全面优于 Opus / EnCodec / DAC”。
3. 不要说“LCA 在所有指标上全面提升”。
4. 不要说“clean 条件收益很大”。clean 收益只是小幅稳定。
5. 不要说“扰动 ASR/WER 全部改善”。`test-clean_100` 的 L2/L3 是略差的。
6. 不要说“NAS encoder 已证明质量不损”。只能说效率显著提升。
7. 不要说“已完成跨语料/跨语言/主观听测/真实网络验证”。这些仍是 limitation / future work。
8. 不要把 `.npy`、JSON、TCP 包头等工程存储开销混同为理想索引负载。

## 6. 推荐的摘要方向

摘要应包含以下逻辑：

1. 极低码率语音通信需要在 500-1500 bps 区间保持可懂度和可控负载。
2. SCIT-Speech 将 RVQ 索引从模型内部表示重新定义为信道载荷。
3. 发送端和接收端预共享码本与解码器，运行时只传前 L 层索引。
4. 系统形成 500/1000/1500 bps 三个操作点。
5. baseline 对比显示在该区间优于同码率 DAC/EnCodec，并在 L=3 接近 Opus 6 kbps 的 WER。
6. 全量 LibriSpeech clean/perturb/packet-burst 评估显示 LCA 稳定改善 mel-L1/STOI。
7. ASR/WER 子集显示困难语音和扰动条件下 LCA 可降低 WER/CER，但 clean 高层码率下 ASR 收益不稳定。
8. 消融显示 semantic distillation 和强扰动 + consistency 是关键。
9. 局限：还缺跨语料、跨语言、主观听测、AMR-WB、真实网络和多用户实时验证。

## 7. 推荐图表

请在论文中规划以下图表：

1. **系统图**：输入语音 -> encoder -> RVQ indices -> 只传前 L 层 -> shared codebook/decoder -> 重建语音。
2. **负载-质量曲线**：bps vs WER/STOI/PESQ/mel-L1，突出 500/1000/1500 bps。
3. **全量 clean 泛化表**：主表 A。
4. **index dropout/substitution 鲁棒性表或热力图**：主表 B。
5. **packet/burst loss 鲁棒性表或热力图**：主表 C。
6. **ASR/WER 表**：主表 D/E，可合并成一张较紧凑表。
7. **消融图**：distill0 vs distill30、weak LCA vs strong+consistency、NAS efficiency。

## 8. 写作输出要求

请输出一篇完整中文论文初稿，而不是只给提纲。

要求：

- 语言应像正式中文学术论文。
- 保持论证克制，不夸大。
- 所有数值必须来自上述文件，不要自行编造。
- 如果某个数值不确定，就写“见实验记录”或暂不写，不要猜。
- 引用可以暂用当前草稿中的编号体系 `[1-10]`，不要新增不存在的参考文献。
- 不需要生成 LaTeX，先写 Markdown 版本。
- 保留“局限性”独立章节。
- 在文末给出“图表清单”和“仍需人工确认的点”。


