# Exp22 说话人身份识别与保持实验汇总

生成日期：2026-07-08

## 结论摘要

当前模型能够保留并暴露说话人身份信息。证据分成两类：

1. 内部表征可区分说话人：在 VCTK 110 个说话人的三种子 probe 中，随机 top1 约为 0.009，而 Base `latent_stats L3` 达到 `0.581 ± 0.019`，LCA `latent_stats L3` 达到 `0.547 ± 0.041`。
2. 解码语音仍保留说话人身份：ECAPA-TDNN 后端在原始音频上 top1 为 `0.996 ± 0.004`，验证后端可靠；Base `L3` 重建语音 top1 为 `0.575 ± 0.042`，LCA `L3` 为 `0.345 ± 0.021`。

三用户通信 demo 中，客户端已能在真实路由通信时输出说话人识别字段；该 demo 当前使用轻量 MFCC 后端，适合作为工程链路验证，不作为论文级 speaker verification 主结果。

## 实验协议

数据集：`data/VCTK/wav48_silence_trimmed`

说话人数：110

随机种子：`41, 42, 43`

模型：SCIT-Speech Base 与 SCIT-Speech LCA v2

层数：`L=1/2/3`

### Experiment A: Codes/Latent Speaker Probe

每个说话人 5 条训练语音、5 条测试语音。特征包括：

- `codes_hist`：前 L 个 RVQ 层的 code ID 归一化直方图。
- `latent_stats`：前 L 层 `forward_feature()` 输出的 mean/std 池化特征。

分类器为标准化后的 logistic regression。该实验回答：SCIT-Speech 传输索引或内部特征中是否含有可恢复的说话人身份信息。

### Experiment B: ECAPA Speaker Preservation

每个说话人 3 条 enrollment、5 条 test。用原始 enrollment 音频建立 speaker profile，然后评估 original、Base 重建、LCA 重建的说话人匹配结果。

后端：SpeechBrain ECAPA `speechbrain/spkrec-ecapa-voxceleb`

主要指标：

- Top-1 speaker identification accuracy
- EER
- TAR@FAR=1%
- correct/impostor cosine margin

该实验回答：解码后的 waveform 是否仍能被强说话人识别模型判为同一说话人。

## 主结果 1：内部表征说话人可分性

随机基线：top1 约 `1/110 = 0.009`，top5 约 `5/110 = 0.045`。

| model | feature | L | top1 | top5 | macro-F1 |
|---|---|---:|---:|---:|---:|
| base | codes_hist | 1 | 0.344 ± 0.008 | 0.673 ± 0.017 | 0.325 ± 0.005 |
| base | codes_hist | 2 | 0.489 ± 0.028 | 0.809 ± 0.014 | 0.470 ± 0.031 |
| base | codes_hist | 3 | 0.510 ± 0.007 | 0.821 ± 0.009 | 0.493 ± 0.003 |
| base | latent_stats | 1 | 0.372 ± 0.019 | 0.688 ± 0.023 | 0.356 ± 0.019 |
| base | latent_stats | 2 | 0.558 ± 0.008 | 0.825 ± 0.007 | 0.542 ± 0.006 |
| base | latent_stats | 3 | 0.581 ± 0.019 | 0.827 ± 0.027 | 0.568 ± 0.022 |
| lca | codes_hist | 1 | 0.325 ± 0.011 | 0.646 ± 0.004 | 0.304 ± 0.016 |
| lca | codes_hist | 2 | 0.451 ± 0.014 | 0.783 ± 0.009 | 0.430 ± 0.019 |
| lca | codes_hist | 3 | 0.510 ± 0.034 | 0.807 ± 0.017 | 0.487 ± 0.033 |
| lca | latent_stats | 1 | 0.341 ± 0.009 | 0.645 ± 0.011 | 0.325 ± 0.010 |
| lca | latent_stats | 2 | 0.527 ± 0.023 | 0.803 ± 0.014 | 0.513 ± 0.025 |
| lca | latent_stats | 3 | 0.547 ± 0.041 | 0.799 ± 0.028 | 0.535 ± 0.044 |

解读：

- `L=2/3` 显著高于 `L=1`，说明传输更多 RVQ 层会携带更多说话人身份线索。
- `latent_stats` 通常强于 `codes_hist`，说明连续/池化表征比纯 code 直方图更容易被线性 probe 利用。
- Base 与 LCA 都远高于随机基线，因此模型表征不是说话人匿名的。

## 主结果 2：ECAPA 解码语音说话人保持

| model | L | top1 | verified | EER | TAR@FAR=1% | margin |
|---|---:|---:|---:|---:|---:|---:|
| original | 0 | 0.996 ± 0.004 | 1.000 ± 0.000 | 0.034 ± 0.005 | 0.938 ± 0.008 | 0.309 ± 0.003 |
| base | 1 | 0.108 ± 0.017 | 0.964 ± 0.010 | 0.767 ± 0.021 | 0.004 ± 0.006 | -0.116 ± 0.007 |
| base | 2 | 0.412 ± 0.036 | 0.996 ± 0.001 | 0.544 ± 0.021 | 0.046 ± 0.015 | -0.022 ± 0.009 |
| base | 3 | 0.575 ± 0.042 | 1.000 ± 0.000 | 0.442 ± 0.019 | 0.099 ± 0.018 | 0.018 ± 0.010 |
| lca | 1 | 0.075 ± 0.014 | 0.987 ± 0.007 | 0.810 ± 0.026 | 0.004 ± 0.002 | -0.140 ± 0.016 |
| lca | 2 | 0.233 ± 0.040 | 0.999 ± 0.001 | 0.666 ± 0.039 | 0.015 ± 0.006 | -0.080 ± 0.017 |
| lca | 3 | 0.345 ± 0.021 | 0.999 ± 0.001 | 0.599 ± 0.040 | 0.033 ± 0.013 | -0.046 ± 0.015 |

解读：

- ECAPA 在 original 上接近满分，说明说话人识别后端在本 split 上有效。
- Base 的 speaker preservation 随 L 增加明显增强，`L3` top1 达 `0.575 ± 0.042`，EER 降至 `0.442 ± 0.019`。
- LCA 也随 L 增加增强，但整体低于 Base，说明 LCA 的鲁棒/低负载适配可能牺牲了一部分说话人个性线索。
- 该结果支持“重建语音保留说话人身份”，但不支持“说话人完全不变”或“可作为高可靠身份认证系统”。

## 三用户通信 demo 验证

运行目录：`output/experiments/exp22_three_user_speaker_demo_smoke_20260708_seed42`

| client | sent | received | decoded | drops | speaker eval | correct | verified | speaker acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p225 | 35 | 70 | 70 | 0 | 32 | 14 | 32 | 0.438 |
| p226 | 35 | 70 | 70 | 0 | 32 | 17 | 32 | 0.531 |
| p227 | 35 | 70 | 70 | 0 | 32 | 19 | 32 | 0.594 |

demo 结论：

- 三用户通信链路已能输出 `spk`, `score`, `margin`, `verified` 与 summary CSV 统计字段。
- 该 demo 当前使用 MFCC 后端，准确率只作为“链路可用”证据。
- 若需要 demo 展示更强识别效果，下一步应把 demo 的 speaker backend 也切换为 ECAPA 或 x-vector。

## 实验完整性与异常记录

- 三种子 probe：`seed41/42/43` 已完成。
- 三种子 ECAPA preservation：`seed41/42/43` 已完成。
- ECAPA `seed41` 曾有一次漏传 `--seed 41` 的启动，发现 split 与 `seed42` 完全相同后已覆盖重跑；最终 JSON 已确认 `seed=41`。
- SpeechBrain ECAPA 在 Windows 上会在写完结果后返回 native teardown code `-1073740791`；每个有效 run 均以 `metrics/*.csv` 非空、日志含 `status=completed` 作为成功判据。
- 聚合 CSV 中除 mean/std 外，还提供小样本 95% t-interval half-width。

## 论文写法建议

可以写：

> We evaluate speaker identity along two axes: linear recoverability from transmitted/internal representations, and speaker preservation in decoded waveforms using a frozen ECAPA-TDNN verifier. On VCTK with 110 speakers and three random splits, speaker labels are recoverable from SCIT-Speech representations far above chance, and decoded speech remains substantially speaker-identifiable, especially at higher RVQ layers.

建议避免写：

- “模型完全识别说话人”。
- “该系统可用于身份认证”。
- “LCA 比 Base 更能保留说话人身份”。当前结果恰好相反，Base 在 ECAPA preservation 上更强。

## 关键文件

- `output/experiments/exp22_paper_grade_20260708/metrics/speaker_probe_multiseed_summary.csv`
- `output/experiments/exp22_paper_grade_20260708/metrics/speaker_identity_ecapa_multiseed_summary.csv`
- `output/experiments/exp22_paper_grade_20260708/reports/exp22_paper_grade_summary.md`
- `scripts/aggregate_exp22_speaker_identity.py`
- `scripts/evaluate_speaker_probe.py`
- `scripts/evaluate_speaker_identity.py`
