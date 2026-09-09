# 样本语音质量定量评估

- 实验编号：exp2_scit_speech_distill30_retrain_20260528_seed42
- 评估对象：`samples/fixed` 与 `samples/full_utterance` 中的原始语音和 L1/L2/L3 重建语音。
- 评估方式：逐条比较原始语音与重建语音，计算侵入式客观指标；未运行 ASR，因此不包含 WER/CER。

## 汇总结果

| 样本集 | 层数 | 样本数 | 码率 bps | 波形 L1 ↓ | Mel L1 ↓ | SI-SNR dB ↑ | 相关系数 ↑ | STOI ↑ | PESQ-WB ↑ | RMS 比例 dB | 裁剪比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_utterance | L1 | 4 | 500 | 0.033677 | 1.424405 | -12.190 | 0.3619 | 0.7338 | 1.203 | -3.866 | 0.000000 |
| full_utterance | L2 | 4 | 1000 | 0.027108 | 1.202103 | -5.400 | 0.5499 | 0.8085 | 1.487 | -1.972 | 0.000000 |
| full_utterance | L3 | 4 | 1500 | 0.025141 | 1.145732 | -3.961 | 0.5931 | 0.8292 | 1.618 | -1.522 | 0.000000 |

## 结论

- L3 仍是当前最佳层数，通常优于 L1/L2。
- 是否接受本阶段结果，应优先看相对实验二 best-dev 基线的 L3 PESQ-WB、STOI、SI-SNR 与 Mel L1 是否同时改善。
- 裁剪比例用于排查满幅削波；若为 0，听感问题更可能来自解码细节、噪声、相位或 token 信息不足。

## 数据文件

- 样本级明细 CSV：`output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\metrics\full_utterance_audio_quality_eval_bestdev.csv`
- 汇总 JSON：`output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\metrics\full_utterance_audio_quality_eval_bestdev_summary.json`
