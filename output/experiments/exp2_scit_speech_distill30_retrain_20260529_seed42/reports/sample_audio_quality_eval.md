# 样本语音质量定量评估

- 实验编号：exp2_scit_speech_distill30_retrain_20260529_seed42
- 评估对象：`samples/fixed` 与 `samples/full_utterance` 中的原始语音和 L1/L2/L3 重建语音。
- 评估方式：逐条比较原始语音与重建语音，计算侵入式客观指标；未运行 ASR，因此不包含 WER/CER。

## 汇总结果

| 样本集 | 层数 | 样本数 | 码率 bps | 波形 L1 ↓ | Mel L1 ↓ | SI-SNR dB ↑ | 相关系数 ↑ | STOI ↑ | PESQ-WB ↑ | RMS 比例 dB | 裁剪比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | L1 | 8 | 500 | 0.028960 | 1.179168 | -7.753 | 0.4257 | 0.7861 | 1.385 | -3.700 | 0.000000 |
| fixed | L2 | 8 | 1000 | 0.023187 | 0.924071 | -3.002 | 0.5992 | 0.8420 | 1.611 | -1.431 | 0.000000 |
| fixed | L3 | 8 | 1500 | 0.021153 | 0.861874 | -0.588 | 0.6730 | 0.8770 | 1.823 | -0.700 | 0.000000 |
| full_utterance | L1 | 4 | 500 | 0.032032 | 1.250304 | -6.689 | 0.4665 | 0.7810 | 1.296 | -3.325 | 0.000000 |
| full_utterance | L2 | 4 | 1000 | 0.025090 | 0.970630 | -2.462 | 0.6216 | 0.8394 | 1.677 | -1.474 | 0.000000 |
| full_utterance | L3 | 4 | 1500 | 0.022815 | 0.906845 | -0.681 | 0.6714 | 0.8636 | 1.889 | -0.908 | 0.000000 |

## 结论

- L3 仍是当前最佳层数，通常优于 L1/L2。
- 是否接受本阶段结果，应优先看相对实验二 best-dev 基线的 L3 PESQ-WB、STOI、SI-SNR 与 Mel L1 是否同时改善。
- 裁剪比例用于排查满幅削波；若为 0，听感问题更可能来自解码细节、噪声、相位或 token 信息不足。

## 数据文件

- 样本级明细 CSV：`output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\metrics\sample_audio_quality_eval.csv`
- 汇总 JSON：`output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\metrics\sample_audio_quality_eval_summary.json`
