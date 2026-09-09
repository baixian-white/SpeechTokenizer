# 实验二总结

- run_id: exp2_scit_speech_training_20260527_232327_seed42
- 状态: completed
- 运行模式: formal
- 配置文件: output\experiments\exp2_scit_speech_training_20260527_232327_seed42\configs\scit_speech_base_config.json
- checkpoint: available
- loss 曲线: available

## 实验一交接信息
- candidate_id: nas_seed42_000896
- encoder_strides: [5, 4, 4, 4]
- handoff_schema: encoder_only_nas_v1
- decoder_condition_note: 实验一中的 `frozen_teacher_decoder` 仅作为交接来源说明；实验二中的 decoder 正常训练。

## 产物可用性
- layer_reconstruction: yes (output\experiments\exp2_scit_speech_training_20260527_232327_seed42\metrics\layer_reconstruction.json)
- codebook_usage: yes (output\experiments\exp2_scit_speech_training_20260527_232327_seed42\metrics\codebook_usage.json)
- loss_curves: yes (output\experiments\exp2_scit_speech_training_20260527_232327_seed42\metrics\loss_curves.csv)
- sample_audio_quality_eval: yes (output\experiments\exp2_scit_speech_training_20260527_232327_seed42\metrics\sample_audio_quality_eval.csv)
- exp2_supplementary_experiments: yes (output\experiments\exp2_scit_speech_training_20260527_232327_seed42\reports\exp2_supplementary_experiments.md)

## 样本语音质量定量评估

`samples/fixed` 与 `samples/full_utterance` 下的语音样本由 best-dev checkpoint 预先生成。下表为原始语音与重建语音的侵入式客观指标；这里未运行 ASR，因此不包含 WER/CER。

| 样本集 | 层数 | 样本数 | 码率 bps | 波形 L1 ↓ | Mel L1 ↓ | SI-SNR dB ↑ | 相关系数 ↑ | STOI ↑ | PESQ-WB ↑ | RMS 比例 dB | 裁剪比例 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | L1 | 8 | 500 | 0.033071 | 1.482141 | -17.334 | 0.1945 | 0.7414 | 1.185 | -5.436 | 0.000000 |
| fixed | L2 | 8 | 1000 | 0.026872 | 1.103436 | -6.649 | 0.4652 | 0.8232 | 1.504 | -2.669 | 0.000000 |
| fixed | L3 | 8 | 1500 | 0.025561 | 1.051497 | -4.042 | 0.5483 | 0.8512 | 1.655 | -1.330 | 0.000000 |
| full_utterance | L1 | 4 | 500 | 0.036629 | 1.450744 | -14.578 | 0.2669 | 0.7298 | 1.164 | -4.682 | 0.000000 |
| full_utterance | L2 | 4 | 1000 | 0.029362 | 1.143400 | -7.507 | 0.5025 | 0.8034 | 1.427 | -2.243 | 0.000000 |
| full_utterance | L3 | 4 | 1500 | 0.026936 | 1.075713 | -4.922 | 0.5645 | 0.8291 | 1.607 | -1.426 | 0.000000 |

- 样本级明细 CSV: output\experiments\exp2_scit_speech_training_20260527_232327_seed42\metrics\sample_audio_quality_eval.csv
- 汇总 JSON: output\experiments\exp2_scit_speech_training_20260527_232327_seed42\metrics\sample_audio_quality_eval_summary.json

## 范围说明
- 本报告不编造训练结果、loss 数值、WER/CER 或 checkpoint 来源。
- 上表中的 STOI 与 PESQ-WB 仅针对已生成样本音频对计算，不代表完整验证集或通信信道评估结果。
- `evaluate_layer_reconstruction.py` 与 `codebook_usage_report.py` 仅生成 sanity/proxy 诊断结果。
