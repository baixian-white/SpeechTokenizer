# S2-B 语义蒸馏权重 ablation 启动指南

## 目的

从实验二 best-dev checkpoint 出发，调整 `distill_loss_lambda`，验证语义蒸馏权重是否过度挤压声学重建质量。

## 当前变体

- 变体：`S2-B`
- 蒸馏权重：`distill_loss_lambda=60.0`
- 训练范围：generator 全量可训练，判别器正常训练。
- 对照对象：实验二 best-dev 基线样本指标。

## 启动命令

```powershell
accelerate launch scripts/train_distill_weight_ablation.py --config output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_B_distill60_20260528_seed42\configs\distill_weight_ablation_config.json
```

## 关键文件

- 运行目录：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_B_distill60_20260528_seed42`
- 配置文件：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_B_distill60_20260528_seed42\configs\distill_weight_ablation_config.json`
- 命令文件：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_B_distill60_20260528_seed42\commands\run_distill_weight_ablation.txt`
- 训练日志：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_B_distill60_20260528_seed42\logs`
- checkpoint 输出：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_B_distill60_20260528_seed42\checkpoints`

## 训练完成后的评估

1. 使用本 run 的 `checkpoints/SpeechTokenizer_best_dev.pt` 或最后一个 checkpoint 导出 fixed/full_utterance 样本。
2. 运行 `scripts/evaluate_sample_audio_quality.py`，与实验二 best-dev 基线做同口径对比。
3. 优先判定 L3 的 PESQ-WB、STOI、SI-SNR 与 Mel L1。
4. 将结果追加到 `exp2_supplementary_experiments.md`。
