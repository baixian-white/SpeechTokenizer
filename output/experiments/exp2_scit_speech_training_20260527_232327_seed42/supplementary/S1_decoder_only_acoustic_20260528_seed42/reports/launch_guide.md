# S1 decoder-only acoustic finetune 启动指南

## 目的

从实验二 best-dev checkpoint 出发，冻结 encoder 与 RVQ/codebook，只训练 decoder 和判别器，验证听感质量是否能在不改变 token 表示的前提下提升。

## 启动命令

```powershell
accelerate launch scripts/train_decoder_only_finetune.py --config output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42\configs\decoder_only_acoustic_config.json
```

## 关键文件

- 运行目录：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42`
- 配置文件：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42\configs\decoder_only_acoustic_config.json`
- 命令文件：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42\commands\run_decoder_only_acoustic_finetune.txt`
- 训练日志：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42\logs`
- checkpoint 输出：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42\checkpoints`

## 训练完成后的建议评估

1. 选取 finetune run 中的 best-dev checkpoint 或最终 checkpoint。
2. 复用实验二固定样本与完整句子样本，生成 L1/L2/L3 重建语音。
3. 运行样本级 PESQ-WB、STOI、SI-SNR、Mel L1 评估。
4. 将结果追加到 `exp2_supplementary_experiments.md` 的“后续追加记录”小节。
