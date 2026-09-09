# distill30 正式重训 epoch 10 中期检查

- 检查时间：2026-05-29
- run_id：`exp2_scit_speech_distill30_retrain_20260528_seed42`
- 配置：`distill_loss_lambda=30.0`
- 训练模式：formal，从头训练，不加载 `pretrained_generator_checkpoint`
- 训练集条数：27039
- batch_size：8
- 估算每 epoch batch step：3379
- 当前训练标量最新 step：36500，约 10.80 epoch
- 当前 dev 最新 step：35000，约 10.36 epoch

## 当前 checkpoint

可见 checkpoint：

| checkpoint | 时间 |
|---|---|
| `SpeechTokenizerTrainer_00020000` | 2026-05-29 05:08:03 |
| `SpeechTokenizerTrainer_00022500` | 2026-05-29 05:56:25 |
| `SpeechTokenizerTrainer_00025000` | 2026-05-29 06:44:42 |
| `SpeechTokenizerTrainer_00027500` | 2026-05-29 07:33:10 |
| `SpeechTokenizerTrainer_00030000` | 2026-05-29 08:21:12 |
| `SpeechTokenizerTrainer_00032500` | 2026-05-29 09:09:33 |
| `SpeechTokenizerTrainer_00035000` | 2026-05-29 09:57:45 |

`SpeechTokenizer_best_dev.pt` 更新时间为 2026-05-29 05:56:24，对应当前最佳 dev 点 step 22500。

## dev 曲线

| step | dev/mel error |
|---:|---:|
| 2500 | 5.222447 |
| 5000 | 4.797072 |
| 7500 | 4.522492 |
| 10000 | 4.303859 |
| 12500 | 4.082685 |
| 15000 | 3.915208 |
| 17500 | 3.806236 |
| 20000 | 3.649731 |
| 22500 | 3.475089 |
| 25000 | 3.822558 |
| 27500 | 3.534792 |
| 30000 | 3.640551 |
| 32500 | 3.763602 |
| 35000 | 3.549350 |

当前最佳：step 22500，`dev/mel error=3.475089`。

## 与实验二基线对比

| 对比项 | 实验二基线 | distill30 正式重训 | 差值 |
|---|---:|---:|---:|
| 同 step 22500 dev/mel error | 3.867209 | 3.475089 | -0.392120 |
| 同 step 35000 dev/mel error | 3.715640 | 3.549350 | -0.166291 |
| 截至 35000 的最佳 dev/mel error | 3.703312 | 3.475089 | -0.228223 |
| 实验二全程 best-dev | 3.470752 | 3.475089 | +0.004337 |

distill30 在前 10 个 epoch 明显快于原实验二；当前 best-dev 已经基本贴近实验二全程 best-dev，但还没有明确超过。

## 训练标量摘要

| 指标 | 当前/最佳 | 数值 |
|---|---|---:|
| train/mel error 最新 | step 36500 | 1.109498 |
| train/mel error 最低 | step 26700 | 0.895387 |
| train/generator loss 最新 | step 36500 | 113.706154 |
| train/generator loss 最低 | step 26700 | 90.531471 |
| train/distillation loss 最新 | step 36500 | 0.475324 |
| train/distillation loss 最低 | step 36000 | 0.448301 |
| train/learning_rate 最新 | step 36500 | 9.2213e-5 |

## 中期判断

这是一个正向中期结果：`distill_loss_lambda=30` 的正式重训在前 10 个 epoch 内显著快于原实验二，并且当前 best-dev 已接近原实验二全程最佳点。

但 step 22500 后 dev/mel 有波动，35000 时仍未刷新 best。现在不建议中断训练；建议继续观察 37500、40000、42500、45000 几个验证点。如果 45000 附近仍没有超过 3.475089，则先保留当前 best-dev，再做样本级 PESQ/STOI/SI-SNR/Mel L1 评估。

本次未导出样本评估，原因是训练进程仍在运行，避免抢占 GPU 干扰训练。
