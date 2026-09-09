# 实验二补充实验记录

- 主实验编号：exp2_scit_speech_training_20260527_232327_seed42
- 建档日期：2026-05-28
- 主 checkpoint：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\checkpoints\SCIT-Speech-Base_best.pt`
- 基准评估：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\reports\sample_audio_quality_eval.md`
- 本文用途：记录实验二之后围绕听感质量提升开展的所有补充实验，包括动机、配置、结果、结论和下一步决策。

## 当前问题

实验二 best-dev checkpoint 的语义可懂度已经有基础，但听感质量仍明显偏弱。完整句子样本上，L3 达到 `STOI=0.8291`、`PESQ-WB=1.607`、`SI-SNR=-4.922 dB`；固定 1 秒片段上，L3 达到 `STOI=0.8512`、`PESQ-WB=1.655`、`SI-SNR=-4.042 dB`。

这说明模型已经能保留一部分语义和可懂度，但重建语音的自然度、音色、细节和信号保真度仍不足。后续补充实验优先解决“同样的离散 token 如何解码得更自然”，其次再处理 codebook 利用率和训练目标权衡。

## 关键诊断

- L3 是当前最佳层数，优于 L1/L2；L1 到 L2 是主要提升，L2 到 L3 仍有小幅收益。
- PESQ-WB 绝对值偏低，说明听感质量仍有明显提升空间。
- 所有样本裁剪比例为 0，当前问题不是削波。
- layer 1 codebook 使用率偏低：`usage_rate=0.175781`，`dead_code_ratio=0.824219`。L2/L3 使用率约 95%，问题集中在第一层。
- 手动重置较大学习率后，`dev/mel error` 明显回退；补充实验应避免大 LR 续训。

## 优先实验路线

### S1：decoder-only acoustic finetune

目标：冻结 encoder 与 RVQ/codebook，只微调 decoder 以及必要的判别器，让已有离散 token 解码出的语音更自然。

建议设置：

- 起点：实验二 best-dev checkpoint。
- 冻结：encoder、quantizer、codebooks。
- 训练：decoder；判别器是否训练作为子实验开关。
- 学习率：`1e-6 ~ 5e-6`。
- 蒸馏权重：`distill_loss_lambda=0` 或很小。
- early stop：优先看 PESQ-WB、STOI、Mel L1、SI-SNR，不以 distillation loss 作为主要停止条件。

预期：在不改变 token 表示的前提下提升 PESQ-WB、SI-SNR 和主观听感。若有效，说明主要瓶颈在声学解码器；若无效，说明 token 本身或 codebook 信息不足。

### S2：语义蒸馏权重 ablation

目标：验证当前 `distill_loss_lambda=120` 是否过度挤压声学重建质量。

建议对比：

| 变体 | distill_loss_lambda | 说明 |
|---|---:|---|
| S2-A | 120 | 当前基线 |
| S2-B | 60 | 中等蒸馏 |
| S2-C | 30 | 弱蒸馏 |
| S2-D | decay 到 0-30 | 训练后段逐步让声学目标主导 |

判定重点：如果降低蒸馏权重后 PESQ-WB、Mel L1、SI-SNR 改善而 STOI 不明显下降，则后续正式训练应采用较低或退火式 distill 权重。

### S3：L1 codebook 利用率修复

目标：解决第一层 codebook 大量死码问题，提升低码率层与分层结构质量。

候选策略：

- dead code reinit。
- k-means 初始化 codebook。
- codebook usage / entropy regularization。
- 调整 commitment loss，或按层设置不同 commitment 权重。
- 训练早期先稳定 L1，再逐步启用 L2/L3。

判定重点：L1 usage rate 是否提升、dead code ratio 是否下降，以及 L1/L2/L3 的 PESQ-WB、STOI 是否同步改善。

### S4：增加声学频谱损失

目标：补足 mel loss 对听感细节约束不足的问题。

候选损失：

- multi-resolution STFT loss。
- spectral convergence loss。
- log magnitude loss。

判定重点：PESQ-WB 和主观听感是否改善；同时观察 SI-SNR、Mel L1 是否稳定改善，避免只优化频谱指标但产生伪影。

### S5：更长训练片段

目标：缓解 `segment_size=16000` 只训练 1 秒片段带来的长句一致性不足。

候选设置：

- 2 秒片段：`segment_size=32000`。
- 3 秒片段：`segment_size=48000`。

判定重点：完整句子样本上的 PESQ-WB、STOI、SI-SNR 是否优于 1 秒基线；同时记录显存、训练速度和稳定性。

## 当前推荐执行顺序

1. 先做 S1 decoder-only acoustic finetune，成本低、风险小、目标直接。
2. 并行或随后做 S2 distill 权重 ablation，确定语义蒸馏和声学质量的权衡点。
3. 若 S1 有效但 L1 仍差，推进 S3 codebook 修复。
4. 若 S1 提升有限，优先尝试 S4 频谱损失，再考虑 S5 更长片段。

## 统一评估口径

每个补充实验至少记录以下指标：

| 指标 | 方向 | 用途 |
|---|---:|---|
| PESQ-WB | 越高越好 | 听感质量代理指标 |
| STOI | 越高越好 | 可懂度代理指标 |
| SI-SNR | 越高越好 | 信号保真度 |
| Mel L1 | 越低越好 | 频谱重建差异 |
| wave L1 / RMSE | 越低越好 | 波形误差 |
| corr | 越高越好 | 波形相关性 |
| codebook usage / dead code ratio | usage 越高、dead 越低越好 | codebook 健康度 |
| 主观听感备注 | 定性 | 记录噪声、闷、金属感、失真、断裂等现象 |

固定样本集：

- `samples/fixed`
- `samples/full_utterance`

必要时额外记录困难样本，例如 `1737-148989-0006`。

## 实验记录模板

### Sx-y：实验名称

- 日期：
- 目的：
- 起点 checkpoint：
- 代码/配置变更：
- 训练命令：
- 训练范围：
- 学习率：
- 冻结模块：
- loss 权重：
- 训练步数：
- early stop 标准：
- 输出目录：
- 生成样本目录：
- 定量结果：

| 样本集 | L | n | PESQ-WB | STOI | SI-SNR dB | Mel L1 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| fixed | L1 |  |  |  |  |  |  |
| fixed | L2 |  |  |  |  |  |  |
| fixed | L3 |  |  |  |  |  |  |
| full_utterance | L1 |  |  |  |  |  |  |
| full_utterance | L2 |  |  |  |  |  |  |
| full_utterance | L3 |  |  |  |  |  |  |

- 主观听感：
- 与实验二 best-dev 基线对比：
- 结论：
- 下一步：

## 决策标准

补充实验优先进入下一轮的条件：

- L3 PESQ-WB 有稳定提升，且 STOI 不显著下降。
- full_utterance 指标优先级高于 fixed 1 秒片段。
- 听感主观改善必须与至少一个客观指标改善一致。
- 若 PESQ-WB 提升但 STOI 明显下降，需要单独标记为“声学改善但语义受损”，不能直接作为主线。
- 若 codebook 修复导致 L1 usage 提升但 L3 听感不变，需要继续检查分层解码是否真正使用了低层信息。

## 后续追加记录

后续所有实验二补充实验记录统一追加到本节之后，按 `S1-*`、`S2-*`、`S3-*` 编号。

### S1-0：decoder-only acoustic finetune 代码准备

- 日期：2026-05-28
- 状态：代码已编写，训练尚未启动。
- 目的：为 S1 decoder-only acoustic finetune 建立可复用的准备脚本、训练入口和启动记录，后续实验结果继续追加到本文档。

#### 编写内容

新增代码文件：

- `scripts/exp2_supplementary.py`
  - 提供补充实验通用逻辑。
  - `prepare_decoder_only_acoustic_run(...)`：从实验二基线配置派生 decoder-only acoustic finetune 配置，创建运行目录、命令文件和启动指南。
  - `apply_generator_train_scope(...)`：应用 `decoder_only_acoustic` 冻结策略，冻结 `encoder`、`quantizer`、`transform`，仅保留 `decoder` 可训练。
  - `load_generator_checkpoint(...)`：从 generator-only 或 trainer checkpoint 加载模型权重。

- `scripts/prepare_exp2_supplementary.py`
  - 命令行准备入口。
  - 当前支持 `decoder-only` 子命令，用于生成 S1 运行目录。

- `scripts/train_decoder_only_finetune.py`
  - S1 训练入口。
  - 从 `pretrained_generator_checkpoint` 加载实验二 best-dev checkpoint。
  - 应用 decoder-only 冻结策略。
  - 写出 `reports/freeze_report.json`，记录冻结模块、可训练参数量和冻结参数量。
  - 使用现有 `SpeechTokenizerTrainer` 执行训练。

新增测试文件：

- `tests/test_exp2_supplementary.py`
  - 验证 decoder-only 冻结后只有 decoder 参数可训练。
  - 验证配置派生会记录 checkpoint、学习率、distill 权重、运行命令和启动指南。
  - 验证短实验的 `save_model_steps` 不超过 `max_train_steps`，避免短跑不保存 checkpoint。

#### 已生成的 S1 运行目录

- 运行目录：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42`
- 配置文件：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42\configs\decoder_only_acoustic_config.json`
- 启动命令文件：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42\commands\run_decoder_only_acoustic_finetune.txt`
- 启动指南：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42\reports\launch_guide.md`

#### S1 当前配置摘要

| 项目 | 值 |
|---|---|
| 起点 checkpoint | `output\experiments\exp2_scit_speech_training_20260527_232327_seed42\checkpoints\SCIT-Speech-Base_best.pt` |
| finetune_scope | `decoder_only_acoustic` |
| 冻结模块 | `encoder`, `quantizer`, `transform` |
| 可训练模块 | `decoder` |
| learning_rate | `3e-6` |
| distill_loss_lambda | `0` |
| epochs | `3` |
| max_train_steps | `1200` |
| save_model_steps | `600` |
| valid_num_workers | `0` |

#### 启动方式

在项目根目录 `H:\H-CODE\speechtokenizer` 下启动：

```powershell
accelerate launch scripts/train_decoder_only_finetune.py --config output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42\configs\decoder_only_acoustic_config.json
```

也可以直接读取命令文件执行：

```powershell
Get-Content output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42\commands\run_decoder_only_acoustic_finetune.txt
```

如果当前 shell 没有激活 `speechtokenizer` 环境，先进入对应环境，或者使用该环境里的 Python/accelerate。训练脚本会在启动时加载 best-dev checkpoint 并写出冻结报告。

#### 训练后评估建议

训练完成后，优先检查：

1. `reports/freeze_report.json`：确认只有 decoder 可训练。
2. `checkpoints/SpeechTokenizer_best_dev.pt` 或最新 `SpeechTokenizerTrainer_*`：确认 finetune 有 checkpoint 产物。
3. 复用实验二样本评估口径生成 L1/L2/L3 语音。
4. 重新计算 `PESQ-WB`、`STOI`、`SI-SNR`、`Mel L1`，并与实验二 best-dev 基线对比。

#### 验证情况

已执行：

```powershell
& 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe' -m unittest discover -s tests -p test_exp2_supplementary.py
```

结果：`Ran 2 tests ... OK`。

### S1-1：decoder-only acoustic finetune 第一阶段结果检查

- 日期：2026-05-28
- 状态：第一阶段已跑完并完成同口径样本评估。
- 运行目录：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42`
- 评估报告：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42\reports\sample_audio_quality_eval.md`
- 样本级明细：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42\metrics\sample_audio_quality_eval.csv`
- 汇总 JSON：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42\metrics\sample_audio_quality_eval_summary.json`
- TensorBoard 标量摘要：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S1_decoder_only_acoustic_20260528_seed42\metrics\tensorboard_scalar_summary.json`

#### 训练产物与冻结检查

本次 S1 确认只训练 decoder：

| 模块 | 参数量 | 可训练参数量 | 状态 |
|---|---:|---:|---|
| encoder | 8,873,360 | 0 | 冻结 |
| quantizer | 0 | 0 | 冻结 |
| transform | 787,200 | 0 | 冻结 |
| decoder | 35,182,530 | 35,182,530 | 可训练 |

可见 checkpoint：

- `checkpoints\SpeechTokenizer_best_dev.pt`
- `checkpoints\SpeechTokenizerTrainer_00000600`

未看到 `SpeechTokenizerTrainer_00001200`。TensorBoard 训练标量记录到 step 1100，dev 标量只在 step 600 出现一次，因此本轮验证点偏少。

#### 训练曲线摘要

| 指标 | step | 数值 | 备注 |
|---|---:|---:|---|
| 实验二 best-dev `dev/mel error` | 45000 | 3.470752 | 主实验基线最佳点 |
| S1 `dev/mel error` | 600 | 3.583195 | 比基线高 0.112444，约 +3.24% |
| S1 `dev/distillation loss` | 600 | 0.590310 | 本轮 `distill_loss_lambda=0`，只作观测 |
| S1 `train/mel error` 最低 | 200 | 0.942020 | 训练集短片段最低点 |
| S1 `train/mel error` 最后 | 1100 | 1.022162 | 后段没有继续下降 |
| S1 `train/generator loss` 最低 | 600 | 92.039925 | 之后回升到 103.183907 |
| S1 `train/learning_rate` 首次/末次 | 0 / 1100 | 3e-6 / 5.01e-8 | 短跑内学习率基本衰减到 0 |

#### 样本级定量结果

下表为 S1 第一阶段结果，括号中为相对实验二 best-dev 基线的变化量。PESQ-WB、STOI、SI-SNR 越高越好；Mel L1 越低越好。

| 样本集 | L | n | PESQ-WB | STOI | SI-SNR dB | Mel L1 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| fixed | L1 | 8 | 1.186 (+0.001) | 0.7511 (+0.0098) | -19.897 (-2.563) | 1.476695 (-0.005446) | 未达显著改善 |
| fixed | L2 | 8 | 1.472 (-0.032) | 0.8258 (+0.0026) | -8.510 (-1.861) | 1.096674 (-0.006762) | 退化 |
| fixed | L3 | 8 | 1.643 (-0.011) | 0.8482 (-0.0029) | -4.381 (-0.340) | 1.019605 (-0.031892) | 未达显著改善 |
| full_utterance | L1 | 4 | 1.153 (-0.011) | 0.7332 (+0.0033) | -15.556 (-0.977) | 1.432305 (-0.018438) | 未达显著改善 |
| full_utterance | L2 | 4 | 1.420 (-0.007) | 0.8044 (+0.0010) | -7.401 (+0.106) | 1.129259 (-0.014141) | 未达显著改善 |
| full_utterance | L3 | 4 | 1.591 (-0.016) | 0.8313 (+0.0021) | -4.794 (+0.128) | 1.054883 (-0.020830) | 未达显著改善 |

裁剪比例全部为 0，说明没有新增满幅削波问题。

#### 结论

S1 第一阶段不建议作为主线改进结果接受。它在 Mel L1 上有小幅下降，说明 decoder-only finetune 确实让频谱距离略微贴近原音；但 PESQ-WB 没有提升，full_utterance L3 反而从 1.607 降到 1.591，fixed L3 从 1.655 降到 1.643。对“听感质量改善”这个目标来说，本轮没有通过。

这轮更像是一次弱负结果：不是全面崩掉，但没有把关键听感代理指标推上去。训练配置上也有明显问题：`max_train_steps=1200` 搭配当前学习率调度，使学习率在短跑内从 `3e-6` 衰减到接近 0，且只有 step 600 一个 dev 点，难以判断最佳停点。

#### 下一步建议

不要基于 S1 第一阶段 checkpoint 继续作为主线。后续优先从实验二 best-dev checkpoint 直接做 S2 蒸馏权重 ablation，验证 `distill_loss_lambda=120` 是否压制了声学质量。

如果仍要继续 decoder-only 路线，建议单独开 `S1-b`，不要沿用当前短跑调度：

- `learning_rate=1e-6`
- `max_train_steps=3000` 或更长
- `save_model_steps=500`
- 增加 dev 评估点，至少保留 step 500/1000/1500/2000/2500/3000
- 避免短跑内 cosine 学习率直接衰减到 0，或改成较小学习率的近似 constant 续训

当前主线判断：S1 第一阶段记录为“未达显著改善”，下一步进入 S2 更划算。

### S2-0：语义蒸馏权重 ablation 代码准备

- 日期：2026-05-28
- 状态：代码已编写，S2-B/S2-C/S2-D 三个运行目录已准备，训练尚未启动。
- 目的：从实验二 best-dev checkpoint 出发，降低或调度 `distill_loss_lambda`，验证原始 `distill_loss_lambda=120` 是否过度挤压声学重建质量。

#### 编写内容

新增/修改代码文件：

- `scripts/exp2_supplementary.py`
  - 新增 `DISTILL_WEIGHT_ABLATION_SCOPE = "distill_weight_ablation"`。
  - 新增 `prepare_distill_weight_ablation_run(...)`，用于生成单个 S2 变体的配置、命令、启动指南和 preparation 记录。
  - `apply_generator_train_scope(...)` 新增 S2 支持：S2 不冻结 generator，`encoder`、`quantizer`、`transform`、`decoder` 全部可训练；判别器按原训练逻辑正常训练。
  - 支持 `distill_loss_schedule`，当前实现 `linear_decay`。

- `scripts/prepare_exp2_supplementary.py`
  - 新增 `distill-ablation` 子命令：准备单个 S2 变体。
  - 新增 `distill-ablation-suite` 子命令：默认一次准备 S2-B/S2-C/S2-D。

- `scripts/train_distill_weight_ablation.py`
  - S2 训练入口。
  - 从 `pretrained_generator_checkpoint` 加载实验二 best-dev checkpoint。
  - 确认 `finetune_scope=distill_weight_ablation`。
  - 写出 `reports/train_scope_report.json`，记录 generator 全量可训练情况。
  - 使用现有 `SpeechTokenizerTrainer` 训练。

- `speechtokenizer/trainer/trainer.py`
  - 新增 `resolve_distill_loss_lambda(...)`。
  - 训练时如果配置中存在 `distill_loss_schedule`，每个 step 动态计算当前蒸馏权重。
  - TensorBoard 额外记录 `train/distillation lambda`。

- `tests/test_exp2_supplementary.py`
  - 新增 S2 静态权重配置测试。
  - 新增 S2-D 线性衰减配置测试。
  - 新增蒸馏权重调度解析测试。

#### 已准备的 S2 运行目录

| 变体 | distill 设置 | 运行目录 |
|---|---|---|
| S2-B | `distill_loss_lambda=60` | `output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_B_distill60_20260528_seed42` |
| S2-C | `distill_loss_lambda=30` | `output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_C_distill30_20260528_seed42` |
| S2-D | `distill_loss_lambda: 60 -> 30 linear_decay, decay_steps=3000` | `output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_D_distill60to30_20260528_seed42` |

三者公共设置：

| 项目 | 值 |
|---|---|
| 起点 checkpoint | `output\experiments\exp2_scit_speech_training_20260527_232327_seed42\checkpoints\SCIT-Speech-Base_best.pt` |
| finetune_scope | `distill_weight_ablation` |
| 冻结模块 | 无 |
| 可训练模块 | `encoder`, `quantizer`, `transform`, `decoder` |
| learning_rate | `1e-5` |
| epochs | `3` |
| max_train_steps | `3000` |
| save_model_steps | `500` |
| valid_num_workers | `0` |

#### 启动方式

在项目根目录 `H:\H-CODE\speechtokenizer` 下，建议先跑 S2-C，再跑 S2-B，最后跑 S2-D。S2-C 成本最低、信息最直接：如果 `distill_loss_lambda=30` 的 PESQ/STOI 明显改善，就说明原始蒸馏权重偏大。

S2-C：

```powershell
accelerate launch scripts/train_distill_weight_ablation.py --config output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_C_distill30_20260528_seed42\configs\distill_weight_ablation_config.json
```

S2-B：

```powershell
accelerate launch scripts/train_distill_weight_ablation.py --config output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_B_distill60_20260528_seed42\configs\distill_weight_ablation_config.json
```

S2-D：

```powershell
accelerate launch scripts/train_distill_weight_ablation.py --config output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_D_distill60to30_20260528_seed42\configs\distill_weight_ablation_config.json
```

也可以直接读取各自目录下的命令文件：

- `S2_B_distill60_20260528_seed42\commands\run_distill_weight_ablation.txt`
- `S2_C_distill30_20260528_seed42\commands\run_distill_weight_ablation.txt`
- `S2_D_distill60to30_20260528_seed42\commands\run_distill_weight_ablation.txt`

#### 训练完成后的评估方式

每个 S2 变体训练完成后，使用该 run 的 `checkpoints\SpeechTokenizer_best_dev.pt` 导出同一批 fixed/full_utterance 样本，然后运行统一评估脚本：

```powershell
& 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe' scripts\evaluate_sample_audio_quality.py --run-dir <S2_RUN_DIR> --config <S2_RUN_DIR>\configs\distill_weight_ablation_config.json --baseline-summary output\experiments\exp2_scit_speech_training_20260527_232327_seed42\metrics\sample_audio_quality_eval_summary.json
```

判定标准沿用 S1 后的口径：L3 PESQ-WB 必须有实质提升，STOI 不能明显下降；full_utterance 优先于 fixed 1 秒片段。

#### 验证情况

已执行红绿测试流程：新增测试先因缺少 S2 函数和调度解析失败，补实现后通过。

```powershell
& 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe' -m unittest discover -s tests -p test_exp2_supplementary.py
```

结果：`Ran 5 tests ... OK`。

### S2-C：`distill_loss_lambda=30` 第一阶段结果检查

- 日期：2026-05-28
- 状态：S2-C 已跑完并完成同口径样本评估；S2-B/S2-D 当前仍只有配置，未看到 checkpoint。
- 运行目录：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_C_distill30_20260528_seed42`
- 评估报告：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_C_distill30_20260528_seed42\reports\sample_audio_quality_eval.md`
- 样本级明细：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_C_distill30_20260528_seed42\metrics\sample_audio_quality_eval.csv`
- 汇总 JSON：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_C_distill30_20260528_seed42\metrics\sample_audio_quality_eval_summary.json`
- TensorBoard 标量摘要：`output\experiments\exp2_scit_speech_training_20260527_232327_seed42\supplementary\S2_C_distill30_20260528_seed42\metrics\tensorboard_scalar_summary.json`

#### 训练产物与训练范围检查

可见 checkpoint：

- `checkpoints\SpeechTokenizer_best_dev.pt`
- `checkpoints\SpeechTokenizerTrainer_00000500`
- `checkpoints\SpeechTokenizerTrainer_00001000`
- `checkpoints\SpeechTokenizerTrainer_00001500`
- `checkpoints\SpeechTokenizerTrainer_00002000`
- `checkpoints\SpeechTokenizerTrainer_00002500`

未看到 `SpeechTokenizerTrainer_00003000`，原因与当前 trainer 停止逻辑有关：到达 `max_train_steps=3000` 前，最后一个可见验证/保存点是 2500；训练标量记录到 step 2900。

训练范围报告显示：

| 模块 | 参数量 | 可训练参数量 | 状态 |
|---|---:|---:|---|
| encoder | 8,873,360 | 8,873,360 | 可训练 |
| quantizer | 0 | 0 | 无可训练参数记录 |
| transform | 787,200 | 787,200 | 可训练 |
| decoder | 35,182,530 | 35,182,530 | 可训练 |

generator 可训练参数合计 `44,843,090`。`quantizer` 在 `.parameters()` 统计中为 0，因此报告里显示为无可训练参数记录；这不是额外冻结造成的。

#### 训练曲线摘要

| 指标 | step | 数值 | 备注 |
|---|---:|---:|---|
| 实验二 best-dev `dev/mel error` | 45000 | 3.470752 | 主实验基线最佳点 |
| S2-C `dev/mel error` 最低 | 2000 | 3.460102 | 比基线低 0.010649，约 -0.31% |
| S2-C `dev/mel error` 最后 | 2500 | 3.691823 | 后段回退 |
| S2-C `dev/distillation loss` 最低/最高 | 500 / 2500 | 0.592286 / 0.594196 | 基本稳定 |
| S2-C `train/mel error` 最低 | 1500 | 0.847469 | 训练短片段最低点 |
| S2-C `train/generator loss` 最低 | 1500 | 86.907654 | 之后回升 |
| S2-C `train/distillation lambda` | 0-2900 | 30.0 | 静态权重正确 |
| S2-C `train/learning_rate` 首次/末次 | 0 / 2900 | 1e-5 / 2.68e-8 | 短跑末端接近 0 |

本轮 best-dev checkpoint 的时间戳对应 step 2000，和 `dev/mel error` 最低点一致。

#### 样本级定量结果

下表为 S2-C 第一阶段结果，括号中为相对实验二 best-dev 基线的变化量。PESQ-WB、STOI、SI-SNR 越高越好；Mel L1 越低越好。

| 样本集 | L | n | PESQ-WB | STOI | SI-SNR dB | Mel L1 | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| fixed | L1 | 8 | 1.202 (+0.018) | 0.7595 (+0.0181) | -18.785 (-1.451) | 1.390247 (-0.091893) | 未达显著改善 |
| fixed | L2 | 8 | 1.517 (+0.013) | 0.8300 (+0.0068) | -6.274 (+0.375) | 1.074631 (-0.028805) | 未达显著改善 |
| fixed | L3 | 8 | 1.702 (+0.047) | 0.8587 (+0.0075) | -3.569 (+0.472) | 1.021147 (-0.030350) | 通过 |
| full_utterance | L1 | 4 | 1.190 (+0.026) | 0.7424 (+0.0126) | -12.064 (+2.514) | 1.369337 (-0.081406) | 未达显著改善 |
| full_utterance | L2 | 4 | 1.496 (+0.069) | 0.8213 (+0.0179) | -5.149 (+2.358) | 1.095406 (-0.047995) | 通过 |
| full_utterance | L3 | 4 | 1.655 (+0.048) | 0.8436 (+0.0144) | -3.677 (+1.245) | 1.032754 (-0.042959) | 通过 |

裁剪比例全部为 0，说明没有新增削波问题。

#### 结论

S2-C 是目前第一个明确过线的补充实验。与实验二 best-dev 相比，S2-C 在关键的 full_utterance L3 上同时改善：

- PESQ-WB：`1.607 -> 1.655`
- STOI：`0.8291 -> 0.8436`
- SI-SNR：`-4.922 -> -3.677 dB`
- Mel L1：`1.075713 -> 1.032754`

这支持当前假设：原始 `distill_loss_lambda=120` 很可能偏大，确实压制了声学重建质量。将权重降到 30 后，听感代理指标和可懂度代理指标没有冲突，反而同时改善。

需要注意的是，S2-C 的 `dev/mel error` 在 step 2000 最好，step 2500 已经回退；训练不宜继续沿当前调度无脑拉长。下一步应围绕 step 1500-2500 的区间做更细评估或早停。

#### 下一步建议

当前建议把 S2-C 标记为“有效候选”，优先做两件事：

1. 听 S2-C 的 full_utterance L3 样本，确认 PESQ/STOI 的提升是否和主观听感一致。
2. 继续跑 S2-B 或 S2-D 中的一个对照分支。优先顺序建议：
   - 若想确认最优权重区间：跑 S2-B，看 60 是否比 30 更稳。
   - 若担心 30 损伤语义表示：跑 S2-D，看 60 到 30 的退火是否兼顾稳定性。

当前主线判断：S2-C 是正结果，可以作为后续调参的新基准；S1 不再作为主线。

### S2-C-formal：`distill_loss_lambda=30` 正式完整重训准备

- 日期：2026-05-28
- 状态：配置与运行目录已准备，训练尚未启动。
- 运行目录：`output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42`
- 配置文件：`output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\configs\scit_speech_base_config.json`
- 启动命令文件：`output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\commands\run_command.txt`
- 配置准备记录：`output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\reports\prepare_config.md`

#### 动机

S2-C 短程续训已经证明：从实验二 best-dev checkpoint 出发，将 `distill_loss_lambda` 从 120 降到 30 后，full_utterance L3 的 PESQ-WB、STOI、SI-SNR、Mel L1 同时改善。因此下一步需要做一次正式完整重训，验证“从训练一开始使用 `distill_loss_lambda=30`”是否能得到更好的最终模型。

这次正式重训不是从 S2-C checkpoint 继续，也不是从实验二 best-dev 继续，而是沿用实验二同一套数据、同一套 Exp1 NAS encoder handoff、同一 seed，从头训练 SCIT-Speech-Base。

#### 配置摘要

| 项目 | 值 |
|---|---|
| run_id | `exp2_scit_speech_distill30_retrain_20260528_seed42` |
| 训练模式 | `formal` |
| 起点 | 从头训练，不加载 `pretrained_generator_checkpoint` |
| Exp1 handoff | `best_seanet_config.json`, candidate `nas_seed42_000896` |
| distill_loss_lambda | `30.0` |
| distill_loss_lambda_source | `override` |
| experiment_tag | `distill30_retrain` |
| seed | `42` |
| epochs | `60` |
| learning_rate | `1e-4` |
| save_model_steps | `2500` |
| batch_size | `8` |
| gradient_accumulation_steps | `4` |
| n_q | `3` |
| segment_size | `16000` |

#### 启动命令

在项目根目录 `H:\H-CODE\speechtokenizer` 下运行：

```powershell
conda run -n speechtokenizer accelerate launch scripts/train_example.py --config output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\configs\scit_speech_base_config.json
```

也可以直接读取命令文件：

```powershell
Get-Content output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\commands\run_command.txt
```

#### 训练后评估要求

训练结束后，不看最后 checkpoint 直接下结论，仍以 best-dev checkpoint 为准。建议按实验二同口径执行：

1. package best-dev checkpoint 为 `SCIT-Speech-Base_best.pt`。
2. 导出 fixed 与 full_utterance 的 L1/L2/L3 样本。
3. 运行 `scripts/evaluate_sample_audio_quality.py`，与实验二 best-dev 与 S2-C finetune 结果同时对比。
4. 记录 L3 的 PESQ-WB、STOI、SI-SNR、Mel L1；full_utterance 优先级高于 fixed 1 秒片段。
5. 如果完整重训的 full_utterance L3 不优于 S2-C finetune，需要回看训练曲线，确认是否出现更早 step 的最佳点。

#### 代码准备情况

- `scripts/prepare_exp2_config.py` 已支持 `--distill-loss-lambda`、`--experiment-tag`、`--experiment-note`。
- `scripts/run_exp2_training.ps1` 已支持 `-DistillLossLambda`、`-ExperimentTag`、`-ExperimentNote`。
- `tests/test_exp2_script_chain.py` 已新增 `distill30` 正式重训配置测试。

#### 验证情况

已执行：

```powershell
& 'C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe' -m unittest discover -s tests -p test_exp2_script_chain.py -k distill30
```

结果：`Ran 1 test ... OK`。

### S2-C-formal：epoch 10 中期检查

- 日期：2026-05-29
- 状态：训练仍在进行，已检查到 epoch 10 附近的 TensorBoard 标量与 checkpoint。
- 中期报告：`output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\reports\interim_epoch10_status.md`
- TensorBoard 标量摘要：`output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\metrics\interim_tensorboard_scalar_summary.json`

#### 当前进度

- 训练集条数：27039
- batch_size：8
- 估算每 epoch batch step：3379
- 当前训练标量最新 step：36500，约 10.80 epoch
- 当前 dev 最新 step：35000，约 10.36 epoch
- 当前 `SpeechTokenizer_best_dev.pt` 更新时间：2026-05-29 05:56:24，对应 step 22500。

#### dev/mel error 曲线摘要

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

#### 与实验二基线对比

| 对比项 | 实验二基线 | distill30 正式重训 | 差值 |
|---|---:|---:|---:|
| 同 step 22500 dev/mel error | 3.867209 | 3.475089 | -0.392120 |
| 同 step 35000 dev/mel error | 3.715640 | 3.549350 | -0.166291 |
| 截至 35000 的最佳 dev/mel error | 3.703312 | 3.475089 | -0.228223 |
| 实验二全程 best-dev | 3.470752 | 3.475089 | +0.004337 |

#### 中期判断

`distill_loss_lambda=30` 正式重训在前 10 个 epoch 明显快于原实验二：同 step 22500 与 35000 都显著优于实验二基线，当前 best-dev 已经基本贴近实验二全程 best-dev。

但 step 22500 后 dev/mel 出现波动，35000 仍未刷新当前最佳。因此现在不建议中断训练，建议继续观察 37500、40000、42500、45000 几个验证点。如果 45000 附近仍没有超过 step 22500，则以当前 best-dev 做一次样本级 PESQ/STOI/SI-SNR/Mel L1 评估。

本次未导出样本评估，因为训练进程仍在运行，避免抢占 GPU 干扰训练。

### S2-C-formal：22500 best-dev 与 42500 checkpoint 长音频听感样本对比

- 日期：2026-05-29
- 对比报告：`output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\reports\checkpoint_22500_vs_42500_full_utterance.md`
- 22500 best-dev 试听页：`output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\samples\full_utterance\listen_full.html`
- 42500 checkpoint 试听页：`output\experiments\exp2_scit_speech_distill30_retrain_20260528_seed42\checkpoint_exports\SpeechTokenizerTrainer_00042500\samples\full_utterance\listen_full.html`
- 样本集：同一批 4 条 full_utterance 长音频。

#### 汇总结果

| L | checkpoint | PESQ-WB ↑ | STOI ↑ | SI-SNR dB ↑ | Mel L1 ↓ | Wave L1 ↓ |
|---:|---|---:|---:|---:|---:|---:|
| L1 | 22500 best-dev | 1.203 | 0.7338 | -12.190 | 1.424405 | 0.033677 |
| L1 | 42500 | 1.249 (+0.046) | 0.7585 (+0.0247) | -7.851 (+4.339) | 1.345840 (-0.078565) | 0.032427 (-0.001250) |
| L2 | 22500 best-dev | 1.487 | 0.8085 | -5.400 | 1.202103 | 0.027108 |
| L2 | 42500 | 1.573 (+0.086) | 0.8280 (+0.0195) | -3.369 (+2.031) | 1.106687 (-0.095417) | 0.025780 (-0.001327) |
| L3 | 22500 best-dev | 1.618 | 0.8292 | -3.961 | 1.145732 | 0.025141 |
| L3 | 42500 | 1.713 (+0.096) | 0.8540 (+0.0248) | -1.624 (+2.336) | 1.042575 (-0.103157) | 0.023614 (-0.001527) |

#### 结论

42500 checkpoint 在这 4 条长音频上明显优于 22500 best-dev，且与主观听感“后面一个更好”一致。关键 L3 上，PESQ-WB 提升约 0.096，STOI 提升约 0.0248，SI-SNR 提升约 2.34 dB，Mel L1 降低约 0.103。

这说明当前 validation `dev/mel error` 与长音频听感/客观质量并不完全一致。后续选择最终 checkpoint 时，不能只看 `SpeechTokenizer_best_dev.pt`，需要同时对候选周期 checkpoint 做 full_utterance 样本评估。
