# 实验名称：实验二 SCIT-Speech-Base 训练

## 1. 实验目的

本实验训练 `SCIT-Speech-Base`，用于实例化共享码本索引传输框架。实验二不是最终质量结论，而是实验三低负载适配、实验四 baseline 对比、实验五消融和实验六系统验证的前置模型来源。

本实验需要训练：

- 发送端表示 encoder `E_theta`
- shared RVQ codebooks `C*`
- speech reconstruction decoder `D_psi`
- 训练阶段的判别器和感知损失模块

本实验不训练通信信道、packetization、三用户路由、ASR 评价模型或传统 codec。论文叙事中统一使用 `SCIT-Speech`、`shared RVQ codebooks`、`index-only transmission` 等术语，不把具体开源 tokenizer 名称作为方法主语。

## 2. 实验输入

| 输入 | 当前路径或建议路径 | 说明 |
|---|---|---|
| 实验手册 | `output/doc/实验手册.md` | 实验二定义、训练路线和 codebook sanity check 来源 |
| 主配置 | `config/spt_base_cfg.json` | 已配置 `sample_rate=16000`、`strides=[8,5,4,2]`、`dimension=1024`、`n_q=3`、`codebook_size=1024` |
| 训练脚本 | `scripts/train_example.py`、`scripts/train_example.sh` | 当前完整训练入口，使用 `SpeechTokenizerTrainer` 和 Accelerate |
| HuBERT 表征提取 | `scripts/hubert_rep_extract.py`、`scripts/hubert_rep_extract.sh` | 若 `train_files.txt`/`valid_files.txt` 缺失，需要先预处理 |
| 训练/验证清单 | `data/SpeechPretrain/hubert_rep/LibriSpeech/train_files.txt`、`valid_files.txt` | 当前项目已有；每行 `<audio>\t<hubert.npy>` |
| 模型代码 | `speechtokenizer/model.py` | `SpeechTokenizer.forward/encode/decode` 支持 RVQ 截层 |
| 训练器 | `speechtokenizer/trainer/trainer.py` | 记录 loss、TensorBoard、checkpoint、validation samples |
| 数据集 | `speechtokenizer/trainer/dataset.py` | 读取音频和 `.hubert.npy`，按 segment 对齐 |
| 损失函数 | `speechtokenizer/trainer/loss.py` | recon、mel、distillation、GAN loss |
| 实验一产物 | `output/experiments/{exp1_run_id}/artifacts/best_architecture/` | 若采用 NAS encoder，必须引用实验一选定架构；若未完成，则使用 hand-designed encoder 作为训练路径 |

需要新增脚本：

| 建议路径 | 职责 | 输入 | 输出 |
|---|---|---|---|
| `scripts/prepare_exp2_config.py` | 从主配置生成本 run 的冻结训练配置，可改 `results_folder` 指向 run 目录 | base config、run_id、可选 NAS config | `configs/scit_speech_base_config.json` |
| `scripts/collect_environment.py` | 保存环境和 git 状态 | run 目录 | `reports/environment.md/json` |
| `scripts/evaluate_layer_reconstruction.py` | 对固定样本生成 `L=1/2/3` 截层重建音频和指标 | config、checkpoint、sample list | `samples/`、`metrics/layer_reconstruction.*` |
| `scripts/codebook_usage_report.py` | 统计 usage rate、perplexity、dead code ratio、layer-wise distribution | checkpoint、eval list | `metrics/codebook_usage.json/csv`、`reports/codebook_usage.md` |
| `scripts/export_loss_curves.py` | 从 TensorBoard 或训练日志导出 loss 曲线 | `logs/` | `metrics/loss_curves.csv`、`artifacts/loss_curves/*.png` |

## 3. 实验输出

每次执行保存到：

```text
output/experiments/{run_id}/
```

推荐 `run_id`：

```text
exp2_scit_speech_training_YYYYMMDD_HHMMSS_seed{seed}
```

应产生：

- `configs/scit_speech_base_config.json`：本次训练配置副本，`results_folder` 指向当前 run。
- `configs/data_split.json`：训练/验证/固定样本清单摘要和 seed。
- `commands/run_command.txt`：实际训练、续训、评估命令。
- `logs/stdout.log`、`logs/stderr.log`：完整训练日志。
- `logs/tensorboard/`：TensorBoard event 文件副本或软链接记录。
- `checkpoints/SpeechTokenizerTrainer_*`、`checkpoints/SCIT-Speech-Base_best.pt`：本 run checkpoint。
- `samples/fixed/original/`：固定样本原始音频。
- `samples/fixed/recon_L1/`、`recon_L2/`、`recon_L3/`：截层重建样本。
- `metrics/loss_curves.csv`、`metrics/train_metrics.jsonl`。
- `metrics/codebook_usage.json`、`metrics/codebook_usage.csv`。
- `reports/codebook_usage.md`、`reports/layer_reconstruction.md`、`reports/summary.md`。

## 4. 实验变量与对照

主变量：

- 训练得到的 `SCIT-Speech-Base` checkpoint。
- 若实验一已完成，可比较 `hand-designed encoder` 和 `NAS encoder` 两条训练配置，但主输出必须明确一个作为后续实验的 Base。

固定变量：

- 数据集：第一版使用 LibriSpeech train-clean-100 训练清单和固定验证清单。
- 采样率：16 kHz 单声道。
- `M=3`、`K=1024`、`latent_rate=50 steps/s`。
- semantic teacher：配置中的 `facebook/hubert-base-ls960` 和 `semantic_model_layer=avg`，只作为训练辅助。
- 训练 seed、batch size、segment size、优化器、loss 权重、checkpoint 保存间隔。

对照与诊断：

- `L=1/2/3` 截层重建不是三组模型，而是同一 Base checkpoint 的三种通信操作点。
- codebook usage sanity check 用于判断 RVQ 是否 collapse。
- 不把已有 `Log/spt_base` 目录中的历史 checkpoint 直接写成新实验结果；若复用，必须复制到 run 目录并记录 provenance。

## 5. 详细执行步骤

### 步骤 1：创建 run 目录

- 操作目标：建立独立实验目录，避免训练产物混入 `Log/spt_base` 历史目录。
- 涉及文件或脚本：需要新增 `scripts/create_experiment_run.py`。
- 输入：`run_id=exp2_scit_speech_training_YYYYMMDD_HHMMSS_seed{seed}`。
- 输出：标准 run 目录。
- 检查点：`configs/`、`commands/`、`logs/`、`checkpoints/`、`metrics/`、`samples/`、`reports/`、`artifacts/` 均存在。
- 失败时如何判断问题：若训练配置仍写到 `Log/spt_base`，必须先改配置副本的 `results_folder`。

### 步骤 2：冻结训练配置

- 操作目标：生成本 run 的 `SCIT-Speech-Base` 配置。
- 涉及文件或脚本：`config/spt_base_cfg.json`、需要新增 `scripts/prepare_exp2_config.py`。
- 输入：base config、实验一 best architecture 路径或 hand-designed encoder 标记。
- 输出：`configs/scit_speech_base_config.json`。
- 检查点：配置中 `results_folder` 应指向 `output/experiments/{run_id}/checkpoints` 或 run 内训练目录；`n_q=3`、`codebook_size=1024`、`dimension=1024`。
- 失败时如何判断问题：若配置缺少 `train_files` 或 `valid_files`，需要先补数据预处理。

### 步骤 3：检查数据清单和样本列表

- 操作目标：确认训练/验证输入可读，并固定后续音频样本。
- 涉及文件或脚本：`speechtokenizer/trainer/dataset.py`、需要新增 `scripts/build_fixed_sample_list.py`。
- 输入：`train_files.txt`、`valid_files.txt`。
- 输出：`configs/data_split.json`、`artifacts/train_files.txt`、`artifacts/valid_files.txt`、`artifacts/fixed_sample_list.txt`。
- 检查点：每行音频和 `.hubert.npy` 存在；固定样本覆盖不同 speaker、时长和文本。
- 失败时如何判断问题：若 `.hubert.npy` 文件缺失或 shape 不对，必须先运行 `scripts/hubert_rep_extract.py` 或修复清单。

### 步骤 4：保存环境信息

- 操作目标：记录可复现实验环境。
- 涉及文件或脚本：需要新增 `scripts/collect_environment.py`。
- 输入：当前环境。
- 输出：`reports/environment.md`、`reports/environment.json`。
- 检查点：至少包含 Python、PyTorch、CUDA、GPU、依赖版本、git commit hash、git status 摘要。
- 失败时如何判断问题：环境信息缺失时，不能宣称 run 可复现。

### 步骤 5：记录训练命令

- 操作目标：保存实际训练入口。
- 涉及文件或脚本：`scripts/train_example.py`、`scripts/train_example.sh`。
- 输入：冻结配置。
- 输出：`commands/run_command.txt`。
- 命令模板：

```bash
accelerate launch scripts/train_example.py --config output/experiments/{run_id}/configs/scit_speech_base_config.json
```

- 检查点：若断点续训，命令必须追加 `--continue_train` 并记录起始 checkpoint。
- 失败时如何判断问题：未保存命令或命令引用原始配置路径而非 run 内配置副本，会破坏复现。

### 步骤 6：执行基础训练

- 操作目标：训练 encoder、shared RVQ codebooks、decoder。
- 涉及文件或脚本：`scripts/train_example.py`、`speechtokenizer/trainer/trainer.py`。
- 输入：`configs/scit_speech_base_config.json`、训练/验证清单。
- 输出：checkpoint、TensorBoard 日志、validation audio、stdout/stderr。
- 检查点：训练日志出现 generator loss、mel error、quantizer loss、distillation loss；validation 可保存 checkpoint。
- 失败时如何判断问题：OOM、loss NaN、数据加载报错、validation dataset 不足、warmup 配置触发 `self.lrz` 风险时，保存失败报告，不要删除 run。

### 步骤 7：复制和规范 checkpoint

- 操作目标：把训练器产物整理成实验二标准命名。
- 涉及文件或脚本：需要新增 `scripts/package_checkpoint.py`。
- 输入：训练器输出的 `SpeechTokenizerTrainer_*` 和 `SpeechTokenizer_best_dev.pt`。
- 输出：`checkpoints/SCIT-Speech-Base_best.pt`、`checkpoints/checkpoint_manifest.json`。
- 检查点：checkpoint 可被 `SpeechTokenizer.load_from_checkpoint(config, ckpt)` 加载。
- 失败时如何判断问题：若 checkpoint 是训练器包 dict 而非纯 generator state dict，需要明确记录加载方式，不可混用。

### 步骤 8：生成 `L=1/2/3` 截层重建样本

- 操作目标：验证 Base checkpoint 在三个传输层数下均可解码。
- 涉及文件或脚本：`speechtokenizer/model.py` 的 `encode/decode`；可参考 `demo_nature/多人嘈杂环境/example.py`；需要新增 `scripts/evaluate_layer_reconstruction.py`。
- 输入：`configs/scit_speech_base_config.json`、`checkpoints/SCIT-Speech-Base_best.pt`、`artifacts/fixed_sample_list.txt`。
- 输出：`samples/fixed/original/`、`samples/fixed/recon_L1/`、`recon_L2/`、`recon_L3/`、`metrics/layer_reconstruction.json/csv`。
- 检查点：`L=1/2/3` 都能产生非静音音频；输出采样率 16 kHz；长度与输入近似对齐。
- 失败时如何判断问题：若 `decode(codes[:L])` 崩溃，检查 `codes` shape `(n_q, B, T)` 和 `st=0`；若大面积静音，标记模型质量失败。

### 步骤 9：运行 codebook usage sanity check

- 操作目标：确认 shared RVQ codebooks 没有明显 collapse。
- 涉及文件或脚本：`speechtokenizer/quantization/vq.py`、需要新增 `scripts/codebook_usage_report.py`。
- 输入：checkpoint、验证集或固定评估清单。
- 输出：`metrics/codebook_usage.json`、`metrics/codebook_usage.csv`、`reports/codebook_usage.md`、可选直方图。
- 检查点：统计每层 usage rate、perplexity、dead code ratio、layer-wise distribution。
- 失败时如何判断问题：第一层只集中在极少数 codeword、后续层大面积 dead code、perplexity 过低时，需要记录并考虑重跑或调参。

### 步骤 10：导出 loss 曲线和训练摘要

- 操作目标：把 TensorBoard/日志转换为论文和复现可读格式。
- 涉及文件或脚本：需要新增 `scripts/export_loss_curves.py`、`scripts/summarize_exp2.py`。
- 输入：`logs/`、`metrics/`、`checkpoints/checkpoint_manifest.json`。
- 输出：`metrics/loss_curves.csv`、`artifacts/loss_curves/*.png`、`reports/summary.md`。
- 检查点：summary 不含假结果；缺失的客观指标写“待实验三/四评估”。
- 失败时如何判断问题：若只有 TensorBoard event 但没有 CSV，应补导出；若日志缺失，summary 必须标注不完整。

## 6. 指标与统计方式

| 指标 | 统计方式 | 说明 |
|---|---|---|
| Train generator loss | 训练日志/TensorBoard 导出 | 只用于训练监控 |
| Train mel loss / mel error | `speechtokenizer/trainer/loss.py` | 用于观察重建趋势 |
| Quantizer loss | trainer 中 `loss_q` | codebook 学习稳定性 |
| Distillation loss | `d_axis_distill_loss` 或配置指定类型 | 语义蒸馏诊断 |
| Validation mel error | trainer validation 阶段 | checkpoint 选择依据之一 |
| L=1/2/3 decode success | 固定样本截层解码是否成功 | 必须保存音频 |
| Usage rate | 每层被使用过的 codeword 数 / 1024 | `metrics/codebook_usage.csv` |
| Perplexity | `exp(H(index_distribution))` | 每层索引分布复杂度 |
| Dead code ratio | 长时间未被选中的 codeword 比例 | 需定义统计窗口 |
| Ideal raw index load | `L * 50 * 10 bps` | L=1/2/3 分别为 500/1000/1500 bps |

实验二不要求完整 WER/PESQ/STOI 主结果；这些应在实验三/四用统一评估脚本计算。但实验二必须保留固定重建样本和 codebook 使用诊断。

## 7. 结果记录格式

### Table A：训练运行摘要

| run_id | seed | encoder_source | config_path | train_samples | valid_samples | epochs_planned | steps_completed | best_checkpoint | best_dev_mel_error | status | notes |
|---|---:|---|---|---:|---:|---:|---:|---|---:|---|---|

### Table B：Loss 曲线导出

| run_id | step | epoch | train_generator_loss | train_mel_loss | train_mel_error | train_quantizer_loss | train_distillation_loss | dev_mel_error | dev_distillation_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### Table C：截层重建样本

| run_id | sample_id | speaker_id | utterance_id | duration_sec | original_path | recon_L1_path | recon_L2_path | recon_L3_path | notes |
|---|---|---|---|---:|---|---|---|---|---|

### Table D：Codebook usage

| run_id | layer | codebook_size | used_codes | usage_rate | perplexity | dead_codes | dead_code_ratio | top1_frequency | notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|

## 8. 成功标准

最低完成标准：

- run 目录和留存结构完整。
- 配置副本、命令、环境、stdout/stderr、数据清单均已保存。
- checkpoint 可加载，且明确是 generator state dict 或 trainer package。
- `M=3`、`K=1024`、sample rate 16 kHz、downsample rate 320 与实验手册一致。
- 固定样本的 `L=1/2/3` 均可解码并保存。
- codebook usage sanity check 已完成，未发现明显 collapse，或已记录问题和重跑建议。

失败或需要重跑：

- loss NaN 或训练无法持续。
- checkpoint 无法加载。
- `L=1/2/3` 任一层数解码崩溃、长度严重错位或输出大面积静音。
- codebook 大面积 collapse 且无法解释。
- 缺少训练日志或环境信息。

## 9. 风险与注意事项

- 训练器当前会把配置写到 `results_folder/config.json`，必须确保 `results_folder` 在 run 目录内。
- `trainer.py` 中 `warmup()` 的 `self.lrz` 存在潜在风险；若启用 warmup，需要先修复或记录。
- `gradient_accumulation_steps` 与判别器 `zero_grad` 的行为需要在训练日志中记录，避免复现实验时 batch 口径不一致。
- semantic teacher 只作为训练辅助，不作为论文方法主语。
- 不要把 public checkpoint 当作本文模型；若用历史 checkpoint，必须记录 provenance 并不混入新实验 run。
- `L=1` 自然度差不等于实验失败；若内容可懂，可作为极低负载操作点保留。
- 码本使用率不应只看全局；必须 layer-wise 分析。

## 10. 实验数据与执行过程留存

`run_id`：

```text
exp2_scit_speech_training_YYYYMMDD_HHMMSS_seed{seed}
```

标准目录：

```text
output/experiments/{run_id}/
  configs/
  commands/
  logs/
  checkpoints/
  metrics/
  samples/
  reports/
  artifacts/
```

必须保存：

- 配置副本：`configs/scit_speech_base_config.json`。
- 数据划分：`artifacts/train_files.txt`、`artifacts/valid_files.txt`、`artifacts/fixed_sample_list.txt`、`configs/data_split.json`。
- 命令：`commands/run_command.txt`。
- 环境：`reports/environment.md`，含 Python、PyTorch、CUDA、GPU、依赖、git commit、git status 摘要。
- 日志：`logs/stdout.log`、`logs/stderr.log`、`logs/tensorboard/`。
- checkpoint：`checkpoints/` 和 `checkpoints/checkpoint_manifest.json`。
- metrics：`metrics/loss_curves.csv`、`metrics/codebook_usage.json/csv`、`metrics/layer_reconstruction.json/csv`。
- audio samples：固定样本原始音频、`L=1/2/3` 重建音频和条件说明。
- failure：若失败，写 `reports/failure_report.md`，保留已生成日志和中间 checkpoint。

可复现实验所必需文件：

- `configs/scit_speech_base_config.json`
- `commands/run_command.txt`
- `reports/environment.md`
- `artifacts/train_files.txt`
- `artifacts/valid_files.txt`
- `checkpoints/SCIT-Speech-Base_best.pt` 或明确的 trainer checkpoint
- `logs/stdout.log`
- `logs/stderr.log`

论文作图/写表所需文件：

- `metrics/loss_curves.csv`
- `metrics/codebook_usage.csv`
- `metrics/layer_reconstruction.csv`
- `samples/fixed/**`
- `reports/codebook_usage.md`
- `reports/summary.md`

失败实验如何留存：

- 不删除 run。
- 保留配置、命令、日志、最后 checkpoint 或 crash 前 checkpoint。
- `reports/failure_report.md` 写明失败阶段、错误摘要、怀疑原因、是否建议从 checkpoint 续训。

多次 run 如何区分：

- 使用不同 `run_id`。
- 如果同一配置不同 seed，seed 必须进入 run_id 和 `configs/data_split.json`。
- 如果续训，记录父 checkpoint 和续训命令。

最终 summary 如何生成：

- 由 `metrics/loss_curves.csv`、`metrics/codebook_usage.csv`、`metrics/layer_reconstruction.csv` 汇总。
- summary 只写实际训练和诊断结果；未跑 WER/PESQ/STOI 则标为实验三/四待评估。

## 11. 交给执行型 AI 的提示词

你要执行“实验二：SCIT-Speech-Base 训练”。请先阅读 `output/doc/experiment_plans/exp2_scit_speech_training.md`、`output/doc/实验手册.md`、`config/spt_base_cfg.json`、`scripts/train_example.py`、`scripts/hubert_rep_extract.py`、`speechtokenizer/model.py`、`speechtokenizer/trainer/trainer.py`、`speechtokenizer/trainer/dataset.py`、`speechtokenizer/trainer/loss.py` 和 `output/experiments/README.md`。不得编造训练结果、loss、checkpoint、音频样本或 codebook 指标。

请生成唯一 `run_id=exp2_scit_speech_training_YYYYMMDD_HHMMSS_seed{seed}`，在 `output/experiments/{run_id}/` 下创建 `configs/`、`commands/`、`logs/`、`checkpoints/`、`metrics/`、`samples/`、`reports/`、`artifacts/`。必须复制训练配置到 run 目录，确保 `results_folder` 指向当前 run，保存实际命令到 `commands/run_command.txt`，保存 stdout/stderr 和 TensorBoard 日志，保存环境信息到 `reports/environment.md`，保存随机种子、训练/验证清单、固定样本清单、checkpoint、loss 曲线、`L=1/2/3` 重建样本，以及 codebook usage report。

本实验聚焦训练 `SCIT-Speech-Base` 的 encoder、shared RVQ codebooks 和 decoder。训练完成后必须对固定样本导出 `L=1/2/3` 截层重建音频，并统计 usage rate、perplexity、dead code ratio。遇到缺失数据、缺失脚本、checkpoint 加载失败、loss NaN、码本 collapse 或截层解码失败时，不要编造结果；请保留失败 run，并写 `reports/failure_report.md`，说明缺失脚本职责、输入、输出和建议处理。
