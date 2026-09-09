# 实验名称：实验三 低负载/轻量信道感知适配

## 1. 实验目的

本实验从 `SCIT-Speech-Base` 微调得到 `SCIT-Speech-LCA`，验证面向通信场景的低负载和轻量索引扰动目标是否提升 `L=1/2` 操作点的可用性。它承接实验二的基础模型，并为实验四 baseline 对比和实验五消融提供主模型条件。

核心问题：

- 只传输前 `L` 层 discrete codebook indices 时，模型是否仍能恢复可懂语音。
- `L=1/2/3` 是否形成清晰的负载-可用性曲线。
- `SCIT-Speech-LCA` 是否相对 `SCIT-Speech-Base` 改善 `L=1/2`。
- mild index-level perturbation 下，LCA 是否比 Base 退化更小。

默认策略：end-to-end fine-tuning，即 encoder、shared RVQ codebooks 和 decoder 一起微调。`decoder-only fine-tuning` 只作为诊断消融，不作为主路线。

## 2. 实验输入

| 输入 | 当前路径或建议路径 | 说明 |
|---|---|---|
| 实验手册 | `output/doc/实验手册.md` | LCA 训练流程、random L、ChannelSim 定义 |
| Base 训练计划和产物 | `output/doc/experiment_plans/exp2_scit_speech_training.md`、`output/experiments/{exp2_run_id}/` | 必须有可加载的 `SCIT-Speech-Base` checkpoint |
| Base 配置 | `output/experiments/{exp2_run_id}/configs/scit_speech_base_config.json` | 作为 LCA 微调配置的来源 |
| Base checkpoint | `output/experiments/{exp2_run_id}/checkpoints/SCIT-Speech-Base_best.pt` | 初始化 LCA |
| 训练/验证清单 | `data/SpeechPretrain/hubert_rep/LibriSpeech/train_files.txt`、`valid_files.txt` 或实验二副本 | 微调数据 |
| 模型 API | `speechtokenizer/model.py` | `encode/decode` 支持截层 |
| Trainer 基础 | `speechtokenizer/trainer/trainer.py` | 现有 Trainer 不含 random L 和 ChannelSim，需要新增 |
| Loss 函数 | `speechtokenizer/trainer/loss.py` | 可复用 recon、mel、distillation、GAN loss |

需要新增脚本或模块：

| 建议路径 | 职责 | 输入 | 输出 |
|---|---|---|---|
| `speechtokenizer/trainer/lca_trainer.py` | 实现 end-to-end LCA 微调：random L、clean/dropout/substitution、full-depth preservation | Base 模型、LCA 配置、数据清单 | LCA checkpoint、训练日志 |
| `scripts/train_lca.py` | 命令行入口，加载 Base checkpoint 并调用 LCA trainer | config、base checkpoint | checkpoint、logs |
| `scripts/channel_sim.py` | 实现 index-level clean、previous-index replacement dropout、light substitution | codes、扰动配置、seed | perturbed codes、扰动记录 |
| `scripts/evaluate_lca_vs_base.py` | Base vs LCA 在 `L=1/2/3` 和 channel conditions 下统一评估 | 两个 checkpoint、测试清单 | metrics、samples、reports |
| `scripts/collect_environment.py` | 保存环境信息 | run 目录 | `reports/environment.md/json` |

## 3. 实验输出

推荐 `run_id`：

```text
exp3_low_load_channel_aware_adaptation_YYYYMMDD_HHMMSS_seed{seed}
```

输出目录：

```text
output/experiments/{run_id}/
```

应产生：

- `configs/lca_finetune_config.json`：微调配置副本。
- `configs/channel_sim_config.json`：clean、dropout、substitution 条件和概率。
- `configs/random_l_sampling.json`：`L` 采样策略、权重、seed。
- `commands/run_command.txt`：微调和评估命令。
- `logs/stdout.log`、`logs/stderr.log`、`logs/train_lca.log`。
- `checkpoints/SCIT-Speech-LCA_best.pt`、`checkpoints/checkpoint_manifest.json`。
- `metrics/lca_train_metrics.jsonl`、`metrics/base_vs_lca_results.json/csv`。
- `samples/base/{condition}/L{1,2,3}/`。
- `samples/lca/{condition}/L{1,2,3}/`。
- `reports/channel_sim_report.md`、`reports/base_vs_lca_summary.md`、`reports/summary.md`。
- 若失败：`reports/failure_report.md`。

## 4. 实验变量与对照

主变量：

- 模型条件：`SCIT-Speech-Base` vs `SCIT-Speech-LCA`。
- 传输层数：`L in {1, 2, 3}`。
- 信道条件：
  - `clean`: `p_drop=0`, `p_sub=0`
  - `dropout-low`: `p_drop=0.01`, `p_sub=0`
  - `dropout-mid`: `p_drop=0.03`, `p_sub=0`
  - `substitution-low`: `p_drop=0`, `p_sub=0.001`
  - `substitution-mid`: `p_drop=0`, `p_sub=0.005`

固定变量：

- `sample_rate=16000`
- `latent_rate=50 steps/s`
- `M=3`
- `K=1024`
- 数据划分、固定测试样本、评价脚本、ASR/metric 模型。

训练机制：

- random L sampling：每个 batch 从 `{1,2,3}` 采样 `L`，保存采样分布。
- index dropout 第一版使用 `previous-index replacement`，不引入未知 token。
- light index substitution 将少量索引替换为另一个合法 codebook 地址。

不允许：

- 不改变 `M/K/latent_rate` 来获得更好结果。
- 不把强信道鲁棒写成主张。
- 不让评价 ASR 模型参与训练。

## 5. 详细执行步骤

### 步骤 1：创建 run 目录

- 操作目标：为 LCA 微调和 Base vs LCA 评估建立独立目录。
- 涉及文件或脚本：需要新增 `scripts/create_experiment_run.py`。
- 输入：`run_id=exp3_low_load_channel_aware_adaptation_YYYYMMDD_HHMMSS_seed{seed}`。
- 输出：标准 run 目录。
- 检查点：所有标准子目录存在。
- 失败时如何判断问题：run 目录重名或缺少子目录时不得继续。

### 步骤 2：验证 Base 产物

- 操作目标：确认实验二产物可用。
- 涉及文件或脚本：`speechtokenizer/model.py`、需要新增 `scripts/check_checkpoint.py`。
- 输入：Base config、Base checkpoint。
- 输出：`reports/base_checkpoint_check.md`。
- 检查点：checkpoint 可加载；`n_q=3`；固定样本可 `encode/decode`。
- 失败时如何判断问题：Base 不可加载时必须返回实验二修复，不能开始 LCA。

### 步骤 3：冻结 LCA 配置

- 操作目标：从 Base 配置派生微调配置。
- 涉及文件或脚本：需要新增 `scripts/prepare_lca_config.py`。
- 输入：Base config、Base checkpoint、LCA 超参数。
- 输出：`configs/lca_finetune_config.json`。
- 检查点：配置中包含 `base_checkpoint`、`random_l_sampling`、`channel_sim`、`alpha/beta/lambda_*`、`results_folder`。
- 失败时如何判断问题：若配置未记录扰动概率或 L 采样策略，run 不可复现。

### 步骤 4：实现或确认 ChannelSim

- 操作目标：确保 index-level perturbation 与手册一致。
- 涉及文件或脚本：需要新增 `scripts/channel_sim.py`。
- 输入：codes `(n_q, B, T)`、`L`、`p_drop`、`p_sub`、seed。
- 输出：perturbed codes、扰动 mask、统计日志。
- 检查点：
  - clean 不改变 codes。
  - dropout 使用 previous-index replacement；第一个时间步可保持原值或使用同层第一个合法索引，必须记录。
  - substitution 只替换为 `[0, K-1]` 内合法索引。
- 失败时如何判断问题：若产生非法 index 或 shape 改变，必须停止并写 failure report。

### 步骤 5：实现或确认 LCA trainer

- 操作目标：微调时同时保留 full-depth 上限和优化低负载通信输出。
- 涉及文件或脚本：需要新增 `speechtokenizer/trainer/lca_trainer.py`、`scripts/train_lca.py`。
- 输入：Base checkpoint、LCA config、训练/验证清单。
- 输出：LCA checkpoint、loss 日志。
- 检查点：每个 batch 记录 sampled `L` 和 channel condition；`L_full` 与 `L_comm` 均被计算。
- 失败时如何判断问题：若 trainer 只训练 full-depth 或只做 decoder-only，需要明确标记为消融/诊断，不可作为主 LCA。

### 步骤 6：记录微调命令

- 操作目标：保存实际命令。
- 涉及文件或脚本：`scripts/train_lca.py`。
- 输入：LCA config、Base checkpoint。
- 输出：`commands/run_command.txt`。
- 命令模板：

```bash
accelerate launch scripts/train_lca.py \
  --config output/experiments/{run_id}/configs/lca_finetune_config.json \
  --base_checkpoint output/experiments/{exp2_run_id}/checkpoints/SCIT-Speech-Base_best.pt
```

- 检查点：命令引用 run 内配置副本，不只引用原始配置。
- 失败时如何判断问题：命令未保存或 checkpoint 路径不明确时不可复现。

### 步骤 7：执行 LCA 微调

- 操作目标：得到 `SCIT-Speech-LCA`。
- 涉及文件或脚本：`scripts/train_lca.py`、`speechtokenizer/trainer/lca_trainer.py`。
- 输入：LCA config、Base checkpoint、训练/验证清单。
- 输出：`checkpoints/SCIT-Speech-LCA_best.pt`、`logs/train_lca.log`、`metrics/lca_train_metrics.jsonl`。
- 检查点：loss 非 NaN；`L=1/2/3` 都被采样；扰动条件都出现；validation 能解码。
- 失败时如何判断问题：若低负载训练导致 full-depth 音质崩坏，需记录并调整 `alpha/beta` 或重跑。

### 步骤 8：Base vs LCA 统一评估

- 操作目标：比较 Base 和 LCA 在所有 `L` 与轻量信道条件下的质量。
- 涉及文件或脚本：需要新增 `scripts/evaluate_lca_vs_base.py`。
- 输入：Base config/checkpoint、LCA config/checkpoint、测试清单、channel config。
- 输出：`metrics/base_vs_lca_results.json`、`metrics/base_vs_lca_results.csv`、samples。
- 检查点：Base 和 LCA 使用同一测试集、同一固定样本、同一评价脚本。
- 失败时如何判断问题：如果评价模型、样本列表或扰动随机种子不同，结果不可比较。

### 步骤 9：保存 `L=1/2/3` 输出样本

- 操作目标：为音频主观检查和附录样本留存。
- 涉及文件或脚本：`scripts/evaluate_lca_vs_base.py`。
- 输入：固定样本列表。
- 输出：`samples/base/{condition}/L*/`、`samples/lca/{condition}/L*/`、`samples/original/`。
- 检查点：每个样本都有原始音频、Base 输出、LCA 输出和条件说明。
- 失败时如何判断问题：缺少任一条件的样本时，summary 必须标注不完整。

### 步骤 10：汇总报告

- 操作目标：生成可读的 Base vs LCA 结论材料。
- 涉及文件或脚本：需要新增 `scripts/summarize_exp3.py`。
- 输入：metrics、samples manifest、logs。
- 输出：`reports/base_vs_lca_summary.md`、`reports/summary.md`。
- 检查点：只汇报真实指标；不使用“强信道鲁棒”等过度表述。
- 失败时如何判断问题：若没有 metric 文件，只能报告执行失败或样本生成结果，不得写定量结论。

## 6. 指标与统计方式

| 指标 | 统计方式 | 保存位置 |
|---|---|---|
| Ideal bitrate | `L * 50 * ceil(log2 1024)`，即 `500L bps` | `metrics/base_vs_lca_results.csv` |
| WER/CER | 固定 ASR 模型识别原始/重建音频后计算 | 需要新增评价脚本 |
| STOI | 原始与重建音频对齐后计算 | 同上 |
| PESQ / ViSQOL | 自然度/质量指标；保存工具版本 | 同上 |
| Semantic similarity | transcript similarity 或 embedding similarity；评价模型不可参与训练 | 同上 |
| RTF | encode + perturb + decode wall time / audio duration | `metrics/latency_rtf.csv` |
| Degradation | 同一模型同一 L 下，clean 指标与扰动条件指标差值 | `reports/base_vs_lca_summary.md` |
| Random L distribution | 训练中 `L=1/2/3` 采样频次 | `metrics/lca_train_metrics.jsonl` |
| Perturbation stats | dropout/substitution 实际位置数、比例、seed | `metrics/channel_sim_stats.json/csv` |

统计原则：

- Base 和 LCA 必须使用相同测试集和相同扰动 seed。
- clean、dropout、substitution 都作用于索引矩阵，而不是波形。
- 指标同时保存机器可读 JSON/CSV 和人类可读 Markdown。

## 7. 结果记录格式

### Table A：Base vs LCA 主结果

| run_id | model | L | ideal_bitrate_bps | channel | p_drop | p_sub | WER | CER | STOI | PESQ | ViSQOL | semantic_similarity | RTF | sample_count | notes |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

### Table B：扰动统计

| run_id | model | L | channel | sample_id | total_indices | replaced_by_previous | substituted | actual_p_drop | actual_p_sub | seed |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|

### Table C：固定样本 manifest

| run_id | sample_id | original_path | model | L | channel | output_path | condition_description | duration_sec |
|---|---|---|---|---:|---|---|---|---:|

### Table D：训练采样统计

| run_id | step | sampled_L | channel | p_drop | p_sub | full_loss | comm_loss | commit_loss | distill_loss | total_loss |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|

## 8. 成功标准

最低完成标准：

- LCA 从 Base checkpoint 成功初始化并完成微调或明确失败留存。
- 训练配置、扰动配置、random L 配置、命令、日志、环境均已保存。
- Base 和 LCA 都在 `L=1/2/3`、clean 条件下完成评估。
- 至少完成 clean、dropout-low、substitution-low 的评估。
- 固定样本的原始音频、Base 输出、LCA 输出和条件说明均保存。
- 指标同时有 JSON/CSV 和 Markdown summary。

需要重跑或判为失败：

- LCA checkpoint 无法加载。
- random L 实际没有覆盖 `L=1/2/3`。
- ChannelSim 产生非法索引或修改 shape。
- Base 和 LCA 使用不同测试集或不同评价脚本。
- LCA 在 `L=3` full-depth 严重退化且没有记录原因。

## 9. 风险与注意事项

- 第一版 `index dropout` 必须使用 previous-index replacement，不引入 mask token，以避免改动量化/解码路径。
- `light index substitution` 必须替换为合法码本地址。
- 不要把本实验写成强信道鲁棒实验；仅称 mild index-level perturbation。
- end-to-end fine-tuning 是主线；decoder-only 只能作为诊断。
- 若 LCA 改善 `L=1` 但损害 `L=3`，需在 summary 中报告 trade-off。
- ASR、speaker embedding、最终 semantic similarity 模型只用于评价，不能参与训练。
- payload 统计应在实验四统一进行；本实验可以报告 ideal bitrate 和 RTF。

## 10. 实验数据与执行过程留存

`run_id`：

```text
exp3_low_load_channel_aware_adaptation_YYYYMMDD_HHMMSS_seed{seed}
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

- Base provenance：`configs/base_run_reference.json`，记录实验二 run_id、config、checkpoint hash。
- LCA 配置：`configs/lca_finetune_config.json`。
- ChannelSim 配置：`configs/channel_sim_config.json`。
- Random L 配置：`configs/random_l_sampling.json`。
- 数据划分和样本列表：`artifacts/train_files.txt`、`valid_files.txt`、`test_files.txt`、`fixed_sample_list.txt`。
- 命令：`commands/run_command.txt`。
- 环境：`reports/environment.md`。
- 日志：`logs/stdout.log`、`logs/stderr.log`、`logs/train_lca.log`。
- checkpoint：`checkpoints/SCIT-Speech-LCA_best.pt` 和 manifest。
- metrics：`metrics/lca_train_metrics.jsonl`、`metrics/base_vs_lca_results.json/csv`、`metrics/channel_sim_stats.json/csv`。
- samples：原始音频、Base/LCA 在 `L=1/2/3` 和各 channel condition 下的输出。
- failure：失败时写 `reports/failure_report.md`，保留失败日志和最后 checkpoint。

可复现实验所必需文件：

- `configs/lca_finetune_config.json`
- `configs/channel_sim_config.json`
- `configs/random_l_sampling.json`
- `configs/base_run_reference.json`
- `commands/run_command.txt`
- `reports/environment.md`
- `artifacts/*_files.txt`
- `checkpoints/SCIT-Speech-LCA_best.pt`
- `logs/stdout.log`
- `logs/stderr.log`

论文作图/写表所需文件：

- `metrics/base_vs_lca_results.csv`
- `metrics/channel_sim_stats.csv`
- `samples/**`
- `reports/base_vs_lca_summary.md`
- `reports/summary.md`

失败实验如何留存：

- 不删除失败 checkpoint 和日志。
- `failure_report.md` 记录失败阶段：Base load、ChannelSim、train、eval、sample export 或 metrics。
- 若缺少新增脚本，写清脚本职责、输入、输出，不编造执行结果。

多次 run 如何区分：

- 每次新的 seed、扰动概率、Base checkpoint 或超参数变化都生成新 `run_id`。
- 在 `configs/base_run_reference.json` 中记录父实验。

最终 summary 如何生成：

- 从 `base_vs_lca_results.csv` 自动汇总均值/方差和退化幅度。
- summary 区分 evidence 和 interpretation；未跑的条件写“未执行”。

## 11. 交给执行型 AI 的提示词

你要执行“实验三：低负载/轻量信道感知适配”。请先阅读 `output/doc/experiment_plans/exp3_low_load_channel_aware_adaptation.md`、`output/doc/实验手册.md`、实验二 run 目录中的配置和 checkpoint、`speechtokenizer/model.py`、`speechtokenizer/trainer/trainer.py`、`speechtokenizer/trainer/loss.py` 和 `output/experiments/README.md`。不得编造 Base/LCA 指标、checkpoint、样本或扰动结果。

请生成唯一 `run_id=exp3_low_load_channel_aware_adaptation_YYYYMMDD_HHMMSS_seed{seed}`，在 `output/experiments/{run_id}/` 下保存配置、命令、日志、checkpoint、metrics、samples、reports 和 artifacts。必须保存 Base checkpoint provenance、LCA 微调配置、random L sampling 配置、ChannelSim 配置、实际命令、stdout/stderr、环境信息、随机种子、训练/验证/测试清单、固定样本列表、LCA checkpoint、Base vs LCA 指标、`L=1/2/3` 输出样本和失败记录。

默认使用 end-to-end fine-tuning。训练时必须实现 random `L in {1,2,3}` sampling，并包含 clean、index dropout 和 light index substitution；index dropout 第一版使用 previous-index replacement，substitution 必须替换为合法 codebook index。评估时比较 `SCIT-Speech-Base` 与 `SCIT-Speech-LCA`。遇到缺失脚本、缺失数据、Base checkpoint 不可加载、ChannelSim 产生非法索引、训练失败或评估失败时，不要编造结果；请写 `reports/failure_report.md`，说明缺失脚本职责、输入、输出和下一步建议。

## 12. 交给 image2 的 PPT 图片提示词

建议标题：

```text
实验三：低负载与轻量信道感知适配，从 Base 到 LCA
```

建议副标题：

```text
通过 random L sampling 和 mild index-level perturbation，让模型更适应只传前 L 层索引的通信条件
```

image2 提示词：

```text
制作一张 16:9 学术汇报 PPT 实验设计图，主题是“实验三：低负载/轻量信道感知适配”。

整体背景必须是白色或极浅灰色，正式科研汇报训练与评估图风格，深蓝标题，绿色表示 RVQ indices 和低负载，青蓝表示 Base/LCA 模型路径，琥珀色表示轻量信道扰动和评价，灰色表示固定条件。不要真实照片，不要人物，不要复杂 3D。

画面左侧标题：Initialize from Base
画一个 SCIT-Speech-Base checkpoint 输入到 fine-tuning 模块。
标注：
encoder + RVQ codebooks + decoder
end-to-end fine-tuning
not decoder-only main route

画面中间标题：Low-load and channel-aware training
画一个 batch 训练流程：
speech x → encode → RVQ indices I → sample L from {1,2,3} → retain I_L → ChannelSim → reconstruct x_hat_{L,c}

在 sample L 模块旁边画三个小按钮：
L=1, 500 bps ideal
L=2, 1000 bps ideal
L=3, 1500 bps ideal
用绿色表示它们是负载操作点。

在 ChannelSim 模块中画三种轻量条件：
clean
index dropout with previous-index replacement
light index substitution
用琥珀色小标识强调这是 mild index-level perturbation，不是强信道鲁棒实验。

画面右侧标题：Base vs LCA evaluation
画一个矩阵表格，行是模型：
SCIT-Speech-Base
SCIT-Speech-LCA
列是：
L=1
L=2
L=3
clean
dropout-low
substitution-low
表格中只画占位勾选和空白指标格，不填真实结果。

右下角放指标框：
WER / CER
STOI
PESQ / ViSQOL
semantic similarity
RTF
degradation under perturbation

底部结论条：
实验三重点验证 LCA 是否改善 L=1 和 L=2 的可用性，并观察轻量索引扰动下的退化幅度；不宣称强信道鲁棒。

视觉风格：
白底或极浅灰背景，Base 到 LCA 的主箭头清楚，random L 用绿色分支，ChannelSim 用琥珀色轻量扰动图标，评估矩阵简洁对齐。文字要短，不要让表格拥挤。

注意事项：
不要把 dropout 画成波形丢失；扰动作用在离散索引矩阵 I_L 上。
不要引入 mask token，第一版 dropout 使用 previous-index replacement。
不要把 substitution 画成非法索引；替换后仍是合法 codebook address。
不要写强信道鲁棒、高 BER、复杂 FEC 等主张。
不要编造 Base/LCA 指标数值。
```
