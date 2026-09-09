# 实验名称：实验五 消融与诊断

## 1. 实验目的

本实验验证 `SCIT-Speech` 实验链条中的关键设计是否必要，并诊断失败或退化来源。它不是扩大版主实验，也不用于强信道鲁棒性宣称；它用于回答：

- NAS encoder 是否相对 hand-designed encoder 带来计算效率收益，并保持低负载可用性。
- `SCIT-Speech-LCA` 是否相对 `SCIT-Speech-Base` 改善低负载操作点。
- semantic distillation 是否有助于低层索引保留内容信息。
- random L sampling 是否有助于截层传输。
- 可选：ChannelSim on/off 是否提升 mild index perturbation 下的稳定性。

本实验应服务论文的“哪些设计真的有用”，避免把消融矩阵扩展到 `M/K` 大范围搜索、高 BER、复杂 FEC 或第一层 probing 大实验。

## 2. 实验输入

| 输入 | 当前路径或建议路径 | 说明 |
|---|---|---|
| 实验手册 | `output/doc/实验手册.md` | 消融边界 |
| 实验一产物 | `output/experiments/{exp1_run_id}/` | hand-designed vs NAS encoder 的候选和 Pareto 记录 |
| 实验二产物 | `output/experiments/{exp2_run_id}/` | `SCIT-Speech-Base` checkpoint |
| 实验三产物 | `output/experiments/{exp3_run_id}/` | `SCIT-Speech-LCA` checkpoint、ChannelSim 配置 |
| 主配置 | `config/spt_base_cfg.json` | 生成消融配置的基础 |
| 训练入口 | `scripts/train_example.py`、需要新增 `scripts/train_lca.py` | Base 和 LCA 训练入口 |
| 模型代码 | `speechtokenizer/model.py`、`nas/custom_model.py` | 消融模型构建 |
| 评估脚本 | 需要新增 `scripts/evaluate_ablation_variants.py` | 统一评估每个 variant |

需要新增脚本：

| 建议路径 | 职责 | 输入 | 输出 |
|---|---|---|---|
| `scripts/build_ablation_configs.py` | 生成每个 ablation variant 的配置，保证只改变目标变量 | base config、variant spec | `configs/variants/*.json` |
| `scripts/train_ablation_variant.py` | 按 variant config 训练或微调 | variant config、可选 checkpoint | checkpoint、logs |
| `scripts/evaluate_ablation_variants.py` | 对所有 variant 统一跑 `L=1/2/3` 和必要 channel 条件 | variant manifest、test list | metrics、samples |
| `scripts/compare_ablation_results.py` | 汇总差异并生成报告 | metrics CSV | `reports/ablation_comparison.md` |
| `scripts/collect_environment.py` | 保存环境 | run 目录 | `reports/environment.md/json` |

## 3. 实验输出

推荐 `run_id`：

```text
exp5_ablation_and_diagnosis_YYYYMMDD_HHMMSS_seed{seed}
```

输出目录：

```text
output/experiments/{run_id}/
```

应产生：

- `configs/ablation_matrix.json`：消融矩阵定义。
- `configs/variants/{variant_id}.json`：每个 variant 的冻结配置。
- `commands/run_command.txt`：全部训练/评估命令。
- `logs/{variant_id}/stdout.log`、`stderr.log`。
- `checkpoints/{variant_id}/`：每个 variant checkpoint 或引用 manifest。
- `metrics/ablation_results.json/csv`。
- `metrics/compute_profile.json/csv`。
- `samples/{variant_id}/L1/`、`L2/`、`L3/`。
- `reports/ablation_comparison.md`、`reports/variant_manifest.md`、`reports/summary.md`。
- `reports/failure_report.md`：如果有 variant 失败。

## 4. 实验变量与对照

必做消融：

| 消融 | 对照组 | 变量组 | 只允许改变 |
|---|---|---|---|
| hand-designed encoder vs NAS encoder | hand-designed encoder | NAS-searched encoder | encoder 架构；decoder 需固定，若不能固定需另列 variant |
| Base vs LCA | `SCIT-Speech-Base` | `SCIT-Speech-LCA` | LCA 训练目标和 ChannelSim/random L |
| with vs without semantic distillation | `distill_loss_lambda > 0` | `distill_loss_lambda = 0` | 语义蒸馏 loss 权重 |
| with vs without random L sampling | random L on | random L off / full-depth only | L 采样策略 |

可选消融：

| 消融 | 对照组 | 变量组 | 条件 |
|---|---|---|---|
| ChannelSim on/off | LCA without ChannelSim | LCA with ChannelSim | 仅当实验三扰动评估稳定时执行 |

固定条件：

- 同一训练/验证/测试划分。
- 同一 `M=3`、`K=1024`、sample rate、latent rate。
- 同一评价脚本、固定样本、ASR/metric 模型。
- 同一 payload 统计口径。

## 5. 详细执行步骤

### 步骤 1：创建 run 目录

- 操作目标：为所有消融 variant 建立统一目录。
- 涉及文件或脚本：需要新增 `scripts/create_experiment_run.py`。
- 输入：`run_id=exp5_ablation_and_diagnosis_YYYYMMDD_HHMMSS_seed{seed}`。
- 输出：标准 run 目录。
- 检查点：目录结构完整。
- 失败时如何判断问题：run 重名或目录缺失时停止。

### 步骤 2：定义 ablation matrix

- 操作目标：冻结本次要跑的 variant，避免中途改变对照口径。
- 涉及文件或脚本：需要新增 `scripts/build_ablation_configs.py`。
- 输入：实验一/二/三 run references、base config。
- 输出：`configs/ablation_matrix.json`、`reports/variant_manifest.md`。
- 检查点：每个 variant 有唯一 `variant_id`、父 run、改变项、固定项。
- 失败时如何判断问题：如果一个 variant 同时改变多个因素，不能作为单因素消融。

### 步骤 3：保存环境和数据划分

- 操作目标：确保多 variant 可复现。
- 涉及文件或脚本：需要新增 `scripts/collect_environment.py`。
- 输入：当前环境、test list。
- 输出：`reports/environment.md`、`artifacts/train_files.txt`、`valid_files.txt`、`test_files.txt`、`fixed_sample_list.txt`。
- 检查点：所有 variant 使用同一数据划分。
- 失败时如何判断问题：variant 使用不同 test list 时，结果不可比较。

### 步骤 4：消融一 hand-designed encoder vs NAS encoder

- 操作目标：验证 NAS 发送端 encoder 的效率和质量影响。
- 涉及文件或脚本：`nas/custom_model.py`、需要新增 `nas/encoder_only_model_variant.py`、`scripts/evaluate_ablation_variants.py`。
- 输入：hand-designed config、NAS encoder config、固定 decoder config。
- 输出：variant checkpoint 或引用、`metrics/compute_profile.csv`、`metrics/ablation_results.csv`。
- 检查点：主口径必须是 fixed decoder；如果只能使用 `NASSpeechTokenizer` 同时替换 decoder，variant 名称必须写 `nas_encoder_mirrored_decoder`。
- 失败时如何判断问题：decoder 变化未拆分时，不能把质量差异归因于 encoder NAS。

### 步骤 5：消融二 Base vs LCA

- 操作目标：验证 LCA 低负载适配收益。
- 涉及文件或脚本：实验二/三产物、`scripts/evaluate_ablation_variants.py`。
- 输入：Base checkpoint、LCA checkpoint、同一 test list。
- 输出：`metrics/ablation_results.csv` 中 Base/LCA 对比行。
- 检查点：重点比较 `L=1/2` clean 和 mild perturbation；`L=3` 作为上限保护。
- 失败时如何判断问题：如果 Base/LCA 来自不同数据或不同 encoder 架构，需在报告中标记 confound。

### 步骤 6：消融三 with/without semantic distillation

- 操作目标：验证语义蒸馏是否帮助低负载内容保持。
- 涉及文件或脚本：`speechtokenizer/trainer/loss.py`、`scripts/train_ablation_variant.py`。
- 输入：两个只差 `distill_loss_lambda` 的配置。
- 输出：两个 checkpoint、loss、`L=1/2/3` 指标和样本。
- 检查点：除 `distill_loss_lambda` 和必要日志标签外，其他配置保持一致。
- 失败时如何判断问题：若 no-distill 训练不收敛，保留失败 run，并作为诊断记录，不补假指标。

### 步骤 7：消融四 with/without random L sampling

- 操作目标：验证训练阶段随机截层是否提升低负载可用性。
- 涉及文件或脚本：`speechtokenizer/trainer/lca_trainer.py`、`scripts/train_ablation_variant.py`。
- 输入：random L on 配置、random L off/full-depth only 配置。
- 输出：checkpoint、训练采样日志、`L=1/2/3` 指标。
- 检查点：random L off variant 仍使用相同数据和训练步数。
- 失败时如何判断问题：若 off variant 实际仍随机采样 L，则配置错误，需重跑。

### 步骤 8：可选 ChannelSim on/off

- 操作目标：诊断 mild index perturbation 训练是否有收益。
- 涉及文件或脚本：`scripts/channel_sim.py`、`speechtokenizer/trainer/lca_trainer.py`。
- 输入：LCA with/without ChannelSim 配置。
- 输出：clean 和扰动条件下指标对比。
- 检查点：只在实验三 ChannelSim 实现稳定时执行。
- 失败时如何判断问题：若扰动收益不稳定，不作为主消融，只在报告中说明。

### 步骤 9：统一评估所有 variant

- 操作目标：用同一脚本生成可比指标和样本。
- 涉及文件或脚本：需要新增 `scripts/evaluate_ablation_variants.py`。
- 输入：`configs/ablation_matrix.json`、所有 checkpoint、test list。
- 输出：`metrics/ablation_results.json/csv`、`samples/{variant_id}/`。
- 检查点：所有 variant 都有 `L=1/2/3`；必要 perturbation 条件一致。
- 失败时如何判断问题：某 variant 缺 checkpoint 时，该 variant 标记 failed/skipped，不影响其他 variant。

### 步骤 10：生成消融报告

- 操作目标：把单因素差异转成论文可用表格和诊断结论。
- 涉及文件或脚本：需要新增 `scripts/compare_ablation_results.py`。
- 输入：`metrics/ablation_results.csv`、`metrics/compute_profile.csv`、samples manifest。
- 输出：`reports/ablation_comparison.md`、`reports/summary.md`。
- 检查点：报告必须区分“已验证”“不稳定”“未执行”“失败”。
- 失败时如何判断问题：如果报告中出现没有 metric 文件支持的数值，必须删除。

## 6. 指标与统计方式

| 指标 | 用途 | 说明 |
|---|---|---|
| WER/CER | 低负载可懂度 | 特别关注 `L=1/2` |
| STOI | 可懂度 | 与 WER/CER 互补 |
| PESQ/ViSQOL | 自然度/质量 | 不作为唯一结论 |
| Semantic similarity | 内容保持 | semantic distillation 消融重点 |
| RTF | 实时性 | NAS 消融重点 |
| Params/MACs | 计算成本 | encoder-only 和 total 都可保存，但主结论看 encoder-only |
| Payload | 传输负载 | 同一 `L` 下固定，不因 variant 改变 |
| Codebook usage/perplexity/dead code ratio | 诊断码本 collapse | distillation/random L 可能影响 |
| Degradation under perturbation | ChannelSim 诊断 | clean vs mild perturbation 差值 |

统计原则：

- 同一消融只改变一个变量。
- 每个 variant 保存独立配置、checkpoint、日志和样本。
- 指标聚合保存 mean/std/sample_count，不填假数。

## 7. 结果记录格式

### Table A：Variant manifest

| run_id | variant_id | ablation_group | parent_run_id | changed_factor | fixed_factors | config_path | checkpoint_path | status | notes |
|---|---|---|---|---|---|---|---|---|---|

### Table B：Ablation results

| run_id | variant_id | ablation_group | L | channel | ideal_bitrate_bps | WER | CER | STOI | PESQ | ViSQOL | semantic_similarity | RTF | sample_count | notes |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|

### Table C：Compute profile

| run_id | variant_id | encoder_params | encoder_macs | encoder_rtf | total_params | total_macs | total_rtf | profile_sample_sec | notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|

### Table D：Codebook diagnostics

| run_id | variant_id | layer | usage_rate | perplexity | dead_code_ratio | top1_frequency | notes |
|---|---|---:|---:|---:|---:|---:|---|

## 8. 成功标准

最低完成标准：

- 必做四类消融的配置、命令、日志、checkpoint 或失败记录均保存。
- 所有成功 variant 用同一测试集、同一评价脚本完成 `L=1/2/3` 评估。
- hand-designed vs NAS encoder 明确拆分 decoder 影响。
- Base vs LCA 对比引用实验二/三真实 checkpoint。
- with/without semantic distillation 和 with/without random L sampling 不改变其他条件。
- 报告中明确失败/skipped variant。

需要重跑或判为失败：

- 一个 variant 同时改变多个关键因素。
- 缺失配置副本或命令。
- 只保存最终指标，没有 stdout/stderr 或 checkpoint provenance。
- NAS 消融把 decoder 变化混入 encoder 结论。
- 缺少固定样本或测试清单。

## 9. 风险与注意事项

- 消融成本高，优先跑最小必要矩阵，不做 `M/K` 大规模搜索。
- 不要把 semantic distillation 写成新 tokenizer 贡献；它是训练辅助。
- NAS 结论必须控制 decoder。
- random L 和 ChannelSim 可能相互作用，先做单因素，再解释组合。
- 若某消融训练失败，不要删除；失败本身可作为诊断。
- 不要让不同训练时长或不同数据划分造成伪差异。

## 10. 实验数据与执行过程留存

`run_id`：

```text
exp5_ablation_and_diagnosis_YYYYMMDD_HHMMSS_seed{seed}
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

- `configs/ablation_matrix.json`。
- 每个 variant 的 `configs/variants/{variant_id}.json`。
- parent run references：实验一/二/三 run_id、config、checkpoint hash。
- 数据清单：train/valid/test/fixed sample list。
- 命令：总命令和每个 variant 命令。
- 环境：`reports/environment.md`。
- 日志：每个 variant 的 stdout/stderr。
- checkpoint：每个成功训练 variant 的 checkpoint 或引用 manifest。
- metrics：`ablation_results.json/csv`、`compute_profile.json/csv`、codebook diagnostics。
- samples：每个 variant 的固定样本原始音频和 `L=1/2/3` 输出。
- failure：`reports/failure_report.md` 和每个失败 variant 的失败原因。

可复现实验所必需文件：

- `configs/ablation_matrix.json`
- `configs/variants/*.json`
- `commands/run_command.txt`
- `reports/environment.md`
- `artifacts/*_files.txt`
- `checkpoints/*` 或 checkpoint reference manifest
- `logs/{variant_id}/stdout.log`
- `logs/{variant_id}/stderr.log`

论文作图/写表所需文件：

- `metrics/ablation_results.csv`
- `metrics/compute_profile.csv`
- `metrics/codebook_diagnostics.csv`
- `samples/sample_manifest.csv`
- `reports/ablation_comparison.md`
- `reports/summary.md`

失败实验如何留存：

- 每个失败 variant 写明失败阶段、错误摘要、是否影响整组消融。
- 失败 variant 在 `variant_manifest` 中状态为 `failed`，不删除目录。

多次 run 如何区分：

- 不同 ablation matrix、不同 parent checkpoint、不同 seed 都生成新 run_id。
- 每个 variant_id 在 run 内唯一。

最终 summary 如何生成：

- 从 `variant_manifest` 和 `ablation_results.csv` 生成。
- 按 ablation group 分段报告，不把不同 group 混成一个总排名。

## 11. 交给执行型 AI 的提示词

你要执行“实验五：消融与诊断”。请先阅读 `output/doc/experiment_plans/exp5_ablation_and_diagnosis.md`、`output/doc/实验手册.md`、实验一/二/三 run 目录、`config/spt_base_cfg.json`、`speechtokenizer/model.py`、`speechtokenizer/trainer/loss.py`、`nas/custom_model.py` 和 `output/experiments/README.md`。不得编造消融结果、checkpoint、指标或样本。

请生成唯一 `run_id=exp5_ablation_and_diagnosis_YYYYMMDD_HHMMSS_seed{seed}`，在 `output/experiments/{run_id}/` 下保存 configs、commands、logs、checkpoints、metrics、samples、reports、artifacts。必须保存 ablation matrix、每个 variant 配置、父实验 run reference、实际命令、stdout/stderr、环境信息、随机种子、数据划分、checkpoint 或失败记录、`L=1/2/3` 样本、指标表和对比报告。

必做消融包括 hand-designed encoder vs NAS encoder、Base vs LCA、with/without semantic distillation、with/without random L sampling；可选 ChannelSim on/off。每个消融只允许改变目标变量。NAS 消融必须提醒并控制 decoder 影响：如果当前工程只能同时替换 decoder，请把该条件标为 implementation variant，不要把差异归因于发送端 encoder。遇到缺失脚本、缺失数据、variant 训练失败或评价失败时，不要补假数；请保存失败日志并写 `reports/failure_report.md`。
