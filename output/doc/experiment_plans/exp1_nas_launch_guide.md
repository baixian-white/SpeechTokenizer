# Experiment 1 Teacher-guided NAS 启动说明

本文档只说明如何手动启动实验一脚本。实验范围仍然是：

```text
teacher-guided encoder-side NAS under a fixed RVQ index interface
```

不要用本文档启动实验二到实验六。

## 1. 脚本入口

调试/单阶段入口：

```text
nas/evaluate_encoder_proxy.py
```

正式四阶段入口：

```text
nas/run_staged_encoder_nas.py
```

注意：本文档描述的是新版 teacher-guided 目标流程。若当前脚本尚未支持 `--teacher-config`、`--teacher-checkpoint`、`--teacher-target` 和 `--teacher-cache-mode`，必须先完成脚本改造；旧版 non-teacher run 不能写成新版实验一结果。

日志说明：Exp1 NAS 脚本会自动把终端输出同步写入 `logs/stdout.log`，把错误输出同步写入 `logs/stderr.log`。启动命令不要再使用 PowerShell `1>` / `2>` 重定向，否则终端不会实时显示日志。

正式 NAS 使用 `run_staged_encoder_nas.py`。新版脚本应执行：

```text
stage1: random sample candidates -> interface/profile only
stage2: top-k -> short distillation -> SpeechTokenizer teacher latent alignment
stage3: top-k -> longer short distillation -> RVQ compatibility with frozen teacher quantizer
stage4: final top-k -> final short distillation + frozen decoder proxy + multi-objective Pareto + best architecture
```

新版选择逻辑不是纯轻量化打分。stage2/3/final 会先只训练候选 B 的 encoder，使其短程模仿冻结 teacher A 的 pre-RVQ latent；随后使用 teacher-guided quality-constrained proxy score：

```text
teacher penalty:
  teacher_latent_smooth_l1
  teacher_latent_cosine_distance
  teacher_temporal_delta_loss

RVQ compatibility penalty:
  rvq_code_agreement
  rvq_quantized_feature_l1

reconstruction proxy penalty:
  proxy_recon_l1
  proxy_mel_loss

resource score:
  encoder_macs
  encoder_params
  encoder_rtf_mean
```

短程蒸馏 loss：

```text
SmoothL1(B(x), A(x))
+ cosine_distance(B(x), A(x))
+ temporal_delta_loss(ΔB(x), ΔA(x))
```

蒸馏时只更新 NAS candidate encoder。SpeechTokenizer teacher、transform、RVQ quantizer 和 decoder 均冻结。proxy reconstruction 使用冻结的预训练 SpeechTokenizer decoder，不再使用随机初始化或 geometry-matched decoder 作为主口径。

默认 `--selection-mode balanced` 会把主搜索空间收紧为：

```text
n_filters >= 24
count(skip) <= 1
lstm in {1, 2}
```

这样资源优势只在质量没有明显劣化的候选之间起主要作用。

Teacher A 是冻结的预训练 SpeechTokenizer encoder。它只作为候选 B 的 pre-RVQ latent 锚点，不作为本文方法主语，也不参与梯度更新。

## 2. 进入仓库

PowerShell：

```powershell
cd H:\H-CODE\speechtokenizer
$PY = "C:\Users\Windows11\.conda\envs\speechtokenizer\python.exe"
& $PY --version
```

## 3. 固定输入路径

默认配置：

```text
config/spt_base_cfg.json
```

默认训练清单：

```text
data/SpeechPretrain/hubert_rep/LibriSpeech/train_files.txt
```

默认 teacher A 路径：

```text
model_hub/speechtokenizer_hubert_avg/config.json
model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt
```

如果本机 teacher A 路径不同，启动命令必须显式传入实际路径，并在 run 的 `configs/teacher_model_reference.json` 中保存。

脚本会检查并固定：

```text
sample_rate = 16000
dimension = 1024
n_q = 3
codebook_size = 1024
prod(encoder_strides) = 320
latent_rate = 50 steps/s
teacher_encoder = frozen
teacher_transform = frozen
teacher_quantizer = frozen, compatibility check only
teacher_decoder = frozen, proxy reconstruction only
```

不要为了跑通而修改这些条件。

## 4. 语法检查

```powershell
& $PY -m py_compile `
  nas\search_space.py `
  nas\teacher_guided_proxy.py `
  nas\validate_short_distill.py `
  nas\evaluate_encoder_proxy.py `
  nas\run_staged_encoder_nas.py `
  nas\encoder_only_model_variant.py `
  nas\SeaNet.py `
  nas\model_components.py
```

如果后续新增 `nas\teacher_guided_proxy.py`、`nas\validate_short_distill.py` 或等价模块，应把它加入语法检查。若 teacher-guided/short-distillation 逻辑缺失，不能把 run 写成新版实验一。

## 5. Staged Smoke

CPU 上完整 proxy forward 可能很慢。建议优先用 GPU 做小 smoke：

```powershell
$SEED = 42
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
$RUN_ID = "exp1_nas_semantic_encoder_stagedsmoke_${TS}_seed${SEED}"
$RUN = "output\experiments\$RUN_ID"
New-Item -ItemType Directory -Force "$RUN\logs" | Out-Null

& $PY nas\run_staged_encoder_nas.py `
  --run-dir $RUN `
  --config config\spt_base_cfg.json `
  --manifest data\SpeechPretrain\hubert_rep\LibriSpeech\train_files.txt `
  --teacher-config model_hub\speechtokenizer_hubert_avg\config.json `
  --teacher-checkpoint model_hub\speechtokenizer_hubert_avg\SpeechTokenizer.pt `
  --teacher-target pre_rvq `
  --seed $SEED `
  --stage1-candidates 8 `
  --stage2-top-k 4 `
  --stage3-top-k 2 `
  --final-top-k 1 `
  --stage2-max-samples 1 `
  --stage2-max-batches 1 `
  --stage3-max-samples 1 `
  --stage3-max-batches 1 `
  --final-max-samples 1 `
  --final-max-batches 1 `
  --teacher-cache-mode none `
  --selection-mode balanced `
  --distill-stage2-steps 1 `
  --distill-stage3-steps 1 `
  --distill-final-steps 1 `
  --distill-log-every 1 `
  --device cuda
```

检查：

```powershell
Get-Content "$RUN\reports\status.json"
Get-Content "$RUN\reports\summary.md"
Get-Content "$RUN\reports\staged_selection.md"
Get-ChildItem "$RUN\metrics"
```

期望至少生成：

```text
metrics/stage1_profile.csv/json
metrics/stage2_teacher_alignment.csv/json
metrics/stage3_rvq_compatibility.csv/json
metrics/stage4_final.csv/json
metrics/teacher_alignment.csv/json
metrics/rvq_compatibility.csv/json
metrics/pareto_frontier.csv/json
artifacts/stages/stage1_to_stage2_selected_candidates.json
artifacts/stages/stage2_to_stage3_selected_candidates.json
artifacts/stages/stage3_to_final_selected_candidates.json
artifacts/best_architecture/best_seanet_config.json
configs/teacher_model_reference.json
reports/teacher_condition.md
reports/teacher_guided_selection.md
reports/staged_selection.md
reports/status.json
```

## 6. 推荐正式 Run

这是当前更合理的主实验预算。相比旧的 `2048 -> 256 -> 32 -> 3`，它扩大初始候选数，并保留更多 final 候选用于 Pareto：

```powershell
$SEED = 42
$TS = Get-Date -Format "yyyyMMdd_HHmmss"
$RUN_ID = "exp1_nas_semantic_encoder_${TS}_seed${SEED}"
$RUN = "output\experiments\$RUN_ID"
New-Item -ItemType Directory -Force "$RUN\logs" | Out-Null

& $PY nas\run_staged_encoder_nas.py `
  --run-dir $RUN `
  --config config\spt_base_cfg.json `
  --manifest data\SpeechPretrain\hubert_rep\LibriSpeech\train_files.txt `
  --teacher-config model_hub\speechtokenizer_hubert_avg\config.json `
  --teacher-checkpoint model_hub\speechtokenizer_hubert_avg\SpeechTokenizer.pt `
  --teacher-target pre_rvq `
  --seed $SEED `
  --stage1-candidates 8192 `
  --stage2-top-k 512 `
  --stage3-top-k 64 `
  --final-top-k 8 `
  --stage2-max-samples 8 `
  --stage2-max-batches 8 `
  --stage3-max-samples 16 `
  --stage3-max-batches 16 `
  --final-max-samples 32 `
  --final-max-batches 32 `
  --teacher-cache-mode disk `
  --selection-mode balanced `
  --distill-stage2-steps 100 `
  --distill-stage3-steps 500 `
  --distill-final-steps 1500 `
  --distill-lr 0.0001 `
  --distill-log-every 0 `
  --device cuda
```

如果这版耗时过长，可以先把预算降到：

```text
4096 -> 256 -> 32 -> 8
```

但报告里必须标记为 reduced-budget proxy NAS。

## 7. 可选质量优先复核

如果 balanced run 的 best architecture 仍然过轻，使用质量优先模式复核：

```powershell
& $PY nas\run_staged_encoder_nas.py `
  --run-dir $RUN `
  --config config\spt_base_cfg.json `
  --manifest data\SpeechPretrain\hubert_rep\LibriSpeech\train_files.txt `
  --teacher-config model_hub\speechtokenizer_hubert_avg\config.json `
  --teacher-checkpoint model_hub\speechtokenizer_hubert_avg\SpeechTokenizer.pt `
  --teacher-target pre_rvq `
  --seed $SEED `
  --stage1-candidates 4096 `
  --stage2-top-k 256 `
  --stage3-top-k 64 `
  --final-top-k 8 `
  --stage2-max-samples 8 `
  --stage2-max-batches 8 `
  --stage3-max-samples 16 `
  --stage3-max-batches 16 `
  --final-max-samples 32 `
  --final-max-batches 32 `
  --teacher-cache-mode disk `
  --selection-mode quality `
  --distill-stage2-steps 100 `
  --distill-stage3-steps 500 `
  --distill-final-steps 1500 `
  --distill-lr 0.0001 `
  --distill-log-every 0 `
  --device cuda
```

`quality` 模式默认：

```text
n_filters >= 32
count(skip) <= 1
quality margins = 5%
resource weights lower than balanced
```

## 8. 监控运行

查看 stdout：

```powershell
Get-Content "$RUN\logs\stdout.log" -Wait -Tail 40
```

查看 stderr：

```powershell
Get-Content "$RUN\logs\stderr.log" -Wait -Tail 40
```

查看最终状态：

```powershell
Get-Content "$RUN\reports\status.json"
```

如果失败：

```powershell
Get-Content "$RUN\reports\failure_report.md" -TotalCount 120
```

## 9. 完成后重点检查

接口检查：

```powershell
Get-Content "$RUN\reports\encoder_interface_check.md"
```

decoder 条件：

```powershell
Get-Content "$RUN\reports\teacher_condition.md"
Get-Content "$RUN\reports\teacher_guided_selection.md"
Get-Content "$RUN\reports\decoder_condition.md"
Get-Content "$RUN\reports\decoder_contamination_check.md"
```

Staged selection 和 Pareto：

```powershell
Get-Content "$RUN\reports\staged_selection.md"
Get-Content "$RUN\reports\pareto_selection.md"
Get-ChildItem "$RUN\artifacts\best_architecture"
```

结果表：

```powershell
Get-ChildItem "$RUN\metrics"
```

`metrics/stage*.csv` 应包含：

```text
stage_teacher_penalty
stage_rvq_penalty
stage_recon_penalty
stage_mel_penalty
stage_quality_penalty
stage_resource_score
```

`metrics/teacher_alignment.csv/json` 应包含：

```text
teacher_latent_smooth_l1
teacher_latent_cosine_distance
teacher_temporal_delta_loss
```

`metrics/rvq_compatibility.csv/json` 应包含：

```text
rvq_code_agreement
rvq_code_flip_rate
rvq_quantized_feature_l1
```

## 10. 成功 Run 的关键产物

下游实验二如果使用 teacher-guided NAS encoder，应引用：

```text
output/experiments/{run_id}/artifacts/best_architecture/best_seanet_config.json
```

并明确写：

```text
encoder route = teacher-guided NAS encoder from Exp1
teacher anchor = frozen pretrained SpeechTokenizer encoder
decoder_condition = frozen_teacher_decoder
```

实验二可以选择只使用 `best_seanet_config.json` 从头完整训练，也可以将 teacher-guided 短训权重作为初始化；无论哪种方式，论文最终结果必须来自实验二完整训练，不得直接引用实验一 proxy run 作为最终通信结果。

如果 Exp1 失败或人工决定不用 NAS，应在实验二写：

```text
encoder route = hand-designed encoder fallback
```

不要声称使用了 NAS encoder。

## 11. 必须停止或判失败的情况

看到以下情况不要继续加大预算：

- `reports/status.json` 中 `status = aborted`
- 出现 `failure_report.md`
- teacher config/checkpoint 无法加载
- teacher encoder 或 teacher quantizer 没有冻结
- `encoder_interface_check.md` 中不是 `[1, 1024, 50]`
- `codes_shape` 第一维不是 `3`
- `teacher_alignment` 指标缺失
- `rvq_code_agreement` 极低且报告未解释
- decoder 输出不是 `[B, 1, 16000]`
- loss 出现 NaN/Inf
- 连续 OOM
- 数据清单归一化后 `written = 0`
- `decoder_contamination_check.md` 显示 decoder op 被搜索

这些情况先修脚本或数据，再重新跑 smoke。
