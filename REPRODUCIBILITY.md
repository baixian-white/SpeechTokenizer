# REPRODUCIBILITY.md — SCIT-Speech 论文复现指南

本文件把论文 `output/doc/paper_drafts/scit_speech_method_cn_draft_20260609.md` 的每张表/图
映射到复现要素：**产出脚本 · 输入数据/sample list · 权重 sha256 + run_id · 随机种子 · 产物路径**。
数据基准为规范实验日志 `output/doc/实验记录.md`（§1–§19）。

> **路径约定**：所有相对路径以仓库根 `H:\H-CODE\speechtokenizer` 为基准。
> **环境**：conda env `speechtokenizer`；`conda activate speechtokenizer`。
> **大权重 caveat**：`.pt/.pth` 已被 `.gitignore` 忽略（不入库），但其 sha256、所属 run_id、
> 配置与指标 CSV 均登记于此与各 run 的 `reports/`，构成可追溯证据。`output/` 整体当前未入库
> （保存于本地磁盘），复现需在原机或恢复 `output/` 后执行。

---

## 0. 核心模型权重台账（溯源链根）

| 模型 | run_id | 权重相对路径 | sha256 | seed |
|---|---|---|---|---|
| SCIT-Speech-Base | `exp2_scit_speech_distill30_retrain_20260529_seed42` | `output/experiments/exp2_.../checkpoints/SCIT-Speech-Base_best.pt` | `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6` | 42 |
| SCIT-Speech-LCA v2（正式）| `exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42` | `output/experiments/exp3_..._v2_.../checkpoints/SCIT-Speech-LCA_best.pt` | `af60223b2f44733f5a43d5a022871dad061dfe2421473c96f5d350c3d4b49a5a` | 42 |
| SCIT-Speech-LCA v1（消融对照）| `exp3_low_load_channel_aware_adaptation_20260530_seed42` | `output/experiments/exp3_..._20260530_.../checkpoints/SCIT-Speech-LCA_best.pt` | `9daa924ff5aeedf2fa4aa1ddb2d7ce7f53297548e7258d0ef8d5b1af255a8609` | 42 |
| A3 distill=0（蒸馏消融）| `exp5_ablation_and_diagnosis_20260601_seed42` | `.../SCIT-Speech-Base_distill0_step42500_extracted.pt` | `84b0ff458fc0084a329795f4d540389fe48dbdcd3a3ccab7df43614c84c877ba` | 42 |
| hand-designed encoder（exp13）| `exp13_handdesigned_encoder_distill30_20260611_seed42` | `.../checkpoints/SCIT-Speech-Base_best.pt` | `e45416b00852e710ef51f7075db504b93b09b0a205b3557317e4b8bbd93836a0` | 42 |

LCA v2 demo 用快照（demo skeleton §X，sha256 与 LCA v2 一致）：
`exp3_..._v2_.../checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`。

NAS 选定架构（exp1 handoff，§3.4 / §6.4）：candidate `nas_seed42_000896`，
config `output/experiments/exp1_nas_distill_run1_seed42/artifacts/best_architecture/best_seanet_config.json`。

---

## 1. 论文图 → 复现

| 图 | 产出脚本 | 输入数据 | 产物路径 | 备注 |
|---|---|---|---|---|
| **图 1** 系统总览 | 手绘/PPT | — | `output/doc/scit_speech_method_cn_draft_20260609论文素材/框架图.png` | 非脚本生成 |
| **图 2** 码率—质量权衡 | `scripts/plot_fig2_rate_quality_tradeoff.py` | exp4 `metrics/{audio_quality_results,asr_results,payload_summary}.csv` | `output/doc/paper_drafts/assets/fig2_*.{png,pdf}` → 素材目录 | ⚠️ **待确认**：脚本读 exp4(8样本)，论文 §5.1 现用 exp12(300样本)；素材存在 `.bak8sample`。图与表 1a/1b 一致性需复核（见 REORG_MANIFEST 待确认①）|
| **图 3** 扰动鲁棒性 | `scripts/plot_fig3_perturbation_robustness.py` | exp8 `eval_perturb_full_lca_nosave/` + exp9 `eval_packet_burst_full_lca_nosave/` | `.../assets/fig3_*.{png,pdf}` → 素材目录 | test-other 全量 n=2939 |
| **图 4** 扰动 ASR WER | （exp11 后处理）| exp11 `eval_perturbed_asr_wer/test-{clean,other}_100/metrics/*.csv` | `.../paper_drafts/...论文素材/fig4_perturbed_asr_wer.png` | §5.5 |
| **图 5** V0–V4 累积鲁棒性 | （exp5b 后处理）| exp5 `eval_unified_n64_20260609/` V0–V4 n=256 CSV | `.../论文素材/fig5_v0v4_cumulative_robustness.png` | §6.3 / §11.1 |

---

## 2. 论文表 → 复现

| 表 | 内容 | run_id / 目录 | 产出/分析脚本 | 输入 sample list | 产物路径 |
|---|---|---|---|---|---|
| **表 1a/1b** §5.1 同码率对比（test-clean/other_300, 95%CI, Wilcoxon）| exp12 | `exp12_baseline_comparison_test300_20260610_seed42` | `scripts/run_exp4_baselines.py` + `scripts/run_opus_baseline.py`；分析 `commands/analyze_baseline_300_stats.py` | exp7 `test-{clean,other}_all_files.txt` 前 300 | `runs/<split>/metrics/per_method_summary.csv`、`reports/baseline_300_summary.md` |
| **表 2** §5.2 全量 clean 泛化 | exp7 | `exp7_librispeech_test_full_clean_20260606` | `scripts/evaluate_clean_large_nosave.py` | `artifacts/test-{clean,other}_all_files.txt`（2620/2939）| `full_clean_results.csv`、`reports/full_clean_summary.md` |
| **表 3** §5.3 索引前帧覆盖/替换 | exp8 | `exp8_librispeech_test_full_perturb_20260606` | `scripts/evaluate_perturb_large_nosave.py` | 同 exp7 全量 | `eval_perturb_full_lca_nosave/.../*.csv` |
| **表 4** §5.4 丢包/突发 | exp9 | `exp9_librispeech_test_full_packet_burst_20260608` | `scripts/evaluate_packet_burst_large_nosave.py`（核心 `evaluate_packet_burst_loss.py`）| 同 exp7 全量 | `eval_packet_burst_full_lca_nosave/.../*.csv` |
| **表 5** §5.5 扰动 ASR/WER | exp11 | `exp11_perturbed_asr_wer_20260609` | `scripts/evaluate_perturbed_asr_wer_onthefly.py` | test-{clean,other}_100 | `eval_perturbed_asr_wer/<split>_100/metrics/perturbed_asr_wer_results.csv` |
| **表 6** §5.5 无扰动 ASR/WER | exp10 | `exp10_asr_wer_onthefly_20260609` | `scripts/evaluate_asr_wer_onthefly.py`（驱动 `run_asr_evaluation.py`）| test-{clean,other}_300 | `eval_asr_wer_clean/.../*.csv` |
| **表 7** §5.6 跨语料 zero-shot | exp19/19b | `exp19_cross_corpus_zeroshot_20260613_seed42`、`exp19b_aishell_zeroshot_20260613_seed42` | `scripts/evaluate_clean_large_nosave.py`；中文 CER `commands/evaluate_asr_cer_zh.py` | `artifacts/vctk_300.txt`、aishell 300 | `eval_*/metrics/full_clean_results.csv` |
| **表 8** §6.4 NAS 终选代理质量 | exp1/1b | `exp1_nas_distill_run1_seed42` | NAS：`nas/run_staged_encoder_nas.py`；后处理 exp1b `reports/pareto_neighbors_20260610/` | NAS 子集 | `metrics/stage4_final.csv`、`nas_stage4_final_with_proxy_quality.csv` |
| **§6.1** 蒸馏消融（dev mel）| exp5 A3 | `exp5_ablation_and_diagnosis_20260601_seed42` | `scripts/train_distill_weight_ablation.py` + `evaluate_ablation_variants.py` | train-clean-100 valid | `reports/ablation_comparison.md` |
| **§6.2** LCA v1/v2 鲁棒性 | exp3 v1/v2 + exp3c | `exp3_lca_v1_v2_stats_20260610` | `scripts/evaluate_lca_vs_base.py`；统计 exp3c `metrics/*` | fixed_sample_list 8 条 | `metrics/v1_v2_*.csv`、`reports/v1_v2_stats_summary.md` |
| **§6.3** V0–V4 因子化（n=256）| exp5b | `exp5_lca_component_factorial_20260603_seed42` | `scripts/evaluate_lca_vs_base.py` + `aggregate_v0_v4_n256.py`；统计 `reports/statistical_tests_20260610/` | 256 train-clean-100 子集 | `factorial_*.csv`、`factorial_stats_summary.md` |
| **附录 D.1/D.2** 完整 23 方法 + Wilcoxon | exp12 | 同表 1 | 同表 1 | 同表 1 | `runs/<split>/metrics/{per_method_summary,lca_vs_baselines_pairwise}.csv` |
| **附录 D.3** AMR-WB(9)+Codec2(6) | exp20 | `exp20_amrwb_codec2_test300_20260614_seed42` | `commands/run_amrwb_codec2.py` + `analyze_exp20.py`（独立 ffmpeg 8.1.1，见 `tools/`，未入库）| 复用 exp12 300 原始 wav | `runs/<split>/metrics/*.csv` |

## 3. 训练复现入口（方法核心）

| 阶段 | 命令 | 章节 |
|---|---|---|
| exp1 NAS | `python nas/run_staged_encoder_nas.py`（见 `nas/run_nas.sh`）| §3.4 |
| exp2 Base | `accelerate launch scripts/train_example.py --config output/experiments/exp2_.../configs/scit_speech_base_config.json` | §3.2 |
| exp3 LCA | `scripts/train_lca.py`（配置见 exp3 v2 `configs/`）| §3.3 |
| HuBERT 蒸馏特征 | `scripts/hubert_rep_extract.py` / `.sh` | §3.2 |
| 打包 best ckpt | `python scripts/package_checkpoint.py --run-dir <run> --config <cfg>` | §2 |

通用配置（§4.5）：seed=42、batch 8、grad-accum 4、Adam(0.9,0.99)、cosine 无 warmup、bf16；
`λ_recon=500, λ_q=10, λ_distill=30`；mel 4 尺度权重 `[45,1,1,1]`。

## 4. 测试

```powershell
conda run -n speechtokenizer python -m unittest \
  tests.test_exp2_script_chain tests.test_encoder_handoff \
  tests.test_teacher_guided_nas tests.test_trainer_lr_schedule tests.test_exp2_supplementary
```
本次整理后实测 **23 tests OK**（补 `tests/__init__.py` 后恢复文档化 `tests.X` 调用）。
注：`tests/test_public_nq8_layer_sweep.py` 为 broken orphan（import 不存在的 `experiments_semcom`），
非本仓库可运行测试，待人工确认（见 REORG_MANIFEST.md G 类）。

## 5. 已知 caveat（复现前必读）

- `output/` 当前未入库；权重 `.pt` 被 `.gitignore` 忽略。复现客观指标须在保有 `output/` 的本机执行。
- 图 2 数据源一致性见 §1 表注（exp4 8 样本 vs exp12 300 样本）。
- exp20 需独立 ffmpeg 8.1.1（`tools/`，未入库），其余 baseline 用 conda ffmpeg 6.1.2。
- 论文诚实边界（不可外推的结论）详见 `output/doc/实验记录.md` 各节 caveats 与论文 §7 局限性。

