# SCIT-Speech 仓库整理 — 可审计分类清单（Phase 2，移动前）

> 状态：**调查完成，尚未移动任何文件。** 本清单依据论文正文
> (`scit_speech_method_cn_draft_20260609.md`)、规范实验日志 (`实验记录.md` §1–§19)、
> 主表、demo skeleton 与代码 import/路径图归类。每项标注证据与所属实验/章节。

## 0. 决定整理策略的硬约束（先读）

1. **几乎所有东西都未被 git 跟踪。** 已跟踪文件仅 ~45 个（见 `git ls-files`）；
   未跟踪的有：整个 `output/`（16,475 文件，含全部实验证据链）、`scripts/` 中 49 个里的 45 个、
   `nas/` 的 7 个新脚本、`tests/`、`3用户demo/`、`tools/`、`speechtokenizer/trainer/lca_trainer.py`。
   → **`git mv` 只能给 4 个已跟踪脚本保历史**；其余文件没有历史可保。
   → **本仓库最大的可复现性风险不是"目录不美观"，而是整条证据链从未入库**：
   一旦磁盘故障或误清理，论文主张→run_id→sha256→指标CSV→图表 全部消失。

2. **`scripts/*.py` 硬编码 `Path(__file__).resolve().parents[1]` 作为 PROJECT_ROOT**
   （24 个确认），随后 `sys.path.insert(root)` 并据此解析 `output/experiments/...`。
   → 把脚本下移一层（如 `scripts/eval/`）会同时**破坏 sys.path 注入和 output 路径解析**。
   子分 `scripts/` 必须逐个改 `parents[1]→parents[2]` + output 相对路径，且逐个重验。

3. **`nas/` 以包形式被 import**（`from nas.SeaNet import ...`、`from nas.search_space import ...`），
   且测试 `tests/test_teacher_guided_nas.py` 依赖。→ `nas/` 必须留在仓库根、保持包名。

4. **论文正文用相对路径 `../...论文素材/figN.png` 嵌入 5 张图**，且论文是规范溯源源（不可改内容）。
   → 移动论文草稿或素材目录会断 5 个图链接。

5. **`3用户demo/` 完全自包含**（自带 `nas/`、`Log/spt_base/` 权重副本、sys.path 自解析），
   加载 LCA v2 `sha256=af60223…`。→ 勿动其内部结构与相对路径。

---

## A. 方法核心代码（复现方法所必需，原地保留）

| 文件 | 服务 | 证据 |
|---|---|---|
| `speechtokenizer/model.py`, `__init__.py`, `discriminators.py` | 所有训练/评估 | 被 train_example/train_lca/preflight import |
| `speechtokenizer/modules/{conv,lstm,norm,seanet}.py` | 模型骨干 | 包内依赖 |
| `speechtokenizer/quantization/{core_vq,vq,ac,distrib}.py` | RVQ 量化 | exp15 §14 reinit 诊断点名 core_vq.py |
| `speechtokenizer/trainer/{trainer,dataset,loss,optimizer}.py` | Base 训练 | §4.4 LR scheduler 修正点名 trainer.py |
| `speechtokenizer/trainer/lca_trainer.py`（未跟踪）| LCA 训练 + ChannelSim | 被 evaluate_lca_vs_base / evaluate_perturb / evaluate_perturbed_asr import；§3.3 |
| `nas/{SeaNet,model_components,search_space,teacher_guided_proxy,encoder_handoff,evaluate_encoder_proxy,validate_short_distill,run_staged_encoder_nas,search_autoencoder,train_nas,Nas,custom_model,export_best_model,check_baseline,make_subset}.py` | exp1 NAS | §3.4；nas 包内互相 import；tests 依赖 |
| `config/spt_base_cfg.json` | Base 默认配置 | §3.2 默认 distill120 |

## B. 实验代码与配置（按 exp 编号，原地保留）

| 文件 | exp / 章节 | 证据 |
|---|---|---|
| `scripts/prepare_exp2_config.py`, `run_exp2_training.{ps1,bat}`, `train_example.py` | exp2 / §3.2,§4.5 | 实验记录 §2 脚本链路表 |
| `scripts/train_lca.py` | exp3 / §3.3 | 12× 被 exp 的 commands/ 引用 |
| `scripts/train_distill_weight_ablation.py` | exp5 A3 蒸馏消融 / §6.1 | §13/§5 |
| `scripts/train_decoder_only_finetune.py` | LCA 变体训练 | import 全套 model |
| `scripts/run_exp4_baselines.py`, `run_opus_baseline.py` | exp4/exp12 baseline / §5.1 | §10.3 点名 |
| `scripts/channel_sim.py`, `pack_indices.py`, `payload_accounting.py` | ChannelSim/打包/码率 / §3.1,§4.3 | 被多脚本 import；§3.1 packet 开销 |
| `scripts/experiment_utils.py`, `create_experiment_run.py` | run 目录脚手架 | 被 collect_environment/preflight import |
| `scripts/hubert_rep_extract.{py,sh}` | HuBERT 蒸馏特征 / §3.2 | 已跟踪 |

## C. 评估与作图代码（直接产出论文表/图，原地保留）

| 文件 | 产出 | 证据 |
|---|---|---|
| `scripts/plot_fig2_rate_quality_tradeoff.py` | **图 2** | 读 exp4 csv，写 fig2_*.png（注：见下方"待确认①"）|
| `scripts/plot_fig3_perturbation_robustness.py` | **图 3** | 读 exp8+exp9，写 fig3_*.png |
| `scripts/evaluate_clean_large_nosave.py` | 表 2/表 7 (§5.2,§5.6) | exp7/exp19 |
| `scripts/evaluate_perturb_large_nosave.py`, `evaluate_packet_burst_large_nosave.py`, `evaluate_packet_burst_loss.py` | 表 3/表 4 (§5.3,§5.4) | exp8/exp9 |
| `scripts/evaluate_asr_wer_onthefly.py`, `evaluate_perturbed_asr_wer_onthefly.py`, `run_asr_evaluation.py` | 表 5/表 6 (§5.5) | exp10/exp11 |
| `scripts/evaluate_lca_vs_base.py` | exp3/exp5 配对评估 | 12× commands 引用 |
| `scripts/evaluate_ablation_variants.py`, `aggregate_v0_v4_n256.py` | 表 (§6.3 V0–V4) | exp5b §11.1 |
| `scripts/evaluate_sample_audio_quality.py`, `evaluate_layer_reconstruction.py`, `export_full_utterance_samples.py`, `compare_reconstruction_audio.py`, `codebook_usage_report.py` | §5 客观指标/§14 codebook | 实验记录 §2,§5,§14 |
| `scripts/export_loss_curves.py`, `summarize_exp2.py`, `package_checkpoint.py`, `extract_best_from_trainer.py` | 训练后处理/打包 | §2 脚本链路 |

## D. demo（原地保留，勿改内部相对路径）

| 路径 | 说明 |
|---|---|
| `3用户demo/`（93 文件，未跟踪）| 当前论文对齐的三用户中心路由原型；demo skeleton §X；加载 LCA v2 sha256=af60223… |

## E. 实验产物与溯源数据（默认原地保留）

| 路径 | 说明 |
|---|---|
| `output/experiments/exp1…exp20/**`（未跟踪）| checkpoint/metrics/reports/logs/configs/commands/sha256/manifest。大 `.pt` 已被 .gitignore 忽略，轻量证据（CSV/md/json）应入库 |
| `output/doc/**` | 论文草稿、实验记录、手册、素材图（含 figN.png + 框架图）|
| `output/archive/legacy_experiment_files_20260526_pre_new_experiments/` | 已有的旧归档（原地）|
| `output/baseline_model_cache/` | DAC/EnCodec 基线权重缓存 |
| `tools/ffmpeg-8.1.1-full_build/` | exp20 AMR-WB/Codec2 编码器（§19）|

## F. 探索性/已被取代/一次性脚本（候选归档 legacy_explorations/，附 provenance）

| 文件 | 所属/验证假设 | 归档理由 |
|---|---|---|
| `scripts/_exp5c_probe_tb_tags.py` | exp5c TB tag 探针 (§16) | 一次性只读探针（`_` 前缀）|
| `tmp_analyze_v3.py`, `tmp_full_manifest.py`（根目录）| exp7 manifest / v3 分析 | 临时脚本（`tmp_` 前缀）|
| `scripts/migrate_paths_e_to_h.py` | E→H 盘迁移 (§实验记录开头) | 一次性迁移，已完成 |
| `scripts/normalize_filelist_paths.py` | filelist 路径规整 | 一次性数据修复 |
| `scripts/convert_single_flac_to_wav.py` | 单文件转换工具 | 一次性工具 |

## G. 不确定（绝不擅自当 F 处理，待人工确认）

| 文件 | 疑问 |
|---|---|
| 根 `example.py` | 推理示例（已跟踪）；与 demo 重复？保留还是归 docs 示例 |
| 根 `项目情况说明.md`, `训练和推理脚本.md`（已跟踪）| 旧项目说明；是否被新 docs 取代 |
| 根 `对比wav语音差距.ipynb`, `读取wav文件信息.ipynb`（git 已删，working tree 无）| 工作区已删，是否 commit 删除 |
| 已跟踪 `demo_nature/多人嘈杂环境/example.py` | 旧 demo？被 3用户demo 取代？|
| 已跟踪 `实时语音系统/demo_now.{md,py}` | 旧实时 demo？被 3用户demo 取代？|
| `scripts/prefetch_baseline_models.py`, `preflight_experiments.py`, `preflight_smoke_test.py`, `collect_environment.py` | 实验前置/环境采集；A 还是 C？（倾向保留为 B 支撑）|
| `scripts/exp2_supplementary.py`, `prepare_exp2_supplementary.py`, `build_exp4_supplementary.py`, `package_exp2_outputs.py`, `review_experiment_outputs.py` | exp 补充/打包脚本；保留 B/C |

### 待确认①（溯源链潜在断点，最高优先）
`plot_fig2_rate_quality_tradeoff.py` 读 **exp4（8 样本）** 的 csv 写 fig2，但论文 §5.1 现用
**exp12（300 样本 + CI）**；素材里 `fig2_*.png` 与 `fig2_*.png.bak8sample` 并存。
**fig2 当前版可能需改读 exp12 才与正文表 1a/1b 一致** —— 这是图-数据一致性问题，非整理问题，先报告不擅改。
