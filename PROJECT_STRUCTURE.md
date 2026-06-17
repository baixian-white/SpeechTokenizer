# PROJECT_STRUCTURE.md — 重组后目录结构与论文/实验对应

本仓库是论文 **SCIT-Speech（基于共享 RVQ 码本索引传输的极低码率语音通信）** 的研究代码与实验库。
组织主轴是**论文章节 + 实验编号（exp1–exp20）**，而非软件分层。本次整理保持实验中心心智模型，
**未**做 src/modules 式软件分层重构。

## 顶层目录

```
speechtokenizer/          # 【方法核心包】模型 / RVQ 量化 / 训练器 / 损失（复现方法必需）
nas/                      # 【exp1 NAS】教师引导分阶段编码器搜索（以包形式被 import）
scripts/                  # 【实验/评估/作图脚本】扁平结构（硬约束：见下「为何不拆分」）
config/                   # Base 默认配置 spt_base_cfg.json
tests/                    # 单元测试（23 tests OK；1 个 broken orphan 待确认）
3用户demo/                # 【demo】三用户中心路由实时通信原型（论文 demo 章节，加载 LCA v2）
output/                   # 【实验产物/溯源数据】未入库，保存于本地（见 REPRODUCIBILITY §5）
  ├── experiments/expN…/  #   按实验编号：checkpoints/metrics/reports/configs/commands/sha256
  ├── doc/                #   论文草稿 / 实验记录.md / 实验手册.md / 素材图
  ├── archive/            #   既有旧实验归档（2026-05-26 前）
  └── baseline_model_cache/  # DAC/EnCodec 基线权重缓存
tools/                    # 独立 ffmpeg 8.1.1（exp20 AMR-WB/Codec2）— 未入库（大体积）
legacy_explorations/      # 【归档】探索性/已被取代/一次性文件 + provenance
REPRODUCIBILITY.md        # 论文每表/图 → 复现要素
PROJECT_STRUCTURE.md      # 本文件
MOVES.md                  # 本次所有移动的旧→新映射
REORG_MANIFEST.md         # Phase 2 可审计分类清单（A–G + 硬约束 + 待确认）
```

## 方法核心包 `speechtokenizer/`（论文 §3）

| 路径 | 职责 | 论文 |
|---|---|---|
| `model.py`, `__init__.py` | SpeechTokenizer 主模型（encoder/RVQ/decoder）| §3.1–§3.2 |
| `discriminators.py` | 多周期/多尺度/STFT 判别器 | §3.2 GAN |
| `modules/{conv,lstm,norm,seanet}.py` | 模型骨干（SEANet）| §3.2/§3.4 |
| `quantization/{core_vq,vq,ac,distrib}.py` | 分层 RVQ（`core_vq.py` 即 §14 reinit 诊断对象）| §3.1 |
| `trainer/{trainer,dataset,loss,optimizer}.py` | Base 训练（`trainer.py` 含 §4.4 LR 修正）| §3.2/§4.5 |
| `trainer/lca_trainer.py` | LCA 训练 + ChannelSim（前帧覆盖/随机替换/一致性）| §3.3 |

## `nas/`（论文 §3.4 / §6.4，exp1）

四阶段搜索（架构剖面→短程蒸馏→精化代理→Pareto 终选）。核心：
`run_staged_encoder_nas.py`(入口)、`evaluate_encoder_proxy.py`、`teacher_guided_proxy.py`、
`search_space.py`、`encoder_handoff.py`、`validate_short_distill.py`、`SeaNet.py`、`model_components.py`。
**约束**：以包名 `nas.*` 被 import（`tests/test_teacher_guided_nas.py` 等依赖），须留仓库根。

## `scripts/`（论文 §5–§6 + 训练入口）

按论文角色（非物理子目录，见下）：
- **训练**：`train_example.py`(exp2)、`train_lca.py`(exp3)、`train_distill_weight_ablation.py`(exp5 §6.1)、`prepare_exp2_config.py`、`run_exp2_training.{ps1,bat}`、`hubert_rep_extract.py`
- **baseline**：`run_exp4_baselines.py`、`run_opus_baseline.py`（exp4/exp12 §5.1）
- **评估**：`evaluate_clean_large_nosave.py`(表2/7)、`evaluate_perturb_large_nosave.py`(表3)、`evaluate_packet_burst_*`(表4)、`evaluate_asr_wer_onthefly.py`(表6)、`evaluate_perturbed_asr_wer_onthefly.py`(表5)、`evaluate_lca_vs_base.py`、`evaluate_ablation_variants.py`(§6.3)
- **作图**：`plot_fig2_rate_quality_tradeoff.py`(图2)、`plot_fig3_perturbation_robustness.py`(图3)
- **共享库**（被同级脚本 import）：`experiment_utils.py`、`channel_sim.py`、`pack_indices.py`、`payload_accounting.py`
- **后处理/打包**：`package_checkpoint.py`、`summarize_exp2.py`、`export_loss_curves.py`、`codebook_usage_report.py`(§14) 等

### 为何 `scripts/` 不拆成子目录（硬约束）
24 个脚本硬编码 `PROJECT_ROOT = Path(__file__).resolve().parents[1]`，随后 `sys.path.insert`
并据此解析 `output/experiments/...`。下移一层会同时破坏 sys.path 注入与 output 路径解析，
需逐脚本改 `parents[1]→parents[2]` + 重验。研究项目以可复现性优先，保持扁平 + 角色文档化
（本节）比软件分层更安全。详见 REORG_MANIFEST.md「硬约束 2」。

## 实验编号 → 论文章节速查（详见 REPRODUCIBILITY.md）

| exp | 主题 | 论文 |
|---|---|---|
| exp1/1b | NAS 编码器搜索 | §3.4/§6.4 |
| exp2 | SCIT-Speech-Base 训练 | §3.2 |
| exp3/3c | LCA v1/v2 微调 | §3.3/§6.2 |
| exp4/exp12/exp20 | 同码率 baseline 对照 | §5.1/附录D |
| exp5/5b/5c | 消融（蒸馏/组件因子化/consistency 动力学）| §6 |
| exp7 | 全量 clean 泛化 | §5.2 |
| exp8 | 索引前帧覆盖/替换 | §5.3 |
| exp9 | 丢包/突发 | §5.4 |
| exp10/exp11 | ASR/WER（无扰动/扰动）| §5.5 |
| exp13 | hand-designed encoder 全训对照 | §6.4/§7（内部存档）|
| exp14 | test-clean ASR 退化诊断 | §5.5 |
| exp15 | codebook 利用率诊断 | §7-5 |
| exp16/exp17 | entropy 真实码率 / packet 开销 | §3.1 |
| exp18 | 主观 AB 听测脚手架（未启动）| §7-3 |
| exp19/19b | VCTK/AISHELL zero-shot | §5.6 |

## 溯源链（神圣不可断）

论文主张 → 实验 run_id → 权重 sha256 → 指标 CSV → 图表。完整台账见 REPRODUCIBILITY.md §0–§2；
规范事实日志见 `output/doc/实验记录.md`（只追加，本次整理未改其内容）。
