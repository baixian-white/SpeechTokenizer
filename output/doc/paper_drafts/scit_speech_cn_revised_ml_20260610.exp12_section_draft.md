# §5.1 与 §7 局限性条目修订草稿（exp12 占位版）

> 独立草稿（2026-06-10）。本草稿仅包含 `scit_speech_cn_revised_ml_20260610.md` 中 **§5.1 同码率对比** 与 **§7 局限性** 第 7 条两处需要被替换的内容；其余章节保持不动。所有数值以 `<<EXP12_TC:...>>` / `<<EXP12_TO:...>>` 占位符形式给出，待 exp12 跑完后由 `analyze_baseline_300_stats.py` 统一渲染填入。本草稿不直接修改主稿，待 exp12 全部输出落盘并审核通过后再合并。
>
> 占位符约定：
>
> - 前缀 `EXP12_TC` 表示 `test-clean_300`，`EXP12_TO` 表示 `test-other_300`。
> - 主体名遵循 `<method>_<L_or_rate>_<metric>_<stat>` 风格，如 `scit_lca_L1_mel_l1_mean (95% CI)`、`dac_nq2_pesq_wb_mean (95% CI)`、`opus_6kbps_stoi_mean (95% CI)`。
> - 配对显著性占位符使用 `<<EXP12_TC:wilcoxon_<baseline>_vs_scit_lca_L<L>_<metric>_p>>` 与 `<<EXP12_TC:diff_<baseline>_vs_scit_lca_L<L>_<metric>_ci95>>`，渲染脚本输出 `p = 0.0xx` 与 `Δ = -0.0xx [-0.0xx, -0.0xx]` 形式。

## 5.1 同码率对比（exp12，n = 300）

![图 2：码率—质量权衡（exp12，n=300）](../scit_speech_method_cn_draft_20260609论文素材/fig2_rate_quality_tradeoff.png)

**图 2**：在 `test-clean_300` 与 `test-other_300` 上的码率—质量权衡曲线（exp12，每集 n = 300）。三联子图分别展示 mel-L1、STOI、PESQ-WB；横轴为打包后的实际码率（对数刻度）。SCIT-Speech-LCA 与 SCIT-Speech-Base 在 500/1000/1500 bps 三个工作点；对照 Opus、DAC、EnCodec 覆盖各自支持的档位；PCM 给出无损上界。黄色阴影标识 SCIT-Speech 的工作区间。曲线点上标注 95% bootstrap CI，配对 Wilcoxon 显著性见正文。

**表 1a：`test-clean_300` 同码率对比（exp12，n = 300，95% bootstrap CI）**

| 操作点 | 方法 | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | n |
|---|---|---|---|---|---:|
| 500 bps | SCIT-Speech-Base L=1 | <<EXP12_TC:scit_base_L1_mel_l1_mean (95% CI)>> | <<EXP12_TC:scit_base_L1_stoi_mean (95% CI)>> | <<EXP12_TC:scit_base_L1_pesq_wb_mean (95% CI)>> | 300 |
| 500 bps | SCIT-Speech-LCA L=1 | <<EXP12_TC:scit_lca_L1_mel_l1_mean (95% CI)>> | <<EXP12_TC:scit_lca_L1_stoi_mean (95% CI)>> | <<EXP12_TC:scit_lca_L1_pesq_wb_mean (95% CI)>> | 300 |
| 500 bps | DAC n_q=1 | <<EXP12_TC:dac_nq1_mel_l1_mean (95% CI)>> | <<EXP12_TC:dac_nq1_stoi_mean (95% CI)>> | <<EXP12_TC:dac_nq1_pesq_wb_mean (95% CI)>> | 300 |
| 1000 bps | SCIT-Speech-Base L=2 | <<EXP12_TC:scit_base_L2_mel_l1_mean (95% CI)>> | <<EXP12_TC:scit_base_L2_stoi_mean (95% CI)>> | <<EXP12_TC:scit_base_L2_pesq_wb_mean (95% CI)>> | 300 |
| 1000 bps | SCIT-Speech-LCA L=2 | <<EXP12_TC:scit_lca_L2_mel_l1_mean (95% CI)>> | <<EXP12_TC:scit_lca_L2_stoi_mean (95% CI)>> | <<EXP12_TC:scit_lca_L2_pesq_wb_mean (95% CI)>> | 300 |
| 1000 bps | DAC n_q=2 | <<EXP12_TC:dac_nq2_mel_l1_mean (95% CI)>> | <<EXP12_TC:dac_nq2_stoi_mean (95% CI)>> | <<EXP12_TC:dac_nq2_pesq_wb_mean (95% CI)>> | 300 |
| 1500 bps | SCIT-Speech-Base L=3 | <<EXP12_TC:scit_base_L3_mel_l1_mean (95% CI)>> | <<EXP12_TC:scit_base_L3_stoi_mean (95% CI)>> | <<EXP12_TC:scit_base_L3_pesq_wb_mean (95% CI)>> | 300 |
| 1500 bps | SCIT-Speech-LCA L=3 | <<EXP12_TC:scit_lca_L3_mel_l1_mean (95% CI)>> | <<EXP12_TC:scit_lca_L3_stoi_mean (95% CI)>> | <<EXP12_TC:scit_lca_L3_pesq_wb_mean (95% CI)>> | 300 |
| 1500 bps | DAC n_q=3 | <<EXP12_TC:dac_nq3_mel_l1_mean (95% CI)>> | <<EXP12_TC:dac_nq3_stoi_mean (95% CI)>> | <<EXP12_TC:dac_nq3_pesq_wb_mean (95% CI)>> | 300 |
| 1500 bps | EnCodec 1.5 kbps | <<EXP12_TC:encodec_1p5kbps_mel_l1_mean (95% CI)>> | <<EXP12_TC:encodec_1p5kbps_stoi_mean (95% CI)>> | <<EXP12_TC:encodec_1p5kbps_pesq_wb_mean (95% CI)>> | 300 |
| 3 kbps | EnCodec 3 kbps | <<EXP12_TC:encodec_3kbps_mel_l1_mean (95% CI)>> | <<EXP12_TC:encodec_3kbps_stoi_mean (95% CI)>> | <<EXP12_TC:encodec_3kbps_pesq_wb_mean (95% CI)>> | 300 |
| 6 kbps | EnCodec 6 kbps | <<EXP12_TC:encodec_6kbps_mel_l1_mean (95% CI)>> | <<EXP12_TC:encodec_6kbps_stoi_mean (95% CI)>> | <<EXP12_TC:encodec_6kbps_pesq_wb_mean (95% CI)>> | 300 |
| 6 kbps | Opus 6 kbps | <<EXP12_TC:opus_6kbps_mel_l1_mean (95% CI)>> | <<EXP12_TC:opus_6kbps_stoi_mean (95% CI)>> | <<EXP12_TC:opus_6kbps_pesq_wb_mean (95% CI)>> | 300 |
| 8 kbps | Opus 8 kbps | <<EXP12_TC:opus_8kbps_mel_l1_mean (95% CI)>> | <<EXP12_TC:opus_8kbps_stoi_mean (95% CI)>> | <<EXP12_TC:opus_8kbps_pesq_wb_mean (95% CI)>> | 300 |
| 12 kbps | Opus 12 kbps | <<EXP12_TC:opus_12kbps_mel_l1_mean (95% CI)>> | <<EXP12_TC:opus_12kbps_stoi_mean (95% CI)>> | <<EXP12_TC:opus_12kbps_pesq_wb_mean (95% CI)>> | 300 |
| lossless | PCM | <<EXP12_TC:pcm_mel_l1_mean (95% CI)>> | <<EXP12_TC:pcm_stoi_mean (95% CI)>> | <<EXP12_TC:pcm_pesq_wb_mean (95% CI)>> | 300 |

**表 1b：`test-other_300` 同码率对比（exp12，n = 300，95% bootstrap CI）**

| 操作点 | 方法 | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | n |
|---|---|---|---|---|---:|
| 500 bps | SCIT-Speech-Base L=1 | <<EXP12_TO:scit_base_L1_mel_l1_mean (95% CI)>> | <<EXP12_TO:scit_base_L1_stoi_mean (95% CI)>> | <<EXP12_TO:scit_base_L1_pesq_wb_mean (95% CI)>> | 300 |
| 500 bps | SCIT-Speech-LCA L=1 | <<EXP12_TO:scit_lca_L1_mel_l1_mean (95% CI)>> | <<EXP12_TO:scit_lca_L1_stoi_mean (95% CI)>> | <<EXP12_TO:scit_lca_L1_pesq_wb_mean (95% CI)>> | 300 |
| 500 bps | DAC n_q=1 | <<EXP12_TO:dac_nq1_mel_l1_mean (95% CI)>> | <<EXP12_TO:dac_nq1_stoi_mean (95% CI)>> | <<EXP12_TO:dac_nq1_pesq_wb_mean (95% CI)>> | 300 |
| 1000 bps | SCIT-Speech-Base L=2 | <<EXP12_TO:scit_base_L2_mel_l1_mean (95% CI)>> | <<EXP12_TO:scit_base_L2_stoi_mean (95% CI)>> | <<EXP12_TO:scit_base_L2_pesq_wb_mean (95% CI)>> | 300 |
| 1000 bps | SCIT-Speech-LCA L=2 | <<EXP12_TO:scit_lca_L2_mel_l1_mean (95% CI)>> | <<EXP12_TO:scit_lca_L2_stoi_mean (95% CI)>> | <<EXP12_TO:scit_lca_L2_pesq_wb_mean (95% CI)>> | 300 |
| 1000 bps | DAC n_q=2 | <<EXP12_TO:dac_nq2_mel_l1_mean (95% CI)>> | <<EXP12_TO:dac_nq2_stoi_mean (95% CI)>> | <<EXP12_TO:dac_nq2_pesq_wb_mean (95% CI)>> | 300 |
| 1500 bps | SCIT-Speech-Base L=3 | <<EXP12_TO:scit_base_L3_mel_l1_mean (95% CI)>> | <<EXP12_TO:scit_base_L3_stoi_mean (95% CI)>> | <<EXP12_TO:scit_base_L3_pesq_wb_mean (95% CI)>> | 300 |
| 1500 bps | SCIT-Speech-LCA L=3 | <<EXP12_TO:scit_lca_L3_mel_l1_mean (95% CI)>> | <<EXP12_TO:scit_lca_L3_stoi_mean (95% CI)>> | <<EXP12_TO:scit_lca_L3_pesq_wb_mean (95% CI)>> | 300 |
| 1500 bps | DAC n_q=3 | <<EXP12_TO:dac_nq3_mel_l1_mean (95% CI)>> | <<EXP12_TO:dac_nq3_stoi_mean (95% CI)>> | <<EXP12_TO:dac_nq3_pesq_wb_mean (95% CI)>> | 300 |
| 1500 bps | EnCodec 1.5 kbps | <<EXP12_TO:encodec_1p5kbps_mel_l1_mean (95% CI)>> | <<EXP12_TO:encodec_1p5kbps_stoi_mean (95% CI)>> | <<EXP12_TO:encodec_1p5kbps_pesq_wb_mean (95% CI)>> | 300 |
| 3 kbps | EnCodec 3 kbps | <<EXP12_TO:encodec_3kbps_mel_l1_mean (95% CI)>> | <<EXP12_TO:encodec_3kbps_stoi_mean (95% CI)>> | <<EXP12_TO:encodec_3kbps_pesq_wb_mean (95% CI)>> | 300 |
| 6 kbps | EnCodec 6 kbps | <<EXP12_TO:encodec_6kbps_mel_l1_mean (95% CI)>> | <<EXP12_TO:encodec_6kbps_stoi_mean (95% CI)>> | <<EXP12_TO:encodec_6kbps_pesq_wb_mean (95% CI)>> | 300 |
| 6 kbps | Opus 6 kbps | <<EXP12_TO:opus_6kbps_mel_l1_mean (95% CI)>> | <<EXP12_TO:opus_6kbps_stoi_mean (95% CI)>> | <<EXP12_TO:opus_6kbps_pesq_wb_mean (95% CI)>> | 300 |
| 8 kbps | Opus 8 kbps | <<EXP12_TO:opus_8kbps_mel_l1_mean (95% CI)>> | <<EXP12_TO:opus_8kbps_stoi_mean (95% CI)>> | <<EXP12_TO:opus_8kbps_pesq_wb_mean (95% CI)>> | 300 |
| 12 kbps | Opus 12 kbps | <<EXP12_TO:opus_12kbps_mel_l1_mean (95% CI)>> | <<EXP12_TO:opus_12kbps_stoi_mean (95% CI)>> | <<EXP12_TO:opus_12kbps_pesq_wb_mean (95% CI)>> | 300 |
| lossless | PCM | <<EXP12_TO:pcm_mel_l1_mean (95% CI)>> | <<EXP12_TO:pcm_stoi_mean (95% CI)>> | <<EXP12_TO:pcm_pesq_wb_mean (95% CI)>> | 300 |

**统计协议**。表中所有均值与 95% 区间均按文件级 bootstrap（B = 1000 次有放回抽样）在 `test-clean_300` / `test-other_300` 两个固定子集上分别计算；同码率对照采用配对 Wilcoxon 符号秩检验，配对单位为同一条样本，比较对象为各基线相对 SCIT-Speech-LCA 在同一操作点的 mel-L1、STOI、PESQ-WB 三项差值，多重比较使用 Benjamini–Hochberg FDR 校正（α = 0.05）。

**与 SCIT-Speech-LCA 的配对 Wilcoxon 显著性**。在 `test-clean_300` 上，500 bps 工作点对照 DAC n_q=1 的 mel-L1 差值为 <<EXP12_TC:diff_dac_nq1_vs_scit_lca_L1_mel_l1_ci95>>（p = <<EXP12_TC:wilcoxon_dac_nq1_vs_scit_lca_L1_mel_l1_p>>），STOI 差值为 <<EXP12_TC:diff_dac_nq1_vs_scit_lca_L1_stoi_ci95>>（p = <<EXP12_TC:wilcoxon_dac_nq1_vs_scit_lca_L1_stoi_p>>），PESQ-WB 差值为 <<EXP12_TC:diff_dac_nq1_vs_scit_lca_L1_pesq_wb_ci95>>（p = <<EXP12_TC:wilcoxon_dac_nq1_vs_scit_lca_L1_pesq_wb_p>>）；1000 bps 与 1500 bps 工作点对照 DAC n_q=2/3 与 EnCodec 1.5 kbps 的 p 值与 CI 取自 `<<EXP12_TC:wilcoxon_*_vs_scit_lca_L{2,3}_*_p>>` 与 `<<EXP12_TC:diff_*_vs_scit_lca_L{2,3}_*_ci95>>`。`test-other_300` 取自同名 `EXP12_TO` 占位符。

**判定语言**。我们对每个 `(数据集, 操作点, 基线, 指标)` 单元采用三类判定：（i）**显著优于 SCIT-Speech-LCA** —— 差值 95% CI 完全位于改善侧且 BH 校正后 p < 0.05；（ii）**与 SCIT-Speech-LCA 可比** —— 差值 95% CI 跨越 0 或绝对值小于 `<<EXP12_GLOBAL:equiv_margin_<metric>>>` 给出的等效边界（mel-L1 0.02、STOI 0.005、PESQ-WB 0.05），按等效检验视作可比；（iii）**结论尚不充分** —— BH 校正后 p ≥ 0.05 且差值绝对值大于等效边界，样本量不足以拒绝零假设亦不足以判等效，记作"现有证据不足以下结论"。完整 6 (操作点) × 3 (指标) × 2 (数据集) 单元的判定表由 `analyze_baseline_300_stats.py` 渲染至附录 D（exp12 输出后追加）。

## 7 局限性 第 7 条修订（替换原"同码率 codec 对照样本量小"条目）

7. **同码率 codec 对照已扩展但仍不完整**：exp12 已把 SCIT-Speech-Base / SCIT-Speech-LCA、DAC n_q∈{1,2,3}、EnCodec ∈ {1.5, 3, 6} kbps、Opus ∈ {6, 8, 12} kbps 与 PCM 的 mel-L1 / STOI / PESQ-WB 对照扩展到 `test-clean_300` 与 `test-other_300`（各 n = 300），主表给出 95% bootstrap CI 与配对 Wilcoxon p 值（BH-FDR 校正）。但以下三项仍未补齐：（i）Codec2（700 bps、1200 bps、3200 bps 三档）由于当前 ffmpeg 构建缺 `libcodec2` 编码器尚未纳入；（ii）AMR-WB（6.6 / 12.65 / 23.85 kbps）同样受限于 ffmpeg 构建未纳入；（iii）主观 MOS / AB 听测尚未开展，PESQ-WB 与 STOI 不能完全替代主观评价。这三项缺口将在主稿正式版前补齐或保留为未来工作明示。

### 关于 Opus 6/8 kbps 在 mel-L1 指标上的偏差说明

libopus 在 ≤8 kbps 工作时进入 SILK 模式并将内部采样率降到 8 kHz，因此重建波形的高频带（>4 kHz）整体被截断。我们的 mel-L1 指标使用多尺度对数 mel 频谱（包含高频 mel bin），这些被截断的高频 bin 在原始波形上不为零、在 Opus 重建上接近 1e-9 下界、再经过自然对数后形成大数值的 L1 距离。因此 Opus 6/8 kbps 在 mel-L1 上的劣势同时反映了真实压缩失真**与**频带截断的指标系统性偏差。STOI（中位数 0.91 / 0.95）与 PESQ-WB（中位数 2.27 / 2.96）这两个指标对带宽截断更鲁棒，更适合作为 Opus 低码率档的可懂度／自然度对照。本文不把 'LCA L=3 mel-L1 优于 Opus 6 kbps' 作为单独宣称，所有跨码率比较（cross-rate）以多指标共同判断为准。
