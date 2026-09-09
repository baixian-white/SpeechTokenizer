# 实验三：低负载/轻量信道感知自适应（LCA）总结

- run_id: `exp3_low_load_channel_aware_adaptation_20260530_seed42`
- 训练时间：2026-05-31 01:08 启动 → 07:59 结束（约 6h50min）
- 训练总 step：33790（10 epochs）
- LCA 微调起点：实验二 distill30 重训的 SCIT-Speech-Base（sha256=8c23c2b146...，dev/mel=1.124）

## 1. 训练配置

| 字段 | 值 |
|---|---|
| 模式 | end-to-end 微调（encoder + RVQ + decoder 全部参与） |
| 学习率 | 1e-5（cosine 衰减到接近 0） |
| 批大小 | 8 |
| 梯度累计 | 4 |
| epochs | 10 |
| seed | 42 |
| distill_loss_lambda | 30.0（与 base 一致）|
| lambda_full / lambda_comm | 1.0 / 1.0 |
| L 采样 | uniform random over {1, 2, 3} |
| 信道条件 | clean / dropout-low (p=0.01) / dropout-mid (p=0.03) / sub-low (p=0.001) / sub-mid (p=0.005) |
| 数据 | LibriSpeech train-clean-100，27039 训练 + 1500 验证 |

## 2. 训练过程关键节点

13 个 dev 评估点（step 2500 / 5000 / ... / 32500）：

| 阶段 | 现象 |
|---|---|
| step 2500–5000 | LCA 初期失稳，dev/comm_mel 全 15 cell 退化 4-5%（GAN 平衡尚在调整） |
| step 7500–10000 | 第一次反弹改善 |
| step 10000–15000 | 中期波动（dev 在 1.07–1.16 区间震荡） |
| step 17500 | dev/full_mel 全程最低 = 1.013 |
| **step 30000** | **dev/comm_mel 全 15 cell 全局最低**（平均改善 12.97% vs LCA step 2500） |
| step 32500 | 训练末段 cosine 收尾，所有 dev cell 略微回退 |

由此选出两个候选 LCA checkpoint：

- **`SCIT-Speech-LCA_step30000_comm_mel_optimum.pt`**：dev/comm_mel 在所有 (L, channel) 组合上达到全局最低 → 作为正式 `SCIT-Speech-LCA_best.pt`（sha256=`9daa924ff5...`）
- `SCIT-Speech-LCA_step17500_full_mel_optimum.pt`：dev/full_depth_mel_error 全程最低 = 1.013 → 备用对照

## 3. Base vs LCA 评估结果（fixed_sample_list.txt 8 条样本，3 L × 5 channel = 15 组合 × 2 model = 240 评估对）

ChannelSim seed 在 (sample, L, channel) 三元组级别确定，且 Base 和 LCA 共享同一 seed → 任何指标差异来自模型权重，与扰动随机性无关。

### 3.1 LCA (step 30000) vs Base 平均改善

| 指标 | 平均 Δ | 解读 |
|---|---:|---|
| Δwave_l1 | -0.0006 | 0.0006 量级，LCA 略优（所有 cell 都为负）|
| Δmel_l1 | **-0.058** | LCA 显著降低 mel L1（5.8% 量级）|
| Δsi_snr_db | +0.19 dB | LCA 平均高 0.19 dB（部分 cell 略低）|
| Δstoi | **+0.016** | LCA 全 15 cell 一致提升 STOI |
| Δpesq_wb | +0.010 | LCA 整体改善 PESQ-WB，但部分 cell 略低 |

### 3.2 LCA (step 17500) vs Base 平均改善

| 指标 | 平均 Δ |
|---|---:|
| Δwave_l1 | -0.0005 |
| Δmel_l1 | **-0.045** |
| Δsi_snr_db | **+0.35 dB** |
| Δstoi | +0.015 |
| Δpesq_wb | **+0.023** |

### 3.3 两个 snapshot 横向对比

| 指标 | step 30000 优势 | step 17500 优势 |
|---|---|---|
| mel_l1 | ✅ 改善 5.8%（更优）| 改善 4.5% |
| stoi | ✅ +0.0160（略优）| +0.0152 |
| si_snr_db | +0.19 dB | ✅ +0.35 dB（更优） |
| pesq_wb | +0.010 | ✅ +0.023（更优） |

**结论**：step 30000 在频谱（mel_l1）和短时可懂度（stoi）上更优；step 17500 在波形保真度（si_snr）和宽带感知质量（pesq_wb）上更优。两者各有侧重，差异不大。

## 4. 按 L 维度的 mel_l1 改善（Base → LCA step 30000）

| L | bps | Base mel_l1 | LCA mel_l1 | 改善 |
|---:|---:|---:|---:|---:|
| 1 | 500 | 1.208 | **1.136** | -6.0% |
| 2 | 1000 | 0.963 | **0.918** | -4.7% |
| 3 | 1500 | 0.900 | **0.843** | -6.4% |

**所有 L 都改善**，最大改善在 L=3（虽然 L=3 已经是高负载、改善空间小）。L=1（最低负载）改善 6%，是 LCA 微调的核心目标场景。

## 5. 信道扰动鲁棒性（实验三核心目标的诚实评估）

实验三的核心问题之一（[计划文档](../../../doc/experiment_plans/exp3_low_load_channel_aware_adaptation.md) 第 1 节）是：

> mild index-level perturbation 下，LCA 是否比 Base 退化更小

定义鲁棒性改善：`robustness_improvement = base_degradation - lca_degradation`（degradation = clean 与扰动条件之差），正值表示 LCA 退化更小（更鲁棒）。

12 个 (L, 4 个扰动条件) 单元的聚合：

| 指标 | 平均 robustness_imp | LCA 更鲁棒的 cell |
|---|---:|---:|
| si_snr_db | -0.018 dB | 4/12 |
| stoi | -0.0001 | 6/12 |
| pesq_wb | -0.002 | 7/12 |
| corr | -0.0003 | 5/12 |
| mel_l1 | -0.0003 | 3/12 |
| wave_l1 | -0.000004 | 6/12 |

**所有 6 个指标的平均 robustness_improvement 都接近 0 或微负**，且每个指标只有约一半 cell 显示 LCA 更鲁棒——属于噪声水平。

**结论**：LCA 微调**没有改善信道扰动鲁棒性**——这是与实验三核心目标不一致的负面结果。

LCA vs Base 的对比可以这样说：LCA 把整条质量曲线上抬了（clean 和扰动条件下都提升），但 clean → 扰动 的退化幅度几乎没变。**绝对质量改善源自 random-L 训练目标，不是 ChannelSim 训练目标**。

可能的原因（写论文时应作为 limitation / future work）：

1. **扰动概率过低**：5 个训练条件平均 p_drop ≈ 0.008、p_sub ≈ 0.0012，模型大部分时间面对 clean 输入，没有学会"识别和补偿扰动"
2. **扰动模式过于平和**：previous-index replacement 在低 dropout 率下几乎不破坏信号——前一帧索引与当前帧索引在大部分语音上相似，模型不需要学习鲁棒性也能恢复
3. **GAN 训练偏向听感而非鲁棒性**：lambda_full=lambda_comm=1.0 让两个分支同等权重，但都用同样的 mel/recon loss，没有区分"鲁棒性"目标
4. **可能的修复方向**：提高训练扰动概率（如 p_drop=0.05、p_sub=0.01）、引入显式的鲁棒性 loss（如 clean 和 perturbed 输出的一致性 loss）、或加 mask token 让模型显式知道哪些位置被扰动

## 6. 可以说和不能说

可以说：

- LCA end-to-end 微调成功完成了 10 个 epoch 的训练，没有发散
- LCA (step 30000) 相对 Base 在所有 15 个 (L, channel) 组合上一致改善 mel_l1 与 STOI
- L=1 低负载操作点的 mel_l1 改善 6%、STOI 提升 0.016——表明 LCA 在低负载场景产生了实质改善
- 训练侧 dev/comm_mel 全 15 cell 在 step 30000 全局最优，平均改善 12.97%（dev set 内部口径）
- 两个 LCA 候选 checkpoint（step 30000 / 17500）已固化并保留 sha256

不能说：

- **不能说 LCA 改善了信道扰动鲁棒性**——这是实验三的核心目标之一，但本次实验设计没有产生鲁棒性改善（详见第 5 节）。绝对质量改善源自 random-L 训练目标，不是 ChannelSim 训练目标
- 不能说 PESQ-WB 改善（+0.01）有显著实际听感差异——量级偏小，可能在主观感知阈值之下
- 不能说当前评估是论文级最终结论——只在 8 条 train-clean 同源测试样本上评估，未覆盖 test-other / 真实通信场景
- 不能说 si_snr / pesq_wb 上 step 17500 优于 step 30000 是定论——指标互有胜负，需要更多样本量
- 不能说 LCA 取代了 Base 作为底模——Base 在 dev/full_mel 上仍然是 1.124 最优，LCA 不优于它在全深度无扰动场景下的表现

## 7. 产物清单

| 类型 | 路径 |
|---|---|
| 正式 LCA ckpt | `checkpoints/SCIT-Speech-LCA_best.pt`（sha256=9daa924f...，来自 step 30000）|
| LCA snapshot 1 | `checkpoints/SCIT-Speech-LCA_step30000_comm_mel_optimum.pt` |
| LCA snapshot 2 | `checkpoints/SCIT-Speech-LCA_step17500_full_mel_optimum.pt` |
| Trainer ckpts | step 17500 / 20000 / 22500 / 25000 / 27500 / 30000 / 32500 共 7 个 |
| manifest | `checkpoints/checkpoint_manifest.json` |
| 评估指标 (step 30000) | `metrics/base_vs_lca_results.csv/json` |
| 评估报告 (step 30000) | `reports/base_vs_lca_summary.md` |
| 评估指标 (step 17500) | `metrics/base_vs_lca_results_step17500.csv/json` |
| 评估报告 (step 17500) | `reports/base_vs_lca_summary_step17500.md` |
| 样本：原始 | `samples/original/*.wav`（8 条）|
| 样本：Base 输出 | `samples/base/{condition}/L{1,2,3}/*.wav` |
| 样本：LCA (step 30000) 输出 | `samples/lca/{condition}/L{1,2,3}/*.wav` |
| 样本：LCA (step 17500) 输出 | `samples/lca_step17500/{condition}/L{1,2,3}/*.wav` |
| 训练日志 | `checkpoints/logs/events.out.tfevents.*` |
| TB 训练标量 | 82 个 tag，含 dev-matrix 30 cell + per-L 分层 + 采样分布健康度等 |

## 8. 下一步建议

1. **跨数据集评估**：在 LibriSpeech test-other 或更大 test-clean 子集上重测，验证泛化
2. **WER/CER 评估**：装一个 ASR（如 Whisper-tiny）跑 8 条样本 × 15 组合的 ASR，给"语义可懂度"留客观数据
3. **更强信道扰动**：当前 p_drop≤0.03、p_sub≤0.005，扰动幅度太小不足以分辨鲁棒性差异。可以补一个 p_drop=0.10 / p_sub=0.05 的诊断评估
4. **更新 [实验记录.md](output/doc/实验记录.md)**：把实验三状态从"未开始"改为"已完成"，记录上述 caveats
5. **决定是否进入实验四（baseline 对比）**：可用 SCIT-Speech-LCA_best.pt 作为本文系统的最终模型；实验四需要先解决 ffmpeg/opusenc 工具链缺失问题
