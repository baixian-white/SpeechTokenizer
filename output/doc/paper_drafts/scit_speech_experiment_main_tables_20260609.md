# SCIT-Speech 论文实验主表整理（2026-06-09）

> 本文件把已完成实验压缩成论文正文可用的主表和结论边界。数值来自 `output/experiments/` 与 `output/doc/实验记录.md`，用于更新 `scit_speech_cn_draft.md` 的实验设置、结果和局限性。

## 1. 当前实验覆盖范围

| 实验 | 目录 | 作用 | 当前状态 |
|---|---|---|---|
| Exp1 | `output/experiments/exp1_nas_distill_run1_seed42` | 轻量发送端 encoder 搜索与 profiling | 完成 |
| Exp2 | `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42` | SCIT-Speech-Base 训练 | 完成 |
| Exp3 v2 | `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42` | LCA 强扰动 + consistency 微调 | 完成 |
| Exp4 | `output/experiments/exp4_baseline_comparison_20260531_seed42` | 8 条固定样本上的 baseline 对比 | 完成 |
| Exp5 | `output/experiments/exp5_ablation_and_diagnosis_20260601_seed42` 与 `output/experiments/exp5_lca_component_factorial_20260603_seed42` | 蒸馏、LCA、组件因子消融 | 完成 |
| Exp7 | `output/experiments/exp7_librispeech_test_full_clean_20260606` | LibriSpeech test-clean/test-other 全量 clean 泛化 | 完成 |
| Exp8 | `output/experiments/exp8_librispeech_test_full_perturb_20260606` | 全量 index dropout/substitution 鲁棒性 | 完成 |
| Exp9 | `output/experiments/exp9_librispeech_test_full_packet_burst_20260608` | 全量 packet/burst loss 鲁棒性 | 完成 |
| Exp10 | `output/experiments/exp10_asr_wer_onthefly_20260609` | clean 条件 ASR/WER 子集 | 完成 |
| Exp11 | `output/experiments/exp11_perturbed_asr_wer_20260609` | 扰动条件 ASR/WER 子集 | 完成 |

## 2. 主表 A：全量 clean 泛化（Exp7）

该表用于替换旧稿中“仅 8 条 train-clean 同源样本”的表述。结论是：LCA 在 `test-clean` 与 `test-other` 全量 clean 条件下对 mel-L1 和 STOI 有小幅一致收益，但 PESQ/SI-SNR 不是全面提升。

| 数据集 | n | L | Base mel-L1 ↓ | LCA mel-L1 ↓ | Δmel-L1 | Base STOI ↑ | LCA STOI ↑ | ΔSTOI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test-clean | 2620 | 1 | 1.226 | 1.205 | -0.021 | 0.801 | 0.813 | +0.013 |
| test-clean | 2620 | 2 | 0.976 | 0.946 | -0.030 | 0.864 | 0.871 | +0.007 |
| test-clean | 2620 | 3 | 0.907 | 0.871 | -0.036 | 0.885 | 0.892 | +0.007 |
| test-other | 2939 | 1 | 1.397 | 1.384 | -0.013 | 0.769 | 0.782 | +0.013 |
| test-other | 2939 | 2 | 1.146 | 1.116 | -0.030 | 0.832 | 0.839 | +0.008 |
| test-other | 2939 | 3 | 1.072 | 1.039 | -0.033 | 0.854 | 0.862 | +0.007 |

正文建议：

> 在未加扰动的 LibriSpeech 全量测试集上，LCA 并没有以牺牲 clean 重建为代价换取鲁棒性。相反，在 `test-clean` 和 `test-other` 的全部 `L=1/2/3` 工作点上，LCA 均降低 mel-L1 并提高 STOI；不过改善幅度较小，且 PESQ/SI-SNR 存在分层差异，因此 clean 条件结果应被解释为“小幅稳定收益”而非大幅质量提升。

## 3. 主表 B：全量 index dropout/substitution 鲁棒性（Exp8）

正文主表只保留 `dropout-high` 和 `substitution-high`。完整 `mid/high` 可放附录。

| 数据集 | n | L | 条件 | Δmel-L1 ↓ | ΔSTOI ↑ | ΔPESQ-WB ↑ | ΔSI-SNR ↑ |
|---|---:|---:|---|---:|---:|---:|---:|
| test-clean | 2620 | 1 | dropout-high | -0.0302 | +0.0168 | +0.0246 | -0.307 |
| test-clean | 2620 | 2 | dropout-high | -0.0405 | +0.0122 | +0.0132 | -0.167 |
| test-clean | 2620 | 3 | dropout-high | -0.0475 | +0.0123 | +0.0185 | +0.133 |
| test-clean | 2620 | 1 | substitution-high | -0.0319 | +0.0134 | +0.0245 | -0.393 |
| test-clean | 2620 | 2 | substitution-high | -0.0393 | +0.0084 | +0.0012 | -0.323 |
| test-clean | 2620 | 3 | substitution-high | -0.0451 | +0.0082 | +0.0122 | -0.030 |
| test-other | 2939 | 1 | dropout-high | -0.0184 | +0.0163 | +0.0223 | -0.310 |
| test-other | 2939 | 2 | dropout-high | -0.0375 | +0.0125 | +0.0062 | -0.269 |
| test-other | 2939 | 3 | dropout-high | -0.0412 | +0.0124 | +0.0075 | +0.070 |
| test-other | 2939 | 1 | substitution-high | -0.0283 | +0.0133 | +0.0259 | -0.384 |
| test-other | 2939 | 2 | substitution-high | -0.0432 | +0.0090 | +0.0085 | -0.407 |
| test-other | 2939 | 3 | substitution-high | -0.0429 | +0.0085 | +0.0087 | -0.072 |

正文建议：

> 在索引级 dropout 和 substitution 下，LCA 的优势比 clean 条件更稳定。对于两个测试集、三档码率和两类 high 扰动，LCA 均降低 mel-L1 并提高 STOI；PESQ 大多改善，SI-SNR 仍然混合。这说明 LCA 更可靠地改善了频谱稳定性和可懂度相关指标，而不是简单提升所有波形级指标。

## 4. 主表 C：全量 packet/burst loss 鲁棒性（Exp9）

正文主表建议只放 `packet-loss-5p` 与 `burst-10f`，它们代表较强包级/突发丢失场景。

| 数据集 | n | L | 条件 | Δmel-L1 ↓ | ΔSTOI ↑ | ΔPESQ-WB ↑ | ΔSI-SNR ↑ |
|---|---:|---:|---|---:|---:|---:|---:|
| test-clean | 2620 | 1 | packet-loss-5p | -0.0220 | +0.0136 | +0.0230 | -0.353 |
| test-clean | 2620 | 2 | packet-loss-5p | -0.0307 | +0.0081 | -0.0092 | -0.233 |
| test-clean | 2620 | 3 | packet-loss-5p | -0.0377 | +0.0078 | -0.0030 | +0.039 |
| test-clean | 2620 | 1 | burst-10f | -0.0213 | +0.0131 | +0.0231 | -0.354 |
| test-clean | 2620 | 2 | burst-10f | -0.0302 | +0.0077 | -0.0124 | -0.280 |
| test-clean | 2620 | 3 | burst-10f | -0.0369 | +0.0074 | -0.0090 | +0.030 |
| test-other | 2939 | 1 | packet-loss-5p | -0.0134 | +0.0137 | +0.0236 | -0.287 |
| test-other | 2939 | 2 | packet-loss-5p | -0.0303 | +0.0085 | -0.0053 | -0.344 |
| test-other | 2939 | 3 | packet-loss-5p | -0.0339 | +0.0080 | -0.0084 | -0.025 |
| test-other | 2939 | 1 | burst-10f | -0.0130 | +0.0133 | +0.0231 | -0.295 |
| test-other | 2939 | 2 | burst-10f | -0.0299 | +0.0083 | -0.0058 | -0.400 |
| test-other | 2939 | 3 | burst-10f | -0.0334 | +0.0076 | -0.0102 | -0.057 |

正文建议：

> packet/burst 评估更接近通信载荷在非独立错误下的表现。LCA 在所有代表性 packet/burst 条件下保持 mel-L1 与 STOI 的一致改善，说明其收益不只局限于独立索引替换；但 PESQ 在 L2/L3 下有时略低，SI-SNR 也不稳定，因此该实验应作为“通信化扰动下的频谱与可懂度鲁棒性证据”，而不是“所有音质指标提升”。

## 5. 主表 D：clean ASR/WER 子集（Exp10）

| 数据集 | n | L | Base WER ↓ | LCA WER ↓ | ΔWER | Base CER ↓ | LCA CER ↓ | ΔCER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test-clean | 300 | 1 | 0.3350 | 0.3089 | -0.0261 | 0.1908 | 0.1749 | -0.0160 |
| test-clean | 300 | 2 | 0.1304 | 0.1320 | +0.0016 | 0.0658 | 0.0695 | +0.0037 |
| test-clean | 300 | 3 | 0.1167 | 0.0933 | -0.0234 | 0.0618 | 0.0441 | -0.0178 |
| test-other | 300 | 1 | 0.6986 | 0.6941 | -0.0045 | 0.4671 | 0.4454 | -0.0217 |
| test-other | 300 | 2 | 0.4791 | 0.4625 | -0.0166 | 0.2867 | 0.2772 | -0.0095 |
| test-other | 300 | 3 | 0.3929 | 0.3665 | -0.0263 | 0.2201 | 0.2123 | -0.0077 |

正文建议：

> ASR 子集结果提供了任务层可懂度证据。在更困难的 `test-other_300` 上，LCA 在全部三档码率降低 WER；在 `test-clean_300` 上，L1 和 L3 改善，L2 基本持平。这与客观指标一致：LCA 更稳定地改善低负载和困难条件下的可懂度相关表现，但不能宣称每个 clean ASR 工作点都改善。

## 6. 主表 E：扰动 ASR/WER 子集（Exp11）

| 数据集 | n | L | 条件 | Base WER ↓ | LCA WER ↓ | ΔWER | Base CER ↓ | LCA CER ↓ | ΔCER |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| test-clean | 100 | 1 | dropout-high | 0.3321 | 0.3226 | -0.0095 | 0.1801 | 0.1721 | -0.0080 |
| test-clean | 100 | 1 | substitution-high | 0.3685 | 0.3272 | -0.0413 | 0.1981 | 0.1757 | -0.0224 |
| test-clean | 100 | 2 | dropout-high | 0.1514 | 0.1610 | +0.0096 | 0.0704 | 0.0783 | +0.0079 |
| test-clean | 100 | 2 | substitution-high | 0.1464 | 0.1528 | +0.0064 | 0.0617 | 0.0792 | +0.0175 |
| test-clean | 100 | 3 | dropout-high | 0.0971 | 0.1088 | +0.0117 | 0.0398 | 0.0527 | +0.0129 |
| test-clean | 100 | 3 | substitution-high | 0.1199 | 0.1278 | +0.0079 | 0.0494 | 0.0529 | +0.0036 |
| test-other | 100 | 1 | dropout-high | 0.6690 | 0.5921 | -0.0769 | 0.4466 | 0.3799 | -0.0667 |
| test-other | 100 | 1 | substitution-high | 0.6556 | 0.6409 | -0.0147 | 0.4179 | 0.4165 | -0.0014 |
| test-other | 100 | 2 | dropout-high | 0.5088 | 0.4449 | -0.0639 | 0.3212 | 0.2596 | -0.0616 |
| test-other | 100 | 2 | substitution-high | 0.4810 | 0.4348 | -0.0462 | 0.2976 | 0.2613 | -0.0364 |
| test-other | 100 | 3 | dropout-high | 0.4852 | 0.4066 | -0.0786 | 0.3113 | 0.2321 | -0.0791 |
| test-other | 100 | 3 | substitution-high | 0.3691 | 0.3205 | -0.0486 | 0.2201 | 0.1868 | -0.0333 |

正文建议：

> 在扰动 ASR/WER 上，两个子集呈现不同模式。`test-clean_100` 的 WER 收益主要集中在最低码率 L=1；L=2/3 下 LCA 略差。相反，在更困难的 `test-other_100` 上，LCA 对所有 L 和两类扰动均降低 WER/CER。这说明 LCA 的可懂度鲁棒性在困难语音和信道扰动叠加时更明显，但论文需要保留 clean 子集高层码率收益不稳定的边界。

## 7. 正文结构建议

建议把结果章节更新为：

1. **5.1 低负载 baseline 对比**：保留旧 Exp4 8 条固定样本表，说明 500/1000/1500 bps 操作带和同码率优势。
2. **5.2 全量 clean 泛化**：新增 Exp7 表，说明在 test-clean/test-other 全量上 clean 不退化并小幅改善。
3. **5.3 索引级扰动鲁棒性**：新增 Exp8 表，覆盖 dropout/substitution。
4. **5.4 packet/burst 丢包鲁棒性**：新增 Exp9 表，体现更通信化错误模型。
5. **5.5 ASR/WER 可懂度验证**：合并 Exp10/Exp11，clean ASR 与扰动 ASR 分两段。
6. **5.6 设计消融**：保留蒸馏、LCA v1/v2、NAS 效率与组件因子消融。

## 8. 需要同步改掉的旧说法

- 摘要中“当前评估仍限于小样本、同源语料”应改为：主 baseline 对比仍基于固定样本，但已经补充 LibriSpeech test-clean/test-other 全量 clean 与扰动评估；仍缺 VCTK/AISHELL、主观听测和真实网络验证。
- 实验设置 4.1 中“固定评估样本为 8 条”应改为分层说明：训练/主 baseline 固定样本、全量 LibriSpeech clean/perturb、ASR 子集。
- 局限性中“尚不能代表 test-clean/test-other”应删除，改为“尚未覆盖跨语料、跨语言、主观听测、真实网络”。
- 结论中可新增：LCA 在 LibriSpeech 全量 clean/perturb 评估中稳定改善 mel-L1/STOI，在 test-other 扰动 ASR 子集上降低 WER。
