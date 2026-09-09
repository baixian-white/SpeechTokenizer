# SCIT-Speech 后续补充实验执行文档

> 用途：供后续 AI / 研究助理按步骤补齐论文修订版 `output/doc/paper_drafts/scit_speech_cn_revised_ml_20260610.md` 中尚未完成的实验证据。本文档只列需要执行的新实验或后处理，不重复已经完成的 exp1–exp11。
>
> 总原则：所有新实验必须写入 `output/experiments/` 下新的 run 目录；所有结果必须包含机器可读 CSV/JSON、Markdown summary、命令记录、配置快照、checkpoint sha256（如适用）。不得直接覆盖现有 exp1–exp11。

## 0. 当前论文缺口总览

当前修订稿已经把强主张降级，但仍有以下证据缺口：

1. 同码率 baseline 只有 8 条固定样本，无置信区间；
2. Codec2 / AMR-WB / 主观听测缺失；
3. NAS 只有效率指标，无 Pareto 邻居代理质量表，也无 hand-designed encoder 同条件全训质量对照；
4. LCA v1/v2 与 V0–V4 的部分统计仍需统一后处理；
5. `test-clean` 高码率 ASR 退化模式尚无诊断；
6. 一致性损失的训练动力学机制尚未验证；
7. codebook 利用率不健康，需要修复实验；
8. 真实 entropy-coded 比特率与 packetized payload 开销尚未核算。

### 完成状态总表（截至 2026-06-12 更新）

下表把本文档的 E 编号映射到 `expN` 实际 run，并标注完成状态与实验记录对应小节（详见 `output/doc/实验记录.md`）。**多数项已完成，但尚未回写进论文**（按用户要求结构 review 先行）。

| E 编号 | expN | 名称 | 状态 | 实验记录 |
|---|---|---|---|---|
| E1 | exp12 | baseline 扩到 test-clean_300 / test-other_300（+95% CI + Wilcoxon） | ✅ 完成 | §10 |
| E2 | exp12/codec2 | Codec2 baseline | ⛔ 工具链阻塞 | §10.5 |
| E3 | exp5b | V0–V4 因子化 n=256 配对统计 | ✅ 完成 | §11.1 |
| E4 | exp3c | LCA v1/v2 统计后处理 | ✅ 完成 | §11.2 |
| E5 | exp1b | NAS Pareto 邻居与代理质量表 | ✅ 完成 | §11.3 |
| E6 | exp13 | hand-designed encoder 同条件全训（方案 B：30 epoch） | 🔄 训练中（约 68%） | §13.1 |
| E7 | exp5c | 一致性损失训练动力学诊断 | 未启动 | — |
| E8 | exp14 | test-clean 高码率 ASR 退化诊断 | ✅ 完成 | §13.2 |
| E9 | exp15 | codebook 利用率修复 | 未启动 | — |
| E10 | exp16 | entropy-coded 真实比特率核算 | ✅ 完成 | §13.3 |
| E11 | exp17 | packetized payload 工程开销建模 | ✅ 完成 | §13.4 |
| E12 | exp18 | 主观 AB / MOS 听测 | 未启动（需外部听者） | — |
| E13 | exp19 | 跨语料 / 跨语言 zero-shot | 未启动 | — |

**论文回写状态**：以上完成项均**尚未写入论文**。回写优先级与落点见实验记录 §13.5。下文各 E 节的"执行步骤"保留作复现参考。

---

## P0：最优先，直接影响当前论文可信度

### E1. 扩展同码率 baseline 到 test-clean_300 / test-other_300

**目的**：把当前 exp4 的 8 条样本快照升级为有统计意义的低码率 baseline 对照。

**输入**：
- SCIT-Speech-Base checkpoint：`output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- SCIT-Speech-LCA checkpoint：`output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`
- LibriSpeech `test-clean`、`test-other`
- baseline codec：Opus、EnCodec、DAC；优先补 Codec2（见 E2）

**输出目录建议**：
`output/experiments/exp12_baseline_comparison_test300_20260610_seed42/`

**执行步骤**：
1. 生成 `test-clean_300` 与 `test-other_300` 固定 sample list，seed=42，与 exp10 子集保持一致。
2. 对每条样本导出原始音频、SCIT Base/LCA L=1/2/3 重建音频。
3. 对 Opus、DAC、EnCodec 运行与 exp4 相同配置。
4. 如已完成 E2，则加入 Codec2 700/1200/2400 bps。
5. 计算 mel-L1、STOI、PESQ-WB、SI-SNR、WER/CER。
6. 对每个方法/码率 cell 报告 mean、median、95% bootstrap CI。
7. 对关键比较做 paired bootstrap / Wilcoxon：
   - LCA L=1 vs Codec2 700 bps；
   - LCA L=2 vs Codec2 1200 bps；
   - LCA L=3 vs EnCodec 1.5 kbps / DAC n_q=3 / Opus 6 kbps。

**产出文件**：
- `metrics/baseline_results.csv`
- `metrics/bootstrap_ci.csv`
- `reports/baseline_test300_summary.md`
- `reports/key_pairwise_tests.md`
- `artifacts/sample_lists/test-clean_300.txt`
- `artifacts/sample_lists/test-other_300.txt`

**论文落点**：替换 §5.1 表 1 与图 2；若结果支持，可把“8 样本快照”改为“300 样本子集统计对照”。

**成功标准**：
- 所有方法在 ≥95% 样本上成功产出音频与指标；
- bootstrap CI 不含明显异常值；
- 关键表中明确标注 n 与 CI。

---

### E2. Codec2 baseline

**目的**：补齐 500–1500 bps 区间最直接的传统语音 codec 对照。

**输出目录建议**：
并入 E1：`output/experiments/exp12_baseline_comparison_test300_20260610_seed42/codec2/`

**配置建议**：
- Codec2 700 bps
- Codec2 1200 bps
- Codec2 2400 bps（超出主区间，但作为参考）

**执行步骤**：
1. 确认可用 codec2 CLI 或 Python binding。
2. 对 E1 的 sample list 批量编码/解码。
3. 保持输入输出 16 kHz；若 Codec2 要求 8 kHz，必须记录 resampling 过程，并在表中标注。
4. 计算与 E1 相同指标。

**产出文件**：
- `metrics/codec2_results.csv`
- `reports/codec2_summary.md`
- `reports/codec2_failure_report.md`

**论文落点**：§4.2 baseline、§5.1 主表、§7 删除/弱化 Codec2 缺失局限性。

**成功标准**：Codec2 700/1200 至少两档完整评估；失败样本数 <5%。

---

### E3. V0–V4 因子化统计补全

**目的**：修订稿 §6.3 中 `V3−V1` 的 CI、t、p、显著单元数仍标为“待补”；需要从已有 n=256 结果直接后处理补全。

**输入目录**：
`output/experiments/exp5_lca_component_factorial_20260603_seed42/eval_unified_20260605/`

**输出目录建议**：
`output/experiments/exp5_lca_component_factorial_20260603_seed42/reports/statistical_tests_20260610/`

**执行步骤**：
1. 读取 V0/V1/V2/V3/V4 在 n=256 × L × condition 上的 per-sample mel-L1。
2. 对每个样本、L、扰动条件，计算：
   - degradation = perturbed_mel_l1 − clean_mel_l1
   - robust_imp = Base_degradation − Variant_degradation
3. 计算以下配对差值：
   - V3 − V1（在 random-L 上加入 ChannelSim 的干净边际路径）
   - V4 − V3（加入 consistency）
   - V3 − V2（在 ChannelSim 上加入 random-L）
   - V2 − V1（配置对比，非纯边际）
   - V4 − V0（端到端总效益）
4. 对 12×256 个配对点做 paired t-test，并同时输出 bootstrap CI。
5. 对每个单元单独做 paired t-test，统计 p<0.05 的单元数。

**产出文件**：
- `metrics/factorial_per_sample_robust_imp.csv`
- `metrics/factorial_pairwise_tests.csv`
- `reports/factorial_stats_summary.md`

**论文落点**：替换 §6.3 表中“待补”的 `V3−V1` 行；更新摘要中对三组件贡献的简述。

**成功标准**：`V3−V1` 的 CI、t、p、显著单元数补全；所有表格可复现当前 §6.3 已有数字。

---

### E4. LCA v1 vs v2 统计后处理

**目的**：让 §6.2 与 §6.3 的统计标准对齐。

**输入目录**：
- v1：`output/experiments/exp3_low_load_channel_aware_adaptation_20260530_seed42/`
- v2：`output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/`

**输出目录建议**：
`output/experiments/exp3_lca_v1_v2_stats_20260610/`

**执行步骤**：
1. 找到 v1/v2 的 per-sample Base vs LCA evaluation CSV。
2. 按相同样本、L、扰动条件配对。
3. 对每个指标计算 robust_imp。
4. 计算 v2 − v1 的 paired mean、95% CI、p 值。
5. 输出全指标表，至少包含 mel-L1、STOI、PESQ-WB、SI-SNR。

**产出文件**：
- `metrics/v1_v2_pairwise_robust_imp.csv`
- `reports/v1_v2_stats_summary.md`

**论文落点**：§6.2 增加统计显著性行；如果样本量仍是 8，则明确“n=8，仅辅助对照”。

**成功标准**：能复现当前 §6.2 均值与 cell 计数，并新增 CI/p 值。

---

## P1：强化方法贡献与可解释性

### E5. NAS Pareto 邻居与代理质量表

**目的**：补齐 NAS 不只有效率指标、还处于合理 Pareto 位置的证据。

**输入目录**：
`output/experiments/exp1_nas_distill_run1_seed42/`

**输出目录建议**：
`output/experiments/exp1_nas_distill_run1_seed42/reports/pareto_neighbors_20260610/`

**执行步骤**：
1. 读取 stage4 top candidates 的代理质量与资源指标。
2. 找到最终候选 `seed42-000896`，以及 Pareto 邻居：
   - 更轻但质量略差的候选 1–2 个；
   - 更重但质量略好的候选 1–2 个；
   - hand-designed encoder profile。
3. 输出表格字段：candidate_id、params、MACs、RTF、teacher latent SmoothL1、cosine distance、temporal diff、quantized feature L1、semantic proxy、mel proxy、wave proxy、Pareto rank。
4. 画 Pareto 图：x=MACs 或 params，y=proxy mel / latent loss，标出 `000896`。

**产出文件**：
- `metrics/nas_stage4_pareto.csv`
- `figures/nas_pareto_macs_vs_proxy.png`
- `reports/nas_pareto_summary.md`

**论文落点**：§6.4 增加代理质量表；§7 删除“Pareto 代理质量未纳入正文”局限性。

**成功标准**：至少 5 个候选 + hand-designed profile；能说明为什么选择 `000896`。

---

### E6. Hand-designed encoder 同条件完整训练

**目的**：回答 NAS encoder 是否仅提升效率，还是影响质量；消除当前最大未控变量。

**输出目录建议**：
`output/experiments/exp13_handdesigned_encoder_distill30_20260610_seed42/`

**实验设计**：
- 固定 train-clean-100、RVQ 层数、码本大小、decoder、loss、训练 schedule 与 exp2 distill30 尽量一致；
- 唯一变量：encoder 使用 hand-designed SEANet；
- 训练预算：优先完整 60 epoch；如资源不足，至少跑到 dev/mel 收敛平台并记录 stop reason。

**执行步骤**：
1. 准备 config：把 NAS encoder config 替换为 hand-designed encoder。
2. 从头训练 Base，不加载 NAS Base 权重。
3. 打包 best checkpoint，记录 sha256。
4. 在 exp7 的 test-clean/test-other clean 全量上至少评估 L=1/2/3 的 mel-L1、STOI、PESQ-WB。
5. 与 NAS Base 对比 params/MACs/RTF/quality。

**产出文件**：
- `checkpoints/SCIT-Speech-Base_handdesigned_best.pt`
- `metrics/full_clean_eval.csv`
- `reports/handdesigned_vs_nas_summary.md`

**论文落点**：§6.4 或新 §6.5；如果质量接近，可强化“NAS 是效率优化”；如果质量差距大，需重写贡献声明。

**成功标准**：完成同条件训练与至少 clean 全量评估。

---

### E7. 一致性损失训练动力学诊断

**目的**：验证 §3.3 中“consistency 与 reconstruction 配合，避免扰动下复制偏差”的机制解释。

**输入目录**：
`output/experiments/exp5_lca_component_factorial_20260603_seed42/`

**输出目录建议**：
`output/experiments/exp5_lca_component_factorial_20260603_seed42/reports/consistency_dynamics_20260610/`

**执行步骤**：
1. 从 V3 与 V4 的训练 logs/TensorBoard 中提取：
   - dev/full_mel
   - dev/comm_mel by L/condition
   - clean-vs-perturbed mel distance（如已有）
2. 若 clean-vs-perturbed mel distance 未记录，则在 V3/V4 checkpoints 上离线计算：
   - `mel(x, xhat_clean)`
   - `mel(x, xhat_pert)`
   - `mel(xhat_clean, xhat_pert)`
3. 对 L=1/2/3 与 dropout/substitution-high 输出表格。
4. 判断 V4 是否在降低 clean-pert distance 的同时没有显著损害 clean reconstruction。

**产出文件**：
- `metrics/consistency_diagnostics.csv`
- `figures/consistency_dynamics.png`
- `reports/consistency_dynamics_summary.md`

**论文落点**：§3.3 或 §6.3 末尾增加机制诊断；若不支持，保留“设计意图”而不写成机制结论。

**成功标准**：至少 V3/V4 final checkpoint 离线诊断完成；最好有训练过程曲线。

---

### E8. test-clean 高码率 ASR 退化诊断

**目的**：解释 LCA 在 `test-clean` L=2/L=3 扰动 ASR 上略差的模式。

**输出目录建议**：
`output/experiments/exp14_testclean_asr_regression_diagnosis_20260610/`

**执行步骤**：
1. 使用 exp10/exp11 的 per-sample ASR 输出。
2. 找出 LCA 比 Base WER 更差的样本，按：
   - 原始 PCM Whisper 是否已接近 0 错误；
   - Base/LCA mel-L1 差异；
   - STOI/PESQ 差异；
   - utterance 长度、speaker、文本长度；
   - 具体替换/置 0 位置比例；
   分组统计。
3. 导出 10 个代表性 regression cases 的音频与转写对比。
4. 对 test-clean_100 与 test-other_100 分别画 ΔWER vs Base WER scatter。

**产出文件**：
- `metrics/asr_regression_cases.csv`
- `reports/testclean_asr_regression_summary.md`
- `samples/regression_case_audio/`

**论文落点**：§6.5 从假设性解释升级为数据诊断。

**成功标准**：能说明退化是否集中在 Base 已很好的样本、短句、特定 speaker 或特定扰动条件。

---

## P2：期刊版/增强版实验

### E9. Codebook 利用率修复实验

**目的**：解决当前 codebook usage 17–27%、L1 dead code 高的问题。

**候选方法**：
1. dead code reinit；
2. k-means 初始化；
3. usage entropy regularization；
4. 分层 commitment 权重调整。

**输出目录建议**：
`output/experiments/exp15_codebook_health_20260610_seed42/`

**最低设计**：先做 dead-code reinit + LCA v2 微调一组，与原 LCA v2 对比：usage、mel-L1、STOI、WER、robustness。

**论文落点**：若成功，可作为新 ablation；若不成功，保留为局限性。

---

### E10. Entropy-coded 真实比特率核算

**目的**：把 `500L bps` 从 naive bit-packed 上界扩展到 empirical entropy rate。

**执行步骤**：
1. 在 test-clean/test-other 上统计各 L 层索引分布。
2. 计算每层经验 entropy `H(I_l)`、联合/条件 entropy（如可行）。
3. 报告 `R_entropy(L) = f_q · Σ_{l=1}^L H(I_l)`。
4. 与 `500L` bps 对比。

**输出目录建议**：
`output/experiments/exp16_entropy_rate_20260610/`

**论文落点**：§3.1 或 §7；可将 codebook dead code 从负面 caveat 转化为“当前 bit-packed 上界保守”。

---

### E11. Packetized payload 工程开销

**目的**：将理想索引负载映射到实际 packet payload。

**执行步骤**：
1. 设计简单 packet 格式：header + timestamp + L + packed indices。
2. 以不同 packet duration（20/40/80 ms）打包。
3. 计算 payload bps、UDP/IP overhead 后总 bps。
4. 对 packet-loss 与 burst loss 模型给出对应帧数解释。

**输出目录建议**：
`output/experiments/exp17_packetized_payload_20260610/`

**论文落点**：§3.1、§7 或工程附录。

---

### E12. 主观 AB / MOS 小规模听测

**目的**：补客观指标不能代表听感的缺口。

**最低设计**：
- 30–50 条样本；
- 方法：Original、SCIT LCA L=1/2/3、DAC 1.5 kbps、EnCodec 1.5 kbps、Opus 6 kbps；
- AB preference 或 MOS 1–5；
- 至少 10 名听者。

**输出目录建议**：
`output/experiments/exp18_subjective_listening_20260610/`

**论文落点**：新增 §5.x 主观评估；或作为 future work。

---

### E13. 跨语料 / 跨语言 zero-shot

**目的**：验证不只在 LibriSpeech 上成立。

**数据建议**：
- VCTK（英文跨说话人）；
- AISHELL-1（中文，跨语言）；
- 如有真实通信语音，也加入。

**执行步骤**：
1. 不重训，仅用 Base/LCA checkpoint zero-shot 重建。
2. 评估 mel-L1、STOI、PESQ-WB、SI-SNR；中文可用 CER ASR（如 Whisper multilingual 或另一个中文 ASR）。
3. 报告相对 LibriSpeech 的退化。

**输出目录建议**：
`output/experiments/exp19_cross_corpus_zeroshot_20260610/`

**论文落点**：扩展 §5 或 §7。

---

## 推荐执行顺序

1. **E3 V0–V4 因子化统计补全**：最快，直接填当前修订稿“待补”。
2. **E5 NAS Pareto 邻居与代理质量表**：无须训练，补 §6.4 说服力。
3. **E1 + E2 扩展 baseline + Codec2**：最能提升 §5.1。
4. **E8 test-clean ASR 退化诊断**：补 §6.5。
5. **E4 v1/v2 统计后处理**：与 §6.2 对齐。
6. **E7 consistency 动力学诊断**：支持机制解释。
7. **E6 hand-designed encoder 完整训练**：耗时最大，但最关键。
8. P2 组（E9–E13）按投稿时间与算力选择。

## 后续 AI 执行注意事项

- 每个实验必须先写 `commands/` 目录保存完整命令。
- 所有 summary 必须区分“已完成事实”和“解释/推测”。
- 不要覆盖 exp1–exp11。
- 不要把 smoke test 或 1 条样本结果写入论文主表。
- 新结果若与当前修订稿矛盾，应以新结果为准并回写论文。
- 所有 WER 必须注明 ASR 模型与解码参数。
- 所有统计检验必须注明样本粒度：utterance 级、frame 级，还是 12×n cell pooling。·
