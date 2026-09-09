# SCIT-Speech 方法学路线补充实验设计

> 用途：在方法学路线下，把现有实验（exp1-5）的证据短板补完，使方法学声明可以无懈可击地立住
> 这份文件是给 **你自己执行** 的，不是给 AI 的 prompt
> 完成后再使用 `prompt_methodology_full_experiments.md` 启动论文写作

---

## 总览

按优先级三档分级。Tier 1 是 contribution 闭环必做；Tier 2 是审稿稳健性强烈推荐；Tier 3 是加分项。

| Tier | 实验编号 | 实验名 | GPU 时长 | 工程时长 | 闭合的短板 |
|---|---|---|---|---|---|
| 1 | exp7 | Hand-designed encoder 同条件全训对照 | ~30-36 h | 0.5 d | C4 NAS protocol 闭环 |
| 1 | exp8 | λ_sem distillation strength sweep (5 点) | ~75-120 h | 0.5 d | A3 反直觉发现升级为完整 sweep |
| 1 | exp9 | LCA 4 组件 factorial 消融 | ~48 h | 1 d | C3 confound 解开 |
| 2 | exp10 | 跨语料评估 (test-clean / test-other / VCTK) | 0 h (推理) | 0.5 d | n=8 同源问题降级 |
| 2 | exp11 | Bit-level BSC + packet-level GE 信道 | 0 h (推理) | 1.5 d | "channel-aware" 名副其实 |
| 2 | exp12 | MOS-lite 主观听测 | 0 h | 1.5 d | 主观维度补背书 |
| 3 | exp13 | Codebook utilization repair | ~90-120 h | 1 d | codebook 健康度从 caveat 翻正 |
| 3 | exp14 | Base 直接做信道扰动评估 | 0 h | 0.5 d | 强化 C3 对照 |

**Tier 1+2 总投入估计**：约 177 GPU 小时（6-8 天单 GPU）+ 5-7 工作日工程时间，单人单 GPU 全套约 **2-3 周**。

---

## Tier 1：必做实验（闭合 4 条 contribution 的方法学可信度）

### 实验 7：Hand-designed encoder 同条件全训对照

| 字段 | 值 |
|---|---|
| 目的 | 闭环 C4（NAS protocol 的方法学贡献），验证 NAS encoder 与 hand-designed encoder 在 matched-budget 下的质量等价 |
| 基础 | 复用 SCIT-Speech-Base 配置，仅替换 encoder 为 hand-designed SeaNet（strides [8,5,4,2]，n_filters=32，default 配置）|
| 训练 | 60 epochs, λ_sem=30, seed=42，与 exp2 完全一致 |
| 预计 GPU 时长 | 30-36 小时（hand-designed encoder 计算量大约是 NAS encoder 的 8-9 倍）|
| 关键产出 | best dev/mel、WER@L=1/2/3、codebook usage 三层分布、fixed_sample 8 条客观指标 |
| 论文用途 | §6 增加 Table "NAS vs Hand-designed at matched compute"；§4.2 NAS protocol 主张可加上 "with verified quality equivalence" 限定语 |
| run_id 建议 | `exp7_hand_designed_encoder_baseline_<date>_seed42` |

### 实验 8：λ_sem distillation strength sweep

| 字段 | 值 |
|---|---|
| 目的 | 把 A3 从二值消融升级为可信因果声明 |
| 配置 | λ_sem ∈ {0, 10, 30, 60, 120} 共 5 个训练 run，每 run 60 epoch |
| 节省版（推荐）| 各 run 只跑到 step 47500（约 14.8 epoch），因为 exp5 已证明该 step 的差距已经决定性。每 run ~15 小时，5 run = 75 GPU 小时 |
| 完整版 | 5 run × 24 小时 = 120 GPU 小时；产出 trained-to-convergence 的最优 λ_sem |
| 关键产出 | dev/mel 与 WER 关于 λ_sem 的曲线；codebook usage 关于 λ_sem 的曲线 |
| 论文用途 | §6.4 A3 改写为完整 sweep 图（曲线显示 sweet spot）；C3+C4 之外加上一条 "auxiliary methodological observation"："excessive distillation strength saturates / hurts; too little fails to organize content"。把"反直觉发现"升级为"sweet-spot 系统观察" |
| run_id 建议 | `exp8_distill_sweep_lambda<value>_<date>_seed42` |

### 实验 9：LCA 训练目标 4 组件 factorial 消融

| 字段 | 值 |
|---|---|
| 目的 | 闭环 C3（the proposed channel-aware objective），把 confound 解开 |
| 基础 | 全部从 SCIT-Speech-Base 起做 10 epoch fine-tune |
| 配置 4 个 run | A: random-L only（L=1,2,3 随机采样，不做 ChannelSim，不做 consistency）<br>B: ChannelSim only（强扰动，但 L 固定 = 3，不做 consistency）<br>C: random-L + ChannelSim（无 consistency）<br>D: random-L + ChannelSim + consistency（= 现有 SCIT-Speech-LCA）|
| 预计 GPU 时长 | 4 × 12 小时 = 48 GPU 小时 |
| 评估 | 每个 run 跑现有 base_vs_lca_summary 的 8×3×5×6 = 720 个评估对，加上 robust_imp 的 12-cell 聚合 |
| 关键产出 | 2x2x2 factorial table（每个 run 的 clean mel_l1、clean WER、平均 robust_imp）|
| 论文用途 | §6.3 升级为 per-component contribution within the proposed objective，使 C3 的方法学声明具体化为 each component is necessary; the combination is optimal |
| run_id 建议 | `exp9_lca_factorial_<variant>_<date>_seed42` |

---

## Tier 2：强烈推荐实验（避免被审稿人逼出"补实验"决议）

### 实验 10：跨语料泛化评估

| 字段 | 值 |
|---|---|
| 目的 | 把 n=8 train-clean 同源问题降级到 acceptable 水平 |
| 数据 | LibriSpeech test-clean: 50 utterances（held-out，10-15 秒长度匹配现有 fixed_sample）<br>LibriSpeech test-other: 50 utterances<br>VCTK: 30 utterances（不同 speaker、不同麦克风、英语）|
| 不需要训练 | 复用现有 SCIT-Base、SCIT-LCA、DAC、EnCodec、Opus checkpoints 做推理 |
| 评估 | 全部 6 项客观指标 + Whisper WER |
| 预计耗时 | 8-12 小时（推理 + 评估）|
| 关键产出 | 扩展 Table 2 增加 test-clean / test-other / VCTK 三列 |
| 论文用途 | §6.1 主结果章节直接用跨语料数据 footing；§8 第 3 条降级为"未覆盖远场 / 噪声 / 多语言泛化"|
| run_id 建议 | `exp10_cross_corpus_eval_<date>` |

### 实验 11：Bit-level / packet-level channel model

| 字段 | 值 |
|---|---|
| 目的 | 让 channel-aware 措辞名副其实，挡审稿人 this is just denoising autoencoder 的反驳 |
| 11a bit-level | BSC + index 重映射：把 index bit-pack 成比特流；BSC with BER ∈ {1e-4, 1e-3, 5e-3, 1e-2}；解码侧若 unpacked index 越界，按 previous-frame replacement 处理；评估现有 SCIT-Base 与 SCIT-LCA |
| 11b packet-level | Gilbert-Elliott burst loss：每 N 个连续 latent frame 打成一个包（N=10, 20, 50）；使用 GE 模型，平均 loss rate ∈ {0.05, 0.10, 0.15}；解码侧丢失 packet 替换 |
| 11c 可选 | 用上述 channel 重新 fine-tune LCA，看是否进一步提升 |
| 预计耗时 | 实现 1-2 天 + 评估 8-12 小时（不重训）|
| 关键产出 | WER/PESQ 关于 BER 的曲线；WER/PESQ 关于 packet loss rate 的曲线；与 Glaris 的 PLC results 对照 |
| 论文用途 | §3.4 新增物理信道章节；§7 新增独立小节 "From Index-Level to Physical-Layer Channels"；§8 第 4 条降级 |
| run_id 建议 | `exp11_physical_channel_eval_<date>` |

### 实验 12：MOS-lite 主观听测

| 字段 | 值 |
|---|---|
| 目的 | 客观指标补一个最小主观背书 |
| 配置 | 8-12 名听者；A-B preference 测试，每对 6-10 个语音对 |
| 对比对 | SCIT-LCA L=2 (1000 bps) vs Opus 6 kbps<br>SCIT-LCA L=3 (1500 bps) vs DAC n_q=3<br>SCIT-LCA L=3 vs Opus 6 kbps<br>SCIT-Base L=3 vs SCIT-LCA L=3（看 LCA 是否在听感上劣化）|
| 预计耗时 | 1-2 天（招募 + 听测 + 统计）|
| 关键产出 | 每对的 preference 比例 + 95% CI |
| 论文用途 | §6 新增 Table subjective AB preference；§8 第 3 条进一步降级 |
| run_id 建议 | `exp12_subjective_listening_<date>` |

---

## Tier 3：加分项实验

### 实验 13：Codebook utilization repair

| 字段 | 值 |
|---|---|
| 目的 | 把 codebook 利用率从 caveat 翻成 positive contribution |
| 配置 多个 run | SCIT-Base + dead-code reinit（每 N step 重置长期未用 codeword）<br>SCIT-Base + k-means initialization<br>SCIT-Base + usage entropy regularization (λ=1e-3)|
| 预计 GPU 时长 | 3 × 60 epoch ≈ 90-120 GPU 小时 |
| 关键产出 | 每种修复策略下的 codebook usage 与下游 WER；选出最优策略 |
| 论文用途 | 升级 C1 包含 with codebook utilization safeguards；§6.4 增加独立 codebook health 小节 |
| run_id 建议 | `exp13_codebook_repair_<strategy>_<date>_seed42` |

### 实验 14：Base 直接做信道扰动评估

| 字段 | 值 |
|---|---|
| 目的 | 进一步加强 C3 的对照（避免审稿人怀疑 LCA 的提升只是因为 fine-tune 时间长）|
| 配置 | 拿 exp2 的 SCIT-Base 直接在 ChannelSim 各档下评估，不微调 |
| 耗时 | 评估 6 小时 |
| 论文用途 | §6.3 增加 Base under perturbation without LCA 行 |
| run_id 建议 | `exp14_base_under_perturbation_<date>` |

---

## 执行建议顺序

1. **Day 1-2**：启动 exp7（30-36 h, 后台跑）+ 同时实现 exp10 跨语料评估脚本
2. **Day 2-3**：exp7 跑完 + exp10 推理 + 启动 exp9 factorial 4 个 run（~48h, 后台跑）
3. **Day 4-5**：exp9 跑完 + 启动 exp8 distillation sweep 5 个节省版 run（~75h, 后台跑）
4. **Day 6-8**：exp8 跑完 + 同时实现 exp11 物理信道评估脚本 + exp12 招募听者
5. **Day 9-10**：exp11 推理 + exp12 听测
6. **Day 11-14**：所有结果汇总到 实验记录.md，更新 base_vs_lca_summary，写论文（用 `prompt_methodology_full_experiments.md`）

---

## 完成后必须更新的文件

补完实验后，启动论文写作 prompt 之前，请务必更新：

1. `output/doc/实验记录.md`：每个新 exp 加一节，包含 run_id、配置、关键数字、caveats
2. `output/doc/citation_pool.md`：如新增 baseline 或 channel model 引用
3. `output/doc/literature_review_2026_06_02.md`：在 caveat 节注明 Q1-Q5 缺口闭合状态变化

更新完毕再使用 `prompt_methodology_full_experiments.md` 写论文。
