# exp19 跨语料 zero-shot 评估 — 实施计划

## 目标
补论文最大局限（§7-1"全部基于 LibriSpeech，跨语料未覆盖"）。用现有 SCIT-Speech-Base + LCA-v2 ckpt（**不重训**），在 VCTK（英文跨说话人）上做 zero-shot 客观评估（mel-L1/STOI/PESQ-WB/SI-SNR），与 LibriSpeech test 结果对比，证明方法跨语料泛化。

## 范围决定（已与用户确认）
- **数据集自己下**（用户已批准下载，H: 剩 477GB，充足）
- 优先 VCTK；AISHELL 中文留作可选第二阶段（中文 ASR 需 multilingual Whisper，复杂度高）
- 本计划只做 VCTK 的**客观重建评估**（不做 ASR/WER），与 exp7 同口径

## 输入（全部已确认存在）
- Base: `exp2.../configs/scit_speech_base_config.json` + `SCIT-Speech-Base_best.pt`
- LCA: `exp6.../configs/full_lca_clean_eval_config.json` + `exp3v2.../SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`
- 评估器: `scripts/evaluate_clean_large_nosave.py`（exp7 已验证；sample-list 每行一个音频路径即可，无需 feature 列）

## 步骤

### 1. 下载 VCTK 0.92（~11GB，不可逆占盘）
- URL: `https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip?sequence=2&isAllowed=y`
- 落到 `data/VCTK-Corpus-0.92.zip`，解压到 `data/VCTK/`
- 后台下载（10GB+，用 Monitor 盯完成）

### 2. 建 run 目录 + 采样 sample list
- `output/experiments/exp19_cross_corpus_zeroshot_20260613_seed42/`
- VCTK 是 48kHz wav（评估器内部会 resample 到 16kHz）；格式确认后建 sample list
- **子集采样**：VCTK 全量 ~44000 条太多（评估按 exp7 节奏每条数秒）。采 300 条（跨多说话人均匀采样，seed=42），与 exp10/exp12 的 300 子集口径一致，控制在 ~1-2h GPU
- sample list 每行一个 wav 绝对路径

### 3. 运行客观评估（GPU，~1-2h）
- 复用 exp7 的精确调用：`evaluate_clean_large_nosave.py --base-config/-checkpoint --lca-config/-checkpoint --sample-list <vctk_300> --run-dir <exp19> --device cuda`
- 产出 `metrics/full_clean_results.csv`（300×3L×2model=1800 行）+ summary
- 用廉价心跳监控（ckpt 无、改盯 CSV 行数增长 + 进程存活）

### 4. 完整性检查 + 写记录
- CSV 行数校验（1800）、NaN/Inf 检查、独立复算关键 cell
- 与 exp7 LibriSpeech test 结果对比（VCTK vs test-clean/other 的 mel-L1/STOI/PESQ 退化幅度）
- 写 `实验记录.md` §15（事实+数字+caveats），更新 §12 状态表

## 诚实边界（写进记录）
- VCTK 48kHz→16kHz 重采样会损失高频，PESQ/STOI 绝对值受影响，但 Base/LCA 同处理、相对对比有效
- 300 子集非全量，跨说话人采样需说明采样方法
- 仅客观指标，无 VCTK ASR/WER（无 ground-truth transcript 对齐成本高）
- AISHELL 中文未做（留 future work 或第二阶段）

## 不做 / 风险
- 不重训任何模型
- 不碰主代码、不碰论文正文（结论先落实验记录）
- 下载失败/解压失败有兜底：先验证 zip 完整性再解压
- VCTK 若实际是 flac/不同采样率，step 2 会先验证再继续

## 预估
下载 ~30min（取决于网速）+ 解压 ~10min + 评估 ~1-2h GPU。总计半天内。
