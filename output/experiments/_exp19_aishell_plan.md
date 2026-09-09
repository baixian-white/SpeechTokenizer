# exp19-AISHELL 中文跨语料评估 — 实施计划

## 目标
扩展 exp19 到中文：AISHELL-1 zero-shot（不重训）客观重建质量 + 中文 ASR/CER 可懂度。
证明 SCIT-Speech 跨**语言**（非仅跨语料）泛化。补论文 §7-1 局限性的中文部分。

## 范围（已与用户确认）：客观指标 + 中文 CER

## 数据
- AISHELL-1，OpenSLR slr33，`data_aishell.tgz` ~15GB，**原生 16kHz**（无需重采样，比 VCTK 省事）
- URL: `https://openslr.trmal.net/resources/33/data_aishell.tgz`（备 EU/CN 镜像）
- **嵌套结构**：外层 tgz 解压后，`data_aishell/wav/{train,dev,test}/SXXXX.tar.gz` 每说话人一个内层 tar，需二次解压
- transcript: `data_aishell/transcript/aishell_transcript_v0.8.txt`（格式：`BAC009SXXXX_WNNN 汉字 汉字 ...`，空格分隔的词，但 CER 按字符算）
- 磁盘：H: 剩 454GB，充足

## 步骤

### 1. 下载（~15GB，后台，最长步骤）
- 落 `data/_downloads/data_aishell.tgz`，curl 后台 + Monitor 盯完成（判据用 curl DONE 行，不用大小阈值——记取 VCTK 教训）

### 2. 解压（两层）
- 外层：`tar xzf data_aishell.tgz -C data/AISHELL/`
- 内层：遍历 `wav/test/*.tar.gz` 全部解压（**只解 test 集**即可，~7176 条，省时省盘；train/dev 不需要）

### 3. 采样 + transcript 映射
- 用 test 集，跨说话人采 300 条（seed=42，round-robin，复用 VCTK 采样器逻辑改路径）
- 解析 transcript 建 {utt_id: 中文文本} 映射，给 CER 用

### 4. 客观评估（GPU，复用 exp7 口径，~5min）
- `evaluate_clean_large_nosave.py` 跑 Base vs LCA clean L=1/2/3，1800 行
- 与 VCTK/LibriSpeech 同表对比

### 5. 中文 ASR/CER 评估（GPU，需改造脚本）
- 新脚本 `evaluate_asr_cer_zh.py`（基于 evaluate_asr_wer_onthefly.py 改）：
  - whisper 模型换 multilingual（`small` 或 `medium`，非 base.en；中文需 multilingual）
  - `transcribe(language="zh")`
  - CER：jiwer char-transform（exp10 已有），中文按字符，不做分词
  - 对 Base/LCA 解码音频 + 原始音频，算 CER vs ground-truth transcript
- 子集：300 条（与客观同 list）× 3L × 2model + 原始 = ASR 调用量适中
- whisper 模型大小待定：medium 更准但慢，small 快；中文建议 medium（先确认下载/显存）

### 6. 完整性 + 写记录
- CSV 行数校验、NaN/Inf、独立复算
- 写 `实验记录.md` §15.7（AISHELL 客观）+ §15.8（中文 CER），与 VCTK 并列
- 飞书里程碑播报（复用 VCTK 那套监控，改路径）

## 诚实边界（写进记录）
- AISHELL 16kHz 原生，无重采样损失（优于 VCTK 对比条件）
- 300 子集（test 集全量 7176），跨说话人采样
- whisper multilingual 中文 ASR 自身有错误率地板，CER 差异读作"相对退化"
- LCA 为英文 LibriSpeech 训练，中文 zero-shot 是真正的跨语言考验，结果可能比 VCTK 差，诚实记录
- 中文 CER vs 英文 WER 不可直接比（字符 vs 词），仅各自语言内 Base/LCA/原始对比

## 风险/不做
- 不重训、不微调（纯 zero-shot）
- 不碰主代码、不碰论文正文
- 嵌套 tar 解压失败有兜底（先验证外层再内层）
- whisper medium 若显存/下载有问题，降级 small

## 预估
下载 ~30-60min（15GB）+ 解压 ~10min + 客观评估 ~5min + 中文 ASR ~20-40min（whisper medium 慢）。总计 1-2h（主要是下载）。
