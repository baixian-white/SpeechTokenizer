# exp21：ViSQOL 感知质量补测（test-clean/other_300）

**日期**：2026-06-18  **run_id**：`exp21_visqol_perceptual_test300_20260618_seed42`

## 目的
补论文 §7-2 长期缺口"ViSQOL 仍缺"。ViSQOL（Virtual Speech Quality Objective Listener，Google）是逼近主观 MOS 的客观感知指标，对神经/生成式 codec 比 PESQ-WB 更鲁棒（PESQ 设计时只见过波形 codec，对神经 codec 系统性偏低）。EnCodec/DAC/SoundStream 等论文普遍报 ViSQOL，补此列使同台对照更可信。

## 工具链（Windows 无官方 wheel / 无 PyPI 包）
- 在 WSL2 Ubuntu-22.04 用 bazel 源码构建 **google/visqol C++ CLI**（conformance v333）。只建 C++ `:visqol`（不建 Python 绑定，系统 python 是 3.14 已无 distutils）。
- 构建踩坑：(1) Git-Bash 把 `/mnt/h` 参数改写成 `D:/Git/mnt/h` → 改为脚本拷到 WSL home 再跑；(2) `python_configure` 需 distutils → 指 `PYTHON_BIN_PATH=/usr/bin/python3.10`；(3) 还需 numpy → 为 python3.10 bootstrap pip + 装 numpy 1.26.4（均 user 级，无 sudo）。
- 二进制：`~/visqol/bazel-bin/visqol`（WSL home，未纳入仓库）。

## 数据与协议
- **复用 exp12/exp20 已解码 wav**（不重新生成）：23 方法 ×2 集，共 **22,800 对**（reference=exp12 original，degraded=各方法解码）。全部 16 kHz mono → ViSQOL 语音模式原生，无重采样。
- 7 shard 并行跑（28 核 WSL），全 22,800 对 0 缺失。
- 聚合：per (split, method, codec_setting) 均值 + 95% bootstrap CI（B=10000，seed=42），与 exp12/exp20 `analyze` 脚本同协议。配对检验 `scipy.stats.wilcoxon`，配对单位同 sample_id。
- **PCM 直通 sanity**：ViSQOL ≈ 4.443 (clean) / 4.342 (other) ≈ 语音模式上界，验证全链路正确。

## 核心结论（ΔViSQOL = LCA − baseline）
1. **对神经 codec 结论一致并强化**：vs DAC，clean L=1/2/3 = +0.33/+0.74/+0.79，other = +0.36/+0.15/+0.05（除 other L=3 p=0.16 外均 p<0.001）；vs EnCodec 1.5k = +0.37(clean)/+0.19(other)，p<0.001。
2. **对 Codec2 的"碾压"叙事被 ViSQOL 修正（诚实标注）**：Codec2 作为专用低码率语音 codec，感知 MOS 代理上差距远小于 mel-L1/STOI，且 **2 个点反超 LCA**：clean L=1 vs 700C Δ=-0.073(p=0.028)、other L=2 vs 1200 Δ=-0.227(p<0.001)；clean L=2 vs 1200 持平(p=0.44)；其余 LCA 仍占优（L=3 vs 1300 = +0.48/+0.32）。
3. **跨码率 Opus 6k**：split 相关——clean LCA L=3 反超 (+0.147)，other Opus 更优 (-0.362)，均 p<0.001。
4. **LCA vs Base**：除 clean L=1（-0.108，L=1-clean 例外同源）外，L=2/L=3 两集均 LCA 显著更高（+0.17~0.26）。

## 论文回写（已完成，写入 `scit_speech_method_cn_draft_20260609.md`）
- §4.4 评价指标：加 ViSQOL 条目。
- §5.1 表 1a/1b：加 `ViSQOL ↑` 列（关键工作点）。
- §5.1 正文：新增「ViSQOL 感知质量」段，含上述两层结论 + Codec2 诚实修正。
- §7-2 局限性："ViSQOL 仍缺" → "已补齐"，四类客观度量已覆盖，仅主观 MOS/AB 缺。
- 附录 D 表 D.1（神经）/D.3（Codec2/AMR-WB）：每张 per-method 表加 ViSQOL 列。
- 附录 D.4：ViSQOL 同码率配对检验表（22 对，含诚实标注劣于对照的 3 点）。

## 产物
- `metrics/visqol_per_sample.csv`（22800 行，含 method/setting/sample_id/visqol）
- `metrics/visqol_per_method_summary.csv`（76 组，mean + 95% CI）
- `metrics/visqol_paired_tests.csv`（22 对配对检验）
- `metrics/visqol_key.csv`（pairing 映射）
- `commands/*.py *.sh`（构建/批处理/聚合/配对/回写脚本）
- `logs/`（7 shard 原始 batch/results CSV + 构建日志）

## 诚实边界
- ViSQOL 是主观 MOS 的**算法代理**，非真实听测；主观 AB/MOS（exp18 脚手架）仍缺。
- ViSQOL 二进制在 WSL home，未纳入仓库；复现需按 `commands/build_visqol.sh` 重建（已记录 3 个踩坑修复）。
- 统计口径：train-clean/other 同源评估集，与 §5.1 其他指标一致。
