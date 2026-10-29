# 英文跨语料泛化实验阅读入口

交接日期：2026-10-29。实验日期：2026-10-28。

本次工作已经完成。三个外部英文语料为 VCTK、TED-LIUM、FLEURS en_us，LibriSpeech test-clean 与 test-other 另作参照。计算已在 lab-suer 的 suer 账户上执行，使用冻结的 Base/LCA 检查点，没有更新模型权重。阅读或核对结论不需要重新训练、下载数据或启动评测。

## 建议阅读顺序

1. [中文评测报告](output/doc/english_generalization_report_20261028/英文跨语料泛化评测报告.md)：面向汇报的正式正文，包含三个外部语料、LibriSpeech 参照、图表、结论和独立补充附录；[Word 版](output/doc/english_generalization_report_20261028/英文跨语料泛化评测报告.docx)内容一致。
2. [实验协议](output/experiments/english_generalization_3corpora_20261028_seed42/PROTOCOL.md)：样本范围、模型身份、处理方式、固定 ASR、统计方法及补充对照的事后性质。
3. [主实验完整汇总](output/experiments/english_generalization_3corpora_20261028_seed42/full/reports/REPORT.md)：各语料的客观指标、WER、区间与失败分母。
4. [WER 汇总](output/experiments/english_generalization_3corpora_20261028_seed42/full/reports/asr_summary.csv)、[WER 配对差异](output/experiments/english_generalization_3corpora_20261028_seed42/full/reports/asr_paired_deltas.csv)、[客观指标汇总](output/experiments/english_generalization_3corpora_20261028_seed42/full/reports/objective_summary.csv)：引用数值时以这些归档表及逐条结果为依据。
5. [运行说明](output/experiments/english_generalization_3corpora_20261028_seed42/RUNBOOK.md)、[实验核验](output/experiments/english_generalization_3corpora_20261028_seed42/checks/final_validation.json)、[报告核验](output/doc/english_generalization_report_20261028/work/validation.json)：定位运行环境、完成状态和重算验证结果。

## 本轮数据与主要结果

三个外部语料共 4,002 条，加两个 LibriSpeech 参照子集各 300 条，共 4,602 条。两模型在 L1/L2/L3 上共完成 27,612 组客观评测；4,582 条有参考文本的录音完成原音、Base L3、LCA L3 共 13,746 次 ASR 识别。

以下是主实验 L3、名义载荷码率 1.5 kbit/s 的语料词数加权 WER，单位为百分比。

| 数据集 | 客观评测条数 | WER 条数 | 原音 | Base | LCA |
|---|---:|---:|---:|---:|---:|
| VCTK | 2,200 | 2,180 | 1.68 | 6.45 | 5.55 |
| TED-LIUM | 1,155 | 1,155 | 4.39 | 9.27 | 9.28 |
| FLEURS en_us | 647 | 647 | 7.30 | 60.59 | 57.17 |
| LibriSpeech test-clean 参照 | 300 | 300 | 4.03 | 7.05 | 6.23 |
| LibriSpeech test-other 参照 | 300 | 300 | 7.35 | 18.91 | 19.46 |

结果支持 VCTK 和 TED-LIUM 上一定的英文跨语料迁移能力。VCTK 上 LCA 相比 Base 的 WER 降低 0.91 个百分点，配对 95% 区间为 [−1.56, −0.28]。TED-LIUM 两者基本持平，差异为 +0.01 个百分点，区间为 [−0.28, +0.39]。FLEURS 的主实验绝对错误率仍高，不能据此声称三个语料都泛化良好。

## 引用结果时的范围

- 用户要求仅评价英文。中文评测属于更早的背景工作，不纳入本轮结果。
- 正式报告正文不展开输入电平的原因分析。FLEURS 增益对照单列补充附录；它是在观察主实验后设计的，不能将其 LCA WER 13.98% 替换主实验的 57.17%。
- L3 的收益不能外推到所有码率。VCTK 的 LCA mel-L1 在 L1/L2 比 Base 更差；L1/L2 本轮没有运行 ASR。
- WER 是固定 Whisper small.en 的自动识别指标，不是人工听测或说话人识别准确率。
- 区间由 2,000 次聚类 bootstrap 得到。VCTK/LibriSpeech 按说话人，TED-LIUM 按演讲，FLEURS 按文本 prompt；仅一对训练种子的检查点，不包含训练种子不确定性，未校正多重比较。
- VCTK p315 的 20 条缺少参考转写，仅参与客观评测。VCTK 的客观指标覆盖 110 个说话人，WER 覆盖 109 个。
- TED-LIUM 依据源数据评分标记排除了 314 条。保留样本中的 ted_00769 在两模型、三个工作点均计算 PESQ 失败，共 6 项；各 TED PESQ 分组分母为 1,154，其他客观指标为 1,155。
- VCTK 扩展样本含此前的 300 条和新增 1,900 条。当前比较全部采用同一轮评测环境，未混入历史分数。
- 已核查现有训练/验证清单，未发现测试样本 ID 重叠或三个外部语料的路径记录；这不等于已完成全部预训练数据污染审计。

## 文件和执行目录

仓库中的实验归档根目录为 `output/experiments/english_generalization_3corpora_20261028_seed42/`。

| 相对归档路径 | 内容 |
|---|---|
| `commands/` | 数据准备、冻结模型评测、流程调度、汇总、绘图与补充对照脚本 |
| `full/objective.jsonl`、`full/asr.jsonl` | 主实验逐条指标、识别文本与编辑计数 |
| `full/reports/` | 主实验统计表、配对差异、置信区间、覆盖率和失败记录 |
| `manifests/` | 固定样本、排除信息、下载来源与版本 |
| `provenance/`、`full/*_identity.json` | 运行时源码快照、环境、检查点/代码/数据哈希与训练清单核查 |
| `figures/` | 结果图，含 PDF 与 PNG |
| `level_control/`、`analysis/` | 独立补充对照及配对分析 |
| `checks/`、`deliverable_sha256.json` | 完成核查及归档校验信息 |

lab-suer 上原有的完整执行目录为 `/home/suer/english_generalization_20261028`，原项目目录为 `/home/suer/workspace/NMI/speechtokenizer`。原始音频、L3 重建缓存、模型权重与运行依赖位于远端既有目录，不在本次 Git 提交中。Git 归档用于阅读、核查及复现依据；直接从新 checkout 启动流水线前仍需要完整运行资产。

本轮归档记录的 Python 为 `/home/suer/sixuser_rerun_20261012/venv/bin/python`，运行时设置 `PYTHONNOUSERSITE=1`。历史执行使用物理 GPU 1；如开展新的实验，应重新检查 GPU 可用情况。

本次没有改写模型或训练主干。`provenance/source/` 是当次运行所用代码的只读快照。新增 `.gitattributes` 规则保留归档字节，避免跨平台换行转换使已有 SHA256 失效。

## 在已有远端仓库中先读取入口

提交所在分支为 `chore/repo-reorg-research`。以下命令获取远端分支并读取入口，不切换当前工作分支：

```bash
cd /home/suer/workspace/NMI/speechtokenizer
git fetch origin
git show origin/chore/repo-reorg-research:ENGLISH_GENERALIZATION_HANDOFF.md
```

后续按入口读取同一分支中的报告和 CSV 即可。检出或合并前先检查远端 AI 所在工作区的分支与未提交改动，保留已有工作。
