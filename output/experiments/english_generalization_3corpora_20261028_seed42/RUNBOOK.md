# 远端英文泛化评测运行说明

本实验在 `lab-suer`（suer 用户）的 GPU 1 上执行。远端目录为 `/home/suer/english_generalization_20261028`，代码库为 `/home/suer/workspace/NMI/speechtokenizer`。本地目录保存协议、脚本、清单和回收的结果。

本轮使用冻结模型评测，不更新模型权重。具体数据和评测口径见 PROTOCOL.md。

## 查看进度

```powershell
ssh -T -o BatchMode=yes lab-suer 'cat /home/suer/english_generalization_20261028/pipeline_status.json'
ssh -T -o BatchMode=yes lab-suer 'cat /home/suer/english_generalization_20261028/full/objective_status.json'
ssh -T -o BatchMode=yes lab-suer 'cat /home/suer/english_generalization_20261028/full/asr_status.json'
```

完整流水线依次进行：10 条跨语料音质试跑、同 10 条 ASR 试跑、全量 4,602 条音质评测、全量 ASR、分语料汇总。音质为 Base/LCA × L1/L2/L3，共 27,612 组；ASR 为 4,582 条有参考转写的原始音频及两种 L3 重建，共 13,746 次识别。

## 恢复与重新汇总

若进程已退出，在远端使用同一环境执行：

```bash
PYTHONNOUSERSITE=1 /home/suer/sixuser_rerun_20261012/venv/bin/python -u \
  /home/suer/english_generalization_20261028/commands/pipeline.py \
  --run /home/suer/english_generalization_20261028 \
  --project /home/suer/workspace/NMI/speechtokenizer
```

流水线自动指定 GPU 1 和独立依赖目录。文件锁阻止重复写入；已完成样本不会重复计入。修改清单、检查点、评测代码或记录中的版本后，旧输出不能直接续跑，需要新建实验目录。

`full/objective.jsonl` 和 `full/asr.jsonl` 为逐条原始结果；`full/cache/` 为无 PCM 裁剪的 float32 L3 重建。`full/reports/summary.json`、CSV 和 REPORT.md 是可重新计算的聚合结果。`smoke/` 和 `preflight_local/` 只用于工程验证，不计入正式统计。

## 复现证据

- `manifests/all_samples.jsonl`：固定样本、转写、分组、原始音频 SHA256。
- `manifests/exclusions.json`：按 TED 官方评分标记排除的 314 条。
- `manifests/downloads.json`：数据及 ASR 模型的固定 revision 和文件哈希。
- `provenance/source_lfs_sha256.json`：与上游 LFS 元数据匹配的三个大文件哈希。
- `provenance/training_manifest_audit.json`：训练／验证清单与测试样本 ID、外部语料路径的交叉检查。
- `provenance/source/`、`source_hashes.json`、`pip_freeze.txt`：模型源码快照和运行环境。
- `full/*_identity.json`：各阶段的检查点、配置、NAS、评测脚本、数据、ASR 和版本身份。

## 已执行的工程核查

- Base/LCA 两个检查点严格加载，NAS 操作配置沿用原实验。
- 包含五个语料分组和各组最长录音的音质／ASR 试跑通过。
- WER 按词数加权的例子、配对差异、单簇样本和空数据处理通过核查。
- 中断尾行修复、缺失换行修复和重复结果拒绝通过核查。
- ASR 编辑计数和英文数字统一化通过核查。
- 仅 VCTK p315 的 20 条样本缺参考；不存在其他被文本规范化成空参考的样本。

历史实验和本轮运行环境不同，重建结果不保证逐位相同；本轮跨语料比较使用同一次评测环境。历史结果仅作背景参照，不与本轮样本混合统计。
