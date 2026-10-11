# 六用户 Demo 完整材料包

打包日期：2026-09-20。

这是实际源码、模型、数据和实验产物的 ZIP64 交接包，不只是路径清单。按用户要求，ZIP 不设密码、不包含旧演示 key 文件，复制后即可正常解压。

## 解压后目录

```text
six_user_demo_handoff_20260920/
  START_HERE_zh.md
  PACKAGE_CONTENTS.json
  PACKAGE_SHA256SUMS.csv
  package_tools.py
  handoff/                  12 项交接清单、原始核查记录、环境和启动日志
  project/                  Exp23 分支源码及运行所需材料
    3用户demo/               客户端、router、LCA 模型和配置，不含旧 key
    config/
    speechtokenizer/
    nas/
    scripts/
    tests/
    docs/
    data/VCTK/              完整本地音频目录和转写目录
    output/experiments/     选定 Base/LCA checkpoint、配置及历史 demo 记录
    output/models/          SpeechBrain ECAPA 本地权重
    tmp/exp23_formal/       canonical 缓存、seed41/42/43、汇总、seed42 bundle
  reference_legacy_source/  旧分支源码快照，仅用于对照和追溯
  reference_hf_snapshot/    ECAPA 原始 snapshot 文件与来源说明
```

推荐先解压到较短路径，例如 `D:\scit-demo`，以减少 Windows 深层目录路径问题。请使用支持 ZIP64 的解压工具。

## 对应 12 项材料

1. 三用户实际 LCA v2 权重：`project/3用户demo/Log/spt_base/SpeechTokenizer_best_dev.pt`。
2. LCA config、finetune config 和 NAS config：demo 模型目录及 `project/output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/configs/`。
3. Base：`project/output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/`。
4. VCTK 音频：`project/data/VCTK/wav48_silence_trimmed/`，110 个说话人目录。
5. VCTK 转写：`project/data/VCTK/txt/`，109 个说话人目录。
6. canonical manifest：`project/tmp/exp23_formal/cache_lca_l3/cache_manifest/split_manifest.json`；新生成的 109 类 manifest：`handoff/text_controlled_109_split.generated.json`。
7. 已有 canonical LCA L3 缓存：`project/tmp/exp23_formal/cache/`，附 `cache_lca_l3/cache_manifest/`。
8. ECAPA：`project/output/models/speechbrain_spkrec_ecapa_voxceleb/`；版本记录见 `handoff/environment_current.json`。
9. 三个 seed 的 checkpoint、validation/test 指标和温度：`project/tmp/exp23_formal/seed41/`、`seed42/`、`seed43/`。
10. 最终 bundle：`project/tmp/exp23_formal/export/seed42_bundle/`，使用固定的 110 类标签。
11. 按用户要求不交付旧 `46.txt` key；环境和已执行命令见 `handoff/verification/`。历史记录可能保留它的原路径和哈希，但没有 key 明文。
12. 包内所有 payload 文件，包括 VCTK 音频和转写，逐文件 SHA-256 见根目录 `PACKAGE_SHA256SUMS.csv`；来源与 Git/run ID 见 `PACKAGE_CONTENTS.json` 和 `handoff/materials_audit.json`。

原始交接文档 `handoff/README_zh.md` 保留的是打包前本机路径和核查状态。其中“没有复制数据”等文字描述的是前一轮核查，不是本完整 ZIP 的内容。以本页和包内清单为准：此 ZIP 包含本地 VCTK 数据，但按最新要求排除了旧 key。

这里排除的 key 是通信程序的演示性 XOR key，不是 ZIP 密码。解压、读取模型和实验结果不需要它；原始通信代码仍有 `--key_path` 依赖，运行该路径前需由开发者另行处理测试 key 或加解密逻辑。本次打包没有擅自移除通信代码中的加解密步骤。

## 完整性检查

在解压后的包根目录执行：

```powershell
python package_tools.py --verify
```

该命令逐文件校验 `PACKAGE_SHA256SUMS.csv`；不会修改模型或数据。生成后的迁移文件不在原包哈希清单内，原始文件仍可继续校验。

## 迁移绝对路径

原始 manifest 按原样保留，包含打包机器的 H:/D: 路径。执行：

```powershell
python package_tools.py --prepare-paths
```

该工具验证目标文件存在后，把三个 manifest 的路径改为解压位置，另存到：

```text
project/tmp/exp23_portable/cache_records.json
project/tmp/exp23_portable/canonical_110_split.json
project/tmp/exp23_portable/text_controlled_109_split.json
```

不改写历史 manifest、说话人标签、划分、温度、权重或测试结果。训练/评估脚本需要 `--cache-manifest` 时，指向这个迁移后的 `cache_records.json`。输出目录非空时工具会拒绝覆盖。

后续运行 Python 脚本前应将工作目录切到 `project/`，并核对所选配置内部的相对路径。历史启动日志中的用户名、解释器及绝对路径仅用于追溯，需要替换为接收机实际位置。

## 环境与代码状态

主交付源码来自 `feature/exp23-speaker-classifier`，当前 commit：

```text
8ee038bdb7e3588b70e88e785f5a05edb78a679c
```

历史参考源码来自 `chore/repo-reorg-research`，当前 commit：

```text
077df2a1e19f8898e6f7c278662c43e1f8347af0
```

本机核查环境为 Python 3.10.19、torch/torchaudio 2.9.1+cu128、SpeechBrain 1.1.0。完整包版本清单在 `handoff/environment_current.json`。没有把 Conda 环境或操作系统依赖直接复制进 ZIP；旧 demo 的 `environment.yml` 与核查环境不一致，不应直接视为正确锁文件。

本包保留了旧代码的真实状态，没有在打包时悄悄改模型或修复实验逻辑。已知问题：

- 旧三用户路径能收发、解码、识别，但本轮定时退出检查出现异常退出。
- Exp23 bundle 可独立加载推理；直接启动原客户端有内嵌旧 `speechtokenizer` 包遮蔽根包的导入错误。
- text-controlled 缓存 CLI 对当前划分函数的调用签名不匹配。
- 109 类 manifest 是交接时新生成的；109 类缓存和正式测试、Base 全条件缓存不包含在本包，因为它们尚未完成。
- Base/LCA 历史训练 commit 未记录；Exp23 历史训练环境标记 `dirty=true`，完整历史源码状态不能只凭 commit 还原。

详细错误、复现命令与边界见 `handoff/README_zh.md` 和 `handoff/verification/verification_summary.json`。本次 ZIP 完整性通过不代表六用户系统已经验收。

## 六用户的身份范围

若演示的是 VCTK 中六个已训练身份的录音回放，可以基于现有 bundle 做集成。

若演示的是六位新同学的真人麦克风，需要另外注册匹配或训练适配；现有 Exp23 分类器不能通过修改 `user_id` 自动认识新用户。

## 未纳入范围

未纳入 `.git/`、其他数据集、无关实验、历史大归档、Python 字节码、训练环境二进制和论文文件。本次任务是材料交接，不重训模型、不重跑已锁定正式测试、不上传到外部服务。
