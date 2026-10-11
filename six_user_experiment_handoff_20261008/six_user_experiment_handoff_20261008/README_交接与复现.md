# 六用户低码率语音通信与 Exp23 识别：交接包

## 交接目的

本包用于让另一台 Windows 机器上的同学及其 Agent 分析、检查并复现六用户单机 Router 实验。运行主体是 `project/`，来源于 GPU 版本；同一份代码支持两个后端：

- `local_cpu`：六客户端各自走 CPU 编解码/识别路径，用于 CPU 基线；
- `gpu_worker`：六个轻量客户端 + Router + 一个共享 GPU Worker，LCA 编码/解码使用一份 CUDA 模型，Exp23 分类器当前使用 CPU 回退。

固定用户映射为：A→`p225`、B→`p226`、C→`p227`、D→`p228`、E→`p229`、F→`p230`。实验为 VCTK 已知说话人闭集子集，不是六名新真人身份注册实验。

## 目录说明

```text
README_交接与复现.md              本文件，给 Agent 的入口
Run-SystemExperiment.ps1          顶层启动器
project/                          代码、LCA 权重、Exp23 bundle、六人 VCTK
  3用户demo/.../三用户中心路由通信demo/  Router、客户端、指标与 GPU Worker
  data/VCTK/...                    六个说话人的固定测试音频
  tmp/exp23_formal/...             split manifest 和 seed42 分类器 bundle
  3用户demo/Log/spt_base/          LCA 配置与 SpeechTokenizer_best_dev.pt
results/cpu/                       CPU 精选汇总结果
results/gpu/                       GPU 精选汇总结果
reports/                           CPU/GPU 科研实验报告
reports/识别优先Demo交付参数_20261009.md  识别率、实时性和待补测参数清单
environment/                       环境与依赖说明
FILE_SHA256.csv                    运行前的材料哈希清单
six_user_demo_metrics_requirements_20260924.md  M1–M8 指标要求
```

`results/` 是从原始运行目录提取的可复算核心记录，包含 manifest、summary、repeat summary、逐流/逐说话人指标、混淆矩阵、预测、质量、资源、流量和失败清单。原始大体积 `events.csv`、`blocks.csv`、重建音频和全部临时日志保留在发送方电脑，原始路径写在每个结果目录的 `SOURCE_ORIGINAL_PATH.txt`；交接包不依赖这些大文件才能重新运行。

## 解压后的完整性验证

解压后先在包根目录运行：

```powershell
& .\VERIFY_PACKAGE.ps1
if ($LASTEXITCODE -ne 0) { throw 'Handoff hash verification failed' }
```

看到 `HANDOFF SHA-256: PASS` 后，再执行下面的 CPU/GPU `Checks`。

## Agent 首先要做的检查

1. 先阅读本文件、`README_GPU.md`、`six_user_demo_metrics_requirements_20260924.md` 和 `reports/` 两份报告。
2. 检查 `project/` 下模型、bundle、VCTK、split manifest 是否存在。
3. 不要把 `results/` 中的历史 `run_manifest.json` 当作当前机器配置；其中的绝对路径是历史记录。
4. 运行哈希与代码检查；若 Python 路径不同，显式传入 `-Python` 和 `-QualityPython`。

## 直接可复制的 PowerShell 命令

进入解压后的交接包根目录：

```powershell
Set-Location 'D:\six_user_experiment_handoff_20261008'
$py = (Get-Command python).Source
```

若使用已有 `st_audio_py310`：

```powershell
$py = 'E:\Miniconda\envs\st_audio_py310\python.exe'
```

### 识别优先的交互 Demo

如果目标是展示六个已知用户的识别率，而不是测六路并发吞吐，直接使用
`group_client.py` 的交互入口。该入口默认采用 1.5 秒编码分块和
`p225,p226,p227,p228,p229,p230` 六人候选集；候选集只在决策时屏蔽其他
类别，不会修改 110 类分类器权重：

```powershell
python .\project\3用户demo\speechtokenizer_now\speechtokenizer\三用户中心路由通信demo\group_client.py `
  --user_id B --room_id demo --speaker_classifier_bundle <bundle_dir> `
  --speaker_id_enable --no_play
```

如需复现原始开放集流式基线，显式传入 `--speaker_classifier_candidates all
--chunk_seconds 0.5`。该识别优先默认只作用于交互入口；冻结的 Stage 1
回放和 `run_metrics_experiment.py` 仍保留论文所用的 0.5 秒通信块与 110
类输出空间，不能把两种条件混为一个结果。交互入口默认会尝试播放恢复语音；
若使用 1.5 秒分块并需要可听播放，建议另行设置 `--max_lat 1.75`，否则默认
`max_lat=0.80` 小于单个解码块，可能触发抖动队列裁剪。

### 1. CPU 代码与材料检查

```powershell
& .\Run-SystemExperiment.ps1 -Mode Checks -Python $py -QualityPython $py
if ($LASTEXITCODE -ne 0) { throw 'Checks failed' }
```

### 2. CPU 冒烟（六用户同时发送，一条验证语音）

```powershell
& .\Run-SystemExperiment.ps1 -Mode Smoke -Conditions D -Scenarios concurrent -ComputeBackend local_cpu -MemoryPolicy balanced -Python $py -QualityPython $py
```

### 3. CPU 正式轮流发送

```powershell
& .\Run-SystemExperiment.ps1 -Mode Formal -Conditions D -Scenarios sequential -Repeats 1 -ComputeBackend local_cpu -MemoryPolicy balanced -Python $py -QualityPython $py
```

### 4. GPU 预检与冒烟

```powershell
& .\Run-SystemExperiment.ps1 -Mode Preflight -Conditions D -Scenarios concurrent -ComputeBackend gpu_worker -MemoryPolicy balanced -Python $py -QualityPython $py
& .\Run-SystemExperiment.ps1 -Mode Smoke -Conditions D -Scenarios concurrent -ComputeBackend gpu_worker -MemoryPolicy balanced -Python $py -QualityPython $py
```

预检必须看到 `cuda_available=true`、`gpu_ready=true`、`measurement_ready=true`。没有 CUDA 时不要选择 `gpu_worker`，改用 `local_cpu`。

### 5. GPU 正式实验

```powershell
& .\Run-SystemExperiment.ps1 -Mode Formal -Conditions D -Scenarios sequential -Repeats 1 -ComputeBackend gpu_worker -MemoryPolicy balanced -Python $py -QualityPython $py
& .\Run-SystemExperiment.ps1 -Mode Formal -Conditions D -Scenarios concurrent -Repeats 1 -ComputeBackend gpu_worker -MemoryPolicy balanced -Python $py -QualityPython $py
```

### 6. GPU 30 分钟稳定性

```powershell
& .\Run-SystemExperiment.ps1 -Mode Stability -Dataset formal -Conditions D -DurationSeconds 1800 -Repeats 1 -ComputeBackend gpu_worker -MemoryPolicy balanced -Python $py -QualityPython $py
```

每次运行会在 `project/tmp/system_metrics_YYYYMMDD_HHMMSS/` 生成 `run_manifest.json`、`summary.csv`、逐条 CSV、`resources.csv`、`traffic.csv`、图件和日志。不要只查看终端最后一行；用 `failures.json`、`summary.csv` 和 `analysis_status.json` 判断是否完整。

## 复现时必须报告的内容

依据 M1–M8，至少报告：

- 30 条有向链路的在线用户数、活跃发送路数、发送/接收/解码块、缺失率、期限失败率；
- 每源语句和逐接收端 STOI、PESQ-WB、mel-L1；
- 采集到解码、编码、传输、解码、分类推理的均值/中位数/P95；
- `N_expected`、`N_predicted`、`N_correct`、有效预测 Top-1、覆盖率、端到端正确输出率、Macro-F1、每人准确率和混淆矩阵；
- 首次身份响应、切换响应；没有切换实验时写“未测”，不能填 0；
- CPU/GPU、内存/显存、RSS、Router/接收队列和积压趋势；
- 载荷码率、应用层总流量及发送/接收方向；
- 30 分钟运行时长、断连、重连、崩溃、异常退出和播放欠载。

识别报告必须同时给出：

```text
有效预测 Top-1 = N_correct / N_predicted
预测覆盖率     = N_predicted / N_expected
端到端正确率   = N_correct / N_expected
```

## 当前实验结论摘要

- CPU 和 GPU 轮流发送都能完整处理 2700/2700 个通信块；450 个识别窗口全部产生预测，Top-1 为 85.56%、Macro-F1 为 0.9183、覆盖率为 100%。
- GPU 轮流发送 LCA 编码/解码 P95 约 15.18/18.71 ms，显著低于 CPU；这反映的是编解码 GPU 加速，Exp23 分类器仍走 CPU 回退。
- GPU 同时发送正式实验收到 2700/2700 块，完成解码 1527 块，预测覆盖率 16.44%，端到端正确率 14.44%，GPU 利用率峰值约 89%，显存峰值约 1.15 GiB。
- GPU 30 分钟运行收到 108000/108000 块，完成解码 57194 块，预测覆盖率 2.89%，端到端正确率 2.54%，断连和异常退出为 0，但存在接收队列丢弃和期限失败。
- 当前结果支持“GPU 能显著改善共享 LCA 编解码吞吐”，不支持“六路同时发送已经达到实时满载”，也不应称为“全模型 GPU”。

## 重要限制

1. 这是单机 TCP loopback，不等同于六台机器或真实局域网。
2. 麦克风和播放关闭，报告的边界是采集到解码，不能宣称完整声学通话延迟。
3. 当前未做同一流内说话人切换，M5 切换指标必须标记未测。
4. Exp23 bundle 直接使用 CUDA 会因 TorchScript CPU 常量报错，GPU Worker 当前采用 CPU 分类回退。
5. 110 类分类器仍保留原始 110 类输出；六个用户只是测试子集，不是把模型改成六分类。
6. 运行重复是系统重复，不等于训练随机种子重复；不能把重复回放当成独立语音样本。
