# 六用户系统实验：运行与复算说明

本入口按照 `six_user_demo_metrics_requirements_20260924.md` 组织后续实验，记录通信完整性、语音质量、延迟、身份识别与响应速度、资源、应用层通信量和稳定性。结果写入独立的 `project/tmp/system_metrics_时间戳/` 目录。

## 本机环境

- 通信、LCA 和 Exp23：`E:\Miniconda\envs\st_audio_py310\python.exe`，当前安装为 CUDA 版 `torch==2.11.0+cu128` / `torchaudio==2.11.0+cu128`，但现有系统测量客户端默认仍按 CPU 基线运行。
- STOI：上述环境中的 `pystoi==0.4.1`。
- PESQ-WB：通过 `st_audio_py310` 中的 `pesq==0.0.4` 评分。评分在运行结束后执行，不加载通信客户端模型。
- 图件：matplotlib，输出 PNG、PDF，并保留来源 CSV。
- 可用物理内存：A 本地对照至少 2 GiB，B 三用户至少 4 GiB，C/D 六用户严格策略至少 8 GiB。
- 内存策略默认是 `strict`。如果本机确认可用内存不足 8 GiB，可以使用 `-MemoryPolicy balanced`，六用户预检门槛为 6 GiB；运行清单会记录警告，实验期间必须检查交换、队列积压、超时和退出码。该策略适合开发验证，论文主结果建议使用 `strict`。

以下命令均可直接粘贴到新的 PowerShell 终端，不需要预先定义变量，也不需要切换当前目录。`Run-SystemExperiment.ps1` 会定位本包 `project` 并使用指定解释器；提示符显示 `(base)` 时也会调用上述环境。

## 实验安排

| 条件 | 运行方式 | 识别 |
| --- | --- | --- |
| A | 六个发送源依次进行单进程、本地分块编解码 | Exp23 开启 |
| B | ABC、DEF 两组三用户分别运行 | Exp23 开启 |
| C | 六个客户端与一个 Router | 关闭 |
| D | 六个客户端与一个 Router | Exp23 开启 |

| 场景 | 实际发送方式 |
| --- | --- |
| sequential | 每个网络任务只有一端发送，所有接收端准备后开始 |
| concurrent | 组内所有客户端在共同时间屏障后开始发送，并分别接收其余用户的流 |
| switch | 同一个流 ID 内，真实身份每六秒轮换一次，形成连续识别窗口 |
| stability | 按冻结清单连续循环发送，默认至少 1,800 秒 |

A 是不经过网络的单进程基线，按六个源分别测量；报告中不把它标成六进程并发。B 的每个接收端处理两路，C/D 并发时每端处理五路。

开发运行用每人一条锁定验证语音；正式运行用每人十五条锁定测试语音。正式 A/B/C/D 默认独立重启重复三次。身份模型始终保留 seed42 的 110 类输出，温度与分类阈值按交付配置加载。

## 检查与开发冒烟

只执行单元测试、关键脚本编译和本包 SHA-256 校验：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode Checks
```

检查依赖、音频、模型与内存，不启动客户端：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode Preflight
```

先验证六用户同时发送的 D 组。此命令会自动执行代码测试与文件校验，再进行模型和资源预检；检查失败会停止，检查通过后开始实验：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode Smoke -Conditions D -Scenarios concurrent
```

在可用内存达到约 6 GiB、但尚未达到严格策略门槛时，可以使用平衡策略进行开发冒烟：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode Smoke -Conditions D -Scenarios concurrent -MemoryPolicy balanced
```

平衡策略冒烟通过后，可以先用一次重复运行正式 D 组同时发送，确认内存、队列和退出状态：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode Formal -Conditions D -Scenarios concurrent -Repeats 1 -MemoryPolicy balanced
```

平衡策略会把门槛降到 6 GiB，但六个客户端仍然各自加载编解码模型。若出现交换使用、队列持续增长、超时或非零退出，应停止该运行并释放内存；不要把这类运行与严格资源条件下的论文主结果混写。

然后运行所有对照的轮流/同时发送开发场景：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode Smoke -Conditions A,B,C,D -Scenarios sequential,concurrent
```

## 冻结与正式实验

本次已生成以下三份计划，当前状态均为 `prepared`。它们尚未产生正式测量结果：

| project/tmp/ 下的目录 | 配置 |
| --- | --- |
| system_suite_formal_20260928 | A/B/C/D，sequential + concurrent，十五条/人，三次系统重复 |
| system_switch_formal_20260928 | D 组，同流身份切换，三次系统重复 |
| system_stability_formal_20260928 | D 组，连续六路，1,800 秒，一次运行 |

先完成六用户开发冒烟，再启动已准备的正式对照计划：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode RunPlan -PlanPath 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\project\tmp\system_suite_formal_20260928\experiment_plan.json'
```

在正式运行前生成配置和音频清单，默认每人十五条、三次重复。`Prepare` 只准备文件，可以在内存尚未释放时执行：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode Prepare -Dataset formal -Conditions A,B,C,D -Scenarios sequential,concurrent -OutputDirectory 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\project\tmp\system_protocol_v1_formal'
```

审阅该目录的 `experiment_plan.json`、`audio_manifest.csv` 和 `run_manifest.json` 后，执行冻结计划。准备目录应当是本次新建的目录；已有任务目录的运行不会覆盖旧数据：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode RunPlan -PlanPath 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\project\tmp\system_protocol_v1_formal\experiment_plan.json'
```

也可一次完成准备和正式运行：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode Formal -Conditions A,B,C,D -Scenarios sequential,concurrent
```

单次 D 组正式同时发送时，预期有 30 条有向链路、450 个三秒窗口和 2,700 个接收块位置。三次重复是三次系统测量；源语句仍然是原来的 90 条。

## 同流身份切换与稳定性

先用开发语音验证同一流内切换：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode Switch -Dataset smoke -Conditions D -Repeats 1
```

在固定协议下运行正式切换测量：

切换场景固定取每个身份所选划分中的第一条语音，每条连续回放两次构成六秒驻留，共六个身份；它的独立语句数为六条，主要评价响应与过渡行为。十五条/人的识别性能由前述正式 sequential/concurrent 场景评价。

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode Switch -Dataset formal -Conditions D -Repeats 3
```

六用户同时发送、30 分钟稳定性测量：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode Stability -Dataset formal -Conditions D -DurationSeconds 1800 -Repeats 1
```

稳定性测量会保留音频与逐块记录，预计使用数 GiB 磁盘；先检查预检报告中的磁盘空间。重复回放用于稳定性与响应测量，不增加独立测试语句数量。

## 计时与指标口径

- 使用单机 `time.perf_counter_ns`，单位纳秒。源时间轴由启动屏障确定；WAV 回放等待到每块尾样本时才启动编码。
- 默认块等待期限为源块尾后十秒，预测等待期限为窗口结束后十五秒；这是固定观测期限，不是性能达标阈值。需要修改期限时，使用 Python 入口的 `--block-deadline-seconds`、`--prediction-deadline-seconds`，并在正式测量前冻结。
- `capture_to_decode_ms` 从源块首样本的计划释放时间测到解码完成，包含分块等待；同时提供块尾到解码、编码、传输、接收队列、解码分项。
- 默认关闭播放。播放和设备声学输出延迟为未测量。可加入 `-Playback device` 测量到音频输出回调，记录缓冲丢弃和欠载；该回调也不等于声音实际离开设备。
- STOI、PESQ-WB、mel-L1 对每路混音前重建与对应三秒源音频评分。缺块位置保留补零，不删除时间空隙，不做相关性搜索对齐。
- 预先声明三秒窗口。独立三秒语句每条一次预测；连续场景三秒上下文，每新增一秒源音频产生一个应预测窗口。
- 跨身份窗口保存为 `transition=True`，按窗口末样本身份记录真值；不计入稳态准确率，保留在切换时间线与响应统计中。
- 连续流的首次/切换响应在该身份的整个驻留段内查找；只有在驻留结束前输出的首次按时正确预测计为响应成功。身份离开后才到达的正确输出保留为 `correct_after_identity_left`，并计入响应失败。独立三秒语句允许在预定预测期限内完成首次输出。
- Top-1 分母是有效、按时的预测；覆盖率分母是预定窗口；端到端正确输出率分母同样是预定窗口。无输出、超时和推理失败均保留。Macro-F1 使用六个真实身份，原始预测仍保存 110 类标签。
- 载荷与应用层字节分别计数，上下行和 Router 转发分别报告。每路码率使用源音频时长作为分母。完整 TCP/IP 或链路层流量没有抓包测量，相关值保持缺失。
- 同源的多个接收副本、重叠窗口和重复回放有相关性。质量先在同源副本间汇总，再逐语句等权平均；跨运行表报告系统重复的均值和样本标准差。

## 文件与复算

| 文件 | 用途 |
| --- | --- |
| experiment_plan.json、run_manifest.json | 冻结配置、模型/代码哈希、解释器、计时规则、运行范围 |
| audio_manifest.csv | 独立身份真值、原始/准备音频及哈希 |
| predictions.csv | 应预测窗口、原始预测标签、状态、置信度、推理耗时和输出时间 |
| blocks.csv | 每个预定源块与接收端位置、事件时间、字节数、处理状态 |
| audio_metrics.csv | 每个重建副本的质量指标、缺失与失败原因 |
| resources.csv | 从加载到退出的 CPU、内存、队列及阶段标签；CPU 运行的显存项注明未使用 |
| per_stream_metrics.csv、summary.csv | 逐链路与逐次运行汇总 |
| repeat_summary.csv、per_speaker_metrics.csv | 系统重复统计与每人结果 |
| confusion_matrix.csv、identity_responses.csv | 六行八列混淆数据、首次/切换正确响应与失败 |
| traffic.csv、events.csv | 分方向应用层流量及全部事件 |
| jobs/ | 每个任务的配置、退出记录、原始 JSONL、日志和接收端混音前音频 |
| figures/ | 由 CSV 生成的收发覆盖、混淆、延迟分布、资源及切换时间线图 |
| failures.json、analysis_status.json | 运行失败和指标生成状态 |

将路径替换为实际输出目录后复算：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode Summarize -RunDirectory 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\project\tmp\system_protocol_v1_formal'
```

每个数字可以通过 `job_id → 源语句/流/序号/窗口 → 原始 JSONL` 追溯。缺失值留空并保留状态，不把未测量项写成零。`failures.json=[]` 表示没有记录的进程运行失败；还需要检查 `summary.csv` 中的覆盖率、截止期失败和质量计算失败数量。

## 额外代码验证

六客户端传输与退出测试使用明确标记的合成编解码器，不测模型性能：

```powershell
& 'E:\Miniconda\envs\st_audio_py310\python.exe' 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\project\tests\run_metrics_transport_check.py'
```

真实模型的单流、六身份切换开发检查：

```powershell
& 'E:\Miniconda\envs\st_audio_py310\python.exe' 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\project\tests\run_metrics_switch_check.py'
```

本实现提供单机回环测量。跨机器部署、可靠时钟同步、声学设备端到端测量，以及身份认证/未知身份拒识，需要相应的独立实验设置。`verified` 是闭集置信度展示标志。
