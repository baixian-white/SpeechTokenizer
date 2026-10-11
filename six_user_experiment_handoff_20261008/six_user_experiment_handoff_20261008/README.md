# 六用户单机路由通信与 Exp23 说话人识别：阶段一

完整系统测量的运行方式见 [README_系统实验.md](README_系统实验.md)，包含 A/B/C/D 对照、同时发送、同流说话人切换、稳定性测试和逐条原始记录。可在任意新 PowerShell 终端直接执行预检：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\Run-SystemExperiment.ps1' -Mode Preflight
```

本文件夹是可独立运行的阶段一实验副本，包含实验源码、LCA v2 模型、seed42 Exp23 说话人分类器、固定划分清单，以及本实验使用的 96 条 VCTK 语音。实验采用一个 Router 和 A–F 六个客户端进程，在同一台电脑上完成低码率 RVQ 传输、解码和说话人识别。

固定身份映射为：A→`p225`、B→`p226`、C→`p227`、D→`p228`、E→`p229`、F→`p230`。每轮只有一个客户端发送，其他五个客户端接收并识别，六轮共覆盖 30 条有向链路。本阶段使用 `clean` 条件、CPU 推理、16 kHz 单声道 3 秒语音和三层 RVQ。

## 目录和运行环境

项目根目录：

```text
B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment
```

代码运行目录：

```text
B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\project
```

已验证的 Python 解释器：

```text
E:\Miniconda\envs\st_audio_py310\python.exe
```

运行时通过绝对解释器路径调用 `st_audio_py310` 环境。每次从新 PowerShell 终端开始，都应先执行下面的初始化区块，使 `$py` 和 `$demo` 在当前终端中可用。

六客户端实验要求至少 **8 GiB 可用物理内存**。这是启动前硬性检查；模型、依赖、音频和文件哈希也会在预检中自动检查。

## 从新终端完整执行：检查并启动阶段一冒烟

下面的 PowerShell 代码可以整体复制到一个新终端中执行。它会依次完成环境检查、单元测试、Python 语法检查、文件哈希检查、阶段一资源预检，并在全部检查通过后启动六轮验证集冒烟实验。

```powershell
$ErrorActionPreference = 'Stop'

$project = 'B:\工作\科研项目\Speech_tokenizer\six_user_stage1_experiment\project'
$py = 'E:\Miniconda\envs\st_audio_py310\python.exe'
$demo = Join-Path $project '3用户demo\speechtokenizer_now\speechtokenizer\三用户中心路由通信demo'

Set-Location -LiteralPath $project

if (-not (Test-Path -LiteralPath $py -PathType Leaf)) {
    throw "Python interpreter not found: $py"
}
if (-not (Test-Path -LiteralPath $demo -PathType Container)) {
    throw "Stage 1 demo directory not found: $demo"
}

Write-Host '检查 Python 环境...'
& $py --version
if ($LASTEXITCODE -ne 0) { throw 'Python interpreter check failed.' }
& $py -c "import sys, torch; print('executable=' + sys.executable); print('torch=' + torch.__version__); print('cuda=' + str(torch.cuda.is_available()))"
if ($LASTEXITCODE -ne 0) { throw 'PyTorch environment check failed.' }

Write-Host '运行项目单元测试...'
& $py -m unittest tests.test_stage1_six_user tests.test_demo_exp23_speaker_identity tests.test_demo_speaker_identity -v
if ($LASTEXITCODE -ne 0) { throw 'Unit tests failed.' }

Write-Host '检查关键脚本语法...'
& $py -m py_compile `
    "$demo\group_client.py" `
    "$demo\router_server.py" `
    "$demo\stage1_protocol.py" `
    "$demo\run_stage1_six_user.py"
if ($LASTEXITCODE -ne 0) { throw 'Python syntax check failed.' }

Write-Host '校验项目文件哈希...'
& $py -c "import csv,hashlib,pathlib; root=pathlib.Path('..'); rows=csv.DictReader((root/'FILE_SHA256.csv').open(encoding='utf-8')); bad=[r['relative_path'] for r in rows if (not (root/r['relative_path']).is_file()) or (root/r['relative_path']).stat().st_size!=int(r['bytes']) or hashlib.sha256((root/r['relative_path']).read_bytes()).hexdigest()!=r['sha256']]; print('PASS' if not bad else 'MISMATCH: '+', '.join(bad)); raise SystemExit(0 if not bad else 1)"
if ($LASTEXITCODE -ne 0) { throw 'FILE_SHA256.csv verification failed.' }

Write-Host '运行阶段一实验前预检...'
& $py "$demo\run_stage1_six_user.py" --preflight-only
if ($LASTEXITCODE -ne 0) {
    throw 'Stage 1 preflight failed. Read the resource_ready and BLOCKED lines above, resolve the reported condition, and rerun this block.'
}

Write-Host '实验前预检通过，启动六轮验证集冒烟实验...'
& $py "$demo\run_stage1_six_user.py" --mode smoke
if ($LASTEXITCODE -ne 0) { throw 'Stage 1 smoke run failed. Inspect the generated round logs and failures.json.' }

Write-Host '阶段一冒烟实验完成。'
```

预检通过时应看到：

```text
"resource_ready": true
```

并且预检命令退出码为 `0`。冒烟实验会依次让 A–F 各发送 1 条验证集语音，其他五端接收并识别；总计应产生 30 条“发送端→接收端”验证结果。结果目录会打印在最后一行，并位于：

```text
project\tmp\stage1_six_user_时间戳\
```

## 冒烟通过后运行正式统计

完成冒烟实验并检查结果目录后，在同一个终端中执行正式测试集统计：

```powershell
& $py "$demo\run_stage1_six_user.py" --mode formal
if ($LASTEXITCODE -ne 0) { throw 'Formal Stage 1 run failed. Inspect the generated round logs and failures.json.' }
```

正式实验使用每位说话人的 15 条固定测试集语音，六轮共 90 个发送语音，预期产生 450 条“语音×接收端”逐条记录和 30 条链路汇总记录。

如果要在一个命令中依次完成验证集冒烟和正式测试，也可以执行：

```powershell
& $py "$demo\run_stage1_six_user.py" --mode both
if ($LASTEXITCODE -ne 0) { throw 'Combined Stage 1 run failed. Inspect the generated round logs and failures.json.' }
```

## 产物和指标

每次运行会生成独立时间戳目录，包含：

- `preflight.json`：运行时 Python、模型哈希、资源检查和身份映射；
- `run_info.json`：运行模式和实验配置；
- 每轮 Router 日志、六个客户端日志和 `receipt.json`；
- 逐条语音识别结果 CSV；
- 30 条“发送端→接收端”链路汇总 CSV；
- `failures.json`：运行过程中记录的失败信息。

报告同时记录识别准确率、端到端正确率、识别覆盖率、收包数、解码数、缺包数和耗时。`accuracy_when_evaluated` 只统计已经产生预测的语音；`end_to_end_accuracy` 以该发送端全部测试语音为分母，缺包或未产生预测的语音计为端到端未成功。

## 结果检查命令

在运行目录中，可以用下面的命令快速查看最近一次结果目录：

```powershell
Get-ChildItem '.\tmp' -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

查看失败清单：

```powershell
Get-Content '.\tmp\stage1_six_user_时间戳\failures.json'
```

将上面的 `stage1_six_user_时间戳` 替换为实际生成的目录名。

