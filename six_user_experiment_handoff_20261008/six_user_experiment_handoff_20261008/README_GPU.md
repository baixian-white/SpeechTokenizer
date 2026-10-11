# 六用户共享 GPU Worker 实验

这是 `six_user_stage1_experiment` 的 GPU 实验副本。CPU 基线保留在原目录；本目录使用一个共享 GPU Worker，六个客户端和 Router 仍保持独立。

## 架构

- `gpu_worker.py`：只加载一份 LCA 模型到 CUDA，处理六个客户端的编码和解码请求，并记录 GPU 显存、GPU 利用率和 Worker 队列。
- `metrics_client.py`：六个轻量客户端，负责 WAV 时间轴、Router TCP、包序号、接收音频和原始结果记录。
- `router_server.py`：继续验证 30 条有向收发关系。
- Exp23：当前默认在 GPU Worker 的 CPU 回退路径运行，因为原始 TorchScript bundle 含有 CPU 常量；`run_manifest.json` 会记录 `classifier_device=cpu_fallback_torchscript_constants`。

系统用一份 GPU LCA 模型承载六路，适配 GTX 1650 4 GiB。它测量的是“共享 GPU Worker 部署条件”，不等于六个客户端各自持有一份 GPU 模型。

GPU Worker 版本的 CPU 预检门槛为六用户至少 3 GiB 可用内存；客户端不再各自加载 LCA 模型。CPU 基线仍使用原来的严格门槛。

## 运行前检查

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_gpu_experiment\Run-SystemExperiment.ps1' `
  -Mode Checks
```

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_gpu_experiment\Run-SystemExperiment.ps1' `
  -Mode Preflight `
  -Conditions D `
  -Scenarios concurrent `
  -ComputeBackend gpu_worker `
  -MemoryPolicy balanced
```

预检必须同时满足：

```text
cuda_available = true
gpu_ready = true
measurement_ready = true
```

## GPU 六用户冒烟

先运行六用户轮流发送，验证 Worker、Router、客户端和指标回写：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_gpu_experiment\Run-SystemExperiment.ps1' `
  -Mode Smoke `
  -Conditions D `
  -Scenarios sequential `
  -ComputeBackend gpu_worker `
  -MemoryPolicy balanced
```

然后运行六用户同时发送：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_gpu_experiment\Run-SystemExperiment.ps1' `
  -Mode Smoke `
  -Conditions D `
  -Scenarios concurrent `
  -ComputeBackend gpu_worker `
  -MemoryPolicy balanced
```

结果目录中应查看：

- `run_manifest.json`：`compute_backend=gpu_worker`、GPU 名称、显存和分类器设备；
- `summary.csv`：覆盖率、端到端正确率、延迟和 GPU/Worker 资源字段；
- `blocks.csv`：逐块 Router、Worker、接收和解码时间；
- `predictions.csv`：Exp23 预测；
- `resources.csv`：CPU、RSS、GPU 利用率、显存和 Worker 队列；
- `events.csv`：客户端、Router 和 GPU Worker 原始事件；
- `failures.json`：超时、OOM、Worker 错误和异常退出。

## GPU 正式实验

轮流发送正式测试：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_gpu_experiment\Run-SystemExperiment.ps1' `
  -Mode Formal `
  -Conditions D `
  -Scenarios sequential `
  -Repeats 1 `
  -ComputeBackend gpu_worker `
  -MemoryPolicy balanced
```

并发正式测试：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_gpu_experiment\Run-SystemExperiment.ps1' `
  -Mode Formal `
  -Conditions D `
  -Scenarios concurrent `
  -Repeats 1 `
  -ComputeBackend gpu_worker `
  -MemoryPolicy balanced
```

30 分钟稳定性测试：

```powershell
& 'B:\工作\科研项目\Speech_tokenizer\six_user_gpu_experiment\Run-SystemExperiment.ps1' `
  -Mode Stability `
  -Dataset formal `
  -Conditions D `
  -DurationSeconds 1800 `
  -Repeats 1 `
  -ComputeBackend gpu_worker `
  -MemoryPolicy balanced
```

GPU 实验主指标与原 requirements 文件一致：M1 通信完整性、M2 STOI/PESQ-WB/mel-L1、M3 分阶段延迟、M4 Top-1/Macro-F1/覆盖率、M5 识别响应、M6 GPU/CPU/队列资源、M7 Router/客户端应用层流量、M8 稳定性。`capture_to_decode_ms` 是采集到解码边界；播放关闭时不报告声学输出延迟。

## 当前限制

当前环境已经是 `torch==2.11.0+cu128`，LCA GPU 编解码通过。Exp23 bundle 直接切到 CUDA 会触发 TorchScript CPU 常量与 CUDA 张量混用，因此保留 CPU 分类器回退并在运行清单中记录。要做到 GPU 分类，需要重新导出设备安全的 Exp23 bundle；完成前不要把本实验称作“全模型 GPU”。

