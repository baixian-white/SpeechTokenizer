# 环境与依赖

推荐使用 Python 3.10 环境。原始实验使用：

- Python 3.10.20
- torch 2.11.0+cu128
- torchaudio 2.11.0+cu128
- numpy 2.2.6
- soundfile 0.13.1
- sounddevice 0.5.5
- scipy 1.15.3
- psutil 7.2.2
- pystoi 0.4.1
- pesq 0.0.4
- matplotlib 3.10.9

ZIP 不包含 Conda 环境本身。运行机器需要自行准备 Python 环境；GPU 后端还需要可用的 CUDA 版 PyTorch 和 NVIDIA 驱动。若没有 CUDA，使用 `-ComputeBackend local_cpu`。

在 Windows PowerShell 中先确认：

```powershell
$py = (Get-Command python).Source
& $py -c "import sys,torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available())"
```

GPU 实验原始机器为 NVIDIA GeForce GTX 1650 4 GiB。
