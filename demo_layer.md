# SpeechTokenizer 推理脚本（支持 RVQ 选层 + GPU 加速 + 全流程计时）

本脚本提供 **SpeechTokenizer 的离线推理功能**，支持：

- 任意输入语音文件（自动重采样）
- RVQ 编码（encode）
- 任意 RVQ 层选择与组合
- RVQ 解码（decode）
- 输出重建音频
- GPU 加速
- 全流程耗时统计：加载、重采样、编码、选层、解码、写盘、总耗时
- 可选从 HuggingFace 自动下载模型

适用于科研复现、压缩率实验、声码器对比、RVQ 层消融实验等场景。

---

# ✨ 功能特性

- 灵活选择任意 RVQ 层（如 `all`、`0,1,2`、`1-4`、`2:5` 等）
- 自动选择推理设备（CUDA > CPU）
- 自动重采样到模型采样率
- 全流程计时输出，便于性能分析
- 自动裁剪输出波形到 [-1, 1]
- 可选 `--download` 一键拉取模型

---

# 📦 环境安装

```bash
pip install torch torchaudio scipy numpy soundfile
pip install speechtokenizer
pip install huggingface_hub   # 如需使用 --download
```

---

# 🚀 使用方法

## 最简启动命令（默认使用全部 RVQ 层）

```bash
python infer_st.py --speech_file input.wav --output_file output.wav
```

---

# 🧠 命令行参数说明

## 1）输入输出

| 参数 | 说明 | 默认 |
|------|------|--------|
| `--speech_file` | 输入音频路径（WAV/任意采样率） | **必填** |
| `--output_file` | 输出音频路径 | `example_output.wav` |

---

## 2）模型路径

| 参数 | 说明 | 默认路径 |
|------|------|---------|
| `--config_path` | model config.json | `model_hub/speechtokenizer_hubert_avg/config.json` |
| `--ckpt_path`   | checkpoint（.pt） | `model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt` |
| `--download`    | 从 HuggingFace 自动下载模型 | 关闭 |

---

## 3）推理设备

| 参数 | 描述 |
|------|------|
| `--device auto` | 自动选择（CUDA 优先） |
| `--device cuda` | 强制使用 GPU |
| `--device cpu` | 使用 CPU |

---

## 4）RVQ 层选择（本脚本的核心）

| 写法 | 含义 |
|------|------|
| `all` | 全层（最高质量） |
| `0` | 仅第一层（最低码率） |
| `0,1,2` | 指定多个层 |
| `1-4` | 连续区间 |
| `2:5` | 连续区间 |

示例：

```bash
--rvq_layers all
--rvq_layers 0,1,2
--rvq_layers 1-3
--rvq_layers 2:5
```

---

# 📌 常用命令示例

## ✔ 使用全部 RVQ 层（最常用）

```bash
python infer_st.py     --speech_file input.wav     --output_file out.wav     --rvq_layers all     --device cuda
```

## ✔ 使用第 0 层（最低码率）

```bash
python infer_st.py     --speech_file input.wav     --output_file out_0layer.wav     --rvq_layers 0
```

## ✔ 使用前 3 层

```bash
python infer_st.py     --speech_file input.wav     --output_file out_3layers.wav     --rvq_layers 0,1,2
```

## ✔ 使用 1-4 层

```bash
python infer_st.py     --speech_file input.wav     --output_file out_1to4.wav     --rvq_layers 1-4
```

## ✔ 自动从 HuggingFace 下载模型

```bash
python infer_st.py     --download     --speech_file input.wav     --output_file out.wav
```

---

# 📄 输出示例

```
[TIME] 加载模型耗时: 240.5 ms
[TIME] 加载+重采样耗时: 30.1 ms
[TIME] 编码耗时: 18.2 ms
[INFO] 模型 RVQ 层数: 12
[INFO] 选中层: [0, 1, 2]
[TIME] 解码耗时: 20.4 ms
[TIME] 写盘耗时: 2.1 ms
[TIME] 总耗时: 340.2 ms
[DONE] 输出文件: out.wav
```

---

# 📂 推荐目录结构

```
project/
│
├── infer_st.py
├── model_hub/
│     └── speechtokenizer_hubert_avg/
│           ├── config.json
│           ├── SpeechTokenizer.pt
│
└── input.wav
```

---

# 🧩 应用场景

- RVQ 层消融实验
- 压缩率 / 音质研究
- 声码器对比
- 快速离线推理
- 论文实验复现

---

# 📄 License

根据项目需求选择 MIT / Apache-2 / GPL 等许可证。
