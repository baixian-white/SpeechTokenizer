# 注意：目前测的实时是在win环境下跑的，因为wsl是我本地构建的虚拟环境，为避免麻烦，采取了win跑实时的方案
# 在win原生环境下启动st_audio_py310这个conda环境
conda activate st_audio_py310

# 知道设备索引
python -m sounddevice


# 直通模式（不走模型，测试设备链路是否正常
python demo_now.py --passthrough --monitor --mic_device 1 --spk_device 4

# 完整实时编解码（使用 SpeechTokenizer）
假设模型目录为

model_hub/speechtokenizer_hubert_avg/config.json
model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt


# 只用前3层的实施编码
python demo_now.py --config_path model_hub/speechtokenizer_hubert_avg/config.json --ckpt_path model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt --device cuda --monitor --chunk_seconds 0.25 --rvq_layers 3

# 命令行参数解析

## 1. 模型文件参数（Model Loading）
### 1.1
--config_path
含义： SpeechTokenizer 的 config.json 路径
是否必需： 非直通模式必须
示例：
--config_path model_hub/speechtokenizer_hubert_avg/config.json

### 1.2
--ckpt_path
含义： 模型权重（.pt 文件）路径
是否必需： 非直通模式必须
示例：
--ckpt_path model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt

## 2. 设备参数（Device Settings）
### 2.1
--device
模型计算设备
选项：cuda / cpu
建议：
有 GPU → cuda
调试 → cpu
### 2.2
--mic_device
麦克风设备索引
默认为 None（系统默认麦克风）
用以下方式查看索引：
python -m sounddevice
### 2.3
--spk_device
扬声器设备索引
默认为 None（系统默认扬声器）
### 2.4
--mic_sr
麦克风采样率
0 或不填 = 使用设备默认采样率（通常 48000）
一般不用改
### 2.5
--out_sr
输出播放采样率
0 = 使用扬声器默认 / 模型采样率
一般保持默认即可

## 3. 实时性能参数（Low Latency Options）
### 3.1
--frame_seconds
麦克风采集回调帧长
默认：0.02（20ms）
越小：延迟低，但 CPU 压力更高
建议范围：0.01 ~ 0.02
### 3.2
--chunk_seconds
模型每次处理的音频块时长
默认：0.25 秒
越小：延迟低，听感更碎
推荐：
稳定：0.25
超低延迟：0.12 ~ 0.2
### 3.3
--rvq_layers
使用前 L 层 RVQ
0 = 全部层（音质最佳）
>0 = 截层（码率更低、速度更快）
### 3.4
--gain_db
输出增益（单位：dB）
默认：0
例如：6 dB ≈ 声音 ×2

注意不要开太大以免削顶

## 4. 功能辅助参数（Functional Options）
### 4.1
--passthrough
直通模式：不走模型，仅麦克风 → 重采样 → 扬声器
用于排查设备是否正常
强烈建议第一次一定先跑它
示例：
python demo_now.py --passthrough --monitor
### 4.2
--monitor
打印每秒输入与输出 RMS（能量监控）
易判断链路是否有声音
RMS=0 → 音频没进来或没出去

### 4.3
--save_output
将实时播放的音频同时保存为 WAV
示例：
--save_output out.wav
## 5.内部参数（一般不需要修改）
### 5.1
--prebuffer_chunks
输出端的抖动缓冲数量
默认：2
建议不要改
### 5.2
--frame_out_seconds
输出端播放回调的帧长（固定 20ms）
不在命令行中暴露
不建议改动


