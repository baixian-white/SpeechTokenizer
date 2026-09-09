#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
demo_send_A.py （A 端：发送端）

整体流程（与你的 demo_now 思路严格一致）：

1️⃣ 麦克风实时采集音频（sounddevice）
2️⃣ mic_sr → model_sr 重采样（torchaudio）
3️⃣ 按 chunk_seconds 聚合为“模型输入块”
4️⃣ SpeechTokenizer.encode → 离散语音 codes（RVQ）
5️⃣ 可选：只保留前 L 层 RVQ codes
6️⃣ 对整数 codes 进行 XOR 加密
7️⃣ 通过 TCP 发送给 B 端

⚠️ 注意：
- 本文件 **只发送 codes，不发送 wav**
- B 端负责：解密 → decode → 播放
"""

from __future__ import annotations

# =========================
# 标准库
# =========================
import argparse          # 命令行参数解析
import queue             # 线程安全队列（音频回调 → 主线程）
import socket            # TCP 通信
import struct            # 二进制封包（网络字节序）
import json              # TCP header 序列化
import sys
import time
from dataclasses import dataclass
from typing import Optional

# =========================
# 第三方库
# =========================
import numpy as np
import sounddevice as sd         # 麦克风采集
import torch
import torchaudio                # 重采样

# =========================
# 项目内模块
# =========================
from speechtokenizer import SpeechTokenizer

from userA.common_config import HOST, PORT_B_SERVER, KEY_FILENAME
from userA.encryption_utils import xor_encrypt_int_ndarray


# ============================================================
# 一、TCP 消息封包协议
# ============================================================
# 格式：
# [4 字节 header 长度][header_json][body_bytes]
#
# 这样做的好处：
# - TCP 是流协议，没有“消息边界”
# - 明确 header 长度，B 端可以稳定解析
# ============================================================

def _send_all(sock: socket.socket, data: bytes) -> None:
    """
    确保 data 中的所有字节都被完整发送
    （socket.send 可能一次只发送一部分）
    """
    view = memoryview(data)
    while view:
        n = sock.send(view)
        if n <= 0:
            raise ConnectionError("socket send failed")
        view = view[n:]


def _send_message(sock: socket.socket, header: dict, body: bytes) -> None:
    """
    发送一条完整消息：
    - header：JSON（描述 body 的 dtype / shape 等）
    - body：原始二进制（这里是加密后的 codes）
    """
    header_bytes = json.dumps(
        header,
        ensure_ascii=False
    ).encode("utf-8")

    # ① 发送 header 长度（4 字节，大端）
    _send_all(sock, struct.pack("!I", len(header_bytes)))
    # ② 发送 header 本体
    _send_all(sock, header_bytes)
    # ③ 发送 body
    _send_all(sock, body)


# ============================================================
# 二、麦克风采集模块
# ============================================================
# 使用 sounddevice 的 callback 机制：
# - 回调线程只负责“采集 + 入队”
# - 主线程负责计算（encode / resample）
# ============================================================

@dataclass
class MicConfig:
    """
    麦克风配置参数
    """
    device: Optional[int]         # 输入设备编号（None = 默认）
    mic_sr: Optional[float]       # 麦克风采样率（None = 设备默认）
    channels: int = 1             # 单声道
    dtype: str = "float32"
    frame_seconds: float = 0.02   # 每帧时长（20ms）


class MicProducer:
    """
    麦克风采集生产者（Producer）

    ⚠️ 设计原则：
    - callback 中不做任何“重计算”
    - 队列满了就丢（实时性 > 完整性）
    """

    def __init__(self, q: "queue.Queue[np.ndarray]", cfg: MicConfig):
        self.q = q
        self.cfg = cfg
        self.stream: Optional[sd.InputStream] = None

    def _callback(self, indata, frames, time_info, status):
        """
        sounddevice 的音频回调函数
        """
        if status:
            print(f"[MIC] status: {status}", file=sys.stderr)

        # 多声道 → 只取第 1 个通道
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata[:, 0]
        else:
            mono = indata.reshape(-1)

        # 非阻塞入队（队列满就丢）
        try:
            self.q.put_nowait(mono.astype(np.float32, copy=True))
        except queue.Full:
            pass

    def start(self) -> float:
        """
        启动麦克风流，并返回实际使用的采样率
        """
        if self.cfg.mic_sr is None:
            mic_sr = float(
                sd.query_devices(self.cfg.device, "input")["default_samplerate"]
            )
        else:
            mic_sr = float(self.cfg.mic_sr)

        # 每次 callback 的采样点数
        blocksize = int(max(1, mic_sr * self.cfg.frame_seconds))

        # 低延迟模式
        sd.default.latency = ("low", "low")

        self.stream = sd.InputStream(
            samplerate=mic_sr,
            channels=self.cfg.channels,
            dtype=self.cfg.dtype,
            device=self.cfg.device,
            blocksize=blocksize,
            callback=self._callback,
        )
        self.stream.start()
        print(f"[MIC] started device={self.cfg.device} sr={mic_sr} block={blocksize}")
        return mic_sr

    def stop(self):
        """
        停止麦克风
        """
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
        self.stream = None
        print("[MIC] stopped.")


# ============================================================
# 三、辅助函数
# ============================================================

def _rms(x: np.ndarray) -> float:
    """
    计算 RMS（用于麦克风音量监控）
    """
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


# ============================================================
# 四、主逻辑
# ============================================================

def main():
    # ==========================================================
    # 1️⃣ 解析命令行参数（决定：模型路径、设备、音频参数、网络参数）
    # ==========================================================

    # 创建 ArgumentParser：用于从命令行读取参数
    # 例如：python demo_send_A.py --config_path ... --ckpt_path ...
    parser = argparse.ArgumentParser(
        description="A: Mic -> encode -> encrypt(codes) -> TCP send to B"
    )

    # SpeechTokenizer 的配置文件路径（通常是 json/yaml 之类）
    # 必填：不传就报错并退出
    parser.add_argument("--config_path", required=True)

    # SpeechTokenizer 的 checkpoint 权重文件路径（.ckpt/.pt）
    # 必填：不传就报错并退出
    parser.add_argument("--ckpt_path", required=True)

    # 推理设备选择：默认 cuda；如果机器没 GPU 或 cuda 不可用，后面会自动退回 cpu
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # 选择麦克风输入设备：
    # - "None" 表示使用系统默认输入设备
    # - 传数字字符串如 "3" 表示使用第3号输入设备
    parser.add_argument("--mic_device", default="None")

    # 指定麦克风采样率：
    # - 默认 0.0 表示不指定，直接使用声卡默认采样率（sd.query_devices得到）
    # - 如果传 44100/48000 等，则强制用该采样率采集
    parser.add_argument("--mic_sr", type=float, default=0.0)

    # 声卡回调每次采集的帧时长（秒）
    # 默认 0.02 = 20ms
    # 这会影响 callback 的 blocksize（= mic_sr * frame_seconds）
    parser.add_argument("--frame_seconds", type=float, default=0.02)

    # 模型输入的“聚合块”时长（秒）
    # 默认 0.25 = 250ms
    # 注意：这会决定端到端延迟的重要组成部分
    parser.add_argument("--chunk_seconds", type=float, default=0.25)

    # RVQ 层数截断：
    # - 0 表示不截断，用模型输出全部 n_q 层
    # - >0 表示只保留前 L 层（降低码率、可能降低音质）
    parser.add_argument("--rvq_layers", type=int, default=0,
                        help="0=all, >0 only first L RVQ layers")

    # 是否打印监控信息（例如 RMS 音量）
    # 不影响核心逻辑，只是调试/观察用
    parser.add_argument("--monitor", action="store_true")

    # B 端监听的端口号（A端要 connect 到这个端口）
    # 默认来自配置 PORT_B_SERVER
    parser.add_argument("--peer_port", type=int, default=PORT_B_SERVER)

    # 真正解析命令行参数：执行后 args.xxx 才能取到用户传的值
    args = parser.parse_args()

    # ==========================================================
    # 2️⃣ 加载 SpeechTokenizer 模型（决定 model_sr、encode 输出形态）
    # ==========================================================

    # 从配置 + checkpoint 加载模型
    # 这里通常会构建网络结构，并加载权重
    model = SpeechTokenizer.load_from_checkpoint(
        args.config_path,
        args.ckpt_path
    )

    # eval 模式：关闭 dropout / 使用 BN 推理行为
    # 推理必须设 eval，否则输出可能不稳定
    model.eval()

    # 判断是否真的能用 GPU：
    # - args.device == "cuda" 表示用户希望用 GPU
    # - torch.cuda.is_available() 表示当前环境确实有 CUDA 可用
    use_cuda = (args.device == "cuda") and torch.cuda.is_available()

    # 选择 torch.device：
    # - 能用 GPU -> cuda:0
    # - 否则 -> cpu
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # 把模型参数搬到对应设备（GPU 或 CPU）
    model.to(device)

    # 获取模型训练/推理所需采样率（关键：后续必须重采样到这个 sr）
    # 如果模型对象没有 sample_rate 属性，则默认 16000
    model_sr = int(getattr(model, "sample_rate", 16000))

    # 打印：模型采样率 + 推理设备
    print(f"[MODEL] sr={model_sr} device={device}")

    # ==========================================================
    # 3️⃣ 启动麦克风采集（得到 mic_sr，并把音频帧塞入队列 q_mic）
    # ==========================================================

    # 解析 mic_device：
    # - 命令行传 "None" -> 用默认设备 -> mic_dev = None
    # - 否则把字符串转成 int（设备编号）
    mic_dev = None if args.mic_device == "None" else int(args.mic_device)

    # 创建麦克风帧队列：
    # - callback 线程往里 put_nowait（满了就丢）
    # - 主线程在 while True 里 get() 来处理
    # maxsize=80 表示最多缓存 80 帧（每帧默认 20ms）
    # 80帧≈1.6秒缓存上限（只是上限，实时情况下不会堆这么多）
    q_mic: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=80)

    # 构建麦克风配置：
    # - mic_sr：若 args.mic_sr<=0，表示用设备默认（传 None）
    # - 否则使用用户指定的采样率
    # frame_seconds：决定每次采集块大小
    mic_cfg = MicConfig(
        device=mic_dev,
        mic_sr=None if args.mic_sr <= 0 else args.mic_sr,
        frame_seconds=args.frame_seconds,
    )

    # 创建麦克风采集器：内部会创建 sounddevice.InputStream
    producer = MicProducer(q_mic, mic_cfg)

    # 启动采集：
    # 返回实际使用的 mic_sr（可能是设备默认 44100/48000，也可能是你指定的）
    mic_sr = producer.start()

    # 到这里为止：
    # - q_mic 中每个元素：np.float32, shape≈[mic_sr*frame_seconds]
    # - 采样率还在 mic_sr（声卡世界）

    # ==========================================================
    # 4️⃣ 创建重采样器（把 mic_sr 的音频转成 model_sr）
    # ==========================================================

    # torchaudio 的 Resample：
    # 输入：采样率 mic_sr
    # 输出：采样率 model_sr
    # 这是为了匹配模型要求（否则 encode 的频带/时间尺度会错）
    rs_mic2model = torchaudio.transforms.Resample(
        orig_freq=mic_sr,
        new_freq=model_sr
    )

    # ==========================================================
    # 5️⃣ 建立 TCP 连接（A端主动连接到 B端监听的地址/端口）
    # ==========================================================

    # 创建 TCP socket（IPv4 + TCP）
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 打印：准备连接的目标
    print(f"[TCP] connecting to {HOST}:{args.peer_port} ...")

    # 发起连接：如果 B 端没开服务/端口不通，会在这里报错
    sock.connect((HOST, args.peer_port))

    # 连接成功后才会走到这里
    print("[TCP] connected.")

    # ==========================================================
    # 6️⃣ Chunk 聚合缓冲（把重采样后的连续小帧，攒成固定长度 chunk）
    # ==========================================================

    # 计算一个 chunk 需要多少个“模型采样点”
    # chunk_seconds=0.25, model_sr=16000 -> chunk_model_samples=4000
    chunk_model_samples = int(args.chunk_seconds * model_sr)

    # model_buf 用来存放重采样后的片段（每个片段是 np.ndarray）
    # 这样做的目的：
    # - 避免频繁拼接（只在够一个 chunk 时才拼接）
    model_buf = []

    # model_buf_len 记录当前缓冲区一共有多少“模型采样点”
    # 方便判断是否已经够一个 chunk
    model_buf_len = 0

    # last_ts 用于控制监控打印节奏
    last_ts = time.time()

    # ema_mic 是麦克风 RMS 的指数滑动平均（避免波动太大）
    ema_mic = 0.0

    # ==========================================================
    # 7️⃣ 主循环：实时处理链路
    #    mic帧 -> 重采样 -> 放入缓冲 -> 够chunk就encode->加密->发送
    # ==========================================================

    try:
        while True:
            # -----------------------------
            # 7.1 从队列取一帧麦克风数据
            # -----------------------------
            # timeout=0.1：最多等100ms
            # - 若没有新音频，就抛 queue.Empty
            # - 这样不会卡死在 get()，主循环还能做其它事（比如监控/退出）
            try:
                mic_block = q_mic.get(timeout=0.1)
            except queue.Empty:
                # 没取到音频帧：直接进入下一轮循环继续等
                continue

            # 此时 mic_block：
            # - dtype float32
            # - shape 约等于 [mic_sr * frame_seconds]
            # - 采样率是 mic_sr

            # -----------------------------
            # 7.2 （可选）监控音量：计算 RMS 并做 EMA 平滑
            # -----------------------------
            if args.monitor:
                # 当前帧的 RMS（反映响度）
                r = _rms(mic_block)

                # ema_mic=0 表示第一次初始化
                # 后续使用 ema: 0.9旧 + 0.1新
                ema_mic = r if ema_mic == 0 else 0.9 * ema_mic + 0.1 * r

            # -----------------------------
            # 7.3 重采样：mic_sr -> model_sr
            # -----------------------------
            # torch.no_grad(): 推理阶段关闭梯度，减少显存和开销
            with torch.no_grad():
                # torchaudio Resample 输入期望是 tensor
                # unsqueeze(0) 把 [T] 变成 [1,T]（batch维度）
                t = torch.from_numpy(mic_block).unsqueeze(0)

                # 重采样输出：
                # y: [1, T']，T' ≈ T * model_sr / mic_sr
                y = rs_mic2model(t).squeeze(0)  # squeeze(0) 回到 [T']

            # 转回 numpy（仍是 float，仍是“波形”）
            rs = y.cpu().numpy()

            # 此时 rs：
            # - dtype 通常 float32（可能跟 torch 默认有关，但整体是 float）
            # - shape [T']
            # - 采样率已经变成 model_sr（模型世界）

            # -----------------------------
            # 7.4 放入“模型采样率缓冲区”，用于凑 chunk
            # -----------------------------
            model_buf.append(rs)         # 把这一段重采样后的数据存起来
            model_buf_len += rs.size     # 更新缓冲区总长度（采样点数）

            # -----------------------------
            # 7.5 只要缓冲够一个 chunk，就循环处理（可能一次处理多个chunk）
            # -----------------------------
            while model_buf_len >= chunk_model_samples:
                # need 表示我们还需要多少采样点才能凑满一个 chunk
                need = chunk_model_samples

                # pieces 用来收集凑 chunk 的若干小段
                pieces = []

                # 从 model_buf 头部不断取数据，直到凑够 need
                while need > 0:
                    # 取队首片段
                    seg = model_buf[0]

                    # take：本次从 seg 里取多少点
                    take = min(need, seg.size)

                    # 收集 seg 的前 take 个点
                    pieces.append(seg[:take])

                    # 如果 seg 已经被取完：弹出队首
                    if take == seg.size:
                        model_buf.pop(0)
                    else:
                        # seg 还剩一部分：把 seg 剩余部分放回队首
                        model_buf[0] = seg[take:]

                    # 更新总长度与需求
                    model_buf_len -= take
                    need -= take

                # 把 pieces 拼成一个完整 chunk（长度=chunk_model_samples）
                chunk = np.concatenate(pieces)

                # 此时 chunk：
                # - dtype float
                # - shape [chunk_model_samples]
                # - 采样率 = model_sr
                # - 时长 = chunk_seconds

                # -----------------------------
                # 7.6 encode：chunk(波形) -> codes(离散token)
                # -----------------------------

                # 模型 encode 输入期望形状一般是 [B, C, T]
                # - B=1（批次）
                # - C=1（单声道）
                # - T=chunk_model_samples
                wav_t = torch.from_numpy(chunk).to(device)\
                            .unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    # 进行编码：输出通常是 RVQ codes
                    # 常见形状：[n_q, B, T_q]
                    # - n_q：量化器层数（RVQ层数）
                    # - B：batch=1
                    # - T_q：离散时间步长度（与模型内部下采样有关）
                    codes_t = model.encode(wav_t)

                    # 如果用户指定 rvq_layers>0：
                    # 只保留前 L 层 codes，相当于降低码率
                    if args.rvq_layers > 0:
                        codes_t = codes_t[:args.rvq_layers]

                # 转到 CPU + numpy + int64
                # 注意：从这一刻开始 codes 不再是“采样点”，也不再谈采样率
                codes = codes_t.cpu().numpy().astype(np.int64)

                # ===============================
                # DEBUG / 学习用途：保存一份 codes
                # ===============================
                np.savez(
                    "debug_codes.npz",
                    codes=codes,
                    rvq_layers=codes.shape[0],
                    T_q=codes.shape[-1],
                )

                # -----------------------------
                # 7.7 加密：对整数 codes 做 XOR（不改变 shape）
                # -----------------------------
                enc_codes = xor_encrypt_int_ndarray(
                    codes,
                    KEY_FILENAME
                )

                # -----------------------------
                # 7.8 发送：封装 header + body
                # -----------------------------

                # 把 enc_codes 按行优先（C order）序列化成 bytes
                # body 就是“网络真正传输的数据载荷”
                body = enc_codes.tobytes()

                # header 用 JSON 描述 body 的元信息，让 B 端知道如何还原 ndarray
                header = {
                    "kind": "codes_v1",                # 协议类型/版本标识（便于以后扩展）
                    "dtype": str(enc_codes.dtype),     # 数据类型（B端要按这个还原）
                    "shape": list(enc_codes.shape),    # 维度形状（B端要按这个 reshape）
                    "body_len": len(body),             # body 字节长度（B端精确读取）
                }

                # 发送一条“完整消息”：
                # [4B header_len][header_json][body_bytes]
                _send_message(sock, header, body)

    # Ctrl+C 会触发 KeyboardInterrupt
    except KeyboardInterrupt:
        print("\n[SEND] stopping ...")

    finally:
        # finally 保证无论正常退出还是异常，都能释放资源

        # 停止麦克风采集流
        producer.stop()

        # 关闭 TCP socket
        sock.close()

        print("[SEND] done.")


if __name__ == "__main__":
    main()


