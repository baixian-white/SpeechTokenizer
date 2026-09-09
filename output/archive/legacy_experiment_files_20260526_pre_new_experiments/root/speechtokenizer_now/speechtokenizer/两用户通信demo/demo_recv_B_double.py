#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
demo_recv_B.py (B 端接收播放)
流程（严格贴近 demo_now 的 decode+播放思路）：
TCP recv -> decrypt(codes) -> SpeechTokenizer.decode -> (model_sr -> out_sr 重采样可选)
-> sounddevice OutputStream 播放（含简单抖动缓冲 + 实时限幅）

关键优化（最优实时策略）：
- 不允许 play_buf_len 无限增长（否则延迟越来越大）
- 超过 MAX_LAT_SEC：丢弃最旧音频，回到 TARGET_LAT_SEC
- buffer 过载时：直接丢包不 decode（节省 CPU/GPU，防止雪崩）
"""

from __future__ import annotations

import argparse
import queue
import socket
import struct
import json
import sys
import time
from collections import deque
from contextlib import nullcontext
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import torch
import torchaudio

from speechtokenizer import SpeechTokenizer

from userB.common_config import HOST, PORT_B_SERVER, KEY_FILENAME
from userB.encryption_utils import xor_decrypt_int_ndarray


# =============================
# TCP 接收： [4B header_len][header_json][body_bytes]
# =============================

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while receiving")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(sock: socket.socket) -> Tuple[dict, bytes]:
    header_len = struct.unpack("!I", _recv_exact(sock, 4))[0]
    header = json.loads(_recv_exact(sock, header_len).decode("utf-8"))
    body_len = int(header["body_len"])
    body = _recv_exact(sock, body_len)
    return header, body


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


# =============================
# 播放输出：更稳的声道/采样率处理
# =============================

def _get_output_channels(device_index: Optional[int]) -> int:
    """
    根据设备能力选 1 或 2（常用就这两种）
    """
    try:
        info = sd.query_devices(device_index, "output")
        ch = int(info.get("max_output_channels", 0))
        if ch >= 2:
            return 2
        if ch >= 1:
            return 1
        return 0
    except Exception:
        return 1


def main():
    # ==========================================================
    # 1️⃣ 命令行参数解析（决定：模型、网络、播放、实时策略）
    # ==========================================================

    parser = argparse.ArgumentParser(
        description="B: TCP recv -> decrypt(codes) -> decode -> speaker"
    )

    # SpeechTokenizer 配置文件路径（必须与 A 端一致）
    parser.add_argument("--config_path", required=True)

    # SpeechTokenizer checkpoint 路径（必须与 A 端一致）
    parser.add_argument("--ckpt_path", required=True)

    # 推理设备选择（cuda / cpu）
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # B 端监听的 TCP 端口（A 端会 connect 到这里）
    parser.add_argument("--listen_port", type=int, default=PORT_B_SERVER)

    # 输出音频设备（None = 默认扬声器）
    parser.add_argument("--spk_device", default="None")

    # 输出采样率：
    # - 0 表示：优先使用声卡默认采样率
    # - 若获取失败，则回退到 model_sr
    parser.add_argument(
        "--out_sr", type=int, default=0,
        help="0=use device default or model_sr fallback"
    )

    # 是否打印播放监控信息（RMS / buffer 状态）
    parser.add_argument("--monitor", action="store_true")

    # 播放回调的时间分辨率（默认 20ms）
    parser.add_argument("--frame_out_seconds", type=float, default=0.02)

    # ==========================================================
    # ✅ 实时播放控制的核心参数（非常重要）
    # ==========================================================

    # 目标播放延迟（秒）
    # 理想情况下，播放缓冲区长度 ≈ target_lat
    parser.add_argument(
        "--target_lat", type=float, default=0.25,
        help="target playback latency (sec)"
    )

    # 最大允许延迟（秒）
    # 超过这个阈值说明网络或 decode 跟不上，必须丢数据
    parser.add_argument(
        "--max_lat", type=float, default=0.80,
        help="max playback latency (sec), exceed -> trim old audio"
    )

    # 当 buffer 已经过载时：
    # - True：直接丢弃收到的 codes（不 decode，节省算力）
    # - False：仍然 decode，但之后会 trim 播放缓冲
    parser.add_argument(
        "--drop_when_over", action="store_true",
        help="if buffer over max_lat, drop packets before decode"
    )

    parser.add_argument(
    "--listen_ip", type=str, default="0.0.0.0",
    help="The IP address to listen on. Use 0.0.0.0 for all interfaces."
)
    

    args = parser.parse_args()

    # ==========================================================
    # 2️⃣ 加载 SpeechTokenizer 模型（解码端）
    # ==========================================================

    model = SpeechTokenizer.load_from_checkpoint(
        args.config_path,
        args.ckpt_path
    )

    # 推理模式
    model.eval()

    # 判断是否可以使用 GPU
    use_cuda = (args.device == "cuda") and torch.cuda.is_available()

    # 选择推理设备
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # 把模型放到对应设备
    model.to(device)

    # 获取模型采样率（decode 输出的 wav 就是这个 sr）
    model_sr = int(getattr(model, "sample_rate", 16000))

    print(f"[MODEL] sr={model_sr} device={device}")

    # ==========================================================
    # 3️⃣ 输出音频设备 & 采样率配置
    # ==========================================================

    # 解析扬声器设备编号
    spk_dev = None if args.spk_device == "None" else int(args.spk_device)

    # 确定最终输出采样率 out_sr
    if args.out_sr > 0:
        # 用户强制指定
        out_sr = int(args.out_sr)
    else:
        try:
            # 尝试从声卡查询默认采样率
            info = sd.query_devices(spk_dev, "output")
            out_sr = int(round(float(info.get("default_samplerate", model_sr))))
        except Exception:
            # 查询失败则回退到模型采样率
            out_sr = model_sr

    # 查询输出设备的声道数（1=单声道，2=立体声）
    out_channels = _get_output_channels(spk_dev)

    if out_channels <= 0:
        raise RuntimeError(
            f"Selected output device {spk_dev} has no output channels."
        )

    print(f"[PLAY] out_sr={out_sr} spk_device={spk_dev} channels={out_channels}")

    # 如果播放采样率 ≠ 模型采样率，需要一个重采样器
    rs_model2out = None
    if out_sr != model_sr:
        rs_model2out = torchaudio.transforms.Resample(
            orig_freq=model_sr,
            new_freq=out_sr
        )

    # ==========================================================
    # 4️⃣ 抖动缓冲（Jitter Buffer）结构
    # ==========================================================

    # play_queue：
    # - 主线程 decode 完的 wav 放这里
    # - 播放回调线程从这里取
    play_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)

    # play_buf：
    # - deque，用于真正“播放前”的连续缓冲
    # - 允许细粒度 trim / slice
    play_buf = deque()

    # 当前播放缓冲区的总样本数（以 out_sr 为基准）
    play_buf_len = 0

    # ==========================================================
    # 5️⃣ 实时播放的关键阈值（以样本数表示）
    # ==========================================================

    # 目标缓冲长度（样本数）
    target_samples = max(1, int(float(args.target_lat) * out_sr))

    # 最大缓冲长度（样本数）
    max_samples = max(
        target_samples + 1,
        int(float(args.max_lat) * out_sr)
    )

    ema_out = 0.0
    last_ts = time.time()

    # ==========================================================
    # 6️⃣ 辅助函数：播放缓冲管理
    # ==========================================================

    def _merge_play_chunks() -> None:
        """
        从 play_queue 中把新解码的音频段
        合并进 play_buf（deque）
        """
        nonlocal play_buf_len

        while True:
            try:
                seg = play_queue.get_nowait()
            except queue.Empty:
                break

            if seg.size > 0:
                play_buf.append(seg)
                play_buf_len += seg.size

    def _trim_play_buffer() -> None:
        """
        ✅ 核心实时策略：
        - 播放缓冲不能无限增长
        - 若超过 max_samples：
            丢弃最旧音频，回退到 target_samples
        """
        nonlocal play_buf_len

        if play_buf_len <= max_samples:
            return

        drop = play_buf_len - target_samples

        while drop > 0 and play_buf_len > 0 and len(play_buf) > 0:
            seg = play_buf[0]

            if drop >= seg.size:
                # 整段丢弃
                play_buf.popleft()
                play_buf_len -= seg.size
                drop -= seg.size
            else:
                # 丢弃段头一部分
                play_buf[0] = seg[drop:]
                play_buf_len -= drop
                drop = 0

    # ==========================================================
    # 7️⃣ 声卡播放回调（真正“消耗”音频的地方）
    # ==========================================================

    def _play_callback(outdata, frames, time_info, status):
        """
        每次声卡需要 frames 个样本时触发
        """
        nonlocal play_buf_len, ema_out

        # ① 把新解码的音频合并进播放缓冲
        _merge_play_chunks()

        # ② 控制缓冲上限（防止延迟无限增长）
        _trim_play_buffer()

        # ③ 从 play_buf 中取 frames 个样本用于播放
        out = np.zeros(frames, dtype=np.float32)
        filled = 0

        while filled < frames and play_buf_len > 0:
            seg = play_buf[0]
            take = min(frames - filled, seg.size)

            out[filled:filled + take] = seg[:take]

            if take == seg.size:
                play_buf.popleft()
            else:
                play_buf[0] = seg[take:]

            play_buf_len -= take
            filled += take

        # 写入声卡输出缓冲
        if out_channels == 1:
            outdata[:, 0] = out
        else:
            outdata[:, 0] = out
            outdata[:, 1] = out

        # 可选：监控播放 RMS
        if args.monitor:
            r = _rms(out)
            ema_out = 0.9 * ema_out + 0.1 * r if ema_out > 0 else r

    # ==========================================================
    # 8️⃣ 打开声卡输出流
    # ==========================================================

    sd.default.latency = ("low", "low")

    # 每次播放回调请求的样本数
    out_blocksize = max(
        1,
        int(out_sr * float(args.frame_out_seconds))
    )

    out_stream = sd.OutputStream(
        samplerate=out_sr,
        channels=out_channels,
        dtype="float32",
        device=spk_dev,
        blocksize=out_blocksize,
        callback=_play_callback,
    )

    out_stream.start()

    print(f"[PLAY] stream started block={out_blocksize}")
    print(
        f"[BUF] target={args.target_lat:.3f}s({target_samples} samples) "
        f"max={args.max_lat:.3f}s({max_samples} samples)"
    )

    # ==========================================================
    # 9️⃣ TCP Server：等待 A 端连接
    # ==========================================================

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind((args.listen_ip, int(args.listen_port))) 
    server.listen(1)

    print(f"[TCP] listening on {args.listen_ip}:{args.listen_port} ...")

    conn, addr = server.accept()
    print(f"[TCP] accepted from {addr}")

    # ==========================================================
    # 🔟 主接收循环：recv → decrypt → decode → enqueue
    # ==========================================================

    try:
        while True:
            # 接收一条完整消息（应用层协议）
            header, body = _recv_message(conn)

            # 非 codes 消息直接跳过
            if header.get("kind") != "codes_v1":
                continue

            # 若播放缓冲已严重过载，直接丢包（不 decode）
            if args.drop_when_over and play_buf_len > max_samples:
                continue

            # 根据 header 还原 ndarray
            dtype = np.dtype(header["dtype"])
            shape = tuple(header["shape"])

            enc_codes = np.frombuffer(body, dtype=dtype).reshape(shape)

            # 解密 XOR
            try:
                codes = xor_decrypt_int_ndarray(
                    enc_codes,
                    KEY_FILENAME
                ).astype(np.int64, copy=False)
            except Exception as e:
                print(f"[RECV] decrypt failed: {e}", file=sys.stderr)
                continue

            # 转为 torch tensor，送入 decode
            codes_t = torch.from_numpy(codes).to(device)

            # decode（可选 AMP）
            use_amp = (device.type == "cuda")
            amp_ctx = (
                torch.amp.autocast(device_type="cuda")
                if use_amp else nullcontext()
            )

            with torch.no_grad(), amp_ctx:
                wav_t = model.decode(codes_t)

            # [1,1,T] → [T]
            wav = (
                wav_t.detach()
                .cpu()
                .squeeze(0)
                .squeeze(0)
                .to(torch.float32)
                .numpy()
            )

            # 若输出采样率 ≠ model_sr，重采样
            if rs_model2out is not None:
                with torch.no_grad():
                    wav = (
                        rs_model2out(
                            torch.from_numpy(wav).unsqueeze(0)
                        )
                        .squeeze(0)
                        .numpy()
                    )

            wav = wav.astype(np.float32, copy=False)

            # 推入播放队列（非阻塞）
            try:
                play_queue.put_nowait(wav)
            except queue.Full:
                # 实时系统：宁可丢音，也不阻塞
                pass

            # 主线程也同步维护一次缓冲状态
            _merge_play_chunks()
            _trim_play_buffer()

            # 监控信息
            if args.monitor and time.time() - last_ts >= 1.0:
                buf_sec = play_buf_len / float(out_sr)
                print(
                    f"[MON] Out RMS(EMA)={ema_out:.4f}  "
                    f"play_buf_len={play_buf_len} ({buf_sec:.3f}s)"
                )
                last_ts = time.time()

    except KeyboardInterrupt:
        print("\n[RECV] stopping ...")
    except ConnectionError:
        print("[RECV] connection closed.")
    finally:
        # 资源释放
        try:
            conn.close()
        except Exception:
            pass

        try:
            server.close()
        except Exception:
            pass

        try:
            out_stream.stop()
            out_stream.close()
        except Exception:
            pass

        print("[RECV] done.")



if __name__ == "__main__":
    main()
