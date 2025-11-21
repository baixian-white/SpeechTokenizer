#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(Annotated) 本机直连实时链路：麦克风 → SpeechTokenizer 编解码 → 扬声器播放（可选同时写盘）

用途：
- 便于排查 “无声 / 卡顿 / 延迟大” 等问题；
- 可切换直通（不走模型）与真实编解码路径，逐段定位问题；
- 提供 RMS 监控、抖动缓冲（jitter buffer）、队列满丢旧等“实时优先”策略。

改进点（与常见实现相比）：
1) OutputStream 在构造时直接传入 callback，避免后设回调偶发无声；
2) --monitor 每秒打印 Mic 与 Out 的 RMS 指标（EMA），快速判断整链是否“有能量”；
3) --passthrough 跳过模型，仅做重采样直通播放，隔离设备与模型问题；
4) 采用 torchaudio.transforms.Resample 做流式重采样；
5) 播放侧维护独立队列 + 抖动缓冲；队列满时丢旧，保证实时性优先于完整性；
6) Windows 上优先尝试 WASAPI 输出设备，降低延迟与不兼容概率；
7) 可选 --gain_db 输出端做线性增益（谨慎，防削顶）。

自检建议：
- 第一步：--passthrough --monitor  应能听到原声，且日志显示 Mic/Out RMS > 0；
- 第二步：去掉 --passthrough，仅留 --monitor，应能听到重建语音，且 Out RMS > 0。

快速示例（Windows/WSL 外置声卡示意）：
  python realtime_st_pipeline_annotated.py \
      --config_path model_hub/speechtokenizer_hubert_avg/config.json \
      --ckpt_path   model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt \
      --device cuda --monitor --chunk_seconds 0.25 --rvq_layers 0

直通（设备链路自检，不走模型）：
  python realtime_st_pipeline_annotated.py --passthrough --monitor

常见排障要点：
- 听不到声：先跑直通；若直通也无声，优先检查设备索引、采样率、驱动；
- 有声但断续：适当增大 --chunk_seconds（如 0.25→0.32），或调小 --frame_seconds（更频繁采集）；
- 破音/削顶：减少 --gain_db，或确认原始 RMS 是否过大；
- GPU 慢：先用 --device cpu 验证逻辑，再切回 CUDA；
- 输出设备选错：用 sounddevice.query_devices() 查看索引，或不传索引让系统默认。
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional
from contextlib import nullcontext

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio

from speechtokenizer import SpeechTokenizer


# =============================
# 采集端：麦克风 → 线程安全队列
# =============================
@dataclass
class MicConfig:
    """麦克风采集相关配置。
    - device: 输入设备索引（None 表示默认设备）；
    - mic_sr: 期望的采样率（None 表示使用设备默认）；
    - channels: 采集声道数（此脚本使用单声道）；
    - dtype: 浮点格式；
    - frame_seconds: 采集回调帧长（越小回调越频繁，端到端延迟更低，但 CPU 压力更大）。
    """
    device: Optional[int]
    mic_sr: Optional[float]
    channels: int = 1
    dtype: str = "float32"
    frame_seconds: float = 0.02  # 20ms 回调


class MicProducer:
    """使用 sounddevice.InputStream 采集音频，并将每个回调块放入队列。

    设计要点：
    - 回调中不做重逻辑，仅做“转单声道 + 入队”；
    - 队列满（消费者跟不上）则丢弃当前块（实时优先）。
    """

    def __init__(self, q: "queue.Queue[np.ndarray]", cfg: MicConfig):
        self.q = q
        self.cfg = cfg
        self.stream: Optional[sd.InputStream] = None

    def _callback(self, indata, frames, time_info, status):
        # status 包含 XRuns 等状态，打印到 stderr 便于观察
        if status:
            print(f"[MIC] status: {status}", file=sys.stderr)
        # 转单声道：若是多声道，取第 1 声道；若已单声道，reshape 成一维
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata[:, 0].copy()
        else:
            mono = indata.reshape(-1).copy()
        # 非阻塞入队；满则丢弃当前块（保证实时）
        try:
            self.q.put_nowait(mono)
        except queue.Full:
            pass

    def start(self) -> float:
        """打开输入流并开始录音，返回实际使用的采样率。"""
        # 若未指定采样率，读取设备默认采样率；否则使用用户指定值
        if self.cfg.mic_sr is None:
            dev_info = sd.query_devices(self.cfg.device, "input")
            mic_sr = float(dev_info["default_samplerate"])  # e.g., 48000.0
        else:
            mic_sr = float(self.cfg.mic_sr)

        # blocksize = 每次回调的样本数 = 采样率 * 帧长；默认采样率48000，帧长为0.001
        blocksize = int(max(1, mic_sr * self.cfg.frame_seconds))

        # 降低系统端到端缓冲/延迟（并非所有平台都严格遵守）
        sd.default.latency = ("low", "low")

        # 构造并启动输入流（将回调直接传入构造函数）
        self.stream = sd.InputStream(
            samplerate=mic_sr,
            channels=self.cfg.channels,
            dtype=self.cfg.dtype,
            device=self.cfg.device,
            blocksize=blocksize,
            callback=self._callback,
        )
        self.stream.start()
        print(f"[MIC] started  device={self.cfg.device}  sr={mic_sr}  blocksize={blocksize}")
        return mic_sr

    def stop(self):
        """关闭采集流（可重复调用，容错处理）。"""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        print("[MIC] stopped.")


# =============================
# 实时管线：重采样 → (直通/编解码) → 播放
# =============================

def rms(x: np.ndarray) -> float:
    """计算一段波形的 RMS；用于健康度监控。"""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


class RealtimePipeline(threading.Thread):
    """从采集队列消费数据，聚合为模型块，做直通或编解码，然后送入播放缓冲。"""

    def __init__(
        self,
        q_mic: "queue.Queue[np.ndarray]",
        model: Optional[SpeechTokenizer],
        device: torch.device,
        mic_sr: float,
        model_sr: int,
        out_sr: int,
        spk_device: Optional[int],
        chunk_seconds: float,
        rvq_layers: int,
        save_output: Optional[str] = None,
        frame_out_seconds: float = 0.02,
        prebuffer_chunks: int = 2,
        monitor: bool = False,
        passthrough: bool = False,
        gain_db: float = 0.0,
        daemon: bool = True,
    ):
        # 将线程声明为守护线程，主进程退出时自动结束
        super().__init__(daemon=daemon)

        # —— 输入与模型参数 ——
        self.q_mic = q_mic
        self.model = model                      # passthrough=True 时允许为 None
        self.device = device
        self.mic_sr = float(mic_sr)
        self.model_sr = int(model_sr)
        self.out_sr = int(out_sr)
        self.spk_device = spk_device
        self.chunk_seconds = float(chunk_seconds)  # 模型端单块时长（权衡时延与算力吞吐）
        self.rvq_layers = max(0, int(rvq_layers))  # 0 表示“使用全部层”
        self.save_output = save_output             # 若不为 None，则边播边写 wav 文件

        # —— 播放端帧长与抖动缓冲配置 ——
        self.frame_out_seconds = float(frame_out_seconds) #0.02
        self.prebuffer_chunks = max(0, int(prebuffer_chunks))  # 预缓冲块数（当前实现中通过队列自然形成）

        # —— 监控与模式 ——
        self.monitor = monitor
        self.passthrough = passthrough
        self.gain_lin = float(10 ** (gain_db / 20.0)) if gain_db != 0.0 else 1.0

        # —— 重采样器：麦克风采样率 → 模型采样率 ——
        self.rs_mic2model = torchaudio.transforms.Resample(orig_freq=self.mic_sr, new_freq=self.model_sr)

        # 模型采样率 → 输出采样率（若两者相同则无需二次重采样）
        self.rs_model2out = None
        if self.out_sr != self.model_sr:
            self.rs_model2out = torchaudio.transforms.Resample(orig_freq=self.model_sr, new_freq=self.out_sr)

        # —— 模型侧累积缓冲：将零碎采集块拼成“模型块” ——
        self._model_buf = deque()   # 存储若干一维 np.float32 段
        self._model_buf_len = 0     # 当前缓存样本总数（模型采样率下）
        self._chunk_model_samples = max(1, int(self.chunk_seconds * self.model_sr))

        # —— 播放侧抖动缓冲：消费端需要连续帧 ——
        self._play_buf = deque()    # 已合并待播的片段
        self._play_buf_len = 0      # 总样本数（输出采样率下）
        self._play_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)  # 生产者→消费者的桥

        # —— 运行状态 & 资源句柄 ——
        self._running = True
        self._out_stream: Optional[sd.OutputStream] = None
        self._sf_writer: Optional[sf.SoundFile] = None

        # —— 监控指标（指数滑动均值，减抖） ——
        self._last_mon_ts = time.time()
        self._ema_mic = 0.0
        self._ema_out = 0.0

    def stop(self):
        """请求线程优雅退出（主线程会 join）。"""
        self._running = False

    # --- 设备选择：Windows 优先 WASAPI，尽量匹配同名设备 ---
    def _prefer_wasapi_output_index(self, wanted_index: Optional[int]) -> Optional[int]:
        try:
            hostapis = sd.query_hostapis()             # 列举 API（MME/DirectSound/WASAPI/CoreAudio...）
            wasapi_id = None
            for i, api in enumerate(hostapis):
                if "wasapi" in api.get("name", "").lower():
                    wasapi_id = i
                    break

            devices = sd.query_devices()

            def info(idx):
                d = devices[idx]
                return d["name"], d["hostapi"], d["max_output_channels"]

            if wasapi_id is None:
                # 平台无 WASAPI（如 Linux/macOS），按用户索引或默认设备
                return wanted_index

            if wanted_index is not None:
                # 若用户给了索引但不是 WASAPI，尝试找到“同名的 WASAPI 设备”
                name, hostapi, ch = info(wanted_index)
                if hostapi != wasapi_id or ch < 1:
                    for i, d in enumerate(devices):
                        if d["hostapi"] == wasapi_id and d["max_output_channels"] >= 1 and d["name"] == name:
                            return i
                return wanted_index
            else:
                # 未指定索引：优先挑“耳机/扬声器”等关键词的 WASAPI 设备
                cands = [(i, d["name"]) for i, d in enumerate(devices)
                         if d["hostapi"] == wasapi_id and d["max_output_channels"] >= 1]
                for kw in ["headphones", "耳机", "speaker", "扬声器"]:
                    for i, nm in cands:
                        if kw.lower() in nm.lower():
                            return i
                if cands:
                    return cands[0][0]
                return wanted_index
        except Exception:
            return wanted_index

    def _open_output_stream_with_fallbacks(self, target_sr: int, device_index: Optional[int], blocksize: int):
        """带回退策略地打开输出流：
        1) 优先尝试 WASAPI（若可用）；2) 其后使用原索引；3) 最后走默认设备。
        另外对 blocksize 也做两档尝试：[指定值, 0(由驱动决定)]。
        """
        trial_indices = []
        wasapi_idx = self._prefer_wasapi_output_index(device_index)
        if wasapi_idx is not None:
            trial_indices.append((wasapi_idx, "WASAPI-preferred"))
        if device_index is not None and device_index != wasapi_idx:
            trial_indices.append((device_index, "as-is"))
        trial_indices.append((None, "default-device"))

        blocksize_plan = [blocksize, 0]

        last_err = None
        for idx, tag in trial_indices:
            for bs in blocksize_plan:
                try:
                    stream = sd.OutputStream(
                        samplerate=target_sr,
                        channels=1,
                        dtype="float32",
                        device=idx,
                        blocksize=bs,
                        callback=self._play_callback,  # ✅ 回调在构造时传入
                    )
                    stream.start()
                    print(f"[PLAY] opened ({tag}) device={idx} sr={target_sr} blocksize={bs}")
                    return stream
                except Exception as e:
                    last_err = e
                    continue
        raise last_err if last_err else RuntimeError("Failed to open any output stream")

    # --- 重采样：mic_sr → model_sr（按块处理，避免长数组增长） ---
    def _rs_block_mic2model(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        t = torch.from_numpy(x).float().unsqueeze(0)  # [1, T]
        with torch.no_grad():
            y = self.rs_mic2model(t)                  # [1, T']
        return y.squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    # --- 编解码或直通 ---
    def _encode_decode(self, chunk_model: np.ndarray) -> np.ndarray:
        if chunk_model.size == 0:
            return np.zeros(0, dtype=np.float32)

        if self.passthrough or self.model is None:
            # 直通路径：模型采样率 → 输出采样率（必要时）
            t = torch.from_numpy(chunk_model).float()
            if self.rs_model2out is not None:
                with torch.no_grad():
                    y = self.rs_model2out(t.unsqueeze(0)).squeeze(0)
            else:
                y = t
            out = y.numpy()
        else:
            # 模型路径：encode → (截层) → decode
            wav_t = torch.from_numpy(chunk_model).to(self.device).float().unsqueeze(0).unsqueeze(0)  # [1,1,Tm]

            # GPU 上启用 autocast（混合精度）降低延时；CPU 上为 no-op
            use_amp = (self.device.type == "cuda")
            amp_ctx = (torch.amp.autocast(device_type="cuda") if use_amp else nullcontext())

            with torch.no_grad(), amp_ctx:
                codes = self.model.encode(wav_t)  # 形状约为 [n_q, B, Tq]
                if self.rvq_layers > 0:
                    # 仅取前 L 层（L 不超过实际层数）
                    L = min(self.rvq_layers, int(codes.shape[0]))
                    codes = codes[:L, :, :]
                wav_rec = self.model.decode(codes)  # [B=1, C=1, Tm]

            wav_model = wav_rec.detach().cpu().squeeze(0).squeeze(0).to(torch.float32)

            # 若模型采样率 != 输出采样率，再做一次重采样
            if self.rs_model2out is not None:
                with torch.no_grad():
                    y = self.rs_model2out(wav_model.unsqueeze(0)).squeeze(0)
            else:
                y = wav_model
            out = y.numpy()

        # 可选线性增益（限制到 [-1,1]）
        if self.gain_lin != 1.0 and out.size > 0:
            out = np.clip(out * self.gain_lin, -1.0, 1.0, out=out)
        return out

    # --- 将播放队列中的新块尽量搬到“连续播放缓冲”里 ---
    def _merge_play_chunks(self) -> int:
        pulled = 0
        while True:
            try:
                seg = self._play_queue.get_nowait()
            except queue.Empty:
                break
            if seg.size > 0:
                self._play_buf.append(seg)
                self._play_buf_len += seg.size
                pulled += 1
        return pulled

    # --- 播放回调：被 sounddevice 在音频线程周期性调用 ---
    def _play_callback(self, outdata, frames, time_info, status):
        if status:
            # 播放 XRuns 等在极端情况下可能出现（CPU 忙/块太大/驱动问题）
            pass
        needed = frames                           # 本次回调需要写入的样本数
        out = np.zeros(needed, dtype=np.float32)  # 预填零，避免读空造成杂音

        # 合并生产者最近放入的若干块（减小切片操作成本）
        self._merge_play_chunks()

        filled = 0
        while needed > 0 and self._play_buf_len > 0:
            seg = self._play_buf[0]
            take = min(needed, seg.size)
            out[filled:filled + take] = seg[:take]
            if take == seg.size:
                self._play_buf.popleft()
            else:
                self._play_buf[0] = seg[take:]
            self._play_buf_len -= take
            needed -= take
            filled += take

        # 将单声道波形写入 outdata（shape: [frames, channels]）
        outdata[:, 0] = out

        # 仅在开启监控时，维护输出 RMS 的 EMA（在主循环中打印，避免线程争用）
        if self.monitor:
            r = rms(out)
            self._ema_out = 0.9 * self._ema_out + 0.1 * r if self._ema_out > 0 else r

    # --- 主循环：从采集队列取块 → 拼成“模型块” → 编解码/直通 → 投递到播放队列 ---
    def run(self):
        # 尽量走低延迟配置
        sd.default.latency = ("low", "low")

        # 播放端 blocksize：每次回调帧长（越小越实时，CPU 越高）
        out_blocksize = max(1, int(self.out_sr * self.frame_out_seconds))

        # 预热模型（一次 encode/decode 触发权重加载与缓存初始化）
        if not self.passthrough and self.model is not None:
            with torch.no_grad():
                dummy = torch.zeros(1, 1, max(16, self._chunk_model_samples), device=self.device)
                _ = self.model.encode(dummy)
                _ = self.model.decode(self.model.encode(dummy))

        # 若需要写盘，先准备文件句柄（单声道 16-bit PCM）
        if self.save_output:
            self._sf_writer = sf.SoundFile(self.save_output, mode="w",
                                           samplerate=self.out_sr, channels=1, subtype="PCM_16")
            print(f"[PLAY] also writing to {self.save_output}")

        print(f"[PLAY] preparing buffer  target_sr={self.out_sr}  preferred_device={self.spk_device}  block={out_blocksize}")
        # 打开输出流（包含设备与 blocksize 的多重回退）
        self._out_stream = self._open_output_stream_with_fallbacks(
            target_sr=self.out_sr,
            device_index=self.spk_device,
            blocksize=out_blocksize,
        )

        # === 实时循环 ===
        while self._running:
            try:
                # 以 100ms 超时从采集端取一块（避免永久阻塞，便于打印监控）
                mic_block = self.q_mic.get(timeout=0.1)
            except queue.Empty:
                # 队列暂时无数据：周期性打印监控信息
                if self.monitor and time.time() - self._last_mon_ts >= 1.0:
                    print(f"[MON] Mic RMS(EMA)={self._ema_mic:.4f}  Out RMS(EMA)={self._ema_out:.4f}")
                    self._last_mon_ts = time.time()
                continue

            # 采集 RMS 更新（EMA 降抖）
            if self.monitor:
                r_mic = rms(mic_block)
                self._ema_mic = 0.9 * self._ema_mic + 0.1 * r_mic if self._ema_mic > 0 else r_mic

            # mic_sr → model_sr 的流式重采样
            rs = self._rs_block_mic2model(mic_block)

            # 将重采样后的数据放入“模型侧聚合缓冲”
            if rs.size > 0:
                self._model_buf.append(rs)
                self._model_buf_len += rs.size

            # 只要累计到一个“模型块”大小，就立刻处理（降低总时延）
            while self._model_buf_len >= self._chunk_model_samples:
                need = self._chunk_model_samples
                pieces = []
                # 从 deque 头部依次取片段，拼到一个整块数组
                while need > 0:
                    seg = self._model_buf[0]
                    take = min(need, seg.size)
                    pieces.append(seg[:take])
                    if take == seg.size:
                        self._model_buf.popleft()
                    else:
                        self._model_buf[0] = seg[take:]
                    need -= take
                    self._model_buf_len -= take
                send_chunk = np.concatenate(pieces, dtype=np.float32)

                # 送入模型（或直通）
                wav_out = self._encode_decode(send_chunk)

                if wav_out.size > 0:
                    # 投递到播放队列（队列满则丢弃队头：实时优先）
                    try:
                        self._play_queue.put_nowait(wav_out)
                    except queue.Full:
                        try:
                            _ = self._play_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self._play_queue.put_nowait(wav_out)

                    # 同步写盘（若开启）
                    if self._sf_writer is not None:
                        self._sf_writer.write(wav_out)

            # 每秒打印一次监控
            if self.monitor and time.time() - self._last_mon_ts >= 1.0:
                print(f"[MON] Mic RMS(EMA)={self._ema_mic:.4f}  Out RMS(EMA)={self._ema_out:.4f}")
                self._last_mon_ts = time.time()

        # === 退出前：尽量把模型缓冲里的剩余音频处理完 ===
        if self._model_buf_len > 0:
            remain = []
            while self._model_buf:
                remain.append(self._model_buf.popleft())
            send_chunk = np.concatenate(remain, dtype=np.float32)
            wav_out = self._encode_decode(send_chunk)
            if wav_out.size > 0:
                try:
                    self._play_queue.put_nowait(wav_out)
                except queue.Full:
                    pass

        # 给播放线程一些时间拉取最后的块
        time.sleep(0.2)

        # 关闭输出流与文件句柄（容错）
        try:
            if self._out_stream.active:
                self._out_stream.stop()
            self._out_stream.close()
        except Exception:
            pass
        if self._sf_writer is not None:
            self._sf_writer.close()
        print("[PLAY] stopped.")


# =============================
# 程序入口：参数解析 → 初始化 → 运行
# =============================

def main():
    parser = argparse.ArgumentParser(description="(Local) Mic → Realtime SpeechTokenizer → Speaker (No TCP, De-jitter)")
    # —— 模型文件 ——
    parser.add_argument("--config_path", type=str, required=False, default="",
                        help="config.json（直通模式下可不填）")
    parser.add_argument("--ckpt_path", type=str, required=False, default="",
                        help="SpeechTokenizer.pt（直通模式下可不填）")

    # —— 设备/性能选项 ——
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="推理设备")
    parser.add_argument("--mic_device", type=str, default="None",
                        help="麦克风设备索引或 'None'（默认设备）。")
    parser.add_argument("--spk_device", type=str, default="None",
                        help="扬声器设备索引或 'None'（默认）。")
    parser.add_argument("--mic_sr", type=float, default=0.0, help="指定麦克风采样率（0=使用设备默认）")
    parser.add_argument("--out_sr", type=float, default=0.0, help="指定输出采样率（0=使用设备默认/模型采样率）")

    # —— 实时性控制 ——
    parser.add_argument("--frame_seconds", type=float, default=0.02, help="采集回调帧长（秒）")
    parser.add_argument("--chunk_seconds", type=float, default=0.25, help="编码块时长（秒）")
    parser.add_argument("--rvq_layers", type=int, default=0, help="0=全部层，>0 仅前 L 层")

    # —— 辅助功能 ——
    parser.add_argument("--save_output", type=str, default=None, help="可选：边播边写出的 wav 路径")
    parser.add_argument("--monitor", action="store_true", help="每秒打印 Mic/Out RMS 监控")
    parser.add_argument("--passthrough", action="store_true", help="跳过模型，直通播放（用于设备链路自检）")
    parser.add_argument("--gain_db", type=float, default=0.0, help="对输出施加增益（dB），谨慎使用")

    args = parser.parse_args()

    # 限制 PyTorch 线程数（避免与 PortAudio/回调线程抢占过多 CPU）
    try:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))
    except Exception:
        pass

    # 解析设备索引：字符串 "None" → Python None；否则转 int
    mic_dev = None if args.mic_device == "None" else int(args.mic_device)
    spk_dev = None if args.spk_device == "None" else int(args.spk_device)

    # 默认：直通模式无需加载模型；若非直通，则加载 SpeechTokenizer
    model = None
    model_sr = 16000  # 兜底，若加载模型成功会被覆盖
    device = torch.device("cpu")

    if not args.passthrough:
        if not args.config_path or not args.ckpt_path:
            raise ValueError("非直通模式需要 --config_path 与 --ckpt_path")
        print("[MAIN] loading model ...")
        model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.ckpt_path)
        model.eval()
        use_cuda = (args.device == "cuda") and torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        model.to(device)
        model_sr = int(getattr(model, "sample_rate", 16000))
        print(f"[MAIN] model ready: sr={model_sr}, device={device}")
        if use_cuda:
            try:
                print("[MAIN] CUDA:", torch.cuda.get_device_name(0))
            except Exception:
                pass
    else:
        print("[MAIN] passthrough mode (no model).")

    # 建立采集队列（容量偏大，容纳短时抖动）
    q_mic: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=80)

    # 启动麦克风采集线程
    mic_cfg = MicConfig(
        device=mic_dev,
        mic_sr=(None if args.mic_sr <= 0 else float(args.mic_sr)),
        channels=1,
        dtype="float32",
        frame_seconds=args.frame_seconds,
    )
    producer = MicProducer(q=q_mic, cfg=mic_cfg)
    mic_sr = producer.start()

    # 计算输出采样率（优先用户指定；否则取扬声器默认；再不行，使用模型采样率）
    if args.out_sr > 0:
        out_sr = int(args.out_sr)
    else:
        try:
            dev_info = sd.query_devices(spk_dev, "output")
            out_sr = int(round(dev_info.get("default_samplerate", model_sr)))
        except Exception:
            out_sr = model_sr
    print(f"[MAIN] playback sr={out_sr} (speaker device={spk_dev})")

    # 启动实时管线线程
    pipe = RealtimePipeline(
        q_mic=q_mic,
        model=model if not args.passthrough else None,
        device=device,
        mic_sr=mic_sr,
        model_sr=model_sr,
        out_sr=out_sr,
        spk_device=spk_dev,
        chunk_seconds=args.chunk_seconds,
        rvq_layers=args.rvq_layers,
        save_output=args.save_output,
        frame_out_seconds=0.02,
        prebuffer_chunks=2,
        monitor=args.monitor,
        passthrough=args.passthrough,
        gain_db=args.gain_db,
        daemon=True,
    )
    pipe.start()

    # 主线程仅负责存活与 Ctrl+C 捕获
    print("[MAIN] running (press Ctrl+C to stop)")
    try:
        while pipe.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[MAIN] stopping ...")
    finally:
        pipe.stop()           # 请求管线线程退出
        pipe.join(timeout=2.0)
        producer.stop()       # 停止麦克风
        print("[MAIN] done.")


if __name__ == "__main__":
    main()
