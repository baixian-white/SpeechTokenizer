#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
demo_duplex.py (全双工实时通讯终端 - 详细注释版)

核心架构：
1. 单进程双线程模型：
   - 线程 A (Sender): 负责“采集 -> 编码 -> 加密 -> 发送”
   - 线程 B (Receiver): 负责“接收 -> 解密 -> 解码 -> 播放”
2. 共享资源：
   - SpeechTokenizer 模型对象（节省显存，推理时线程安全）
3. 网络拓扑：
   - User A 和 User B 互相作为对方的 TCP 客户端和服务器
   - 建立了由两条单向 TCP 连接组成的双向通道

使用方法：
User A: python demo_duplex.py --role A --config_path ... --ckpt_path ...
User B: python demo_duplex.py --role B --config_path ... --ckpt_path ...
"""

from __future__ import annotations

import argparse
import queue
import socket
import struct
import json
import sys
import time
import threading
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import torch
import torchaudio

from speechtokenizer import SpeechTokenizer

# 引入项目特定的配置和加密工具
# 确保 userA 文件夹在 PYTHONPATH 中或者与脚本在同级目录
from userA.common_config import HOST, PORT_A_SERVER, PORT_B_SERVER, KEY_FILENAME
from userA.encryption_utils import xor_decrypt_int_ndarray, xor_encrypt_int_ndarray

# ============================================================
# 网络通信辅助函数 (TCP 封包/解包)
# ============================================================
# 协议格式设计：
# [4字节 Header长度] + [Header JSON字符串] + [Body 二进制数据]
# 目的：解决 TCP 粘包/拆包问题，并让接收端知道如何解析 Body (dtype, shape)
# ============================================================

def _send_all(sock: socket.socket, data: bytes) -> None:
    """确保所有字节都发送出去，处理 socket.send 可能只发送部分数据的情况"""
    view = memoryview(data)
    while view:
        n = sock.send(view)
        if n <= 0:
            raise ConnectionError("socket send failed")
        view = view[n:]

def _send_message(sock: socket.socket, header: dict, body: bytes) -> None:
    """封装并发送一条完整消息"""
    # 1. 序列化 Header
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    # 2. 发送 Header 长度 (4字节大端整数)
    _send_all(sock, struct.pack("!I", len(header_bytes)))
    # 3. 发送 Header 内容
    _send_all(sock, header_bytes)
    # 4. 发送 Body (加密后的 codes)
    _send_all(sock, body)

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """精确接收 n 个字节，如果连接断开则抛出异常"""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

def _recv_message(sock: socket.socket) -> Tuple[dict, bytes]:
    """接收并解析一条完整消息"""
    # 1. 读 Header 长度
    header_len = struct.unpack("!I", _recv_exact(sock, 4))[0]
    # 2. 读 Header JSON
    header = json.loads(_recv_exact(sock, header_len).decode("utf-8"))
    # 3. 根据 Header 中的 body_len 读 Body
    body_len = int(header["body_len"])
    body = _recv_exact(sock, body_len)
    return header, body

# ============================================================
# 麦克风采集模块
# ============================================================

@dataclass
class MicConfig:
    """麦克风配置参数"""
    device: Optional[int]         # 设备 ID
    mic_sr: Optional[float]       # 采样率
    channels: int = 1             # 声道数 (默认单声道)
    dtype: str = "float32"        # 数据类型
    frame_seconds: float = 0.02   # 每次回调的音频帧时长 (20ms)

class MicProducer:
    """
    音频生产者：
    使用 sounddevice 的 callback 机制将音频数据放入队列
    """
    def __init__(self, q: "queue.Queue[np.ndarray]", cfg: MicConfig):
        self.q = q
        self.cfg = cfg
        self.stream: Optional[sd.InputStream] = None

    def _callback(self, indata, frames, time_info, status):
        """音频硬件的中断回调，切记不能在此做耗时操作"""
        if status:
            print(f"[MIC] {status}", file=sys.stderr)
        
        # 确保只要第一个声道 (单声道处理)
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata[:, 0]
        else:
            mono = indata.reshape(-1)
            
        # 非阻塞入队，队列满了就丢弃旧帧 (保证实时性)
        try:
            self.q.put_nowait(mono.astype(np.float32, copy=True))
        except queue.Full:
            pass

    def start(self) -> float:
        """启动采集流"""
        # 如果未指定采样率，查询设备默认值
        if self.cfg.mic_sr is None:
            mic_sr = float