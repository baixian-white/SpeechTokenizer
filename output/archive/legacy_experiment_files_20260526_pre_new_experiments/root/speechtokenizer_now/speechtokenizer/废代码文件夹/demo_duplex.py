#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
demo_duplex.py (通用全双工终端：支持单机回环 & 双机局域网)

核心架构：
- 线程 1 (Sender): 采集麦克风 -> 编码 -> 加密 -> 发送给 peer_ip
- 线程 2 (Receiver): 绑定 0.0.0.0 -> 接收 -> 解密 -> 解码 -> 播放

使用场景：
1. 单机测试：直接运行，peer_ip 默认为 127.0.0.1
2. 双机通讯：运行时添加参数 --peer_ip 192.168.x.x (对方IP)
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

# 引入项目配置
from userA.common_config import PORT_A_SERVER, PORT_B_SERVER, KEY_FILENAME
from userA.encryption_utils import xor_decrypt_int_ndarray, xor_encrypt_int_ndarray

# ============================================================
# 网络协议封装
# ============================================================

def _send_all(sock: socket.socket, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = sock.send(view)
        if n <= 0:
            raise ConnectionError("socket send failed")
        view = view[n:]

def _send_message(sock: socket.socket, header: dict, body: bytes) -> None:
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    _send_all(sock, struct.pack("!I", len(header_bytes)))
    _send_all(sock, header_bytes)
    _send_all(sock, body)
    

def _recv_exact(sock: socket.socket, n: int) -> bytes:
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
    header_len = struct.unpack("!I", _recv_exact(sock, 4))[0]
    header = json.loads(_recv_exact(sock, header_len).decode("utf-8"))
    body_len = int(header["body_len"])
    body = _recv_exact(sock, body_len)
    return header, body

# ============================================================
# 音频采集与处理类
# ============================================================

@dataclass
class MicConfig:
    device: Optional[int]
    mic_sr: Optional[float]
    channels: int = 1
    dtype: str = "float32"
    frame_seconds: float = 0.02

class MicProducer:
    def __init__(self, q: "queue.Queue[np.ndarray]", cfg: MicConfig):
        self.q = q
        self.cfg = cfg
        self.stream: Optional[sd.InputStream] = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[MIC] {status}", file=sys.stderr)
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata[:, 0]
        else:
            mono = indata.reshape(-1)
        try:
            self.q.put_nowait(mono.astype(np.float32, copy=True))
        except queue.Full:
            pass

    def start(self) -> float:
        if self.cfg.mic_sr is None:
            mic_sr = float(sd.query_devices(self.cfg.device, "input")["default_samplerate"])
        else:
            mic_sr = float(self.cfg.mic_sr)
        blocksize = int(max(1, mic_sr * self.cfg.frame_seconds))
        sd.default.latency = ("low", "low")
        self.stream = sd.InputStream(
            samplerate=mic_sr, channels=self.cfg.channels, dtype=self.cfg.dtype,
            device=self.cfg.device, blocksize=blocksize, callback=self._callback,
        )
        self.stream.start()
        return mic_sr

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

# ============================================================
# 核心通讯节点 (Sender + Receiver)
# ============================================================

class DuplexNode:
    def __init__(self, args, model, device):
        self.args = args
        self.model = model
        self.device = device
        self.model_sr = int(getattr(model, "sample_rate", 16000))
        self.running = True

        # 端口分配逻辑
        if args.role == 'A':
            self.my_listen_port = PORT_A_SERVER
            self.target_port = PORT_B_SERVER
            print(f"=== Role A === Listen Port: {self.my_listen_port} | Target Port: {self.target_port}")
        else:
            self.my_listen_port = PORT_B_SERVER
            self.target_port = PORT_A_SERVER
            print(f"=== Role B === Listen Port: {self.my_listen_port} | Target Port: {self.target_port}")
        
        # 目标 IP (默认为 127.0.0.1，双机时为对方 IP)
        self.peer_ip = args.peer_ip
        print(f"=== Target IP: {self.peer_ip} ===")

    def run_sender_thread(self):
        """发送线程：Mic -> Encode -> Encrypt -> TCP Send"""
        print("[TX] Sender thread started.")
        
        mic_dev = None if self.args.mic_device == "None" else int(self.args.mic_device)
        q_mic = queue.Queue(maxsize=80)
        mic_cfg = MicConfig(
            device=mic_dev,
            mic_sr=None if self.args.mic_sr <= 0 else self.args.mic_sr,
            frame_seconds=self.args.frame_seconds
        )
        producer = MicProducer(q_mic, mic_cfg)
        
        try:
            mic_sr = producer.start()
            print(f"[TX] Mic active. sr={mic_sr}")
        except Exception as e:
            print(f"[TX] Failed to start mic: {e}")
            return

        rs_mic2model = torchaudio.transforms.Resample(orig_freq=mic_sr, new_freq=self.model_sr)
        
        # 连接重试循环
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connected = False
        while self.running and not connected:
            try:
                # 使用传入的 peer_ip 进行连接
                sock.connect((self.peer_ip, self.target_port))
                connected = True
                print(f"[TX] Connected to {self.peer_ip}:{self.target_port}!")
            except ConnectionRefusedError:
                time.sleep(2)
            except Exception as e:
                print(f"[TX] Connect error: {e}")
                time.sleep(2)

        if not self.running:
            producer.stop()
            return

        chunk_model_samples = int(self.args.chunk_seconds * self.model_sr)
        model_buf = []
        model_buf_len = 0

        try:
            while self.running:
                try:
                    mic_block = q_mic.get(timeout=0.5)
                except queue.Empty:
                    continue

                with torch.no_grad():
                    t = torch.from_numpy(mic_block).unsqueeze(0)
                    y = rs_mic2model(t).squeeze(0)
                rs = y.cpu().numpy()

                model_buf.append(rs)
                model_buf_len += rs.size

                while model_buf_len >= chunk_model_samples:
                    need = chunk_model_samples
                    pieces = []
                    while need > 0:
                        seg = model_buf[0]
                        take = min(need, seg.size)
                        pieces.append(seg[:take])
                        if take == seg.size:
                            model_buf.pop(0)
                        else:
                            model_buf[0] = seg[take:]
                        model_buf_len -= take
                        need -= take
                    
                    chunk = np.concatenate(pieces)

                    wav_t = torch.from_numpy(chunk).to(self.device).unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        codes_t = self.model.encode(wav_t)
                        if self.args.rvq_layers > 0:
                            codes_t = codes_t[:self.args.rvq_layers]
                    
                    codes = codes_t.cpu().numpy().astype(np.int64)
                    enc_codes = xor_encrypt_int_ndarray(codes, KEY_FILENAME)

                    header = {
                        "kind": "codes_v1",
                        "dtype": str(enc_codes.dtype),
                        "shape": list(enc_codes.shape),
                        "body_len": enc_codes.nbytes,
                    }
                    _send_message(sock, header, enc_codes.tobytes())
                    print(".", end="", flush=True)

        except Exception as e:
            print(f"[TX] Error: {e}")
        finally:
            producer.stop()
            sock.close()
            print("[TX] Stopped.")

    def run_receiver_thread(self):
        """接收线程：TCP Recv -> Decrypt -> Decode -> Play"""
        print("[RX] Receiver thread started.")
        
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            # 绑定 0.0.0.0 以支持局域网访问
            server.bind(("0.0.0.0", self.my_listen_port))
            server.listen(1)
        except Exception as e:
            print(f"[RX] Bind failed: {e}")
            return

        print(f"[RX] Listening on 0.0.0.0:{self.my_listen_port}...")
        
        server.settimeout(1.0)
        conn = None
        while self.running and conn is None:
            try:
                conn, addr = server.accept()
                print(f"[RX] Accepted from {addr}")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[RX] Accept error: {e}")
                return
        
        if not self.running:
            server.close()
            return

        conn.settimeout(None)

        spk_dev = None if self.args.spk_device == "None" else int(self.args.spk_device)
        try:
            info = sd.query_devices(spk_dev, "output")
            out_sr = int(round(float(info.get("default_samplerate", self.model_sr))))
        except:
            out_sr = self.model_sr

        rs_model2out = None
        if out_sr != self.model_sr:
            rs_model2out = torchaudio.transforms.Resample(orig_freq=self.model_sr, new_freq=out_sr)

        play_queue = queue.Queue(maxsize=32)
        play_buf = deque()
        play_buf_len = 0
        target_samples = max(1, int(float(self.args.target_lat) * out_sr))
        max_samples = max(target_samples + 1, int(float(self.args.max_lat) * out_sr))

        def _callback(outdata, frames, time_info, status):
            nonlocal play_buf_len
            while True:
                try:
                    seg = play_queue.get_nowait()
                    play_buf.append(seg)
                    play_buf_len += seg.size
                except queue.Empty:
                    break
            
            if play_buf_len > max_samples:
                drop = play_buf_len - target_samples
                while drop > 0 and play_buf_len > 0:
                    seg = play_buf[0]
                    if drop >= seg.size:
                        play_buf.popleft()
                        play_buf_len -= seg.size
                        drop -= seg.size
                    else:
                        play_buf[0] = seg[drop:]
                        play_buf_len -= drop
                        drop = 0
            
            out = np.zeros(frames, dtype=np.float32)
            filled = 0
            while filled < frames and play_buf_len > 0:
                seg = play_buf[0]
                take = min(frames - filled, seg.size)
                out[filled:filled+take] = seg[:take]
                if take == seg.size:
                    play_buf.popleft()
                else:
                    play_buf[0] = seg[take:]
                play_buf_len -= take
                filled += take
            
            if outdata.shape[1] == 1:
                outdata[:, 0] = out
            else:
                outdata[:, 0] = out
                outdata[:, 1] = out

        out_stream = sd.OutputStream(
            samplerate=out_sr, channels=2, callback=_callback,
            blocksize=int(out_sr * 0.02), device=spk_dev
        )
        out_stream.start()

        try:
            while self.running:
                header, body = _recv_message(conn)
                if header.get("kind") != "codes_v1": continue
                
                if self.args.drop_when_over and play_buf_len > max_samples:
                    continue

                dtype = np.dtype(header["dtype"])
                shape = tuple(header["shape"])
                enc_codes = np.frombuffer(body, dtype=dtype).reshape(shape)

                codes = xor_decrypt_int_ndarray(enc_codes, KEY_FILENAME).astype(np.int64)
                codes_t = torch.from_numpy(codes).to(self.device)
                
                with torch.no_grad():
                    wav_t = self.model.decode(codes_t)
                
                wav = wav_t.detach().cpu().squeeze(0).squeeze(0).float().numpy()

                if rs_model2out:
                    wav = rs_model2out(torch.from_numpy(wav).unsqueeze(0)).squeeze(0).numpy()
                
                wav = wav.astype(np.float32)
                
                try:
                    play_queue.put_nowait(wav)
                    print(">", end="", flush=True)
                except queue.Full:
                    pass

        except Exception as e:
            print(f"[RX] Error: {e}")
        finally:
            out_stream.stop()
            conn.close()
            server.close()
            print("[RX] Stopped.")

def main():
    parser = argparse.ArgumentParser(description="Full Duplex Secure Audio")
    parser.add_argument("--role", required=True, choices=["A", "B"], help="Identifies if this is User A or User B")
    
    # === 关键新增：默认连接本地，双机时修改此参数 ===
    parser.add_argument("--peer_ip", default="127.0.0.1", help="Target IP address (default: 127.0.0.1 for local test)")
    # =================================================
    
    parser.add_argument("--config_path", default="model_hub/speechtokenizer_hubert_avg/config.json")
    parser.add_argument("--ckpt_path", default="model_hub/speechtokenizer_hubert_avg/SpeechTokenizer.pt")
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--mic_device", default="None")
    parser.add_argument("--spk_device", default="None")
    parser.add_argument("--mic_sr", type=float, default=0.0)
    parser.add_argument("--frame_seconds", type=float, default=0.02)
    parser.add_argument("--chunk_seconds", type=float, default=0.20)
    parser.add_argument("--rvq_layers", type=int, default=3)
    parser.add_argument("--target_lat", type=float, default=0.25)
    parser.add_argument("--max_lat", type=float, default=0.80)
    parser.add_argument("--drop_when_over", action="store_true")

    args = parser.parse_args()

    print("Loading model...")
    model = SpeechTokenizer.load_from_checkpoint(args.config_path, args.ckpt_path)
    model.eval()
    
    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")

    node = DuplexNode(args, model, device)

    t_tx = threading.Thread(target=node.run_sender_thread)
    t_rx = threading.Thread(target=node.run_receiver_thread)

    t_tx.start()
    t_rx.start()

    print(f"\n=== Node {args.role} Running. Press Ctrl+C to stop ===\n")

    try:
        while True:
            time.sleep(1)
            if not t_tx.is_alive() or not t_rx.is_alive():
                print("One of the threads died, exiting...")
                break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        node.running = False
        t_tx.join()
        t_rx.join()
        print("Done.")

if __name__ == "__main__":
    main()