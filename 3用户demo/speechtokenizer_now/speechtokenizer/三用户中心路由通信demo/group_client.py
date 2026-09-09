#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import sys
import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

DEMO_DIR = Path(__file__).resolve().parent
PARENT_DIR = DEMO_DIR.parent
PROJECT_ROOT = DEMO_DIR.parents[2]
TWO_USER_DIR = PARENT_DIR / "两用户通信demo"
if (TWO_USER_DIR / "speechtokenizer").exists():
    sys.path.insert(0, str(TWO_USER_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio

from common_protocol import recv_message, send_message
from crypto_utils import xor_bytes
from channel_perturb import apply_packet_burst, CONDITIONS
from speechtokenizer import SpeechTokenizer
from speaker_identity import StreamingSpeakerIdentifier


# 默认指向本 bundle 内捆绑的本地训练模型（即论文 SCIT-Speech-LCA v2：
# Log/spt_base/config.json + SpeechTokenizer_best_dev.pt）。
# 若该目录不存在，回退到两用户 demo 的 model_hub（仅当其确实存在时）。
BUNDLED_MODEL_DIR = PROJECT_ROOT / "Log" / "spt_base"
LEGACY_MODEL_DIR = TWO_USER_DIR / "model_hub" / "speechtokenizer_hubert_avg"
if (BUNDLED_MODEL_DIR / "config.json").exists():
    DEFAULT_CONFIG_PATH = BUNDLED_MODEL_DIR / "config.json"
    DEFAULT_CKPT_PATH = BUNDLED_MODEL_DIR / "SpeechTokenizer_best_dev.pt"
else:
    DEFAULT_CONFIG_PATH = LEGACY_MODEL_DIR / "config.json"
    DEFAULT_CKPT_PATH = LEGACY_MODEL_DIR / "SpeechTokenizer.pt"
DEFAULT_KEY_PATH = TWO_USER_DIR / "userA" / "46.txt"


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def parse_device(value: str) -> Optional[int]:
    return None if str(value) == "None" else int(value)


def get_output_channels(device_index: Optional[int]) -> int:
    try:
        info = sd.query_devices(device_index, "output")
        channels = int(info.get("max_output_channels", 0))
        if channels >= 2:
            return 2
        if channels >= 1:
            return 1
    except Exception:
        return 1
    return 0


def packet_wire_size(header: dict, body: bytes) -> int:
    header_with_len = dict(header)
    header_with_len["body_len"] = len(body)
    header_bytes = json.dumps(header_with_len, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return 4 + len(header_bytes) + len(body)


def mean_value(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def median_value(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def percentile_value(values: list[float], percent: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * percent / 100.0
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    weight = pos - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def max_value(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(max(values))


def fmt_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}ms"


def fmt_num(value: Optional[float], digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_sec(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}s"


def fmt_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _round_or_none(value: Optional[float], digits: int = 1):
    return None if value is None else round(float(value), digits)


def pack_indices_bits(codes: np.ndarray, bits: int) -> bytes:
    """把整数索引数组按每个 bits 位 MSB-first 紧凑打包成字节流。

    对应 §3.1 的名义码率 R(L)=L·f_q·⌈log₂K⌉：K=1024 → bits=10，
    每索引精确占 10 bit（而非 uint16 的 16 bit），body 字节数 = ceil(n_values·bits/8)。
    """
    flat = np.ascontiguousarray(codes).reshape(-1).astype(np.uint16, copy=False)
    if flat.size == 0:
        return b""
    # 取每个值的低 bits 位，MSB-first 展开成位平面
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint16)
    bitplane = ((flat[:, None] >> shifts) & 1).astype(np.uint8)  # (n_values, bits)
    return np.packbits(bitplane.reshape(-1)).tobytes()


def unpack_indices_bits(body: bytes, bits: int, n_values: int) -> np.ndarray:
    """pack_indices_bits 的逆：从字节流还原 n_values 个 bits 位索引（一维）。"""
    if n_values == 0:
        return np.zeros(0, dtype=np.int64)
    all_bits = np.unpackbits(np.frombuffer(body, dtype=np.uint8))
    used = all_bits[: n_values * bits].reshape(n_values, bits).astype(np.uint16)
    weights = (np.uint16(1) << np.arange(bits - 1, -1, -1, dtype=np.uint16))
    return (used * weights).sum(axis=1).astype(np.int64)


@dataclass
class MicConfig:
    device: Optional[int]
    mic_sr: Optional[float]
    frame_seconds: float = 0.02
    channels: int = 1
    dtype: str = "float32"


class MicProducer:
    def __init__(self, out_queue: "queue.Queue[np.ndarray]", cfg: MicConfig):
        self.out_queue = out_queue
        self.cfg = cfg
        self.stream: Optional[sd.InputStream] = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[MIC] status: {status}", file=sys.stderr)
        if indata.ndim == 2 and indata.shape[1] > 1:
            mono = indata[:, 0].copy()
        else:
            mono = indata.reshape(-1).copy()
        try:
            self.out_queue.put_nowait(mono.astype(np.float32, copy=False))
        except queue.Full:
            pass

    def start(self) -> float:
        if self.cfg.mic_sr is None:
            mic_sr = float(sd.query_devices(self.cfg.device, "input")["default_samplerate"])
        else:
            mic_sr = float(self.cfg.mic_sr)
        blocksize = max(1, int(mic_sr * self.cfg.frame_seconds))
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
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        print("[MIC] stopped")


@dataclass
class IncomingStream:
    sender_id: str
    packet_queue: "queue.Queue[tuple[dict, bytes]]" = field(default_factory=lambda: queue.Queue(maxsize=16))
    jitter: deque = field(default_factory=deque)
    jitter_len: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    packets: int = 0
    drops: int = 0
    decoded: int = 0
    last_seq: Optional[int] = None
    last_rms: float = 0.0
    last_decode_ms: float = 0.0
    recv_body_bytes: int = 0
    recv_total_bytes: int = 0
    decode_ms_samples: list[float] = field(default_factory=list)
    jitter_sec_samples: list[float] = field(default_factory=list)
    last_speaker_pred: str = ""
    last_speaker_score: float = float("nan")
    last_speaker_margin: float = float("nan")
    last_speaker_verified: bool = False
    speaker_eval_count: int = 0
    speaker_correct_count: int = 0
    speaker_verified_count: int = 0


class GroupClient:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.user_id = args.user_id
        self.room_id = args.room_id
        self.running = threading.Event()
        self.running.set()
        self.sock: Optional[socket.socket] = None
        self.send_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.streams: Dict[str, IncomingStream] = {}
        self.streams_lock = threading.Lock()
        self.seq_id = 0
        self.start_time = time.time()
        self.stats_lock = threading.Lock()
        self.sent_packets = 0
        self.recv_packets = 0
        self.encode_ms = 0.0
        self.sent_body_bytes = 0
        self.sent_total_bytes = 0
        self.recv_body_bytes = 0
        self.recv_total_bytes = 0
        self.encode_ms_samples: list[float] = []
        self.model = None
        self.device = torch.device("cpu")
        self.model_sr = 16000
        self.out_sr = 16000
        self.rs_model2out = None
        self.output_stream: Optional[sd.OutputStream] = None
        self.out_channels = 1
        self.mic: Optional[MicProducer] = None
        self.codebook_size = 1024
        self.index_bits = 10
        self._channel_cond = CONDITIONS.get(getattr(args, "channel", "clean"), CONDITIONS["clean"])
        self.speaker_identifier: Optional[StreamingSpeakerIdentifier] = None

    def load_model(self):
        print("[MODEL] loading ...")
        self.model = self.load_checkpoint_model(self.args.config_path, self.args.ckpt_path)
        self.model.eval()
        use_cuda = self.args.device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.model.to(self.device)
        self.model_sr = int(getattr(self.model, "sample_rate", 16000))
        try:
            with open(self.args.config_path, "r", encoding="utf-8") as f:
                self.codebook_size = int(json.load(f).get("codebook_size", 1024))
        except Exception:
            self.codebook_size = 1024
        self.index_bits = max(1, (self.codebook_size - 1).bit_length())
        print(f"[MODEL] ready sr={self.model_sr} device={self.device} "
              f"codebook_size={self.codebook_size} index_bits={self.index_bits}")

    def init_speaker_identifier(self):
        if not getattr(self.args, "speaker_id_enable", False):
            return
        profile_dir = str(getattr(self.args, "speaker_profile_dir", "") or "").strip()
        if not profile_dir:
            print("[SPKID] disabled: --speaker_profile_dir is empty")
            return
        sample_rate = self.out_sr if self.rs_model2out is not None else self.model_sr
        speaker_device = str(getattr(self.args, "speaker_device", "cpu"))
        if speaker_device == "auto":
            speaker_device = "cuda" if self.device.type == "cuda" and torch.cuda.is_available() else "cpu"
        self.speaker_identifier = StreamingSpeakerIdentifier(
            profile_dir=profile_dir,
            sample_rate=sample_rate,
            window_sec=float(self.args.speaker_window_sec),
            hop_sec=float(self.args.speaker_hop_sec),
            threshold=float(self.args.speaker_threshold),
            n_mfcc=int(self.args.speaker_n_mfcc),
            backend=str(self.args.speaker_backend),
            speaker_device=speaker_device,
            ecapa_source=str(self.args.ecapa_source),
            ecapa_savedir=str(self.args.ecapa_savedir),
        )
        if not self.speaker_identifier.enabled:
            print(f"[SPKID] disabled: no profile wav/flac found under {profile_dir}")
            self.speaker_identifier = None
            return
        print(
            f"[SPKID] enabled speakers={self.speaker_identifier.speaker_count} "
            f"backend={self.args.speaker_backend} device={speaker_device} "
            f"sr={sample_rate} window={self.args.speaker_window_sec:.2f}s "
            f"hop={self.args.speaker_hop_sec:.2f}s threshold={self.args.speaker_threshold:.2f}"
        )

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[TCP] connecting to {self.args.router_ip}:{self.args.router_port} ...")
        self.sock.connect((self.args.router_ip, self.args.router_port))
        hello = {
            "kind": "hello_v1",
            "room_id": self.room_id,
            "user_id": self.user_id,
            "timestamp_ms": int(time.time() * 1000),
        }
        send_message(self.sock, hello, b"")
        print("[TCP] connected and registered")

    def get_stream(self, sender_id: str) -> IncomingStream:
        with self.streams_lock:
            stream = self.streams.get(sender_id)
            if stream is None:
                stream = IncomingStream(sender_id=sender_id)
                self.streams[sender_id] = stream
                t = threading.Thread(target=self.decode_loop, args=(stream,), daemon=True)
                t.start()
                print(f"[STREAM] created from_{sender_id}")
            return stream

    def elapsed_sec(self) -> float:
        return max(1e-6, time.time() - self.start_time)

    def chunk_ms(self) -> float:
        return max(1e-6, float(self.args.chunk_seconds) * 1000.0)

    def encode_rtf(self, encode_ms: float) -> float:
        return encode_ms / self.chunk_ms()

    def bitrate_kbps(self, byte_count: int) -> float:
        return byte_count * 8.0 / self.elapsed_sec() / 1000.0

    def load_checkpoint_model(self, config_path: str, ckpt_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        handoff = cfg.get("exp1_handoff", {})
        if handoff.get("handoff_schema") != "encoder_only_nas_v1":
            return SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)
        return self.load_encoder_only_nas_model(cfg, ckpt_path, config_path)

    def resolve_nas_config_path(self, raw_path: str, config_path: str) -> Optional[Path]:
        if not raw_path:
            return None
        raw = Path(raw_path)
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            cfg_dir = Path(config_path).resolve().parent
            candidates.append(cfg_dir / raw)
            candidates.append(PROJECT_ROOT / raw)
            candidates.append(PROJECT_ROOT / raw.name)
            run_name = next((part for part in raw.parts if part.startswith("exp3_low_load_channel_aware_adaptation_v2_strong")), None)
            if run_name:
                candidates.append(PROJECT_ROOT / run_name / "configs" / raw.name)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        matches = list(PROJECT_ROOT.glob(f"**/configs/{raw.name}"))
        return matches[0] if matches else None

    def load_nas_encoder_config(self, cfg: dict, config_path: str) -> tuple[Optional[dict], Optional[Path]]:
        handoff = cfg.get("exp1_handoff", {})
        for raw_path in (cfg.get("nas_encoder_config"), handoff.get("config_path"), handoff.get("source_path")):
            path = self.resolve_nas_config_path(str(raw_path or ""), config_path)
            if path is None:
                continue
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path
        return None, None
    def infer_nas_encoder_ops(self, params: dict, ratios: list[int], n_residual_layers: int) -> tuple[list[str], list[bool]]:
        block_indices = []
        module_idx = 1
        for _ in ratios:
            for _ in range(n_residual_layers):
                block_indices.append(module_idx)
                module_idx += 1
            module_idx += 2

        ops = []
        use_se = []
        for idx in block_indices:
            prefix = f"encoder.model.{idx}."
            keys = [k for k in params if k.startswith(prefix)]
            use_se.append(any(".se." in k for k in keys))
            if not keys:
                ops.append("skip")
                continue
            depthwise_key = next((k for k in keys if ".op.depthwise.conv.conv.weight_v" in k), None)
            if depthwise_key is not None:
                kernel = int(params[depthwise_key].shape[-1])
                ops.append(f"sep_k{kernel}")
                continue
            conv_key = next((k for k in keys if ".op.conv.conv.weight_v" in k), None)
            if conv_key is not None:
                kernel = int(params[conv_key].shape[-1])
                ops.append(f"std_k{kernel}")
                continue
            raise RuntimeError(f"cannot infer NAS op for encoder block index={idx}")
        return ops, use_se

    def infer_encoder_lstm_layers(self, params: dict) -> int:
        layers = set()
        for key in params:
            marker = "encoder.model.13.lstm.weight_ih_l"
            if key.startswith(marker) and "reverse" not in key:
                suffix = key[len(marker) :]
                if suffix.isdigit():
                    layers.add(int(suffix))
        return max(layers) + 1 if layers else int(self.args.rvq_layers > 0)

    def load_encoder_only_nas_model(self, cfg: dict, ckpt_path: str, config_path: str):
        from nas.SeaNet import SEANetEncoder

        params = torch.load(ckpt_path, map_location="cpu")
        encoder_n_filters = int(params["encoder.model.0.conv.conv.bias"].shape[0])
        handoff = cfg.get("exp1_handoff", {})
        nas_cfg, nas_cfg_path = self.load_nas_encoder_config(cfg, config_path)
        encoder_strides = list(handoff.get("encoder_strides") or [5, 4, 4, 4])
        encoder_ratios = list(reversed([int(v) for v in encoder_strides]))
        n_residual_layers = int(cfg.get("n_residual_layers", 1))
        encoder_lstm_layers = self.infer_encoder_lstm_layers(params)
        encoder_bidirectional = any(k.startswith("encoder.") and k.endswith("_reverse") for k in params)
        encoder_activation = "Snake" if any(k.startswith("encoder.") and k.endswith(".alpha") for k in params) else cfg.get("activation", "ELU")
        inferred_ops, inferred_se = self.infer_nas_encoder_ops(params, encoder_ratios, n_residual_layers)
        layer_ops, layer_se = inferred_ops, inferred_se
        if nas_cfg is not None:
            encoder_n_filters = int(nas_cfg.get("n_filters", encoder_n_filters))
            encoder_strides = list(nas_cfg.get("encoder_strides") or encoder_strides)
            encoder_ratios = list(reversed([int(v) for v in encoder_strides]))
            encoder_lstm_layers = int(nas_cfg.get("lstm", encoder_lstm_layers))
            encoder_activation = str(nas_cfg.get("activation", encoder_activation))
            layer_ops = list(nas_cfg.get("layer_ops_list") or layer_ops)
            layer_se = list(nas_cfg.get("layer_se_list") or layer_se)
        expected_blocks = len(encoder_ratios) * n_residual_layers
        if len(layer_ops) != expected_blocks or len(layer_se) != expected_blocks:
            raise RuntimeError(
                f"NAS config block count mismatch: ops={len(layer_ops)} se={len(layer_se)} expected={expected_blocks}"
            )

        model = SpeechTokenizer(cfg)
        model.encoder = SEANetEncoder(
            n_filters=encoder_n_filters,
            dimension=int(cfg.get("dimension", 1024)),
            ratios=encoder_ratios,
            lstm=encoder_lstm_layers,
            bidirectional=encoder_bidirectional,
            dilation_base=int(cfg.get("dilation_base", 2)),
            residual_kernel_size=int(cfg.get("residual_kernel_size", 3)),
            n_residual_layers=n_residual_layers,
            activation=encoder_activation,
            norm="weight_norm",
            causal=False,
            pad_mode="reflect",
            layer_ops_list=layer_ops,
            layer_se_list=layer_se,
        )
        model.load_state_dict(params)
        print(
            "[MODEL] encoder_only_nas_v1 "
            f"encoder_n_filters={encoder_n_filters} encoder_strides={encoder_strides} "
            f"ops={layer_ops} decoder_n_filters={cfg.get('n_filters')} "
            f"nas_config={nas_cfg_path if nas_cfg_path is not None else 'inferred'}"
        )
        return model

    def send_codes(self, codes: np.ndarray):
        # 紧凑封装（与论文 §3.1 名义码率 R(L)=L·f_q·⌈log₂K⌉ 逐位对齐）：
        # 每个 RVQ 索引精确占 ⌈log₂K⌉=10 bit（K=1024），位打包成字节流后逐字节 XOR。
        # body 字节数 = ceil(n_values·index_bits/8)，实测 body 码率即名义 500L（仅末字节补零的
        # 取整误差）；total 含 header 开销 > 500L，与 §3.1 "payload 上界 + 协议栈开销" 一致。
        if codes.max(initial=0) >= self.codebook_size:
            raise ValueError(f"RVQ index exceeds codebook_size={self.codebook_size}: max={int(codes.max())}")
        n_values = int(codes.size)
        packed = pack_indices_bits(codes, self.index_bits)
        body = xor_bytes(packed, self.args.key_path)
        header = {
            "kind": "codes_v1",
            "room_id": self.room_id,
            "sender_id": self.user_id,
            "seq_id": self.seq_id,
            "timestamp_ms": int(time.time() * 1000),
            "rvq_layers": int(codes.shape[0]),
            "packing": "bitpack_v1",
            "bits": int(self.index_bits),
            "n_values": n_values,
            "shape": list(codes.shape),
        }
        self.seq_id += 1
        if self.sock is None:
            return
        total_bytes = packet_wire_size(header, body)
        with self.send_lock:
            send_message(self.sock, header, body)
        with self.stats_lock:
            self.sent_packets += 1
            self.sent_body_bytes += len(body)
            self.sent_total_bytes += total_bytes

    def encode_and_send_chunk(self, chunk_model: np.ndarray):
        if chunk_model.size == 0 or self.model is None:
            return
        wav_t = torch.from_numpy(chunk_model).to(self.device).float().unsqueeze(0).unsqueeze(0)
        use_amp = self.device.type == "cuda"
        amp_ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
        t0 = time.perf_counter()
        with torch.no_grad(), amp_ctx, self.model_lock:
            codes_t = self.model.encode(wav_t)
            if self.args.rvq_layers > 0:
                codes_t = codes_t[: self.args.rvq_layers]
        encode_ms = (time.perf_counter() - t0) * 1000.0
        with self.stats_lock:
            self.encode_ms = encode_ms
            self.encode_ms_samples.append(encode_ms)
        codes = codes_t.detach().cpu().numpy().astype(np.int64, copy=False)
        self.send_codes(codes)

    def mic_send_loop(self):
        if self.args.no_mic and not self.args.wav_input:
            print("[SEND] no_mic enabled; this client only receives")
            return
        if self.args.wav_input:
            self.wav_send_loop()
            return

        q_mic: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=80)
        mic_dev = parse_device(self.args.mic_device)
        cfg = MicConfig(
            device=mic_dev,
            mic_sr=None if self.args.mic_sr <= 0 else self.args.mic_sr,
            frame_seconds=self.args.frame_seconds,
        )
        self.mic = MicProducer(q_mic, cfg)
        mic_sr = self.mic.start()
        rs_mic2model = torchaudio.transforms.Resample(orig_freq=mic_sr, new_freq=self.model_sr)
        chunk_samples = max(1, int(self.args.chunk_seconds * self.model_sr))
        parts = deque()
        parts_len = 0

        try:
            while self.running.is_set():
                try:
                    mic_block = q_mic.get(timeout=0.1)
                except queue.Empty:
                    continue
                with torch.no_grad():
                    rs = rs_mic2model(torch.from_numpy(mic_block).float().unsqueeze(0)).squeeze(0).numpy()
                parts.append(rs.astype(np.float32, copy=False))
                parts_len += rs.size
                while parts_len >= chunk_samples:
                    need = chunk_samples
                    chunk_parts = []
                    while need > 0:
                        seg = parts[0]
                        take = min(need, seg.size)
                        chunk_parts.append(seg[:take])
                        if take == seg.size:
                            parts.popleft()
                        else:
                            parts[0] = seg[take:]
                        parts_len -= take
                        need -= take
                    self.encode_and_send_chunk(np.concatenate(chunk_parts).astype(np.float32, copy=False))
        finally:
            if self.mic is not None:
                self.mic.stop()

    def wav_send_loop(self):
        wav_path = Path(self.args.wav_input)
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(audio.T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.model_sr:
            wav = torchaudio.functional.resample(wav, sr, self.model_sr)
        audio = wav.squeeze(0).numpy().astype(np.float32, copy=False)
        chunk_samples = max(1, int(self.args.chunk_seconds * self.model_sr))
        print(f"[WAV] sending {wav_path} sr={self.model_sr} chunk={chunk_samples}")
        while self.running.is_set():
            for start in range(0, len(audio), chunk_samples):
                if not self.running.is_set():
                    break
                chunk = audio[start : start + chunk_samples]
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                t0 = time.perf_counter()
                self.encode_and_send_chunk(chunk)
                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, self.args.chunk_seconds - elapsed))
            if not self.args.loop_wav:
                break
        print("[WAV] send loop ended")

    def recv_loop(self):
        assert self.sock is not None
        try:
            while self.running.is_set():
                header, body = recv_message(self.sock)
                if header.get("kind") != "codes_v1":
                    continue
                sender_id = str(header.get("sender_id", ""))
                if sender_id == self.user_id or not sender_id:
                    continue
                stream = self.get_stream(sender_id)
                total_bytes = packet_wire_size(header, body)
                if stream.packet_queue.full():
                    try:
                        stream.packet_queue.get_nowait()
                        with stream.lock:
                            stream.drops += 1
                    except queue.Empty:
                        pass
                stream.packet_queue.put_nowait((header, body))
                with stream.lock:
                    stream.packets += 1
                    stream.last_seq = header.get("seq_id")
                    stream.recv_body_bytes += len(body)
                    stream.recv_total_bytes += total_bytes
                with self.stats_lock:
                    self.recv_packets += 1
                    self.recv_body_bytes += len(body)
                    self.recv_total_bytes += total_bytes
        except Exception as exc:
            if self.running.is_set():
                print(f"[RECV] stopped: {exc}")
            self.running.clear()

    def decode_loop(self, stream: IncomingStream):
        while self.running.is_set():
            try:
                header, body = stream.packet_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                shape = tuple(header["shape"])
                bits = int(header.get("bits", self.index_bits))
                n_values = int(header.get("n_values", int(np.prod(shape))))
                packed = xor_bytes(body, self.args.key_path)
                codes = unpack_indices_bits(packed, bits, n_values).reshape(shape)
                codes_t = torch.from_numpy(codes).to(self.device)
                # 信道索引扰动（与论文 §5.4 逐行一致）：在 decode 前按 --channel 施加
                # packet/burst 丢失 + previous-index 拼接，确定性种子由 (sender, seq) 派生，
                # 在真实链路上精确重放论文丢包条件（默认 clean，不扰动）。
                if self._channel_cond is not None and self._channel_cond["name"] != "clean":
                    seq_id = int(header.get("seq_id", 0))
                    seed = (hash((stream.sender_id, seq_id)) ^ self.args.channel_seed) & 0x7FFFFFFF
                    gen = torch.Generator(device=self.device)
                    gen.manual_seed(seed)
                    codes_t, _ = apply_packet_burst(codes_t, self._channel_cond, gen)
                use_amp = self.device.type == "cuda"
                amp_ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()
                t0 = time.perf_counter()
                with torch.no_grad(), amp_ctx, self.model_lock:
                    wav_t = self.model.decode(codes_t)
                decode_ms = (time.perf_counter() - t0) * 1000.0
                wav = wav_t.detach().cpu().squeeze(0).squeeze(0).to(torch.float32)
                if self.rs_model2out is not None:
                    with torch.no_grad():
                        wav = self.rs_model2out(wav.unsqueeze(0)).squeeze(0)
                pcm = wav.numpy().astype(np.float32, copy=False)
                self.update_speaker_identity(stream, pcm)
                pcm_rms = rms(pcm)
                jitter_sec = self.add_pcm_to_jitter(stream, pcm)
                with stream.lock:
                    stream.last_decode_ms = decode_ms
                    stream.last_rms = pcm_rms
                    stream.decoded += 1
                    stream.decode_ms_samples.append(decode_ms)
                    stream.jitter_sec_samples.append(jitter_sec)
            except Exception as exc:
                print(f"[DECODE:{stream.sender_id}] failed: {exc}", file=sys.stderr)

    def update_speaker_identity(self, stream: IncomingStream, pcm: np.ndarray) -> None:
        if self.speaker_identifier is None:
            return
        try:
            prediction = self.speaker_identifier.update(stream.sender_id, pcm)
        except Exception as exc:
            print(f"[SPKID:{stream.sender_id}] failed: {exc}", file=sys.stderr)
            return
        if prediction is None:
            return
        correct = prediction.predicted_speaker == stream.sender_id
        with stream.lock:
            stream.last_speaker_pred = prediction.predicted_speaker
            stream.last_speaker_score = prediction.score
            stream.last_speaker_margin = prediction.margin
            stream.last_speaker_verified = prediction.verified
            stream.speaker_eval_count += 1
            if correct:
                stream.speaker_correct_count += 1
            if prediction.verified:
                stream.speaker_verified_count += 1

    def add_pcm_to_jitter(self, stream: IncomingStream, pcm: np.ndarray) -> float:
        max_samples = max(1, int(self.args.max_lat * self.out_sr))
        target_samples = max(1, int(self.args.target_lat * self.out_sr))
        with stream.lock:
            stream.jitter.append(pcm)
            stream.jitter_len += pcm.size
            if stream.jitter_len > max_samples:
                drop = stream.jitter_len - target_samples
                while drop > 0 and stream.jitter:
                    seg = stream.jitter[0]
                    take = min(drop, seg.size)
                    if take == seg.size:
                        stream.jitter.popleft()
                    else:
                        stream.jitter[0] = seg[take:]
                    stream.jitter_len -= take
                    drop -= take
            return stream.jitter_len / float(self.out_sr)

    def pop_pcm(self, stream: IncomingStream, frames: int) -> np.ndarray:
        out = np.zeros(frames, dtype=np.float32)
        filled = 0
        with stream.lock:
            while filled < frames and stream.jitter_len > 0:
                seg = stream.jitter[0]
                take = min(frames - filled, seg.size)
                out[filled : filled + take] = seg[:take]
                if take == seg.size:
                    stream.jitter.popleft()
                else:
                    stream.jitter[0] = seg[take:]
                stream.jitter_len -= take
                filled += take
        return out

    def play_callback(self, outdata, frames, time_info, status):
        if status:
            pass
        with self.streams_lock:
            streams = list(self.streams.values())
        active = []
        for stream in streams:
            pcm = self.pop_pcm(stream, frames)
            if np.any(pcm):
                active.append(pcm)
        if active:
            mixed = np.sum(active, axis=0) / max(1, len(active))
            mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32, copy=False)
        else:
            mixed = np.zeros(frames, dtype=np.float32)

        if self.out_channels == 1:
            outdata[:, 0] = mixed
        else:
            outdata[:, 0] = mixed
            outdata[:, 1] = mixed

    def start_playback(self):
        if self.args.no_play:
            print("[PLAY] no_play enabled; decoded audio will not open speaker")
            return
        spk_dev = parse_device(self.args.spk_device)
        if self.args.out_sr > 0:
            self.out_sr = int(self.args.out_sr)
        else:
            try:
                info = sd.query_devices(spk_dev, "output")
                self.out_sr = int(round(float(info.get("default_samplerate", self.model_sr))))
            except Exception:
                self.out_sr = self.model_sr
        if self.out_sr != self.model_sr:
            self.rs_model2out = torchaudio.transforms.Resample(orig_freq=self.model_sr, new_freq=self.out_sr)
        self.out_channels = get_output_channels(spk_dev)
        if self.out_channels <= 0:
            raise RuntimeError(f"selected output device has no output channels: {spk_dev}")
        blocksize = max(1, int(self.out_sr * self.args.frame_out_seconds))
        sd.default.latency = ("low", "low")
        self.output_stream = sd.OutputStream(
            samplerate=self.out_sr,
            channels=self.out_channels,
            dtype="float32",
            device=spk_dev,
            blocksize=blocksize,
            callback=self.play_callback,
        )
        self.output_stream.start()
        print(f"[PLAY] started device={spk_dev} sr={self.out_sr} channels={self.out_channels} block={blocksize}")

    def monitor_loop(self):
        while self.running.is_set():
            time.sleep(1.0)
            if not self.args.monitor:
                continue
            with self.streams_lock:
                streams = list(self.streams.values())
            with self.stats_lock:
                sent_packets = self.sent_packets
                recv_packets = self.recv_packets
                encode_ms = self.encode_ms
                body_bytes = self.sent_body_bytes + self.recv_body_bytes
                total_bytes = self.sent_total_bytes + self.recv_total_bytes
            enc_rtf = self.encode_rtf(encode_ms)
            body_kbps = self.bitrate_kbps(body_bytes)
            total_kbps = self.bitrate_kbps(total_bytes)
            details = []
            for s in streams:
                with s.lock:
                    jitter_sec = s.jitter_len / float(self.out_sr)
                    packets = s.packets
                    drops = s.drops
                    last_seq = s.last_seq
                    last_rms = s.last_rms
                    last_decode_ms = s.last_decode_ms
                    spk_pred = s.last_speaker_pred
                    spk_score = s.last_speaker_score
                    spk_margin = s.last_speaker_margin
                    spk_verified = s.last_speaker_verified
                    spk_eval = s.speaker_eval_count
                    spk_correct = s.speaker_correct_count
                drop_rate = drops / max(1, packets + drops)
                dec_rtf = last_decode_ms / self.chunk_ms()
                spk_detail = ""
                if self.speaker_identifier is not None:
                    spk_acc = spk_correct / max(1, spk_eval)
                    spk_detail = (
                        f" spk={spk_pred or 'n/a'} score={fmt_num(spk_score, 3)} "
                        f"margin={fmt_num(spk_margin, 3)} verified={'yes' if spk_verified else 'no'} "
                        f"spk_acc={fmt_pct(spk_acc)}"
                    )
                details.append(
                    f"from_{s.sender_id}:seq={last_seq} q={s.packet_queue.qsize()} "
                    f"jit={jitter_sec:.3f}s rms={last_rms:.4f} dec={last_decode_ms:.1f}ms "
                    f"dec_rtf={dec_rtf:.2f} drop={drops} drop_rate={fmt_pct(drop_rate)}"
                    f"{spk_detail}"
                )
            print(
                f"[MON:{self.user_id}] sent={sent_packets} recv={recv_packets} "
                f"enc={encode_ms:.1f}ms enc_rtf={enc_rtf:.2f} "
                f"body={body_kbps:.1f}kbps total={total_kbps:.1f}kbps | "
                + ("; ".join(details) if details else "no streams")
            )

    def print_summary(self):
        duration = self.elapsed_sec()
        with self.stats_lock:
            sent_packets = self.sent_packets
            recv_packets = self.recv_packets
            sent_body = self.sent_body_bytes
            recv_body = self.recv_body_bytes
            sent_total = self.sent_total_bytes
            recv_total = self.recv_total_bytes
            encode_samples = list(self.encode_ms_samples)
        with self.streams_lock:
            streams = list(self.streams.values())

        decode_samples: list[float] = []
        jitter_samples: list[float] = []
        decoded_packets = 0
        dropped_packets = 0
        per_stream = []
        for stream in streams:
            with stream.lock:
                stream_decode = list(stream.decode_ms_samples)
                stream_jitter = list(stream.jitter_sec_samples)
                stream_packets = stream.packets
                stream_drops = stream.drops
                stream_decoded = stream.decoded
                stream_body = stream.recv_body_bytes
                stream_total = stream.recv_total_bytes
                spk_eval = stream.speaker_eval_count
                spk_correct = stream.speaker_correct_count
                spk_verified = stream.speaker_verified_count
                spk_pred = stream.last_speaker_pred
                spk_score = stream.last_speaker_score
                spk_margin = stream.last_speaker_margin
                spk_last_verified = stream.last_speaker_verified
            decode_samples.extend(stream_decode)
            jitter_samples.extend(stream_jitter)
            decoded_packets += stream_decoded
            dropped_packets += stream_drops
            stream_drop_rate = stream_drops / max(1, stream_packets + stream_drops)
            speaker_suffix = ""
            if self.speaker_identifier is not None:
                speaker_suffix = (
                    f" speaker_eval={spk_eval} speaker_acc={fmt_pct(spk_correct / max(1, spk_eval))} "
                    f"speaker_verified={spk_verified} last_speaker={spk_pred or 'n/a'} "
                    f"score={fmt_num(spk_score, 3)} margin={fmt_num(spk_margin, 3)} "
                    f"verified={'yes' if spk_last_verified else 'no'}"
                )
            per_stream.append(
                f"  from_{stream.sender_id}: packets={stream_packets} decoded={stream_decoded} "
                f"drops={stream_drops} drop_rate={fmt_pct(stream_drop_rate)} "
                f"body={stream_body * 8.0 / duration / 1000.0:.1f}kbps "
                f"total={stream_total * 8.0 / duration / 1000.0:.1f}kbps"
                f"{speaker_suffix}"
            )

        total_body = sent_body + recv_body
        total_wire = sent_total + recv_total
        drop_rate = dropped_packets / max(1, recv_packets + dropped_packets)
        encode_rtf_samples = [v / self.chunk_ms() for v in encode_samples]
        decode_rtf_samples = [v / self.chunk_ms() for v in decode_samples]

        print(f"\n[SUMMARY:{self.user_id}] duration={duration:.1f}s")
        print(
            f"[SUMMARY:{self.user_id}] sent={sent_packets} recv={recv_packets} "
            f"decoded={decoded_packets} drops={dropped_packets} drop_rate={fmt_pct(drop_rate)}"
        )
        print(
            f"[SUMMARY:{self.user_id}] body={total_body * 8.0 / duration / 1000.0:.1f}kbps "
            f"total={total_wire * 8.0 / duration / 1000.0:.1f}kbps"
        )
        print(
            f"[SUMMARY:{self.user_id}] encode_ms mean={fmt_ms(mean_value(encode_samples))} "
            f"median={fmt_ms(median_value(encode_samples))} "
            f"p95={fmt_ms(percentile_value(encode_samples, 95))} max={fmt_ms(max_value(encode_samples))}"
        )
        print(
            f"[SUMMARY:{self.user_id}] decode_ms mean={fmt_ms(mean_value(decode_samples))} "
            f"median={fmt_ms(median_value(decode_samples))} "
            f"p95={fmt_ms(percentile_value(decode_samples, 95))} max={fmt_ms(max_value(decode_samples))}"
        )
        print(
            f"[SUMMARY:{self.user_id}] encode_rtf mean={fmt_num(mean_value(encode_rtf_samples))} "
            f"p95={fmt_num(percentile_value(encode_rtf_samples, 95))}"
        )
        print(
            f"[SUMMARY:{self.user_id}] decode_rtf mean={fmt_num(mean_value(decode_rtf_samples))} "
            f"p95={fmt_num(percentile_value(decode_rtf_samples, 95))}"
        )
        print(
            f"[SUMMARY:{self.user_id}] jitter mean={fmt_sec(mean_value(jitter_samples))} "
            f"p95={fmt_sec(percentile_value(jitter_samples, 95))} max={fmt_sec(max_value(jitter_samples))}"
        )
        if per_stream:
            print(f"[SUMMARY:{self.user_id}] per_stream:")
            for line in per_stream:
                print(line)

        if self.args.summary_csv:
            self.write_summary_csv(
                duration=duration,
                sent_packets=sent_packets,
                recv_packets=recv_packets,
                decoded_packets=decoded_packets,
                dropped_packets=dropped_packets,
                drop_rate=drop_rate,
                total_body=total_body,
                total_wire=total_wire,
                encode_samples=encode_samples,
                decode_samples=decode_samples,
                encode_rtf_samples=encode_rtf_samples,
                decode_rtf_samples=decode_rtf_samples,
                jitter_samples=jitter_samples,
                streams=streams,
            )

    def write_summary_csv(self, **m) -> None:
        import csv

        duration = m["duration"]
        # 单向延迟预算（近似）：发送侧 chunk 缓冲 + 接收侧抖动缓冲目标
        one_way_latency_ms = (float(self.args.chunk_seconds) + float(self.args.target_lat)) * 1000.0
        row = {
            "user_id": self.user_id,
            "device": self.device.type,
            "rvq_layers": int(self.args.rvq_layers),
            "chunk_seconds": float(self.args.chunk_seconds),
            "duration_s": round(duration, 2),
            "sent_packets": m["sent_packets"],
            "recv_packets": m["recv_packets"],
            "decoded_packets": m["decoded_packets"],
            "dropped_packets": m["dropped_packets"],
            "drop_rate": round(m["drop_rate"], 4),
            "body_kbps": round(m["total_body"] * 8.0 / duration / 1000.0, 2),
            "total_kbps": round(m["total_wire"] * 8.0 / duration / 1000.0, 2),
            "encode_ms_mean": _round_or_none(mean_value(m["encode_samples"])),
            "encode_ms_p95": _round_or_none(percentile_value(m["encode_samples"], 95)),
            "decode_ms_mean": _round_or_none(mean_value(m["decode_samples"])),
            "decode_ms_p95": _round_or_none(percentile_value(m["decode_samples"], 95)),
            "encode_rtf_mean": _round_or_none(mean_value(m["encode_rtf_samples"]), 3),
            "encode_rtf_p95": _round_or_none(percentile_value(m["encode_rtf_samples"], 95), 3),
            "decode_rtf_mean": _round_or_none(mean_value(m["decode_rtf_samples"]), 3),
            "decode_rtf_p95": _round_or_none(percentile_value(m["decode_rtf_samples"], 95), 3),
            "jitter_s_mean": _round_or_none(mean_value(m["jitter_samples"]), 3),
            "jitter_s_p95": _round_or_none(percentile_value(m["jitter_samples"], 95), 3),
            "one_way_latency_budget_ms": round(one_way_latency_ms, 1),
        }
        streams = m.get("streams", [])
        speaker_eval = sum(stream.speaker_eval_count for stream in streams)
        speaker_correct = sum(stream.speaker_correct_count for stream in streams)
        speaker_verified = sum(stream.speaker_verified_count for stream in streams)
        row.update(
            {
                "speaker_id_enabled": int(self.speaker_identifier is not None),
                "speaker_eval_count": int(speaker_eval),
                "speaker_correct_count": int(speaker_correct),
                "speaker_verified_count": int(speaker_verified),
                "speaker_accuracy_mean": _round_or_none(
                    speaker_correct / max(1, speaker_eval) if speaker_eval else None,
                    4,
                ),
            }
        )
        csv_path = Path(self.args.summary_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"[SUMMARY:{self.user_id}] appended to {csv_path}")

    def run(self):
        self.load_model()
        self.connect()
        self.start_playback()
        self.init_speaker_identifier()
        threads = [
            threading.Thread(target=self.recv_loop, daemon=True),
            threading.Thread(target=self.mic_send_loop, daemon=True),
            threading.Thread(target=self.monitor_loop, daemon=True),
        ]
        for t in threads:
            t.start()
        print(f"[MAIN] user={self.user_id} room={self.room_id} running; Ctrl+C to stop")
        deadline = None
        if getattr(self.args, "run_seconds", 0) and self.args.run_seconds > 0:
            deadline = time.time() + float(self.args.run_seconds)
        try:
            while self.running.is_set():
                time.sleep(0.2)
                if deadline is not None and time.time() >= deadline:
                    print(f"\n[MAIN] run_seconds={self.args.run_seconds} reached; stopping cleanly")
                    break
        except KeyboardInterrupt:
            print("\n[MAIN] stopping ...")
        finally:
            self.running.clear()
            try:
                self.print_summary()
            except Exception as exc:
                print(f"[SUMMARY:{self.user_id}] failed: {exc}", file=sys.stderr)
            if self.sock is not None:
                try:
                    self.sock.close()
                except Exception:
                    pass
            if self.mic is not None:
                self.mic.stop()
            if self.output_stream is not None:
                try:
                    self.output_stream.stop()
                    self.output_stream.close()
                except Exception:
                    pass
            print("[MAIN] done")


def main():
    parser = argparse.ArgumentParser(description="Three-user SpeechTokenizer group client")
    parser.add_argument("--user_id", required=True, help="A/B/C or another stable user id")
    parser.add_argument("--room_id", default="demo")
    parser.add_argument("--router_ip", default="127.0.0.1")
    parser.add_argument("--router_port", type=int, default=12350)
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--ckpt_path", default=str(DEFAULT_CKPT_PATH))
    parser.add_argument("--key_path", default=str(DEFAULT_KEY_PATH))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mic_device", default="None")
    parser.add_argument("--spk_device", default="None")
    parser.add_argument("--mic_sr", type=float, default=0.0)
    parser.add_argument("--out_sr", type=int, default=0)
    parser.add_argument("--frame_seconds", type=float, default=0.02)
    parser.add_argument("--frame_out_seconds", type=float, default=0.02)
    parser.add_argument("--chunk_seconds", type=float, default=0.5,
                        help="编码分块时长。须使 chunk*sr 为下采样率(320)整数倍以避免边界退化；"
                             "0.5s@16kHz=8000 样点=25 帧，质量与论文整句一致（0.25s 会塌方，勿用）")
    parser.add_argument("--rvq_layers", type=int, default=3)
    parser.add_argument("--target_lat", type=float, default=0.25)
    parser.add_argument("--max_lat", type=float, default=0.80)
    parser.add_argument("--no_mic", action="store_true")
    parser.add_argument("--no_play", action="store_true")
    parser.add_argument("--wav_input", default="")
    parser.add_argument("--loop_wav", action="store_true")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--channel", default="clean", choices=list(CONDITIONS.keys()),
                        help="接收端信道索引扰动条件，与论文 §5.4 一致；默认 clean 不扰动")
    parser.add_argument("--channel_seed", type=int, default=42, help="信道扰动确定性种子基值")
    parser.add_argument("--run_seconds", type=float, default=0.0, help="若>0，运行该秒数后自动干净退出并写 CSV（跑批用，避免依赖信号）")
    parser.add_argument("--summary_csv", default="", help="若指定，退出时把本客户端汇总指标追加写入该 CSV（用于论文跑批填表）")
    parser.add_argument("--speaker_id_enable", action="store_true", help="启用本地 decoded-audio 说话人识别监控")
    parser.add_argument("--speaker_profile_dir", default="", help="说话人档案目录，格式为 <dir>/<speaker_id>/*.wav 或 *.flac")
    parser.add_argument("--speaker_window_sec", type=float, default=3.0, help="说话人识别滑动窗口秒数")
    parser.add_argument("--speaker_hop_sec", type=float, default=1.0, help="说话人识别更新间隔秒数")
    parser.add_argument("--speaker_threshold", type=float, default=0.65, help="说话人 verified 判定阈值")
    parser.add_argument("--speaker_n_mfcc", type=int, default=40, help="MFCC 说话人特征维度")
    parser.add_argument("--speaker_backend", default="mfcc", choices=["mfcc", "ecapa"], help="说话人识别后端：mfcc 或 ECAPA-TDNN")
    parser.add_argument("--speaker_device", default="cpu", choices=["auto", "cuda", "cpu"], help="ECAPA 后端运行设备")
    parser.add_argument("--ecapa_source", default="speechbrain/spkrec-ecapa-voxceleb", help="SpeechBrain ECAPA 模型来源")
    parser.add_argument("--ecapa_savedir", default="output/models/speechbrain_spkrec_ecapa_voxceleb", help="SpeechBrain ECAPA 本地缓存目录")
    args = parser.parse_args()

    try:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))
    except Exception:
        pass

    GroupClient(args).run()


if __name__ == "__main__":
    main()
