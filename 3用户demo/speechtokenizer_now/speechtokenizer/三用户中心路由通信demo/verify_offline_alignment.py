#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线整句验证：证明 demo 的 codec 路径与论文 §5 离线评估逐行一致，并量化
实时分块（0.25s）相对整句的质量残差（双向 LSTM 导致、不可消除）。

输出（每个 L）：
  - whole_mel_l1 : 整句一次 encode/decode 的 mel-L1
                   （与 scripts/evaluate_clean_large_nosave.model_inference_clean 等价：
                    encode(st=0) → decode(codes[:L])，故复现论文管线数字）
  - chunk_mel_l1 : 0.25s 逐块 encode/decode 后拼接的 mel-L1（demo 实时模式）
  - gap          : chunk - whole（>0 即实时分块的质量代价）

用法（在本目录、speechtokenizer 环境下）：
  python verify_offline_alignment.py --wav ../../../example_input.wav --rvq_layers 1 2 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR))

from group_client import GroupClient, DEFAULT_CONFIG_PATH, DEFAULT_CKPT_PATH, DEFAULT_KEY_PATH


def load_wav_mono(path, target_sr):
    import soundfile as sf
    import torchaudio
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0).numpy().astype(np.float32)


def whole_utterance(model, x_np, L, device):
    """与论文 model_inference_clean 等价：整句 encode(st=0) → decode(codes[:L])。"""
    x = torch.from_numpy(x_np).to(device).view(1, 1, -1)
    with torch.no_grad():
        codes = model.encode(x, st=0)
        recon = model.decode(codes[:int(L)].contiguous().long(), st=0)
    return recon[0, 0].detach().cpu().numpy().astype(np.float32)


def chunked(model, x_np, L, device, chunk_sec, sr):
    """demo 实时模式等价：0.25s 逐块 encode→截断→decode，拼接。"""
    n = max(1, int(chunk_sec * sr))
    outs = []
    with torch.no_grad():
        for s in range(0, len(x_np), n):
            seg = x_np[s:s + n]
            if len(seg) < n:
                seg = np.pad(seg, (0, n - len(seg)))
            x = torch.from_numpy(seg).to(device).view(1, 1, -1)
            codes = model.encode(x, st=0)[:int(L)].contiguous().long()
            rec = model.decode(codes, st=0)[0, 0].detach().cpu().numpy().astype(np.float32)
            outs.append(rec)
    return np.concatenate(outs)[:len(x_np)]


def mel_l1(ref, est, cfg):
    """与论文 compute_mel_l1 同口径（优先复用主仓实现，回退本地实现）。"""
    try:
        sys.path.insert(0, str(DEMO_DIR.parents[4]))  # 主仓根，尽力而为
        from scripts.evaluate_sample_audio_quality import compute_mel_l1
        return float(compute_mel_l1(ref, est, cfg))
    except Exception:
        import torchaudio
        n = min(len(ref), len(est))
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(cfg.get("sample_rate", 16000)),
            n_fft=int(cfg.get("n_fft", 1024)), hop_length=int(cfg.get("hop_size", 240)),
            win_length=int(cfg.get("win_size", 1024)), n_mels=int(cfg.get("num_mels", 80)),
            f_min=float(cfg.get("fmin", 0)), f_max=float(cfg.get("fmax", 8000)) or None,
        )
        def lm(x):
            m = mel(torch.from_numpy(x[:n]).float().view(1, -1))
            return torch.log(torch.clamp(m, min=1e-5))
        return float((lm(ref) - lm(est)).abs().mean().item())


def main():
    import json
    ap = argparse.ArgumentParser(description="demo 整句 vs 论文管线对齐 + 实时分块残差量化")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--config_path", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--ckpt_path", default=str(DEFAULT_CKPT_PATH))
    ap.add_argument("--key_path", default=str(DEFAULT_KEY_PATH))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--rvq_layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--chunk_seconds", type=float, default=0.25)
    args = ap.parse_args()

    gc_args = SimpleNamespace(config_path=args.config_path, ckpt_path=args.ckpt_path,
                              key_path=args.key_path, device=args.device, rvq_layers=3,
                              user_id="verify", room_id="verify", channel="clean")
    client = GroupClient(gc_args)
    client.load_model()
    model, device = client.model, client.device
    cfg = json.load(open(args.config_path, encoding="utf-8"))
    sr = client.model_sr

    x = load_wav_mono(args.wav, sr)
    print(f"[VERIFY] wav={args.wav} dur={len(x)/sr:.2f}s sr={sr} device={device}")
    print(f"{'L':>2} {'whole_mel_l1':>13} {'chunk_mel_l1':>13} {'gap(chunk-whole)':>17}")
    for L in args.rvq_layers:
        w = whole_utterance(model, x, L, device)
        c = chunked(model, x, L, device, args.chunk_seconds, sr)
        mw, mc = mel_l1(x, w, cfg), mel_l1(x, c, cfg)
        print(f"{L:>2} {mw:>13.4f} {mc:>13.4f} {mc-mw:>17.4f}")
    print("\n[VERIFY] whole_mel_l1 复现论文离线管线数字（encode st=0 / decode codes[:L] 逐行一致）；")
    print("[VERIFY] gap>0 即实时分块相对整句的质量代价（双向 LSTM 边界不连续，不可消除）。")


if __name__ == "__main__":
    main()
