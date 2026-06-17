"""Exp4 baselines: SCIT-Speech-Base/LCA + PCM + EnCodec + DAC unified evaluation.

For each test sample, produce:
  - SCIT-Speech-Base reconstructions at L=1/2/3 (clean operating points)
  - SCIT-Speech-LCA reconstructions at L=1/2/3
  - PCM passthrough (lossless 16-bit upper bound)
  - EnCodec reconstructions at multiple bandwidths {1.5, 3.0, 6.0, 12.0} kbps
  - DAC reconstructions at multiple n_codebooks values

For each (method, sample) compute:
  - audio quality: wave_l1, mel_l1, si_snr_db, corr, stoi, pesq_wb
  - payload: ideal_bitrate (when applicable), packed_payload_bps, packetized_payload_bps, overhead_ratio
  - duration_sec, decode wall-clock ms, sha256 of decoded wav

All decoded audio is written as 16 kHz mono WAV. EnCodec runs natively at 24 kHz
internally; resampling is recorded but evaluation is in 16 kHz.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import write_csv, write_json
from scripts.evaluate_sample_audio_quality import (
    si_snr_db,
    pearson_corr,
    compute_mel_l1,
    maybe_compute_stoi,
    maybe_compute_pesq,
)


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_audio(path, sample_rate):
    try:
        audio, sr = torchaudio.load(path)
    except Exception:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        audio = torch.from_numpy(data.T).contiguous()
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio.float()


def write_wav(path, tensor, sample_rate):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.squeeze(tensor.detach().cpu().float().numpy())
    sf.write(p, arr, sample_rate)


def file_sha256(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def read_test_list(path, max_samples):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            raw = line.strip().lstrip("﻿")
            if not raw:
                continue
            audio = raw.split("\t")[0].strip()
            rows.append({"audio_path": audio, "sample_id": Path(audio).stem})
            if len(rows) >= max_samples:
                break
    return rows


def compute_audio_quality(ref_np, est_np, sample_rate, mel_cfg):
    length = min(len(ref_np), len(est_np))
    ref = ref_np[:length].astype(np.float32, copy=False)
    est = est_np[:length].astype(np.float32, copy=False)
    diff = est - ref
    metrics = {
        "duration_sec": float(length / sample_rate),
        "length_samples": int(length),
        "wave_l1": float(np.mean(np.abs(diff))) if length else float("nan"),
        "rmse": float(np.sqrt(np.mean(diff ** 2))) if length else float("nan"),
        "mel_l1": compute_mel_l1(ref, est, mel_cfg),
        "si_snr_db": si_snr_db(ref, est),
        "corr": pearson_corr(ref, est),
    }
    s = maybe_compute_stoi(ref, est, sample_rate)
    p = maybe_compute_pesq(ref, est, sample_rate)
    metrics["stoi"] = s if s is not None else ""
    metrics["pesq_wb"] = p if p is not None else ""
    return metrics


def packed_indices_bytes(num_indices, codebook_size):
    bits_per_code = math.ceil(math.log2(codebook_size))
    total_bits = num_indices * bits_per_code
    return math.ceil(total_bits / 8), bits_per_code, total_bits


def packetized_bytes(packed_b, header_bytes=16):
    return packed_b + header_bytes


def make_payload(model, L, sample_id, duration_sec, codes_shape, codebook_size, ideal_bps, header_bytes=16):
    L_, B, T = codes_shape
    num_indices = L_ * T
    packed_b, bits_per_code, _ = packed_indices_bytes(num_indices, codebook_size)
    packetized_b = packetized_bytes(packed_b, header_bytes=header_bytes)
    packed_bps = packed_b * 8 / duration_sec if duration_sec > 0 else float("nan")
    packetized_bps = packetized_b * 8 / duration_sec if duration_sec > 0 else float("nan")
    overhead = (packetized_bps - ideal_bps) / ideal_bps if ideal_bps > 0 else float("nan")
    return {
        "ideal_bitrate_bps": ideal_bps,
        "packed_payload_bytes": packed_b,
        "packed_payload_bps": packed_bps,
        "packetized_payload_bytes": packetized_b,
        "packetized_payload_bps": packetized_bps,
        "overhead_ratio_vs_ideal": overhead,
        "bits_per_code": bits_per_code,
        "num_indices": num_indices,
        "header_bytes": header_bytes,
    }


def build_scit(cfg):
    if cfg.get("nas_encoder_config"):
        from nas.encoder_only_model_variant import NASEncoderOnlySpeechTokenizer
        return NASEncoderOnlySpeechTokenizer(cfg, cfg["nas_encoder_config"])
    from speechtokenizer import SpeechTokenizer
    return SpeechTokenizer(cfg)


def load_scit_state(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    return model.load_state_dict(state, strict=False)


def eval_scit(model_name, model, sample_rate, n_q, codebook_size, x_np, sid, run_dir, mel_cfg, device, header_bytes):
    rows = []
    x = torch.from_numpy(x_np).to(device).view(1, 1, -1)
    with torch.no_grad():
        full_codes = model.encode(x, n_q=int(n_q), st=0)
    for L in (1, 2, 3):
        truncated = full_codes[:L].contiguous().long()
        t0 = time.time()
        with torch.no_grad():
            recon = model.decode(truncated, st=0)
        wall_ms = (time.time() - t0) * 1000.0
        recon_np = recon[0, 0].detach().cpu().numpy().astype(np.float32)
        out = run_dir / "samples" / model_name / f"L{L}" / f"{sid}.wav"
        write_wav(out, torch.from_numpy(recon_np), sample_rate)
        q = compute_audio_quality(x_np, recon_np, sample_rate, mel_cfg)
        ideal_bps = L * 50 * math.ceil(math.log2(codebook_size))
        payload = make_payload(model_name, L, sid, q["duration_sec"], tuple(truncated.shape), codebook_size, ideal_bps, header_bytes)
        rows.append({
            "method": model_name,
            "model": model_name,
            "L": int(L),
            "codec_setting": f"L={L}",
            "sample_id": sid,
            "decoded_path": str(out),
            "decoded_sha256": file_sha256(out),
            "decode_wall_ms": wall_ms,
            **q,
            **payload,
        })
    return rows


def eval_pcm(x_np, sample_rate, sid, run_dir, mel_cfg):
    int16 = np.clip(np.round(x_np * 32767.0), -32768, 32767).astype(np.int16)
    recon_np = (int16.astype(np.float32) / 32767.0)
    out = run_dir / "samples" / "pcm" / f"{sid}.wav"
    write_wav(out, torch.from_numpy(recon_np), sample_rate)
    q = compute_audio_quality(x_np, recon_np, sample_rate, mel_cfg)
    bps = sample_rate * 16
    return {
        "method": "pcm",
        "model": "pcm_16bit_16khz",
        "L": "",
        "codec_setting": "16bit_16khz_passthrough",
        "sample_id": sid,
        "decoded_path": str(out),
        "decoded_sha256": file_sha256(out),
        "decode_wall_ms": 0.0,
        "ideal_bitrate_bps": bps,
        "packed_payload_bps": bps,
        "packetized_payload_bps": bps,
        "packed_payload_bytes": int(int16.nbytes),
        "packetized_payload_bytes": int(int16.nbytes),
        "overhead_ratio_vs_ideal": 0.0,
        **q,
    }


_encodec_cache = {}


def get_encodec(bw, device):
    key = float(bw)
    if key not in _encodec_cache:
        from encodec import EncodecModel
        m = EncodecModel.encodec_model_24khz()
        m.set_target_bandwidth(float(bw))
        _encodec_cache[key] = m.to(device).eval()
    return _encodec_cache[key]


def eval_encodec(x_np, src_sr, sid, run_dir, mel_cfg, device, bws):
    rows = []
    x_t = torch.from_numpy(x_np).view(1, 1, -1).to(device)
    x_24 = torchaudio.functional.resample(x_t, src_sr, 24000)
    duration_sec = x_np.shape[0] / src_sr
    for bw in bws:
        m = get_encodec(bw, device)
        t0 = time.time()
        with torch.no_grad():
            ef = m.encode(x_24)
            recon_24 = m.decode(ef)
        wall_ms = (time.time() - t0) * 1000.0
        recon_16 = torchaudio.functional.resample(recon_24, 24000, src_sr)
        recon_np = recon_16[0, 0].detach().cpu().numpy().astype(np.float32)
        out = run_dir / "samples" / "encodec" / f"bw{bw}kbps" / f"{sid}.wav"
        write_wav(out, torch.from_numpy(recon_np), src_sr)
        q = compute_audio_quality(x_np, recon_np, src_sr, mel_cfg)
        codes = ef[0][0]  # (B, n_codebooks, T_24k)
        n_cb = int(codes.shape[1])
        T_24 = int(codes.shape[2])
        num_indices = n_cb * T_24
        packed_b, bits_per_code, _ = packed_indices_bytes(num_indices, codebook_size=1024)
        packed_bps = packed_b * 8 / duration_sec
        packetized_b = packetized_bytes(packed_b, header_bytes=16)
        packetized_bps = packetized_b * 8 / duration_sec
        rows.append({
            "method": "encodec",
            "model": "encodec_24khz",
            "L": n_cb,
            "codec_setting": f"bw{bw}kbps_n_cb{n_cb}",
            "sample_id": sid,
            "decoded_path": str(out),
            "decoded_sha256": file_sha256(out),
            "decode_wall_ms": wall_ms,
            "ideal_bitrate_bps": float(bw) * 1000,
            "packed_payload_bytes": packed_b,
            "packed_payload_bps": packed_bps,
            "packetized_payload_bytes": packetized_b,
            "packetized_payload_bps": packetized_bps,
            "overhead_ratio_vs_ideal": (packetized_bps - float(bw) * 1000) / (float(bw) * 1000),
            "bits_per_code": bits_per_code,
            "num_indices": num_indices,
            "T_native": T_24,
            "native_sample_rate": 24000,
            **q,
        })
    return rows


_dac_cache = {}


def get_dac(device):
    if "model" not in _dac_cache:
        import dac
        mp = dac.utils.download(model_type="16khz")
        m = dac.DAC.load(mp)
        _dac_cache["model"] = m.to(device).eval()
    return _dac_cache["model"]


def eval_dac(x_np, sample_rate, sid, run_dir, mel_cfg, device, n_qs):
    rows = []
    m = get_dac(device)
    duration_sec = x_np.shape[0] / sample_rate
    x = torch.from_numpy(x_np).to(device).view(1, 1, -1)
    x_pre = m.preprocess(x, sample_rate)
    for n_q_use in n_qs:
        t0 = time.time()
        with torch.no_grad():
            z, codes_full, latents, _, _ = m.encode(x_pre, n_quantizers=int(n_q_use))
            recon = m.decode(z)
        wall_ms = (time.time() - t0) * 1000.0
        recon_np = recon[0, 0].detach().cpu().numpy().astype(np.float32)
        out = run_dir / "samples" / "dac" / f"n_q{n_q_use}" / f"{sid}.wav"
        write_wav(out, torch.from_numpy(recon_np), sample_rate)
        q = compute_audio_quality(x_np, recon_np, sample_rate, mel_cfg)
        n_total = int(codes_full.shape[1])
        T_dac = int(codes_full.shape[2])
        actual_n_q = min(int(n_q_use), n_total)
        num_indices = actual_n_q * T_dac
        cb_size = m.codebook_size
        packed_b, bits_per_code, _ = packed_indices_bytes(num_indices, codebook_size=cb_size)
        packed_bps = packed_b * 8 / duration_sec
        packetized_b = packetized_bytes(packed_b, header_bytes=16)
        packetized_bps = packetized_b * 8 / duration_sec
        ideal_bps = packed_bps  # DAC ideal == bit-packed indices
        rows.append({
            "method": "dac",
            "model": "dac_16khz",
            "L": actual_n_q,
            "codec_setting": f"n_q_{actual_n_q}",
            "sample_id": sid,
            "decoded_path": str(out),
            "decoded_sha256": file_sha256(out),
            "decode_wall_ms": wall_ms,
            "ideal_bitrate_bps": ideal_bps,
            "packed_payload_bytes": packed_b,
            "packed_payload_bps": packed_bps,
            "packetized_payload_bytes": packetized_b,
            "packetized_payload_bps": packetized_bps,
            "overhead_ratio_vs_ideal": (packetized_bps - ideal_bps) / ideal_bps if ideal_bps > 0 else float("nan"),
            "bits_per_code": bits_per_code,
            "num_indices": num_indices,
            "T_native": T_dac,
            "native_sample_rate": sample_rate,
            **q,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_json(args.config)
    sample_rate = int(cfg["audio_format"]["sample_rate"])
    n_q = int(cfg["scit_n_q"])
    codebook_size = int(cfg["scit_codebook_size"])
    header_bytes = int(cfg["payload_packet_schema"]["header_bytes"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = read_test_list(cfg["test_files"], args.max_samples)
    print(f"loaded {len(rows)} test samples", flush=True)
    for r in rows:
        a = load_audio(r["audio_path"], sample_rate)
        r["np"] = a.squeeze(0).numpy().astype(np.float32)
        out = run_dir / "samples" / "original" / f"{r['sample_id']}.wav"
        if not out.exists():
            write_wav(out, a[0], sample_rate)
        r["original_path"] = str(out)

    print("loading SCIT-Speech-Base", flush=True)
    base_cfg = load_json(cfg["scit_base"]["config"])
    base = build_scit(base_cfg).to(device).eval()
    bm, bu = load_scit_state(base, cfg["scit_base"]["checkpoint"])
    print(f"  Base load: {len(bm)} miss / {len(bu)} extra", flush=True)

    print("loading SCIT-Speech-LCA (v2)", flush=True)
    lca_cfg = load_json(cfg["scit_lca"]["config"])
    lca = build_scit(lca_cfg).to(device).eval()
    lm, lu = load_scit_state(lca, cfg["scit_lca"]["checkpoint"])
    print(f"  LCA load: {len(lm)} miss / {len(lu)} extra", flush=True)

    encodec_bws = cfg["baselines"]["encodec"]["bandwidths_kbps"] if cfg["baselines"]["encodec"]["enabled"] else []
    dac_n_qs = [1, 2, 3, 4, 6, 9, 12] if cfg["baselines"]["dac"]["enabled"] else []
    do_pcm = cfg["baselines"]["pcm"]["enabled"]

    all_rows = []
    for r in rows:
        sid, x_np = r["sample_id"], r["np"]
        print(f"--- {sid} ---", flush=True)
        for row in eval_scit("scit_base", base, sample_rate, n_q, codebook_size, x_np, sid, run_dir, base_cfg, device, header_bytes):
            row["original_path"] = r["original_path"]
            all_rows.append(row)
        for row in eval_scit("scit_lca", lca, sample_rate, n_q, codebook_size, x_np, sid, run_dir, base_cfg, device, header_bytes):
            row["original_path"] = r["original_path"]
            all_rows.append(row)
        if do_pcm:
            row = eval_pcm(x_np, sample_rate, sid, run_dir, base_cfg)
            row["original_path"] = r["original_path"]
            all_rows.append(row)
        if encodec_bws:
            for row in eval_encodec(x_np, sample_rate, sid, run_dir, base_cfg, device, encodec_bws):
                row["original_path"] = r["original_path"]
                all_rows.append(row)
        if dac_n_qs:
            for row in eval_dac(x_np, sample_rate, sid, run_dir, base_cfg, device, dac_n_qs):
                row["original_path"] = r["original_path"]
                all_rows.append(row)

    audio_fields = ["method", "model", "L", "codec_setting", "sample_id",
                    "duration_sec", "length_samples",
                    "wave_l1", "rmse", "mel_l1", "si_snr_db", "corr", "stoi", "pesq_wb",
                    "decode_wall_ms", "decoded_path", "decoded_sha256", "original_path"]
    payload_fields = ["method", "model", "L", "codec_setting", "sample_id",
                      "duration_sec",
                      "ideal_bitrate_bps", "packed_payload_bytes", "packed_payload_bps",
                      "packetized_payload_bytes", "packetized_payload_bps", "overhead_ratio_vs_ideal",
                      "bits_per_code", "num_indices"]

    def normalize(rows, keys):
        return [{k: r.get(k, "") for k in keys} for r in rows]

    audio_csv = run_dir / "metrics" / "audio_quality_results.csv"
    payload_csv = run_dir / "metrics" / "payload_summary.csv"
    audio_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(audio_csv, normalize(all_rows, audio_fields), audio_fields)
    write_csv(payload_csv, normalize(all_rows, payload_fields), payload_fields)
    write_json(run_dir / "metrics" / "audio_quality_results.json", {"run_id": run_dir.name, "rows": all_rows})
    write_json(run_dir / "metrics" / "payload_summary.json", {"run_id": run_dir.name, "rows": [{k: r.get(k, "") for k in payload_fields} for r in all_rows]})
    print(json.dumps({"status": "completed", "rows": len(all_rows), "audio_csv": str(audio_csv)}, indent=2))


if __name__ == "__main__":
    main()
