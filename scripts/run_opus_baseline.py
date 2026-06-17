"""Run Opus baseline for exp4 and merge into existing results.

Encodes 8 test samples through ffmpeg + libopus at multiple bitrates
{6, 8, 12, 16, 24} kbps, decodes back, computes the same audio quality
metrics as run_exp4_baselines.py, and merges the new rows into the existing
audio_quality_results.csv/json + payload_summary.csv/json.

ffmpeg location: C:\\Users\\Windows11\\.conda\\envs\\speechtokenizer\\Library\\bin\\ffmpeg.exe
(installed via `conda install -n speechtokenizer -c conda-forge ffmpeg=6.1`)

We run a fresh-start strategy: read the existing audio_quality_results.json,
strip any prior 'opus' rows (idempotent), append new opus rows, write back.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import write_csv, write_json
from scripts.evaluate_sample_audio_quality import (
    si_snr_db, pearson_corr, compute_mel_l1, maybe_compute_stoi, maybe_compute_pesq,
)


FFMPEG = r"C:\Users\Windows11\.conda\envs\speechtokenizer\Library\bin\ffmpeg.exe"


def load_json(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def file_sha256(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def load_wav_np(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False), int(sr)


def opus_encode(ffmpeg, src_wav, out_opus, bitrate_bps, sample_rate=16000):
    """Encode a wav to .opus at the requested bitrate. Returns the encoded file size in bytes."""
    out_opus.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src_wav),
        "-c:a", "libopus",
        "-b:a", str(int(bitrate_bps)),
        "-ar", str(sample_rate),
        "-ac", "1",
        str(out_opus),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    encode_ms = (time.time() - t0) * 1000.0
    if r.returncode != 0:
        raise RuntimeError(f"opus encode failed: {r.stderr}")
    return out_opus.stat().st_size, encode_ms, " ".join(cmd)


def opus_decode(ffmpeg, src_opus, out_wav, sample_rate=16000):
    """Decode .opus back to 16 kHz mono wav."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src_opus),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "wav",
        str(out_wav),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    decode_ms = (time.time() - t0) * 1000.0
    if r.returncode != 0:
        raise RuntimeError(f"opus decode failed: {r.stderr}")
    return decode_ms, " ".join(cmd)


def compute_quality(ref_np, est_np, sample_rate, mel_cfg):
    length = min(len(ref_np), len(est_np))
    ref = ref_np[:length].astype(np.float32, copy=False)
    est = est_np[:length].astype(np.float32, copy=False)
    diff = est - ref
    out = {
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
    out["stoi"] = s if s is not None else ""
    out["pesq_wb"] = p if p is not None else ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True, help="baseline_comparison_config.json")
    ap.add_argument("--bitrates", nargs="+", type=int, default=[6000, 8000, 12000, 16000, 24000])
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--ffmpeg", default=FFMPEG)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_json(args.config)
    sample_rate = int(cfg["audio_format"]["sample_rate"])

    # mel kwargs from base config
    base_cfg = load_json(cfg["scit_base"]["config"])

    # Load test list (use the one inside the exp4 run directory)
    test_files = cfg["test_files"]
    rows = []
    with open(test_files, encoding="utf-8-sig") as f:
        for line in f:
            raw = line.strip().lstrip("﻿")
            if not raw:
                continue
            audio = raw.split("\t")[0].strip()
            rows.append({"audio_path": audio, "sample_id": Path(audio).stem})
            if len(rows) >= args.max_samples:
                break
    print(f"loaded {len(rows)} test samples")

    # Verify ffmpeg works
    r = subprocess.run([args.ffmpeg, "-version"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"ffmpeg unusable: exit {r.returncode}")
    print(f"ffmpeg OK: {r.stdout.splitlines()[0]}")

    new_rows = []
    log_path = run_dir / "logs" / "codec_commands"
    log_path.mkdir(parents=True, exist_ok=True)

    for r in rows:
        sid = r["sample_id"]
        src = r["audio_path"]
        # Use the ORIGINAL (saved) wav from the run dir, not the source flac, so all
        # baselines see the same 16 kHz mono wav.
        original_wav = run_dir / "samples" / "original" / f"{sid}.wav"
        if not original_wav.exists():
            raise FileNotFoundError(f"missing exp4 original wav: {original_wav}")
        ref_np, ref_sr = load_wav_np(original_wav)
        if ref_sr != sample_rate:
            raise RuntimeError(f"unexpected sr={ref_sr} for {original_wav}")

        for br in args.bitrates:
            print(f"  {sid} @ {br} bps", flush=True)
            opus_path = run_dir / "artifacts" / "encoded" / "opus" / f"br{br}" / f"{sid}.opus"
            wav_path = run_dir / "samples" / "opus" / f"br{br}" / f"{sid}.wav"
            try:
                opus_bytes, enc_ms, enc_cmd = opus_encode(args.ffmpeg, original_wav, opus_path, br, sample_rate)
                dec_ms, dec_cmd = opus_decode(args.ffmpeg, opus_path, wav_path, sample_rate)
            except Exception as e:
                print(f"    FAILED: {e}")
                continue
            est_np, est_sr = load_wav_np(wav_path)
            assert est_sr == sample_rate, f"decoded sr mismatch: {est_sr}"
            q = compute_quality(ref_np, est_np, sample_rate, base_cfg)
            duration_sec = q["duration_sec"]
            actual_payload_bps = opus_bytes * 8 / duration_sec if duration_sec > 0 else float("nan")
            packetized_bps = (opus_bytes + 16) * 8 / duration_sec if duration_sec > 0 else float("nan")

            row = {
                "method": "opus",
                "model": "libopus",
                "L": "",
                "codec_setting": f"opus_{br}bps",
                "sample_id": sid,
                "decoded_path": str(wav_path),
                "decoded_sha256": file_sha256(wav_path),
                "decode_wall_ms": dec_ms,
                "encode_wall_ms": enc_ms,
                "ideal_bitrate_bps": float(br),
                "packed_payload_bytes": int(opus_bytes),
                "packed_payload_bps": float(actual_payload_bps),
                "packetized_payload_bytes": int(opus_bytes + 16),
                "packetized_payload_bps": float(packetized_bps),
                "overhead_ratio_vs_ideal": (packetized_bps - br) / br if br > 0 else float("nan"),
                "bits_per_code": "",
                "num_indices": "",
                "T_native": "",
                "native_sample_rate": 48000,  # libopus internal SR (decoded back to 16k)
                "encode_cmd": enc_cmd,
                "decode_cmd": dec_cmd,
                "original_path": str(original_wav),
                **q,
            }
            new_rows.append(row)

            # Log per-sample command
            log_file = log_path / f"opus_{sid}_br{br}.log"
            log_file.write_text(
                f"# encode\n{enc_cmd}\n# encoded_bytes={opus_bytes}\n# encode_wall_ms={enc_ms:.1f}\n"
                f"\n# decode\n{dec_cmd}\n# decode_wall_ms={dec_ms:.1f}\n",
                encoding="utf-8",
            )

    print(f"\nencoded {len(new_rows)} (sample, bitrate) pairs")

    # Merge with existing audio_quality_results.json + payload_summary.json
    aq_json = run_dir / "metrics" / "audio_quality_results.json"
    pl_json = run_dir / "metrics" / "payload_summary.json"
    aq = load_json(aq_json)
    pl = load_json(pl_json)
    # Strip prior 'opus' rows (idempotent re-run)
    aq["rows"] = [r for r in aq["rows"] if r.get("method") != "opus"]
    pl["rows"] = [r for r in pl["rows"] if r.get("method") != "opus"]
    aq["rows"].extend(new_rows)
    pl["rows"].extend([{k: r.get(k, "") for k in [
        "method", "model", "L", "codec_setting", "sample_id", "duration_sec",
        "ideal_bitrate_bps", "packed_payload_bytes", "packed_payload_bps",
        "packetized_payload_bytes", "packetized_payload_bps", "overhead_ratio_vs_ideal",
        "bits_per_code", "num_indices",
    ]} for r in new_rows])

    write_json(aq_json, aq)
    write_json(pl_json, pl)

    # Re-write CSVs from merged data
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

    aq_csv = run_dir / "metrics" / "audio_quality_results.csv"
    pl_csv = run_dir / "metrics" / "payload_summary.csv"
    write_csv(aq_csv, normalize(aq["rows"], audio_fields), audio_fields)
    write_csv(pl_csv, normalize(pl["rows"], payload_fields), payload_fields)

    print(json.dumps({"status": "completed", "added_rows": len(new_rows), "total_rows": len(aq["rows"])}, indent=2))


if __name__ == "__main__":
    main()
