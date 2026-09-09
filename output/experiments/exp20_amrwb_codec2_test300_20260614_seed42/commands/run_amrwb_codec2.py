"""exp20: AMR-WB + Codec2 baselines on test-clean_300 / test-other_300.

Isolated run (does NOT touch exp12). Uses the standalone gyan.dev ffmpeg 8.1.1
full build (has libvo_amrwbenc + libcodec2), NOT the conda-forge 6.1.2 the other
baselines used. Reuses the exact original 16 kHz mono wavs exp12 saved, so rows
are directly comparable / mergeable into exp12's per-sample tables.

AMR-WB: 16 kHz wideband, 9 modes {6.6 .. 23.85 kbps}, .amr container.
Codec2: 8 kHz narrowband only -> resample 16k->8k, encode, decode, ffmpeg
        resamples back to 16k for metric parity. 6 modes {700,1200,1300,1600,2400,3200}.

Metrics: same wave_l1/mel_l1/si_snr/corr/stoi/pesq_wb as run_exp4_baselines /
run_opus_baseline (imported from scripts.evaluate_sample_audio_quality).
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import write_csv, write_json
from scripts.evaluate_sample_audio_quality import (
    si_snr_db, pearson_corr, compute_mel_l1, maybe_compute_stoi, maybe_compute_pesq,
)

FFMPEG = r"H:\H-CODE\speechtokenizer\tools\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
AMRWB_MODES = [6600, 8850, 12650, 14250, 15850, 18250, 19850, 23050, 23850]
# (ffmpeg -mode name, nominal bps). 700C = canonical modern 700 bps variant.
CODEC2_MODES = [("700C", 700), ("1200", 1200), ("1300", 1300), ("1600", 1600), ("2400", 2400), ("3200", 3200)]

EXP12 = PROJECT_ROOT / "output/experiments/exp12_baseline_comparison_test300_20260610_seed42/runs"
BASE_CFG = PROJECT_ROOT / "output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json"


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


def run_ff(cmd):
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    ms = (time.time() - t0) * 1000.0
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{r.stderr}")
    return ms, " ".join(cmd)


def amrwb_roundtrip(ff, src, enc, dec, br):
    enc.parent.mkdir(parents=True, exist_ok=True)
    dec.parent.mkdir(parents=True, exist_ok=True)
    e_ms, e_cmd = run_ff([ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
                          "-c:a", "libvo_amrwbenc", "-b:a", str(br), "-ar", "16000", "-ac", "1", str(enc)])
    d_ms, d_cmd = run_ff([ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(enc),
                          "-ar", "16000", "-ac", "1", "-f", "wav", str(dec)])
    return enc.stat().st_size, e_ms, d_ms, e_cmd, d_cmd


def codec2_roundtrip(ff, src, enc, dec, mode_name):
    enc.parent.mkdir(parents=True, exist_ok=True)
    dec.parent.mkdir(parents=True, exist_ok=True)
    # 16k -> 8k resample happens inside encode; mode selected via -mode; decode back to 16k
    e_ms, e_cmd = run_ff([ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
                          "-ar", "8000", "-ac", "1", "-c:a", "libcodec2", "-mode", str(mode_name), str(enc)])
    d_ms, d_cmd = run_ff([ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(enc),
                          "-ar", "16000", "-ac", "1", "-f", "wav", str(dec)])
    return enc.stat().st_size, e_ms, d_ms, e_cmd, d_cmd


def compute_quality(ref_np, est_np, sr, mel_cfg):
    length = min(len(ref_np), len(est_np))
    ref = ref_np[:length].astype(np.float32, copy=False)
    est = est_np[:length].astype(np.float32, copy=False)
    diff = est - ref
    out = {
        "duration_sec": float(length / sr), "length_samples": int(length),
        "wave_l1": float(np.mean(np.abs(diff))) if length else float("nan"),
        "rmse": float(np.sqrt(np.mean(diff ** 2))) if length else float("nan"),
        "mel_l1": compute_mel_l1(ref, est, mel_cfg),
        "si_snr_db": si_snr_db(ref, est), "corr": pearson_corr(ref, est),
    }
    s = maybe_compute_stoi(ref, est, sr); p = maybe_compute_pesq(ref, est, sr)
    out["stoi"] = s if s is not None else ""
    out["pesq_wb"] = p if p is not None else ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["test-clean_300", "test-other_300"])
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--ffmpeg", default=FFMPEG)
    args = ap.parse_args()

    sr = 16000
    base_cfg = load_json(BASE_CFG)
    orig_dir = EXP12 / args.split / "samples" / "original"
    out_run = PROJECT_ROOT / "output/experiments/exp20_amrwb_codec2_test300_20260614_seed42/runs" / args.split
    (out_run / "metrics").mkdir(parents=True, exist_ok=True)

    wavs = sorted(orig_dir.glob("*.wav"))[:args.max_samples]
    print(f"[{args.split}] {len(wavs)} original wavs from exp12", flush=True)
    r = subprocess.run([args.ffmpeg, "-version"], capture_output=True, text=True)
    print("ffmpeg:", r.stdout.splitlines()[0])

    rows = []
    for i, ow in enumerate(wavs):
        sid = ow.stem
        ref_np, ref_sr = load_wav_np(ow)
        assert ref_sr == sr, f"{ow} sr={ref_sr}"
        for br in AMRWB_MODES:
            enc = out_run / "artifacts" / "amrwb" / f"br{br}" / f"{sid}.amr"
            dec = out_run / "samples" / "amrwb" / f"br{br}" / f"{sid}.wav"
            nb, e_ms, d_ms, ec, dc = amrwb_roundtrip(args.ffmpeg, ow, enc, dec, br)
            est_np, _ = load_wav_np(dec)
            q = compute_quality(ref_np, est_np, sr, base_cfg)
            dur = q["duration_sec"]
            rows.append({"method": "amrwb", "model": "libvo_amrwbenc", "L": "",
                         "codec_setting": f"amrwb_{br}bps", "sample_id": sid,
                         "decoded_path": str(dec), "decoded_sha256": file_sha256(dec),
                         "decode_wall_ms": d_ms, "encode_wall_ms": e_ms,
                         "ideal_bitrate_bps": float(br), "packed_payload_bytes": int(nb),
                         "packed_payload_bps": nb * 8 / dur if dur > 0 else float("nan"),
                         "packetized_payload_bytes": int(nb + 16),
                         "packetized_payload_bps": (nb + 16) * 8 / dur if dur > 0 else float("nan"),
                         "overhead_ratio_vs_ideal": ((nb + 16) * 8 / dur - br) / br if (dur > 0 and br > 0) else float("nan"),
                         "bits_per_code": "", "num_indices": "", "native_sample_rate": 16000,
                         "encode_cmd": ec, "decode_cmd": dc, "original_path": str(ow), **q})
        for mode_name, br in CODEC2_MODES:
            enc = out_run / "artifacts" / "codec2" / f"m{mode_name}" / f"{sid}.c2"
            dec = out_run / "samples" / "codec2" / f"m{mode_name}" / f"{sid}.wav"
            nb, e_ms, d_ms, ec, dc = codec2_roundtrip(args.ffmpeg, ow, enc, dec, mode_name)
            est_np, _ = load_wav_np(dec)
            q = compute_quality(ref_np, est_np, sr, base_cfg)
            dur = q["duration_sec"]
            rows.append({"method": "codec2", "model": "libcodec2", "L": "",
                         "codec_setting": f"codec2_{br}bps", "sample_id": sid,
                         "decoded_path": str(dec), "decoded_sha256": file_sha256(dec),
                         "decode_wall_ms": d_ms, "encode_wall_ms": e_ms,
                         "ideal_bitrate_bps": float(br), "packed_payload_bytes": int(nb),
                         "packed_payload_bps": nb * 8 / dur if dur > 0 else float("nan"),
                         "packetized_payload_bytes": int(nb + 16),
                         "packetized_payload_bps": (nb + 16) * 8 / dur if dur > 0 else float("nan"),
                         "overhead_ratio_vs_ideal": ((nb + 16) * 8 / dur - br) / br if (dur > 0 and br > 0) else float("nan"),
                         "bits_per_code": "", "num_indices": "", "native_sample_rate": 8000,
                         "encode_cmd": ec, "decode_cmd": dc, "original_path": str(ow), **q})
        if (i + 1) % 25 == 0:
            print(f"  [{args.split}] {i+1}/{len(wavs)} done", flush=True)

    aq_fields = ["method", "model", "L", "codec_setting", "sample_id", "duration_sec", "length_samples",
                 "wave_l1", "rmse", "mel_l1", "si_snr_db", "corr", "stoi", "pesq_wb",
                 "decode_wall_ms", "decoded_path", "decoded_sha256", "original_path"]
    pl_fields = ["method", "model", "L", "codec_setting", "sample_id", "duration_sec",
                 "ideal_bitrate_bps", "packed_payload_bytes", "packed_payload_bps",
                 "packetized_payload_bytes", "packetized_payload_bps", "overhead_ratio_vs_ideal",
                 "bits_per_code", "num_indices"]
    write_csv(out_run / "metrics" / "audio_quality_results.csv", [{k: r.get(k, "") for k in aq_fields} for r in rows], aq_fields)
    write_csv(out_run / "metrics" / "payload_summary.csv", [{k: r.get(k, "") for k in pl_fields} for r in rows], pl_fields)
    write_json(out_run / "metrics" / "audio_quality_results.json", {"rows": rows})
    print(json.dumps({"status": "completed", "split": args.split, "rows": len(rows),
                      "n_samples": len(wavs), "settings": len(AMRWB_MODES) + len(CODEC2_MODES)}, indent=2))


if __name__ == "__main__":
    main()
