import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import write_csv, write_json, write_text
from speechtokenizer.trainer.loss import mel_loss


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_wav(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False), int(sample_rate)


def si_snr_db(reference, estimate, eps=1e-8):
    reference = reference - float(np.mean(reference))
    estimate = estimate - float(np.mean(estimate))
    ref_energy = float(np.sum(reference ** 2)) + eps
    scale = float(np.sum(estimate * reference)) / ref_energy
    target = scale * reference
    noise = estimate - target
    return 10.0 * math.log10((float(np.sum(target ** 2)) + eps) / (float(np.sum(noise ** 2)) + eps))


def pearson_corr(reference, estimate, eps=1e-8):
    ref = reference - float(np.mean(reference))
    est = estimate - float(np.mean(estimate))
    denom = math.sqrt(float(np.sum(ref ** 2)) * float(np.sum(est ** 2))) + eps
    return float(np.sum(ref * est) / denom)


def compute_mel_l1(reference, estimate, cfg):
    x = torch.from_numpy(reference).view(1, 1, -1)
    y = torch.from_numpy(estimate).view(1, 1, -1)
    return float(
        mel_loss(
            x,
            y,
            n_fft=int(cfg["n_fft"]),
            num_mels=int(cfg["num_mels"]),
            sample_rate=int(cfg["sample_rate"]),
            hop_size=int(cfg["hop_size"]),
            win_size=int(cfg["win_size"]),
            fmin=cfg.get("fmin", 0),
            fmax=cfg.get("fmax_for_loss", None),
        ).item()
    )


def compare_pair(original_path, recon_path, cfg):
    reference, sr_ref = load_wav(original_path)
    estimate, sr_est = load_wav(recon_path)
    if sr_ref != sr_est:
        raise ValueError(f"Sample-rate mismatch: {original_path}={sr_ref}, {recon_path}={sr_est}")
    length = min(len(reference), len(estimate))
    reference = reference[:length]
    estimate = estimate[:length]
    diff = estimate - reference
    ref_rms = float(np.sqrt(np.mean(reference ** 2)))
    est_rms = float(np.sqrt(np.mean(estimate ** 2)))
    rms_ratio_db = 20.0 * math.log10((est_rms + 1e-8) / (ref_rms + 1e-8))
    return {
        "duration_sec": length / sr_ref,
        "wave_l1": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "si_snr_db": si_snr_db(reference, estimate),
        "corr": pearson_corr(reference, estimate),
        "mel_l1": compute_mel_l1(reference, estimate, cfg),
        "original_rms": ref_rms,
        "recon_rms": est_rms,
        "rms_ratio_db": rms_ratio_db,
        "recon_peak": float(np.max(np.abs(estimate))) if length else 0.0,
        "clip_ratio": float(np.mean(np.abs(estimate) >= 0.999)) if length else 0.0,
        "length_samples": int(length),
    }


def summarize(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def compare_run(run_dir, config):
    run_dir = Path(run_dir)
    cfg = load_json(config)
    sample_root = run_dir / "samples" / "full_utterance"
    original_dir = sample_root / "original"
    rows = []
    for original_path in sorted(original_dir.glob("*.wav")):
        sample_id = original_path.stem
        for layer in (1, 2, 3):
            recon_path = sample_root / f"recon_L{layer}" / f"{sample_id}.wav"
            if not recon_path.exists():
                continue
            metrics = compare_pair(original_path, recon_path, cfg)
            rows.append(
                {
                    "sample_id": sample_id,
                    "L": layer,
                    "ideal_bitrate_bps": layer * 50 * 10,
                    "original_path": str(original_path),
                    "recon_path": str(recon_path),
                    **metrics,
                }
            )
    if not rows:
        raise ValueError(f"No full-utterance reconstruction pairs found under {sample_root}")

    fields = [
        "sample_id",
        "L",
        "ideal_bitrate_bps",
        "duration_sec",
        "wave_l1",
        "rmse",
        "si_snr_db",
        "corr",
        "mel_l1",
        "original_rms",
        "recon_rms",
        "rms_ratio_db",
        "recon_peak",
        "clip_ratio",
        "length_samples",
        "original_path",
        "recon_path",
    ]
    csv_path = run_dir / "metrics" / "full_utterance_audio_quality.csv"
    json_path = run_dir / "metrics" / "full_utterance_audio_quality.json"
    write_csv(csv_path, rows, fields)

    by_layer = {}
    for layer in (1, 2, 3):
        layer_rows = [row for row in rows if row["L"] == layer]
        by_layer[f"L{layer}"] = {
            "count": len(layer_rows),
            "wave_l1": summarize([row["wave_l1"] for row in layer_rows]),
            "mel_l1": summarize([row["mel_l1"] for row in layer_rows]),
            "si_snr_db": summarize([row["si_snr_db"] for row in layer_rows]),
            "corr": summarize([row["corr"] for row in layer_rows]),
            "rms_ratio_db": summarize([row["rms_ratio_db"] for row in layer_rows]),
            "clip_ratio": summarize([row["clip_ratio"] for row in layer_rows]),
        }
    write_json(json_path, {"run_id": run_dir.name, "rows": rows, "summary": by_layer})

    md = [
        "# Full Utterance Audio Quality",
        "",
        "Objective proxy comparison between original full utterances and generated L1/L2/L3 reconstructions.",
        "These metrics do not replace human listening, WER, PESQ, STOI, or channel evaluation.",
        "",
        "| Layer | n | wave L1 mean | mel L1 mean | SI-SNR mean | corr mean | RMS ratio mean | clip ratio mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in (1, 2, 3):
        item = by_layer[f"L{layer}"]
        md.append(
            f"| L{layer} | {item['count']} | {item['wave_l1']['mean']:.6f} | "
            f"{item['mel_l1']['mean']:.6f} | {item['si_snr_db']['mean']:.3f} | "
            f"{item['corr']['mean']:.4f} | {item['rms_ratio_db']['mean']:.3f} dB | "
            f"{item['clip_ratio']['mean']:.6f} |"
        )
    md.extend(["", f"- CSV: `{csv_path}`", f"- JSON: `{json_path}`", ""])
    write_text(run_dir / "reports" / "full_utterance_audio_quality.md", "\n".join(md))
    return rows, by_layer


def build_parser():
    parser = argparse.ArgumentParser(description="Compare original and reconstructed full-utterance audio.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    rows, summary = compare_run(args.run_dir, args.config)
    print(json.dumps({"status": "completed", "rows": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
