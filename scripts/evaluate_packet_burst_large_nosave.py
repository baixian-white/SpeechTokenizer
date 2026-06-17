"""Large-set packet/burst loss evaluation without saving all WAVs.

Same packet/burst protocol as evaluate_packet_burst_loss.py, but intended for
full LibriSpeech test-clean/test-other: writes metrics only and optionally saves
WAVs for the first N samples.
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_lca_vs_base import (
    load_json,
    build_model,
    load_state,
    load_audio_for_inference,
    write_wav,
    read_sample_rows,
    hash_file,
)
from scripts.evaluate_packet_burst_loss import apply_packet_burst
from scripts.evaluate_sample_audio_quality import (
    si_snr_db,
    pearson_corr,
    compute_mel_l1,
    maybe_compute_stoi,
    maybe_compute_pesq,
)


def model_inference(model, x_np, L, condition, n_q, device, channel_seed):
    x = torch.from_numpy(x_np).to(device).view(1, 1, -1)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(channel_seed))
    with torch.no_grad():
        full_codes = model.encode(x, n_q=int(n_q), st=0)
        truncated = full_codes[:int(L)].contiguous().long()
        perturbed, stats = apply_packet_burst(truncated, condition, generator)
        recon = model.decode(perturbed, st=0)
    return recon[0, 0].detach().cpu().numpy().astype(np.float32), stats


def evaluate_one(model_name, model, base_audio, sample_id, L, condition, n_q, cfg, sample_rate, channel_seed, device, run_dir=None, save_wav=False):
    recon, stats = model_inference(model, base_audio, L, condition, n_q, device, channel_seed)
    length = min(len(base_audio), len(recon))
    ref = base_audio[:length].astype(np.float32, copy=False)
    est = recon[:length].astype(np.float32, copy=False)
    cond_name = condition["name"]
    diff = est - ref
    ref_rms = float(np.sqrt(np.mean(ref**2))) if len(ref) else 0.0
    est_rms = float(np.sqrt(np.mean(est**2))) if len(est) else 0.0
    recon_path = ""
    if save_wav and run_dir is not None:
        recon_path = str(run_dir / "samples" / model_name / cond_name / f"L{L}" / f"{sample_id}.wav")
        write_wav(recon_path, torch.from_numpy(est), sample_rate)
    return {
        "model": model_name,
        "sample_id": sample_id,
        "L": int(L),
        "channel": cond_name,
        "condition_type": condition.get("type", "clean"),
        "ideal_bitrate_bps": int(L) * 50 * 10,
        "duration_sec": float(length / sample_rate),
        "wave_l1": float(np.mean(np.abs(diff))) if len(diff) else float("nan"),
        "rmse": float(np.sqrt(np.mean(diff**2))) if len(diff) else float("nan"),
        "mel_l1": compute_mel_l1(ref, est, cfg),
        "si_snr_db": si_snr_db(ref, est),
        "corr": pearson_corr(ref, est),
        "stoi": maybe_compute_stoi(ref, est, sample_rate) or "",
        "pesq_wb": maybe_compute_pesq(ref, est, sample_rate) or "",
        "rms_ratio_db": 20.0 * math.log10((est_rms + 1e-8) / (ref_rms + 1e-8)),
        "recon_peak": float(np.max(np.abs(est))) if len(est) else 0.0,
        "clip_ratio": float(np.mean(np.abs(est) >= 0.999)) if len(est) else 0.0,
        "length_samples": int(length),
        "actual_index_loss": float(stats.get("actual_index_loss", 0.0)),
        "affected_indices": int(stats.get("affected_indices", 0)),
        "packet_frames": stats.get("packet_frames", ""),
        "p_packet": stats.get("p_packet", ""),
        "burst_frames": stats.get("burst_frames", ""),
        "channel_seed": int(channel_seed),
        "recon_path": recon_path,
    }


def write_csv(path, rows, fields):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, key):
    vals = []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)) and not math.isnan(val):
            vals.append(val)
    return sum(vals) / len(vals) if vals else float("nan")


def write_summary(run_dir, rows, base_ckpt, lca_ckpt, channel_seed):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["model"], row["L"], row["channel"])].append(row)
    channels = sorted({row["channel"] for row in rows})
    lines = [
        "# Full LibriSpeech packet/burst evaluation summary",
        "",
        f"- Base: `{base_ckpt}`",
        f"- LCA: `{lca_ckpt}`",
        f"- Channel seed: `{channel_seed}`",
        "",
        "| model | L | condition | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ | actual loss |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ["base", "lca"]:
        for L in [1, 2, 3]:
            for channel in channels:
                rows_here = buckets[(model, L, channel)]
                if not rows_here:
                    continue
                lines.append(
                    f"| {model} | {L} | {channel} | {len(rows_here)} | "
                    f"{mean(rows_here, 'mel_l1'):.4f} | {mean(rows_here, 'stoi'):.4f} | "
                    f"{mean(rows_here, 'pesq_wb'):.4f} | {mean(rows_here, 'si_snr_db'):+.3f} | "
                    f"{mean(rows_here, 'actual_index_loss'):.4f} |"
                )
    lines += [
        "",
        "## LCA - Base deltas",
        "",
        "Negative mel-L1 means LCA is better; positive STOI/PESQ/SI-SNR means LCA is better.",
        "",
        "| L | condition | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for L in [1, 2, 3]:
        for channel in channels:
            base_rows = buckets[("base", L, channel)]
            lca_rows = buckets[("lca", L, channel)]
            if not base_rows or not lca_rows:
                continue
            lines.append(
                f"| {L} | {channel} | "
                f"{mean(lca_rows, 'mel_l1') - mean(base_rows, 'mel_l1'):+.4f} | "
                f"{mean(lca_rows, 'stoi') - mean(base_rows, 'stoi'):+.4f} | "
                f"{mean(lca_rows, 'pesq_wb') - mean(base_rows, 'pesq_wb'):+.4f} | "
                f"{mean(lca_rows, 'si_snr_db') - mean(base_rows, 'si_snr_db'):+.3f} |"
            )
    out = run_dir / "reports" / "packet_burst_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--lca-config", required=True)
    parser.add_argument("--lca-checkpoint", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--channel-seed", type=int, default=123)
    parser.add_argument("--save-sample-count", type=int, default=0, help="Save wavs for first N samples only")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    base_cfg = load_json(args.base_config)
    lca_cfg = load_json(args.lca_config)
    sample_rate = int(base_cfg.get("sample_rate", 16000))
    n_q = int(base_cfg.get("n_q", 3))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"loading Base {args.base_checkpoint}", flush=True)
    base_model = build_model(base_cfg).to(device).eval()
    load_state(base_model, args.base_checkpoint)
    print(f"loading LCA {args.lca_checkpoint}", flush=True)
    lca_model = build_model(lca_cfg).to(device).eval()
    load_state(lca_model, args.lca_checkpoint)

    conditions = [
        {"name": "clean", "type": "clean"},
        {"name": "packet-loss-1p", "type": "packet_loss", "p_packet": 0.01, "packet_frames": 5},
        {"name": "packet-loss-3p", "type": "packet_loss", "p_packet": 0.03, "packet_frames": 5},
        {"name": "packet-loss-5p", "type": "packet_loss", "p_packet": 0.05, "packet_frames": 5},
        {"name": "burst-2f", "type": "single_burst", "burst_frames": 2},
        {"name": "burst-5f", "type": "single_burst", "burst_frames": 5},
        {"name": "burst-10f", "type": "single_burst", "burst_frames": 10},
    ]
    l_values = [1, 2, 3]
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else 10**9
    samples = read_sample_rows(args.sample_list, max_samples)
    print(f"evaluating {len(samples)} samples x {len(l_values)} L x {len(conditions)} conditions x 2 models", flush=True)

    rows = []
    tic = time.time()
    for idx, sample in enumerate(samples):
        audio = load_audio_for_inference(sample["audio"], sample_rate).squeeze(0).numpy().astype(np.float32)
        save_this_sample = (idx + 1) <= max(0, int(args.save_sample_count))
        if save_this_sample:
            original_path = run_dir / "samples" / "original" / f"{sample['sample_id']}.wav"
            write_wav(original_path, torch.from_numpy(audio), sample_rate)
        for L in l_values:
            for condition in conditions:
                seed = args.channel_seed + idx * 10007 + int(L) * 1009 + abs(hash(condition["name"])) % 1000
                rows.append(evaluate_one("base", base_model, audio, sample["sample_id"], L, condition, n_q, base_cfg, sample_rate, seed, str(device), run_dir, save_this_sample))
                rows.append(evaluate_one("lca", lca_model, audio, sample["sample_id"], L, condition, n_q, base_cfg, sample_rate, seed, str(device), run_dir, save_this_sample))
        if (idx + 1) % 25 == 0 or (idx + 1) == len(samples):
            print(f"done {idx + 1}/{len(samples)} elapsed={time.time() - tic:.1f}s", flush=True)

    fields = ["model", "sample_id", "L", "channel", "condition_type", "ideal_bitrate_bps", "duration_sec", "wave_l1", "rmse", "mel_l1", "si_snr_db", "corr", "stoi", "pesq_wb", "rms_ratio_db", "recon_peak", "clip_ratio", "length_samples", "actual_index_loss", "affected_indices", "packet_frames", "p_packet", "burst_frames", "channel_seed", "recon_path"]
    out_csv = run_dir / "metrics" / "packet_burst_results.csv"
    out_json = run_dir / "metrics" / "packet_burst_results.json"
    write_csv(out_csv, rows, fields)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "rows": rows,
        "sample_count": len(samples),
        "conditions": conditions,
        "channel_seed": args.channel_seed,
        "save_sample_count": int(args.save_sample_count),
        "base_checkpoint": args.base_checkpoint,
        "lca_checkpoint": args.lca_checkpoint,
        "base_sha256": hash_file(args.base_checkpoint),
        "lca_sha256": hash_file(args.lca_checkpoint),
    }, ensure_ascii=False), encoding="utf-8")
    summary = write_summary(run_dir, rows, args.base_checkpoint, args.lca_checkpoint, args.channel_seed)
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(out_csv), "summary": str(summary)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
