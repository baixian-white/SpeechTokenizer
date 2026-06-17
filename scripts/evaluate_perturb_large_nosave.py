"""Large-set Base vs LCA channel perturbation evaluation without writing WAVs.

Uses the same ChannelSim protocol as evaluate_lca_vs_base.py, but writes only
CSV/JSON/Markdown metrics. Intended for full LibriSpeech test-clean/test-other.
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
    read_sample_rows,
    hash_file,
    write_wav,
)
from scripts.evaluate_sample_audio_quality import (
    si_snr_db,
    pearson_corr,
    compute_mel_l1,
    maybe_compute_stoi,
    maybe_compute_pesq,
)
from speechtokenizer.trainer.lca_trainer import apply_channel_sim_torch


def model_inference(model, x_np, L, p_drop, p_sub, n_q, codebook_size, device, channel_seed):
    x = torch.from_numpy(x_np).to(device).view(1, 1, -1)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(channel_seed))
    with torch.no_grad():
        full_codes = model.encode(x, n_q=int(n_q), st=0)
        truncated = full_codes[:int(L)].contiguous().long()
        pert, sim_stats = apply_channel_sim_torch(
            truncated,
            codebook_size=int(codebook_size),
            p_drop=float(p_drop),
            p_sub=float(p_sub),
            generator=generator,
        )
        recon = model.decode(pert, st=0)
    return recon[0, 0].detach().cpu().numpy().astype(np.float32), sim_stats


def evaluate_one(model_name, model, base_audio, sample_id, L, cond, n_q, cfg, sample_rate, channel_seed, device, run_dir=None, save_wav=False):
    p_drop = float(cond.get("p_drop", 0.0))
    p_sub = float(cond.get("p_sub", 0.0))
    cond_name = cond["name"]
    recon, sim_stats = model_inference(
        model, base_audio, L=L, p_drop=p_drop, p_sub=p_sub,
        n_q=n_q, codebook_size=int(cfg.get("codebook_size", 1024)),
        device=device, channel_seed=channel_seed,
    )
    length = min(len(base_audio), len(recon))
    ref = base_audio[:length].astype(np.float32, copy=False)
    est = recon[:length].astype(np.float32, copy=False)
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
        "p_drop": p_drop,
        "p_sub": p_sub,
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
        "actual_p_drop": float(sim_stats.get("actual_p_drop", 0.0)),
        "actual_p_sub": float(sim_stats.get("actual_p_sub", 0.0)),
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
        "# Full LibriSpeech perturbation evaluation summary",
        "",
        f"- Base: `{base_ckpt}`",
        f"- LCA: `{lca_ckpt}`",
        f"- Channel seed: `{channel_seed}`",
        "",
        "| model | L | channel | n | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ | actual_drop | actual_sub |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ["base", "lca"]:
        for L in [1, 2, 3]:
            for channel in channels:
                rs = buckets[(model, L, channel)]
                if not rs:
                    continue
                lines.append(
                    f"| {model} | {L} | {channel} | {len(rs)} | "
                    f"{mean(rs, 'mel_l1'):.4f} | {mean(rs, 'stoi'):.4f} | "
                    f"{mean(rs, 'pesq_wb'):.4f} | {mean(rs, 'si_snr_db'):+.3f} | "
                    f"{mean(rs, 'actual_p_drop'):.4f} | {mean(rs, 'actual_p_sub'):.4f} |"
                )
    lines += [
        "",
        "## LCA - Base deltas",
        "",
        "Negative mel-L1 means LCA is better; positive STOI/PESQ/SI-SNR means LCA is better.",
        "",
        "| L | channel | Δmel-L1 | ΔSTOI | ΔPESQ-WB | ΔSI-SNR |",
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
    out = run_dir / "reports" / "base_vs_lca_perturb_summary.md"
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
    parser.add_argument("--channel-seed", type=int, default=42)
    parser.add_argument("--save-sample-count", type=int, default=0, help="Save reconstructed wavs for the first N samples only; 0 means save none")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    base_cfg = load_json(args.base_config)
    lca_cfg = load_json(args.lca_config)
    sample_rate = int(base_cfg.get("sample_rate", 16000))
    n_q = int(base_cfg.get("n_q", 3))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"loading Base model from {args.base_checkpoint}", flush=True)
    base_model = build_model(base_cfg).to(device).eval()
    load_state(base_model, args.base_checkpoint)
    print(f"loading LCA model from {args.lca_checkpoint}", flush=True)
    lca_model = build_model(lca_cfg).to(device).eval()
    load_state(lca_model, args.lca_checkpoint)

    channel_conditions = lca_cfg["channel_sim"]["conditions"]
    l_values = lca_cfg.get("random_l_sampling", {}).get("values", [1, 2, 3])
    max_samples = args.max_samples if args.max_samples and args.max_samples > 0 else 10**9
    samples = read_sample_rows(args.sample_list, max_samples)
    total_pairs = len(samples) * len(l_values) * len(channel_conditions) * 2
    print(f"evaluating {len(samples)} samples x {len(l_values)} L x {len(channel_conditions)} channels x 2 models = {total_pairs} pairs", flush=True)

    rows = []
    tic = time.time()
    for idx, sample in enumerate(samples, 1):
        audio = load_audio_for_inference(sample["audio"], sample_rate).squeeze(0).numpy().astype(np.float32)
        save_this_sample = idx <= max(0, int(args.save_sample_count))
        if save_this_sample:
            original_path = run_dir / "samples" / "original" / f"{sample['sample_id']}.wav"
            write_wav(original_path, torch.from_numpy(audio), sample_rate)
        for L in l_values:
            for cond in channel_conditions:
                pair_seed = (
                    args.channel_seed
                    + abs(hash(sample["sample_id"])) % (2**16)
                    + int(L) * 1000
                    + abs(hash(cond["name"])) % 1000
                )
                rows.append(evaluate_one("base", base_model, audio, sample["sample_id"], L, cond, n_q, base_cfg, sample_rate, pair_seed, str(device), run_dir, save_this_sample))
                rows.append(evaluate_one("lca", lca_model, audio, sample["sample_id"], L, cond, n_q, base_cfg, sample_rate, pair_seed, str(device), run_dir, save_this_sample))
        if idx % 25 == 0 or idx == len(samples):
            print(f"done {idx}/{len(samples)} elapsed={time.time() - tic:.1f}s", flush=True)

    fields = [
        "model", "sample_id", "L", "channel", "p_drop", "p_sub", "ideal_bitrate_bps",
        "duration_sec", "wave_l1", "rmse", "mel_l1", "si_snr_db", "corr", "stoi", "pesq_wb",
        "rms_ratio_db", "recon_peak", "clip_ratio", "length_samples", "actual_p_drop", "actual_p_sub", "channel_seed", "recon_path",
    ]
    csv_path = run_dir / "metrics" / "base_vs_lca_perturb_results.csv"
    json_path = run_dir / "metrics" / "base_vs_lca_perturb_results.json"
    write_csv(csv_path, rows, fields)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "sample_count": len(samples),
        "base_checkpoint": args.base_checkpoint,
        "base_sha256": hash_file(args.base_checkpoint),
        "lca_checkpoint": args.lca_checkpoint,
        "lca_sha256": hash_file(args.lca_checkpoint),
        "L_values": list(l_values),
        "channel_conditions": channel_conditions,
        "channel_seed": args.channel_seed,
        "save_sample_count": int(args.save_sample_count),
        "rows": rows,
    }, ensure_ascii=False), encoding="utf-8")
    summary = write_summary(run_dir, rows, args.base_checkpoint, args.lca_checkpoint, args.channel_seed)
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(csv_path), "summary": str(summary)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
