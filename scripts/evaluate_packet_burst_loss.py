"""Evaluate Base vs LCA under packet/burst index loss.

This is a communication-oriented extension of evaluate_lca_vs_base.py.
It keeps the same intrusive metrics but replaces i.i.d. ChannelSim with:
  - packet loss: divide latent frames into fixed-size packets and drop packets
  - single burst loss: replace one contiguous span of latent frames

Loss concealment uses previous-index replacement, matching the existing
index-dropout implementation semantics.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import write_csv, write_json
from scripts.evaluate_lca_vs_base import (
    load_json,
    build_model,
    load_state,
    load_audio_for_inference,
    write_wav,
    read_sample_rows,
    hash_file,
)
from scripts.evaluate_sample_audio_quality import (
    si_snr_db,
    pearson_corr,
    compute_mel_l1,
    maybe_compute_stoi,
    maybe_compute_pesq,
)


def previous_index_replace(codes, mask):
    out = codes.clone()
    if out.shape[-1] <= 1 or not mask.any():
        return out
    mask = mask.clone()
    mask[..., 0] = False
    prev = torch.cat([out[..., :1], out[..., :-1]], dim=-1)
    return torch.where(mask, prev, out)


def apply_packet_burst(codes, condition, generator):
    """codes shape (L, B, T). Returns perturbed codes and stats."""
    out = codes.clone()
    L, B, T = out.shape
    total = L * B * T
    name = condition["name"]
    affected = 0

    if name == "clean":
        mask = torch.zeros_like(out, dtype=torch.bool)
    elif condition["type"] == "packet_loss":
        packet_frames = int(condition.get("packet_frames", 5))
        p_packet = float(condition.get("p_packet", 0.03))
        num_packets = int(math.ceil(T / packet_frames))
        packet_mask = torch.rand((B, num_packets), device=out.device, generator=generator) < p_packet
        frame_mask = torch.zeros((B, T), device=out.device, dtype=torch.bool)
        for packet_idx in range(num_packets):
            start = packet_idx * packet_frames
            end = min(T, start + packet_frames)
            frame_mask[:, start:end] = packet_mask[:, packet_idx:packet_idx+1]
        mask = frame_mask.unsqueeze(0).expand(L, B, T)
        out = previous_index_replace(out, mask)
        affected = int(mask.sum().item())
    elif condition["type"] == "single_burst":
        burst_frames = int(condition.get("burst_frames", 5))
        burst_frames = max(1, min(burst_frames, T))
        frame_mask = torch.zeros((B, T), device=out.device, dtype=torch.bool)
        if T > burst_frames:
            starts = torch.randint(1, T - burst_frames + 1, (B,), device=out.device, generator=generator)
        else:
            starts = torch.zeros((B,), device=out.device, dtype=torch.long)
        for b in range(B):
            start = int(starts[b].item())
            frame_mask[b, start:start+burst_frames] = True
        frame_mask[:, 0] = False
        mask = frame_mask.unsqueeze(0).expand(L, B, T)
        out = previous_index_replace(out, mask)
        affected = int(mask.sum().item())
    else:
        raise ValueError(f"unknown condition type: {condition}")

    return out, {
        "condition": name,
        "shape": list(out.shape),
        "total_indices": total,
        "affected_indices": affected,
        "actual_index_loss": affected / total if total else 0.0,
        "packet_frames": condition.get("packet_frames", ""),
        "p_packet": condition.get("p_packet", ""),
        "burst_frames": condition.get("burst_frames", ""),
    }


def model_inference(model, x_np, L, condition, n_q, cfg, device, channel_seed):
    x = torch.from_numpy(x_np).to(device).view(1, 1, -1)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(channel_seed))
    with torch.no_grad():
        full_codes = model.encode(x, n_q=int(n_q), st=0)
        truncated = full_codes[:int(L)].contiguous().long()
        pert, stats = apply_packet_burst(truncated, condition, gen)
        recon = model.decode(pert, st=0)
    return recon[0, 0].detach().cpu().numpy().astype(np.float32), stats


def evaluate_one(model_name, model, base_audio, sample_id, L, condition, n_q, cfg, run_dir, sample_rate, channel_seed, device):
    recon, stats = model_inference(model, base_audio, L, condition, n_q, cfg, device, channel_seed)
    length = min(len(base_audio), len(recon))
    ref = base_audio[:length].astype(np.float32, copy=False)
    est = recon[:length].astype(np.float32, copy=False)
    cond_name = condition["name"]
    out_path = run_dir / "samples" / model_name / cond_name / f"L{L}" / f"{sample_id}.wav"
    write_wav(out_path, torch.from_numpy(est), sample_rate)

    diff = est - ref
    ref_rms = float(np.sqrt(np.mean(ref**2))) if len(ref) else 0.0
    est_rms = float(np.sqrt(np.mean(est**2))) if len(est) else 0.0
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
        "recon_path": str(out_path),
    }


def write_summary(run_dir, rows, base_ckpt, lca_ckpt):
    from collections import defaultdict
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["model"], row["L"], row["channel"])].append(row)
    def avg(rs, key):
        vals = [r[key] for r in rs if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else float("nan")
    lines = ["# Packet/Burst Evaluation Summary", "", f"- Base: `{base_ckpt}`", f"- LCA: `{lca_ckpt}`", ""]
    lines += ["| model | L | condition | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR ↑ | actual loss |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for key in sorted(buckets):
        model, L, cond = key
        rs = buckets[key]
        lines.append(f"| {model} | {L} | {cond} | {avg(rs,'mel_l1'):.3f} | {avg(rs,'stoi'):.3f} | {avg(rs,'pesq_wb'):.3f} | {avg(rs,'si_snr_db'):+.2f} | {avg(rs,'actual_index_loss'):.3f} |")
    out = run_dir / "reports" / "packet_burst_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--lca-config", required=True)
    ap.add_argument("--lca-checkpoint", required=True)
    ap.add_argument("--sample-list", required=True)
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--channel-seed", type=int, default=123)
    args = ap.parse_args()

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
    L_values = [1, 2, 3]
    samples = read_sample_rows(args.sample_list, args.max_samples)
    print(f"evaluating {len(samples)} samples x {len(L_values)} L x {len(conditions)} conditions x 2 models", flush=True)
    for s in samples:
        ref_audio = load_audio_for_inference(s["audio"], sample_rate)
        s["np"] = ref_audio.squeeze(0).numpy().astype(np.float32)
        orig_out = run_dir / "samples" / "original" / f"{s['sample_id']}.wav"
        if not orig_out.exists():
            write_wav(orig_out, ref_audio[0], sample_rate)
        s["original_path"] = str(orig_out)

    rows = []
    for idx, s in enumerate(samples):
        for L in L_values:
            for cond in conditions:
                seed = args.channel_seed + idx * 10007 + int(L) * 1009 + abs(hash(cond["name"])) % 1000
                for model_name, model in [("base", base_model), ("lca", lca_model)]:
                    row = evaluate_one(model_name, model, s["np"], s["sample_id"], L, cond, n_q, base_cfg, run_dir, sample_rate, seed, str(device))
                    row["original_path"] = s["original_path"]
                    rows.append(row)
        print(f"done {idx+1}/{len(samples)} {s['sample_id']}", flush=True)

    fields = ["model","sample_id","L","channel","condition_type","ideal_bitrate_bps","duration_sec","wave_l1","rmse","mel_l1","si_snr_db","corr","stoi","pesq_wb","rms_ratio_db","recon_peak","clip_ratio","length_samples","actual_index_loss","affected_indices","packet_frames","p_packet","burst_frames","channel_seed","original_path","recon_path"]
    out_csv = run_dir / "metrics" / "packet_burst_results.csv"
    out_json = run_dir / "metrics" / "packet_burst_results.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out_csv, rows, fields)
    write_json(out_json, {"rows": rows, "sample_count": len(samples), "conditions": conditions, "base_checkpoint": args.base_checkpoint, "lca_checkpoint": args.lca_checkpoint, "base_sha256": hash_file(args.base_checkpoint), "lca_sha256": hash_file(args.lca_checkpoint)})
    summary = write_summary(run_dir, rows, args.base_checkpoint, args.lca_checkpoint)
    print(json.dumps({"status":"completed","rows":len(rows),"csv":str(out_csv),"summary":str(summary)}, indent=2), flush=True)

if __name__ == "__main__":
    main()
