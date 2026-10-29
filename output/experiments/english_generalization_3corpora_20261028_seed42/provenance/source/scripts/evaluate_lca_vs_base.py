"""Base vs LCA evaluation across L and channel conditions.

For each fixed test sample, run:
  - Base model: encode(n_q=3) -> truncate to L -> ChannelSim -> decode -> wav
  - LCA  model: encode(n_q=3) -> truncate to L -> ChannelSim -> decode -> wav

Then compute per-pair audio-quality metrics (wave_l1, mel_l1, si_snr, corr,
stoi, pesq_wb). The same ChannelSim seed is shared across both models per
(sample, L, channel) triple so any difference is attributable to model
weights only.

Outputs:
  metrics/base_vs_lca_results.csv
  metrics/base_vs_lca_results.json
  reports/base_vs_lca_summary.md
  samples/base/{condition}/L{1,2,3}/{sample_id}.wav
  samples/lca/{condition}/L{1,2,3}/{sample_id}.wav
  samples/original/{sample_id}.wav
"""

import argparse
import json
import math
import os
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
from scripts.evaluate_sample_audio_quality import (
    si_snr_db,
    pearson_corr,
    compute_mel_l1,
    maybe_compute_stoi,
    maybe_compute_pesq,
)
from speechtokenizer.trainer.lca_trainer import apply_channel_sim_torch


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_model(cfg):
    if cfg.get("nas_encoder_config"):
        from nas.encoder_only_model_variant import NASEncoderOnlySpeechTokenizer
        return NASEncoderOnlySpeechTokenizer(cfg, cfg["nas_encoder_config"])
    from speechtokenizer import SpeechTokenizer
    return SpeechTokenizer(cfg)


def load_state(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    return model.load_state_dict(state, strict=False)


def load_audio_for_inference(path, sample_rate):
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


def read_sample_rows(sample_list, max_samples):
    rows = []
    with open(sample_list, "r", encoding="utf-8-sig") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            audio = raw.split("\t")[0].lstrip("﻿").strip()
            rows.append({"audio": audio, "sample_id": Path(audio).stem})
            if len(rows) >= max_samples:
                break
    return rows


def hash_file(path):
    import hashlib
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def model_inference(model, x_np, L, p_drop, p_sub, n_q, codebook_size, device, channel_seed):
    x = torch.from_numpy(x_np).to(device).view(1, 1, -1)
    cuda_gen = torch.Generator(device=device)
    cuda_gen.manual_seed(int(channel_seed))
    with torch.no_grad():
        full_codes = model.encode(x, n_q=int(n_q), st=0)
        truncated = full_codes[:int(L)].contiguous().long()
        pert, sim_stats = apply_channel_sim_torch(
            truncated,
            codebook_size=int(codebook_size),
            p_drop=float(p_drop),
            p_sub=float(p_sub),
            generator=cuda_gen,
        )
        recon = model.decode(pert, st=0)
    return recon[0, 0].detach().cpu().numpy().astype(np.float32), sim_stats


def evaluate_one(model_name, model, base_audio, sample_id, L, cond, n_q, cfg, run_dir, sample_rate, channel_seed, device, save_wavs=True, skip_pesq=False):
    p_drop = float(cond.get("p_drop", 0.0))
    p_sub = float(cond.get("p_sub", 0.0))
    cond_name = cond["name"]
    recon, sim_stats = model_inference(
        model, base_audio,
        L=L, p_drop=p_drop, p_sub=p_sub,
        n_q=n_q, codebook_size=int(cfg.get("codebook_size", 1024)),
        device=device, channel_seed=channel_seed,
    )
    length = min(len(base_audio), len(recon))
    ref = base_audio[:length].astype(np.float32, copy=False)
    est = recon[:length].astype(np.float32, copy=False)

    out_path = run_dir / "samples" / model_name / cond_name / f"L{L}" / f"{sample_id}.wav"
    if save_wavs:
        write_wav(out_path, torch.from_numpy(est), sample_rate)

    diff = est - ref
    ref_rms = float(np.sqrt(np.mean(ref**2))) if len(ref) else 0.0
    est_rms = float(np.sqrt(np.mean(est**2))) if len(est) else 0.0
    wave_l1 = float(np.mean(np.abs(diff))) if len(diff) else float("nan")
    rmse = float(np.sqrt(np.mean(diff**2))) if len(diff) else float("nan")
    rms_ratio_db = 20.0 * math.log10((est_rms + 1e-8) / (ref_rms + 1e-8))
    recon_peak = float(np.max(np.abs(est))) if len(est) else 0.0
    clip_ratio = float(np.mean(np.abs(est) >= 0.999)) if len(est) else 0.0
    mel_l1 = compute_mel_l1(ref, est, cfg)
    si_snr = si_snr_db(ref, est)
    corr = pearson_corr(ref, est)
    stoi_v = maybe_compute_stoi(ref, est, sample_rate)
    pesq_wb = None if skip_pesq else maybe_compute_pesq(ref, est, sample_rate)
    ideal_bitrate = int(L) * 50 * 10

    return {
        "model": model_name,
        "sample_id": sample_id,
        "L": int(L),
        "channel": cond_name,
        "p_drop": p_drop,
        "p_sub": p_sub,
        "ideal_bitrate_bps": ideal_bitrate,
        "duration_sec": float(length / sample_rate),
        "wave_l1": wave_l1,
        "rmse": rmse,
        "mel_l1": mel_l1,
        "si_snr_db": si_snr,
        "corr": corr,
        "stoi": stoi_v if stoi_v is not None else "",
        "pesq_wb": pesq_wb if pesq_wb is not None else "",
        "rms_ratio_db": rms_ratio_db,
        "recon_peak": recon_peak,
        "clip_ratio": clip_ratio,
        "length_samples": int(length),
        "recon_path": str(out_path) if save_wavs else "",
        "actual_p_drop": float(sim_stats.get("actual_p_drop", 0.0)),
        "actual_p_sub": float(sim_stats.get("actual_p_sub", 0.0)),
        "channel_seed": int(channel_seed),
    }


def write_summary_md(run_dir, all_rows, base_meta, lca_meta, channel_seed):
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in all_rows:
        # Resume path may produce string-typed L values from csv reader; coerce.
        try:
            L_val = int(r["L"])
        except (TypeError, ValueError):
            L_val = r["L"]
        buckets[(r["model"], L_val, r["channel"])].append(r)

    def _mean(rows, key):
        vals = []
        for r in rows:
            v = r.get(key)
            if v is None or v == "":
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        return sum(vals) / len(vals) if vals else float("nan")

    n_samples = len({r["sample_id"] for r in all_rows})
    lines = []
    lines.append("# Base vs LCA evaluation summary")
    lines.append("")
    lines.append(f"- Base ckpt: `{base_meta['path']}`")
    lines.append(f"  - sha256: `{base_meta.get('sha256', 'unknown')}`")
    lines.append(f"- LCA ckpt:  `{lca_meta['path']}`")
    lines.append(f"  - sha256: `{lca_meta.get('sha256', 'unknown')}`")
    lines.append(f"- Sample count per (L, channel): {n_samples}")
    lines.append(f"- ChannelSim seed offset: {channel_seed} (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)")
    lines.append("")
    lines.append("## Mean metrics by (model, L, channel)")
    lines.append("")
    lines.append("| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for key in sorted(buckets.keys()):
        model, L, channel = key
        rows = buckets[key]
        lines.append(
            f"| {model} | {L} | {channel} "
            f"| {_mean(rows, 'wave_l1'):.4f} "
            f"| {_mean(rows, 'mel_l1'):.4f} "
            f"| {_mean(rows, 'si_snr_db'):+.3f} "
            f"| {_mean(rows, 'corr'):.4f} "
            f"| {_mean(rows, 'stoi'):.4f} "
            f"| {_mean(rows, 'pesq_wb'):.4f} |"
        )

    lines.append("")
    lines.append("## LCA - Base improvements by (L, channel)")
    lines.append("")
    lines.append("Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.")
    lines.append("")
    lines.append("| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")
    Ls = sorted({int(r["L"]) for r in all_rows})
    chans = sorted({r["channel"] for r in all_rows})
    for L in Ls:
        for ch in chans:
            base_rows = buckets.get(("base", L, ch), [])
            lca_rows = buckets.get(("lca", L, ch), [])
            if not base_rows or not lca_rows:
                continue
            d_wave = _mean(lca_rows, "wave_l1") - _mean(base_rows, "wave_l1")
            d_mel = _mean(lca_rows, "mel_l1") - _mean(base_rows, "mel_l1")
            d_sis = _mean(lca_rows, "si_snr_db") - _mean(base_rows, "si_snr_db")
            d_stoi = _mean(lca_rows, "stoi") - _mean(base_rows, "stoi")
            d_pesq = _mean(lca_rows, "pesq_wb") - _mean(base_rows, "pesq_wb")
            lines.append(
                f"| {L} | {ch} "
                f"| {d_wave:+.4f} | {d_mel:+.4f} | {d_sis:+.3f} "
                f"| {d_stoi:+.4f} | {d_pesq:+.4f} |"
            )

    lines.append("")
    lines.append("## Notes")
    lines.append("- Intrusive objective metrics, sample-wise then averaged across the test set.")
    lines.append("- WER/CER not computed (no ASR model integration).")
    lines.append("- PESQ-WB requires sample_rate=16000.")
    lines.append("- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).")
    lines.append("- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.")

    out = run_dir / "reports" / "base_vs_lca_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--lca-config", required=True, help="LCA config (channel_sim conditions read here)")
    ap.add_argument("--lca-checkpoint", required=True)
    ap.add_argument("--sample-list", required=True)
    ap.add_argument("--max-samples", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--channel-seed", type=int, default=42)
    ap.add_argument("--no-save-wavs", action="store_true",
                    help="Skip writing reconstructed/original WAVs to disk; only emit metrics csv/json. "
                         "Useful for large-scale evaluation runs where audio artifacts are not needed.")
    ap.add_argument("--skip-samples-file", default=None,
                    help="Optional path to a text file listing sample_ids (one per line) to skip. "
                         "Used by the driver to avoid retrying samples that previously caused a "
                         "native crash (segfault / DLL fault) outside Python's try/except reach.")
    ap.add_argument("--skip-pesq", action="store_true",
                    help="Disable PESQ-WB computation. The pypesq C extension can segfault on "
                         "long/silent/edge-case inputs and the crash is not catchable from Python; "
                         "set this flag for large-batch runs where mel-L1 / STOI / SI-SNR suffice.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    base_cfg = load_json(args.base_config)
    lca_cfg = load_json(args.lca_config)
    sample_rate = int(base_cfg.get("sample_rate", 16000))
    n_q = int(base_cfg.get("n_q", 3))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"loading Base model from {args.base_checkpoint}", flush=True)
    base_model = build_model(base_cfg).to(device).eval()
    bm, bu = load_state(base_model, args.base_checkpoint)
    if bm:
        print(f"  Base missing keys: {len(bm)} (first 5: {bm[:5]})", flush=True)
    if bu:
        print(f"  Base unexpected keys: {len(bu)} (first 5: {bu[:5]})", flush=True)

    print(f"loading LCA model from {args.lca_checkpoint}", flush=True)
    lca_model = build_model(lca_cfg).to(device).eval()
    lm, lu = load_state(lca_model, args.lca_checkpoint)
    if lm:
        print(f"  LCA missing keys: {len(lm)} (first 5: {lm[:5]})", flush=True)
    if lu:
        print(f"  LCA unexpected keys: {len(lu)} (first 5: {lu[:5]})", flush=True)

    channel_conditions = lca_cfg["channel_sim"]["conditions"]
    L_values = lca_cfg.get("random_l_sampling", {}).get("values", [1, 2, 3])

    rows = read_sample_rows(args.sample_list, args.max_samples)
    skip_ids = set()
    if args.skip_samples_file and Path(args.skip_samples_file).exists():
        with Path(args.skip_samples_file).open("r", encoding="utf-8") as f:
            for line in f:
                sid = line.strip()
                if sid:
                    skip_ids.add(sid)
        before = len(rows)
        rows = [r for r in rows if r["sample_id"] not in skip_ids]
        print(f"skip-list: dropped {before - len(rows)} samples ({len(skip_ids)} ids in skip file)", flush=True)
    total_pairs = len(rows) * len(L_values) * len(channel_conditions) * 2
    print(f"evaluating {len(rows)} samples x {len(L_values)} L x {len(channel_conditions)} channels x 2 models = {total_pairs} pairs", flush=True)

    for s in rows:
        ref_audio = load_audio_for_inference(s["audio"], sample_rate)
        s["np"] = ref_audio.squeeze(0).numpy().astype(np.float32)
        orig_out = run_dir / "samples" / "original" / f"{s['sample_id']}.wav"
        if not args.no_save_wavs and not orig_out.exists():
            write_wav(orig_out, ref_audio[0], sample_rate)
        s["original_path"] = "" if args.no_save_wavs else str(orig_out)

    fields = [
        "model", "sample_id", "L", "channel", "p_drop", "p_sub", "ideal_bitrate_bps",
        "duration_sec", "wave_l1", "rmse", "mel_l1", "si_snr_db", "corr", "stoi", "pesq_wb",
        "rms_ratio_db", "recon_peak", "clip_ratio", "length_samples",
        "actual_p_drop", "actual_p_sub", "channel_seed", "original_path", "recon_path",
    ]
    csv_path = run_dir / "metrics" / "base_vs_lca_results.csv"
    json_path = run_dir / "metrics" / "base_vs_lca_results.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: scan already-completed sample_ids from existing csv (if any).
    # A sample is treated as completed only when ALL (L, channel, model) rows for it
    # are already in the csv (= 2 * len(L) * len(channel) rows). Partially-finished
    # samples are dropped and re-run to avoid mixing pre/post-crash rows.
    expected_rows_per_sample = 2 * len(L_values) * len(channel_conditions)
    completed_ids = set()
    pre_existing_rows = []
    if csv_path.exists():
        import csv as _csv
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = _csv.DictReader(f)
            from collections import Counter as _Counter
            counts = _Counter()
            cached = []
            for r in reader:
                cached.append(r)
                counts[r["sample_id"]] += 1
            for sid, c in counts.items():
                if c == expected_rows_per_sample:
                    completed_ids.add(sid)
            pre_existing_rows = [r for r in cached if r["sample_id"] in completed_ids]
        print(
            f"resume: found existing csv with {len(completed_ids)} fully-completed samples, "
            f"{len(pre_existing_rows)} rows kept; partial samples will be re-run.",
            flush=True,
        )

    # Open csv in append mode if pre-existing rows survive the resume filter; else fresh write.
    import csv as _csv
    write_header = not (csv_path.exists() and pre_existing_rows)
    if not write_header and pre_existing_rows:
        # Rewrite csv with only fully-completed rows so the file is consistent.
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in pre_existing_rows:
                w.writerow(r)
    csv_fh = csv_path.open("a", encoding="utf-8", newline="")
    csv_writer = _csv.DictWriter(csv_fh, fieldnames=fields, extrasaction="ignore")
    if write_header:
        csv_writer.writeheader()
        csv_fh.flush()

    all_rows = list(pre_existing_rows)
    consecutive_oom = 0
    for s in rows:
        if s["sample_id"] in completed_ids:
            continue
        try:
            sample_rows = []
            for L in L_values:
                for cond in channel_conditions:
                    pair_seed = (
                        args.channel_seed
                        + abs(hash(s["sample_id"])) % (2**16)
                        + int(L) * 1000
                        + abs(hash(cond["name"])) % 1000
                    )
                    row_base = evaluate_one(
                        "base", base_model, s["np"], s["sample_id"], L, cond,
                        n_q, base_cfg, run_dir, sample_rate, pair_seed, str(device),
                        save_wavs=not args.no_save_wavs,
                        skip_pesq=args.skip_pesq,
                    )
                    row_base["original_path"] = s["original_path"]
                    sample_rows.append(row_base)
                    row_lca = evaluate_one(
                        "lca", lca_model, s["np"], s["sample_id"], L, cond,
                        n_q, base_cfg, run_dir, sample_rate, pair_seed, str(device),
                        save_wavs=not args.no_save_wavs,
                        skip_pesq=args.skip_pesq,
                    )
                    row_lca["original_path"] = s["original_path"]
                    sample_rows.append(row_lca)
        except torch.cuda.OutOfMemoryError as e:
            consecutive_oom += 1
            print(f"  WARN: sample {s['sample_id']} OOM ({consecutive_oom}/3): {e}; clearing cache and skipping.", flush=True)
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
            if consecutive_oom >= 3:
                # CUDA context likely corrupted; exiting with code 2 so the driver
                # restarts the python process (fresh cuda context). Resume logic
                # picks up where csv left off.
                print(
                    "  ERROR: 3 consecutive OOMs; CUDA context likely corrupted, "
                    "exiting with code 2 to trigger driver-level restart.",
                    flush=True,
                )
                csv_fh.close()
                sys.exit(2)
            continue
        except Exception as e:
            print(f"  WARN: sample {s['sample_id']} raised {type(e).__name__}: {e}; skipping.", flush=True)
            continue
        consecutive_oom = 0

        # Atomic per-sample append: only commit rows after all (L, channel, model) finished.
        for r in sample_rows:
            csv_writer.writerow(r)
        csv_fh.flush()
        try:
            os.fsync(csv_fh.fileno())
        except OSError:
            pass
        all_rows.extend(sample_rows)
        completed_ids.add(s["sample_id"])
        print(f"  done sample {s['sample_id']}", flush=True)

    csv_fh.close()
    write_json(
        json_path,
        {
            "run_id": run_dir.name,
            "base_checkpoint": args.base_checkpoint,
            "base_checkpoint_sha256": hash_file(args.base_checkpoint),
            "lca_checkpoint": args.lca_checkpoint,
            "lca_checkpoint_sha256": hash_file(args.lca_checkpoint),
            "sample_count": len(rows),
            "L_values": list(L_values),
            "channel_conditions": channel_conditions,
            "channel_seed": args.channel_seed,
            "rows": all_rows,
        },
    )
    summary_md = write_summary_md(
        run_dir, all_rows,
        base_meta={"path": args.base_checkpoint, "sha256": hash_file(args.base_checkpoint)},
        lca_meta={"path": args.lca_checkpoint, "sha256": hash_file(args.lca_checkpoint)},
        channel_seed=args.channel_seed,
    )
    print(json.dumps({"status": "completed", "rows": len(all_rows), "csv": str(csv_path), "summary": str(summary_md)}, indent=2))


if __name__ == "__main__":
    main()
