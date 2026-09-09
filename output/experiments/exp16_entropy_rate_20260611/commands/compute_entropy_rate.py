"""Exp16: empirical RVQ index entropy for SCIT-Speech-Base on the LibriSpeech test sets.

Goal
----
Compute the EMPIRICAL entropy of SCIT-Speech-Base RVQ indices on a held-out
test set so we can compare

    R_entropy(L) = L * f_q * H(I_l)         bits/sec
    R_naive(L)   = L * 50 * 10 = 500 * L     bits/sec   (codebook=1024 -> 10 bits/code)

This justifies the section 3.1 / section 7 caveat that 500*L bps is only a
CONSERVATIVE upper bound on the channel rate: with f_q=50 Hz and a 1024-entry
codebook, each code consumes 10 bits when bit-packed, but the empirical
per-code entropy H(I_l) is typically below log2(1024)=10 because of skewed
usage and dead codes.

The script does NOT execute training. It only runs the encoder + RVQ on a
fixed number of test utterances and accumulates per-layer histograms over
codebook indices.

Outputs (under output/experiments/exp16_entropy_rate_20260611/)
---------------------------------------------------------------
  metrics/index_histogram_per_layer.csv
      cols: split, L, code_id, count, prob
  metrics/entropy_per_layer.csv
      cols: split, L, n_frames, n_used, dead_code_ratio, top1_freq, top10_freq,
            H_bits, H_uniform_used_bits, R_entropy_bps, R_naive_bps,
            ratio_entropy_to_naive
  metrics/cumulative_rate_per_L.csv
      cols: split, L_cum, sum_H_bits, R_entropy_bps_cum, R_naive_bps_cum,
            savings_pct
  reports/entropy_rate_summary.md
      human-readable: bits saved per layer, biggest entropy gap,
      paper-ready 3-sentence interpretation
  logs/encode.log
      stdout/stderr from the encode loop

Usage
-----
    python compute_entropy_rate.py \
        --split both \
        --max-samples 300 \
        --device cuda
"""
import argparse
import csv
import io
import json
import math
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the SCIT builders + state loaders from the Exp4 baselines script.
from scripts.run_exp4_baselines import build_scit, load_scit_state  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = EXP_DIR / "metrics"
REPORTS_DIR = EXP_DIR / "reports"
LOGS_DIR = EXP_DIR / "logs"
for _d in (METRICS_DIR, REPORTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

EXP2_RUN = (
    PROJECT_ROOT
    / "output"
    / "experiments"
    / "exp2_scit_speech_distill30_retrain_20260529_seed42"
)
SCIT_BASE_CONFIG = EXP2_RUN / "configs" / "scit_speech_base_config.json"
SCIT_BASE_NAS_ENCODER = EXP2_RUN / "configs" / "best_seanet_config.json"
SCIT_BASE_CKPT = EXP2_RUN / "checkpoints" / "SCIT-Speech-Base_best.pt"

EXP7_TEST_LISTS = {
    "test-clean": PROJECT_ROOT
    / "output"
    / "experiments"
    / "exp7_librispeech_test_full_clean_20260606"
    / "artifacts"
    / "test-clean_all_files.txt",
    "test-other": PROJECT_ROOT
    / "output"
    / "experiments"
    / "exp7_librispeech_test_full_clean_20260606"
    / "artifacts"
    / "test-other_all_files.txt",
}

SAMPLE_RATE = 16000
F_Q = 50  # codes per second (256 / 16000 ... actually 320x downsample @ 16 kHz = 50 Hz)
CODEBOOK_SIZE = 1024
N_Q = 3
H_MAX_BITS = math.log2(CODEBOOK_SIZE)  # = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load mono float32 audio and resample to target_sr. Returns shape [1, T]."""
    try:
        audio, sr = torchaudio.load(path)
    except Exception:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        audio = torch.from_numpy(data.T).contiguous()
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio.float()


def read_test_list(list_path: Path, max_samples: int):
    rows = []
    with open(list_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            raw = line.strip().lstrip("﻿")
            if not raw:
                continue
            audio = raw.split("\t")[0].strip()
            rows.append(audio)
            if len(rows) >= max_samples:
                break
    return rows


def build_model(device: torch.device):
    """Build SCIT-Speech-Base and load Exp2 best checkpoint."""
    cfg = load_json(SCIT_BASE_CONFIG)
    # Make NAS encoder config path absolute so we don't depend on cwd.
    cfg["nas_encoder_config"] = str(SCIT_BASE_NAS_ENCODER)
    model = build_scit(cfg).to(device).eval()
    missing, unexpected = load_scit_state(model, str(SCIT_BASE_CKPT))
    print(
        f"[model] loaded SCIT-Speech-Base: {len(missing)} missing / "
        f"{len(unexpected)} unexpected keys",
        flush=True,
    )
    return model


# ---------------------------------------------------------------------------
# Histogram + entropy
# ---------------------------------------------------------------------------
def empty_histograms(n_layers: int = N_Q, codebook_size: int = CODEBOOK_SIZE):
    return [np.zeros(codebook_size, dtype=np.int64) for _ in range(n_layers)]


def update_histograms(histograms, codes_lbt: torch.Tensor):
    """codes_lbt: LongTensor of shape [L, B, T]."""
    L = codes_lbt.shape[0]
    flat = codes_lbt.reshape(L, -1).cpu().numpy().astype(np.int64)
    for l in range(L):
        # bincount is fast, but cap to codebook_size in case of stray ids.
        bc = np.bincount(flat[l], minlength=CODEBOOK_SIZE)
        if bc.shape[0] > CODEBOOK_SIZE:
            bc = bc[:CODEBOOK_SIZE]
        histograms[l] += bc


def shannon_entropy_bits(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts.astype(np.float64) / float(total)
    nz = p > 0
    return float(-(p[nz] * np.log2(p[nz])).sum())


def per_layer_stats(counts: np.ndarray):
    total = int(counts.sum())
    n_used = int((counts > 0).sum())
    dead_code_ratio = float(1.0 - n_used / CODEBOOK_SIZE)
    if total > 0:
        sorted_counts = np.sort(counts)[::-1]
        top1_freq = float(sorted_counts[0] / total)
        top10_freq = float(sorted_counts[:10].sum() / total)
    else:
        top1_freq = float("nan")
        top10_freq = float("nan")
    H_bits = shannon_entropy_bits(counts)
    H_uniform_used_bits = math.log2(n_used) if n_used > 0 else float("nan")
    R_entropy_bps = H_bits * F_Q if not math.isnan(H_bits) else float("nan")
    R_naive_bps = H_MAX_BITS * F_Q  # per-layer naive: 10 * 50 = 500
    ratio = H_bits / H_MAX_BITS if not math.isnan(H_bits) else float("nan")
    return {
        "n_frames": total,
        "n_used": n_used,
        "dead_code_ratio": dead_code_ratio,
        "top1_freq": top1_freq,
        "top10_freq": top10_freq,
        "H_bits": H_bits,
        "H_uniform_used_bits": H_uniform_used_bits,
        "R_entropy_bps": R_entropy_bps,
        "R_naive_bps": R_naive_bps,
        "ratio_entropy_to_naive": ratio,
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def write_csv_rows(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------
@torch.no_grad()
def process_split(model, device, split_name: str, list_path: Path, max_samples: int):
    files = read_test_list(list_path, max_samples)
    print(f"[{split_name}] {len(files)} files (capped at {max_samples})", flush=True)
    histograms = empty_histograms()
    n_skipped = 0
    t0 = time.time()
    for i, p in enumerate(files):
        try:
            x = load_audio(p, SAMPLE_RATE).to(device).view(1, 1, -1)
            codes = model.encode(x, n_q=N_Q, st=0)  # [L, B, T]
            if codes.dim() != 3:
                raise RuntimeError(f"unexpected codes shape {tuple(codes.shape)}")
            update_histograms(histograms, codes.long())
        except Exception as e:
            n_skipped += 1
            print(f"[{split_name}] skip {p}: {type(e).__name__}: {e}", flush=True)
            continue
        if (i + 1) % 25 == 0 or (i + 1) == len(files):
            elapsed = time.time() - t0
            print(
                f"[{split_name}] processed {i + 1}/{len(files)} "
                f"(skipped={n_skipped}, elapsed={elapsed:.1f}s)",
                flush=True,
            )
    n_frames_per_layer = [int(h.sum()) for h in histograms]
    print(
        f"[{split_name}] done. frames/layer={n_frames_per_layer}, skipped={n_skipped}",
        flush=True,
    )
    return histograms, n_frames_per_layer, n_skipped


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def build_summary_md(per_split_stats, cumulative_rows):
    lines = []
    lines.append("# Exp16: Empirical RVQ Index Entropy (SCIT-Speech-Base)")
    lines.append("")
    lines.append(
        f"- Codebook size: {CODEBOOK_SIZE} (H_max = log2 = {H_MAX_BITS:.3f} bits/code)"
    )
    lines.append(f"- Frame rate f_q = {F_Q} Hz (latent rate)")
    lines.append(f"- Naive per-layer rate: {F_Q} * {H_MAX_BITS:.0f} = {F_Q * H_MAX_BITS:.0f} bps")
    lines.append("")
    lines.append("## Per-layer entropy")
    lines.append("")
    lines.append(
        "| split | L | n_frames | n_used | dead% | top1 | H (bits) | "
        "R_entropy (bps) | R_naive (bps) | H / H_max |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    biggest_gap = {"split": None, "L": None, "gap": -1.0}
    for split_name, layer_stats in per_split_stats.items():
        for L, s in enumerate(layer_stats, start=1):
            gap = (
                (H_MAX_BITS - s["H_bits"])
                if not math.isnan(s["H_bits"])
                else -1.0
            )
            if gap > biggest_gap["gap"]:
                biggest_gap = {"split": split_name, "L": L, "gap": gap}
            lines.append(
                f"| {split_name} | {L} | {s['n_frames']} | {s['n_used']} | "
                f"{s['dead_code_ratio'] * 100:.1f}% | {s['top1_freq']:.3f} | "
                f"{s['H_bits']:.3f} | {s['R_entropy_bps']:.1f} | "
                f"{s['R_naive_bps']:.1f} | {s['ratio_entropy_to_naive']:.3f} |"
            )
    lines.append("")
    lines.append("## Cumulative bits saved per L (entropy vs naive)")
    lines.append("")
    lines.append(
        "| split | L_cum | sum_H (bits) | R_entropy_cum (bps) | "
        "R_naive_cum (bps) | savings vs naive |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in cumulative_rows:
        lines.append(
            f"| {r['split']} | {r['L_cum']} | {r['sum_H_bits']:.3f} | "
            f"{r['R_entropy_bps_cum']:.1f} | {r['R_naive_bps_cum']:.1f} | "
            f"{r['savings_pct']:.2f}% |"
        )
    lines.append("")
    lines.append("## Paper-ready interpretation")
    lines.append("")
    if biggest_gap["split"] is not None:
        bg_split = biggest_gap["split"]
        bg_L = biggest_gap["L"]
        bg_gap = biggest_gap["gap"]
        first_split = next(iter(per_split_stats))
        per_layer = per_split_stats[first_split]
        l1_H = per_layer[0]["H_bits"] if per_layer else float("nan")
        l3_H = per_layer[-1]["H_bits"] if per_layer else float("nan")
        l1_savings_pct = (
            (1.0 - l1_H / H_MAX_BITS) * 100.0 if not math.isnan(l1_H) else float("nan")
        )
        l3_savings_pct = (
            (1.0 - l3_H / H_MAX_BITS) * 100.0 if not math.isnan(l3_H) else float("nan")
        )
        lines.append(
            f"On the {first_split} split, layer-1 indices have empirical entropy "
            f"H(I_1) = {l1_H:.2f} bits versus the codebook ceiling of "
            f"{H_MAX_BITS:.0f} bits, giving an entropy-coded rate of "
            f"{l1_H * F_Q:.0f} bps per layer instead of the naive {F_Q * H_MAX_BITS:.0f} bps "
            f"(a {l1_savings_pct:.1f}% reduction)."
        )
        lines.append(
            f"By layer 3 the per-layer entropy drops to H(I_3) = {l3_H:.2f} bits "
            f"({l3_savings_pct:.1f}% below the ceiling), with the largest "
            f"H_max - H gap of {bg_gap:.2f} bits observed at split={bg_split}, L={bg_L}."
        )
        lines.append(
            "These measurements support the paper's caveat that the 500*L bps "
            "figure used in section 3.1 / section 7 is a CONSERVATIVE upper bound: "
            "an entropy coder operating on the empirical index distribution would "
            "require strictly fewer bits per second than the naive bit-packed limit."
        )
    else:
        lines.append("(No per-layer statistics were collected.)")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args):
    if args.split == "both":
        splits = ["test-clean", "test-other"]
    else:
        splits = [args.split]

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    print(f"[init] device={device}", flush=True)

    model = build_model(device)

    histogram_rows = []
    entropy_rows = []
    cumulative_rows = []
    per_split_stats = {}
    n_frames_per_split = {}

    for split_name in splits:
        list_path = EXP7_TEST_LISTS[split_name]
        if not list_path.exists():
            print(f"[{split_name}] missing list: {list_path}", flush=True)
            continue
        histograms, n_frames_per_layer, _ = process_split(
            model, device, split_name, list_path, args.max_samples
        )
        n_frames_per_split[split_name] = n_frames_per_layer

        layer_stats = []
        for L_idx, counts in enumerate(histograms, start=1):
            stats = per_layer_stats(counts)
            layer_stats.append(stats)
            entropy_rows.append({
                "split": split_name,
                "L": L_idx,
                **stats,
            })
            total = max(int(counts.sum()), 1)
            for code_id, c in enumerate(counts):
                if c == 0:
                    continue
                histogram_rows.append({
                    "split": split_name,
                    "L": L_idx,
                    "code_id": code_id,
                    "count": int(c),
                    "prob": float(c) / float(total),
                })
        per_split_stats[split_name] = layer_stats

        # cumulative-rate-per-L
        sum_H = 0.0
        for L_cum in range(1, N_Q + 1):
            H_l = layer_stats[L_cum - 1]["H_bits"]
            sum_H = sum_H + H_l if not math.isnan(H_l) else sum_H
            R_entropy_cum = sum_H * F_Q
            R_naive_cum = 500.0 * L_cum
            savings_pct = (
                (R_naive_cum - R_entropy_cum) / R_naive_cum * 100.0
                if R_naive_cum > 0
                else float("nan")
            )
            cumulative_rows.append({
                "split": split_name,
                "L_cum": L_cum,
                "sum_H_bits": sum_H,
                "R_entropy_bps_cum": R_entropy_cum,
                "R_naive_bps_cum": R_naive_cum,
                "savings_pct": savings_pct,
            })

    # Write CSVs
    write_csv_rows(
        METRICS_DIR / "index_histogram_per_layer.csv",
        ["split", "L", "code_id", "count", "prob"],
        histogram_rows,
    )
    write_csv_rows(
        METRICS_DIR / "entropy_per_layer.csv",
        [
            "split", "L", "n_frames", "n_used", "dead_code_ratio",
            "top1_freq", "top10_freq", "H_bits", "H_uniform_used_bits",
            "R_entropy_bps", "R_naive_bps", "ratio_entropy_to_naive",
        ],
        entropy_rows,
    )
    write_csv_rows(
        METRICS_DIR / "cumulative_rate_per_L.csv",
        ["split", "L_cum", "sum_H_bits", "R_entropy_bps_cum", "R_naive_bps_cum", "savings_pct"],
        cumulative_rows,
    )

    # Summary markdown
    md = build_summary_md(per_split_stats, cumulative_rows)
    (REPORTS_DIR / "entropy_rate_summary.md").write_text(md, encoding="utf-8")

    status = {
        "split_done": list(per_split_stats.keys()),
        "n_frames_per_split": n_frames_per_split,
        "exit": "ok",
    }
    print(json.dumps(status), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["test-clean", "test-other", "both"], default="both")
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    log_path = LOGS_DIR / "encode.log"
    # Tee stdout/stderr to logs/encode.log while still printing to console.
    class Tee(io.TextIOBase):
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for st in self.streams:
                try:
                    st.write(s)
                    st.flush()
                except Exception:
                    pass
            return len(s)
        def flush(self):
            for st in self.streams:
                try:
                    st.flush()
                except Exception:
                    pass

    f = open(log_path, "w", encoding="utf-8")
    tee_out = Tee(sys.stdout, f)
    tee_err = Tee(sys.stderr, f)
    try:
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            try:
                run(args)
            except Exception:
                traceback.print_exc()
                err_status = {"split_done": [], "n_frames_per_split": {}, "exit": "error"}
                print(json.dumps(err_status), flush=True)
                raise
    finally:
        f.close()


if __name__ == "__main__":
    main()
