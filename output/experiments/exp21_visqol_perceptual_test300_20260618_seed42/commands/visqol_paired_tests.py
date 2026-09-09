"""Paired ViSQOL tests (LCA - baseline) for the paper's key same-rate pairs,
both splits, mirroring exp12's lca_vs_baselines_pairwise protocol:
paired by sample_id, scipy.stats.wilcoxon + bootstrap CI on the paired diff
(B=10000, seed=42). Operates on visqol_per_sample.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SETUP = Path(r"H:/H-CODE/speechtokenizer/output/experiments/_visqol_setup")
B = 10000
SEED = 42

# (label, lca method+setting, baseline method+setting)
PAIRS = [
    ("L=1 vs DAC n_q=1 (500 bps)", ("scit_lca", "L=1"), ("dac", "n_q_1")),
    ("L=2 vs DAC n_q=2 (1000 bps)", ("scit_lca", "L=2"), ("dac", "n_q_2")),
    ("L=3 vs DAC n_q=3 (1500 bps)", ("scit_lca", "L=3"), ("dac", "n_q_3")),
    ("L=3 vs EnCodec 1.5k (1500 bps)", ("scit_lca", "L=3"), ("encodec", "bw1.5kbps_n_cb2")),
    ("L=3 vs Opus 6k (cross-rate)", ("scit_lca", "L=3"), ("opus", "opus_6000bps")),
    ("L=1 vs Codec2 700C (~815 bps)", ("scit_lca", "L=1"), ("codec2", "codec2_700bps")),
    ("L=2 vs Codec2 1200 (~1.2k)", ("scit_lca", "L=2"), ("codec2", "codec2_1200bps")),
    ("L=3 vs Codec2 1300 (~1.4k)", ("scit_lca", "L=3"), ("codec2", "codec2_1300bps")),
    # LCA vs Base self-comparison
    ("L=1 LCA vs Base", ("scit_lca", "L=1"), ("scit_base", "L=1")),
    ("L=2 LCA vs Base", ("scit_lca", "L=2"), ("scit_base", "L=2")),
    ("L=3 LCA vs Base", ("scit_lca", "L=3"), ("scit_base", "L=3")),
]


def boot_ci_diff(d, b=B, seed=SEED):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(b, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    df = pd.read_csv(SETUP / "visqol_per_sample.csv")
    rows = []
    for split in ["test-clean_300", "test-other_300"]:
        s = df[df.split == split]
        for label, (m1, c1), (m2, c2) in PAIRS:
            a = s[(s.method == m1) & (s.codec_setting == c1)][["sample_id", "visqol"]].rename(columns={"visqol": "lca"})
            b_ = s[(s.method == m2) & (s.codec_setting == c2)][["sample_id", "visqol"]].rename(columns={"visqol": "base"})
            mrg = a.merge(b_, on="sample_id")
            if len(mrg) == 0:
                continue
            diff = (mrg["lca"] - mrg["base"]).values
            mean, lo, hi = boot_ci_diff(diff)
            try:
                p = wilcoxon(mrg["lca"].values, mrg["base"].values).pvalue
            except ValueError:
                p = float("nan")
            rows.append({
                "split": split, "pair": label, "n": len(mrg),
                "lca_mean": mrg["lca"].mean(), "base_mean": mrg["base"].mean(),
                "d_visqol": mean, "ci_low": lo, "ci_high": hi, "wilcoxon_p": p,
            })
    res = pd.DataFrame(rows)
    res.to_csv(SETUP / "visqol_paired_tests.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    for split in ["test-clean_300", "test-other_300"]:
        print(f"\n===== {split} (ΔViSQOL = LCA - baseline) =====")
        for _, r in res[res.split == split].iterrows():
            sig = "p<0.001" if r["wilcoxon_p"] < 0.001 else f"p={r['wilcoxon_p']:.3f}"
            print(f"  {r['pair']:34s} LCA={r['lca_mean']:.3f} base={r['base_mean']:.3f} "
                  f"Δ={r['d_visqol']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] {sig}")


if __name__ == "__main__":
    main()
