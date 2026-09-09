"""exp20 analysis: per-method mean + 95% bootstrap CI for AMR-WB / Codec2,
both splits, plus a nearest-rate reference comparison against SCIT-LCA
(pulled from exp12 per_method_summary). Self-contained; stdlib + numpy/pandas.

Outputs per split:
  runs/<split>/metrics/per_method_summary.csv
  reports/exp20_summary.md  (combined)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"H:/H-CODE/speechtokenizer/output/experiments/exp20_amrwb_codec2_test300_20260614_seed42")
EXP12 = Path(r"H:/H-CODE/speechtokenizer/output/experiments/exp12_baseline_comparison_test300_20260610_seed42/runs")
SPLITS = ["test-clean_300", "test-other_300"]
METRICS = ["wave_l1", "mel_l1", "si_snr_db", "corr", "stoi", "pesq_wb"]
RNG_SEED = 42
B = 10000


def boot_ci(x, b=B, seed=RNG_SEED):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(b, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(split):
    df = pd.read_csv(ROOT / "runs" / split / "metrics" / "audio_quality_results.csv")
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    out = []
    for (method, setting), g in df.groupby(["method", "codec_setting"]):
        row = {"method": method, "codec_setting": setting, "n_samples": len(g)}
        for m in METRICS:
            mean, lo, hi = boot_ci(g[m].values)
            row[f"{m}_mean"], row[f"{m}_ci_low"], row[f"{m}_ci_high"] = mean, lo, hi
        out.append(row)
    res = pd.DataFrame(out)
    res.to_csv(ROOT / "runs" / split / "metrics" / "per_method_summary.csv", index=False)
    return res


def fmt(r, m):
    return f"{r[m+'_mean']:.3f} [{r[m+'_ci_low']:.3f}, {r[m+'_ci_high']:.3f}]"


def main():
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    lines = ["# exp20: AMR-WB + Codec2 baselines (test-clean/other_300)\n",
             "ffmpeg 8.1.1 gyan.dev full build (libvo_amrwbenc + libcodec2). "
             "Codec2 is 8 kHz narrowband (16k->8k->16k); AMR-WB is 16 kHz wideband. "
             "n=300/split, 95% bootstrap CI (B=10000, seed=42). Original wavs reused from exp12.\n"]
    for split in SPLITS:
        res = summarize(split)
        # SCIT-LCA reference rows from exp12
        ex12 = pd.read_csv(EXP12 / split / "metrics" / "per_method_summary.csv")
        lca = ex12[ex12.method == "scit_lca"].copy()
        lines.append(f"\n## {split}\n")
        lines.append("**Codec2 (区间内 / 跨区间)**\n")
        lines.append("| 档位 | 实际码率(bps) | mel-L1 | STOI | PESQ-WB |")
        lines.append("|---|---:|---|---|---|")
        pay = pd.read_csv(ROOT / "runs" / split / "metrics" / "payload_summary.csv")
        rate = pay.groupby("codec_setting")["packed_payload_bps"].mean().round(0)
        c2 = res[res.method == "codec2"].copy()
        c2["bps"] = c2["codec_setting"].map(rate)
        for _, r in c2.sort_values("bps").iterrows():
            lines.append(f"| {r['codec_setting']} | {int(r['bps'])} | {fmt(r,'mel_l1')} | {fmt(r,'stoi')} | {fmt(r,'pesq_wb')} |")
        lines.append("\n**SCIT-LCA 参照（exp12 同集）**\n")
        lines.append("| 操作点 | 码率(bps) | mel-L1 | STOI | PESQ-WB |")
        lines.append("|---|---:|---|---|---|")
        lca_rate = {"L=1": 500, "L=2": 1000, "L=3": 1500}
        for _, r in lca.iterrows():
            lines.append(f"| SCIT-LCA {r['codec_setting']} | {lca_rate.get(r['codec_setting'],'')} | "
                         f"{fmt(r,'mel_l1')} | {fmt(r,'stoi')} | {fmt(r,'pesq_wb')} |")
        lines.append("\n**AMR-WB (全 9 档, 均在区间外 ≥6.6 kbps)**\n")
        lines.append("| 档位 | 实际码率(bps) | mel-L1 | STOI | PESQ-WB |")
        lines.append("|---|---:|---|---|---|")
        aw = res[res.method == "amrwb"].copy()
        aw["bps"] = aw["codec_setting"].map(rate)
        for _, r in aw.sort_values("bps").iterrows():
            lines.append(f"| {r['codec_setting']} | {int(r['bps'])} | {fmt(r,'mel_l1')} | {fmt(r,'stoi')} | {fmt(r,'pesq_wb')} |")
    (ROOT / "reports" / "exp20_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote reports/exp20_summary.md")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
