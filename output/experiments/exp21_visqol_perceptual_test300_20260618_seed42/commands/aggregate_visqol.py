"""Aggregate ViSQOL MOS-LQO results: join shard outputs onto the key,
then per (exp, split, method, codec_setting) compute mean + 95% bootstrap CI
(B=10000, seed=42) mirroring exp12/exp20 analyze scripts. Also write the
joined per-sample CSV and a markdown summary focused on the in-band table rows.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SETUP = Path(r"H:/H-CODE/speechtokenizer/output/experiments/_visqol_setup")
N_SHARDS = 7
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


def norm(p):
    return str(p).replace("\\", "/").strip().lower()


def main():
    key = pd.read_csv(SETUP / "visqol_key.csv")
    key["degraded_n"] = key["degraded"].map(norm)

    # gather shard results
    parts = []
    for s in range(N_SHARDS):
        f = SETUP / f"results_{s}.csv"
        if not f.exists():
            print(f"WARN missing {f}")
            continue
        parts.append(pd.read_csv(f))
    res = pd.concat(parts, ignore_index=True)
    # visqol results columns: reference, degraded, moslqo
    res.columns = [c.strip().lower() for c in res.columns]
    moscol = "moslqo" if "moslqo" in res.columns else res.columns[-1]
    res["degraded_n"] = res["degraded"].map(norm)
    res = res[["degraded_n", moscol]].rename(columns={moscol: "visqol"})

    merged = key.merge(res, on="degraded_n", how="left")
    n_missing = merged["visqol"].isna().sum()
    print(f"merged rows: {len(merged)}, missing visqol: {n_missing}")
    merged.drop(columns=["degraded_n"]).to_csv(SETUP / "visqol_per_sample.csv", index=False)

    # per-group summary
    out = []
    for (exp, split, method, setting), g in merged.groupby(["exp", "split", "method", "codec_setting"], dropna=False):
        mean, lo, hi = boot_ci(g["visqol"].values)
        out.append({
            "exp": exp, "split": split, "method": method, "codec_setting": setting,
            "n_samples": len(g), "n_valid": int(g["visqol"].notna().sum()),
            "visqol_mean": mean, "visqol_ci_low": lo, "visqol_ci_high": hi,
        })
    summ = pd.DataFrame(out).sort_values(["split", "method", "codec_setting"])
    summ.to_csv(SETUP / "visqol_per_method_summary.csv", index=False)
    print(f"wrote visqol_per_method_summary.csv ({len(summ)} groups)")
    # quick sanity print: PCM passthrough should be near-max
    pcm = summ[summ.method == "pcm"]
    if len(pcm):
        print("PCM passthrough (sanity, expect ~4.6-5.0):")
        print(pcm[["split", "visqol_mean", "visqol_ci_low", "visqol_ci_high"]].to_string(index=False))
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
