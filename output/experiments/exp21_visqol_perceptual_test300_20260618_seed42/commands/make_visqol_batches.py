"""Prepare ViSQOL batch-input CSVs from the exp12/exp20 per-sample metric CSVs.

ViSQOL CLI batch mode wants a CSV with columns: reference, degraded
We also keep a sidecar mapping (key.csv) so we can join MOS-LQO back to
(exp, split, method, codec_setting, sample_id) by the unique degraded path.

Paths in source CSVs are Windows-style; exp12 is repo-relative, exp20 is H:\ absolute.
Normalize everything to WSL-absolute POSIX: /mnt/h/H-CODE/speechtokenizer/...
Output batch CSVs are sharded for parallel ViSQOL runs.
"""
from pathlib import Path
import pandas as pd

REPO_WIN = r"H:/H-CODE/speechtokenizer"
REPO_WSL = "/mnt/h/H-CODE/speechtokenizer"
OUTDIR = Path(REPO_WIN) / "output/experiments/_visqol_setup"
N_SHARDS = 7  # parallel ViSQOL processes

SOURCES = [
    ("exp12", "test-clean_300", "output/experiments/exp12_baseline_comparison_test300_20260610_seed42/runs/test-clean_300/metrics/audio_quality_results.csv"),
    ("exp12", "test-other_300", "output/experiments/exp12_baseline_comparison_test300_20260610_seed42/runs/test-other_300/metrics/audio_quality_results.csv"),
    ("exp20", "test-clean_300", "output/experiments/exp20_amrwb_codec2_test300_20260614_seed42/runs/test-clean_300/metrics/audio_quality_results.csv"),
    ("exp20", "test-other_300", "output/experiments/exp20_amrwb_codec2_test300_20260614_seed42/runs/test-other_300/metrics/audio_quality_results.csv"),
]


def to_wsl(p: str) -> str:
    p = str(p).replace("\\", "/")
    low = p.lower()
    if low.startswith("h:/h-code/speechtokenizer"):
        return REPO_WSL + p[len("H:/H-CODE/speechtokenizer"):]
    if low.startswith("output/"):
        return REPO_WSL + "/" + p
    raise ValueError(f"unexpected path: {p}")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for exp, split, rel in SOURCES:
        df = pd.read_csv(Path(REPO_WIN) / rel)
        for _, r in df.iterrows():
            rows.append({
                "exp": exp,
                "split": split,
                "method": r["method"],
                "codec_setting": r["codec_setting"],
                "sample_id": r["sample_id"],
                "reference": to_wsl(r["original_path"]),
                "degraded": to_wsl(r["decoded_path"]),
            })
    full = pd.DataFrame(rows)
    # unique degraded paths are the join key; assert uniqueness
    assert full["degraded"].is_unique, "degraded paths not unique!"
    full.to_csv(OUTDIR / "visqol_key.csv", index=False)
    print(f"total pairs: {len(full)}")

    # shard the (reference,degraded) batch input
    for s in range(N_SHARDS):
        sub = full.iloc[s::N_SHARDS][["reference", "degraded"]]
        sub.to_csv(OUTDIR / f"batch_{s}.csv", index=False)
        print(f"shard {s}: {len(sub)} pairs -> batch_{s}.csv")


if __name__ == "__main__":
    main()
