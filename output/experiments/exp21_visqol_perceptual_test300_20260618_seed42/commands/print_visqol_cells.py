"""Print ViSQOL per-method cells formatted for the paper, grouped by split.
Reads visqol_per_method_summary.csv (produced by aggregate_visqol.py).
Output is a lookup I transcribe into the paper tables (1a/1b, D.1, D.3).
"""
from pathlib import Path
import pandas as pd

SETUP = Path(r"H:/H-CODE/speechtokenizer/output/experiments/_visqol_setup")


def cell(r):
    return f"{r['visqol_mean']:.3f} [{r['visqol_ci_low']:.3f}, {r['visqol_ci_high']:.3f}]"


def main():
    df = pd.read_csv(SETUP / "visqol_per_method_summary.csv")
    for split in ["test-clean_300", "test-other_300"]:
        print(f"\n===== {split} =====")
        sub = df[df.split == split].copy()
        # stable ordering: method then codec_setting
        for _, r in sub.sort_values(["method", "codec_setting"]).iterrows():
            print(f"  {r['method']:10s} | {str(r['codec_setting']):28s} | n={int(r['n_valid']):3d} | {cell(r)}")


if __name__ == "__main__":
    main()
