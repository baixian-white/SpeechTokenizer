"""exp5c analysis: does consistency loss (V4) reduce the perturbation gap vs V3 (no consistency),
without hurting clean full-depth quality?

gap = comm_mel(perturbed channel) - comm_mel(clean) at each (L, channel, step).
A smaller gap = perturbed decode is closer to clean decode = consistency mechanism working.

CAVEAT (printed): V0-V3 are the exp5 factorial run; V4 is the exp3-v2 run (different run, not a
clean same-run A/B). This is a dynamics-level corroboration, not a controlled ablation; the clean
controlled V3-vs-V4 comparison lives in exp5b's n=256 eval CSVs.
"""
import csv
import collections

CSV = r"h:\H-CODE\speechtokenizer\output\experiments\exp5c_consistency_dynamics_20260611_seed42\metrics\robustness_gap_dynamics.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))

def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

# group gap by (variant, L, channel) -> list over steps; report final-window mean (last 3 dev points)
series = collections.defaultdict(list)
for r in rows:
    series[(r["variant"], r["L"], r["channel"])].append((int(r["step"]), fnum(r["gap"])))

def tail_mean(pairs, k=3):
    pairs = sorted(pairs)
    vals = [v for _s, v in pairs[-k:]]
    return sum(vals) / len(vals) if vals else float("nan")

channels = ["dropout-mid", "dropout-high", "substitution-mid", "substitution-high"]
print("=== perturbation gap (comm_mel_pert - comm_mel_clean), final-window mean (last 3 dev pts) ===")
print(f"{'L':2} {'channel':18} {'V3_gap':>9} {'V4_gap':>9} {'V4-V3':>9}")
v3v4 = []
for L in ("1", "2", "3"):
    for ch in channels:
        v3 = series.get(("V3", L, ch))
        v4 = series.get(("V4", L, ch))
        if not v3 or not v4:
            continue
        g3, g4 = tail_mean(v3), tail_mean(v4)
        d = g4 - g3
        v3v4.append(d)
        print(f"{L:2} {ch:18} {g3:9.4f} {g4:9.4f} {d:+9.4f}")

import statistics
print(f"\nV4-V3 gap delta: mean={statistics.mean(v3v4):+.4f} "
      f"(negative = V4 has SMALLER gap = consistency reduces perturbation deviation)")
print(f"cells where V4 gap < V3 gap: {sum(1 for d in v3v4 if d < 0)}/{len(v3v4)}")

print("\n=== CAVEAT: V0-V3 = exp5 factorial run; V4 = exp3-v2 run (different run). ===")
print("Dynamics-level corroboration only; controlled n=256 ablation is exp5b §11.1.")
