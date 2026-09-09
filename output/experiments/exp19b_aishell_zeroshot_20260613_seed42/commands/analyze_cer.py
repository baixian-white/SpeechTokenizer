"""AISHELL Chinese CER analysis: per (model,L) mean CER + LCA-Base delta."""
import csv
import collections
import math

CSV = r"h:\H-CODE\speechtokenizer\output\experiments\exp19b_aishell_zeroshot_20260613_seed42\metrics\asr_cer_zh_results.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
print("rows:", len(rows), "unique sid:", len(set(r["sample_id"] for r in rows)))

def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

agg = collections.defaultdict(lambda: collections.defaultdict(list))
nan = collections.Counter()
for r in rows:
    for m in ("cer_vs_gt", "cer_vs_original_whisper"):
        v = fnum(r[m])
        if math.isnan(v):
            nan[m] += 1
        else:
            agg[(r["model"], r["L"])][m].append(v)
print("NaN:", dict(nan) if nan else "none")

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
print(f"\n{'model':5} {'L':2} {'CER_vs_GT':>10} {'CER_vs_origWhisper':>18}")
for k in sorted(agg):
    m = agg[k]
    print(f"{k[0]:5} {k[1]:2} {mean(m['cer_vs_gt']):10.4f} {mean(m['cer_vs_original_whisper']):18.4f}")

print("\n=== LCA - Base delta CER (negative = LCA better) ===")
for L in ("1","2","3"):
    b,l = agg[("base",L)], agg[("lca",L)]
    print(f"L={L}: dCER_vs_GT={mean(l['cer_vs_gt'])-mean(b['cer_vs_gt']):+.4f} "
          f"dCER_vs_orig={mean(l['cer_vs_original_whisper'])-mean(b['cer_vs_original_whisper']):+.4f}")

# original-audio whisper CER floor (how well whisper does on clean AISHELL = lower bound)
print("\nNote: cer_vs_gt includes whisper's own errors on Chinese; cer_vs_original_whisper isolates codec degradation.")
