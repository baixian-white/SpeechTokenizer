"""AISHELL objective analysis + cross-corpus comparison."""
import csv
import collections
import math

CSV = r"h:\H-CODE\speechtokenizer\output\experiments\exp19b_aishell_zeroshot_20260613_seed42\eval_aishell_300\metrics\full_clean_results.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
print("total rows:", len(rows), "unique sid:", len(set(r["sample_id"] for r in rows)))

def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

metrics = ["mel_l1", "stoi", "pesq_wb", "si_snr_db"]
bad = collections.Counter()
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    for m in metrics:
        v = fnum(r[m])
        if math.isnan(v) or math.isinf(v):
            bad[m] += 1
        agg[(r["model"], r["L"])][m].append(v)
print("NaN/Inf:", dict(bad) if bad else "none")

def mean(xs): return sum(xs) / len(xs)
print(f"\n{'model':5} {'L':2} {'mel_l1':>8} {'stoi':>7} {'pesq_wb':>8} {'si_snr':>8}")
for k in sorted(agg):
    m = agg[k]
    print(f"{k[0]:5} {k[1]:2} {mean(m['mel_l1']):8.4f} {mean(m['stoi']):7.4f} {mean(m['pesq_wb']):8.4f} {mean(m['si_snr_db']):8.3f}")

print("\n=== LCA - Base delta (AISHELL) ===")
for L in ("1", "2", "3"):
    b, l = agg[("base", L)], agg[("lca", L)]
    print(f"L={L}: Dmel={mean(l['mel_l1'])-mean(b['mel_l1']):+.4f} Dstoi={mean(l['stoi'])-mean(b['stoi']):+.4f} "
          f"Dpesq={mean(l['pesq_wb'])-mean(b['pesq_wb']):+.4f} Dsisnr={mean(l['si_snr_db'])-mean(b['si_snr_db']):+.3f}")
