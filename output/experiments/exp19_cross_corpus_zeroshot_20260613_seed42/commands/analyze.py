"""exp19 integrity check + VCTK-vs-LibriSpeech comparison.
Verifies row breakdown, NaN/Inf, and computes per-(model,L) means; loads exp7 LibriSpeech
test means for cross-corpus comparison. CPU only."""
import csv
import collections
import math

EXP19 = r"h:\H-CODE\speechtokenizer\output\experiments\exp19_cross_corpus_zeroshot_20260613_seed42\eval_vctk_300\metrics\full_clean_results.csv"


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def fnum(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


rows = load(EXP19)
print(f"total rows: {len(rows)}")
c = collections.Counter((r["model"], r["L"]) for r in rows)
for k, v in sorted(c.items()):
    print(f"  {k}: {v}")
print("unique sample_id:", len(set(r["sample_id"] for r in rows)))

# NaN/Inf check on key metrics
metrics = ["mel_l1", "stoi", "pesq_wb", "si_snr_db"]
bad = collections.Counter()
for r in rows:
    for m in metrics:
        v = fnum(r[m])
        if math.isnan(v) or math.isinf(v):
            bad[m] += 1
print("NaN/Inf counts:", dict(bad) if bad else "none")

# per (model,L) means
print("\n=== VCTK 300 per (model,L) means ===")
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    key = (r["model"], r["L"])
    for m in metrics:
        agg[key][m].append(fnum(r[m]))
print(f"{'model':6} {'L':2} {'mel_l1':>8} {'stoi':>7} {'pesq_wb':>8} {'si_snr_db':>9}")
def mean(xs): return sum(xs) / len(xs)
for key in sorted(agg):
    m = agg[key]
    print(f"{key[0]:6} {key[1]:2} {mean(m['mel_l1']):8.4f} {mean(m['stoi']):7.4f} "
          f"{mean(m['pesq_wb']):8.4f} {mean(m['si_snr_db']):9.3f}")

# LCA - Base delta per L
print("\n=== LCA - Base delta (VCTK) ===")
for L in ("1", "2", "3"):
    b = agg[("base", L)]; l = agg[("lca", L)]
    if not b or not l:
        continue
    print(f"L={L}: Dmel={mean(l['mel_l1'])-mean(b['mel_l1']):+.4f} "
          f"Dstoi={mean(l['stoi'])-mean(b['stoi']):+.4f} "
          f"Dpesq={mean(l['pesq_wb'])-mean(b['pesq_wb']):+.4f} "
          f"Dsisnr={mean(l['si_snr_db'])-mean(b['si_snr_db']):+.3f}")
