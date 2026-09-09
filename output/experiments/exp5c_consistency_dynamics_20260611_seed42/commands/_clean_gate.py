import csv, collections, statistics
rows = list(csv.DictReader(open(r"metrics/robustness_gap_dynamics.csv", encoding="utf-8")))
clean = collections.defaultdict(dict)
for r in rows:
    clean[(r["variant"], r["L"])][int(r["step"])] = float(r["comm_mel_clean"])
for L in ("1","2","3"):
    v3 = sorted(clean[("V3",L)].items()); v4 = sorted(clean[("V4",L)].items())
    m3 = statistics.mean([v for s,v in v3[-3:]]); m4 = statistics.mean([v for s,v in v4[-3:]])
    print(f"L={L}: V3_clean={m3:.4f} V4_clean={m4:.4f} delta={m4-m3:+.4f}")
