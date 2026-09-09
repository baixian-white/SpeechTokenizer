"""Diagnose Opus 6/8 kbps mel_l1 anomaly in exp12 test-clean_300.

Looks at length alignment between original 16 kHz wav and decoded Opus wav for the
worst samples by mel_l1, and reports whether libopus added a pre-roll / encoder delay
that the run_opus_baseline.py truncate-to-min-length step does not correct.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

RUN = Path(__file__).resolve().parents[1] / "runs" / "test-clean_300"
CSV = RUN / "metrics" / "audio_quality_results.csv"

df = pd.read_csv(CSV)
out = {}
for setting in ["opus_6000bps", "opus_8000bps", "opus_12000bps", "opus_16000bps", "opus_24000bps"]:
    sub = df[(df.method == "opus") & (df.codec_setting == setting)].sort_values("mel_l1", ascending=False)
    out[setting] = {
        "n": int(len(sub)),
        "mel_l1_min": float(sub.mel_l1.min()),
        "mel_l1_median": float(sub.mel_l1.median()),
        "mel_l1_max": float(sub.mel_l1.max()),
        "stoi_median": float(sub.stoi.median()),
        "pesq_wb_median": float(sub.pesq_wb.median()),
        "si_snr_median": float(sub.si_snr_db.median()),
    }

# Now align the worst Opus 6 kbps sample.
op6 = df[(df.method == "opus") & (df.codec_setting == "opus_6000bps")].sort_values("mel_l1", ascending=False)
align_rows = []
for _, row in op6.head(3).iterrows():
    o, sr1 = sf.read(row.original_path)
    e, sr2 = sf.read(row.decoded_path)
    if o.ndim > 1:
        o = o.mean(axis=1)
    if e.ndim > 1:
        e = e.mean(axis=1)
    L = min(len(o), len(e))
    diff = e[:L] - o[:L]
    rms = float(np.sqrt(np.mean(diff ** 2)))
    # Try to find best alignment within [-1024, 1024] samples by minimizing RMS
    best_lag = 0
    best_rms = rms
    for lag in range(-1024, 1025, 8):
        if lag >= 0:
            a = o[: len(o) - lag]
            b = e[lag : lag + len(a)]
        else:
            b = e[: len(e) + lag]
            a = o[-lag : -lag + len(b)]
        if len(a) == 0:
            continue
        L2 = min(len(a), len(b))
        d = b[:L2] - a[:L2]
        r = float(np.sqrt(np.mean(d ** 2)))
        if r < best_rms:
            best_rms = r
            best_lag = lag
    align_rows.append({
        "sample_id": row.sample_id,
        "mel_l1_reported": float(row.mel_l1),
        "len_orig": int(len(o)),
        "len_decoded": int(len(e)),
        "len_diff_samples": int(len(e) - len(o)),
        "len_diff_ms": float((len(e) - len(o)) / sr1 * 1000.0),
        "rms_naive": rms,
        "rms_best_aligned": best_rms,
        "best_lag_samples": int(best_lag),
        "rms_reduction_pct": (rms - best_rms) / rms * 100.0 if rms > 0 else 0.0,
    })

print(json.dumps({"per_setting_distribution": out, "alignment_top3_worst_op6kbps": align_rows}, indent=2))
