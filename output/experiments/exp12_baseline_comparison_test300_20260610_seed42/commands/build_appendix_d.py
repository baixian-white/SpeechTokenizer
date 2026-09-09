"""Build Appendix D markdown for exp12: full per-method table (both splits) +
full paired Wilcoxon table. Writes a markdown fragment to stdout-captured file.
"""
from pathlib import Path
import pandas as pd

ROOT = Path("H:/H-CODE/speechtokenizer/output/experiments/exp12_baseline_comparison_test300_20260610_seed42/runs")
OUT = Path("H:/H-CODE/speechtokenizer/output/experiments/exp12_baseline_comparison_test300_20260610_seed42/reports/appendix_d_fragment.md")

# bitrate map
def rate_map(split):
    pay = pd.read_csv(ROOT / split / "metrics" / "payload_summary.csv")
    return pay.groupby(["method", "codec_setting"])["packed_payload_bps"].mean().round(0).astype(int)

# friendly method/setting name
def name(m, s):
    if m == "scit_lca": return f"SCIT-LCA {s}"
    if m == "scit_base": return f"SCIT-Base {s}"
    if m == "dac": return f"DAC {s.replace('n_q_', 'n_q=')}"
    if m == "encodec":
        kb = s.split("_")[0].replace("bw", "").replace("kbps", "")
        return f"EnCodec {kb} kbps"
    if m == "opus":
        return f"Opus {int(s.replace('opus_', '').replace('bps', ''))//1000} kbps"
    if m == "pcm": return "PCM (无损)"
    return f"{m} {s}"

# method ordering by bitrate then family
fam_order = {"scit_base": 0, "scit_lca": 1, "dac": 2, "encodec": 3, "opus": 4, "pcm": 5}

lines = []
lines.append("## 附录 D：exp12 同码率对照完整结果（n=300）\n")
lines.append("本附录给出 §5.1 同码率对照的完整 23 方法 × 2 数据集 per-method 表，"
             "以及 SCIT-Speech-LCA 相对各同码率基线的配对 Wilcoxon 检验。"
             "统计协议：每集独立 bootstrap（B=10 000，seed=42，percentile CI）；"
             "同码率配对采用 `scipy.stats.wilcoxon`，配对单位为同一样本，差值方向为 LCA − baseline。\n")

for split in ["test-clean_300", "test-other_300"]:
    pm = pd.read_csv(ROOT / split / "metrics" / "per_method_summary.csv")
    rm = rate_map(split)
    pm["bps"] = pm.apply(lambda r: rm.get((r["method"], r["codec_setting"]), -1), axis=1)
    pm["fam"] = pm["method"].map(fam_order)
    pm = pm.sort_values(["bps", "fam"])
    lines.append(f"\n**表 D.1（{split}）：完整 per-method 均值与 95% bootstrap CI**\n")
    lines.append("| 方法 | 实际码率 (bps) | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR (dB) |")
    lines.append("|---|---:|---|---|---|---|")
    for _, r in pm.iterrows():
        def cell(p):
            return f"{r[p+'_mean']:.3f} [{r[p+'_ci_low']:.3f}, {r[p+'_ci_high']:.3f}]"
        si = f"{r['si_snr_db_mean']:.2f} [{r['si_snr_db_ci_low']:.2f}, {r['si_snr_db_ci_high']:.2f}]"
        lines.append(f"| {name(r['method'], r['codec_setting'])} | {int(r['bps'])} | "
                     f"{cell('mel_l1')} | {cell('stoi')} | {cell('pesq_wb')} | {si} |")

# pairwise
lines.append("\n**表 D.2：SCIT-Speech-LCA vs 同码率基线的配对 Wilcoxon（差值 = LCA − baseline）**\n")
lines.append("| 数据集 | 操作点 | 对照 | Δmel-L1 [95% CI] | ΔSTOI [95% CI] | ΔPESQ-WB [95% CI] | p (各指标) |")
lines.append("|---|---|---|---|---|---|---|")
for split in ["test-clean_300", "test-other_300"]:
    pw = pd.read_csv(ROOT / split / "metrics" / "lca_vs_baselines_pairwise.csv", skiprows=1)
    for _, r in pw.iterrows():
        def d(metric):
            return f"{r[metric+'_mean_diff']:+.4f} [{r[metric+'_ci_low']:+.4f}, {r[metric+'_ci_high']:+.4f}]"
        comp = str(r["comparison"]).replace("scit_lca/", "").replace("dac/n_q_", "DAC n_q=").replace(
            "encodec/bw1.5kbps_n_cb2", "EnCodec 1.5k").replace("opus/opus_6000bps", "Opus 6k")
        rl = str(r["rate_label"]).replace("_", " ")
        if str(r["cross_rate"]).strip().lower() == "yes":
            rl = rl + "（跨码率）"
        # all p reported <0.001 (smallest ~6e-51); report literal max p across 3 metrics
        ps = [r["mel_l1_wilcoxon_p"], r["stoi_wilcoxon_p"], r["pesq_wb_wilcoxon_p"]]
        pmax = max(ps)
        pstr = "<0.001" if pmax < 1e-3 else f"{pmax:.3g}"
        lines.append(f"| {split} | {rl} | {comp} | {d('mel_l1')} | {d('stoi')} | {d('pesq_wb')} | {pstr} |")

lines.append("\n注：跨码率对 Opus 6 kbps 因 SILK 窄带模式带宽截断，mel-L1 强负向但 STOI/PESQ-WB 上 Opus 略优，"
             "见 §5.1 末段说明；本文不据此单独宣称 LCA L=3 优于 Opus 6 kbps。"
             "DAC、EnCodec、Opus 各自的完整档位（n_q∈{1,2,3,4,6,9,12}、EnCodec {1.5,3,6,12} kbps、"
             "Opus {6,8,12,16,24} kbps）已含于表 D.1。\n")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT} ({len(lines)} lines)")
print("\n--- PREVIEW (first 40 lines) ---")
print("\n".join(lines[:40]))
