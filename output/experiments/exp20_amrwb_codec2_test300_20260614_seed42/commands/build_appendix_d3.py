"""Append exp20 (AMR-WB + Codec2) full tables to Appendix D of the draft."""
from pathlib import Path
import pandas as pd

ROOT = Path(r"H:/H-CODE/speechtokenizer/output/experiments/exp20_amrwb_codec2_test300_20260614_seed42/runs")
OUT = Path(r"H:/H-CODE/speechtokenizer/output/experiments/exp20_amrwb_codec2_test300_20260614_seed42/reports/appendix_d_exp20_fragment.md")

def fmt(r, m):
    return f"{r[m+'_mean']:.3f} [{r[m+'_ci_low']:.3f}, {r[m+'_ci_high']:.3f}]"

lines = ["\n### 附录 D.3：AMR-WB 与 Codec2 完整结果（exp20，n=300）\n",
         "AMR-WB 与 Codec2 用独立 ffmpeg 8.1.1 完整构建（libvo_amrwbenc / libcodec2）编解码，"
         "复用 exp12 的同一批 300 条原始 wav，95% bootstrap CI（B=10000，seed=42）。"
         "Codec2 为 8 kHz 窄带（16k→8k 编码→16k 解码），其 mel-L1 偏高含高频截断成分（与 Opus SILK 同理），"
         "STOI 对窄带更鲁棒；AMR-WB 为 16 kHz 宽带。实际码率为打包后 payload（含帧头）。\n"]

for split in ["test-clean_300", "test-other_300"]:
    res = pd.read_csv(ROOT / split / "metrics" / "per_method_summary.csv")
    pay = pd.read_csv(ROOT / split / "metrics" / "payload_summary.csv")
    rate = pay.groupby("codec_setting")["packed_payload_bps"].mean().round(0)
    for method, title in [("codec2", "Codec2（6 档）"), ("amrwb", "AMR-WB（9 档）")]:
        sub = res[res.method == method].copy()
        sub["bps"] = sub["codec_setting"].map(rate)
        sub = sub.sort_values("bps")
        lines.append(f"\n**表 D.3（{split}）— {title}**\n")
        lines.append("| 档位 | 实际码率(bps) | mel-L1 ↓ | STOI ↑ | PESQ-WB ↑ | SI-SNR(dB) |")
        lines.append("|---|---:|---|---|---|---|")
        for _, r in sub.iterrows():
            si = f"{r['si_snr_db_mean']:.2f} [{r['si_snr_db_ci_low']:.2f}, {r['si_snr_db_ci_high']:.2f}]"
            lines.append(f"| {r['codec_setting']} | {int(r['bps'])} | {fmt(r,'mel_l1')} | {fmt(r,'stoi')} | {fmt(r,'pesq_wb')} | {si} |")

lines.append("\n注：Codec2 700C 标称 700 bps，实际打包 ~815 bps（帧头开销）；其余档位标称与实际接近。"
             "AMR-WB 最低 6.6 kbps 标称、实际打包 ~7.2 kbps，已是该 codec 下限，仍约为 SCIT-LCA L=3（1.5 kbps）的 4.8× 带宽。"
             "**SI-SNR 一列对 Codec2/AMR-WB 不可作质量解读**：这两个传统 codec 引入算法延迟（编码前瞻/帧对齐），"
             "解码波形相对原始整体时移，使逐样本对齐的 SI-SNR 大幅塌陷（与真实失真无关）；"
             "感知/频谱指标（mel-L1、STOI、PESQ-WB）不受时移影响，应以这三项为准。\n")

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {OUT}, {len(lines)} lines")
