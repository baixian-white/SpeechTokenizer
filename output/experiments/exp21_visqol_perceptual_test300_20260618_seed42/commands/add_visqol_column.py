"""Add a 'ViSQOL ↑' column to the 8 same-rate quality tables in the paper draft.
Strict: every data row label must resolve to a ViSQOL value, else abort.
Writes to <paper>.visqol_preview.md for inspection; does NOT touch the original.
"""
import re
from pathlib import Path
import pandas as pd

PAPER = Path(r"H:/H-CODE/speechtokenizer/output/doc/paper_drafts/scit_speech_method_cn_draft_20260609.md")
SUMMARY = Path(r"H:/H-CODE/speechtokenizer/output/experiments/_visqol_setup/visqol_per_method_summary.csv")

# lookup[(split, method, setting)] = (mean, lo, hi)
LUT = {}
for _, r in pd.read_csv(SUMMARY).iterrows():
    LUT[(r["split"], r["method"], str(r["codec_setting"]))] = (
        r["visqol_mean"], r["visqol_ci_low"], r["visqol_ci_high"])


def parse_label(raw):
    t = raw.replace("**", "").strip()
    t = re.sub(r"（.*?）", "", t).strip()          # drop Chinese parenthetical notes
    if t.startswith("SCIT-Base"):
        return ("scit_base", t.split()[-1])
    if t.startswith("SCIT-LCA"):
        return ("scit_lca", t.split()[-1])
    if t.startswith("DAC"):
        return ("dac", f"n_q_{t.split('n_q=')[1].strip()}")
    if t.startswith("EnCodec"):
        m = {1.5: "bw1.5kbps_n_cb2", 3.0: "bw3.0kbps_n_cb4",
             6.0: "bw6.0kbps_n_cb8", 12.0: "bw12.0kbps_n_cb16"}
        return ("encodec", m[float(t.split()[1])])
    if t.startswith("Opus"):
        return ("opus", f"opus_{int(round(float(t.split()[1])*1000))}bps")
    if t.startswith("AMR-WB"):
        return ("amrwb", f"amrwb_{int(round(float(t.split()[1])*1000))}bps")
    if t.startswith("Codec2"):
        num = re.match(r"(\d+)", t.split()[1]).group(1)   # "700C"->700, "1200"->1200
        return ("codec2", f"codec2_{num}bps")
    if t.startswith("PCM"):
        return ("pcm", "16bit_16khz_passthrough")
    if t.startswith("codec2_"):
        return ("codec2", t)
    if t.startswith("amrwb_"):
        return ("amrwb", t)
    raise ValueError(f"unparsed label: {raw!r}")


def main():
    lines = PAPER.read_text(encoding="utf-8").splitlines()
    cur_split = None
    out = []
    modified_tables = 0
    i = 0
    while i < len(lines):
        ln = lines[i]
        if "test-clean_300" in ln:
            cur_split = "test-clean_300"
        elif "test-other_300" in ln:
            cur_split = "test-other_300"
        # detect quality-table header row
        if ln.startswith("|") and "PESQ-WB ↑" in ln:
            assert cur_split, f"table at line {i} has no split context"
            # header
            out.append(ln.rstrip() + " ViSQOL ↑ |")
            # separator
            sep = lines[i + 1]
            assert set(sep.replace("|", "").replace(":", "").strip()) <= {"-", " "}, f"bad sep at {i+1}: {sep}"
            out.append(sep.rstrip() + "---|")
            # data rows
            j = i + 2
            while j < len(lines) and lines[j].startswith("|"):
                row = lines[j].rstrip()
                cells = [c.strip() for c in row.strip("|").split("|")]
                # tables 1a/1b have an 操作点 first column (e.g. "500 bps", "lossless",
                # "6 kbps（区间外）"); D.1/D.3 have the method/档位 directly in col0.
                # An operating-point cell starts with a digit+space or is lossless/区间外.
                is_oppoint = (cells[0] == "lossless") or ("区间外" in cells[0]) \
                    or bool(re.match(r"^[\d.]+\s+k?bps", cells[0]))
                lab_cell = cells[1] if is_oppoint else cells[0]
                method, setting = parse_label(lab_cell)
                key = (cur_split, method, setting)
                assert key in LUT, f"no ViSQOL for {key} (row label {lab_cell!r}, line {j})"
                mean, lo, hi = LUT[key]
                use_ci = any("[" in c for c in cells[1:])
                val = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]" if use_ci else f"{mean:.3f}"
                assert row.endswith("|"), f"row missing trailing pipe at line {j}: {row}"
                out.append(row + f" {val} |")
                j += 1
            modified_tables += 1
            i = j
            continue
        out.append(ln)
        i += 1
    preview = PAPER.with_suffix(".visqol_preview.md")
    preview.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"modified {modified_tables} tables -> {preview.name}")


if __name__ == "__main__":
    main()
