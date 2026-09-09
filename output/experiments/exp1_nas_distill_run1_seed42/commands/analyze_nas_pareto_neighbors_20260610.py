from pathlib import Path
import pandas as pd

ROOT = Path(r"h:/H-CODE/speechtokenizer/output/experiments/exp1_nas_distill_run1_seed42")
OUT = ROOT / "reports" / "pareto_neighbors_20260610"
OUT.mkdir(parents=True, exist_ok=True)

stage4 = pd.read_csv(ROOT / "metrics" / "stage4_final.csv")
frontier_path = ROOT / "metrics" / "pareto_frontier.csv"
frontier = pd.read_csv(frontier_path) if frontier_path.exists() else stage4.copy()

selected_id = "nas_seed42_000896"
cols = [
    "candidate_id",
    "encoder_type",
    "stage_rank",
    "stage_score",
    "selected_for_next_stage",
    "selected",
    "encoder_params",
    "encoder_macs_g",
    "encoder_rtf_mean",
    "encoder_rtf_std",
    "semantic_proxy_loss",
    "proxy_recon_l1",
    "proxy_mel_loss",
    "teacher_latent_smooth_l1",
    "teacher_latent_cosine_distance",
    "teacher_temporal_delta_loss",
    "rvq_code_agreement",
    "rvq_code_flip_rate",
    "rvq_quantized_feature_l1",
    "train_last_total",
    "encoder_strides",
    "n_filters",
    "compress",
    "lstm",
    "activation",
    "layer_ops_list",
    "layer_se_list",
]
for c in cols:
    if c not in stage4.columns:
        stage4[c] = pd.NA

# Human-readable frontier table: all stage4 final candidates plus hand-designed reference.
table = stage4[cols].copy()
table["is_selected"] = table["candidate_id"].eq(selected_id)
table["params_M"] = table["encoder_params"] / 1e6
table["macs_G_per_s"] = table["encoder_macs_g"]
# Lower is better for these proxy quality columns; agreement higher is better.
quality_cols = [
    "semantic_proxy_loss",
    "proxy_recon_l1",
    "proxy_mel_loss",
    "teacher_latent_smooth_l1",
    "teacher_latent_cosine_distance",
    "teacher_temporal_delta_loss",
    "rvq_quantized_feature_l1",
    "train_last_total",
]

# Resource deltas vs hand encoder and selected candidate.
hand = table[table["candidate_id"] == "hand_encoder"].iloc[0]
sel = table[table["candidate_id"] == selected_id].iloc[0]
for base_name, base in [("vs_hand", hand), ("vs_selected", sel)]:
    table[f"params_ratio_{base_name}"] = table["encoder_params"] / base["encoder_params"]
    table[f"macs_ratio_{base_name}"] = table["encoder_macs_g"] / base["encoder_macs_g"]
    table[f"rtf_ratio_{base_name}"] = table["encoder_rtf_mean"] / base["encoder_rtf_mean"]

# Pareto neighbor categories.
nas = table[table["candidate_id"].ne("hand_encoder")].copy()
sel_macs = float(sel["encoder_macs_g"])
sel_mel = float(sel["proxy_mel_loss"])
sel_stage = float(sel["stage_score"])

lighter = nas[nas["encoder_macs_g"] < sel_macs].sort_values("encoder_macs_g").head(3)
heavier_better_mel = nas[(nas["encoder_macs_g"] >= sel_macs) & (nas["proxy_mel_loss"] < sel_mel)].sort_values("proxy_mel_loss").head(3)
better_stage = nas[nas["stage_score"] < sel_stage].sort_values("stage_score").head(3)
rank_neighbors = nas.sort_values("stage_rank").head(8)
neighbor_ids = pd.concat([lighter, heavier_better_mel, better_stage, rank_neighbors, table[table["candidate_id"].isin([selected_id, "hand_encoder"])]], ignore_index=True)["candidate_id"].drop_duplicates()
neighbors = table[table["candidate_id"].isin(neighbor_ids)].sort_values(["candidate_id".replace("candidate_id", "is_selected"), "stage_rank"], ascending=[False, True])

# Save full and compact outputs.
table.to_csv(OUT / "nas_stage4_final_with_proxy_quality.csv", index=False)
neighbors.to_csv(OUT / "nas_pareto_neighbors.csv", index=False)

compact_cols = [
    "candidate_id", "stage_rank", "stage_score", "is_selected", "params_M", "macs_G_per_s", "encoder_rtf_mean",
    "proxy_mel_loss", "proxy_recon_l1", "semantic_proxy_loss",
    "teacher_latent_smooth_l1", "teacher_latent_cosine_distance", "teacher_temporal_delta_loss",
    "rvq_code_agreement", "rvq_code_flip_rate", "rvq_quantized_feature_l1",
    "params_ratio_vs_hand", "macs_ratio_vs_hand", "rtf_ratio_vs_hand",
    "encoder_strides", "n_filters", "compress", "lstm", "activation", "layer_ops_list", "layer_se_list"
]
neighbors[compact_cols].to_csv(OUT / "nas_pareto_neighbors_compact.csv", index=False)

# Optional plot.
plot_status = "not generated"
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_df = nas.copy()
    ax.scatter(plot_df["encoder_macs_g"], plot_df["proxy_mel_loss"], s=50, alpha=0.75, label="stage4 NAS candidates")
    ax.scatter([sel["encoder_macs_g"]], [sel["proxy_mel_loss"]], s=120, marker="*", label="selected 000896")
    ax.scatter([hand["encoder_macs_g"]], [hand["proxy_mel_loss"]], s=80, marker="s", label="hand encoder")
    for _, r in rank_neighbors.iterrows():
        ax.annotate(str(r["candidate_id"]).replace("nas_seed42_", ""), (r["encoder_macs_g"], r["proxy_mel_loss"]), fontsize=7)
    ax.set_xlabel("Encoder MACs (G/s)")
    ax.set_ylabel("Proxy mel loss (lower is better)")
    ax.set_title("NAS stage4 Pareto neighborhood")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "nas_pareto_macs_vs_proxy_mel.png", dpi=200)
    plot_status = "generated: nas_pareto_macs_vs_proxy_mel.png"
except Exception as e:
    plot_status = f"not generated: {e}"

with open(OUT / "nas_pareto_summary.md", "w", encoding="utf-8") as f:
    f.write("# E5 NAS Pareto Neighbors and Proxy Quality\n\n")
    f.write(f"Input stage4 table: `{ROOT / 'metrics' / 'stage4_final.csv'}`\n\n")
    f.write(f"Selected candidate: `{selected_id}`\n\n")
    f.write("## Selected vs hand-designed encoder\n\n")
    f.write("| Metric | Hand encoder | Selected NAS | Ratio | Reduction |\n")
    f.write("|---|---:|---:|---:|---:|\n")
    for label, col in [("Params", "encoder_params"), ("MACs G/s", "encoder_macs_g"), ("RTF mean", "encoder_rtf_mean")]:
        hv = float(hand[col]); sv = float(sel[col]); ratio = sv / hv; red = 1 - ratio
        f.write(f"| {label} | {hv:.6g} | {sv:.6g} | {ratio:.4f} | {red*100:.1f}% |\n")
    f.write("\n## Stage4 final candidates (compact)\n\n")
    md_cols = ["candidate_id", "stage_rank", "stage_score", "params_M", "macs_G_per_s", "encoder_rtf_mean", "proxy_mel_loss", "teacher_latent_smooth_l1", "rvq_quantized_feature_l1", "is_selected"]
    f.write("| " + " | ".join(md_cols) + " |\n")
    f.write("|" + "|".join(["---" for _ in md_cols]) + "|\n")
    for _, rr in neighbors[md_cols].iterrows():
        vals = []
        for c in md_cols:
            v = rr[c]
            if isinstance(v, float):
                vals.append(f"{v:.6g}")
            else:
                vals.append(str(v))
        f.write("| " + " | ".join(vals) + " |\n")
    f.write("\n\n## Interpretation\n\n")
    f.write("- `stage_rank=1` selected the minimum final quality-constrained proxy score among the stage4 Pareto candidates.\n")
    f.write("- `proxy_mel_loss`, teacher latent losses, and RVQ compatibility metrics are proxy-quality diagnostics, not final SCIT-Speech reconstruction quality.\n")
    f.write("- Final quality still requires full Base training and downstream evaluation; this report only documents the NAS selection neighborhood.\n")
    f.write(f"- Plot status: {plot_status}.\n")
    f.write("\n## Output files\n\n")
    f.write("- `nas_stage4_final_with_proxy_quality.csv`\n")
    f.write("- `nas_pareto_neighbors.csv`\n")
    f.write("- `nas_pareto_neighbors_compact.csv`\n")
    f.write("- `nas_pareto_macs_vs_proxy_mel.png` if matplotlib was available\n")

print("Wrote", OUT)
print(neighbors[compact_cols].to_string(index=False))
