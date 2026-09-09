# Exp1 Staged NAS Agent Handoff

- status: completed
- selected_encoder: nas_seed42_000896
- selection_mode: balanced
- short_distillation_steps: stage2=100, stage3=500, final=1500
- selection rule: short-distilled quality-constrained proxy score plus multi-objective Pareto.
- staged outputs are in metrics/stage1_profile.*, stage2_proxy.*, stage3_refined.*, and stage4_final.*.
- downstream Exp2 may use artifacts/best_architecture/best_seanet_config.json as the NAS encoder route.
- raw selected NAS search item is preserved in artifacts/best_architecture/best_candidate_raw.json.
