# Experiment Orchestrator State

- controller_run_id: controller_20260526_202329
- controller_dir: output/experiments/controller_20260526_202329
- created_at: 2026-05-26T20:23:29+08:00
- updated_at: 2026-05-26T21:08:09+08:00
- seed: 42
- git_commit: cb9929cb60e05540b24d59bc68f117187219d810
- git_status_summary: dirty worktree present before orchestration; existing changes are treated as user/project state and must not be reverted.

## Required Documents Read

- output/doc/实验手册.md
- output/experiments/README.md
- output/doc/experiment_plans/exp1_nas_semantic_encoder.md
- output/doc/experiment_plans/exp2_scit_speech_training.md
- output/doc/experiment_plans/exp3_low_load_channel_aware_adaptation.md
- output/doc/experiment_plans/exp4_baseline_comparison.md
- output/doc/experiment_plans/exp5_ablation_and_diagnosis.md

## Global Fixed Conditions

- sample_rate: 16000
- strides: [8, 5, 4, 2]
- encoder_downsample_rate: 320
- latent_rate: 50 steps/s
- latent_dimension: 1024
- M / n_q: 3
- K / codebook_size: 1024
- transmitted L: 1, 2, 3
- experiment 6 is explicitly out of scope and was not executed.

## Current Experiment Status

| Experiment | Run ID | Status | Latest log | Latest checkpoint / artifact | Blocking item |
|---|---|---|---|---|---|
| exp1 | exp1_nas_semantic_encoder_20260526_202329_seed42 | completed | output/experiments/exp1_nas_semantic_encoder_20260526_202329_seed42/logs/stdout_rerun1.log | output/experiments/exp1_nas_semantic_encoder_20260526_202329_seed42/artifacts/best_architecture/best_seanet_config.json | none |
| exp2 | exp2_scit_speech_training_20260526_202329_seed42 | completed | output/experiments/exp2_scit_speech_training_20260526_202329_seed42/logs/package_stdout.log | output/experiments/exp2_scit_speech_training_20260526_202329_seed42/checkpoints/SCIT-Speech-Base_best.pt | none |
| exp3 | exp3_low_load_channel_aware_adaptation_20260526_202329_seed42 | completed | output/experiments/exp3_low_load_channel_aware_adaptation_20260526_202329_seed42/logs/eval_stdout.log | output/experiments/exp3_low_load_channel_aware_adaptation_20260526_202329_seed42/checkpoints/SCIT-Speech-LCA_best.pt | none |
| exp4 | exp4_baseline_comparison_20260526_202329_seed42 | aborted | output/experiments/exp4_baseline_comparison_20260526_202329_seed42/logs/stdout.log | none | required traditional codec tool missing: ffmpeg/opusenc not found |
| exp5 | exp5_ablation_and_diagnosis_20260526_202329_seed42 | completed | output/experiments/exp5_ablation_and_diagnosis_20260526_202329_seed42/logs/ablation_eval_stdout.log | output/experiments/exp5_ablation_and_diagnosis_20260526_202329_seed42/checkpoints/random_l_off/SCIT-Speech-LCA_best.pt | none |

## Subagents

- Preflight Agent: simulated; prompt at output/experiments/controller_20260526_202329/task_prompts/preflight_agent.md.
- Exp5 Ablation Agent: simulated; prompt at output/experiments/controller_20260526_202329/task_prompts/exp5_ablation_agent.md.
- Monitor / Reviewer Agent: simulated; prompt at output/experiments/controller_20260526_202329/task_prompts/monitor_reviewer_agent.md.

## Monitor / Reviewer Result

- status: passed
- report: output/experiments/controller_20260526_202329/reports/monitor_review.md
- json: output/experiments/controller_20260526_202329/reports/monitor_review.json
- out-of-scope Exp6-like directories: none

## Caveats

- Exp1 is a constrained encoder-only proxy NAS run, not a full long NAS search.
- Exp2/Exp3/Exp5 are short debug/tracer executions with documented config changes, not paper-grade full training conclusions.
- Exp4 is aborted because required traditional codec tooling is missing. SCIT and PCM artifacts created before abort were preserved; Opus/AMR-WB results were not fabricated.

## Next Step

Human inspection should start from output/experiments/exp4_baseline_comparison_20260526_202329_seed42/reports/failure_report.md and the debug/tracer caveats in each completed run summary.
