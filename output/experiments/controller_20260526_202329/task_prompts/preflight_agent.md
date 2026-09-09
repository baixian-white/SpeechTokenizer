# Preflight Agent Task Prompt

You are the Preflight Agent for this repository execution.

Read-only scope:
- output/doc/实验手册.md
- output/experiments/README.md
- output/doc/experiment_plans/exp1_nas_semantic_encoder.md
- output/doc/experiment_plans/exp2_scit_speech_training.md
- output/doc/experiment_plans/exp3_low_load_channel_aware_adaptation.md
- output/doc/experiment_plans/exp4_baseline_comparison.md
- output/doc/experiment_plans/exp5_ablation_and_diagnosis.md
- config/spt_base_cfg.json
- scripts/train_example.py
- scripts/hubert_rep_extract.py
- speechtokenizer/model.py
- speechtokenizer/trainer/dataset.py
- speechtokenizer/trainer/trainer.py
- speechtokenizer/trainer/loss.py
- nas/search_autoencoder.py
- nas/make_subset.py
- nas/export_best_model.py
- nas/custom_model.py
- nas/train_nas.py
- output/experiments/controller_20260526_202329/orchestrator_state.md
- output/experiments/controller_20260526_202329/orchestrator_state.json

Write scope:
- output/experiments/controller_20260526_202329/reports/preflight_report.md
- output/experiments/controller_20260526_202329/reports/preflight_status.json
- output/experiments/controller_20260526_202329/reports/agent_handoff.md
- output/experiments/controller_20260526_202329/reports/agent_status.json
- output/experiments/exp*/reports/preflight_report.md
- output/experiments/exp*/reports/preflight_status.json

You may create the reports directory under the controller if it is missing. Do not edit code. Do not edit experiment metrics, checkpoints, samples, or configs.

Allowed scripts: no new scripts. You may run short inspection commands only. Do not launch long training, NAS search, or background jobs.

Required checks:
- data paths and file lists exist; inspect a few listed entries for audio and .hubert.npy existence.
- config fixed conditions: sample_rate=16000, strides=[8,5,4,2], dimension=1024, n_q=3, codebook_size=1024.
- checkpoint availability for Base/LCA if any existing new run references exist.
- Python, PyTorch, CUDA, GPU, disk free space, key dependency importability.
- critical script existence for exp1-exp5.
- whether minimal smoke test is feasible: data load 1-2 batches, model instantiate, forward/encode/decode L=1/2/3, payload toy example. If not feasible, say exactly why; do not run heavy training.
- disk free space must be >=30GB or current experiment must abort.

Major-deviation stop rules:
- fixed conditions changed.
- missing data manifest or missing sample files.
- checkpoint cannot load or model/config mismatch.
- codes shape, L slicing, payload definition, latent rate, M, K inconsistent.
- same error repeated 3 times.
- any output directory cannot preserve logs/config/commands.

Output:
- Write machine-readable status JSON with per-experiment status: preflight_passed, preflight_failed, or blocked.
- Write human-readable report with commands run, observations, blockers, and recommended next step.
- Write agent handoff and agent status.
- Do not fabricate results, metrics, checkpoints, logs, samples, or Pareto frontier.
