# LCA Component Factorial Ablation Status

- run_id: `exp5_lca_component_factorial_20260603_seed42`
- purpose: isolate `random-L`, `ChannelSim`, and `consistency loss` contributions in LCA.
- base checkpoint: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
- full LCA reference: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42`

## Variant Matrix

| Variant | random-L | ChannelSim | Consistency | Status |
|---|---|---|---|---|
| `V0_full_depth_clean_control` | no, fixed `L=3` | clean only | no | completed; last metric step 33700 |
| `V1_random_l_only` | yes, `L=1/2/3` | clean only | no | pending |
| `V2_channelsim_only` | no, fixed `L=3` | strong | no | pending |
| `V3_random_l_channelsim` | yes, `L=1/2/3` | strong | no | pending |
| `V4_full_lca_reference` | yes, `L=1/2/3` | strong | yes, `λ=0.5` | already completed in exp3 v2 |

## Current Run

`V0_full_depth_clean_control` was briefly started from Codex to verify the pipeline.
The run reached roughly step 590 with normal logs and about `0.5s/step`, then was stopped at user request.
The partial output directory `runs/V0_full_depth_clean_control` was removed so the user can rerun from a clean state.

User restarted `V0_full_depth_clean_control` at 2026-06-03 20:44. Monitoring confirmed:

- process active under the `speechtokenizer` conda environment;
- `lca_train_metrics.jsonl` contains step `0`, `100`, and `200` records;
- sampled `L=3`, channel `clean`, no perturbation, and `consistency_used=false`, as intended for V0;
- training completed overnight; last JSONL metric step is `33700` and latest saved trainer checkpoint is `SpeechTokenizerTrainer_00032500`.
- TensorBoard dev metrics show `dev/mel error` best at step `5000` (`1.09745`) and `dev/comm_mel/L3_clean` best at step `30000` (`0.366516`).

## Next Steps

1. Let `V0_full_depth_clean_control` finish or reach the selected checkpoint budget.
2. Run `V1_random_l_only` with the generated config.
3. Run `V2_channelsim_only` with the generated config.
4. Run `V3_random_l_channelsim` with the generated config.
5. Evaluate all new checkpoints with the same `evaluate_lca_vs_base.py` protocol and compare against the existing full LCA reference.
