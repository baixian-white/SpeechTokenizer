# E5 NAS Pareto Neighbors and Proxy Quality

Input stage4 table: `h:\H-CODE\speechtokenizer\output\experiments\exp1_nas_distill_run1_seed42\metrics\stage4_final.csv`

Selected candidate: `nas_seed42_000896`

## Selected vs hand-designed encoder

| Metric | Hand encoder | Selected NAS | Ratio | Reduction |
|---|---:|---:|---:|---:|
| Params | 6.77059e+07 | 8.87336e+06 | 0.1311 | 86.9% |
| MACs G/s | 5.86936 | 0.618394 | 0.1054 | 89.5% |
| RTF mean | 0.00952724 | 0.003688 | 0.3871 | 61.3% |

## Stage4 final candidates (compact)

| candidate_id | stage_rank | stage_score | params_M | macs_G_per_s | encoder_rtf_mean | proxy_mel_loss | teacher_latent_smooth_l1 | rvq_quantized_feature_l1 | is_selected |
|---|---|---|---|---|---|---|---|---|---|
| nas_seed42_000896 | 1 | 62.1459 | 8.87336 | 0.618394 | 0.003688 | 1.50591 | 1.90328 | 2.0047 | True |
| nas_seed42_002496 | 2 | 62.2231 | 8.55234 | 0.532992 | 0.00477066 | 1.50313 | 1.98073 | 2.02677 | False |
| nas_seed42_007851 | 3 | 62.2276 | 8.51814 | 0.538277 | 0.00742248 | 1.49272 | 1.96212 | 2.0429 | False |
| nas_seed42_000187 | 4 | 62.2977 | 8.75346 | 0.581915 | 0.00475164 | 1.61376 | 1.9866 | 2.07963 | False |
| nas_seed42_006303 | 5 | 62.3413 | 8.59486 | 0.52693 | 0.00701886 | 1.53342 | 2.04541 | 2.09257 | False |
| nas_seed42_005944 | 6 | 62.4286 | 8.45281 | 0.56185 | 0.0075841 | 1.60675 | 2.10507 | 2.13305 | False |
| nas_seed42_007861 | 7 | 62.5141 | 8.47009 | 0.550598 | 0.0073926 | 1.65536 | 2.14781 | 2.14769 | False |
| nas_seed42_003333 | 8 | 62.5339 | 8.47134 | 0.569109 | 0.00484188 | 1.62871 | 2.16048 | 2.22332 | False |
| hand_encoder | nan | nan | 67.7059 | 5.86936 | 0.00952724 | 1.053 | 0 | 0 | False |


## Interpretation

- `stage_rank=1` selected the minimum final quality-constrained proxy score among the stage4 Pareto candidates.
- `proxy_mel_loss`, teacher latent losses, and RVQ compatibility metrics are proxy-quality diagnostics, not final SCIT-Speech reconstruction quality.
- Final quality still requires full Base training and downstream evaluation; this report only documents the NAS selection neighborhood.
- Plot status: not generated: No module named 'matplotlib'.

## Output files

- `nas_stage4_final_with_proxy_quality.csv`
- `nas_pareto_neighbors.csv`
- `nas_pareto_neighbors_compact.csv`
- `nas_pareto_macs_vs_proxy_mel.png` if matplotlib was available
