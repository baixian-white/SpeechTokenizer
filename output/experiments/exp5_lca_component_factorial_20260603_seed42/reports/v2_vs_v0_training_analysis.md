# V2 ChannelSim-only vs V0 Clean-control Training Analysis

- date: 2026-06-04
- V0: `V0_full_depth_clean_control` = fixed `L=3`, clean only, no consistency
- V2: `V2_channelsim_only` = fixed `L=3`, strong ChannelSim, no consistency

## Completion

Both V0 and V2 completed the 10-epoch LCA fine-tuning schedule.

| Variant | Last metric step | Last epoch | Last LR | Latest trainer ckpt |
|---|---:|---:|---:|---|
| V0 | 33700 | 9 | 1.71e-10 | `SpeechTokenizerTrainer_00032500` |
| V2 | 33700 | 9 | 1.71e-10 | `SpeechTokenizerTrainer_00032500` |

## Training JSONL Summary

### V0

| Window | Channel | n | full branch | comm branch | actual drop | actual sub |
|---|---|---:|---:|---:|---:|---:|
| all | clean | 338 | 87.500 | 49.655 | 0.0000 | 0.0000 |
| last 5k | clean | 51 | 87.526 | 49.209 | 0.0000 | 0.0000 |

### V2

| Window | Channel | n | full branch | comm branch | actual drop | actual sub |
|---|---|---:|---:|---:|---:|---:|
| all | clean | 59 | 93.654 | 51.594 | 0.0000 | 0.0000 |
| all | dropout-mid | 79 | 94.573 | 55.413 | 0.0503 | 0.0000 |
| all | dropout-high | 58 | 94.327 | 58.158 | 0.0984 | 0.0000 |
| all | substitution-mid | 57 | 95.052 | 53.246 | 0.0000 | 0.0100 |
| all | substitution-high | 85 | 93.252 | 55.073 | 0.0000 | 0.0293 |
| last 5k | clean | 9 | 94.512 | 50.835 | 0.0000 | 0.0000 |
| last 5k | dropout-mid | 13 | 96.265 | 55.159 | 0.0492 | 0.0000 |
| last 5k | dropout-high | 11 | 97.115 | 58.850 | 0.0977 | 0.0000 |
| last 5k | substitution-mid | 11 | 97.742 | 53.445 | 0.0000 | 0.0091 |
| last 5k | substitution-high | 7 | 94.672 | 53.822 | 0.0000 | 0.0283 |

## Dev TensorBoard Summary

| Metric | V0 best | V0 last | V2 best | V2 last |
|---|---:|---:|---:|---:|
| `dev/full_depth_mel_error` | 1.097450 @ 5000 | 1.172973 @ 32500 | 0.977949 @ 2500 | 1.092991 @ 32500 |
| `dev/comm_mel/L3_clean` | 0.366516 @ 30000 | 0.407384 @ 32500 | 0.423478 @ 17500 | 0.485670 @ 32500 |
| `dev/comm_mel/L3_dropout-mid` | n/a | n/a | 0.444177 @ 17500 | 0.503522 @ 32500 |
| `dev/comm_mel/L3_dropout-high` | n/a | n/a | 0.460853 @ 17500 | 0.523265 @ 32500 |
| `dev/comm_mel/L3_substitution-mid` | n/a | n/a | 0.471729 @ 17500 | 0.562132 @ 32500 |
| `dev/comm_mel/L3_substitution-high` | n/a | n/a | 0.574930 @ 17500 | 0.709280 @ 32500 |

## Interpretation

1. V2 is a valid ChannelSim-only run: all logged samples have fixed `L=3`, nonzero dropout/substitution appears at the intended rates, and `consistency_used=false` throughout.
2. V2 sacrifices clean communication quality relative to V0. V0 reaches `dev/comm_mel/L3_clean = 0.366516`, while V2 reaches `0.423478` at best.
3. V2's perturbation dev metrics peak around step `17500`; later checkpoints drift upward on comm-mel. For V2 evaluation, prioritize `SpeechTokenizerTrainer_00017500`, then compare `00020000` and `00032500` as secondary checkpoints.
4. V2 alone cannot prove robustness improvement over V0 yet, because V0's dev matrix did not include perturbed channels. The next evaluation must run V0 and V2 through the same clean/dropout/substitution evaluation protocol and compute perturbation degradation and robust_imp.
5. The observed pattern is consistent with the experimental hypothesis: ChannelSim-only training exposes the model to perturbations but pays a clean-quality cost. Whether this cost buys robustness must be decided by matched evaluation, not by training loss alone.

## Recommended Next Evaluation

Evaluate these checkpoints with the same fixed sample list and channel conditions:

- V0: `SpeechTokenizerTrainer_00030000`, `SpeechTokenizerTrainer_00032500`, `SCIT-Speech-LCA_best.pt`
- V2: `SpeechTokenizerTrainer_00017500`, `SpeechTokenizerTrainer_00020000`, `SpeechTokenizerTrainer_00032500`, `SCIT-Speech-LCA_best.pt`

Primary comparison:

```text
degradation(model, channel) = metric(model, channel) - metric(model, clean)
robust_imp(V2 vs V0) = degradation(V0, channel) - degradation(V2, channel)
```

Use mel-L1 first, then confirm with STOI/PESQ/WER if evaluation cost allows.
