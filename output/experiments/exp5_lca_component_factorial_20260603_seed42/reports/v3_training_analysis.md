# V3 random-L + ChannelSim Training Analysis

- date: 2026-06-05
- V3: `V3_random_l_channelsim` = random `L ∈ {1,2,3}`, strong ChannelSim, no consistency loss
- comparison controls: `V0_full_depth_clean_control`, `V2_channelsim_only`

## Completion

V3 completed the 10-epoch schedule.

| Variant | Last metric step | Last epoch | Last LR | Latest trainer ckpt |
|---|---:|---:|---:|---|
| V3 | 33700 | 9 | 1.71e-10 | `SpeechTokenizerTrainer_00032500` |

## Training Distribution

V3 sampled all three load points and all five channel conditions as expected.

| L | n | full branch | comm branch |
|---:|---:|---:|---:|
| 1 | 106 | 91.897 | 74.510 |
| 2 | 125 | 91.831 | 58.390 |
| 3 | 107 | 91.264 | 53.852 |

| Channel | n | comm branch | actual drop | actual sub |
|---|---:|---:|---:|---:|
| clean | 60 | 59.303 | 0.0000 | 0.0000 |
| dropout-mid | 69 | 60.832 | 0.0490 | 0.0000 |
| dropout-high | 68 | 63.819 | 0.0977 | 0.0000 |
| substitution-mid | 70 | 61.337 | 0.0000 | 0.0105 |
| substitution-high | 71 | 64.367 | 0.0000 | 0.0303 |

The training loss ordering is sensible: lower `L` has higher communication loss, and stronger perturbations produce higher communication loss.

## Dev TensorBoard Summary

V3 improves throughout training; for most dev metrics, the best point is the final logged step `32500`.

| Metric | Best value |
|---|---:|
| `dev/full_depth_mel_error` | 0.979239 @ 32500 |
| `dev/comm_mel/L1_clean` | 0.553820 @ 2500 |
| `dev/comm_mel/L2_clean` | 0.434456 @ 32500 |
| `dev/comm_mel/L3_clean` | 0.398440 @ 32500 |
| `dev/comm_mel/L3_dropout-mid` | 0.417809 @ 32500 |
| `dev/comm_mel/L3_dropout-high` | 0.434778 @ 32500 |
| `dev/comm_mel/L3_substitution-mid` | 0.452158 @ 32500 |
| `dev/comm_mel/L3_substitution-high` | 0.559574 @ 32500 |

For `L1`, the best dev point occurs early (`2500`) and later training worsens L1. For `L2/L3`, final checkpoint is best.

## V3 vs V2 at L3

V2's best L3 dev point is step `17500`; V3's best L3 dev point is step `32500`. Comparing best observed values:

| Condition | V2 L3 best | V3 L3 best | V3 - V2 |
|---|---:|---:|---:|
| clean | 0.423478 | 0.398440 | -0.025038 |
| dropout-mid | 0.444177 | 0.417809 | -0.026368 |
| dropout-high | 0.460853 | 0.434778 | -0.026075 |
| substitution-mid | 0.471729 | 0.452158 | -0.019571 |
| substitution-high | 0.574930 | 0.559574 | -0.015356 |

At L3, V3 is better than V2 in every dev condition, including clean and all perturbations. This suggests random-L does not harm the L3 dev matrix and may improve optimization relative to ChannelSim-only.

## Degradation Relative to Clean

| Variant / L | dropout-mid | dropout-high | substitution-mid | substitution-high |
|---|---:|---:|---:|---:|
| V2 L3 best | +0.020699 | +0.037375 | +0.048251 | +0.151452 |
| V3 L3 best | +0.019369 | +0.036338 | +0.053718 | +0.161134 |
| V3 L2 best | +0.018006 | +0.034709 | +0.050412 | +0.171791 |
| V3 L1 best | +0.013843 | +0.028180 | +0.060132 | +0.195069 |

V3 improves absolute mel-L1 at L3 relative to V2. Robustness degradation is slightly better for dropout but slightly worse for substitution-high. The main observed gain is absolute reconstruction quality, not a uniformly smaller degradation ratio.

## Interpretation

1. V3 is a successful run and should be kept.
2. Random-L + ChannelSim is better than ChannelSim-only at L3 across the dev matrix.
3. V3 gives usable dev results for all `L=1/2/3`, while V2 only trains/evaluates `L=3`.
4. For V3, use final or near-final checkpoints for L2/L3 evaluation, but consider early checkpoint `2500` if focusing specifically on L1 clean mel.
5. V3 still lacks consistency loss; the next comparison against full LCA (`exp3 v2`) will tell whether consistency adds robustness beyond random-L + ChannelSim.

## Recommended Checkpoints for Evaluation

- V3 primary: `SpeechTokenizerTrainer_00032500`
- V3 secondary: `SCIT-Speech-LCA_best.pt`
- V3 L1 diagnostic: `SpeechTokenizerTrainer_00002500`
- Compare against:
  - V0: `SpeechTokenizerTrainer_00030000` and `00032500`
  - V2: `SpeechTokenizerTrainer_00017500` and `00032500`
  - Full LCA reference: `exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42`
