# Base vs LCA evaluation summary

- Base ckpt: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output/experiments/exp5_ablation_and_diagnosis_20260601_seed42/checkpoints/A3_distill0/SCIT-Speech-Base_distill0_step42500_extracted.pt`
  - sha256: `84b0ff458fc0084a329795f4d540389fe48dbdcd3a3ccab7df43614c84c877ba`
- Sample count per (L, channel): 8
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0321 | 1.2022 | -7.183 | 0.4327 | 0.7866 | 1.3330 |
| base | 2 | clean | 0.0246 | 0.9549 | -1.447 | 0.6514 | 0.8442 | 1.7072 |
| base | 3 | clean | 0.0222 | 0.8916 | +0.354 | 0.7089 | 0.8697 | 1.9123 |
| lca | 1 | clean | 0.0295 | 1.2243 | -6.647 | 0.4762 | 0.7625 | 1.3130 |
| lca | 2 | clean | 0.0238 | 1.0441 | -1.853 | 0.6475 | 0.8286 | 1.6160 |
| lca | 3 | clean | 0.0216 | 0.9899 | -0.324 | 0.6968 | 0.8543 | 1.7946 |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | -0.0027 | +0.0222 | +0.535 | -0.0241 | -0.0200 |
| 2 | clean | -0.0008 | +0.0892 | -0.407 | -0.0156 | -0.0913 |
| 3 | clean | -0.0006 | +0.0983 | -0.678 | -0.0154 | -0.1178 |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.