# Base vs LCA evaluation summary

- Base ckpt: `output\experiments\exp2_scit_speech_distill30_retrain_20260529_seed42\checkpoints\SCIT-Speech-Base_best.pt`
  - sha256: `8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6`
- LCA ckpt:  `output\experiments\exp5_lca_component_factorial_20260603_seed42\runs\V2_channelsim_only\checkpoints\SpeechTokenizerTrainer_00017500`
  - sha256: `b59e3282a46b132ed0432e4fc01663439714664f8f154c4b7c1835905739a03f`
- Sample count per (L, channel): 256
- ChannelSim seed offset: 42 (deterministic per (sample, L, channel) triple, shared between Base and LCA for fair comparison)

## Mean metrics by (model, L, channel)

| model | L | channel | wave_l1 ↓ | mel_l1 ↓ | si_snr_db ↑ | corr ↑ | stoi ↑ | pesq_wb ↑ |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| base | 1 | clean | 0.0312 | 1.2080 | -6.181 | 0.4546 | 0.7971 | nan |
| base | 1 | dropout-high | 0.0328 | 1.2678 | -7.856 | 0.3903 | 0.7690 | nan |
| base | 1 | dropout-mid | 0.0320 | 1.2377 | -7.029 | 0.4220 | 0.7831 | nan |
| base | 1 | substitution-high | 0.0316 | 1.2564 | -6.574 | 0.4391 | 0.7796 | nan |
| base | 1 | substitution-mid | 0.0314 | 1.2242 | -6.310 | 0.4496 | 0.7910 | nan |
| base | 2 | clean | 0.0239 | 0.9491 | -0.921 | 0.6662 | 0.8596 | nan |
| base | 2 | dropout-high | 0.0270 | 1.0319 | -3.283 | 0.5729 | 0.8231 | nan |
| base | 2 | dropout-mid | 0.0255 | 0.9920 | -2.157 | 0.6180 | 0.8407 | nan |
| base | 2 | substitution-high | 0.0248 | 1.0056 | -1.561 | 0.6421 | 0.8395 | nan |
| base | 2 | substitution-mid | 0.0242 | 0.9672 | -1.107 | 0.6591 | 0.8530 | nan |
| base | 3 | clean | 0.0217 | 0.8831 | +0.654 | 0.7224 | 0.8811 | nan |
| base | 3 | dropout-high | 0.0254 | 0.9749 | -2.025 | 0.6221 | 0.8410 | nan |
| base | 3 | dropout-mid | 0.0235 | 0.9297 | -0.695 | 0.6737 | 0.8613 | nan |
| base | 3 | substitution-high | 0.0227 | 0.9423 | -0.051 | 0.6978 | 0.8600 | nan |
| base | 3 | substitution-mid | 0.0221 | 0.9035 | +0.407 | 0.7140 | 0.8737 | nan |
| lca | 1 | clean | 0.0316 | 1.2268 | -6.675 | 0.4365 | 0.7994 | nan |
| lca | 1 | dropout-high | 0.0330 | 1.2785 | -8.311 | 0.3762 | 0.7751 | nan |
| lca | 1 | dropout-mid | 0.0323 | 1.2522 | -7.493 | 0.4061 | 0.7872 | nan |
| lca | 1 | substitution-high | 0.0320 | 1.2712 | -7.099 | 0.4203 | 0.7789 | nan |
| lca | 1 | substitution-mid | 0.0317 | 1.2410 | -6.826 | 0.4310 | 0.7924 | nan |
| lca | 2 | clean | 0.0244 | 0.9426 | -1.382 | 0.6478 | 0.8634 | nan |
| lca | 2 | dropout-high | 0.0272 | 1.0185 | -3.600 | 0.5590 | 0.8305 | nan |
| lca | 2 | dropout-mid | 0.0258 | 0.9809 | -2.530 | 0.6021 | 0.8467 | nan |
| lca | 2 | substitution-high | 0.0253 | 1.0026 | -2.055 | 0.6214 | 0.8400 | nan |
| lca | 2 | substitution-mid | 0.0247 | 0.9613 | -1.591 | 0.6397 | 0.8558 | nan |
| lca | 3 | clean | 0.0216 | 0.8597 | +0.569 | 0.7198 | 0.8872 | nan |
| lca | 3 | dropout-high | 0.0249 | 0.9445 | -2.013 | 0.6227 | 0.8509 | nan |
| lca | 3 | dropout-mid | 0.0232 | 0.9025 | -0.753 | 0.6714 | 0.8696 | nan |
| lca | 3 | substitution-high | 0.0226 | 0.9236 | -0.188 | 0.6929 | 0.8626 | nan |
| lca | 3 | substitution-mid | 0.0219 | 0.8810 | +0.302 | 0.7105 | 0.8787 | nan |

## LCA - Base improvements by (L, channel)

Negative wave_l1 / mel_l1 deltas mean LCA is better. Positive si_snr / stoi / pesq deltas mean LCA is better.

| L | channel | Δwave_l1 | Δmel_l1 | Δsi_snr_db | Δstoi | Δpesq_wb |
|---:|---|---:|---:|---:|---:|---:|
| 1 | clean | +0.0004 | +0.0188 | -0.494 | +0.0023 | +nan |
| 1 | dropout-high | +0.0002 | +0.0107 | -0.455 | +0.0062 | +nan |
| 1 | dropout-mid | +0.0003 | +0.0145 | -0.465 | +0.0041 | +nan |
| 1 | substitution-high | +0.0004 | +0.0148 | -0.525 | -0.0007 | +nan |
| 1 | substitution-mid | +0.0004 | +0.0168 | -0.515 | +0.0014 | +nan |
| 2 | clean | +0.0005 | -0.0065 | -0.460 | +0.0038 | +nan |
| 2 | dropout-high | +0.0002 | -0.0134 | -0.317 | +0.0074 | +nan |
| 2 | dropout-mid | +0.0003 | -0.0111 | -0.373 | +0.0060 | +nan |
| 2 | substitution-high | +0.0005 | -0.0030 | -0.494 | +0.0006 | +nan |
| 2 | substitution-mid | +0.0005 | -0.0059 | -0.484 | +0.0028 | +nan |
| 3 | clean | -0.0002 | -0.0235 | -0.085 | +0.0061 | +nan |
| 3 | dropout-high | -0.0005 | -0.0304 | +0.012 | +0.0099 | +nan |
| 3 | dropout-mid | -0.0003 | -0.0272 | -0.058 | +0.0083 | +nan |
| 3 | substitution-high | -0.0001 | -0.0187 | -0.137 | +0.0026 | +nan |
| 3 | substitution-mid | -0.0001 | -0.0225 | -0.105 | +0.0050 | +nan |

## Notes
- Intrusive objective metrics, sample-wise then averaged across the test set.
- WER/CER not computed (no ASR model integration).
- PESQ-WB requires sample_rate=16000.
- Channel-sim conditions: clean, dropout-low/mid (p_drop=0.01/0.03 with previous-index replacement), substitution-low/mid (p_sub=0.001/0.005 with uniform legal codebook indices).
- Same ChannelSim seed applied to Base and LCA for each (sample, L, channel) triple so improvements reflect model differences only.