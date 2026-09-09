# Experiment 5 Summary: Ablation and Diagnosis

- run_id: `exp5_ablation_and_diagnosis_20260601_seed42`
- status: **completed (one new training run + four ablations from existing artifacts)**
- date: 2026-06-01

## Scope

5 single-factor ablations testing whether key SCIT-Speech design choices are necessary:

1. **A1 — NAS vs hand-designed encoder** (data from exp1)
2. **A2 — SCIT-Speech-Base vs SCIT-Speech-LCA v2** (data from exp4)
3. **A3 — with vs without semantic distillation** (NEW training run, this experiment)
4. **A4 — with vs without random L sampling** (data from exp4, with confound caveat)
5. **A5 — ChannelSim weak vs strong + consistency loss** (data from exp3 v1 vs v2)

## Headline findings

| Ablation | Conclusion | Strength |
|---|---|---|
| **A3 distillation** | **distill_lambda=30 ≫ distill_lambda=0**; +1.4 dev/mel gap, +7-9pp WER, -0.118 PESQ@L=3 | strong evidence (controlled training) |
| A1 NAS encoder | -87% params, -89% MACs, -55% RTF | strong on efficiency; quality requires hand-designed full retrain |
| A5 strong ChannelSim + consistency | weak ChannelSim alone yields ~0 robustness; strong+consistency gives 6/6 positive metrics | strong evidence (exp3 v1 vs v2) |
| A2 LCA v2 vs Base (clean) | LCA marginally better (mel_l1 -3-4%, stoi +0.01) | weak on clean; main LCA benefit is robustness (A5) |
| A4 random-L | Cannot cleanly isolate (confounded with ChannelSim+consistency in exp3 v2) | future work |

## Key A3 result

A new 12-epoch training run with `distill_loss_lambda = 0` (otherwise identical to exp2 distill30 production), early-stopped after the ablation gap became conclusive at step 47500.

| step | distill0 | distill30 | Δ |
|---:|---:|---:|---:|
| 5000 | 4.587 | 4.447 | +0.139 |
| 17500 | 3.923 | 3.218 | +0.706 |
| 47500 | 3.202 | 1.776 | **+1.426** |

distill0 best dev/mel (3.142 @ step 42500) is **~80% worse** than distill30 best (1.124 @ step 107500). At all evaluated L levels, distill0 has higher mel_l1, lower stoi, lower PESQ-WB, and higher WER.

## Components

| File | Content |
|---|---|
| [reports/ablation_comparison.md](ablation_comparison.md) | Detailed per-ablation tables and analysis (A1-A5) |
| [reports/summary.md](summary.md) | This file |
| [configs/ablation_matrix.json](../configs/ablation_matrix.json) | Ablation matrix definition (which variants, fixed conditions, control-treatment mapping) |
| [configs/variants/A3_distill0.json](../configs/variants/A3_distill0.json) | Training config for the new A3 run |
| [checkpoints/A3_distill0/](../checkpoints/A3_distill0/) | A3 checkpoints, best_dev.pt, sha256-stamped extracted ckpt |
| [checkpoints/A3_distill0/checkpoint_manifest.json](../checkpoints/A3_distill0/checkpoint_manifest.json) | A3 ckpt provenance |
| [metrics/ablation_results_preliminary.csv](../metrics/ablation_results_preliminary.csv) | A1+A2 from existing artifacts (early collection) |
| [metrics/base_vs_lca_results.csv](../metrics/base_vs_lca_results.csv) | A3 distill0 vs distill30 audio quality, 48 rows |
| [metrics/codebook_usage.csv](../metrics/codebook_usage.csv) | A3 distill0 codebook diagnostics, 3 layers |
| [metrics/layer_reconstruction.csv](../metrics/layer_reconstruction.csv) | A3 distill0 layer reconstruction L1/L2/L3, 24 rows |
| [metrics/asr_results.csv](../metrics/asr_results.csv) | A3 WER/CER via Whisper base.en, 48 rows |
| [samples/](../samples/) | A3 reconstructed wavs (base/lca subdirs) |

## A3 training timeline (this experiment's only new training)

| event | time | step | dev/mel |
|---|---|---:|---:|
| start | 2026-06-01 07:12 | 0 | — |
| step 2500 dev | 07:41 | 2500 | 5.149 |
| user-initiated pause | 14:30 | ~37500 | — |
| resume | 16:42 | 37500 (continue_train) | — |
| best dev | 17:46 | 42500 | **3.142** |
| stop | ~19:00 | ~50000 | — |
| total wall-time | ~12 hours (~9.3 hours actual training, ~2.7 hours pause) |
| epochs completed | ~14.8 / 60 |

## Limitations

1. **8 test samples only**, all LibriSpeech train-clean-100 (same speaker pool as training). No cross-corpus.
2. **A3 early-stopped at step 47500** (~12 epochs). Justified by determinative ablation signal (+1.4 dev/mel gap, monotonically widening), but a strict "both trained 60 epochs" comparison was not done. The conclusion direction is unaffected.
3. **A1 hand-designed encoder quality numbers come from exp1 short-distill proxy**, not from a 60-epoch full training matching exp2. To validate "NAS doesn't lose quality", a hand-designed full retrain is needed (~24h, listed as future work).
4. **A4 random-L is confounded** with ChannelSim+consistency loss changes between exp2 base and exp3 v2 LCA. A clean A4 ablation requires a dedicated training run (~24h, future work).
5. **No subjective listening test**.

## Statements supported

You can say:

- distill_loss_lambda > 0 is necessary for SCIT-Speech-Base; ablation removed it cleanly produces +1.4 dev/mel and +7-9pp WER worsening
- NAS encoder reduces compute by ~87% params, ~89% MACs, ~55% RTF vs hand-designed
- Strong ChannelSim (p ≥ 0.05) plus consistency loss is necessary for measurable robustness; weak ChannelSim alone produces near-zero robustness (clean negative result, exp3 v1 → v2)
- Without distillation, codebook usage is HIGHER (more codes used) but each code carries less information — distillation produces a more "specialized" codebook, not just a denser one

You should NOT say:

- distill_loss_lambda = 30 is the optimum (it works, but no fine-grained sweep was done)
- NAS encoder doesn't lose quality (verified for efficiency, but full retrain comparison missing)
- Random-L is independently necessary (confounded comparison)
- All ablation evidence generalizes beyond LibriSpeech train-clean

## Next steps (future work)

1. **Fine-grained distill_lambda sweep** (5/15/30/60/120) to find optimum
2. **Hand-designed encoder full retrain** (~24h) to fully validate A1 quality preservation claim
3. **Clean A4 random-L ablation** (~24h): train a base-style 60-epoch run with random-L ON, ChannelSim OFF, lambda_consistency=0
4. **Cross-corpus generalization** (test-other / VCTK)
5. **Subjective MOS / AB test**
