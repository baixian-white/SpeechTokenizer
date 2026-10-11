# Exp23 Frozen SpeechTokenizer Speaker Classifier Runbook

## Boundary

Exp23 is a fixed-110 closed-set classifier. It recognizes only identities present during training. It does not support arbitrary-user enrollment, unknown-speaker rejection, biometric authentication, or spoof detection.

## Execution Order

1. Verify the Conda environment, VCTK manifests, LCA config, checkpoint, and SHA-256 values.
2. Reproduce the Exp22 baseline without changing the frozen SpeechTokenizer.
3. Build canonical 110-speaker and text-controlled 109-speaker manifests.
4. Run cache storage estimation, then build validated frozen L3 caches.
5. Train B1 audio Softmax, B2 audio AAM, and B3 token AAM using validation-only selection.
6. Stop before fusion when both B2 and B3 validation Top-1 are below 0.75.
7. Train B4 score averaging and M1 gated fusion, plus mandatory ablations.
8. If seed 42 validation Top-1 is below 0.85, stop. Between 0.85 and 0.88, allow one bounded revision. At or above 0.88, freeze configuration.
9. Train frozen configurations for seeds 41 and 43.
10. Fit temperature on validation logits, then perform one formal test evaluation per seed.
11. Aggregate mean and standard deviation across seeds 41, 42, and 43. Never select the best seed for the 90% claim.
12. Export the classifier bundle and run CPU latency and demo smoke tests.

## Success Criteria

- Product: mean Top-1 and macro-F1 at least 0.90, with hierarchical-bootstrap Top-1 lower bound at least 0.88.
- Mechanism: gated fusion improves absolute Top-1 by at least 0.02 or reduces relative error by at least 15%.
- Robustness: text-controlled 109-class and two-second 110-class Top-1 are each at least 0.85.
- Negative and stopped runs remain in the final report.

## Commands

```powershell
conda run -n speechtokenizer python scripts/build_exp23_speaker_cache.py --config config/exp23_speaker_classifier.json --run-dir output/experiments/exp23/cache_lca_l3 --cache-root output/cache/exp23 --audio-root data/VCTK-Corpus-0.92/wav48_silence_trimmed --models lca --layers 3 --device cpu --dry-run-estimate
conda run -n speechtokenizer python scripts/train_exp23_speaker_classifier.py --config config/exp23_speaker_classifier.json --cache-manifest output/experiments/exp23/cache_lca_l3/cache_manifest/cache_records.json --run-dir output/experiments/exp23/seed42 --seed 42 --device cuda
conda run -n speechtokenizer python scripts/evaluate_exp23_speaker_classifier.py --checkpoint output/experiments/exp23/seed42/checkpoints/best_macro_f1.pt --cache-manifest output/experiments/exp23/cache_lca_l3/cache_manifest/cache_records.json --output-dir output/experiments/exp23/seed42/metrics --split validation --device cuda
conda run -n speechtokenizer python scripts/evaluate_exp23_speaker_classifier.py --checkpoint output/experiments/exp23/seed42/checkpoints/best_macro_f1.pt --cache-manifest output/experiments/exp23/cache_lca_l3/cache_manifest/cache_records.json --output-dir output/experiments/exp23/seed42/metrics --split test --allow-test-evaluation --device cuda
```

The test command is run only after configuration freeze. Never use test metrics to revise model settings.
