# Exp23 Frozen SpeechTokenizer Speaker Classifier Design

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-02T00:00:00+08:00
- Verification Status: UNVERIFIED
- Version Label: exp23_design_v1.2_revised
- Upstream Dependencies: `exp22_speaker_identity_design`, `exp22_paper_grade_20260708`, `exp23_methodology_review_20260802`
- Repro Lock: null (design stage; implementation configuration does not exist yet)

## 1. Objective and Scope

Exp23 trains a closed-set speaker identity module for the fixed 110 VCTK speakers. The existing SpeechTokenizer Base and LCA models remain completely frozen. Exp23 must not update their encoders, RVQ codebooks, decoders, packet format, or bitrate.

The primary endpoint is the frozen fine-tuned LCA checkpoint at L3. The primary goal is at least 90% Top-1 accuracy and 0.90 macro-F1 on its independent closed-set test set. Base L3 is a secondary reference and diagnostic upper-bound condition. The target is a hypothesis, not a promised result. Failure to reach 90% must be reported without changing the locked test split or removing difficult speakers.

The experiment tests three sources of identity information:

1. Reconstructed waveform features from a codec-aware ECAPA-TDNN branch.
2. Discrete RVQ sequences from a trainable token branch.
3. Gated fusion of the waveform and token embeddings.

This is fixed-class classification. It does not establish open-set speaker verification, arbitrary-user enrollment, biometric security, or spoof resistance.

## 2. Research Questions and Hypotheses

### Research Questions

1. Does supervised codec-aware training outperform the frozen ECAPA profile-matching baseline from Exp22?
2. Do RVQ tokens and reconstructed waveforms contain complementary identity information?
3. Can fusion achieve Top-1 and macro-F1 of at least 0.90 without data leakage?
4. Does the result remain stable for short segments, lower RVQ layers, and mild channel perturbations?

### Falsifiable Hypotheses

- H1: codec-aware ECAPA classification outperforms Exp22 ECAPA profile matching.
- H2: a temporal token encoder outperforms the existing `codes_hist` and `latent_stats` probes.
- H3: gated fusion improves over the best single branch by at least 2 percentage points.
- H4: the LCA L3 fusion model reaches mean Top-1 >= 0.90 and mean macro-F1 >= 0.90 over training seeds 41, 42, and 43 on one canonical test split.
- H5: Base/LCA differences identify whether the limiting factor is upstream identity loss or classifier capacity.

## 3. Existing Baselines

The verified Exp22 VCTK-110 results are:

| Condition | Top-1 |
|---|---:|
| Original audio with ECAPA profile matching | 0.996 +/- 0.004 |
| Base L3 reconstruction with ECAPA profile matching | 0.575 +/- 0.042 |
| LCA L3 reconstruction with ECAPA profile matching | 0.345 +/- 0.021 |
| Base L3 latent probe | approximately 0.581 |
| LCA L3 latent probe | approximately 0.547 |

Evidence files:

- `output/experiments/exp22_paper_grade_20260708/metrics/speaker_identity_ecapa_multiseed_summary.csv`
- `output/experiments/exp22_paper_grade_20260708/metrics/speaker_probe_multiseed_summary.csv`

These numbers show that the pretrained speaker backend is reliable on original speech, while reconstructed speech has a major domain mismatch. The token and latent probes also show that useful identity information remains available for supervised learning.

## 4. Frozen Upstream Models

### Base

- Config: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json`
- Checkpoint: `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt`

### LCA

- Config: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/configs/lca_finetune_config.json`
- Checkpoint: `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt`

### Freeze Contract

- Call `model.eval()` and set every parameter to `requires_grad=False`.
- Generate cached representations under `torch.inference_mode()`.
- Record checkpoint SHA-256, config SHA-256, and Git revision.
- Exclude every SpeechTokenizer parameter from all optimizers.
- Compare state-dict hashes before and after training. Any change is a hard failure.

The current codec operates at 16 kHz with three RVQ layers, 1024 codes per layer, and internal dimension 1024. `model.encode()` exposes discrete codes, `model.decode()` produces reconstructed waveforms, and `model.forward_feature()` exposes per-layer quantized features.

## 5. Dataset and Leakage Control

- Root: `data/VCTK/wav48_silence_trimmed`
- Speakers: 110 audio-bearing speaker directories. Discovery must not assume IDs begin with `p`; the current checkout includes the nonstandard speaker ID `s5`.
- Current minimum audio files per speaker: 172. After mic1/mic2 normalization, the minimum is 123 independent utterance groups for `p362`.
- Task: the same 110 identities occur in train, validation, and test, but utterances must be disjoint.

The split unit is a normalized utterance group, not an audio file. Microphone duplicates must remain together:

```text
p225_001_mic1.flac -> p225_001
p225_001_mic2.flac -> p225_001
```

The default run uses one fixed microphone version. If both microphones are used for augmentation, they must remain in the same split.

Per speaker, sample 80 train, 15 validation, and 15 test utterance groups. This produces 8,800 train, 1,650 validation, and 1,650 test groups. The live checkout has 110 audio-bearing speaker directories and a minimum of 123 independent groups per speaker after mic-copy normalization, so this split covers all 110 speakers and leaves at least 13 reserve groups. Corrupt or invalid files are removed using one rule before sampling, and all classes remain balanced.

The canonical data split uses split seed 42 and is shared by all formal training seeds. Training seeds 41, 42, and 43 vary initialization, minibatch order, crop selection, and augmentation randomness; they do not create different formal test sets. The test manifest is written and locked before training, but test labels and metrics are unavailable to training and stage-gate code.

Training uses random three-second crops. Validation and test use deterministic center crops. Additional one-, two-, three-, and five-second views are evaluation-only.

### Text-Controlled Secondary Split

VCTK transcripts under `data/VCTK/txt` are normalized by lowercasing and whitespace folding. Transcript resolution requires the exact speaker/utterance text file. The current checkout has audio for `p315` but no `data/VCTK/txt/p315` directory, and numeric sentence IDs are not reliable cross-speaker transcript keys. Therefore `p315` remains in the 110-class canonical experiment but is excluded from the transcript-controlled secondary challenge. No inferred or ASR-generated transcript may silently replace missing ground truth.

For the remaining 109 speakers, unique normalized transcript strings are globally assigned to train, validation, or test before per-speaker sampling, so the same text cannot cross splits for any speaker. A dry run over the intersection of available audio groups and exact transcripts found that deterministic text split seed 1 supports exact 80/10/10 samples for all 109 eligible speakers. Before exact subsampling, the minimum available counts for any eligible speaker are 96 train, 14 validation, and 13 test groups. The resulting secondary challenge contains 8,720 train, 1,090 validation, and 1,090 test groups. It is reported as a separate 109-class robustness result and cannot replace or be numerically conflated with the canonical 110-class endpoint.

## 6. Representation Cache

The frozen preprocessing path is:

```text
original audio -> frozen encode -> RVQ codes -> frozen decode -> reconstructed waveform
```

Each cache item records speaker ID, utterance group, transcript hash, split, source path, Base/LCA model, RVQ layer count, codes, reconstruction, original sample count, code-frame count, valid length, and checkpoint hash. LCA L3 is the primary condition. Base L3 is secondary; L1 and L2 support mixed-layer training and robustness evaluation.

The cache stores full-utterance codes and either compressed FLAC reconstructions or chunked tensors. Every training crop is represented by one shared normalized time interval. Audio boundaries are converted to token boundaries using the actual per-item ratio `code_frames / audio_samples`, with an expected rate near 50 frames per second from the 320-sample encoder downsampling rate. Audio and token branches must therefore see the same segment.

Before generation, the cache builder estimates required bytes. The default cache budget is 30 GB, generation requires free space of at least 1.2 times the estimate, and the run aborts before writing if either condition fails. L1/L2 waveforms may be generated on demand from cached codes when storing all six model/layer waveform conditions would exceed the budget. Cache generation must resume safely, validate every file, check finite values and code ranges, and record failures explicitly.

## 7. Identity Module Architecture

### Token Branch

Input shape is approximately `[batch, L, time]`.

1. Reserve code index 1024 as PAD and use one `Embedding(1025, 128, padding_idx=1024)` per RVQ layer.
2. Represent absent L2/L3 inputs with PAD plus an explicit RVQ-layer presence mask.
3. Add learnable RVQ-layer embeddings and temporal position encoding.
4. Concatenate layer features and project to `d_model=256`.
5. Apply four lightweight TDNN/Conformer blocks with dropout 0.1.
6. Apply mask-aware attentive statistics pooling.
7. Project to a 256-dimensional L2-normalized speaker embedding.

The first implementation should choose one stable temporal backbone. The alternative backbone is a later capacity ablation, not a parallel primary implementation.

### Reconstructed-Audio Branch

- Backbone: pretrained ECAPA-TDNN.
- Stage A: freeze ECAPA and train only projection and classification layers.
- Stage B: unfreeze upper ECAPA blocks with a small learning rate for codec-aware adaptation.
- Output: a 256-dimensional normalized embedding.
- WavLM-base-plus is a preregistered fallback only if ECAPA reaches a clear plateau; it must be reported as a separate model-size experiment.

The current `SpeakerEmbeddingExtractor` remains the frozen Exp22 baseline only because it loads `speechbrain.inference.speaker.EncoderClassifier` and evaluates under `torch.no_grad()`. Exp23 must define a separate trainable ECAPA module. Its implementation config records the SpeechBrain package version, source model revision, downloaded artifact hashes, waveform normalization, trainable parameter names, and exact blocks unfrozen in Stage B. A gradient-scope test must prove that frozen ECAPA blocks and all SpeechTokenizer parameters receive no gradients.

### Fusion

Let token and audio embeddings be `z_t` and `z_a`:

```text
g = sigmoid(MLP([z_t; z_a]))
z_f = LayerNorm(g * W_t(z_t) + (1 - g) * W_a(z_a))
```

Keep token, audio, and fusion classification heads for auxiliary supervision and ablations. Normal inference uses the fusion head.

### Classification Head

- Classes: 110
- Primary objective: AAM-Softmax
- Initial scale: 30
- Initial margin: 0.20
- Allowed margin search: 0.15, 0.20, 0.25, 0.30
- Outputs: speaker ID, Top-1 confidence, Top-5 list, and logit margin

## 8. Loss Function

```text
L_total = L_fusion
        + 0.30 * L_token
        + 0.30 * L_audio
        + 0.10 * L_consistency
```

- `L_fusion`, `L_token`, and `L_audio` are classification losses.
- `L_consistency` aligns reconstructed-audio embeddings with a frozen original-audio teacher embedding from the same utterance.
- If consistency hurts validation accuracy, first reduce its weight to 0.03, then run a complete removal ablation.

## 9. Training Stages

### Phase 0: Baseline Reproduction

Reproduce Exp22 summaries and verify speaker parsing, checkpoints, and data roots. A relative deviation greater than 5% blocks the new training run.

### Phase 1: Cache Build

Build and validate the canonical split caches for Base/LCA and L1/L2/L3. Validate the exact 110-speaker label set including `s5`, counts, balance, time alignment, lengths, code ranges, waveform finiteness, cache estimate, and storage budget. Separately validate that the text-controlled manifest contains exactly the 109 transcript-eligible speakers, excludes only `p315`, resolves every included audio group to an exact transcript, and has zero normalized-text overlap.

### Phase 2: Single-Branch Baselines

- B0: Exp22 frozen ECAPA profile matching.
- B1: reconstructed audio, ECAPA, ordinary Softmax.
- B2: reconstructed audio, ECAPA, AAM-Softmax.
- B3: RVQ token encoder, AAM-Softmax.

If B2 and B3 are both below 0.75 Top-1, stop before fusion and diagnose the data and representations.

### Phase 3: Fusion

Initialize from the best B2 and B3 validation checkpoints. Freeze both branches and train fusion for five epochs, then jointly tune the token branch and upper ECAPA blocks for 20-40 epochs. Early-stop on validation macro-F1 with patience 6. Select `best_macro_f1.pt` before viewing formal test results.

### Phase 4: Three-Seed Replication

Use validation results only for stage gates. If the seed42 validation Top-1 is at least 0.88, freeze the complete architecture, loss, hyperparameters, checkpoint-selection rule, calibration procedure, and command configuration. Then train seeds 41 and 43 on the same canonical split. After all three training runs are complete, execute one final test evaluation per seed. No test metric may change the configuration or decide whether another run is launched. The claim of 90% requires the three-seed mean, not the best seed, to reach 0.90.

## 10. Optimization Defaults

| Parameter group | Optimizer | Learning rate | Weight decay |
|---|---|---:|---:|
| Token encoder and fusion | AdamW | 3e-4 | 1e-2 |
| ECAPA projection and head | AdamW | 1e-4 | 1e-2 |
| Unfrozen ECAPA blocks | AdamW | 1e-5 | 1e-3 |

Use cosine decay, 5% warmup, gradient clipping at 1.0, bf16 when supported, and effective batch size 128 through accumulation. Limit hyperparameter search to learning rate, AAM margin, fusion dropout, and segment duration.

Training-only augmentation includes random gain, mild noise at 10-30 dB SNR, mild reverberation, random crop, and mixed L1/L2/L3 sampling with L3 probability at least 0.6. Strong pitch transformation is excluded. A no-augmentation ablation is mandatory.

## 11. Experiment Matrix

| ID | Input | Model | Purpose |
|---|---|---|---|
| B0 | Reconstruction | Frozen ECAPA profile matching | Exp22 baseline |
| B1 | Reconstruction | ECAPA + Softmax | Closed-set supervision baseline |
| B2 | Reconstruction | ECAPA + AAM-Softmax | Strong audio branch |
| B3 | RVQ codes | Token encoder + AAM-Softmax | Token branch |
| B4 | Both | Late score average | Simple fusion baseline |
| M1 | Both | Gated embedding fusion | Primary model |

Mandatory ablations remove each branch, replace gated fusion with concatenation, replace AAM with Softmax, remove consistency, remove augmentation, keep ECAPA fully frozen, and compare L3-only with mixed-layer training.

Stress tests cover Base-to-Base, LCA-to-LCA, cross-model transfer, the transcript-controlled split, one/two/three/five-second segments, L1/L2/L3, mild noise, reverberation, and the existing Exp3 channel perturbation configuration.

## 12. Metrics and Statistics

Primary metrics are Top-1 accuracy and macro-F1 on LCA L3. Secondary metrics are Top-5, balanced accuracy, per-speaker accuracy, worst-decile speaker accuracy, ECE, NLL, confusion matrix, latency, parameter count, and peak memory.

Fit temperature scaling using validation logits only, freeze the temperature, and report both calibrated and uncalibrated test ECE/NLL. Report seed mean and standard deviation as training-randomness summaries; a three-seed t-interval may be included but is not the main uncertainty claim.

Within each seed, use 10,000 hierarchical bootstrap replicates: sample speakers first, then utterance groups within sampled speakers. Compare the primary model with the best single branch using paired hierarchical bootstrap, optionally supported by McNemar's test on matched predictions. Apply Holm correction across mandatory ablations and report effect sizes, not only p-values.

## 13. Success and Stop Criteria

### Primary Product Success

1. SpeechTokenizer hashes remain unchanged.
2. Three-training-seed mean LCA L3 Top-1 >= 0.90 on the canonical test split.
3. Three-training-seed mean LCA L3 macro-F1 >= 0.90.
4. Hierarchical-bootstrap Top-1 95% CI lower bound >= 0.88.
5. All 110 classes are present and correctly labeled.
6. No utterance-group, transcript, or microphone-copy leakage exists for the applicable split.

### Mechanism Success

Gated fusion improves over the strongest single branch by at least 2 percentage points or reduces relative error by at least 15%. Failure of this criterion does not invalidate primary product success; it means the fusion-complementarity hypothesis is unsupported.

### Robustness Success

The LCA L3 fusion model reaches at least 0.85 Top-1 on the 109-class three-second transcript-controlled test and at least 0.85 Top-1 on the two-second canonical 110-class test. One-second, L1/L2, cross-model, and perturbation results are reported without post-hoc threshold changes.

Stage gates:

- Best B2/B3 validation Top-1 < 0.75: stop and diagnose.
- M1 seed42 validation Top-1 < 0.85: do not launch additional seeds.
- M1 seed42 validation Top-1 in 0.85-0.88: allow one bounded revision using validation evidence only.
- M1 seed42 validation Top-1 >= 0.88: freeze the configuration and launch seeds 41 and 43.
- Three-seed mean < 0.90: report that the target was not reached; do not tune on the test set.

## 14. Planned Artifacts

Run directory:

```text
output/experiments/exp23_frozen_tokenizer_speaker_classifier_YYYYMMDD_seedXX/
  configs/ commands/ logs/ checkpoints/ metrics/ reports/ artifacts/ cache_manifest/
```

Required outputs include `speaker_split.json`, `text_disjoint_split.json`, `cache_manifest.json`, `cache_storage_estimate.json`, `experiment_config.json`, `checkpoint_manifest.json`, `dependency_lock.json`, `trainable_parameter_manifest.json`, `train_history.csv`, `test_predictions.csv`, `summary_by_seed.csv`, `per_speaker.csv`, `ablation_summary.csv`, `calibration.json`, `exp23_summary.md`, and `confusion_matrix.png`.

## 15. Planned Implementation Entry Points

These files are planned and do not exist yet:

- `scripts/build_exp23_speaker_cache.py`
- `scripts/train_exp23_speaker_classifier.py`
- `scripts/evaluate_exp23_speaker_classifier.py`
- `scripts/aggregate_exp23_speaker_classifier.py`
- `speechtokenizer/speaker_identity/token_encoder.py`
- `speechtokenizer/speaker_identity/audio_encoder.py`
- `speechtokenizer/speaker_identity/fusion.py`
- `speechtokenizer/speaker_identity/model.py`
- `tests/test_exp23_split_integrity.py`
- `tests/test_exp23_text_split_integrity.py`
- `tests/test_exp23_frozen_tokenizer.py`
- `tests/test_exp23_alignment.py`
- `tests/test_exp23_gradient_scope.py`
- `tests/test_exp23_model_shapes.py`
- `tests/test_exp23_metrics.py`

Planned command shape:

```powershell
$CONFIG_PATH = path/to/experiment_config.json
$RUN_DIR = path/to/seed42_run
$SEED41_RUN = path/to/seed41_run
$SEED42_RUN = path/to/seed42_run
$SEED43_RUN = path/to/seed43_run
python scripts/build_exp23_speaker_cache.py --config $CONFIG_PATH
python scripts/train_exp23_speaker_classifier.py --config $CONFIG_PATH --seed 42
python scripts/evaluate_exp23_speaker_classifier.py --run-dir $RUN_DIR
python scripts/aggregate_exp23_speaker_classifier.py --runs $SEED41_RUN $SEED42_RUN $SEED43_RUN
```

The implementation phase must define real CLI arguments in a checked-in config and record exact commands under `commands/`. This document does not present planned commands as existing functionality.

## 16. Monitoring and Risks

Use a provisional 24-hour hard timeout per training run, revised after throughput measurement. Log loss, learning rate, gradient norm, throughput, and memory every 100 steps. NaN/Inf, missing classes, split overlap, or tokenizer hash changes are hard failures. Crashed runs are not silently retried.

The final model must export a classifier checkpoint, speaker-label map, preprocessing metadata, and calibration temperature. A deployment smoke test integrates these artifacts into the three-user client without changing router behavior. Speaker inference runs outside the packet-decode critical section, and the target CPU p95 is below the one-second speaker update hop for a three-second window.

Key risks:

- The 90% result may depend on closed-set supervision and must not be described as open-set verification.
- Microphone duplicates or cache reuse can inflate accuracy; split-integrity tests are mandatory.
- The classifier may exploit recording or text artifacts; mitigate with utterance-disjoint splits, augmentation, and cross-Base/LCA tests.
- Frozen LCA representations may have insufficient identity information; failure is a valid finding.
- WavLM can hide engineering cost; report parameters, latency, and memory separately if used.

## 17. Primary References

1. Desplanques, B., Thienpondt, J., and Demuynck, K. (2020). ECAPA-TDNN. Interspeech 2020. DOI: 10.21437/Interspeech.2020-2650.
2. Chen, S. et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. IEEE JSTSP 16(6). arXiv: 2110.13900.
3. Deng, J. et al. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR 2019. DOI: 10.1109/CVPR.2019.00482.
4. Chung, J. S., Nagrani, A., and Zisserman, A. (2018). VoxCeleb2: Deep Speaker Recognition. Interspeech 2018. DOI: 10.21437/Interspeech.2018-1929.

These references motivate component selection. Exp23 validity ultimately depends on the repository's leakage-controlled measurements.

## 18. Self-Review Checklist

- [x] SpeechTokenizer remains frozen and bitrate is unchanged.
- [x] The task is explicitly fixed-110 closed-set classification.
- [x] The 90% threshold is a target, not a guaranteed result.
- [x] Utterance-group and microphone-copy leakage rules are explicit.
- [x] The canonical split is feasible for all 110 speakers and formal seeds reuse one test split.
- [x] Speaker discovery includes `s5`; `p315` remains in the canonical 110-class task and is explicitly excluded only from the text-controlled challenge.
- [x] A transcript-controlled 109-class secondary split is defined without inventing missing `p315` ground truth.
- [x] Test metrics are excluded from development stage gates.
- [x] LCA L3 is the unambiguous primary endpoint; Base is secondary.
- [x] Audio/token alignment, RVQ PAD, and layer masks are defined.
- [x] The trainable ECAPA dependency and gradient-scope contract are defined.
- [x] Base/LCA, L1/L2/L3, and three-seed evaluation are included.
- [x] Single-branch baselines, fusion, and mandatory ablations are defined.
- [x] Hierarchical statistics, calibration, stage gates, and stop rules are defined.
- [x] Storage-budget and deployment acceptance rules are defined.
- [x] Planned scripts are clearly marked as not implemented.
- [x] No unresolved placeholders remain.
