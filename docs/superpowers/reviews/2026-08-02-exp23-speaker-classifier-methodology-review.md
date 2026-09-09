# Exp23 Speaker Classifier Methodology Review

## Review Metadata

- Review Date: 2026-08-02
- Reviewed File: `docs/superpowers/specs/2026-08-02-exp23-frozen-tokenizer-speaker-classifier-design.md`
- Review Mode: methodology focus plus devil's advocate
- Overall Recommendation: Major Revision
- Confidence: 5/5
- Manuscript Policy: read-only review; this report does not modify the reviewed design

## Summary Assessment

The design has a sound high-level direction: freeze the deployed SpeechTokenizer, train a fixed-110 closed-set identity module, separate waveform and RVQ-token evidence, and require leakage checks, ablations, multiple seeds, and explicit failure reporting. These choices are substantially stronger than treating the existing ECAPA profile matcher as the final system.

The current version is not ready to become an implementation plan. Two blocking issues must be corrected first. The proposed 120/20/20 utterance-group split is impossible for all 110 speakers because the live VCTK checkout contains one speaker with only 123 independent groups after mic1/mic2 normalization. More importantly, the design uses the seed42 test result to decide whether to launch seeds 41 and 43, which makes the test set part of model-development control flow. That creates adaptive test leakage and invalidates the stated independent-test interpretation.

Several additional major issues affect reproducibility and interpretation: the primary endpoint does not identify whether LCA or Base is the required 90% target; audio/token crop alignment is undefined; mixed RVQ layer masking is unspecified; the current ECAPA utility is an inference-only wrapper rather than a fine-tuning recipe; transcript-content confounding is acknowledged but not controlled; and the statistical plan bootstraps utterances without respecting speaker-level clustering.

## Verified Strengths

1. **Correct system boundary**: SpeechTokenizer freezing is explicit and can be enforced by state-dict hashes.
2. **Appropriate task label**: the document clearly calls the task fixed-class closed-set classification rather than open-set biometric verification.
3. **Strong baseline structure**: Exp22 profile matching, supervised audio, token-only, simple fusion, and gated fusion provide an interpretable progression.
4. **Useful negative-result policy**: the 90% threshold is framed as a target, and the design prohibits test-speaker removal to manufacture success.
5. **Ablation awareness**: branch removal, loss replacement, augmentation removal, and ECAPA freeze/unfreeze are all included.

## Blocking Findings

### B1. The proposed balanced split is not feasible

- Severity: BLOCKER
- Location: reviewed design line 101
- Design claim: 120 train + 20 validation + 20 test groups for every speaker.
- Repository evidence: after normalizing mic1/mic2 copies, VCTK has 110 speakers, but the minimum is 123 independent utterance groups (`p362`). Only 109 speakers have at least 160 groups.
- Impact: the planned split cannot include all 110 speakers, contradicting both the data section and the final success criteria.
- Required revision: use a feasible fixed split such as 80/15/15 groups per speaker, producing 8,800/1,650/1,650 groups, with remaining groups reserved. Alternatively, define a documented adaptive per-speaker split and use class-balanced sampling, but the fixed 80/15/15 design is easier to audit.
- Acceptance criterion: a split-builder dry run proves 110 classes, exact counts, zero group overlap, and zero mic-copy overlap before any model code is implemented.

### B2. The formal test set is used as a development gate

- Severity: BLOCKER
- Location: reviewed design lines 196 and 245-249
- Design claim: launch seeds 41/43 only when seed42 test Top-1 reaches 0.88.
- Impact: seed42 test performance influences experiment continuation and the permitted revision. The test set is therefore not independent, even if no gradient is computed from it.
- Required revision: all stage gates must use validation metrics. Freeze the architecture, hyperparameters, loss weights, checkpoint-selection rule, and stopping policy before opening the canonical test manifest. Evaluate the test set once per training seed after the configuration is locked.
- Acceptance criterion: the implementation has no code path that reads test metrics before the final evaluation command.

## Major Findings

### M1. The primary 90% endpoint is ambiguous between Base and LCA

- Location: lines 17, 41-42, 179, and 233-249
- Problem: the design evaluates both Base and LCA but defines success only as an L3 result. A 92% Base result with a 70% LCA result could be incorrectly presented as success even if the intended deployed model is the fine-tuned LCA checkpoint.
- Recommendation: define `LCA L3 closed-set Top-1` as the primary endpoint if the user's intended upstream is the fine-tuned model. Treat Base L3 as a secondary reference and diagnostic upper bound. If both are intended deployment targets, define two separate success claims.

### M2. Audio/token alignment is underspecified

- Location: lines 105-113 and 115-129
- Problem: random audio cropping and token cropping must select the same temporal interval. The current model has a 320-sample encoder downsampling rate at 16 kHz, corresponding to 50 token frames per second. A three-second segment should therefore align with approximately 150 token frames, subject to exact convolution boundary behavior.
- Recommendation: cache full utterances or aligned chunk metadata, then derive audio sample boundaries and token-frame boundaries from one shared segment record. Add shape and alignment tests using encode-decode round trips.

### M3. Mixed L1/L2/L3 inputs have no missing-layer representation

- Location: lines 121-124 and 221
- Problem: the proposed token encoder uses three per-layer embedding tables, but mixed-layer training does not define what occupies absent L2/L3 positions for an L1 sample.
- Recommendation: reserve a PAD code, for example index 1024 with `Embedding(1025, 128)`, add an RVQ-layer presence mask, and ensure attentive pooling and fusion cannot treat PAD as a normal transmitted code.

### M4. ECAPA fine-tuning is not reproducibly defined

- Location: lines 132-134 and 262-277
- Repository evidence: `scripts/speaker_identity_utils.py` loads `speechbrain.inference.speaker.EncoderClassifier` and computes embeddings inside `torch.no_grad()`. It is a frozen inference wrapper, not a training recipe.
- Problem: the design does not pin a SpeechBrain version, model revision, preprocessing configuration, trainable module path, or exact ECAPA blocks to unfreeze.
- Recommendation: specify the dependency version and model artifact hash, define whether the SpeechBrain embedding model is extracted into a trainable module or reimplemented, list trainable parameter names, and test that gradients are present only in the intended ECAPA blocks.

### M5. Transcript-content confounding needs a stronger control

- Location: line 298
- Repository evidence: the checkout includes VCTK transcripts. Across normalized sentence IDs, 398 IDs occur for only one speaker, while many others are shared across large subsets of speakers.
- Problem: a random per-speaker utterance split may let the classifier exploit speaker-correlated lexical or prompt patterns, especially in the token branch.
- Recommendation: add a secondary text-disjoint evaluation in which normalized transcript hashes are globally assigned to splits, or construct a parallel-text challenge subset from prompts shared across many speakers. The primary 90% claim should report both ordinary utterance-disjoint and text-controlled results.

### M6. Statistical resampling ignores speaker clustering

- Location: lines 227-229
- Problem: flat utterance bootstrap treats all test clips as independent, even though clips from the same speaker share identity and recording characteristics. Three-seed t intervals with n=3 are also too unstable to carry the main uncertainty claim.
- Recommendation: use hierarchical bootstrap: sample speakers first, then utterance groups within speakers. Report seed mean/std as optimization variability, and use the hierarchical interval as the main fixed-population uncertainty estimate. Use paired hierarchical bootstrap for model comparisons.

### M7. Calibration metrics need an explicit calibration protocol

- Location: line 227
- Problem: ECE and NLL are not directly meaningful if raw AAM-Softmax scores are interpreted as calibrated probabilities.
- Recommendation: fit temperature scaling on validation logits only, freeze the temperature, and then report test ECE/NLL. Report uncalibrated and calibrated values separately.

### M8. Performance success and fusion-mechanism success are conflated

- Location: lines 233-242
- Problem: requiring both >=90% and a two-point fusion gain means a 93% token-only model with a 93.5% fusion model would fail the final criterion even though the main product goal was achieved.
- Recommendation: separate the claims:
  - Primary product success: LCA L3 Top-1 and macro-F1 reach 0.90.
  - Mechanism success: gated fusion significantly improves over the strongest branch.
  - Robustness success: predefined short-duration and perturbation thresholds are met.

### M9. Cache storage and deployment acceptance are missing

- Problem: caching Base/LCA x L1/L2/L3 reconstructed waveforms can consume substantial disk, especially if full utterances are stored as float32. The repository already contains evidence of prior disk-full experiment interruptions. The design also stops at offline metrics and does not define export into the three-user demo.
- Recommendation: add a storage preflight, estimated cache size, FLAC or chunked tensor format, maximum cache budget, and cleanup policy. Add a deployment gate that exports the label map and classifier checkpoint, measures CPU/GPU latency, and validates the existing demo on held-out users from the fixed 110 classes.

## Devil's Advocate Challenge

The strongest counter-argument is that a 90% result may demonstrate memorization of a fixed VCTK classification problem rather than recovery of speaker identity from low-bitrate speech. The same 110 people appear in training and testing, some prompts are speaker-correlated, microphone and recording conditions are stable, and the token branch may learn dataset-specific distributions that do not correspond to perceptual identity. A large classifier can therefore exceed the Exp22 general-purpose profile matcher without proving that SpeechTokenizer preserves identity in a transferable sense.

The design remains valuable if it states the narrower claim precisely: a frozen SpeechTokenizer representation supports accurate closed-set recognition for enrolled VCTK identities. To defend even that claim, the experiment needs a text-controlled test, a cross-Base/LCA evaluation, a short-duration challenge, and a clear separation between classification accuracy and open-set verification.

## Revision Roadmap

| Priority | Revision | Verification criterion |
|---|---|---|
| P0 | Replace 120/20/20 with a feasible split | Dry run includes all 110 speakers with zero overlap |
| P0 | Remove all test-driven stage gates | Test metrics are read only by the final evaluator |
| P0 | Declare LCA L3 or Base L3 as the primary endpoint | One unambiguous primary condition is named |
| P1 | Define aligned waveform/token segmentation | Unit tests prove shared time boundaries |
| P1 | Define PAD and layer masks for L1/L2/L3 | Shape and masking tests pass |
| P1 | Pin and specify the trainable ECAPA path | Dependency, artifact hash, and trainable parameters are recorded |
| P1 | Add transcript-controlled evaluation | Text hashes do not cross the controlled split |
| P1 | Replace flat bootstrap with hierarchical bootstrap | CI code resamples speakers and utterances |
| P1 | Separate product, mechanism, and robustness success | Each claim has its own threshold |
| P2 | Add storage budget and demo export acceptance | Cache estimate and inference benchmark are reported |

## Editorial Decision

**Major Revision.** The overall architecture is promising and the experiment can become rigorous, but implementation should not start from the current specification. The infeasible split and adaptive use of the test set are blocking defects. After the P0 items are corrected, the remaining P1 items should be incorporated before creating the implementation plan.

