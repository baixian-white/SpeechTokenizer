# Exp23 Frozen SpeechTokenizer Speaker Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fixed-110-speaker classifier behind the frozen LCA L3 SpeechTokenizer and evaluate whether mean Top-1 and macro-F1 exceed 90%.

**Architecture:** Add a separate `speechtokenizer.speaker_identity` package. Data manifests, frozen caches, token/audio branches, fusion, training, evaluation, and demo inference remain independent modules; test metrics are unavailable to training and stage-gate code.

**Tech Stack:** Python, PyTorch, torchaudio, SpeechBrain ECAPA-TDNN, NumPy, scikit-learn, unittest, JSON/CSV.

**Spec:** `docs/superpowers/specs/2026-08-02-exp23-frozen-tokenizer-speaker-classifier-design.md`

**Policy:** Do not commit unless explicitly requested.

---

### Task 1: Lock Configuration and Provenance

**Files:**
- Create: `config/exp23_speaker_classifier.json`
- Modify: `scripts/experiment_utils.py`
- Create: `tests/test_exp23_config.py`

- [ ] **Step 1: Write the failing contract test**

```python
cfg = json.loads(Path('config/exp23_speaker_classifier.json').read_text(encoding='utf-8'))
self.assertEqual(cfg['task']['speaker_count'], 110)
self.assertEqual(cfg['primary_condition'], {'model': 'lca', 'rvq_layers': 3})
self.assertEqual(cfg['split']['counts'], {'train': 80, 'validation': 15, 'test': 15})
self.assertEqual(cfg['stage_gates']['metric_split'], 'validation')
```

- [ ] **Step 2: Run it and verify failure**

Run: `python -m unittest tests.test_exp23_config -v`

Expected: FAIL because the config is absent.

- [ ] **Step 3: Create the config and provenance helper**

Lock canonical seed/counts `42` and `80/15/15`; text-controlled seed/counts `1` and `80/10/10`, excluding only `p315`; PAD `1024`; AAM scale/margin `30/0.2`; seeds `41/42/43`; cache budget `30 GB`; validation replication gate `0.88`. Add `collect_environment_metadata()` recording Python, platform, Git revision, and installed versions of torch, torchaudio, SpeechBrain, NumPy, and sklearn.

- [ ] **Step 4: Verify pass**

Run: `python -m unittest tests.test_exp23_config -v`

Expected: PASS.

---

### Task 2: Build Leakage-Safe Manifests

**Files:**
- Create: `speechtokenizer/speaker_identity/__init__.py`
- Create: `speechtokenizer/speaker_identity/data.py`
- Create: `tests/test_exp23_split_integrity.py`
- Create: `tests/test_exp23_text_split_integrity.py`
- Modify: `setup.py`

- [ ] **Step 1: Write failing tests**

```python
self.assertEqual(normalize_utterance_group(Path('p225_001_mic1.flac')), 'p225_001')
result = build_text_controlled_split(samples, 2, 1, 1, seed=1, excluded_speakers={'p315'})
self.assertFalse(set(result['transcript_hashes']['train']) & set(result['transcript_hashes']['test']))
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m unittest tests.test_exp23_split_integrity tests.test_exp23_text_split_integrity -v`

Expected: FAIL because the module is absent.

- [ ] **Step 3: Implement manifests**

Canonical discovery must include all audio directories, including `s5`, normalize mic copies into one group, select one microphone, and sample exact `80/15/15` with seed `42`. Text-controlled discovery must require exact transcript files, exclude only `p315`, assign globally unique normalized transcript hashes with seed `1`, sample exact `80/10/10`, and reject overlap.

- [ ] **Step 4: Package and verify real counts**

Add `speechtokenizer.speaker_identity` to `setup.py`. Run the two tests, then a real dry run.

Expected: canonical `110` speakers and `8800` train groups; controlled `109` speakers and `8720` train groups.

---

### Task 3: Build Frozen Aligned Caches

**Files:**
- Create: `speechtokenizer/speaker_identity/cache.py`
- Create: `scripts/build_exp23_speaker_cache.py`
- Create: `tests/test_exp23_alignment.py`
- Create: `tests/test_exp23_cache.py`
- Create: `tests/test_exp23_frozen_tokenizer.py`

- [ ] **Step 1: Write failing tests**

```python
bounds = aligned_crop_bounds(160000, 500, 32000, 48000)
self.assertEqual((bounds.token_start, bounds.token_end), (100, 250))
with self.assertRaisesRegex(RuntimeError, 'SpeechTokenizer state changed'):
    assert_state_dict_unchanged({'w': torch.tensor([1.0])}, {'w': torch.tensor([2.0])})
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m unittest tests.test_exp23_alignment tests.test_exp23_cache tests.test_exp23_frozen_tokenizer -v`

Expected: FAIL because cache APIs are absent.

- [ ] **Step 3: Implement cache contracts and CLI**

Store speaker, label, utterance group, split, model, RVQ layers, code/audio paths, sample/frame lengths, transcript hash, and checkpoint hash. Convert audio and token crops from one shared normalized interval. Reject non-finite audio and codes outside `[0,1023]`. Abort above 30 GB or below `1.2 x estimate` free space.

Use only:

```python
with torch.inference_mode():
    codes = model.encode(waveform.view(1, 1, -1), n_q=layers, st=0)
    reconstruction = model.decode(codes, st=0)
```

Set every tokenizer parameter `requires_grad=False`, record before/after hashes, write atomic NPZ/FLAC cache files, and resume only validated records.

- [ ] **Step 4: Verify**

Run the three unit tests and a two-speaker CPU cache smoke. Expected: PASS and identical tokenizer hashes.

### Task 4: Implement Token and Audio Encoders

**Files:**
- Create: `speechtokenizer/speaker_identity/token_encoder.py`
- Create: `speechtokenizer/speaker_identity/audio_encoder.py`
- Create: `tests/test_exp23_model_shapes.py`
- Create: `tests/test_exp23_gradient_scope.py`

- [ ] **Step 1: Write failing shape and gradient tests**

```python
embedding = token_encoder(codes, frame_mask, layer_mask)
self.assertEqual(embedding.shape, (2, 256))
self.assertTrue(torch.allclose(embedding.norm(dim=-1), torch.ones(2), atol=1e-5))

audio_encoder.configure_stage('B', ['embedding_model.blocks.4'])
self.assertTrue(any('blocks.4' in name for name in audio_encoder.trainable_parameter_names()))
self.assertFalse(any('blocks.0' in name for name in audio_encoder.trainable_parameter_names()))
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m unittest tests.test_exp23_model_shapes tests.test_exp23_gradient_scope -v`

Expected: FAIL because encoders are absent.

- [ ] **Step 3: Implement RVQ encoder**

Use three `Embedding(1025,128,padding_idx=1024)` tables, learned layer embeddings, explicit layer-mask multiplication, projection to 256, four depthwise TDNN residual blocks, mask-aware attentive mean/std pooling, and normalized 256-dimensional output.

- [ ] **Step 4: Implement trainable ECAPA wrapper**

Load `EncoderClassifier.from_hparams`, but do not call its inference-only `encode_batch` under `no_grad`. Call normalization, feature extraction, mean/variance normalization, and `embedding_model` directly. Stage A freezes all ECAPA parameters and trains projection/head. Stage B unfreezes only configured name patterns. Record SpeechBrain version, source, artifact hashes, preprocessing, and trainable parameter names.

- [ ] **Step 5: Verify**

Run the two test modules. Expected: PASS using fake SpeechBrain modules without downloads. A real dependency preflight must stop rather than fall back to MFCC when SpeechBrain is unavailable.

---

### Task 5: Implement Fusion, AAM Loss, and Full Model

**Files:**
- Create: `speechtokenizer/speaker_identity/fusion.py`
- Create: `speechtokenizer/speaker_identity/losses.py`
- Create: `speechtokenizer/speaker_identity/model.py`
- Create: `tests/test_exp23_losses.py`
- Extend: `tests/test_exp23_model_shapes.py`

- [ ] **Step 1: Write failing tests**

```python
output = model(fake_batch())
self.assertEqual(output.fusion_logits.shape, (2, 110))
self.assertEqual(output.fusion_embedding.shape, (2, 256))

logits = aam_head(F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1), torch.tensor([0]))
self.assertLess(logits[0, 0], 30.0)
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m unittest tests.test_exp23_losses tests.test_exp23_model_shapes -v`

Expected: FAIL because model components are absent.

- [ ] **Step 3: Implement gated fusion and heads**

```python
gate = self.gate(torch.cat([token_embedding, audio_embedding], dim=-1))
fused = gate * self.token_projection(token_embedding)
fused = fused + (1.0 - gate) * self.audio_projection(audio_embedding)
fused = F.normalize(self.norm(fused), dim=-1)
```

Implement AAM-Softmax with scale `30` and margin `0.2`, plus ordinary Softmax for B1/ablations. Keep token, audio, and fusion heads.

- [ ] **Step 4: Implement joint loss and output contract**

```python
total = fusion_loss + 0.30 * token_loss + 0.30 * audio_loss
total = total + consistency_weight * consistency_loss
```

The dataclass output contains token/audio/fusion embeddings, three logits tensors, and gate values. The model consumes caches only and never owns SpeechTokenizer.

- [ ] **Step 5: Verify**

Run the two test modules. Expected: PASS.

---

### Task 6: Implement Dataset and Training CLI

**Files:**
- Extend: `speechtokenizer/speaker_identity/cache.py`
- Create: `speechtokenizer/speaker_identity/training.py`
- Create: `scripts/train_exp23_speaker_classifier.py`
- Create: `tests/test_exp23_training.py`

- [ ] **Step 1: Write failing training tests**

```python
first = validation_dataset[0]
second = validation_dataset[0]
self.assertTrue(torch.equal(first['codes'], second['codes']))
self.assertEqual(replication_decision(0.89), 'freeze_and_replicate')
self.assertEqual(replication_decision(0.84), 'stop')
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m unittest tests.test_exp23_training -v`

Expected: FAIL because dataset/training APIs are absent.

- [ ] **Step 3: Implement dataset and augmentation**

Training uses random aligned three-second crops; validation/test use center crops. Collation pads codes with `1024`, builds frame/layer masks, pads waveform with zero, and returns integer labels. Training-only augmentation is random gain, 10-30 dB noise, mild reverb, crop, and mixed L1/L2/L3 with L3 probability at least `0.6`; no pitch shifting.

- [ ] **Step 4: Implement training loop**

Use AdamW groups at `3e-4` token/fusion, `1e-4` projection/head, `1e-5` unfrozen ECAPA; cosine decay; 5% warmup; clip `1.0`; bf16 when supported; effective batch `128`; early-stop on validation macro-F1. Save only validation-selected `best_macro_f1.pt`.

- [ ] **Step 5: Enforce validation-only gates**

Training must never load a test manifest. Seed42 validation below `0.85` stops; `0.85-0.88` permits one bounded revision; at least `0.88` freezes config and launches seeds 41/43.

- [ ] **Step 6: Verify and smoke**

Run `tests.test_exp23_training`, then a 20-step CPU B3 overfit smoke. Expected: decreasing loss and no SpeechTokenizer parameter in the optimizer manifest.

### Task 7: Implement Evaluation and Statistics

**Files:**
- Create: `speechtokenizer/speaker_identity/metrics.py`
- Create: `scripts/evaluate_exp23_speaker_classifier.py`
- Create: `tests/test_exp23_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
temperature = fit_temperature(validation_logits, validation_labels)
self.assertGreater(temperature, 0.0)
ci = hierarchical_bootstrap_accuracy(rows, replicates=200, seed=42)
self.assertLessEqual(ci['ci_low'], ci['point_estimate'])
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m unittest tests.test_exp23_metrics -v`

Expected: FAIL because metrics are absent.

- [ ] **Step 3: Implement metrics and calibration**

Return Top-1, Top-5, macro-F1, balanced accuracy, per-speaker accuracy, worst-decile accuracy, NLL, and 15-bin ECE. Fit one positive temperature on validation logits only; save it before calculating calibrated test metrics.

- [ ] **Step 4: Implement hierarchical comparisons**

Each replicate samples speakers with replacement, then utterance groups within sampled speakers. Paired comparisons resample matched utterance groups and calculate primary-minus-baseline accuracy. Apply Holm correction across mandatory ablations.

- [ ] **Step 5: Isolate final test evaluation**

Require `--allow-test-evaluation` for `split=test`; refuse to overwrite `test_predictions.csv` without `--force`. Load only a frozen checkpoint, locked manifest, label map, and validation-fitted temperature.

- [ ] **Step 6: Verify**

Run: `python -m unittest tests.test_exp23_metrics -v`

Expected: PASS.

---

### Task 8: Aggregate Seeds and Claims

**Files:**
- Create: `scripts/aggregate_exp23_speaker_classifier.py`
- Create: `tests/test_exp23_aggregate.py`

- [ ] **Step 1: Write failing claim-separation test**

```python
summary = summarize_claims(seed_rows, bootstrap_ci_low=0.89, fusion_gain=0.005,
                           relative_error_reduction=0.05)
self.assertTrue(summary['product_success'])
self.assertFalse(summary['mechanism_success'])
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m unittest tests.test_exp23_aggregate -v`

Expected: FAIL because aggregation is absent.

- [ ] **Step 3: Implement three-seed aggregation**

Require exactly seeds `41/42/43`; report mean/std, never best seed. Product success requires mean Top-1 and macro-F1 at least `0.90` plus bootstrap lower bound at least `0.88`. Mechanism success requires two absolute points or 15% relative error reduction. Robustness success requires 109-class controlled and two-second 110-class Top-1 at least `0.85`.

- [ ] **Step 4: Write artifacts and verify**

Write `summary_by_seed.csv`, `ablation_summary.csv`, `success_claims.json`, and `exp23_summary.md`, retaining stopped and negative runs. Run the aggregation test; expected PASS.

---

### Task 9: Export Bundle and Integrate Demo

**Files:**
- Create: `speechtokenizer/speaker_identity/inference.py`
- Create: `scripts/export_exp23_speaker_classifier.py`
- Create: `tests/test_exp23_export.py`
- Modify: `3用户demo/speechtokenizer_now/speechtokenizer/三用户中心路由通信demo/speaker_identity.py`
- Modify: `3用户demo/speechtokenizer_now/speechtokenizer/三用户中心路由通信demo/group_client.py`
- Create: `tests/test_demo_exp23_speaker_identity.py`

- [ ] **Step 1: Write failing export/demo tests**

```python
metadata = validate_export_bundle(bundle_dir)
self.assertEqual(metadata['speaker_count'], 110)
prediction = identifier.update('sender', np.ones(1600, dtype=np.float32))
self.assertEqual(prediction.predicted_speaker, 'p225')
```

- [ ] **Step 2: Run and verify failure**

Run: `python -m unittest tests.test_exp23_export tests.test_demo_exp23_speaker_identity -v`

Expected: FAIL because bundle mode is absent.

- [ ] **Step 3: Implement fixed bundle contract**

Bundle files are `classifier.pt`, `speaker_labels.json`, `preprocessing.json`, `calibration.json`, `model_config.json`, and `manifest.json`. Prediction returns the existing `SpeakerPrediction` fields using calibrated Top-1 confidence and logit margin. `verified` is only a display threshold, not unknown-speaker verification.

- [ ] **Step 4: Preserve legacy demo mode**

Selection order: injected test classifier, Exp23 bundle, existing profile matcher. Add `--speaker_classifier_bundle`, `--speaker_classifier_device`, and `--speaker_classifier_threshold`; reject simultaneous bundle/profile arguments. Do not modify routing or buffering behavior.

- [ ] **Step 5: Benchmark and verify**

Export hashes and metadata, run 100 warmed CPU three-second inferences, and write p50/p95 latency plus peak memory. Acceptance: CPU p95 below one second. Run new tests plus `tests.test_demo_speaker_identity`; expected PASS.

### Task 10: Runbook and Acceptance

**Files:**
- Create: `docs/experiments/exp23-speaker-classifier-runbook.md`
- Modify: `REPRODUCIBILITY.md`
- Modify: `3用户demo/speechtokenizer_now/speechtokenizer/三用户中心路由通信demo/三用户中心路由通信demo使用说明.md`

- [ ] **Step 1: Document exact execution order**

Document: environment/checkpoint preflight; Exp22 reproduction; split dry run; cache estimate/build; B1/B2/B3; validation gate; B4/M1; configuration freeze; seeds 41/43; one final test evaluation per seed; aggregation; export; demo smoke.

- [ ] **Step 2: Document the product boundary**

Use this exact statement:

```text
Exp23 is a fixed-110 closed-set classifier. It recognizes only identities present during training. It does not support arbitrary-user enrollment, unknown-speaker rejection, biometric authentication, or spoof detection.
```

- [ ] **Step 3: Run focused tests**

Run: `python -m unittest tests.test_exp23_config tests.test_exp23_split_integrity tests.test_exp23_text_split_integrity tests.test_exp23_frozen_tokenizer tests.test_exp23_alignment tests.test_exp23_cache tests.test_exp23_gradient_scope tests.test_exp23_model_shapes tests.test_exp23_losses tests.test_exp23_training tests.test_exp23_metrics tests.test_exp23_aggregate tests.test_exp23_export tests.test_demo_exp23_speaker_identity tests.test_demo_speaker_identity -v`

Expected: all PASS.

- [ ] **Step 4: Run syntax and whitespace checks**

Run: `python -m compileall -q speechtokenizer/speaker_identity scripts/build_exp23_speaker_cache.py scripts/train_exp23_speaker_classifier.py scripts/evaluate_exp23_speaker_classifier.py scripts/aggregate_exp23_speaker_classifier.py scripts/export_exp23_speaker_classifier.py`

Run: `git diff --check`

Expected: exit 0 and no new whitespace errors.

- [ ] **Step 5: Audit formal-result integrity**

Prove identical canonical test-manifest hashes for all seeds; no test metrics before replication freeze; all 110 canonical classes present; controlled result labeled 109-class and excluding only `p315`; mean rather than best seed controls the 90% claim; stopped/negative runs remain reported; tokenizer hashes remain unchanged.

- [ ] **Step 6: Review targeted status**

Run: `git status --short -- config/exp23_speaker_classifier.json speechtokenizer/speaker_identity scripts tests docs/experiments/exp23-speaker-classifier-runbook.md`

Expected: only intended Exp23 files plus explicitly modified demo/docs files. Do not stage or commit without approval.

---

## Mandatory Ablations

| Requirement | Configuration |
|---|---|
| B1 audio Softmax | `model_id=b1` |
| B2 audio AAM | `model_id=b2` |
| B3 token AAM | `model_id=b3` |
| B4 score average | `model_id=b4` |
| M1 gated fusion | `model_id=m1` |
| Concatenation fusion | `fusion_type=concat` |
| No consistency | `loss.consistency=0.0` |
| No augmentation | `augmentation=false` |
| Fully frozen ECAPA | `ecapa_stage=A` |
| L3-only versus mixed | `layer_sampling=l3_only` or `mixed` |

## Stop Points

1. Stop before cache build if canonical/text manifests, disk preflight, or checkpoint hashes fail.
2. Stop before training if SpeechBrain is unavailable or gradient scope includes SpeechTokenizer/unintended ECAPA blocks.
3. Stop before fusion if both B2 and B3 validation Top-1 are below `0.75`.
4. Do not launch replication unless seed42 validation Top-1 reaches `0.88` after the allowed bounded revision policy.
5. Never change configuration from test metrics; report failure when the three-seed mean misses `0.90`.

## Self-Review

- [x] P0/P1/P2 requirements map to explicit tasks.
- [x] Canonical 110-class and controlled 109-class results remain separate.
- [x] Test metrics are isolated from training and stage gates.
- [x] SpeechTokenizer freezing, crop alignment, PAD/layer masks, ECAPA gradient scope, calibration, hierarchical statistics, cache budget, export, and demo latency are covered.
- [x] No unresolved implementation placeholders remain.
