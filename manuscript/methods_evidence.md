# Methods Evidence Notes

Updated: 2026-09-18.

## Scope

The English Methods in `manuscript.tex` covers the deep-learning
workflow: index representation, encoder NAS, base codec training, channel-aware
fine-tuning, reconstruction evaluation, speaker-classifier training and
evaluation, and deployment inference. Quantum-link implementation is outside
scope. Six-user experiments are incomplete and have only a placeholder.

This file is an author-side provenance record, not manuscript prose.

## Main-Text and Supplement Split

On 2026-09-18, only the English Methods range in `manuscript.tex` was
condensed. Text before Methods (including Results and all three tables),
the component figure, and the bibliography were preserved.

`supplementary_methods.tex` is a standalone document. Its body in
`supplementary_methods_body.tex` preserves the complete pre-compression
Methods, changing only the section-heading level for S1--S9.
`supplementary_methods_references.tex` preserves the original bibliography.
The supplement is the complete reproducibility record, rather than a
collection of newly executed experiments.

`methods_main_zh.md` translates the condensed main Methods.
`methods_zh.md` remains the original full-length translation.
The previous English source, PDF and complete Chinese Methods were retained
in `archive/20260918_before_methods_split/`.

The quantitative Results were not re-estimated, and the missing speaker
bootstrap metadata and six-user placeholder remain visible in both versions.

## Terminology

| Manuscript term | Meaning | Local aliases |
| --- | --- | --- |
| SCIT | Base NAS-encoder codec used in Results | SCIT-Speech-Base, Exp2 |
| SCIT-LCA | End-to-end channel-aware fine-tuned codec, formal v2 | LCA v2, Exp3 v2 |
| Frozen-codec speaker classifier | Token/audio gated-fusion classifier over 110 VCTK identities | Exp23 |
| Index erasure | Simulated index dropout with previous-index concealment | dropout-high |
| Speaker identification | Fixed-class identity prediction | Not open-set authentication |
| Nominal index rate | 500 bits/s per retained RVQ layer | Not packetized network traffic |

## Completion Pass

The continuation pass filled in the fixed-length serialization convention,
NAS search options and ranking penalties, straight-through quantization,
EMA codebook updates, feature-matching scale, the exact speaker-model
identifier, and the hierarchical bootstrap sampling unit. Table 1 now includes
the archived codec confidence intervals and an Opus bandwidth caveat. Existing
Results point estimates were not changed.

The missing speaker-bootstrap replicate count and RNG seed are marked visibly
in the draft, rather than only in a LaTeX comment. They are not inferred from
the analysis helper's defaults.

## Primary Local Evidence

### Architecture and NAS

- `output/experiments/exp1_nas_distill_run1_seed42/configs/nas_search_space.json`
- `output/experiments/exp1_nas_distill_run1_seed42/configs/staged_nas_runtime_config.json`
- `output/experiments/exp1_nas_distill_run1_seed42/commands/run_command.txt`
- `output/experiments/exp1_nas_distill_run1_seed42/artifacts/best_architecture/best_seanet_config.json`
- `output/experiments/exp1_nas_distill_run1_seed42/metrics/stage1_profile.csv`
- `output/experiments/exp1_nas_distill_run1_seed42/metrics/stage4_final.csv`
- `output/experiments/exp1_nas_distill_run1_seed42/reports/environment.json`
- `nas/run_staged_encoder_nas.py`
- `nas/evaluate_encoder_proxy.py`
- `nas/validate_short_distill.py`

The search executed 8,192 configuration candidates, then 512/64/8 proxy
candidates. Configuration signatures are not guaranteed to correspond to
distinct networks: the recorded `compress` variable is not consumed by the
NAS encoder constructor. Accordingly, the manuscript says "configuration
candidates", not "distinct architectures".

Selected candidate: `nas_seed42_000896`. Selection minimizes the combined
quality/resource score among final nondominated candidates; it is not the
minimum-parameter or minimum-MAC architecture. The corresponding Results
phrase was corrected without changing any reported number.

The approximately 55% RTF reduction uses stage-1 measurements
(0.00818658 versus 0.00367646). Stage-4 timings are different. The proxy
training and scoring reuse small training subsets, not a held-out NAS
validation set. The downstream decoder is frozen during NAS proxies but
trained normally during base codec training.

### Base Training, Fine-Tuning and Reconstruction Evaluation

- `output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json`
- `output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/configs/lca_finetune_config.json`
- `speechtokenizer/trainer/trainer.py`
- `speechtokenizer/trainer/lca_trainer.py`
- `speechtokenizer/trainer/loss.py`
- `speechtokenizer/quantization/vq.py`
- `speechtokenizer/quantization/core_vq.py`
- `speechtokenizer/model.py`
- `output/experiments/exp12_baseline_comparison_test300_20260610_seed42/`
- `output/experiments/exp7_librispeech_test_full_clean_20260606/`
- `output/experiments/exp8_librispeech_test_full_perturb_20260606/`
- `output/experiments/exp9_librispeech_test_full_packet_burst_20260608/`
- `scripts/run_exp4_baselines.py`
- `scripts/run_opus_baseline.py`
- `scripts/evaluate_lca_vs_base.py`
- `scripts/evaluate_packet_burst_loss.py`

Use recorded run configurations rather than generic defaults in the
reproducibility overview. In particular, the LCA v2 training condition samples
clean, 5%/10% index dropout, or 1%/3% substitution. Packet and burst loss are
separate evaluation conditions, not additional LCA training augmentations.

Table 1 means and intervals are taken from
`output/experiments/exp12_baseline_comparison_test300_20260610_seed42/runs/test-clean_300/metrics/per_method_summary.csv`,
using `scit_base/L=3`, `dac/n_q_3`, `encodec/bw1.5kbps_n_cb2`,
`opus/opus_6000bps`, and `pcm/16bit_16khz_passthrough`. The bootstrap protocol
is retained in `commands/analyze_baseline_300_stats.py`; the Opus bandwidth
caveat is documented in `reports/appendix_d_fragment.md` in the same experiment.
Intervals and means are rounded to three decimal places in the manuscript.

The codec constructor does not override the RVQ defaults: codebooks use
50-iteration k-means initialization, EMA decay 0.99 and a dead-code count
threshold of two. Codebook embeddings are buffers, not Adam-updated parameters.
Feature matching in `trainer/loss.py` is twice the sum of feature-map mean
absolute differences.

### Speaker Classifier

The primary checkout contains an older design. The implemented experiment and
formal measurements are in the separate existing worktree:
`D:/tmp/speechtokenizer-exp23`.

- `tmp/exp23_formal/seed{41,42,43}/training_config.json`
- `tmp/exp23_formal/seed42/environment.json`
- `tmp/exp23_formal/cache_lca_l3/cache_manifest/split_manifest.json`
- `tmp/exp23_formal/cache_lca_l3/cache_manifest/tokenizer_state_hashes.json`
- `tmp/exp23_formal/cache_lca_l3/cache_manifest/cache_failures.json`
- `tmp/exp23_formal/aggregate/formal_results.md`
- `tmp/exp23_formal/aggregate/bootstrap_summary.json`
- `tmp/exp23_formal/seed{41,42,43}/metrics/test_predictions.csv`
- `tmp/exp23_formal/export/seed42_bundle/`
- `scripts/train_exp23_speaker_classifier.py`
- `scripts/evaluate_exp23_speaker_classifier.py`
- `speechtokenizer/speaker_identity/{data,cache,token_encoder,audio_encoder,fusion,losses,training,metrics,inference}.py`

All paths in this subsection are relative to that worktree.

The formal run differs materially from the initial design:

| Item | Executed formal run |
| --- | --- |
| ECAPA backbone | Frozen stage A for all three seeds |
| Audio adaptation | Trainable 192-to-256 projection and classifier |
| Token temporal backbone | Four depthwise temporal residual blocks |
| Layer combination | Masked mean, not concatenation |
| Positional embeddings | No separate temporal-position embedding |
| Classification loss | Fusion + 0.3 token + 0.3 audio |
| Consistency loss | Disabled |
| Duration | 30 epochs; no implemented early stopping |
| Optimizer weight decay | 1e-4 |
| Learning-rate schedule | No warm-up; cosine horizon in minibatches, advanced on optimizer updates |
| Augmentation | Gain, Gaussian noise and token-layer masking |
| Mixed-layer audio | Audio stays cached L3; lower layers mask only the token branch |
| Calibration | 400 log-spaced temperatures over [0.05, 10], validation only |
| ECE | 15 equal-width bins |
| Worst decile | Mean accuracy of the 11 lowest-accuracy speakers |

The formal split manifest contains exactly 8,800/1,650/1,650 items and 110
speakers. No cache failures were recorded. Codec state hashes match before and
after cache construction; the codec is absent from classifier optimization.
The canonical split is not globally transcript-disjoint.

The split manifest resolves to `data/VCTK/wav48_silence_trimmed/..._mic1.flac`
and includes speaker `s5`. The local `data/VCTK/update.txt` records the
dual-microphone release, addition of `s5`, and March 2019 silence trimming.
The VCTK reference now identifies version 0.92 (2019), rather than the original
2017 release. Its title, author order, year and DOI were verified on 2026-09-17
against the registered DataCite metadata for `10.7488/ds/2645`
(`https://api.datacite.org/dois/10.7488/ds/2645`).

Fusion-versus-branch ablations, the transcript-controlled 109-speaker
experiment and the two-second challenge were not completed in the retained
formal report. They are not described as established evidence.

### Inference

In addition to the codec's `encode` and `decode` implementations, the inference
description uses these files in the speaker-classifier worktree:

- `speechtokenizer/speaker_identity/inference.py`
- `tmp/exp23_formal/export/seed42_bundle/preprocessing.json`
- `tmp/exp23_formal/export/seed42_bundle/calibration.json`
- `3用户demo/speechtokenizer_now/speechtokenizer/三用户中心路由通信demo/group_client.py`
- `3用户demo/speechtokenizer_now/speechtokenizer/三用户中心路由通信demo/speaker_identity.py`

These files demonstrate the software interface and default windowing, not a
completed six-user experiment. The encoder is not sample-causal. Offline
whole-utterance quality, encoder-only RTF, speaker-window duration and
end-to-end latency must remain separate quantities.

`pack_indices_bits`, `unpack_indices_bits`, and `send_codes` in `group_client.py`
establish the layer-major, MSB-first ten-bit payload convention, final-byte
zero padding and shape/index-count metadata. This description deliberately
does not claim a completed quantum-link experiment or quantify network traffic
as the nominal codec rate.

## Remaining Author Checks

1. The speaker bootstrap summary records per-seed confidence limits but not
   the executed replicate count or RNG seed. The initial plan specifies
   10,000 replicates, while the helper defaults to 2,000. Neither default
   should be asserted as executed without the original analysis command.
   Methods describes the verified hierarchy and percentile rule, with a
   visible author-confirmation placeholder for the missing metadata.
2. The title and abstract retain the original overall-study framing,
   including broader system-level claims. They were not rewritten in this
   Methods task and need review when integration experiments are complete.
3. The Chinese manuscript and Chinese Results working draft were not
   synchronized by this English-Methods task.
