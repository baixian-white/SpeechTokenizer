# Exp22 Speaker Identity Design

## Context

SCIT-Speech is organized around a paper evidence chain rather than a generic software stack. The current method sends only shared RVQ codebook indices at approximately 500, 1000, and 1500 bps for `L=1/2/3`, with the receiver reconstructing audio from pre-shared model weights and codebooks. Existing experiments cover Base/LCA training, clean quality, channel perturbation, packet loss, ASR/WER, codec baselines, VCTK/AISHELL zero-shot quality, ViSQOL, and the three-user TCP router demo.

Speaker identity is still a gap. The experiment manual lists speaker preservation as an optional metric, and the paper text discusses preservation of speaker-related cues, but there is no dedicated experiment showing whether reconstructed speech keeps "who is speaking" or whether RVQ indices/latents carry speaker-discriminative information.

## Goal

Add Exp22 to verify speaker identity preservation and speaker recognizability at three levels:

1. Reconstructed speech remains close to the original speaker.
2. SCIT codes and latent features contain speaker-discriminative information.
3. The three-user communication demo can optionally identify the decoded speaker per incoming stream.

## Scope

Exp22 is an evaluation and demo extension. It does not retrain SCIT-Speech Base or LCA, does not modify RVQ packet semantics, and does not move speaker recognition into the router. The router remains an index-packet forwarder that does not decode, mix, or access codebooks.

## Data

The primary formal dataset is VCTK because the local checkout already contains `data/VCTK/wav48_silence_trimmed`, and Exp19 established VCTK as the cross-corpus English multi-speaker test set. AISHELL can be used as an optional Chinese speaker-boundary check. LibriSpeech can be used for same-corpus comparison, but VCTK is preferred for identity because it has many speakers and clean speaker IDs.

The formal manifest should be built under:

```text
output/experiments/exp22_speaker_identity_YYYYMMDD_seed42/
```

Each run follows the project convention:

```text
configs/
commands/
logs/
checkpoints/
metrics/
samples/
reports/
artifacts/
```

## Experiment A: Speaker Preservation

Use enrollment utterances from each speaker to build a speaker profile from original audio. Test utterances are evaluated in original, Base-reconstructed, and LCA-reconstructed form for `L=1/2/3`.

Primary metrics:

- Top-1 speaker identification accuracy.
- Same-speaker cosine similarity.
- Max impostor cosine similarity.
- Cosine margin between the correct speaker and the nearest impostor.
- Equal error rate.
- TAR at FAR=1%.

The initial implementation uses a dependency-light MFCC statistics backend so the experiment and demo run in the current environment. The backend interface must allow a stronger frozen speaker model such as ECAPA-TDNN to be added later without rewriting the experiment scripts.

## Experiment B: Codes and Latent Probe

Extract speaker features from SCIT internals:

- `codes_hist`: normalized histograms over RVQ code IDs for the first `L` layers.
- `latent_stats`: mean and standard deviation of quantized feature tensors from `forward_feature()` for the first `L` layers.

Train a lightweight scikit-learn classifier on speaker labels with a per-speaker stratified split. Report top-1, top-5, and macro-F1 for `Base/LCA x L=1/2/3 x feature_kind`.

This experiment answers whether speaker identity is separable from the transmitted indices/features, not only from the final reconstructed waveform.

## Demo Extension

The three-user demo adds optional client-side speaker identification:

```text
--speaker_id_enable
--speaker_profile_dir <dir>
--speaker_window_sec 3.0
--speaker_hop_sec 1.0
--speaker_threshold 0.65
```

`speaker_profile_dir` contains one subdirectory per speaker/user, for example:

```text
speaker_profiles/
  A/*.wav
  B/*.wav
  C/*.wav
```

The client computes speaker profiles at startup. During receiving, after decoding each incoming packet to PCM and before playback mixing, it appends the decoded audio to a rolling buffer per sender. Once enough audio is available, it predicts the closest enrolled speaker and prints the result in monitor output.

Example monitor detail:

```text
from_A:seq=52 q=0 jit=0.500s rms=0.0310 dec=8.1ms dec_rtf=0.02 drop=0 drop_rate=0.0% spk=A score=0.82 margin=0.17 verified=yes
```

Recognition accuracy may use `sender_id` only as an evaluation label. The prediction itself must be computed from decoded audio.

## Outputs

Exp22 writes:

- `metrics/speaker_identity_results.csv`
- `metrics/speaker_identity_summary.csv`
- `metrics/speaker_identity_results.json`
- `reports/speaker_identity_summary.md`
- `metrics/speaker_probe_results.csv`
- `metrics/speaker_probe_summary.csv`
- `metrics/speaker_probe_results.json`
- `reports/speaker_probe_summary.md`

The demo summary CSV always includes speaker-ID columns; `speaker_id_enabled=0` when `--speaker_id_enable` is inactive.

## Success Criteria

1. Offline speaker preservation runs on a fixed VCTK manifest and writes CSV/JSON/Markdown outputs.
2. Offline probe runs on the same manifest and reports `L=1/2/3` trends.
3. Demo can be launched with speaker profiles and prints per-stream predicted speaker labels.
4. Existing tests still pass, and the new utilities have focused unit tests for speaker parsing, cosine scoring, EER, and profile classification.

## Caveats

The default MFCC backend is a lightweight reproducibility baseline, not a state-of-the-art biometric verifier. Formal paper claims should either name it as a classical speaker-feature backend or rerun with a frozen speaker-recognition model. Speaker preservation is also a privacy-relevant property: preserving identity can be useful for communication, but it means speaker identity is not anonymized.
