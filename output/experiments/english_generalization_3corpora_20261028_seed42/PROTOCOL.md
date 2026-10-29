# English cross-corpus generalization — frozen protocol

Date: 2026-10-28. Seed: 42. Compute host: lab-suer, user suer, physical GPU 1.

## Question and scope

Measure how the existing SCIT Base and LCA checkpoints transfer from LibriSpeech training to three external English corpora. No training, adaptation, checkpoint selection, or parameter tuning on these test corpora. All results, including degradations and metric failures, must be reported.

- VCTK 0.92: 20 utterances per speaker, mic1, seeded sampling from sorted filenames (110 speakers, target 2,200). Preserve all sampled audio. Missing reference transcripts are explicitly excluded from WER only (p315 has no local transcripts).
- TED-LIUM release3 test mirror: all rows from sanchit-gandhi/tedlium-data at revision 8083dbe64f82e17b6be070a02b9239e9b847b6ce. Source loader distil-whisper/tedlium at 31aa998c1cbc36063c5bd5c0ca87b082d443da02 identifies this data repository and release3. Use human `text`, never `whisper_transcript`. Exclude `ignore_time_segment_in_scoring` and document counts. This is a pinned mirror, not an independently verified bitwise copy of the original distribution.
- FLEURS en_us test: all 647 recordings, google/fleurs at 70bb2e84b976b7e960aa89f1c648e09c59f894dd. Use provided raw transcription; group uncertainty by prompt ID (speaker IDs are unavailable).
- LibriSpeech test-clean and test-other: 300 seeded speaker-balanced utterances each, evaluated with the same new pipeline. Existing full-set objective results (2,620 / 2,939) are historical context, not mixed into the new estimates.

## Frozen models and measures

Base SHA256: 8c23c2b146575d1144b7f2f96e6ded88aa44cd89cf53fa75ab14ff6e279763e6.
LCA SHA256: af60223b2f44733f5a43d5a022871dad061dfe2421473c96f5d350c3d4b49a5a.
Load exact NAS operation configs and require a strict state-dict match. FP32 codec inference, eval mode, no gradients. Encode n_q=3 once and decode prefixes L=1,2,3. Nominal payload rates 0.5,1.0,1.5 kbit/s, excluding transport overhead.

Primary operating point L3: mel-L1, STOI, PESQ-WB, SI-SNR and corpus word-weighted WER. L1/L2 objective metrics are secondary operating-point checks. No packet loss is introduced in this experiment.

Input preprocessing: whole utterance, float32 mono, torchaudio resampling to 16 kHz; no loudness normalization, trimming, time alignment, or test-driven duration filter. Objective comparisons truncate to common reference/reconstruction length, recording both lengths. ASR uses original full audio and common-length L3 reconstructions as float32 arrays, no intermediate PCM clipping.

ASR: fixed Systran/faster-whisper-small.en revision d1d751a5f8271d482d14ca55d9e2deeebbae577f, float16, English, beam size 5, temperature 0, no VAD, no previous-text conditioning, fixed decoding settings across corpora/models. Record original-audio WER and each codec's WER, plus the absolute WER increase over the original. Normalize references and predictions identically with the pinned OpenAI Whisper English normalizer; report S/D/I/N, not just mean utterance WER. ASR is a measurement probe and its own pretraining exposure to benchmark corpora is not established here.

## Reproducibility and uncertainty

Freeze and hash manifests, input audio, checkpoints, configs, source scripts, ASR model, and environment. Exclusions are determined from corpus metadata before inference. Separate smoke-test outputs from full-run outputs. Write one complete utterance record at a time and resume only when run provenance matches. Log failures without silently deleting them or substituting zeros.

Report per-corpus means and 95% percentile bootstrap intervals with 2,000 draws, using speakers for VCTK and LibriSpeech, TED talk IDs for TED-LIUM, and prompt IDs for FLEURS. Use paired resampling for LCA−Base and decoded−original WER. Cross-corpus differences are descriptive and confounded by corpus characteristics. Bootstrap uncertainty describes evaluation-sample variation, not codec training-seed variation; the retained codecs have one training seed. Report cluster counts, successful denominators, missing transcripts, and metric failures. No single pooled score across unequal corpora.

## Execution layout

Remote run directory: /home/suer/english_generalization_20261028.
Repository: /home/suer/workspace/NMI/speechtokenizer.
Runtime uses an isolated dependency directory with the existing sixuser venv and PYTHONNOUSERSITE=1. GPU 0 belongs to another job and is not used. Durable logs/status/manifests/JSONL results are retained remotely and copied locally for reporting.

## Frozen manifest after source validation

SHA256: de007738841b5f87915d856964d6ceef39c58c2d2277c458e9a564b9e20651a1.
4,602 total recordings, reported separately by corpus:

| Corpus | Recordings | Uncertainty clusters | Audio hours | Missing reference text |
|---|---:|---:|---:|---:|
| VCTK | 2,200 | 110 speakers | 2.070 | 20 (p315) |
| TED-LIUM | 1,155 | 11 talks | 2.617 | 0 |
| FLEURS en_us | 647 | 350 prompts | 1.774 | 0 |
| LibriSpeech test-clean | 300 | 40 speakers | 0.738 | 0 |
| LibriSpeech test-other | 300 | 33 speakers | 0.576 | 0 |

TED mirror contains 1,469 rows; exactly 314 `ignore_time_segment_in_scoring` rows were excluded by the prespecified rule. No duration filter was used; the shortest retained TED segment is 0.35 s. Very short clips may fail PESQ/STOI and must be reported with denominators. External recordings total 4,002; ASR-eligible external recordings total 3,982.

FLEURS, TED parquet, and ASR model hashes were matched to the pinned Hugging Face repository LFS SHA256 metadata. FLEURS TSV requires literal-quote parsing (`csv.QUOTE_NONE`); all 647 rows have seven tab-separated fields. The pinned Whisper English normalizer revision is 86098128c0b4f24f0e2aa2994de830614b474227.

The expanded VCTK selection contains all 300 previously evaluated utterances plus 1,900 additional utterances. It is an expanded external-corpus test, not 2,200 entirely new test utterances. TED-LIUM and FLEURS add two external corpora absent from the previous audit.

## Separate post-hoc diagnostic

After completing the raw-level FLEURS objective results, audio-level checks found a median RMS near −60 dBFS, versus roughly −25 dBFS for the LibriSpeech references. The previous official FLEURS loader at revision d7c758a6dceecd54a98cac43404d3d576e721f07 returns original audio bytes; it does not add explicit gain normalization. A copy is retained in checks/fleurs_official_loader.py.

A separate `level_control` experiment applies a single prespecified gain rule to all 647 FLEURS recordings: target RMS −25 dBFS, limited to peak 0.99 without clipping. Models, transcripts, scoring, and test inclusion stay fixed. Six recordings meet the peak cap before reaching the target. This diagnostic was designed after observing the raw FLEURS decline; it does not replace the primary raw-level experiment and is not an independent preregistered confirmation. Its frozen manifest SHA256 is e185e3e51eb702beef7ee14a0e32239ee02f9137e1b9954afca9a56c8e82e6d2.
