# Experiment 4 Summary: Baseline Comparison

- run_id: `exp4_baseline_comparison_20260531_seed42`
- status: **completed (Opus + Whisper WER/CER added; AMR-WB / Codec2 skipped with documented reasons)**
- date: 2026-05-31 (initial run + Opus extension) / 2026-06-01 (WER/CER + supplementary docs)

## Key result

SCIT-Speech operates in a 500-1500 bps regime where Opus is structurally unavailable and neural codecs (DAC, EnCodec) perform poorly. At those bitrates SCIT-LCA dominates all baselines on mel_l1, stoi, pesq, and WER.

| Operating bps | Best baseline | SCIT-LCA |
|---|---|---|
| 500 | DAC n_q=1: stoi 0.61, pesq 1.05, **WER 0.92** | **L=1: stoi 0.80, pesq 1.35, WER 0.46** |
| 1000 | DAC n_q=2: stoi 0.73, pesq 1.15, WER 0.31 | **L=2: stoi 0.86, pesq 1.71, WER 0.19** |
| 1500 | DAC n_q=3: stoi 0.80, pesq 1.27, WER 0.15 | **L=3: stoi 0.88, pesq 1.89, WER 0.14** |

Most striking cross-codec contrast (added 2026-06-01): **SCIT-LCA at 1.5 kbps achieves WER 0.142 — marginally better than Opus at 6 kbps (WER 0.147), at 1/4 the bandwidth**. mel_l1 is also 2.6× lower (0.86 vs 2.24).

Outside SCIT's range (6+ kbps) Opus remains the established choice (pesq 4.48 at 24 kbps, vs PCM 4.64). SCIT does not compete in that regime.

## Components

| File | Content |
|---|---|
| [reports/baseline_comparison_summary.md](baseline_comparison_summary.md) | Detailed method × bitrate table (audio quality + WER/CER) + cross-codec analysis |
| [reports/payload_accounting.md](payload_accounting.md) | Ideal/packed/packetized bps for all methods |
| [reports/audio_samples.md](audio_samples.md) | Per-sample appendix listing all 23 versions of each utterance |
| [reports/environment.md](environment.md) | Repository state, hardware, conda env |
| [reports/tool_versions.md](tool_versions.md) | ffmpeg / Whisper / DAC / EnCodec / jiwer versions |
| [reports/failure_report.md](failure_report.md) | AMR-WB / Codec2 / ViSQOL skip reasons |
| [configs/baseline_comparison_config.json](../configs/baseline_comparison_config.json) | Run config |
| [configs/base_run_reference.json](../configs/base_run_reference.json) | Base ckpt provenance (sha256) |
| [configs/lca_run_reference.json](../configs/lca_run_reference.json) | LCA v2 ckpt provenance (sha256) |
| [metrics/audio_quality_results.csv](../metrics/audio_quality_results.csv) | **184** evaluation rows (8 samples × 23 method-settings) |
| [metrics/payload_summary.csv](../metrics/payload_summary.csv) | 184 rows of payload accounting |
| [metrics/asr_results.csv](../metrics/asr_results.csv) | 184 rows of WER/CER (Whisper base.en) |
| [samples/sample_manifest.csv](../samples/sample_manifest.csv) | Per-sample wide-format manifest indexing all decoded wavs |
| [samples/](../samples/) | original + scit_base + scit_lca + pcm + opus + encodec + dac (184 wavs) |
| [artifacts/asr_transcripts/](../artifacts/asr_transcripts/) | Per-(method,setting,sample) Whisper hypothesis text + GT reference |
| [logs/codec_commands/](../logs/codec_commands/) | Per-sample ffmpeg encode/decode command logs (40 Opus pairs) |

## Methods tested

- ✅ SCIT-Speech-Base at L=1/2/3
- ✅ SCIT-Speech-LCA v2 at L=1/2/3
- ✅ PCM 16-bit @ 16 kHz (lossless upper bound)
- ✅ Opus (libopus via ffmpeg 6.1.2) at 6/8/12/16/24 kbps
- ✅ EnCodec 24kHz at bw ∈ {1.5, 3.0, 6.0, 12.0} kbps
- ✅ DAC 16kHz at n_q ∈ {1, 2, 3, 4, 6, 9, 12}
- ✅ Whisper `base.en` ASR for WER/CER on all 184 decoded wavs
- ⏭️ AMR-WB — skipped (conda-forge ffmpeg lacks `--enable-libvo-amrwbenc`; encoder not in build)
- ⏭️ Codec2 — skipped (optional)
- ⏭️ ViSQOL — skipped (no convenient win-64 build)
- ⏭️ Subjective MOS / AB — out of scope for automated pipeline

## Limitations

1. **Only 8 test samples** from LibriSpeech train-clean-100 (same speaker pool as training). No cross-corpus / test-other.
2. **AMR-WB still missing** — the most direct traditional speech codec comparison (6.6/8.85/15.85/23.85 kbps wideband). Opus + libopus partially fills the gap.
3. **No subjective MOS / AB test**.
4. **Opus 1.5/3 kbps not tested**: libopus minimum useful WB rate is ~6 kbps.
5. **Whisper baseline error**: PCM gets WER 0.075 on these samples (Whisper itself is not perfect on this audio); WER differences between methods are best read as deltas relative to PCM rather than absolutes.

## Honest framing

SCIT carves out a low-load operating regime (500-1500 bps) where existing speech codecs structurally do not work or perform poorly:
- Opus minimum useful wideband: ~6 kbps
- EnCodec minimum: 1.5 kbps (and even at 1.5 kbps it underperforms SCIT-LCA L=3)
- DAC at 500-1500 bps degrades sharply (it's not its training target)

This is the paper's value proposition: a different operating regime, not "we beat everything." SCIT does NOT outperform Opus at 12+ kbps or DAC at 6+ kbps; those remain the established choices in their native bitrate ranges.

## Next steps

1. Cross-corpus generalization on test-other / VCTK / AISHELL
2. AMR-WB baseline: build ffmpeg from source with `--enable-libvo-amrwbenc` or use gyan.dev pre-built full ffmpeg
3. Subjective MOS / AB test on the 8 fixed samples
4. Use SCIT-LCA v2 as the official model for downstream Exp5 ablations

