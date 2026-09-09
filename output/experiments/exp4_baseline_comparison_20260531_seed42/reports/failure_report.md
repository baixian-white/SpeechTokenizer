# Exp4 Failure / Skip Report

This report documents components from the experiment plan that were SKIPPED or FAILED, with explicit reasons. The experiment is otherwise considered complete (see summary.md).

## Skipped components

### AMR-WB
- **status**: skipped (encoder not available)
- **reason**: conda-forge ffmpeg (6.1.2) on Windows is built with `--enable-libopus` but NOT `--enable-libvo-amrwbenc` or `--enable-libopencore-amrwb`. The build provides AMR-WB DECODER only.
- **what was tried**:
  - `conda search -c conda-forge "*amr*"` returned no AMR-WB packages
  - `ffmpeg -hide_banner -encoders | grep amr` returned no AMR-WB encoders (decoder only)
- **next-step options for follow-up**:
  - Build ffmpeg from source with `--enable-libvo-amrwbenc` (requires `vo-amrwbenc` library)
  - Use a pre-built Windows ffmpeg from gyan.dev (full version) which includes amrwb encoder
  - Use external sox build with AMR-WB plugin

### Codec2
- **status**: skipped (optional baseline, not pursued)
- **reason**: implementation cost not justified given other baselines already cover the low-bitrate regime

### ViSQOL
- **status**: skipped (tooling)
- **reason**: ViSQOL requires bazel build or specific Python wheel; not in conda-forge. STOI + PESQ-WB cover the perceptual-quality dimension.

### Semantic similarity (separate from WER/CER)
- **status**: covered indirectly by WER/CER
- **reason**: experiment plan mentions transcript or embedding similarity. Whisper-derived WER/CER computes Levenshtein distance on transcript tokens, which is the most reproducible form of transcript-level semantic similarity. No separate embedding-based score was added.

### MOS / AB preference (subjective)
- **status**: skipped (out of scope for automated pipeline)
- **reason**: requires human listeners; recommended for future paper-supplement work

## What was completed

- 184 (sample, method, setting) evaluation pairs, all with audio_quality + payload + ASR (WER/CER)
- All decoded WAVs preserved (samples/{original,scit_base,scit_lca,pcm,opus,encodec,dac}/...)
- All ffmpeg encode/decode commands logged per (sample, bitrate) under logs/codec_commands/
- Checkpoint provenance with sha256 for both Base and LCA v2
- Tool / environment versions recorded (tool_versions.md, environment.md)

## Summary

The experiment satisfies the calling document's minimum success criteria (Section 8 of `output/doc/experiment_plans/exp4_baseline_comparison.md`): PCM + at least one traditional codec (Opus) + neural codec (EnCodec, DAC) baselines completed; ideal/packed/packetized payloads tracked; all command logs and provenance saved. AMR-WB remains a known gap and is the highest-priority follow-up.