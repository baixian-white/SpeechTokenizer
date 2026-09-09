# Baseline 300-Sample Comparison Summary (TEMPLATE)

> This file is a **template**. The populated report is written to
> `runs/<split>/reports/baseline_300_summary.md` by `analyze_baseline_300_stats.py`.
> Fields marked `<<TBD: filled by analyze_baseline_300_stats.py>>` are filled in
> automatically; everything else is fixed boilerplate.

---

## 1. Run Identity

| Field | Value |
|---|---|
| `run_id` | `<<TBD: filled by analyze_baseline_300_stats.py>>` |
| Split | `test300_20260610_seed42` |
| Date generated (UTC) | `<<TBD: filled by analyze_baseline_300_stats.py>>` |
| Host / device | `<<TBD: filled by analyze_baseline_300_stats.py>>` |

### Checkpoint integrity (sha256)

| Model | Checkpoint path | sha256 |
|---|---|---|
| SCIT Base | `<<TBD: filled by analyze_baseline_300_stats.py>>` | `<<TBD: filled by analyze_baseline_300_stats.py>>` |
| SCIT + LCA | `<<TBD: filled by analyze_baseline_300_stats.py>>` | `<<TBD: filled by analyze_baseline_300_stats.py>>` |

### Baselines included

| Baseline | Version / commit | Notes |
|---|---|---|
| Opus | `<<TBD: filled by analyze_baseline_300_stats.py>>` | Min-bitrate floor applies; see Caveats. |
| EnCodec | `<<TBD: filled by analyze_baseline_300_stats.py>>` | `<<TBD: filled by analyze_baseline_300_stats.py>>` |
| Lyra-v2 | `<<TBD: filled by analyze_baseline_300_stats.py>>` | `<<TBD: filled by analyze_baseline_300_stats.py>>` |
| SpeechTokenizer (orig.) | `<<TBD: filled by analyze_baseline_300_stats.py>>` | `<<TBD: filled by analyze_baseline_300_stats.py>>` |
| SCIT Base (ours) | `<<TBD: filled by analyze_baseline_300_stats.py>>` | Reference for paired tests. |
| SCIT + LCA (ours) | `<<TBD: filled by analyze_baseline_300_stats.py>>` | Reference for paired tests. |

> **Codec2 is intentionally excluded** from this run — see Caveats.

### Tooling

| Tool | Path / version |
|---|---|
| ffmpeg binary | `<<TBD: filled by analyze_baseline_300_stats.py>>` |
| ffmpeg version string | `<<TBD: filled by analyze_baseline_300_stats.py>>` |
| Python | `<<TBD: filled by analyze_baseline_300_stats.py>>` |
| PyTorch | `<<TBD: filled by analyze_baseline_300_stats.py>>` |

---

## 2. Sample List Provenance

| Field | Value |
|---|---|
| Source filelist | `<<TBD: filled by analyze_baseline_300_stats.py>>` |
| Total sample count | `<<TBD: filled by analyze_baseline_300_stats.py>>` |
| RNG seed | `42` |
| Sampling method | `<<TBD: filled by analyze_baseline_300_stats.py>>` |
| sha256 of resolved sample list | `<<TBD: filled by analyze_baseline_300_stats.py>>` |

### First 5 sample_ids (sanity check)

1. `<<TBD: filled by analyze_baseline_300_stats.py>>`
2. `<<TBD: filled by analyze_baseline_300_stats.py>>`
3. `<<TBD: filled by analyze_baseline_300_stats.py>>`
4. `<<TBD: filled by analyze_baseline_300_stats.py>>`
5. `<<TBD: filled by analyze_baseline_300_stats.py>>`

---

## 3. Per-Method Quality at 3 Operating Points

All values are reported as `mean +/- 95% CI` (bootstrap, 10k resamples) over
the 300-sample evaluation set. Arrows indicate the favorable direction:
mel-L1 (lower is better), STOI / PESQ-WB (higher is better),
WER (lower is better).

### 3.1 Operating point: 500 bps

| Method | mel-L1 (down) | STOI (up) | PESQ-WB (up) | WER (down) |
|---|---|---|---|---|
| Opus | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| EnCodec | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| Lyra-v2 | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| SpeechTokenizer (orig.) | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| SCIT Base | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| **SCIT + LCA** | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |

### 3.2 Operating point: 1000 bps

| Method | mel-L1 (down) | STOI (up) | PESQ-WB (up) | WER (down) |
|---|---|---|---|---|
| Opus | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| EnCodec | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| Lyra-v2 | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| SpeechTokenizer (orig.) | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| SCIT Base | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| **SCIT + LCA** | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |

### 3.3 Operating point: 1500 bps

| Method | mel-L1 (down) | STOI (up) | PESQ-WB (up) | WER (down) |
|---|---|---|---|---|
| Opus | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| EnCodec | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| Lyra-v2 | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| SpeechTokenizer (orig.) | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| SCIT Base | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |
| **SCIT + LCA** | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | pending exp10 alignment |

---

## 4. Pairwise Comparisons vs SCIT-LCA

Each cell reports `mean_diff [95% CI]` of (other - SCIT-LCA) on paired
per-sample scores, plus the Wilcoxon signed-rank `p`-value. Verdict is
assigned via the rules in section 5.

### 4.1 At 500 bps

| Other method | mel-L1 diff | STOI diff | PESQ-WB diff | WER diff | Verdict |
|---|---|---|---|---|---|
| Opus | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |
| EnCodec | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |
| Lyra-v2 | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |
| SpeechTokenizer (orig.) | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |
| SCIT Base | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |

### 4.2 At 1000 bps

| Other method | mel-L1 diff | STOI diff | PESQ-WB diff | WER diff | Verdict |
|---|---|---|---|---|---|
| Opus | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |
| EnCodec | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |
| Lyra-v2 | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |
| SpeechTokenizer (orig.) | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |
| SCIT Base | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |

### 4.3 At 1500 bps

| Other method | mel-L1 diff | STOI diff | PESQ-WB diff | WER diff | Verdict |
|---|---|---|---|---|---|
| Opus | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |
| EnCodec | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |
| Lyra-v2 | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |
| SpeechTokenizer (orig.) | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |
| SCIT Base | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` | `<<TBD>>` |

---

## 5. Interpretation Rules (Plain Language)

A verdict is assigned to each `(method, metric, operating_point)` triple using
the paired statistics in section 4:

- **Definitively better** — the 95% CI of `mean_diff` excludes 0 in the
  favorable direction for that metric, **AND** the Wilcoxon signed-rank test
  yields `p < 0.01`. Both conditions must hold.
- **Comparable** — the 95% CI of `mean_diff` includes 0. Treat the methods as
  indistinguishable on this metric at this operating point regardless of the
  Wilcoxon `p`-value.
- **Inconclusive** — the 95% CI of `mean_diff` excludes 0, but the Wilcoxon
  test disagrees (does not reach `p < 0.01`, or points the opposite way). Do
  not claim a winner; flag for follow-up with a larger sample or a different
  test.

Favorable direction by metric: mel-L1 lower, STOI higher, PESQ-WB higher,
WER lower.

---

## 6. Caveats

- **Codec2 is not included.** The toolchain available on the evaluation host
  could not produce the requested operating points reliably; rather than ship
  partial numbers we omit Codec2 from this run. A follow-up run will revisit
  it once the encoder is validated.
- **Opus minimum-bitrate constraint.** Opus has a hard floor below which the
  encoder refuses to operate at the requested sample rate; the 500 bps column
  for Opus is therefore reported at the encoder's actual minimum-allowed
  bitrate, not 500 bps. The actual bitrate used is recorded in
  `runs/<split>/raw/opus_actual_bitrate.json` and surfaced as a footnote when
  the report is rendered.
- **PCM Whisper floor placeholder.** The WER column needs a clean-PCM Whisper
  reference run on these exact 300 samples to anchor degradation. That floor
  is `<<TBD: filled by analyze_baseline_300_stats.py>>` and is currently a
  placeholder.
- **Whisper has not yet been run on these 300 samples.** Until exp10's WER
  pipeline is aligned to this split, every WER cell reads
  `pending exp10 alignment`. Do not draw WER conclusions from this report
  until that line is removed.

---

*End of template. Populated reports are written by
`analyze_baseline_300_stats.py` to
`runs/<split>/reports/baseline_300_summary.md`.*
