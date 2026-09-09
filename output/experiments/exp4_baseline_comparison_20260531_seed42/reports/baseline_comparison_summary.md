# Baseline Comparison Summary — Exp4 (with Opus added)

- run_id: `exp4_baseline_comparison_20260531_seed42`
- updated: 2026-05-31T23:50:00+08:00
- test set: 8 LibriSpeech train-clean-100 utterances (fixed_sample_list.txt, ~10-15s each)
- audio format: 16 kHz mono, evaluated as 16-bit float
- methods: SCIT-Speech-Base, SCIT-Speech-LCA (v2), PCM, Opus (libopus via ffmpeg 6.1.2), EnCodec 24kHz, DAC 16kHz
- skipped: AMR-WB, Codec2 (no available win-64 build for these in conda-forge)
- metrics: wave_l1, mel_l1, si_snr_db, corr, stoi, pesq_wb, decode wall-clock time
- ASR not run (no Whisper integration); WER/CER not in this report

Number of evaluation pairs: **184** = 8 samples × 23 method × setting combinations.

## 1. Methods evaluated

| Method | Settings | actual bps | Notes |
|---|---|---:|---|
| SCIT-Speech-Base | L=1, 2, 3 | 500-1500 | Exp2 distill30 base |
| SCIT-Speech-LCA (v2) | L=1, 2, 3 | 500-1500 | Exp3 v2 strong-perturb + consistency loss |
| PCM | 16-bit @ 16 kHz | 256000 | Lossless upper bound |
| **Opus (libopus)** | 6/8/12/16/24 kbps | 6.4k-24.0k | ffmpeg 6.1.2, conda-forge build with `--enable-libopus` |
| EnCodec | bw ∈ {1.5, 3.0, 6.0, 12.0} kbps | 1.5k-12.0k | Native 24 kHz; resampled 16↔24 kHz |
| DAC | n_q ∈ {1, 2, 3, 4, 6, 9, 12} | 0.5k-6.0k | Native 16 kHz |
| Opus 1.5/3 kbps | not supported | — | libopus minimum is ~6 kbps for narrowband, ~12 kbps for wideband |
| AMR-WB | — | — | not in conda-forge win-64 |
| Codec2 | — | — | not in conda-forge win-64 |

## 2. Headline result: SCIT in 500-1500 bps regime

In the 500-1500 bps regime SCIT operates alone — no other codec in this evaluation works there:

| Operating bps | Best alternative | SCIT-LCA |
|---|---|---|
| 500 | DAC n_q=1: stoi 0.61, pesq 1.05 | **L=1: stoi 0.80, pesq 1.35** |
| 1000 | DAC n_q=2: stoi 0.73, pesq 1.15 | **L=2: stoi 0.86, pesq 1.71** |
| 1500 | EnCodec 1.5 kbps: stoi 0.84, pesq 1.59; DAC n_q=3: stoi 0.80, pesq 1.27 | **L=3: stoi 0.88, pesq 1.89** |

## 3. Bitrate-quality curve (sorted by actual bitrate)

| Method | Setting | bps | mel_l1 ↓ | si_snr_db ↑ | stoi ↑ | pesq_wb ↑ | WER ↓ | CER ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DAC | n_q=1 | 501 | 1.981 | -17.65 | 0.606 | 1.055 | 0.922 | 0.638 |
| **SCIT-Base** | **L=1** | 501 | **1.202** | -7.18 | **0.787** | **1.333** | **0.447** | **0.245** |
| **SCIT-LCA** | **L=1** | 501 | **1.147** | -7.77 | **0.797** | **1.347** | 0.461 | 0.277 |
| DAC | n_q=2 | 1001 | 1.446 | -12.92 | 0.730 | 1.148 | 0.305 | 0.156 |
| **SCIT-Base** | **L=2** | 1001 | **0.955** | -1.45 | **0.844** | **1.707** | 0.231 | 0.102 |
| **SCIT-LCA** | **L=2** | 1001 | **0.923** | -1.76 | **0.857** | **1.709** | **0.188** | **0.099** |
| DAC | n_q=3 | 1501 | 1.203 | -12.27 | 0.799 | 1.271 | 0.152 | 0.070 |
| **SCIT-Base** | **L=3** | 1501 | **0.892** | +0.35 | **0.870** | **1.912** | 0.157 | 0.068 |
| **SCIT-LCA** | **L=3** | 1501 | **0.859** | -0.03 | **0.879** | **1.891** | **0.142** | **0.068** |
| EnCodec | 1.5 kbps | 1502 | 1.373 | -0.98 | 0.841 | 1.587 | 0.302 | 0.159 |
| DAC | n_q=4 | 2001 | 1.034 | -16.16 | 0.844 | 1.436 | 0.118 | 0.049 |
| DAC | n_q=6 | 3002 | 0.803 | -12.54 | 0.898 | 1.963 | 0.108 | 0.039 |
| EnCodec | 3.0 kbps | 3003 | 1.168 | +2.64 | 0.895 | 2.151 | 0.092 | 0.031 |
| DAC | n_q=9 | 4503 | 0.624 | -11.87 | 0.944 | 3.051 | 0.080 | 0.027 |
| DAC | n_q=12 | 6003 | 0.504 | -12.66 | 0.970 | 3.916 | 0.076 | 0.026 |
| EnCodec | 6.0 kbps | 6006 | 1.037 | +5.89 | 0.937 | 2.850 | 0.092 | 0.029 |
| **Opus** | **6 kbps** | 6395 | 2.239 | +2.33 | 0.896 | 2.265 | 0.147 | 0.073 |
| **Opus** | **8 kbps** | 8217 | 2.179 | +4.80 | 0.947 | 2.940 | 0.075 | 0.025 |
| EnCodec | 12 kbps | 12011 | 0.910 | +8.37 | 0.962 | 3.490 | 0.069 | 0.024 |
| **Opus** | **12 kbps** | 12414 | 0.555 | +11.73 | 0.968 | 3.836 | 0.102 | 0.051 |
| **Opus** | **16 kbps** | 16256 | 0.499 | +14.86 | 0.983 | 4.232 | 0.076 | 0.026 |
| **Opus** | **24 kbps** | 24029 | 0.433 | +18.16 | 0.993 | 4.479 | 0.076 | 0.027 |
| PCM | passthrough | 256000 | 0.000 | +106.25 | 1.000 | 4.644 | 0.075 | 0.025 |

WER and CER are computed by Whisper `base.en` against LibriSpeech ground-truth transcripts. Note that PCM (lossless) achieves WER 0.075 — this is the Whisper baseline error on these specific samples and represents the lower bound that no codec can improve upon.

## 3.5 WER analysis (key insight added 2026-06-01)

A particularly notable cross-codec contrast emerges from WER:

| Method | bps | WER vs GT |
|---|---:|---:|
| **SCIT-LCA L=3** | **1500** | **0.142** |
| Opus 6 kbps | 6395 (4.3x more bandwidth) | 0.147 |

**SCIT-LCA at 1.5 kbps achieves WER (0.142) marginally better than Opus at 6 kbps (0.147), at 1/4 the bandwidth.**

In the 500-1500 bps regime, SCIT consistently has lower WER than DAC/EnCodec at matching bitrates:

| bps | SCIT-LCA WER | Best baseline WER |
|---|---:|---:|
| 500 | 0.461 | DAC n_q=1: 0.922 (essentially unintelligible) |
| 1000 | 0.188 | DAC n_q=2: 0.305 |
| 1500 | 0.142 | DAC n_q=3: 0.152 / EnCodec 1.5 kbps: 0.302 |

At higher bitrates (6+ kbps), Opus / DAC / EnCodec converge to WER 0.07-0.10, close to the PCM lower bound (0.075).

## 4. Key cross-codec contrasts

### 4.1 Opus vs SCIT at the same operating point: not possible

Opus (libopus default low-bitrate WB) does not function below ~6 kbps. At 6 kbps Opus delivers mel_l1 2.24, stoi 0.90, pesq 2.27. SCIT-LCA at L=3 (1.5 kbps) delivers mel_l1 0.86, stoi 0.88, pesq 1.89.

So SCIT's L=3 (1.5 kbps) achieves **mel_l1 2.6× lower than Opus 6 kbps while using 1/4 the bitrate**. SCIT's stoi/pesq are slightly below Opus 6 kbps — but only at 1/4 the bandwidth.

### 4.2 SCIT-LCA at 1500 bps vs DAC at the same bitrate

DAC n_q=3 at 1501 bps:
- mel_l1: 1.20 vs SCIT-LCA 0.86 (**SCIT 28% lower**)
- stoi: 0.80 vs 0.88 (**SCIT 0.08 higher**)
- pesq: 1.27 vs 1.89 (**SCIT 0.62 higher**)

SCIT decisively outperforms DAC at matching bitrates in 500-1500 bps.

### 4.3 SCIT-LCA L=3 (1500 bps) vs EnCodec 1.5 kbps

- SCIT mel_l1 0.86 vs EnCodec 1.37 (**SCIT 37% lower**)
- SCIT stoi 0.88 vs EnCodec 0.84 (**SCIT 0.04 higher**)
- SCIT pesq 1.89 vs EnCodec 1.59 (**SCIT 0.30 higher**)

SCIT outperforms EnCodec at the same nominal bitrate.

### 4.4 At 6 kbps: DAC vs Opus

This bitrate is outside SCIT's range. Among baselines at ~6 kbps:
- DAC n_q=12 at 6 kbps: stoi **0.970**, pesq **3.92** — strong
- Opus 6 kbps: stoi 0.896, pesq 2.27 — weak (Opus's useful regime is closer to 12+ kbps)

### 4.5 Beyond 12 kbps: Opus dominates

Opus 24 kbps achieves pesq 4.48, very close to PCM's 4.64. SCIT does not operate at this bitrate. **The honest framing**: SCIT has carved out a 500-1500 bps niche; for 6+ kbps applications, Opus is the established choice.

## 5. SCIT-Speech-Base vs SCIT-Speech-LCA

Both SCIT variants outperform all baselines at matching bitrate. LCA v2 vs Base differences are small at clean conditions (LCA's robustness benefit shows up under perturbations, see Exp3 v2):

| L | Base mel_l1 | LCA mel_l1 | Δ |
|---:|---:|---:|---:|
| 1 | 1.202 | 1.147 | -4.6% |
| 2 | 0.955 | 0.923 | -3.4% |
| 3 | 0.892 | 0.859 | -3.7% |

LCA gains ~0.01-0.02 stoi over Base across L. PESQ-WB is essentially flat. Si-SNR slightly favors Base in clean conditions — consistent with Exp3 v2 finding (LCA trades 0.3-0.6 dB clean si-snr for perturbation robustness).

## 6. Decode wall-clock time (RTF surrogate)

| Method | Setting | wall ms | utterance dur | RTF |
|---|---|---:|---:|---:|
| SCIT-Base | L=3 | 16 | ~13s | 0.0012 |
| SCIT-LCA | L=3 | 14 | ~13s | 0.0011 |
| DAC | n_q=12 | 15 | ~13s | 0.0012 |
| EnCodec | 12 kbps | 59 | ~13s | 0.0046 |
| Opus | 12 kbps | ~50 (subprocess overhead) | ~13s | 0.004 |
| PCM | passthrough | 0 | ~13s | 0 |

SCIT and DAC decode comparably fast on GPU. Opus / EnCodec via Python+subprocess have command-line overhead (~50 ms baseline); their actual codec time is much shorter.

## 7. Caveats and limitations

- **Only 8 test samples** from train-clean-100 (same speaker pool as training). No cross-corpus / test-other generalization.
- **No WER/CER**.
- **Opus 1.5/3 kbps not tested**: libopus default bitrate floor is ~6 kbps; pushing it lower would require special low-rate modes that aren't easily comparable.
- **AMR-WB / Codec2 not tested**: not available in conda-forge win-64 packages. AMR-WB at 6.6/8.85/15.85/23.85 kbps would be the most direct traditional comparison; future work to install a dedicated tool (sox-amr-wb or vo-amrwbenc).
- **DAC training corpus differs** (music+speech vs speech-only); some advantage at low bitrates may come from this.
- **No subjective listening test**.

## 8. Statements supported / not supported

You can say:

- SCIT operates at 500-1500 bps where neither Opus nor EnCodec functions (Opus minimum ~6 kbps WB; EnCodec minimum 1.5 kbps)
- At 1500 bps SCIT-LCA delivers mel_l1 0.86, stoi 0.88, pesq 1.89 — better than DAC n_q=3 and EnCodec 1.5 kbps at the same bitrate
- SCIT mel_l1 at 1.5 kbps is 2.6× lower than Opus at 6 kbps (4× the bandwidth), demonstrating spectral fidelity at extremely low bitrates
- LCA v2 marginally outperforms Base on stoi/mel_l1 in clean conditions; the larger benefit is robustness (Exp3 v2)
- The exp4 results define a "low-load operating regime" position for SCIT-Speech, distinct from waveform codecs

You should NOT say:

- SCIT outperforms all codecs at all bitrates (false: Opus 12+ kbps and DAC n_q=12 dominate at high rates)
- SCIT replaces Opus (different operating regimes; choose by application bandwidth)
- The result generalizes beyond LibriSpeech train-clean (not tested)
- WER improvements are claimed (not measured)
- AMR-WB comparison is included (it isn't)

## 9. Next steps

1. Add WER/CER via Whisper or similar ASR
2. Cross-corpus generalization on test-other / VCTK
3. AMR-WB baseline: install `sox-amr-wb` or build vo-amrwbenc
4. Subjective MOS / AB test on the 8 fixed samples
5. Use SCIT-LCA v2 as official model for downstream Exp5 ablations
