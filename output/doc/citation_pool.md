# Citation Pool for SCIT-Speech

> Source: deep-research workflow w043ohpnt (2026-06-02). 23 verified sources organized by research-question angle (Q1-Q5). For each: arXiv ID + one-line relevance + how SCIT differentiates.

## Tier 1 — Must-cite, central to positioning

### SAC (arXiv 2510.16841, 2025-10)
- **Why critical**: First codec to publish full WER+PESQ+STOI+UTMOS+SIM curve in ≤1 kbps regime (525/875 bps on LibriSpeech test-clean).
- **SCIT differentiator**: SCIT uses **RVQ with L-truncation** (3 operating points 500/1000/1500 bps); SAC uses single-codebook (no RVQ truncation, no L-knob). SCIT has explicit payload accounting; SAC reports only nominal BPS = token-rate × log2(codebook). SCIT uses **HuBERT distillation**; SAC uses frozen glm4voice tokenizer.

### Glaris (arXiv 2512.08203, 2025-12)
- **Why critical**: Closest precedent to index-perturbation training in speech communication.
- **SCIT differentiator**: Glaris operates at **8/12/18 kbps**; SCIT at 500-1500 bps. Glaris uses mask tokens for PLC only, no strong-vs-weak ablation. Glaris loss = α·D_prior(l, l̂_con) + D_prior(l, l̂_rec) — both branches supervised against GT in latent space. SCIT uses **mel-L1 clean-vs-perturbed consistency** (different supervisory signal).

### DualCodec (arXiv 2505.13000, 2025-05)
- **Why critical**: Uses RVQ dropout (random q in [0,N-1]) — closest precedent to SCIT's random-L training.
- **SCIT differentiator**: DualCodec frames RVQ dropout as **TTS tokenizer flexibility** for downstream LMs (VALL-E, SoundStorm, FlattenedAR). SCIT frames L as **transport-layer load-control knob**. DualCodec reports no WER at 0.75-0.93 kbps; SCIT reports WER at 500/1000/1500 bps.

### Codec-SUPERB (arXiv 2409.14085, 2024-09)
- **Why critical**: Cross-codec benchmark sampling 500-1500 bps via SemantiCodec B3-B6 (0.68/0.95/1.35/1.40 kbps).
- **SCIT differentiator**: Codec-SUPERB samples points across heterogeneous codecs; SCIT provides **single-codec dense curve** within RVQ-native L=1/2/3 truncation, with consistent encoder-decoder pair.

### SpeechTokenizer (arXiv 2308.16692, 2023-08)
- **Why critical**: Introduced HuBERT distillation for layer-1 of RVQ — direct ancestor of SCIT's distillation choice.
- **SCIT differentiator**: SpeechTokenizer demonstrates distillation **helps**; SCIT provides **clean necessity ablation** (distill=0 vs distill=30) at ≤1.5 kbps with WER quantification (+9.4/+7.4/+8.4 pp at L=1/2/3) and codebook-usage analysis.

## Tier 2 — Foundational neural codec references

### EnCodec (arXiv 2210.13438, 2022-10)
- **Why cite**: Architectural ancestor of SCIT (RVQ + adversarial training).
- **SCIT differentiator**: EnCodec lowest evaluated bitrate is 1.5 kbps (24kHz/wideband variants). No payload accounting, no transport framing.

### SoundStream (arXiv 2107.03312, 2021-07)
- **Why cite**: Origin of structured RVQ dropout for variable-bitrate codecs.
- **SCIT differentiator**: SoundStream tests 3-18 kbps; SCIT extends to **500-1500 bps**. SoundStream's dropout is for quality-bitrate scalability; SCIT's L-truncation is for transport.

### StreamCodec2 (arXiv 2509.13670, 2025-09)
- **Why cite**: Recent streaming codec at low bitrate.
- **SCIT differentiator**: StreamCodec2 evaluated only at **single 1.7 kbps** point. No bit-error / packet-loss / token-perturbation experiments. No payload accounting.

## Tier 3 — Speech semantic communication references

### SyncSC (arXiv 2408.04535, 2024-08)
- **Why cite**: Recent semantic-communication speech work.
- **SCIT differentiator**: SyncSC overhead discussion is **qualitative only** ("reduce transmission overhead"). SCIT provides quantitative ideal/packed/packetized accounting.

### LargeSC-era (arXiv 2509.15462, 2025-09)
- **Why cite**: 2025 large-model semantic communication for speech.
- **SCIT differentiator**: Reports "2-4× bitrate reduction" without absolute bitrates or payload accounting in abstract. SCIT operates at **explicit 500/1000/1500 bps with full payload accounting**.

### DeepSC-S / DeepSC-ST (referenced in source pool, prior knowledge)
- **Why cite**: Foundational speech semantic communication.
- **SCIT differentiator**: Both use end-to-end JSCC with continuous channel symbols. SCIT uses **discrete index transmission** with shared codebook prior.

## Tier 4 — Supporting / contextual

### arXiv 2501.05859 (semantic codec related)
### arXiv 2207.03067 (variable-bitrate context)
### arXiv 2506.16538 (RVQ-related 2025)
### arXiv 2404.19441v2 (codec evaluation)
### arXiv 2410.14411 (recent neural audio codec)
### arXiv 2406.08900v1 (perturbation/robustness related)
### arXiv 2510.00264v3 (consistency-loss reference)
### arXiv 2505.16845v1 (codec/robustness)
### arXiv 2506.23325v1 (semantic distillation related)
### arXiv 2602.06213 (distillation context)
### arXiv 2512.20944v1 (very recent distillation/codec)
### arXiv 2604.14654v1 (RVQ context)
### arXiv 2604.26296v1 (semantic distillation context)

These are second-tier supporting references — verified by the workflow but not central to the positioning differentiation. Cite as supporting evidence in §2 if a specific claim needs backup.

## Notes on submission-time refresh

- 2025-2026 arXiv landscape moves quickly. Re-run this sweep before submission.
- IEEE JSAC/TWC/TCOM/INFOCOM 2024-2026 venue-specific search was under-sampled by the workflow (relied on arXiv mirrors). Targeted IEEE Xplore search recommended before submission.
- WavTokenizer, AudioDec, FunCodec, Mimi, Moshi codec papers were referenced via SAC/Codec-SUPERB tables but not individually re-verified for the latest arXiv versions.
