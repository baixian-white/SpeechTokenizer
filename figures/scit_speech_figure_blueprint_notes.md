# SCIT-Speech overview figure drawing notes

Use `scit_speech_figure_blueprint.svg` as the technical source of truth and the
existing AI-generated overview as the visual-style reference. Preserve the six
panel positions, all labels below, and every arrow direction. Decorative network
blocks may be simplified or replaced without changing the flow.

## Visual grammar

- Solid blue arrow: forward data flow.
- Dashed slate arrow: training supervision, loss return, or an architecture-only link.
- Orange, green, violet: RVQ layers 1, 2, and 3 respectively.
- Panels 1 and 5 are training-time procedures.
- Panels 2–4 are the communication codec path.
- Panel 6 is an optional receiver-side downstream task.

## Panel requirements

1. **Offline encoder architecture search**
   - Candidate Encoder Space → Teacher-Guided Proxy Training → Resource and
     Quality Profiling → Pareto Selection → NAS-Selected Encoder.
   - Only the selected architecture passes to Panel 2. Proxy weights are not reused.

2. **Speech encoding and three-layer RVQ**
   - 16 kHz speech → NAS-selected SEANet encoder → continuous latent `z` at
     50 Hz with `d = 1024`.
   - Q1/C1: coarse content; Q2/C2: acoustic refinement; Q3/C3: fine acoustic
     refinement. Each codebook has 1,024 entries.
   - The three transmitted index streams are `i1`, `i2`, and `i3`.
   - The reconstructed quantized latent is `zq = q1 + q2 + q3`.

3. **Discrete index transmission**
   - `i1, i2, i3` → index packing and packetization → discrete bitstream →
     communication channel.
   - Nominal payload: 1.5 kbit/s.
   - Only discrete indices are transmitted.

4. **Shared-codebook reconstruction and decoding**
   - Received `i1, i2, i3` are looked up in pre-shared C1, C2, and C3.
   - Outputs must be labeled exactly `q1`, `q2`, `q3`.
   - Sum `q1 + q2 + q3` → recovered `zq` → SEANet decoder → reconstructed speech.
   - Codebooks and decoder are pre-deployed and never transmitted.

5. **Training only**
   - 5A semantic path: real speech → frozen HuBERT teacher → teacher feature → Q1.
   - 5A GAN path: real speech and reconstructed speech separately enter the
     multi-scale discriminators. Adversarial and feature-matching losses return
     to the generator.
   - 5A reconstruction path: real and reconstructed speech are compared directly
     by waveform and Mel reconstruction loss. This path is separate from GAN outputs.
   - RVQ commitment updates the RVQ component.
   - 5B clean path: clean indices → decoder `Dθ` → clean Mel.
   - 5B perturbed path: clean indices → index perturbation → perturbed indices →
     the same decoder `Dθ` → perturbed Mel.
   - Clean and perturbed Mel features produce the consistency loss. The complete
     fine-tuning block produces the SCIT-Speech-LCA checkpoint.

6. **Optional token–audio speaker classification**
   - Token branch: received `i1, i2, i3` → trainable RVQ token encoder → `e_token`.
   - Audio branch: reconstructed speech → frozen ECAPA-TDNN → trainable projection
     → `e_audio`.
   - Fusion must read `e_fused = g e_token + (1 − g) e_audio`.
   - Fused embedding → 110-speaker closed-set classifier → predicted speaker ID.
   - This module runs only at the receiver and requires no additional transmission.

## Do not draw

- Do not show codebooks, decoder weights, speaker embeddings, labels, or classifier
  parameters crossing the communication channel.
- Do not connect waveform/Mel reconstruction loss to GAN discriminator outputs.
- Do not draw Mel features as inputs to the decoder.
- Do not feed the predicted speaker identity back into the speech codec.
