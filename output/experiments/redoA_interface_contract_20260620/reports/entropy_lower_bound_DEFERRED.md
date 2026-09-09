# Entropy Lower-Bound Harness — DEFERRED (BLOCKED-ON-BASE)

**Experiment A (I1) · run `redoA_interface_contract_20260620`**
Status: **NOT COMPUTED.** Blocked on a trained SCIT-Speech-Base model (Experiment B).
This document specifies the exact methodology so Experiment B can produce the
empirical-entropy lower bound that closes the lower side of the I1 trichotomy.

## Why deferred

The I1 trichotomy is:

```
entropy_lower_bound(L)  ≤  500·L  (reported figure)  ≤  real_packetized_rate
        └── this doc ──┘     └─ A2 ─┘                    └──── A3 ────┘
```

The right two terms are closed-form / config-derived and are produced in this run
(`metrics/rate_closed_form.csv`, `metrics/packet_overhead.csv`). The **left** term is
*empirical*: it depends on the actual distribution of RVQ index values a trained model
emits, which does not exist until SCIT-Speech-Base is trained. Computing entropy now
would require either a random/untrained codebook (meaningless) or fabricating a
histogram (forbidden). Hence: deferred.

## What the paper currently claims (target to reproduce/verify in Exp B)

From §3.1 (line 59 of `scit_speech_method_cn_draft_20260609.md`):
- L=1: ~398–415 bps (≈ 17–20% below the 500·L = 500 bps upper bound)
- L=2: ~838–882 bps
- L=3: ~1301–1356 bps
- Layer-1 per-index empirical entropy ≈ 8.0–8.3 bit (< 10-bit nominal capacity),
  corroborating the "layer-1 structurally low codebook utilization" observation.

These ranges (clean vs other split) are the acceptance targets for the Exp B computation.

## Exact computation steps (for Experiment B)

1. **Model + data.** Load the trained SCIT-Speech-Base (per `REPRODUCIBILITY.md`:
   run `exp2_scit_speech_distill30_retrain_20260529_seed42`, seed=42, λ_distill=30).
   Evaluation set = 600 utterances: test-clean_300 + test-other_300 (first 300 of each
   `test-{clean,other}_all_files.txt`), kept as two separate splits so per-split entropy
   ranges can be reported.

2. **Index extraction.** For each utterance, encode and RVQ-quantize to obtain the
   per-layer integer index sequence `I[ℓ, t]`, ℓ∈{1..n_q=3}, t over latent frames
   (f_q = 50 Hz). Persist as `.npy` of shape `(n_q, T)` (dtype int16/int32; K=1024 < 2^15).
   This is the artifact the stub script below consumes.

3. **Per-layer histogram.** Accumulate, **per split** and **per layer ℓ**, a count vector
   `c_ℓ ∈ N^K` (K=1024) over all frames of all 300 utterances:
   `c_ℓ[k] = #{(u,t) : I_u[ℓ,t] == k}`. Accumulate across utterances, not per-utterance,
   so the histogram reflects the corpus-level marginal distribution.

4. **Per-layer Shannon entropy.** `p_ℓ = c_ℓ / Σ_k c_ℓ[k]`;
   `H_ℓ = −Σ_{k: p>0} p_ℓ[k] · log2 p_ℓ[k]` (bits per index, in [0, 10]).
   Layer-1 H₁ is expected ≈ 8.0–8.3 bit.

5. **Cumulative lossless lower bound at level L.** The transmitted prefix is layers 1..L,
   so the lossless per-frame description length is `Σ_{ℓ=1..L} H_ℓ` bits/frame, and the
   lossless **bitrate** lower bound is:
   `entropy_lower_bound(L) = f_q · Σ_{ℓ=1..L} H_ℓ = 50 · Σ_{ℓ=1..L} H_ℓ  bps`.
   (Marginal/independent-coding bound; a true joint/conditional entropy across layers and
   time would be ≤ this, so this is itself an upper estimate of the true entropy — but it
   is the standard reported per-layer-marginal lower bound vs the uniform 500·L. Report it
   as such.)

6. **Verify trichotomy.** Assert `entropy_lower_bound(L) ≤ 500·L` for each L and each split,
   and check the values land in the paper's ranges (step "what the paper claims").

## Stub script

`commands/entropy_lower_bound_stub.py` — takes one or more index `.npy` files of shape
`(n_q, T)`, builds per-layer histograms, computes `H_ℓ`, the cumulative
`entropy_lower_bound(L) = 50·Σ H_ℓ`, and the gap vs 500·L. **Untested / deferred**: it
needs numpy and a real index file, neither available in this run. It encodes steps 3–6
exactly so Exp B can run it directly once Base indices exist.

## Blocking dependency

**BLOCKED-ON-BASE (Experiment B).** No entropy number is reported in Experiment A.
