# Experiment A — I1 Interface Contract Freeze + Rate Accounting

Run: `output/experiments/redoA_interface_contract_20260620/`
Date: 2026-06-20 · Project: SCIT-Speech (基于共享 RVQ 码本索引传输的极低码率语音通信)
Scope: Freeze the I1 interface contract, prove the closed-form rate R(L)=500L, tabulate
packet overhead, reconcile against paper draft §3.1. Entropy lower bound DEFERRED.

## The I1 claim
R(L) = L · f_q · ⌈log₂ K⌉ = 500·L bps, L∈{1,2,3} → 500 / 1000 / 1500 bps, and per op point:

    entropy_lower_bound(L)  ≤  500·L (reported)  ≤  real_packetized_rate
       [DEFERRED — Exp B]        [A2, proven]        [A3, proven]

## A1 — Frozen interface contract (interface_contract.json)
K=1024, n_q=3, strides=[8,5,4,2], sample_rate=16000; total_downsample=320;
f_q=50.0 Hz; b=⌈log₂K⌉=10; R(L)=[500.0,1000.0,1500.0]. All 8 assertions PASS
(`all_assertions_passed: true`).

## A2 — Closed-form rate table (metrics/rate_closed_form.csv)
L=1 → 500.0 | L=2 → 1000.0 | L=3 → 1500.0  (f_q=50.0, b=10)

## A3 — Packet overhead trichotomy right side (metrics/packet_overhead.csv)
Driver imports payload_stats() from scripts/payload_accounting.py (unmodified). One packet
per interval; packed=⌈ideal_bits/8⌉; packetized=packed+header. RTP12/UDP8/IPv4 20/IPv6 40.
All 18 cells: packetized_bps > 500L. Paper-cited cells:
  - RTP-only@100ms      L1/2/3: +204% / +100% / +65.33%
  - UDP+IPv4+RTP@100ms  L1/2/3: +652% / +324% / +214.67%
  - UDP+IPv6+RTP@20ms   L1/2/3: +4860% / +2420% / +1606.67%

## Reconciliation vs paper §3.1 (honesty gate)
8/9 cells exact. UDP+IPv4+RTP@100ms L=2: computed +324.00% vs paper +334% (~10pt).
Clean global pack: 10 idx→100b→13B+40B=53B→+324%. Internally consistent with L1/L3 on same
row (both match paper). Likely stale per-frame-packing artifact in draft (per-frame→15B→+340%
brackets +334% above; global→+324% below; draft L1 uses global +652%). NOT fudged.
Authoritative I1 value: +324.00%. Recommend §3.1 "+334%"→"+324%". No other mismatch.

## A4 — Entropy lower bound (left side): DEFERRED
BLOCKED-ON-BASE (Exp B). Methodology in reports/entropy_lower_bound_DEFERRED.md; untested
stub commands/entropy_lower_bound_stub.py. entropy_lower_bound(L)=50·Σ_{ℓ≤L}H_ℓ over 600
samples (test-clean/other 300). Paper targets: L1 ~398–415, L2 ~838–882, L3 ~1301–1356 bps.

## Gate status
- A1 assertions (8): PASS (all_assertions_passed=true)
- A2 R(L)=500L: PASS [500,1000,1500]
- A3 packetized ≥ 500L (18 cells): PASS
- Reconciliation 8/9: PASS (exact)
- Reconciliation cell #5 (UDP+IPv4@100ms L=2): FLAGGED +324.00% vs paper +334% (artifact, not fudged)
- A4 entropy lower bound: DEFERRED (BLOCKED-ON-BASE)

## Note for Experiment B (out of scope here)
config/spt_base_cfg.json has distill_loss_lambda:120 and seed:1234; paper §4.5 +
REPRODUCIBILITY.md line 74 (run exp2_..._distill30_retrain_20260529_seed42) specify
λ_distill=30 and seed=42. Training hyperparams — irrelevant to the interface contract,
left untouched. Exp B must resolve before computing the entropy lower bound.

## Files written
interface_contract.json, metrics/rate_closed_form.csv, metrics/packet_overhead.csv,
commands/{build_contract_and_rate.py, run_packet_overhead.py, entropy_lower_bound_stub.py},
reports/{packet_overhead_reconciliation.md, entropy_lower_bound_DEFERRED.md}, report.md.
No tracked source modified; no other experiment dir touched.
