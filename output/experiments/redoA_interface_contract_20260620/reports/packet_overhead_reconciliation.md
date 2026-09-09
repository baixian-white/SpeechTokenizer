# Packet Overhead Reconciliation — Computed vs Paper Draft §3.1

**Experiment A (I1) · run `redoA_interface_contract_20260620`**
Source of paper claims: `output/doc/paper_drafts/scit_speech_method_cn_draft_20260609.md` §3.1 (line 61).
Computed by: `commands/run_packet_overhead.py` → `metrics/packet_overhead.csv`,
reusing `payload_stats()` from `scripts/payload_accounting.py` (unmodified).

## Method

For each cell: one packet covers the packing interval.
- `frames_in_interval = f_q · interval` with `f_q = 50 Hz` → 5 frames @100ms, 1 frame @20ms.
- `total_indices = L · frames_in_interval`.
- `ideal_bits = total_indices · ⌈log₂ 1024⌉ = total_indices · 10`.
- `packed_payload_bytes = ⌈ideal_bits / 8⌉` (single global bit-pack of the whole packet).
- `packetized_payload_bytes = packed_payload_bytes + header_bytes` (packet_count = 1).
- `overhead_ratio = (packetized_bytes·8 − ideal_bits) / ideal_bits`.

Header byte sizes: RTP = 12, UDP = 8, IPv4 = 20, IPv6 = 40.
Stacks: RTP-only = 12 B; UDP+IPv4+RTP = 40 B; UDP+IPv6+RTP = 60 B.

## The 9 paper-claimed cells: computed vs claimed

| # | Stack | Interval | L | ideal_bits | packed B | pktz B | Computed overhead | Paper claim | Match |
|---|-------|----------|---|-----------:|---------:|-------:|------------------:|------------:|-------|
| 1 | RTP-only | 100ms | 1 | 50 | 7 | 19 | **+204.00%** | +204% | ✅ exact |
| 2 | RTP-only | 100ms | 2 | 100 | 13 | 25 | **+100.00%** | +100% | ✅ exact |
| 3 | RTP-only | 100ms | 3 | 150 | 19 | 31 | **+65.33%** | +65% | ✅ exact (round) |
| 4 | UDP+IPv4+RTP | 100ms | 1 | 50 | 7 | 47 | **+652.00%** | +652% | ✅ exact |
| 5 | UDP+IPv4+RTP | 100ms | 2 | 100 | 13 | 53 | **+324.00%** | +334% | ⚠️ **−10 pts** |
| 6 | UDP+IPv4+RTP | 100ms | 3 | 150 | 19 | 59 | **+214.67%** | +215% | ✅ exact (round) |
| 7 | UDP+IPv6+RTP | 20ms | 1 | 10 | 2 | 62 | **+4860.00%** | +4860% | ✅ exact |
| 8 | UDP+IPv6+RTP | 20ms | 2 | 20 | 3 | 63 | **+2420.00%** | +2420% | ✅ exact |
| 9 | UDP+IPv6+RTP | 20ms | 3 | 30 | 4 | 64 | **+1606.67%** | +1607% | ✅ exact (round) |

**8 of 9 cells reproduce the paper exactly** (to the paper's reporting precision).

## The one discrepancy — cell #5, UDP+IPv4+RTP @100ms, L=2

Confirmed exactly as flagged in the task brief: clean computation gives **+324%**, paper draft states **+334%** — a ~10-point gap.

### Clean-computation trace (what `metrics/packet_overhead.csv` reports)
- frames_in_interval = 50 × 0.1 = 5; total_indices = 2 × 5 = 10.
- ideal_bits = 10 × 10 = 100 → packed = ⌈100/8⌉ = **13 B**.
- packetized = 13 + 40 = 53 B = 424 bits.
- overhead = (424 − 100) / 100 = 3.24 = **+324.00%**. ✔

This is internally consistent with the surrounding cells, which all match the paper: L=1 (+652%) and L=3 (+214.67%) on the *same* stack/interval use the identical formula and reproduce the paper exactly. Only the L=2 value diverges, which points to an isolated artifact in the original draft rather than a methodological difference (a methodology difference would shift all three L on this row).

### Most likely origin of the paper's +334%: per-frame packing granularity
The clean computation bit-packs all 10 indices of the packet into one contiguous bitstream (13 B). If instead each of the 5 latent frames is byte-aligned independently before assembly (a plausible "implementation" packing the original draft may have measured), then per frame: 2 indices × 10 bits = 20 bits → ⌈20/8⌉ = 3 B, × 5 frames = 15 B payload.
- packetized = 15 + 40 = 55 B = 440 bits → overhead = (440 − 100)/100 = **+340%**.

+340% (per-frame byte alignment) brackets the paper's +334% from above, and +324% (global pack) brackets it from below; +334% sits between the two and matches neither exactly. The cleanest reading is that the draft's +334% is a stale/rounding artifact from an earlier per-frame-ish packing estimate that was not recomputed when the global-pack accounting (`scripts/payload_accounting.py`) was standardized. Note the same per-frame model does **not** break the other cells' agreement at the paper's precision:
- L=1 same row: per-frame 1×10b→2B×5=10B → +560% (≠ paper +652%); so the paper's L=1 used the **global** pack (7 B → +652%, matches).
This inconsistency (L=1 global, L=2 neither) reinforces that +334% is an isolated transcription/rounding slip in the draft, not a deliberate alternative model.

### Disposition
- **Authoritative number for I1: +324.00%** (global bit-pack, `scripts/payload_accounting.py`).
- Numbers were **NOT** fudged to match the paper.
- Recommendation for the paper revision: change §3.1 "L=2 +334%" → "+324%". Optionally add a footnote that packetization granularity (global vs per-frame byte alignment) moves this single cell within [+324%, +340%]; the reported figure uses global packing consistent with all other cells.

## Trichotomy sanity (I1 core relation), per cell
`entropy_lower_bound ≤ 500L (reported) ≤ real_packetized_rate`. The packetized side holds in every cell — every `packetized_payload_bps` exceeds `500L`:

| Stack | Interval | L | 500L bps | packetized bps |
|-------|----------|---|---------:|---------------:|
| RTP-only | 100ms | 1/2/3 | 500/1000/1500 | 1520 / 2000 / 2480 |
| UDP+IPv4+RTP | 100ms | 1/2/3 | 500/1000/1500 | 3760 / 4240 / 4720 |
| UDP+IPv6+RTP | 20ms | 1/2/3 | 500/1000/1500 | 24800 / 25200 / 25600 |

The entropy lower-bound side (`≤ 500L`) is **DEFERRED** — see `reports/entropy_lower_bound_DEFERRED.md` (blocked on a trained SCIT-Speech-Base).
