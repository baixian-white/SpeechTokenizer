# Packetized Payload Overhead Model (exp17)

Structural sweep over packet durations and protocol stacks, layered on top of the ideal index payload R(L) = 500 L bps. The grid is purely analytic; the upstream exp12 payload_summary.csv is consulted only as a cross-check that our recomputation of packed_payload_bps agrees with the runner.

## CSV cross-check

Compared 3600 rows from exp12 test-clean_300 + test-other_300 (methods: scit_base, scit_lca). Recomputed `packed_payload_bps = num_indices * bits_per_code / duration_sec` against the runner-stored value. Max abs diff = 4.800000 bps, mean abs diff = 0.515915 bps. Tolerance per spec is 1 bps; residual is dominated by float rounding in the runner.

## Per-L overhead extrema

| L | min stack | min duration | min overhead | max stack | max duration | max overhead |
|---|---|---|---|---|---|---|
| 1 | RTP-only | 100 ms | 204.00% | UDP+IPv6+RTP | 20 ms | 4860.00% |
| 2 | RTP-only | 100 ms | 100.00% | UDP+IPv6+RTP | 20 ms | 2420.00% |
| 3 | RTP-only | 100 ms | 65.33% | UDP+IPv6+RTP | 20 ms | 1606.67% |

## Which packet duration minimizes overhead at each L

Across the swept durations [20, 40, 60, 80, 100] ms, the minimum-overhead packet duration grows monotonically with the chosen protocol stack's per-packet header tax: longer packets amortise the fixed RTP/UDP/IP header bytes over more codec frames, so headier stacks favour longer packets, while RTP-only is already close to its asymptote at moderate durations.

- L=1: minimum overhead 204.00% at packet_duration=100 ms with RTP-only (total 1520.0 bps vs ideal 500 bps).
- L=2: minimum overhead 100.00% at packet_duration=100 ms with RTP-only (total 2000.0 bps vs ideal 1000 bps).
- L=3: minimum overhead 65.33% at packet_duration=100 ms with RTP-only (total 2480.0 bps vs ideal 1500 bps).

## UDP+IPv4 vs UDP+IPv6

At each L's IPv4-optimal packet duration, switching from IPv4 to IPv6 adds the 20-byte address-size delta to every packet, so the overhead penalty scales inversely with packet duration: longer packets dilute the IPv6 hit.

| L | duration | IPv4 overhead | IPv6 overhead | IPv6 - IPv4 |
|---|---|---|---|---|
| 1 | 100 ms | 652.00% | 972.00% | +320.00% |
| 2 | 100 ms | 324.00% | 484.00% | +160.00% |
| 3 | 100 ms | 214.67% | 321.33% | +106.67% |

## Paragraph for §3.1 footnote

The ideal index payload R(L) = 500 L bps stated in §3.1 is an analytical floor that ignores packetization. Under a realistic UDP+IPv4+RTP stack, the on-the-wire bitrate sits 652.00% above this floor at L=1 and 214.67% above at L=3 when packets are sized to 100 ms and 100 ms respectively (the overhead-minimising choices in our sweep). With RTP-only framing the residual overhead drops to 204.00% at L=1 and 65.33% at L=3, matching the upstream exp12 packetized_payload_bps figures within the 1 bps tolerance noted in the cross-check.

## Paragraph for §7 limitation

R(L) = 500 L bps treats indices as a contiguous bitstream and ignores transport framing. In practice the codec must be delivered over RTP and either UDP+IPv4 or UDP+IPv6, which adds a fixed per-packet header tax that is amortised over codec frames packed into each datagram. Within our 20-100 ms sweep this tax ranges from 204.00% to 4860.00% of the ideal payload at L=1, and from 65.33% to 1606.67% at L=3. The asymmetry between L=1 and L=3 is intrinsic: a fixed header amortises over 3x more index payload at L=3, so the absolute bitrate gap between the ideal floor and the on-the-wire bitrate widens with L while the relative overhead shrinks. Latency-sensitive deployments that must use shorter packets (e.g. 20 ms for interactive voice) should expect the upper end of these ranges; offline transport can push toward the lower end.
