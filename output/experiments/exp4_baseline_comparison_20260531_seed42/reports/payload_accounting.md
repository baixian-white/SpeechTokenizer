# Payload Accounting — Exp4

For all methods we report **ideal**, **packed**, and **packetized** payload, plus `overhead_ratio = (packetized - ideal) / ideal`.

- **ideal**: theoretically minimum bits required to transmit the codes, computed as `num_indices × ceil(log2 codebook_size)`. For SCIT, this equals the headline `500 × L bps`.
- **packed**: bit-packed bytes of the index stream (rounded up to byte boundary). Slight upward rounding from ideal due to byte alignment.
- **packetized**: packed bytes plus a fixed 16-byte header (magic + user_id + session_id + timestamp + length). Models a single-packet wrapper for system-level accounting.
- **overhead**: relative cost of byte alignment + packet header.

All numbers averaged over 8 test samples (LibriSpeech train-clean-100 fixed list, durations ~10-15s each).

## Table

| Method | Setting | Ideal bps | Packed bps | Packetized bps | Overhead |
|---|---|---:|---:|---:|---:|
| pcm | 16bit_16khz | 256000 | 256000 | 256000 | 0.0% |
| scit_base | L=1 | 500 | 501 | 517 | **+3.44%** |
| scit_base | L=2 | 1000 | 1001 | 1017 | +1.72% |
| scit_base | L=3 | 1500 | 1501 | 1517 | +1.16% |
| scit_lca | L=1 | 500 | 501 | 517 | +3.44% |
| scit_lca | L=2 | 1000 | 1001 | 1017 | +1.72% |
| scit_lca | L=3 | 1500 | 1501 | 1517 | +1.16% |
| encodec | bw1.5 kbps (2 cb) | 1500 | 1502 | 1518 | +1.19% |
| encodec | bw3.0 kbps (4 cb) | 3000 | 3003 | 3019 | +0.64% |
| encodec | bw6.0 kbps (8 cb) | 6000 | 6006 | 6022 | +0.37% |
| encodec | bw12.0 kbps (16 cb) | 12000 | 12011 | 12028 | +0.23% |
| dac | n_q=1 | 501 | 501 | 517 | +3.26% |
| dac | n_q=2 | 1001 | 1001 | 1017 | +1.63% |
| dac | n_q=3 | 1501 | 1501 | 1517 | +1.09% |
| dac | n_q=4 | 2001 | 2001 | 2017 | +0.82% |
| dac | n_q=6 | 3002 | 3002 | 3018 | +0.54% |
| dac | n_q=9 | 4503 | 4503 | 4519 | +0.36% |
| dac | n_q=12 | 6003 | 6003 | 6019 | +0.27% |

## Header Schema (16 bytes)

| Field | Bytes | Purpose |
|---|---:|---|
| magic | 4 | stream identifier |
| user_id | 2 | user / session identifier |
| session_id | 4 | session identifier |
| timestamp | 4 | monotonic packet timestamp (ms) |
| length | 2 | payload length in bytes |

## Notes

- **SCIT has slightly higher overhead at low bitrates** (3.44% at L=1 vs 3.26% for DAC n_q=1) because the same fixed 16-byte header amortizes over a smaller payload. This is unavoidable for any low-bitrate codec; it diminishes as L increases.
- **Packed payload ≠ `int64` array storage**: the `.npy` file size for an int64-encoded codes tensor would be `num_indices * 8 bytes`, vastly larger than the bit-packed transmission cost. We deliberately compute the bit-packed size based on `ceil(log2 1024) = 10 bits per code`.
- **Packetized payload assumes a single packet per utterance**. For longer streams, the header amortizes further; for shorter packets (e.g., one packet per 100 ms frame), overhead grows. The 16-byte header is a deliberately conservative choice; in production transports header sizes vary with the protocol.
- Real-world transmission would also include UDP/TCP/RTP framing, codec-config negotiation, and possible FEC. None of these are modeled here — our accounting captures only the application-layer payload.
