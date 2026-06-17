import argparse
import json
from pathlib import Path

import numpy as np


def pack_indices_array(indices, bits_per_index=10):
    arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any(arr < 0):
        raise ValueError("indices contain negative values")
    if np.any(arr >= (1 << bits_per_index)):
        raise ValueError("indices exceed bit width")

    out = bytearray()
    buffer = 0
    buffered_bits = 0
    mask = (1 << bits_per_index) - 1
    for value in arr:
        buffer = (buffer << bits_per_index) | (int(value) & mask)
        buffered_bits += bits_per_index
        while buffered_bits >= 8:
            shift = buffered_bits - 8
            out.append((buffer >> shift) & 0xFF)
            buffer &= (1 << shift) - 1
            buffered_bits -= 8
    if buffered_bits:
        out.append((buffer << (8 - buffered_bits)) & 0xFF)
    return bytes(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", required=True, help="Path to .npy index array shaped L x T or L x B x T")
    parser.add_argument("--output", required=True, help="Packed binary output")
    parser.add_argument("--stats", required=True, help="JSON stats output")
    parser.add_argument("--bits-per-index", type=int, default=10)
    parser.add_argument("--duration-sec", type=float, required=True)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--latent-rate", type=float, default=50.0)
    args = parser.parse_args()

    codes = np.load(args.codes)
    packed = pack_indices_array(codes, bits_per_index=args.bits_per_index)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(packed)

    ideal_bits = int(args.L * args.latent_rate * args.bits_per_index * args.duration_sec)
    stats = {
        "codes_path": args.codes,
        "packed_path": str(output),
        "shape": list(codes.shape),
        "L": args.L,
        "bits_per_index": args.bits_per_index,
        "duration_sec": args.duration_sec,
        "ideal_bits": ideal_bits,
        "packed_payload_bytes": len(packed),
        "packed_payload_bps": (len(packed) * 8.0 / args.duration_sec) if args.duration_sec > 0 else None,
    }
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

