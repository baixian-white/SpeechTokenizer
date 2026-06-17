import argparse
import csv
import json
import math
from pathlib import Path


def pack_indices(values: list[int], bits_per_index: int) -> bytes:
    acc = 0
    acc_bits = 0
    out = bytearray()
    mask = (1 << bits_per_index) - 1
    for value in values:
        if value < 0 or value > mask:
            raise ValueError(f"index {value} outside {bits_per_index}-bit range")
        acc = (acc << bits_per_index) | value
        acc_bits += bits_per_index
        while acc_bits >= 8:
            shift = acc_bits - 8
            out.append((acc >> shift) & 0xFF)
            acc_bits -= 8
            acc &= (1 << acc_bits) - 1 if acc_bits else 0
    if acc_bits:
        out.append((acc << (8 - acc_bits)) & 0xFF)
    return bytes(out)


def payload_stats(
    *,
    total_indices: int,
    duration_sec: float,
    L: int,
    K: int,
    header_bytes: int,
    packet_count: int,
) -> dict:
    bits_per_index = math.ceil(math.log2(K))
    ideal_bits = total_indices * bits_per_index
    packed_bytes = math.ceil(ideal_bits / 8)
    packetized_bytes = packed_bytes + header_bytes * packet_count
    return {
        "L": L,
        "K": K,
        "bits_per_index": bits_per_index,
        "total_indices": total_indices,
        "duration_sec": duration_sec,
        "ideal_bits": ideal_bits,
        "ideal_bitrate_bps": ideal_bits / duration_sec if duration_sec else None,
        "packed_payload_bytes": packed_bytes,
        "packed_payload_bps": packed_bytes * 8 / duration_sec if duration_sec else None,
        "packet_header_bytes": header_bytes,
        "packet_count": packet_count,
        "packetized_payload_bytes": packetized_bytes,
        "packetized_payload_bps": packetized_bytes * 8 / duration_sec if duration_sec else None,
        "overhead_ratio": (packetized_bytes * 8 - ideal_bits) / ideal_bits if ideal_bits else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute bit-packed and packetized payload stats.")
    parser.add_argument("--total_indices", type=int, required=True)
    parser.add_argument("--duration_sec", type=float, required=True)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--header_bytes", type=int, default=16)
    parser.add_argument("--packet_count", type=int, default=1)
    parser.add_argument("--json_out", required=True)
    parser.add_argument("--csv_out", required=True)
    args = parser.parse_args()

    stats = payload_stats(
        total_indices=args.total_indices,
        duration_sec=args.duration_sec,
        L=args.L,
        K=args.K,
        header_bytes=args.header_bytes,
        packet_count=args.packet_count,
    )
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    with Path(args.csv_out).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
        writer.writeheader()
        writer.writerow(stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
