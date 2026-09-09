"""A3: Packet overhead table by importing payload_stats() from the existing
scripts/payload_accounting.py (no modification of that file).

For each (protocol stack x packing interval x L), build one packet covering the
interval and tabulate ideal vs packed vs packetized payload + overhead.

Run from repo root:
    python output/experiments/redoA_interface_contract_20260620/commands/run_packet_overhead.py
"""
import csv
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]

# Import payload_stats from the existing accounting script without modifying it.
_spec = importlib.util.spec_from_file_location(
    "payload_accounting", REPO / "scripts" / "payload_accounting.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
payload_stats = _mod.payload_stats

K = 1024
F_Q = 50.0  # latent frame rate Hz (frozen contract)

STACKS = [
    ("RTP-only", 12),
    ("UDP+IPv4+RTP", 8 + 20 + 12),  # 40
    ("UDP+IPv6+RTP", 8 + 40 + 12),  # 60
]
INTERVALS_MS = [100, 20]


def main() -> None:
    rows = []
    for stack, header_bytes in STACKS:
        for interval_ms in INTERVALS_MS:
            interval = interval_ms / 1000.0
            frames_in_interval = int(round(F_Q * interval))
            for L in (1, 2, 3):
                total_indices = L * frames_in_interval
                s = payload_stats(
                    total_indices=total_indices,
                    duration_sec=interval,
                    L=L,
                    K=K,
                    header_bytes=header_bytes,
                    packet_count=1,
                )
                rows.append(
                    {
                        "stack": stack,
                        "header_bytes": header_bytes,
                        "interval_ms": interval_ms,
                        "L": L,
                        "total_indices": total_indices,
                        "ideal_bits": s["ideal_bits"],
                        "packed_payload_bytes": s["packed_payload_bytes"],
                        "packetized_payload_bytes": s["packetized_payload_bytes"],
                        "packetized_payload_bps": round(s["packetized_payload_bps"], 4),
                        "overhead_ratio": round(s["overhead_ratio"], 6),
                        "overhead_pct": round(s["overhead_ratio"] * 100, 2),
                    }
                )

    out = RUN / "metrics" / "packet_overhead.csv"
    fields = [
        "stack",
        "header_bytes",
        "interval_ms",
        "L",
        "total_indices",
        "ideal_bits",
        "packed_payload_bytes",
        "packetized_payload_bytes",
        "packetized_payload_bps",
        "overhead_ratio",
        "overhead_pct",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print(
            f"{r['stack']:<14} {r['interval_ms']:>4}ms L={r['L']} "
            f"idx={r['total_indices']:>3} ideal={r['ideal_bits']:>4}b "
            f"packed={r['packed_payload_bytes']:>3}B pktz={r['packetized_payload_bytes']:>3}B "
            f"-> +{r['overhead_pct']}%"
        )


if __name__ == "__main__":
    main()
