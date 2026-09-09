"""A1 + A2: Interface contract audit/freeze and closed-form rate table.

Pure stdlib. Reads config/spt_base_cfg.json relative to repo root.
Fails loudly (AssertionError) if any I1 invariant breaks.

Run from repo root:
    python output/experiments/redoA_interface_contract_20260620/commands/build_contract_and_rate.py
"""
import csv
import json
import math
from functools import reduce
from pathlib import Path

# repo root = parents[4] of this file:
# .../commands/build_contract_and_rate.py
# parents[0]=commands parents[1]=run parents[2]=experiments parents[3]=output parents[4]=repo
REPO = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "config" / "spt_base_cfg.json"


def product(xs):
    return reduce(lambda a, b: a * b, xs, 1)


def main() -> None:
    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))

    K = cfg["codebook_size"]
    n_q = cfg["n_q"]
    strides = cfg["strides"]
    sample_rate = cfg["sample_rate"]

    total_downsample = product(strides)
    f_q = sample_rate / total_downsample
    b = math.ceil(math.log2(K))
    R = {L: L * f_q * b for L in (1, 2, 3)}

    assertions = []

    def check(name, condition, got, expected):
        passed = bool(condition)
        assertions.append(
            {"name": name, "passed": passed, "got": got, "expected": expected}
        )
        return passed

    check("K == 1024", K == 1024, K, 1024)
    check("n_q == 3", n_q == 3, n_q, 3)
    check("strides == [8,5,4,2]", strides == [8, 5, 4, 2], strides, [8, 5, 4, 2])
    check("product(strides) == 320", total_downsample == 320, total_downsample, 320)
    check("sample_rate == 16000", sample_rate == 16000, sample_rate, 16000)
    check("f_q == 50.0", f_q == 50.0, f_q, 50.0)
    check("bits_per_index b == 10", b == 10, b, 10)
    check(
        "R(L) == [500,1000,1500]",
        [R[1], R[2], R[3]] == [500.0, 1000.0, 1500.0],
        [R[1], R[2], R[3]],
        [500.0, 1000.0, 1500.0],
    )

    all_passed = all(a["passed"] for a in assertions)

    contract = {
        "i1_claim": "R(L) = L * f_q * ceil(log2 K) = 500L bps for L in {1,2,3}",
        "source_config": str(CFG_PATH.relative_to(REPO)).replace("\\", "/"),
        "source_values": {
            "codebook_size_K": K,
            "n_q": n_q,
            "strides": strides,
            "sample_rate_hz": sample_rate,
        },
        "derived_values": {
            "total_downsample": total_downsample,
            "latent_frame_rate_f_q_hz": f_q,
            "bits_per_index_b": b,
            "R_bps": {str(L): R[L] for L in (1, 2, 3)},
        },
        "assertions": assertions,
        "all_assertions_passed": all_passed,
    }

    (RUN / "interface_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )

    # A2: closed-form rate table
    rate_csv = RUN / "metrics" / "rate_closed_form.csv"
    with rate_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["L", "latent_rate_hz", "bits_per_index", "R_bps"])
        for L in (1, 2, 3):
            w.writerow([L, f_q, b, R[L]])

    print(json.dumps(contract, indent=2))
    if not all_passed:
        failed = [a["name"] for a in assertions if not a["passed"]]
        raise AssertionError(f"I1 contract assertions FAILED: {failed}")
    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
