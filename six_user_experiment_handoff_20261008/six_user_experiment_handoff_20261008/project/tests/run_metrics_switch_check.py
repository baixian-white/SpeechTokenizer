"""Real-codec development check: six identities change inside one local stream."""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = next(ROOT.glob("3*demo/speechtokenizer_now/speechtokenizer/*/run_metrics_experiment.py")).parent
sys.path.insert(0, str(DEMO))
from metrics_protocol import read_json, sha256, write_json
from run_metrics_experiment import DEFAULT_ASSET_ROOT, execute, prepare, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tmp" / ("switch_development_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    args.conditions, args.scenarios, args.mode = ["A"], ["switch"], "smoke"
    args.repeats = args.utterances = 1
    args.quality_python = None
    args.playback = "off"
    args.duration_seconds = 36
    args.queue_blocks = 64
    args.torch_threads = 1
    args.block_deadline_seconds = 10
    args.prediction_deadline_seconds = 15
    args.drain_seconds = 45
    paths, report = verify(args)
    if not report["measurement_ready"]:
        raise RuntimeError("Development check prerequisites not ready")
    output, config = prepare(args, paths, report)
    config["jobs"] = config["jobs"][:1]
    config["validation_scope"] = "one_local_routing_stream; six_validation_identities; component_check"
    write_json(output / "experiment_plan.json", config)
    manifest = read_json(output / "run_manifest.json")
    manifest["plan_sha256"] = sha256(output / "experiment_plan.json")
    manifest["validation_scope"] = config["validation_scope"]
    write_json(output / "run_manifest.json", manifest)
    if execute(output, config):
        raise RuntimeError("Switch component run failed")
    rows = list(csv.DictReader((output / "predictions.csv").open(encoding="utf-8-sig")))
    if len(rows) != 34 or len({r["expected_speaker"] for r in rows}) != 6 or len({r["sender_id"] for r in rows}) != 1:
        raise RuntimeError("Switch timeline or window denominator is incorrect")
    if not any(r["transition"] == "True" for r in rows):
        raise RuntimeError("Cross-speaker windows were not retained")
    print("PASS: same routing stream, six true identities, 34 planned windows, transition and response records retained.")


if __name__ == "__main__":
    main()
