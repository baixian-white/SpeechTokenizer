"""Six-process transport/teardown check with an explicitly synthetic codec."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = next(ROOT.glob("3*demo/speechtokenizer_now/speechtokenizer/*/run_metrics_experiment.py")).parent
sys.path.insert(0, str(DEMO))

from metrics_protocol import read_json
from run_metrics_experiment import DEFAULT_ASSET_ROOT, execute, prepare, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tmp" / ("transport_fixture_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    args.conditions = ["D"]
    args.scenarios = ["concurrent"]
    args.mode = "smoke"
    args.repeats = args.utterances = 1
    args.quality_python = None
    args.playback = "off"
    args.duration_seconds = 3
    args.queue_blocks = 64
    args.torch_threads = 1
    args.block_deadline_seconds = 10
    args.prediction_deadline_seconds = 15
    args.drain_seconds = 30
    paths, report = verify(args)
    output, config = prepare(args, paths, report)
    code = execute(output, config, fixture=True)
    receipt = read_json(output / "jobs" / config["jobs"][0]["job_id"] / "receipt.json")
    analysis = read_json(output / "analysis_status.json")
    if code or receipt.get("router_exit_code") != 0 or analysis["expected_block_rows"] != 180 or analysis["expected_prediction_rows"] != 30:
        raise RuntimeError("Transport fixture failed; inspect " + str(output))
    import csv
    rows = list(csv.DictReader((output / "blocks.csv").open(encoding="utf-8-sig")))
    if not all(row["status"] == "decoded" for row in rows):
        raise RuntimeError("Transport fixture did not decode every directed block")
    predictions = list(csv.DictReader((output / "predictions.csv").open(encoding="utf-8-sig")))
    if not all(row["status"] == "fixture" and int(row["paired_samples"]) == 48000 for row in predictions):
        raise RuntimeError("Synthetic paired-window checks were not explicitly marked")
    print("PASS: 6 clients, 30 directions, 180 synthetic block deliveries, clean process exits. Not a model performance run.")


if __name__ == "__main__":
    main()
