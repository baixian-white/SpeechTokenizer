"""Driver script: run the test-other 300-sample baseline pipeline followed by the
Opus post-processing step, teeing stdout/stderr to log files and emitting a
compact JSON status summary.

This script is intentionally self-contained: every path it touches is derived
from ``__file__`` so it can be invoked from any working directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


# --- Path anchors -----------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
COMMANDS_DIR = SCRIPT_PATH.parent
EXPERIMENT_DIR = COMMANDS_DIR.parent
# Repo root: experiment dir is <repo>/output/experiments/<exp>; go up 3 levels.
REPO_ROOT = EXPERIMENT_DIR.parent.parent.parent

RUN_DIR = EXPERIMENT_DIR / "runs" / "test-other_300"
CONFIG_PATH = EXPERIMENT_DIR / "configs" / "baseline_test_other_300.json"
LOGS_DIR = EXPERIMENT_DIR / "logs"
BASELINE_LOG = LOGS_DIR / "test_other_300.log"
OPUS_LOG = LOGS_DIR / "test_other_300_opus.log"

BASELINE_SCRIPT = REPO_ROOT / "scripts" / "run_exp4_baselines.py"
OPUS_SCRIPT = COMMANDS_DIR / "run_opus_after_baseline.py"

RESULTS_CSV = RUN_DIR / "audio_quality_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the test-other 300-sample baseline evaluation, then the Opus "
            "post-processing step, with tee'd logs and a JSON status summary."
        )
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        default=False,
        help="Skip the baseline subprocess (run_exp4_baselines.py).",
    )
    parser.add_argument(
        "--skip-opus",
        action="store_true",
        default=False,
        help="Skip the Opus post-processing subprocess.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=300,
        help="Number of samples to evaluate (default: 300).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device passed to baseline runner (default: cuda).",
    )
    return parser.parse_args()


def _child_env() -> dict:
    """Environment for child Python processes: force unbuffered I/O."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_and_tee(cmd: list[str], log_path: Path) -> int:
    """Run ``cmd`` and tee combined stdout+stderr to both this process's stdout
    and ``log_path``. Returns the child exit code.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[launch] $ {' '.join(str(c) for c in cmd)}", flush=True)
    print(f"[launch] log -> {log_path}", flush=True)

    with log_path.open("w", encoding="utf-8", buffering=1) as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_child_env(),
            cwd=str(REPO_ROOT),
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
        rc = proc.wait()
    print(f"[launch] exit code: {rc}", flush=True)
    return rc


def count_csv_rows(csv_path: Path) -> Optional[int]:
    """Return the number of data rows in ``csv_path`` (excluding header), or
    ``None`` if the file is missing/unreadable.
    """
    if not csv_path.exists():
        return None
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return 0
        # Subtract 1 for the header row when present.
        return max(0, len(rows) - 1)
    except OSError:
        return None


def main() -> int:
    args = parse_args()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    status = {"baseline": "skipped", "opus": "skipped", "row_count": None}

    # Step 1: baseline
    if args.skip_baseline:
        print("[launch] --skip-baseline set; skipping baseline step.", flush=True)
    else:
        baseline_cmd = [
            sys.executable,
            "-u",
            str(BASELINE_SCRIPT),
            "--run-dir",
            str(RUN_DIR),
            "--config",
            str(CONFIG_PATH),
            "--max-samples",
            str(args.max_samples),
            "--device",
            args.device,
        ]
        rc = run_and_tee(baseline_cmd, BASELINE_LOG)
        if rc == 0:
            status["baseline"] = "ok"
        else:
            status["baseline"] = "failed"
            status["row_count"] = count_csv_rows(RESULTS_CSV)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            return rc

    # Step 2: Opus post-processing (only if baseline didn't fail)
    if args.skip_opus:
        print("[launch] --skip-opus set; skipping Opus step.", flush=True)
    else:
        opus_cmd = [
            sys.executable,
            "-u",
            str(OPUS_SCRIPT),
            "--split",
            "test-other",
            "--max-samples",
            str(args.max_samples),
        ]
        rc = run_and_tee(opus_cmd, OPUS_LOG)
        status["opus"] = "ok" if rc == 0 else "failed"
        if rc != 0:
            status["row_count"] = count_csv_rows(RESULTS_CSV)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            return rc

    status["row_count"] = count_csv_rows(RESULTS_CSV)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
