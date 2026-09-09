"""Run the Opus baseline pass for exp12 after the SCIT/EnCodec/DAC baselines finish.

For a given split (test-clean or test-other), this wrapper resolves the matching
exp12 run directory and baseline config, asserts that the original samples
already exist (run_opus_baseline.py reads them), then invokes
scripts/run_opus_baseline.py via subprocess and tees stdout/stderr to a log
file under exp12 logs/.

Usage:
    python run_opus_after_baseline.py --split test-clean
    python run_opus_after_baseline.py --split test-other --bitrates 6000 12000 24000
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


# ffmpeg location used by scripts/run_opus_baseline.py
DEFAULT_FFMPEG = r"C:\Users\Windows11\.conda\envs\speechtokenizer\Library\bin\ffmpeg.exe"

# Anchored on this wrapper's location:
#   <project_root>/output/experiments/<EXP12>/commands/run_opus_after_baseline.py
THIS_FILE = Path(__file__).resolve()
EXP12_DIR = THIS_FILE.parent.parent
PROJECT_ROOT = EXP12_DIR.parents[2]  # output/experiments/<exp> -> project root

SPLIT_TO_PATHS = {
    "test-clean": {
        "run_dir": EXP12_DIR / "runs" / "test-clean_300",
        "config": EXP12_DIR / "configs" / "baseline_test_clean_300.json",
    },
    "test-other": {
        "run_dir": EXP12_DIR / "runs" / "test-other_300",
        "config": EXP12_DIR / "configs" / "baseline_test_other_300.json",
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run Opus baseline against an exp12 split run-dir.",
    )
    ap.add_argument(
        "--split",
        choices=["test-clean", "test-other"],
        required=True,
        help="Which exp12 split to process.",
    )
    ap.add_argument(
        "--bitrates",
        nargs="+",
        type=int,
        default=[6000, 8000, 12000, 16000, 24000],
        help="Opus bitrates in bps.",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=300,
        help="Maximum number of original samples to encode.",
    )
    ap.add_argument(
        "--ffmpeg",
        default=DEFAULT_FFMPEG,
        help="Path to ffmpeg executable.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    paths = SPLIT_TO_PATHS[args.split]
    run_dir: Path = paths["run_dir"]
    config: Path = paths["config"]

    if not run_dir.is_dir():
        raise SystemExit(f"run-dir does not exist: {run_dir}")
    if not config.is_file():
        raise SystemExit(f"config does not exist: {config}")

    samples_orig = run_dir / "samples" / "original"
    if not samples_orig.is_dir():
        raise SystemExit(
            f"original samples dir is missing: {samples_orig}\n"
            "run_opus_baseline.py depends on it; run the SCIT/EnCodec/DAC pass first."
        )
    wavs = sorted(samples_orig.glob("*.wav"))
    assert len(wavs) >= 1, (
        f"expected at least 1 .wav under {samples_orig}, found {len(wavs)}"
    )

    runner = PROJECT_ROOT / "scripts" / "run_opus_baseline.py"
    if not runner.is_file():
        raise SystemExit(f"runner script not found: {runner}")

    logs_dir = EXP12_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_opus_{args.split}_{ts}.log"

    cmd = [
        sys.executable, "-u",
        str(runner),
        "--run-dir", str(run_dir),
        "--config", str(config),
        "--max-samples", str(args.max_samples),
        "--ffmpeg", args.ffmpeg,
        "--bitrates", *[str(b) for b in args.bitrates],
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[run_opus_after_baseline] split={args.split}")
    print(f"[run_opus_after_baseline] run_dir={run_dir}")
    print(f"[run_opus_after_baseline] config={config}")
    print(f"[run_opus_after_baseline] originals={len(wavs)} wav(s)")
    print(f"[run_opus_after_baseline] log={log_path}")
    print(f"[run_opus_after_baseline] cmd={' '.join(cmd)}")

    rc = 0
    with open(log_path, "w", encoding="utf-8") as logf:
        header = (
            f"# split={args.split}\n"
            f"# run_dir={run_dir}\n"
            f"# config={config}\n"
            f"# bitrates={args.bitrates}\n"
            f"# max_samples={args.max_samples}\n"
            f"# ffmpeg={args.ffmpeg}\n"
            f"# cmd={' '.join(cmd)}\n"
            f"# started={ts}\n"
        )
        logf.write(header)
        logf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        rc = proc.wait()

        logf.write(f"# exit_code={rc}\n")

    if rc != 0:
        print(f"[run_opus_after_baseline] FAILED with exit code {rc}", file=sys.stderr)
        return rc
    print("[run_opus_after_baseline] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
