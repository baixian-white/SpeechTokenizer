"""Reconstruct SpeechTokenizer_best_dev.pt from a trainer checkpoint.

Use case: the trainer's `SpeechTokenizer_best_dev.pt` was truncated by a disk-full
event during a write at step 47500; the trainer pkg at step 45000 still holds the
exact generator state that was last successfully marked as best-dev.
"""
import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer-ckpt", required=True, help="path to SpeechTokenizerTrainer_NNNNNNNN")
    ap.add_argument("--output", required=True, help="path to write best_dev pt")
    args = ap.parse_args()

    src = Path(args.trainer_ckpt)
    dst = Path(args.output)

    print(f"loading {src} ...", flush=True)
    pkg = torch.load(src, map_location="cpu", weights_only=False)
    print(f"keys: {sorted(pkg.keys())}", flush=True)
    if "generator" not in pkg:
        raise SystemExit("trainer pkg missing 'generator' key")

    gen = pkg["generator"]
    if not isinstance(gen, dict):
        raise SystemExit(f"unexpected generator type: {type(gen)}")
    print(f"generator state_dict keys: {len(gen)}", flush=True)
    if "best_dev_mel_loss" in pkg:
        print(f"best_dev_mel_loss in pkg: {pkg['best_dev_mel_loss']}", flush=True)

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gen, dst)
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"wrote {dst} ({size_mb:.2f} MiB)", flush=True)


if __name__ == "__main__":
    main()
