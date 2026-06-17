import argparse
import json
import re
import shutil
from pathlib import Path

try:
    from scripts.experiment_utils import file_sha256, iso_now, write_json
except ImportError:
    from experiment_utils import file_sha256, iso_now, write_json


BEST_NAME = "SCIT-Speech-Base_best.pt"


def _checkpoint_step(path):
    matches = re.findall(r"\d+", Path(path).name)
    return int(matches[-1]) if matches else -1


def find_checkpoint(run_dir):
    checkpoints = Path(run_dir) / "checkpoints"
    best_dev = checkpoints / "SpeechTokenizer_best_dev.pt"
    if best_dev.exists():
        return best_dev, "SpeechTokenizer_best_dev.pt"

    trainer_ckpts = [p for p in checkpoints.glob("SpeechTokenizerTrainer_*") if p.is_file()]
    if trainer_ckpts:
        trainer_ckpts.sort(key=lambda p: (_checkpoint_step(p), p.stat().st_mtime))
        return trainer_ckpts[-1], "latest SpeechTokenizerTrainer_*"

    raise FileNotFoundError(
        f"No checkpoint found under {checkpoints}. Expected SpeechTokenizer_best_dev.pt "
        "or SpeechTokenizerTrainer_*."
    )


def package_checkpoint(run_dir, config, checkpoint=None):
    run_dir = Path(run_dir)
    config = Path(config)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    if checkpoint:
        source = Path(checkpoint)
        selection_reason = "explicit checkpoint"
        if not source.exists():
            raise FileNotFoundError(f"Specified checkpoint does not exist: {source}")
    else:
        source, selection_reason = find_checkpoint(run_dir)

    if not config.exists():
        raise FileNotFoundError(f"Config does not exist: {config}")

    packaged = checkpoints / BEST_NAME
    if source.resolve() != packaged.resolve():
        shutil.copy2(source, packaged)

    manifest = {
        "run_id": run_dir.name,
        "timestamp": iso_now(),
        "source_path": str(source),
        "packaged_path": str(packaged),
        "config_path": str(config),
        "sha256": file_sha256(packaged),
        "selection_reason": selection_reason,
        "note": "Packaged checkpoint only; this script does not train or evaluate.",
    }
    write_json(checkpoints / "checkpoint_manifest.json", manifest)
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Package an Exp2 checkpoint under the standard filename.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    manifest = package_checkpoint(args.run_dir, args.config, args.checkpoint)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
