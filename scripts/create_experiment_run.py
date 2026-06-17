import argparse
import shutil
from pathlib import Path

from experiment_utils import ensure_run_layout, write_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiments-root", default="output/experiments")
    parser.add_argument("--copy-config", action="append", default=[])
    parser.add_argument("--command", action="append", default=[])
    args = parser.parse_args()

    run_dir = ensure_run_layout(Path(args.experiments_root) / args.run_id)

    for config_path in args.copy_config:
        src = Path(config_path)
        if src.exists():
            shutil.copy2(src, run_dir / "configs" / src.name)

    if args.command:
        write_text(run_dir / "commands" / "run_command.txt", "\n".join(args.command) + "\n")
    else:
        (run_dir / "commands" / "run_command.txt").touch(exist_ok=True)

    print(run_dir)


if __name__ == "__main__":
    main()

