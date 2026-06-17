import argparse
from pathlib import Path

from experiment_utils import collect_environment_dict, format_environment_markdown, write_json, write_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--tool", action="append", default=[])
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    env = collect_environment_dict(args.project_root, extra_tools=args.tool)
    write_json(run_dir / "reports" / "environment.json", env)
    write_text(run_dir / "reports" / "environment.md", format_environment_markdown(env))
    print(run_dir / "reports" / "environment.md")


if __name__ == "__main__":
    main()

