import argparse
import json
from pathlib import Path


REQUIRED_DIRS = [
    "configs",
    "commands",
    "logs",
    "checkpoints",
    "metrics",
    "samples",
    "reports",
    "artifacts",
]

REQUIRED_FILES = [
    "commands/run_command.txt",
    "reports/environment.md",
    "reports/status.json",
    "reports/summary.md",
    "metrics/results.json",
    "metrics/results.csv",
]


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def review_run(run_dir):
    run_dir = Path(run_dir)
    missing_dirs = [name for name in REQUIRED_DIRS if not (run_dir / name).is_dir()]
    missing_files = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    status_path = run_dir / "reports" / "status.json"
    status = "missing"
    status_payload = {}
    if status_path.is_file():
        try:
            status_payload = load_json(status_path)
            status = status_payload.get("status", "missing-status-field")
        except Exception as exc:
            status = "invalid-status-json"
            status_payload = {"error": str(exc)}
    failure_report_required = status == "aborted"
    failure_report_present = (run_dir / "reports" / "failure_report.md").is_file()
    sample_count = len(list((run_dir / "samples").rglob("*"))) if (run_dir / "samples").is_dir() else 0
    metric_count = len(list((run_dir / "metrics").glob("*"))) if (run_dir / "metrics").is_dir() else 0
    issues = []
    for item in missing_dirs:
        issues.append(f"missing directory: {item}")
    for item in missing_files:
        issues.append(f"missing required file: {item}")
    if failure_report_required and not failure_report_present:
        issues.append("aborted run is missing reports/failure_report.md")
    return {
        "run_id": run_dir.name,
        "path": str(run_dir),
        "status": status,
        "status_payload": status_payload,
        "missing_dirs": missing_dirs,
        "missing_files": missing_files,
        "failure_report_required": failure_report_required,
        "failure_report_present": failure_report_present,
        "metric_file_count": metric_count,
        "sample_tree_item_count": sample_count,
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-dir", required=True)
    parser.add_argument("--experiments-root", default="output/experiments")
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()

    experiments_root = Path(args.experiments_root)
    reviews = [review_run(experiments_root / run_id) for run_id in args.runs]
    exp6_like = [
        p.name
        for p in experiments_root.iterdir()
        if p.is_dir() and any(token in p.name.lower() for token in ["exp6", "three", "3user", "multi_user"])
    ]
    overall_issues = [f"{row['run_id']}: {issue}" for row in reviews for issue in row["issues"]]
    if exp6_like:
        overall_issues.append(f"out-of-scope experiment-like directories present: {', '.join(exp6_like)}")
    payload = {
        "status": "passed" if not overall_issues else "issues_found",
        "runs": reviews,
        "out_of_scope_exp6_like_directories": exp6_like,
        "issues": overall_issues,
    }

    controller_dir = Path(args.controller_dir)
    write_json(controller_dir / "reports" / "monitor_review.json", payload)
    lines = ["# Monitor / Reviewer Report", ""]
    lines.append(f"- status: {payload['status']}")
    lines.append(f"- out-of-scope experiment-like directories: {', '.join(exp6_like) if exp6_like else 'none'}")
    lines.append("")
    for row in reviews:
        lines.append(f"## {row['run_id']}")
        lines.append(f"- status: {row['status']}")
        lines.append(f"- metrics files: {row['metric_file_count']}")
        lines.append(f"- sample tree items: {row['sample_tree_item_count']}")
        lines.append(f"- failure_report_present: {row['failure_report_present']}")
        if row["issues"]:
            lines.append(f"- issues: {'; '.join(row['issues'])}")
        else:
            lines.append("- issues: none")
        lines.append("")
    write_text(controller_dir / "reports" / "monitor_review.md", "\n".join(lines))
    print(json.dumps({"status": payload["status"], "issues": len(overall_issues)}, indent=2))


if __name__ == "__main__":
    main()
