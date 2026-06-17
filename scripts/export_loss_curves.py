import argparse
import csv
import json
from pathlib import Path

try:
    from scripts.experiment_utils import write_text
except ImportError:
    from experiment_utils import write_text


def _event_files(run_dir):
    run_dir = Path(run_dir)
    roots = [run_dir / "logs", run_dir / "checkpoints" / "logs", run_dir / "logs" / "tensorboard"]
    files = []
    for root in roots:
        if root.exists():
            files.extend(p for p in root.rglob("events.out.tfevents.*") if p.is_file())
    return sorted(set(files))


def export_loss_curves(run_dir):
    run_dir = Path(run_dir)
    report_path = run_dir / "reports" / "loss_curves.md"
    output_path = run_dir / "metrics" / "loss_curves.csv"
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        write_text(
            report_path,
            "# Loss Curves\n\n"
            f"TensorBoard event export was skipped because the dependency is unavailable: {exc}\n",
        )
        return {"status": "skipped", "reason": "tensorboard_dependency_unavailable", "output_path": None}

    files = _event_files(run_dir)
    if not files:
        write_text(report_path, "# Loss Curves\n\nNo TensorBoard event files were found for this run.\n")
        return {"status": "skipped", "reason": "no_event_files", "output_path": None}

    rows = []
    for event_file in files:
        acc = event_accumulator.EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            for event in acc.Scalars(tag):
                rows.append(
                    {
                        "event_file": str(event_file),
                        "tag": tag,
                        "step": event.step,
                        "wall_time": event.wall_time,
                        "value": event.value,
                    }
                )

    if not rows:
        write_text(report_path, "# Loss Curves\n\nTensorBoard event files were present, but no scalar tags were found.\n")
        return {"status": "skipped", "reason": "no_scalar_tags", "output_path": None}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["event_file", "tag", "step", "wall_time", "value"])
        writer.writeheader()
        writer.writerows(rows)
    write_text(
        report_path,
        "# Loss Curves\n\n"
        f"Exported {len(rows)} scalar rows from {len(files)} TensorBoard event file(s) to `{output_path}`.\n",
    )
    return {"status": "completed", "rows": len(rows), "output_path": str(output_path)}


def build_parser():
    parser = argparse.ArgumentParser(description="Export TensorBoard scalar loss curves for Exp2 if available.")
    parser.add_argument("--run-dir", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    print(json.dumps(export_loss_curves(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
