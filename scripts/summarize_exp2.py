import argparse
import json
from pathlib import Path

try:
    from scripts.experiment_utils import iso_now, read_json, validate_fixed_conditions, write_json, write_text
except ImportError:
    from experiment_utils import iso_now, read_json, validate_fixed_conditions, write_json, write_text


def _read_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    return read_json(path)


def _exists(path):
    return bool(path) and Path(path).exists()


def _metric_state(run_dir):
    metrics = {
        "layer_reconstruction": run_dir / "metrics" / "layer_reconstruction.json",
        "codebook_usage": run_dir / "metrics" / "codebook_usage.json",
        "loss_curves": run_dir / "metrics" / "loss_curves.csv",
    }
    return {name: {"path": str(path), "exists": path.exists()} for name, path in metrics.items()}


def _determine_status(config, checkpoint_ok, metrics_state, fixed_problems):
    if config is None or fixed_problems:
        return "failure"
    required_metrics = metrics_state["layer_reconstruction"]["exists"] and metrics_state["codebook_usage"]["exists"]
    if not checkpoint_ok or not required_metrics:
        return "partial"
    if config.get("exp2_run_mode", "formal") != "formal":
        return "partial"
    return "completed"


def summarize_exp2(run_dir, config=None):
    run_dir = Path(run_dir)
    config_path = Path(config) if config else run_dir / "configs" / "scit_speech_base_config.json"
    cfg = _read_json_if_exists(config_path)
    fixed_problems = validate_fixed_conditions(cfg) if cfg else [{"key": "config", "expected": "present", "actual": "missing"}]

    checkpoint_manifest_path = run_dir / "checkpoints" / "checkpoint_manifest.json"
    checkpoint_manifest = _read_json_if_exists(checkpoint_manifest_path)
    packaged_checkpoint = run_dir / "checkpoints" / "SCIT-Speech-Base_best.pt"
    checkpoint_ok = packaged_checkpoint.exists() and checkpoint_manifest is not None

    handoff = None
    if cfg and cfg.get("nas_encoder_config"):
        handoff = _read_json_if_exists(cfg["nas_encoder_config"])

    metrics_state = _metric_state(run_dir)
    status = _determine_status(cfg, checkpoint_ok, metrics_state, fixed_problems)
    missing = []
    if cfg is None:
        missing.append("configs/scit_speech_base_config.json")
    if not checkpoint_ok:
        missing.append("checkpoints/SCIT-Speech-Base_best.pt or checkpoint_manifest.json")
    for name, state in metrics_state.items():
        if name != "loss_curves" and not state["exists"]:
            missing.append(state["path"])

    status_doc = {
        "status": status,
        "run_id": run_dir.name,
        "timestamp": iso_now(),
        "run_mode": (cfg or {}).get("exp2_run_mode", "unknown"),
        "config_path": str(config_path),
        "checkpoint": {
            "packaged_path": str(packaged_checkpoint),
            "exists": packaged_checkpoint.exists(),
            "manifest_path": str(checkpoint_manifest_path),
            "manifest_exists": checkpoint_manifest is not None,
            "sha256": (checkpoint_manifest or {}).get("sha256"),
        },
        "metrics": metrics_state,
        "fixed_condition_problems": fixed_problems,
        "missing_required_items": missing,
        "loss_curve_available": metrics_state["loss_curves"]["exists"],
        "notes": [
            "Exp2 summary only reports generated artifacts.",
            "Layer reconstruction and codebook usage are sanity/proxy diagnostics, not final communication quality conclusions.",
            "WER, PESQ, STOI, ChannelSim, packetization, three-user routing, and baseline codecs are outside this Exp2 script chain.",
        ],
    }

    summary_lines = [
        "# Experiment 2 Summary",
        "",
        f"- run_id: {run_dir.name}",
        f"- status: {status}",
        f"- run_mode: {status_doc['run_mode']}",
        f"- config: {config_path if cfg else 'missing'}",
        f"- checkpoint: {'available' if checkpoint_ok else 'missing'}",
        f"- loss_curves: {'available' if metrics_state['loss_curves']['exists'] else 'not exported'}",
        "",
        "## Exp1 Handoff",
    ]
    if handoff:
        summary_lines.extend(
            [
                f"- candidate_id: {handoff.get('candidate_id')}",
                f"- encoder_strides: {handoff.get('encoder_strides')}",
                f"- handoff_schema: {handoff.get('handoff_schema')}",
                f"- decoder_condition_note: Exp1 `{handoff.get('decoder_condition')}` is handoff provenance only; Exp2 decoder is trained normally.",
            ]
        )
    else:
        summary_lines.append("- missing or unreadable Exp1 handoff config")
    summary_lines.extend(["", "## Artifact Availability"])
    for name, state in metrics_state.items():
        summary_lines.append(f"- {name}: {'yes' if state['exists'] else 'no'} ({state['path']})")
    if missing:
        summary_lines.extend(["", "## Missing Required Items"])
        summary_lines.extend(f"- {item}" for item in missing)
    summary_lines.extend(
        [
            "",
            "## Scope Notes",
            "- This report does not invent training results, loss values, WER, PESQ, STOI, or checkpoint provenance.",
            "- `evaluate_layer_reconstruction.py` and `codebook_usage_report.py` produce sanity/proxy diagnostics only.",
            "",
        ]
    )

    handoff_lines = [
        "# Exp2 Agent Handoff",
        "",
        f"- run_id: {run_dir.name}",
        f"- status: {status}",
        f"- run_mode: {status_doc['run_mode']}",
        f"- config_path: {config_path}",
        f"- checkpoint_manifest: {checkpoint_manifest_path if checkpoint_manifest else 'missing'}",
        f"- layer_reconstruction: {metrics_state['layer_reconstruction']['path'] if metrics_state['layer_reconstruction']['exists'] else 'missing'}",
        f"- codebook_usage: {metrics_state['codebook_usage']['path'] if metrics_state['codebook_usage']['exists'] else 'missing'}",
        "",
        "Next agent should only treat completed formal runs with checkpoint and metrics as reportable Exp2 training outputs.",
        "",
    ]

    write_json(run_dir / "reports" / "status.json", status_doc)
    write_text(run_dir / "reports" / "summary.md", "\n".join(summary_lines))
    write_text(run_dir / "reports" / "agent_handoff.md", "\n".join(handoff_lines))
    return status_doc


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize Exp2 artifacts without inventing results.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    status = summarize_exp2(args.run_dir, args.config)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
