import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiment_utils import (
    FIXED_CONDITIONS,
    collect_environment_dict,
    ensure_run_layout,
    format_environment_markdown,
    normalize_manifest,
    validate_fixed_conditions,
    write_json,
    write_text,
)


def add_project_root_to_path(project_root):
    project_root = str(Path(project_root).resolve())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


EXP_RUN_KEYS = ["exp1", "exp2", "exp3", "exp4", "exp5"]


CRITICAL_SCRIPTS = {
    "shared": [
        "scripts/create_experiment_run.py",
        "scripts/collect_environment.py",
        "scripts/preflight_experiments.py",
        "scripts/pack_indices.py",
        "scripts/channel_sim.py",
    ],
    "exp1": [
        "nas/make_subset.py",
        "nas/search_autoencoder.py",
        "nas/export_best_model.py",
        "nas/custom_model.py",
        "nas/train_nas.py",
    ],
    "exp2": [
        "scripts/train_example.py",
        "scripts/hubert_rep_extract.py",
        "speechtokenizer/model.py",
        "speechtokenizer/trainer/trainer.py",
        "speechtokenizer/trainer/dataset.py",
        "speechtokenizer/trainer/loss.py",
    ],
    "exp3": [
        "scripts/channel_sim.py",
    ],
    "exp4": [
        "scripts/pack_indices.py",
    ],
    "exp5": [
        "scripts/channel_sim.py",
    ],
}


def load_controller_state(controller_dir):
    state_path = Path(controller_dir) / "orchestrator_state.json"
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_lines(path, limit):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            out.append(line)
    return out


def run_smoke_test(project_root, config_path, manifest_path, device):
    add_project_root_to_path(project_root)
    from speechtokenizer.model import SpeechTokenizer
    from speechtokenizer.trainer.dataset import audioDataset, get_dataloader

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    with open(manifest_path, "r", encoding="utf-8") as f:
        file_list = [line for line in f.readlines() if line.strip()]
    if not file_list:
        raise RuntimeError("normalized manifest is empty")

    ds = audioDataset(
        file_list=file_list[:2],
        segment_size=cfg["segment_size"],
        sample_rate=cfg["sample_rate"],
        downsample_rate=320,
        valid=True,
    )
    dl = get_dataloader(ds, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    batch = next(iter(dl))
    x, semantic = batch
    x = x.unsqueeze(1).to(device)

    model = SpeechTokenizer(cfg).to(device)
    model.eval()
    with torch.inference_mode():
        out, commit_loss, feature = model(x)
        codes = model.encode(x, n_q=3, st=0)
        decoded_shapes = {}
        for L in (1, 2, 3):
            decoded = model.decode(codes[:L], st=0)
            decoded_shapes[f"L{L}"] = list(decoded.shape)

    expected_tq = int(cfg["segment_size"] // 320)
    result = {
        "data_batch_audio_shape": list(x.shape),
        "data_batch_semantic_shape": list(semantic.shape),
        "forward_output_shape": list(out.shape),
        "feature_shape": list(feature.shape),
        "commit_loss_is_finite": bool(torch.isfinite(commit_loss).all().item()),
        "codes_shape": list(codes.shape),
        "decoded_shapes": decoded_shapes,
        "expected_latent_frames_for_segment": expected_tq,
        "observed_latent_frames": int(codes.shape[-1]),
        "codes_shape_valid": bool(codes.shape[0] == 3 and codes.shape[1] == x.shape[0]),
        "latent_frame_check": bool(abs(int(codes.shape[-1]) - expected_tq) <= 2),
    }
    if not result["codes_shape_valid"] or not result["latent_frame_check"]:
        raise RuntimeError(f"smoke test shape check failed: {result}")
    return result


def payload_toy_check():
    import numpy as np
    from pack_indices import pack_indices_array
    from channel_sim import apply_channel_sim

    codes = np.array(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14],
        ],
        dtype=np.int64,
    )
    out, sim_stats = apply_channel_sim(codes, L=2, codebook_size=1024, p_drop=0.2, p_sub=0.1, seed=42)
    packed = pack_indices_array(out, bits_per_index=10)
    return {
        "input_shape": list(codes.shape),
        "sim_output_shape": list(out.shape),
        "sim_stats": sim_stats,
        "packed_bytes": len(packed),
        "ideal_bits_L2_for_5_frames": 2 * 5 * 10,
    }


def script_existence(project_root):
    rows = []
    for group, paths in CRITICAL_SCRIPTS.items():
        for rel in paths:
            path = Path(project_root) / rel
            rows.append({"group": group, "path": rel, "exists": path.exists()})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-dir", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-check-lines", type=int, default=200)
    parser.add_argument("--smoke-device", default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    controller_dir = Path(args.controller_dir)
    state = load_controller_state(controller_dir)
    reports_dir = controller_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    env = collect_environment_dict(project_root, extra_tools=["ffmpeg", "opusenc", "opusdec", "codec2"])
    write_json(reports_dir / "preflight_environment.json", env)

    cfg_path = project_root / "config" / "spt_base_cfg.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    fixed_problems = validate_fixed_conditions(cfg)

    script_rows = script_existence(project_root)
    missing_scripts = [row for row in script_rows if not row["exists"]]

    train_src = project_root / cfg["train_files"]
    valid_src = project_root / cfg["valid_files"]
    manifest_stats = {}

    exp_results = {}
    normalized_train = None
    normalized_valid = None

    for exp_key in EXP_RUN_KEYS:
        run_id = state["experiments"][exp_key]["run_id"]
        run_dir = ensure_run_layout(project_root / "output" / "experiments" / run_id)
        shutil.copy2(cfg_path, run_dir / "configs" / "spt_base_cfg.json")
        write_text(
            run_dir / "reports" / "environment.md",
            format_environment_markdown(env),
        )
        write_json(run_dir / "reports" / "environment.json", env)
        (run_dir / "commands" / "run_command.txt").touch(exist_ok=True)

        exp_results[exp_key] = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": "preflight_pending",
            "blockers": [],
            "warnings": [],
        }

        if fixed_problems:
            exp_results[exp_key]["blockers"].append({"kind": "fixed_condition_mismatch", "details": fixed_problems})

    if env["disk_free_gb"] < 30.0:
        for exp in exp_results.values():
            exp["blockers"].append({"kind": "disk_space_below_30gb", "free_gb": env["disk_free_gb"]})

    if not train_src.exists() or not valid_src.exists():
        for exp in exp_results.values():
            exp["blockers"].append(
                {
                    "kind": "manifest_missing",
                    "train_files": str(train_src),
                    "valid_files": str(valid_src),
                }
            )
    else:
        for exp_key in ["exp1", "exp2", "exp3", "exp4", "exp5"]:
            run_id = exp_results[exp_key]["run_id"]
            run_dir = Path(exp_results[exp_key]["run_dir"])
            shutil.copy2(train_src, run_dir / "artifacts" / "train_files.original.txt")
            shutil.copy2(valid_src, run_dir / "artifacts" / "valid_files.original.txt")

            train_out = run_dir / "artifacts" / "train_files.txt"
            valid_out = run_dir / "artifacts" / "valid_files.txt"
            train_stats = normalize_manifest(train_src, train_out, project_root, max_lines=args.manifest_check_lines)
            valid_stats = normalize_manifest(valid_src, valid_out, project_root, max_lines=args.manifest_check_lines)
            write_json(run_dir / "artifacts" / "manifest_normalization.json", {"train": train_stats, "valid": valid_stats})
            write_json(
                run_dir / "configs" / "data_split.json",
                {
                    "seed": args.seed,
                    "source_train_files": str(train_src),
                    "source_valid_files": str(valid_src),
                    "normalized_train_files": str(train_out),
                    "normalized_valid_files": str(valid_out),
                    "normalization_note": "Only file paths were mapped from archived Linux absolute paths to the current repository root; sample ordering and split membership were not changed.",
                    "train_stats": train_stats,
                    "valid_stats": valid_stats,
                },
            )
            manifest_stats[exp_key] = {"train": train_stats, "valid": valid_stats}
            if train_stats["written"] == 0 or valid_stats["written"] == 0:
                exp_results[exp_key]["blockers"].append({"kind": "normalized_manifest_empty", "details": manifest_stats[exp_key]})
            elif train_stats["missing_audio"] or train_stats["missing_feature"] or valid_stats["missing_audio"] or valid_stats["missing_feature"]:
                exp_results[exp_key]["warnings"].append({"kind": "manifest_entries_missing_in_checked_window", "details": manifest_stats[exp_key]})

            if exp_key == "exp2":
                normalized_train = train_out
                normalized_valid = valid_out

    if missing_scripts:
        for exp_key, exp in exp_results.items():
            relevant = [row for row in missing_scripts if row["group"] in ("shared", exp_key)]
            if relevant:
                exp["warnings"].append({"kind": "missing_noncritical_or_future_scripts", "details": relevant})

    smoke_result = None
    smoke_error = None
    if normalized_train is not None and Path(normalized_train).exists() and not fixed_problems:
        smoke_cfg_path = project_root / "output" / "experiments" / exp_results["exp2"]["run_id"] / "configs" / "smoke_spt_base_cfg.json"
        smoke_cfg = dict(cfg)
        smoke_cfg["train_files"] = str(normalized_train)
        smoke_cfg["valid_files"] = str(normalized_valid)
        smoke_cfg["batch_size"] = 1
        smoke_cfg["num_workers"] = 0
        smoke_cfg["segment_size"] = 16000
        smoke_cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        write_json(smoke_cfg_path, smoke_cfg)
        smoke_device = args.smoke_device or smoke_cfg["device"]
        try:
            smoke_result = run_smoke_test(project_root, smoke_cfg_path, normalized_train, smoke_device)
            write_json(project_root / "output" / "experiments" / exp_results["exp2"]["run_id"] / "reports" / "smoke_test.json", smoke_result)
        except Exception as exc:
            smoke_error = {"error": repr(exc), "traceback": traceback.format_exc()}
            for exp_key in ["exp1", "exp2", "exp3", "exp4", "exp5"]:
                exp_results[exp_key]["blockers"].append({"kind": "model_or_data_smoke_failed", "details": smoke_error})

    try:
        payload_result = payload_toy_check()
    except Exception as exc:
        payload_result = {"error": repr(exc), "traceback": traceback.format_exc()}
        for exp_key in ["exp3", "exp4", "exp5"]:
            exp_results[exp_key]["blockers"].append({"kind": "payload_or_channel_toy_failed", "details": payload_result})

    # Dependency gate statuses.
    exp_results["exp2"]["blockers"].append({"kind": "dependency_pending", "details": "Exp2 must wait for Exp1 completed or explicit hand-designed encoder fallback."})
    exp_results["exp3"]["blockers"].append({"kind": "dependency_pending", "details": "Exp3 must wait for a completed, loadable Exp2 Base checkpoint."})
    exp_results["exp4"]["blockers"].append({"kind": "dependency_pending", "details": "SCIT Exp4 results must wait for Exp2/Exp3 checkpoints; codec/tool preflight can proceed."})
    exp_results["exp5"]["blockers"].append({"kind": "dependency_pending", "details": "Exp5 must wait for valid Exp1-Exp3 artifacts."})

    if env["tools"].get("ffmpeg", {}).get("path") is None and env["tools"].get("opusenc", {}).get("path") is None:
        exp_results["exp4"]["warnings"].append({"kind": "traditional_codec_tool_missing", "details": "ffmpeg/opusenc not found; Opus/AMR baseline cannot run until a tool is installed or another documented codec path is supplied."})

    for exp_key, exp in exp_results.items():
        exp["status"] = "preflight_failed" if exp["blockers"] else "preflight_passed"
        run_dir = Path(exp["run_dir"])
        write_json(run_dir / "reports" / "preflight_status.json", exp)
        lines = [
            f"# Preflight Report: {exp_key}",
            "",
            f"- run_id: {exp['run_id']}",
            f"- status: {exp['status']}",
            f"- fixed_conditions: `{FIXED_CONDITIONS}`",
            f"- environment: `{run_dir / 'reports' / 'environment.md'}`",
            "",
            "## Blockers",
        ]
        if exp["blockers"]:
            for item in exp["blockers"]:
                lines.append(f"- {item['kind']}: `{json.dumps(item.get('details', item), ensure_ascii=False)}`")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("## Warnings")
        if exp["warnings"]:
            for item in exp["warnings"]:
                lines.append(f"- {item['kind']}: `{json.dumps(item.get('details', item), ensure_ascii=False)}`")
        else:
            lines.append("- none")
        lines.append("")
        lines.append("## Manifest Normalization")
        lines.append(f"- details: `{run_dir / 'artifacts' / 'manifest_normalization.json'}`")
        lines.append("")
        write_text(run_dir / "reports" / "preflight_report.md", "\n".join(lines))

    aggregate = {
        "status": "preflight_failed" if any(exp["blockers"] for exp in exp_results.values()) else "preflight_passed",
        "environment": env,
        "fixed_condition_problems": fixed_problems,
        "scripts": script_rows,
        "missing_scripts": missing_scripts,
        "manifest_stats": manifest_stats,
        "smoke_result": smoke_result,
        "smoke_error": smoke_error,
        "payload_toy_check": payload_result,
        "experiments": exp_results,
    }
    write_json(reports_dir / "preflight_status.json", aggregate)
    write_json(reports_dir / "agent_status.json", {"agent": "Preflight Agent (simulated)", "status": aggregate["status"], "output": str(reports_dir / "preflight_status.json")})

    report_lines = [
        "# Controller Preflight Report",
        "",
        f"- status: {aggregate['status']}",
        f"- python: `{env['python_executable']}`",
        f"- torch: `{env['packages'].get('torch')}`",
        f"- cuda: `{env['cuda']}`",
        f"- disk_free_gb: {env['disk_free_gb']:.2f}",
        "",
        "## Fixed Condition Check",
    ]
    if fixed_problems:
        for item in fixed_problems:
            report_lines.append(f"- mismatch: `{item}`")
    else:
        report_lines.append("- passed")
    report_lines.extend(["", "## Model/Data Smoke Test"])
    if smoke_result:
        report_lines.append(f"- passed: `{smoke_result}`")
    else:
        report_lines.append(f"- failed or not run: `{smoke_error}`")
    report_lines.extend(["", "## Payload Toy Check", f"- result: `{payload_result}`", "", "## Per-Experiment Status"])
    for exp_key, exp in exp_results.items():
        report_lines.append(f"- {exp_key}: {exp['status']} ({exp['run_id']})")
    report_lines.append("")
    write_text(reports_dir / "preflight_report.md", "\n".join(report_lines))
    write_text(
        reports_dir / "agent_handoff.md",
        "\n".join(
            [
                "# Preflight Agent Handoff",
                "",
                f"- status: {aggregate['status']}",
                f"- preflight_report: {reports_dir / 'preflight_report.md'}",
                f"- preflight_status: {reports_dir / 'preflight_status.json'}",
                "- no long training, NAS search, or background job was started.",
                "- downstream experiments remain dependency-gated until upstream artifacts exist or fallback is explicitly recorded.",
                "",
            ]
        ),
    )

    print(json.dumps({"status": aggregate["status"], "report": str(reports_dir / "preflight_report.md")}, indent=2))
    if aggregate["status"] != "preflight_passed":
        sys.exit(2)


if __name__ == "__main__":
    main()
