import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.experiment_utils import (
        ensure_run_layout,
        normalize_manifest,
        validate_fixed_conditions,
        write_json,
        write_text,
    )
except ImportError:
    from experiment_utils import (
        ensure_run_layout,
        normalize_manifest,
        validate_fixed_conditions,
        write_json,
        write_text,
    )


DEFAULT_EXP1_BEST = (
    "output/experiments/exp1_nas_distill_run1_seed42/"
    "artifacts/best_architecture/best_seanet_config.json"
)
RUN_SUBDIRS = ("configs", "commands", "logs", "checkpoints", "metrics", "samples", "reports", "artifacts")
EXPECTED_CANDIDATE_ID = "nas_seed42_000896"


@dataclass
class PreparedExp2Config:
    run_dir: Path
    config_path: Path
    command_path: Path
    data_split_path: Path
    train_files_path: Path
    valid_files_path: Path
    fixed_sample_list_path: Path


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _copy_to_dir(source, target_dir, target_name=None):
    source = Path(source)
    target = Path(target_dir) / (target_name or source.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _path_for_config(path):
    return str(Path(path))


def _normalize_or_copy_manifest(source, output, project_root, max_lines=None):
    stats = normalize_manifest(source, output, project_root, max_lines=max_lines)
    if stats["written"] > 0:
        stats["fallback_raw_copy"] = False
        return stats

    rows = []
    with open(source, "r", encoding="utf-8-sig") as f:
        for line in f:
            if max_lines is not None and len(rows) >= max_lines:
                break
            if line.strip():
                rows.append(line if line.endswith("\n") else line + "\n")
    Path(output).write_text("".join(rows), encoding="utf-8", newline="\n")
    stats["written"] = len(rows)
    stats["fallback_raw_copy"] = True
    return stats


def _build_fixed_sample_list(valid_files, output, max_samples):
    rows = []
    with open(valid_files, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(line if line.endswith("\n") else line + "\n")
            if len(rows) >= max_samples:
                break
    Path(output).write_text("".join(rows), encoding="utf-8", newline="\n")
    return {"source_path": str(valid_files), "output_path": str(output), "written": len(rows)}


def _apply_run_mode_overrides(cfg, mode):
    cfg["exp2_run_mode"] = mode
    if mode in {"debug", "tracer"}:
        cfg["epochs"] = 1
        cfg["max_train_steps"] = int(cfg.get("max_train_steps", 2) or 2)
        cfg["save_model_steps"] = int(cfg.get("save_model_steps_debug", 1) or 1)
        cfg["log_steps"] = min(int(cfg.get("log_steps", 100) or 100), 1)
        cfg["stdout_steps"] = min(int(cfg.get("stdout_steps", 10) or 10), 1)
        cfg["num_workers"] = 0
        cfg["valid_num_workers"] = 0
    else:
        cfg.pop("max_train_steps", None)
        cfg["exp2_run_mode"] = "formal"
    return cfg


def prepare_exp2_config(
    base_config,
    exp1_best_config,
    run_id,
    seed,
    experiments_root="output/experiments",
    debug=False,
    tracer=False,
    project_root=".",
    fixed_sample_count=8,
    max_train_files=None,
    max_valid_files=None,
    distill_loss_lambda=None,
    experiment_tag=None,
    experiment_note=None,
):
    base_config = Path(base_config)
    exp1_best_config = Path(exp1_best_config)
    if exp1_best_config.name == "best_candidate_raw.json":
        raise ValueError("Exp2 must consume best_seanet_config.json, not best_candidate_raw.json")
    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")
    if not exp1_best_config.exists():
        raise FileNotFoundError(f"Exp1 best_seanet_config.json not found: {exp1_best_config}")

    project_root = Path(project_root).resolve()
    run_dir = ensure_run_layout(Path(experiments_root) / run_id)
    for subdir in RUN_SUBDIRS:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    cfg = load_json(base_config)
    handoff = load_json(exp1_best_config)
    if handoff.get("candidate_id") != EXPECTED_CANDIDATE_ID:
        raise ValueError(
            f"Unexpected Exp1 candidate_id {handoff.get('candidate_id')!r}; expected {EXPECTED_CANDIDATE_ID!r}"
        )
    if handoff.get("handoff_schema") != "encoder_only_nas_v1":
        raise ValueError("Exp1 handoff must use handoff_schema=encoder_only_nas_v1")

    copied_base = _copy_to_dir(base_config, run_dir / "configs")
    copied_handoff = _copy_to_dir(exp1_best_config, run_dir / "configs", "best_seanet_config.json")

    train_files_path = run_dir / "artifacts" / "train_files.txt"
    valid_files_path = run_dir / "artifacts" / "valid_files.txt"
    fixed_sample_list_path = run_dir / "artifacts" / "fixed_sample_list.txt"
    train_stats = _normalize_or_copy_manifest(cfg["train_files"], train_files_path, project_root, max_train_files)
    valid_stats = _normalize_or_copy_manifest(cfg["valid_files"], valid_files_path, project_root, max_valid_files)
    fixed_stats = _build_fixed_sample_list(valid_files_path, fixed_sample_list_path, fixed_sample_count)

    cfg["seed"] = int(seed)
    cfg["results_folder"] = _path_for_config(run_dir / "checkpoints")
    cfg["nas_encoder_config"] = _path_for_config(copied_handoff)
    cfg["train_files"] = _path_for_config(train_files_path)
    cfg["valid_files"] = _path_for_config(valid_files_path)
    cfg["sample_rate"] = 16000
    cfg["strides"] = [8, 5, 4, 2]
    cfg["dimension"] = 1024
    cfg["n_q"] = 3
    cfg["codebook_size"] = 1024
    cfg["encoder_downsample_rate"] = 320
    cfg["latent_rate"] = 50
    cfg["exp2_scope"] = (
        "Train SCIT-Speech-Base from scratch with Exp1 NAS encoder architecture; "
        "RVQ codebooks and decoder are trained in Exp2."
    )
    cfg["exp1_handoff"] = {
        "config_path": _path_for_config(copied_handoff),
        "source_path": _path_for_config(exp1_best_config),
        "candidate_id": handoff.get("candidate_id"),
        "encoder_strides": handoff.get("encoder_strides"),
        "handoff_schema": handoff.get("handoff_schema"),
        "decoder_condition_note": (
            "Exp1 decoder_condition=frozen_teacher_decoder is handoff provenance only; "
            "Exp2 trains the decoder normally."
        ),
    }
    if distill_loss_lambda is not None:
        cfg["distill_loss_lambda"] = float(distill_loss_lambda)
        cfg["exp2_distill_loss_lambda_source"] = "override"
    else:
        cfg["exp2_distill_loss_lambda_source"] = "base_config"
    if experiment_tag:
        cfg["exp2_experiment_tag"] = str(experiment_tag)
    if experiment_note:
        cfg["exp2_experiment_note"] = str(experiment_note)

    mode = "tracer" if tracer else "debug" if debug else "formal"
    cfg = _apply_run_mode_overrides(cfg, mode)
    problems = validate_fixed_conditions(cfg)
    if problems:
        raise ValueError(f"Fixed interface conditions are invalid: {problems}")

    config_path = run_dir / "configs" / "scit_speech_base_config.json"
    write_json(config_path, cfg)
    data_split = {
        "run_id": run_id,
        "seed": int(seed),
        "mode": cfg["exp2_run_mode"],
        "base_config_copy": _path_for_config(copied_base),
        "exp1_best_config_copy": _path_for_config(copied_handoff),
        "train_files": train_stats,
        "valid_files": valid_stats,
        "fixed_sample_list": fixed_stats,
    }
    data_split_path = run_dir / "configs" / "data_split.json"
    write_json(data_split_path, data_split)

    command = (
        f"conda run -n speechtokenizer accelerate launch scripts/train_example.py "
        f"--config {_path_for_config(config_path)}"
    )
    command_path = run_dir / "commands" / "run_command.txt"
    write_text(command_path, command + "\n")

    write_text(
        run_dir / "reports" / "prepare_config.md",
        "\n".join(
            [
                "# Exp2 Config Preparation",
                "",
                f"- run_id: {run_id}",
                f"- mode: {cfg['exp2_run_mode']}",
                f"- config: {config_path}",
                f"- results_folder: {cfg['results_folder']}",
                f"- nas_encoder_config: {cfg['nas_encoder_config']}",
                f"- distill_loss_lambda: {cfg.get('distill_loss_lambda')}",
                f"- distill_loss_lambda_source: {cfg.get('exp2_distill_loss_lambda_source')}",
                f"- experiment_tag: {cfg.get('exp2_experiment_tag', '')}",
                f"- experiment_note: {cfg.get('exp2_experiment_note', '')}",
                "- note: this script prepares training inputs only; it does not train or evaluate the model.",
                "",
            ]
        ),
    )

    return PreparedExp2Config(
        run_dir=run_dir,
        config_path=config_path,
        command_path=command_path,
        data_split_path=data_split_path,
        train_files_path=train_files_path,
        valid_files_path=valid_files_path,
        fixed_sample_list_path=fixed_sample_list_path,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare the formal Exp2 SCIT-Speech-Base training config.")
    parser.add_argument("--base-config", default="config/spt_base_cfg.json")
    parser.add_argument("--exp1-best-config", default=DEFAULT_EXP1_BEST)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--experiments-root", default="output/experiments")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--tracer", action="store_true")
    parser.add_argument("--fixed-sample-count", type=int, default=8)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-valid-files", type=int, default=None)
    parser.add_argument("--distill-loss-lambda", type=float, default=None)
    parser.add_argument("--experiment-tag", default=None)
    parser.add_argument("--experiment-note", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    prepared = prepare_exp2_config(
        base_config=args.base_config,
        exp1_best_config=args.exp1_best_config,
        run_id=args.run_id,
        seed=args.seed,
        experiments_root=args.experiments_root,
        debug=args.debug,
        tracer=args.tracer,
        project_root=args.project_root,
        fixed_sample_count=args.fixed_sample_count,
        max_train_files=args.max_train_files,
        max_valid_files=args.max_valid_files,
        distill_loss_lambda=args.distill_loss_lambda,
        experiment_tag=args.experiment_tag,
        experiment_note=args.experiment_note,
    )
    print(prepared.config_path)


if __name__ == "__main__":
    main()
