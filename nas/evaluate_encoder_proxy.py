import argparse
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import torch
from einops import rearrange
from thop import profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import (
    collect_environment_dict,
    ensure_run_layout,
    format_environment_markdown,
    install_tee_logging,
    normalize_manifest,
    write_csv,
    write_json,
    write_text,
)
from speechtokenizer.model import SpeechTokenizer
from speechtokenizer.modules.seanet import SEANetDecoder as BaseSEANetDecoder
from speechtokenizer.trainer.dataset import audioDataset, get_dataloader
from speechtokenizer.trainer.loss import d_axis_distill_loss, mel_loss, recon_loss
from nas.SeaNet import SEANetEncoder
from nas.encoder_handoff import build_encoder_only_handoff_config
from nas.search_space import (
    OPERATOR_LIBRARY,
    normalize_candidate,
    sample_candidates,
    search_space_with_policy,
)
from nas.teacher_guided_proxy import (
    TEACHER_METRIC_KEYS,
    compute_teacher_guided_metrics,
    disabled_teacher_metrics,
    load_teacher_model,
    teacher_reference_dict,
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_base_interface(cfg):
    problems = []
    checks = {
        "sample_rate": 16000,
        "dimension": 1024,
        "n_q": 3,
        "codebook_size": 1024,
    }
    for key, expected in checks.items():
        if cfg.get(key) != expected:
            problems.append({"key": key, "expected": expected, "actual": cfg.get(key)})

    strides = cfg.get("strides")
    if not isinstance(strides, list) or len(strides) != 4:
        problems.append({"key": "strides", "expected": "list length 4", "actual": strides})
        return problems

    downsample = 1
    for stride in strides:
        downsample *= int(stride)
    if downsample != 320:
        problems.append({"key": "encoder_downsample_rate", "expected": 320, "actual": downsample})
    latent_rate = float(cfg.get("sample_rate", 0)) / float(downsample) if downsample else None
    if latent_rate != 50.0:
        problems.append({"key": "latent_rate", "expected": 50.0, "actual": latent_rate})
    return problems


def make_dataloader(manifest_path, cfg):
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line for line in f.readlines() if line.strip()]
    ds = audioDataset(
        file_list=lines,
        segment_size=cfg["segment_size"],
        sample_rate=cfg["sample_rate"],
        downsample_rate=320,
        valid=True,
    )
    return get_dataloader(ds, batch_size=1, shuffle=False, drop_last=False, num_workers=0), lines


def deterministic_model(cfg, seed):
    torch.manual_seed(int(seed))
    return SpeechTokenizer(cfg)


def build_candidate_model(candidate, cfg, seed, device, teacher_model=None, stage1_only=False):
    model = deterministic_model(cfg, seed).to(device)
    model.encoder = build_nas_encoder(candidate, cfg).to(device)
    if teacher_model is not None:
        model.transform = teacher_model.transform
        model.quantizer = teacher_model.quantizer
        model.decoder = teacher_model.decoder
    elif not stage1_only:
        model.decoder = build_geometry_matched_decoder(candidate, cfg).to(device)
    return model


def build_nas_encoder(candidate, cfg):
    return SEANetEncoder(
        channels=1,
        n_filters=int(candidate["n_filters"]),
        dimension=int(candidate["dimension"]),
        lstm=int(candidate["lstm"]),
        activation=candidate["activation"],
        compress=int(candidate["compress"]),
        layer_ops_list=list(candidate["layer_ops_list"]),
        layer_se_list=list(candidate["layer_se_list"]),
        ratios=list(candidate["seanet_ratios_arg"]),
        norm="weight_norm",
        causal=False,
        pad_mode="reflect",
        dilation_base=cfg.get("dilation_base", 2),
        residual_kernel_size=cfg.get("residual_kernel_size", 3),
        n_residual_layers=cfg.get("n_residual_layers", 1),
        bidirectional=cfg.get("bidirectional", False),
    )


def build_geometry_matched_decoder(candidate, cfg):
    return BaseSEANetDecoder(
        channels=1,
        n_filters=cfg.get("n_filters"),
        dimension=cfg.get("dimension"),
        ratios=list(candidate["decoder_strides"]),
        lstm=cfg.get("lstm_layers"),
        bidirectional=False,
        dilation_base=cfg.get("dilation_base", 2),
        residual_kernel_size=cfg.get("residual_kernel_size", 3),
        n_residual_layers=cfg.get("n_residual_layers", 1),
        activation=cfg.get("activation"),
    )


def count_params(module):
    return int(sum(p.numel() for p in module.parameters()))


def profile_encoder(module, device):
    module.eval()
    dummy = torch.randn(1, 1, 16000, device=device)
    with torch.no_grad():
        flops, thop_params = profile(module, inputs=(dummy,), verbose=False)
        warmup = module(dummy)
        if warmup.shape[1] != 1024:
            raise RuntimeError(f"latent dimension changed: got {list(warmup.shape)}")
        if warmup.shape[-1] != 50:
            raise RuntimeError(f"latent frame count changed for 1s input: got {list(warmup.shape)}")
        times = []
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(5):
            tic = time.perf_counter()
            module(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - tic)

    mean_time = sum(times) / len(times)
    return {
        "encoder_params": count_params(module),
        "thop_params": int(thop_params),
        "encoder_macs": int(flops),
        "encoder_macs_g": float(flops / 1e9),
        "encoder_rtf_mean": float(mean_time),
        "encoder_rtf_std": float((sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5),
        "interface_shape_1s": list(warmup.shape),
    }


def _finite_or_raise(name, value):
    if not math.isfinite(float(value)):
        raise RuntimeError(f"{name} is not finite: {value}")


def evaluate_model(candidate_id, model, dl, cfg, device, max_batches, teacher_model=None):
    model.to(device)
    model.eval()
    if teacher_model is not None:
        teacher_model.to(device)
        teacher_model.eval()
    total_sem = 0.0
    total_rec = 0.0
    total_mel = 0.0
    teacher_totals = {key: 0.0 for key in TEACHER_METRIC_KEYS}
    count = 0
    last_codes_shape = None
    last_out_shape = None

    with torch.inference_mode():
        for idx, batch in enumerate(dl):
            if idx >= max_batches:
                break
            x, semantic = batch
            x = x.unsqueeze(1).to(device)
            semantic = semantic.to(device)
            encoded = model.encoder(x)
            if encoded.shape[1] != cfg["dimension"]:
                raise RuntimeError(f"{candidate_id}: encoded dim mismatch {list(encoded.shape)}")
            if encoded.shape[-1] != x.shape[-1] // 320:
                raise RuntimeError(
                    f"{candidate_id}: latent frame mismatch, got {list(encoded.shape)}, input={list(x.shape)}"
                )
            transformed = model.transform(rearrange(encoded, "b d t -> b t d"))
            sem_loss = d_axis_distill_loss(transformed, semantic)
            quantized, codes, commit_loss, quantized_list = model.quantizer(encoded, n_q=cfg["n_q"], layers=[0])
            if codes.shape[0] != cfg["n_q"]:
                raise RuntimeError(f"{candidate_id}: codes layer mismatch {list(codes.shape)}")
            if teacher_model is not None:
                teacher_encoded = teacher_model.encoder(x)
                teacher_n_q = min(int(cfg["n_q"]), int(getattr(teacher_model, "n_q", cfg["n_q"])))
                teacher_quantized, teacher_codes, _, _ = teacher_model.quantizer(
                    teacher_encoded, n_q=teacher_n_q, layers=[0]
                )
                student_teacher_quantized, student_teacher_codes, _, _ = teacher_model.quantizer(
                    encoded, n_q=teacher_n_q, layers=[0]
                )
                teacher_metrics = compute_teacher_guided_metrics(
                    teacher_latent=teacher_encoded,
                    student_latent=encoded,
                    teacher_codes=teacher_codes,
                    student_codes=student_teacher_codes,
                    teacher_quantized=teacher_quantized,
                    student_quantized=student_teacher_quantized,
                )
                for name, value in teacher_metrics.items():
                    _finite_or_raise(name, value)
                    teacher_totals[name] += float(value)
            out = model.decoder(quantized)
            if out.shape != x.shape:
                raise RuntimeError(f"{candidate_id}: decoder output shape mismatch, got {list(out.shape)}, expected {list(x.shape)}")
            rec = recon_loss(x, out)
            mel = mel_loss(
                x,
                out,
                n_fft=cfg["n_fft"],
                num_mels=cfg["num_mels"],
                sample_rate=cfg["sample_rate"],
                hop_size=cfg["hop_size"],
                win_size=cfg["win_size"],
                fmin=cfg["fmin"],
                fmax=cfg["fmax"],
            )
            for name, value in [("semantic_proxy_loss", sem_loss), ("proxy_recon_l1", rec), ("proxy_mel_loss", mel)]:
                _finite_or_raise(name, value.item())
            total_sem += float(sem_loss.item())
            total_rec += float(rec.item())
            total_mel += float(mel.item())
            count += 1
            last_codes_shape = list(codes.shape)
            last_out_shape = list(out.shape)

    if count == 0:
        raise RuntimeError("no batches evaluated")
    return {
        "semantic_proxy_loss": total_sem / count,
        "proxy_recon_l1": total_rec / count,
        "proxy_mel_loss": total_mel / count,
        "evaluated_batches": count,
        "codes_shape": last_codes_shape,
        "output_shape": last_out_shape,
        **(
            {key: teacher_totals[key] / count for key in TEACHER_METRIC_KEYS}
            if teacher_model is not None
            else disabled_teacher_metrics()
        ),
    }


def pareto_frontier(rows):
    objective_keys = [
        "semantic_proxy_loss",
        "proxy_recon_l1",
        "proxy_mel_loss",
        "encoder_macs",
        "encoder_params",
        "encoder_rtf_mean",
    ]
    if any(any(key not in row or row.get(key) in ("", None) for key in objective_keys) for row in rows):
        objective_keys = ["semantic_proxy_loss", "encoder_macs", "encoder_params", "encoder_rtf_mean"]
    frontier = []
    for row in rows:
        dominated_by = []
        for other in rows:
            if other is row:
                continue
            no_worse = all(float(other[key]) <= float(row[key]) for key in objective_keys)
            strictly_better = any(float(other[key]) < float(row[key]) for key in objective_keys)
            if no_worse and strictly_better:
                dominated_by.append(other["candidate_id"])
        item = dict(row)
        item["dominated_by"] = ";".join(dominated_by)
        item["selected"] = False
        item["selection_reason"] = ""
        frontier.append(item)
    return [row for row in frontier if not row["dominated_by"]]


def flatten_row_for_csv(row):
    item = dict(row)
    for key in ["encoder_strides", "decoder_strides", "layer_ops_list", "layer_se_list", "codes_shape", "output_shape", "interface_shape_1s"]:
        if key in item and isinstance(item[key], (list, dict)):
            item[key] = json.dumps(item[key], ensure_ascii=False)
    return item


def write_operator_report(run_dir):
    lines = ["# NAS Operator Library", ""]
    for name, description in OPERATOR_LIBRARY.items():
        lines.append(f"- `{name}`: {description}")
    lines.extend(
        [
            "",
            "Constraints:",
            "",
            "- `count(skip) <= 2`",
            "- `skip` forces `use_se=false` because the current skip op is identity.",
            "",
        ]
    )
    write_text(run_dir / "reports" / "operator_library.md", "\n".join(lines))


def load_candidates_from_json(paths, search_space):
    candidates = []
    for idx, path in enumerate(paths):
        raw = load_json(path)
        raw.setdefault("candidate_id", f"candidate_json_{idx:06d}")
        raw.setdefault("search_mode", "candidate_json")
        raw.setdefault("seed", None)
        raw.setdefault("sample_index", idx)
        candidates.append(normalize_candidate(raw, search_space=search_space))
    return candidates


def generate_candidates(args, search_space):
    if args.candidate_json:
        return load_candidates_from_json(args.candidate_json, search_space), "candidate_json"
    if args.search_mode != "random":
        raise SystemExit(f"unsupported search mode for this implementation: {args.search_mode}")
    return sample_candidates(args.seed, args.num_candidates, search_space=search_space), "random"


def base_row(run_dir, cfg, seed, device, dl, max_batches, stage1_only, teacher_model=None):
    model = teacher_model if teacher_model is not None else deterministic_model(cfg, seed).to(device)
    profile_row = profile_encoder(model.encoder, device)
    metrics = {} if stage1_only else evaluate_model("hand_encoder", model, dl, cfg, device, max_batches, teacher_model)
    return {
        "run_id": run_dir.name,
        "candidate_id": "hand_encoder",
        "encoder_type": "hand-designed",
        "architecture_json": "configs/teacher_model_reference.json" if teacher_model is not None else "configs/spt_base_cfg.json",
        "encoder_strides": cfg.get("strides"),
        "decoder_strides": cfg.get("strides"),
        "decoder_condition": "frozen_teacher_decoder" if teacher_model is not None else "base_config_decoder",
        "n_filters": cfg.get("n_filters"),
        "compress": 2,
        "lstm": cfg.get("lstm_layers"),
        "activation": cfg.get("activation"),
        "layer_ops_list": "base_seanet_resnet_block",
        "layer_se_list": [],
        "valid_interface": True,
        "status": "completed",
        "error": "",
        **profile_row,
        **(disabled_teacher_metrics() if stage1_only else {}),
        **metrics,
    }


def candidate_row(run_dir, candidate, cfg, seed, device, dl, max_batches, stage1_only, teacher_model=None):
    cid = candidate["candidate_id"]
    torch.manual_seed(int(seed))
    model = build_candidate_model(candidate, cfg, seed, device, teacher_model=teacher_model, stage1_only=stage1_only)
    profile_row = profile_encoder(model.encoder, device)
    metrics = {} if stage1_only else evaluate_model(cid, model, dl, cfg, device, max_batches, teacher_model)
    return {
        "run_id": run_dir.name,
        "candidate_id": cid,
        "encoder_type": "nas-candidate",
        "architecture_json": f"artifacts/candidates/{cid}.json",
        "encoder_strides": candidate["encoder_strides"],
        "decoder_strides": cfg.get("strides") if teacher_model is not None else candidate["decoder_strides"],
        "decoder_condition": "frozen_teacher_decoder" if teacher_model is not None else candidate["decoder_condition"],
        "n_filters": candidate["n_filters"],
        "compress": candidate["compress"],
        "lstm": candidate["lstm"],
        "activation": candidate["activation"],
        "layer_ops_list": candidate["layer_ops_list"],
        "layer_se_list": candidate["layer_se_list"],
        "valid_interface": True,
        "status": "completed",
        "error": "",
        **profile_row,
        **(disabled_teacher_metrics() if stage1_only else {}),
        **metrics,
    }


def write_stage1_outputs(run_dir, rows, fieldnames):
    flat_rows = [flatten_row_for_csv(row) for row in rows]
    write_csv(run_dir / "metrics" / "nas_records.csv", flat_rows, fieldnames)
    write_json(run_dir / "metrics" / "nas_records.json", rows)
    write_csv(run_dir / "metrics" / "results.csv", flat_rows, fieldnames)
    write_json(run_dir / "metrics" / "results.json", {"stage1_only": True, "selected": None, "rows": rows})
    write_text(
        run_dir / "reports" / "summary.md",
        "\n".join(
            [
                "# Experiment 1 Stage-1 Summary",
                "",
                "- status: completed_stage1_only",
                f"- run_id: {run_dir.name}",
                "- scope: candidate generation, encoder interface validation, and encoder-only profiling.",
                "- no semantic proxy, reconstruction proxy, Pareto frontier, or best architecture was selected in this mode.",
                "",
            ]
        ),
    )
    write_json(
        run_dir / "reports" / "status.json",
        {
            "status": "completed_stage1_only",
            "run_id": run_dir.name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "selected_encoder": None,
        },
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-mode", choices=["random"], default="random")
    parser.add_argument("--num-candidates", "--candidate-limit", type=int, default=32)
    parser.add_argument("--search-space-config")
    parser.add_argument("--candidate-json", nargs="*")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--teacher-config")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--teacher-target", choices=["pre_rvq"], default="pre_rvq")
    parser.add_argument("--teacher-cache-mode", choices=["none", "disk"], default="none")
    return parser


def collect_file_list(run_dir, limit=200):
    files = []
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return files
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            files.append(str(path.relative_to(run_dir)))
            if len(files) >= limit:
                files.append(f"... truncated after {limit} files")
                break
    return files


def write_failure_artifacts(run_dir, exc, failed_step="evaluate_encoder_proxy"):
    run_dir = ensure_run_layout(run_dir)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    command = " ".join([sys.executable] + sys.argv)
    if isinstance(exc, SystemExit):
        reason = str(exc.code)
        trace = reason
    else:
        reason = f"{exc.__class__.__name__}: {exc}"
        trace = traceback.format_exc(limit=20)
    files = collect_file_list(run_dir)
    lines = [
        "# Failure Report",
        "",
        f"- failed_experiment: exp1",
        f"- run_id: {Path(run_dir).name}",
        f"- failed_step: {failed_step}",
        f"- timestamp: {timestamp}",
        f"- reason: {reason}",
        "",
        "## Actual Command",
        "",
        f"```text\n{command}\n```",
        "",
        "## Error Log Summary",
        "",
        f"```text\n{trace}\n```",
        "",
        "## Produced Files",
        "",
    ]
    lines.extend(f"- `{path}`" for path in files)
    lines.extend(
        [
            "",
            "## Major Deviation Reason",
            "",
            "The Exp1 proxy evaluator stopped before a completed NAS result could be produced.",
            "",
            "## Suggested Next Step",
            "",
            "Inspect the error summary, verify the fixed interface/data paths/candidate JSON, and rerun with a smoke-sized budget before launching a larger search.",
            "",
        ]
    )
    write_text(run_dir / "reports" / "failure_report.md", "\n".join(lines))
    write_json(
        run_dir / "reports" / "status.json",
        {
            "status": "aborted",
            "reason": reason,
            "failed_step": failed_step,
            "timestamp": timestamp,
            "run_id": Path(run_dir).name,
        },
    )


def run(args):

    run_dir = ensure_run_layout(args.run_dir)
    install_tee_logging(run_dir)
    write_text(run_dir / "commands" / "run_command.txt", " ".join([sys.executable] + sys.argv) + "\n")

    cfg = load_json(args.config)
    shutil.copy2(args.config, run_dir / "configs" / Path(args.config).name)

    fixed_problems = validate_base_interface(cfg)
    if fixed_problems:
        raise SystemExit(f"fixed interface mismatch: {fixed_problems}")

    env = collect_environment_dict(PROJECT_ROOT, extra_tools=["ffmpeg", "opusenc", "opusdec"])
    write_json(run_dir / "reports" / "environment.json", env)
    write_text(run_dir / "reports" / "environment.md", format_environment_markdown(env))

    search_space = load_json(args.search_space_config) if args.search_space_config else search_space_with_policy(
        args.seed, args.num_candidates, args.search_mode
    )
    write_json(run_dir / "configs" / "operator_library.json", OPERATOR_LIBRARY)
    write_json(run_dir / "configs" / "nas_search_space.json", search_space)
    write_json(
        run_dir / "configs" / "nas_runtime_config.json",
        {
            "seed": args.seed,
            "search_mode": args.search_mode,
            "num_candidates": args.num_candidates,
            "max_samples": args.max_samples,
            "max_batches": args.max_batches,
            "stage1_only": bool(args.stage1_only),
            "device_requested": args.device,
            "teacher_enabled": bool(args.teacher_config or args.teacher_checkpoint),
            "teacher_target": args.teacher_target,
            "teacher_cache_mode": args.teacher_cache_mode,
            "proxy_note": "Encoder-side NAS under a fixed index-transmission interface. Decoder is geometry-matched for reconstruction diagnostics only and is not searched.",
        },
    )
    write_operator_report(run_dir)

    candidates, actual_search_mode = generate_candidates(args, search_space)
    if not candidates:
        raise SystemExit("no valid NAS candidates generated")
    for candidate in candidates:
        write_json(run_dir / "artifacts" / "candidates" / f"{candidate['candidate_id']}.json", candidate)

    dl = None
    sample_lines = []
    if not args.stage1_only:
        if not args.manifest:
            raise SystemExit("--manifest is required unless --stage1-only is set")
        proxy_manifest = run_dir / "artifacts" / "train_subset_nas.txt"
        manifest_stats = normalize_manifest(args.manifest, proxy_manifest, PROJECT_ROOT, max_lines=args.max_samples)
        write_json(run_dir / "artifacts" / "proxy_manifest_normalization.json", manifest_stats)
        if manifest_stats["written"] == 0:
            raise SystemExit(f"no usable proxy samples after manifest normalization: {manifest_stats}")
        dl, sample_lines = make_dataloader(proxy_manifest, cfg)
        write_json(
            run_dir / "artifacts" / "proxy_sample_manifest.json",
            {
                "source_manifest": args.manifest,
                "normalized_manifest": str(proxy_manifest),
                "sample_count": len(sample_lines),
                "samples": [line.strip() for line in sample_lines],
            },
        )
    elif args.manifest:
        shutil.copy2(args.manifest, run_dir / "artifacts" / "source_manifest_for_stage1_only.txt")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    teacher_enabled = bool(args.teacher_config or args.teacher_checkpoint)
    if teacher_enabled and not (args.teacher_config and args.teacher_checkpoint):
        raise SystemExit("--teacher-config and --teacher-checkpoint must be provided together")
    teacher_model = load_teacher_model(args.teacher_config, args.teacher_checkpoint, device) if teacher_enabled else None
    teacher_ref = teacher_reference_dict(
        args.teacher_config,
        args.teacher_checkpoint,
        args.teacher_target,
        args.teacher_cache_mode,
        enabled=teacher_enabled,
    )
    write_json(run_dir / "configs" / "teacher_model_reference.json", teacher_ref)
    write_json(
        run_dir / "configs" / "teacher_guided_nas_config.json",
        {
            "teacher_guided": teacher_enabled,
            "teacher_target": args.teacher_target,
            "teacher_cache_mode": args.teacher_cache_mode,
            "metric_keys": TEACHER_METRIC_KEYS,
        },
    )
    write_text(
        run_dir / "reports" / "teacher_condition.md",
        "\n".join(
            [
                "# Teacher Condition",
                "",
                f"- teacher_enabled: {teacher_enabled}",
                f"- teacher_config: {args.teacher_config or ''}",
                f"- teacher_checkpoint: {args.teacher_checkpoint or ''}",
                f"- teacher_target: {args.teacher_target}",
                f"- teacher_cache_mode: {args.teacher_cache_mode}",
                "- teacher_role: frozen SpeechTokenizer encoder pre-RVQ latent anchor.",
                "- teacher_quantizer_role: frozen RVQ compatibility diagnostic only.",
                "- teacher_is_method_contribution: false",
                "",
            ]
        ),
    )
    interface_lines = ["# Encoder Interface Check", ""]
    decoder_lines = [
        "# Decoder Condition",
        "",
        "- Main NAS search target: transmitter-side encoder before latent Z.",
        "- Candidate decoder condition for reconstruction diagnostics: `geometry_matched_decoder`.",
        "- Candidate decoder stride schedule: `reverse(encoder_strides)`.",
        "- Candidate decoder ops, width, depth, activation, and LSTM settings are fixed to the base config and are not searched.",
        "",
    ]
    contamination_lines = [
        "# Decoder Contamination Check",
        "",
        "- The search space is defined in `configs/nas_search_space.json` and contains encoder macro/block choices only.",
        "- RVQ codebooks, M, K, L, payload accounting, and ChannelSim are not searched in this script.",
        "- Decoder geometry matching is used only to make reconstruction diagnostics well-defined for changed stride schedules.",
        "- Encoder complexity metrics report encoder-only params, MACs, and RTF.",
        "",
    ]

    fieldnames = [
        "run_id",
        "candidate_id",
        "encoder_type",
        "architecture_json",
        "encoder_strides",
        "decoder_strides",
        "decoder_condition",
        "n_filters",
        "compress",
        "lstm",
        "activation",
        "layer_ops_list",
        "layer_se_list",
        "valid_interface",
        "status",
        "semantic_proxy_loss",
        "proxy_recon_l1",
        "proxy_mel_loss",
        *TEACHER_METRIC_KEYS,
        "encoder_params",
        "encoder_macs",
        "encoder_macs_g",
        "thop_params",
        "encoder_rtf_mean",
        "encoder_rtf_std",
        "evaluated_batches",
        "codes_shape",
        "output_shape",
        "interface_shape_1s",
        "selection_semantic_penalty",
        "selection_recon_penalty",
        "selection_mel_penalty",
        "selection_quality_penalty",
        "selection_resource_score",
        "error",
    ]

    rows = []
    rows.append(base_row(run_dir, cfg, args.seed, device, dl, args.max_batches, args.stage1_only, teacher_model))
    interface_lines.append(f"- hand_encoder: shape_1s={rows[-1]['interface_shape_1s']}")

    candidate_errors = []
    repeated_errors = {}
    for candidate in candidates:
        cid = candidate["candidate_id"]
        try:
            row = candidate_row(run_dir, candidate, cfg, args.seed, device, dl, args.max_batches, args.stage1_only, teacher_model)
            rows.append(row)
            detail = f"- {cid}: encoder_strides={candidate['encoder_strides']}, shape_1s={row['interface_shape_1s']}"
            if not args.stage1_only:
                detail += f", codes_shape={row['codes_shape']}, output_shape={row['output_shape']}"
            interface_lines.append(detail)
        except Exception as exc:
            message = f"{exc.__class__.__name__}: {exc}"
            repeated_errors[message] = repeated_errors.get(message, 0) + 1
            candidate_errors.append(
                {
                    "candidate_id": cid,
                    "error": message,
                    "traceback": traceback.format_exc(limit=6),
                }
            )
            rows.append(
                {
                    "run_id": run_dir.name,
                    "candidate_id": cid,
                    "encoder_type": "nas-candidate",
                    "architecture_json": f"artifacts/candidates/{cid}.json",
                    "encoder_strides": candidate.get("encoder_strides"),
                    "decoder_strides": candidate.get("decoder_strides"),
                    "decoder_condition": candidate.get("decoder_condition"),
                    "n_filters": candidate.get("n_filters"),
                    "compress": candidate.get("compress"),
                    "lstm": candidate.get("lstm"),
                    "activation": candidate.get("activation"),
                    "layer_ops_list": candidate.get("layer_ops_list"),
                    "layer_se_list": candidate.get("layer_se_list"),
                    "valid_interface": False,
                    "status": "failed",
                    "error": message,
                }
            )
            if "out of memory" in message.lower() and repeated_errors[message] >= 2:
                raise
            if repeated_errors[message] >= 3:
                raise

    write_json(run_dir / "artifacts" / "candidate_errors.json", candidate_errors)
    write_text(run_dir / "reports" / "encoder_interface_check.md", "\n".join(interface_lines) + "\n")
    write_text(run_dir / "reports" / "decoder_condition.md", "\n".join(decoder_lines) + "\n")
    write_text(run_dir / "reports" / "decoder_contamination_check.md", "\n".join(contamination_lines) + "\n")

    if args.stage1_only:
        write_stage1_outputs(run_dir, rows, fieldnames)
        print(json.dumps({"status": "completed_stage1_only", "run_dir": str(run_dir)}, indent=2))
        return

    completed_nas = [
        row
        for row in rows
        if row.get("encoder_type") == "nas-candidate" and row.get("status") == "completed" and row.get("valid_interface")
    ]
    if not completed_nas:
        raise SystemExit("no completed NAS candidate; cannot compute Pareto frontier")

    frontier = pareto_frontier(completed_nas)
    if not frontier:
        raise SystemExit("no NAS candidate on Pareto frontier")

    hand_row = rows[0]
    hand_macs = hand_row["encoder_macs"]
    hand_params = hand_row["encoder_params"]
    hand_sem = hand_row["semantic_proxy_loss"]
    hand_recon = hand_row["proxy_recon_l1"]
    hand_mel = hand_row["proxy_mel_loss"]

    def log_ratio(value, reference):
        return math.log(max(float(value), 1e-12) / max(float(reference), 1e-12))

    def quality_excess(value, reference, margin=0.10):
        return max(0.0, log_ratio(value, reference) - math.log1p(margin))

    def selection_score(row):
        sem_penalty = quality_excess(row["semantic_proxy_loss"], hand_sem)
        recon_penalty = quality_excess(row["proxy_recon_l1"], hand_recon)
        mel_penalty = quality_excess(row["proxy_mel_loss"], hand_mel)
        quality_penalty = sem_penalty + 0.60 * recon_penalty + 0.60 * mel_penalty
        resource_score = (
            0.12 * log_ratio(row["encoder_macs"], hand_macs)
            + 0.12 * log_ratio(row["encoder_params"], hand_params)
            + 0.03 * log_ratio(row["encoder_rtf_mean"], hand_row["encoder_rtf_mean"])
        )
        row["selection_semantic_penalty"] = sem_penalty
        row["selection_recon_penalty"] = recon_penalty
        row["selection_mel_penalty"] = mel_penalty
        row["selection_quality_penalty"] = quality_penalty
        row["selection_resource_score"] = resource_score
        return quality_penalty + resource_score

    best = min(frontier, key=selection_score)
    best["selected"] = True
    best["selection_reason"] = "minimum balanced quality-constrained proxy score among multi-objective Pareto candidates"
    for item in frontier:
        if item["candidate_id"] != best["candidate_id"]:
            selection_score(item)
            item["selection_reason"] = "pareto candidate, not minimum balanced quality-constrained proxy score"

    best_candidate = next(c for c in candidates if c["candidate_id"] == best["candidate_id"])
    best_dir = run_dir / "artifacts" / "best_architecture"
    best_handoff_config = build_encoder_only_handoff_config(best_candidate, best, cfg)
    write_json(best_dir / "best_seanet_config.json", best_handoff_config)
    write_json(best_dir / "best_candidate_raw.json", best_candidate)

    flat_rows = [flatten_row_for_csv(row) for row in rows]
    write_csv(run_dir / "metrics" / "encoder_proxy_results.csv", flat_rows, fieldnames)
    write_json(run_dir / "metrics" / "encoder_proxy_results.json", rows)
    write_csv(run_dir / "metrics" / "teacher_alignment.csv", flat_rows, fieldnames)
    write_json(run_dir / "metrics" / "teacher_alignment.json", rows)
    write_csv(run_dir / "metrics" / "rvq_compatibility.csv", flat_rows, fieldnames)
    write_json(run_dir / "metrics" / "rvq_compatibility.json", rows)
    write_csv(run_dir / "metrics" / "nas_records.csv", flat_rows, fieldnames)
    write_json(run_dir / "metrics" / "nas_records.json", rows)
    write_csv(run_dir / "metrics" / "results.csv", flat_rows, fieldnames)
    write_json(run_dir / "metrics" / "results.json", {"selected": best, "rows": rows, "candidate_errors": candidate_errors})
    write_csv(
        run_dir / "metrics" / "pareto_frontier.csv",
        [flatten_row_for_csv(row) for row in frontier],
        fieldnames + ["dominated_by", "selected", "selection_reason"],
    )
    write_json(run_dir / "metrics" / "pareto_frontier.json", frontier)

    selection_md = [
        "# Pareto Selection",
        "",
        f"- selected_candidate: {best['candidate_id']}",
        "- selected_config: artifacts/best_architecture/best_seanet_config.json",
        "- raw_selected_candidate: artifacts/best_architecture/best_candidate_raw.json",
        "- pareto_objectives: semantic_proxy_loss, proxy_recon_l1, proxy_mel_loss, encoder_macs, encoder_params, encoder_rtf_mean",
        "- selection_score: balanced quality-constrained score with 10% semantic/recon/mel margins plus MAC/param/RTF log ratios",
        f"- hand_encoder_params: {hand_params}",
        f"- hand_encoder_macs: {hand_macs}",
        f"- hand_semantic_proxy_loss: {hand_sem}",
        f"- hand_proxy_recon_l1: {hand_recon}",
        f"- hand_proxy_mel_loss: {hand_mel}",
        f"- selected_encoder_params: {best['encoder_params']}",
        f"- selected_encoder_macs: {best['encoder_macs']}",
        f"- selected_semantic_proxy_loss: {best['semantic_proxy_loss']}",
        f"- selected_proxy_recon_l1: {best['proxy_recon_l1']}",
        f"- selected_proxy_mel_loss: {best['proxy_mel_loss']}",
        f"- selected_encoder_strides: {best['encoder_strides']}",
        f"- selected_decoder_strides: {best['decoder_strides']}",
        f"- decoder_condition: {best['decoder_condition']}",
        "",
        "This is a constrained proxy NAS result. It must not be presented as a fully trained SCIT-Speech result.",
        "",
    ]
    write_text(run_dir / "reports" / "pareto_selection.md", "\n".join(selection_md))
    write_text(run_dir / "reports" / "teacher_guided_selection.md", "\n".join(selection_md))

    completed_count = sum(1 for row in rows if row.get("encoder_type") == "nas-candidate" and row.get("status") == "completed")
    failed_count = sum(1 for row in rows if row.get("encoder_type") == "nas-candidate" and row.get("status") == "failed")
    summary_lines = [
        "# Experiment 1 Summary",
        "",
        "- status: completed",
        f"- run_id: {run_dir.name}",
        f"- search_mode: {actual_search_mode}",
        f"- seed: {args.seed}",
        f"- num_requested: {args.num_candidates}",
        f"- num_generated: {len(candidates)}",
        f"- num_completed: {completed_count}",
        f"- num_failed: {failed_count}",
        f"- proxy_samples: {len(sample_lines)}",
        f"- evaluated_batches_per_candidate: {args.max_batches}",
        f"- selected_encoder: {best['candidate_id']}",
        f"- selected_encoder_strides: {best['encoder_strides']}",
        f"- selected_decoder_strides: {best['decoder_strides']}",
        f"- decoder_condition: {best['decoder_condition']}",
        "- scope: transmitter-side pre-latent encoder proxy search only.",
        "- fixed interface: sample_rate=16000, prod(encoder_strides)=320, latent_rate=50, latent_dimension=1024, M=3, K=1024.",
        "- decoder contamination: decoder ops/width/depth/activation are not searched; geometry matching is only used for reconstruction diagnostics.",
        "- limitation: proxy metrics are not final SCIT-Speech training results.",
        "- metrics: metrics/encoder_proxy_results.csv, metrics/pareto_frontier.csv",
        "- best_architecture: artifacts/best_architecture/best_seanet_config.json",
        "- raw_selected_candidate: artifacts/best_architecture/best_candidate_raw.json",
        "",
    ]
    write_text(run_dir / "reports" / "summary.md", "\n".join(summary_lines))
    write_json(
        run_dir / "reports" / "status.json",
        {
            "status": "completed",
            "run_id": run_dir.name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "selected_encoder": best["candidate_id"],
            "best_architecture": "artifacts/best_architecture/best_seanet_config.json",
            "raw_selected_candidate": "artifacts/best_architecture/best_candidate_raw.json",
            "num_generated": len(candidates),
            "num_completed": completed_count,
            "num_failed": failed_count,
        },
    )
    write_text(
        run_dir / "reports" / "agent_handoff.md",
        "\n".join(
            [
                "# Exp1 NAS Agent Handoff",
                "",
                "- status: completed",
                f"- selected_encoder: {best['candidate_id']}",
                "- wrote configs, commands, logs, metrics, artifacts, and reports inside this run directory.",
                "- downstream Exp2 may use artifacts/best_architecture/best_seanet_config.json as the NAS encoder proxy route.",
                "- raw selected NAS search item is preserved in artifacts/best_architecture/best_candidate_raw.json.",
                "",
            ]
        ),
    )
    write_json(run_dir / "reports" / "agent_status.json", {"agent": "Exp1 NAS Agent", "status": "completed", "run_id": run_dir.name})
    print(json.dumps({"status": "completed", "selected": best["candidate_id"], "run_dir": str(run_dir)}, indent=2))


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except SystemExit as exc:
        if exc.code not in (0, None):
            write_failure_artifacts(args.run_dir, exc)
        raise
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        write_failure_artifacts(args.run_dir, exc)
        raise


if __name__ == "__main__":
    main()
