"""Four-stage Exp1 encoder-side NAS runner.

Stages:
1. Generate and profile a large random candidate set.
2. Short-distill and proxy-evaluate the top-k profile candidates.
3. Short-distill and refined proxy-evaluate the top-k proxy candidates.
4. Short-distill final candidates, export Pareto and best config.

This script keeps the communication interface fixed and only searches the
transmitter-side encoder before latent Z. The pretrained SpeechTokenizer
teacher provides the frozen transform, RVQ, and decoder when teacher guidance is
enabled.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

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
from nas.evaluate_encoder_proxy import (
    base_row,
    build_candidate_model,
    build_nas_encoder,
    evaluate_model,
    flatten_row_for_csv,
    load_json,
    make_dataloader,
    profile_encoder,
    validate_base_interface,
    write_failure_artifacts,
    write_operator_report,
)
from nas.encoder_handoff import build_encoder_only_handoff_config
from nas.search_space import OPERATOR_LIBRARY, sample_candidates, search_space_with_policy
from nas.teacher_guided_proxy import (
    TEACHER_METRIC_KEYS,
    disabled_teacher_metrics,
    load_teacher_model,
    teacher_reference_dict,
)
from nas.validate_short_distill import train_encoder


DISTILL_METRIC_KEYS = [
    "distill_enabled",
    "distill_steps",
    "train_mean_latent_smooth_l1",
    "train_mean_cosine_distance",
    "train_mean_temporal_delta_loss",
    "train_mean_total",
    "train_last_latent_smooth_l1",
    "train_last_cosine_distance",
    "train_last_temporal_delta_loss",
    "train_last_total",
]


BASE_FIELDNAMES = [
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
    *DISTILL_METRIC_KEYS,
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
    "error",
]

STAGE_FIELDNAMES = BASE_FIELDNAMES + [
    "stage",
    "stage_rank",
    "stage_score",
    "stage_semantic_penalty",
    "stage_teacher_penalty",
    "stage_rvq_penalty",
    "stage_recon_penalty",
    "stage_mel_penalty",
    "stage_quality_penalty",
    "stage_resource_score",
    "selected_for_next_stage",
    "stage_selection_reason",
]

PARETO_FIELDNAMES = STAGE_FIELDNAMES + ["dominated_by", "selected", "selection_reason"]


@dataclass(frozen=True)
class SelectionConfig:
    mode: str
    quality_margin_sem: float
    quality_margin_recon: float
    quality_margin_mel: float
    quality_margin_latent: float
    quality_margin_delta: float
    quality_margin_cos: float
    weight_sem: float
    weight_recon: float
    weight_mel: float
    weight_latent: float
    weight_cos: float
    weight_delta: float
    weight_code: float
    weight_qfeat: float
    weight_macs: float
    weight_params: float
    weight_rtf: float


def project_row(row: Dict[str, Any], fieldnames: Iterable[str]) -> Dict[str, Any]:
    return {key: row.get(key, "") for key in fieldnames}


def csv_rows(rows: List[Dict[str, Any]], fieldnames: Iterable[str]) -> List[Dict[str, Any]]:
    return [project_row(flatten_row_for_csv(row), fieldnames) for row in rows]


def write_stage_table(run_dir: Path, stage_name: str, rows: List[Dict[str, Any]]) -> None:
    write_json(run_dir / "metrics" / f"{stage_name}.json", rows)
    write_csv(run_dir / "metrics" / f"{stage_name}.csv", csv_rows(rows, STAGE_FIELDNAMES), STAGE_FIELDNAMES)


def log_ratio(value: float, reference: float) -> float:
    return math.log(max(float(value), 1e-12) / max(float(reference), 1e-12))


def log_quality_excess(value: float, reference: float, relative_margin: float) -> float:
    """Penalty for being worse than the hand-designed reference beyond a margin."""

    allowed = math.log1p(max(float(relative_margin), 0.0))
    return max(0.0, log_ratio(value, reference) - allowed)


def numeric_metric(row: Dict[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def profile_score(row: Dict[str, Any], hand_row: Dict[str, Any]) -> float:
    return (
        0.45 * log_ratio(row["encoder_macs"], hand_row["encoder_macs"])
        + 0.45 * log_ratio(row["encoder_params"], hand_row["encoder_params"])
        + 0.10 * log_ratio(row["encoder_rtf_mean"], hand_row["encoder_rtf_mean"])
    )


def proxy_penalty_breakdown(row: Dict[str, Any], hand_row: Dict[str, Any], cfg: SelectionConfig) -> Dict[str, float]:
    sem = log_quality_excess(row["semantic_proxy_loss"], hand_row["semantic_proxy_loss"], cfg.quality_margin_sem)
    recon = log_quality_excess(row["proxy_recon_l1"], hand_row["proxy_recon_l1"], cfg.quality_margin_recon)
    mel = log_quality_excess(row["proxy_mel_loss"], hand_row["proxy_mel_loss"], cfg.quality_margin_mel)
    teacher_penalty = 0.0
    rvq_penalty = 0.0
    latent = numeric_metric(row, "teacher_latent_smooth_l1")
    latent_ref = numeric_metric(hand_row, "teacher_latent_smooth_l1")
    cos = numeric_metric(row, "teacher_latent_cosine_distance")
    cos_ref = numeric_metric(hand_row, "teacher_latent_cosine_distance")
    delta = numeric_metric(row, "teacher_temporal_delta_loss")
    delta_ref = numeric_metric(hand_row, "teacher_temporal_delta_loss")
    agreement = numeric_metric(row, "rvq_code_agreement")
    agreement_ref = numeric_metric(hand_row, "rvq_code_agreement")
    qfeat = numeric_metric(row, "rvq_quantized_feature_l1")
    qfeat_ref = numeric_metric(hand_row, "rvq_quantized_feature_l1")

    if latent is not None and latent_ref is not None:
        teacher_penalty += cfg.weight_latent * log_quality_excess(latent, latent_ref, cfg.quality_margin_latent)
    if cos is not None and cos_ref is not None:
        teacher_penalty += cfg.weight_cos * max(0.0, cos - cos_ref - cfg.quality_margin_cos)
    if delta is not None and delta_ref is not None:
        teacher_penalty += cfg.weight_delta * log_quality_excess(delta, delta_ref, cfg.quality_margin_delta)
    if agreement is not None and agreement_ref is not None:
        rvq_penalty += cfg.weight_code * max(0.0, agreement_ref - agreement)
    if qfeat is not None and qfeat_ref is not None:
        rvq_penalty += cfg.weight_qfeat * log_quality_excess(qfeat, qfeat_ref, cfg.quality_margin_latent)

    quality = cfg.weight_sem * sem + cfg.weight_recon * recon + cfg.weight_mel * mel + teacher_penalty + rvq_penalty
    resources = (
        cfg.weight_macs * log_ratio(row["encoder_macs"], hand_row["encoder_macs"])
        + cfg.weight_params * log_ratio(row["encoder_params"], hand_row["encoder_params"])
        + cfg.weight_rtf * log_ratio(row["encoder_rtf_mean"], hand_row["encoder_rtf_mean"])
    )
    return {
        "stage_semantic_penalty": sem,
        "stage_teacher_penalty": teacher_penalty,
        "stage_rvq_penalty": rvq_penalty,
        "stage_recon_penalty": recon,
        "stage_mel_penalty": mel,
        "stage_quality_penalty": quality,
        "stage_resource_score": resources,
        "stage_score": quality + resources,
    }


def proxy_score(row: Dict[str, Any], hand_row: Dict[str, Any], cfg: SelectionConfig) -> float:
    return proxy_penalty_breakdown(row, hand_row, cfg)["stage_score"]


def pareto_frontier_by_objectives(rows: List[Dict[str, Any]], objectives: Iterable[str]) -> List[Dict[str, Any]]:
    objectives = list(objectives)
    frontier = []
    for row in rows:
        dominated_by = []
        for other in rows:
            if other is row:
                continue
            no_worse = all(float(other[obj]) <= float(row[obj]) for obj in objectives)
            strictly_better = any(float(other[obj]) < float(row[obj]) for obj in objectives)
            if no_worse and strictly_better:
                dominated_by.append(other["candidate_id"])
        item = dict(row)
        item["dominated_by"] = ";".join(dominated_by)
        item["selected"] = False
        item["selection_reason"] = ""
        frontier.append(item)
    return [row for row in frontier if not row["dominated_by"]]


MODE_DEFAULTS = {
    "lite": {
        "min_n_filters": 16,
        "max_skip_blocks": 2,
        "quality_margin_sem": 0.20,
        "quality_margin_recon": 0.20,
        "quality_margin_mel": 0.20,
        "quality_margin_latent": 0.20,
        "quality_margin_delta": 0.20,
        "quality_margin_cos": 0.10,
        "weight_sem": 1.0,
        "weight_recon": 0.30,
        "weight_mel": 0.30,
        "weight_latent": 1.0,
        "weight_cos": 0.40,
        "weight_delta": 0.40,
        "weight_code": 0.60,
        "weight_qfeat": 0.40,
        "weight_macs": 0.25,
        "weight_params": 0.25,
        "weight_rtf": 0.05,
    },
    "balanced": {
        "min_n_filters": 24,
        "max_skip_blocks": 1,
        "quality_margin_sem": 0.10,
        "quality_margin_recon": 0.10,
        "quality_margin_mel": 0.10,
        "quality_margin_latent": 0.10,
        "quality_margin_delta": 0.10,
        "quality_margin_cos": 0.05,
        "weight_sem": 1.0,
        "weight_recon": 0.60,
        "weight_mel": 0.60,
        "weight_latent": 1.2,
        "weight_cos": 0.50,
        "weight_delta": 0.50,
        "weight_code": 0.80,
        "weight_qfeat": 0.50,
        "weight_macs": 0.12,
        "weight_params": 0.12,
        "weight_rtf": 0.03,
    },
    "quality": {
        "min_n_filters": 32,
        "max_skip_blocks": 1,
        "quality_margin_sem": 0.05,
        "quality_margin_recon": 0.05,
        "quality_margin_mel": 0.05,
        "quality_margin_latent": 0.05,
        "quality_margin_delta": 0.05,
        "quality_margin_cos": 0.025,
        "weight_sem": 1.2,
        "weight_recon": 0.80,
        "weight_mel": 0.80,
        "weight_latent": 1.5,
        "weight_cos": 0.70,
        "weight_delta": 0.70,
        "weight_code": 1.0,
        "weight_qfeat": 0.70,
        "weight_macs": 0.08,
        "weight_params": 0.08,
        "weight_rtf": 0.02,
    },
}


def effective_value(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return MODE_DEFAULTS[args.selection_mode][name]


def build_selection_config(args: argparse.Namespace) -> SelectionConfig:
    return SelectionConfig(
        mode=args.selection_mode,
        quality_margin_sem=float(effective_value(args, "quality_margin_sem")),
        quality_margin_recon=float(effective_value(args, "quality_margin_recon")),
        quality_margin_mel=float(effective_value(args, "quality_margin_mel")),
        quality_margin_latent=float(effective_value(args, "quality_margin_latent")),
        quality_margin_delta=float(effective_value(args, "quality_margin_delta")),
        quality_margin_cos=float(effective_value(args, "quality_margin_cos")),
        weight_sem=float(effective_value(args, "weight_sem")),
        weight_recon=float(effective_value(args, "weight_recon")),
        weight_mel=float(effective_value(args, "weight_mel")),
        weight_latent=float(effective_value(args, "weight_latent")),
        weight_cos=float(effective_value(args, "weight_cos")),
        weight_delta=float(effective_value(args, "weight_delta")),
        weight_code=float(effective_value(args, "weight_code")),
        weight_qfeat=float(effective_value(args, "weight_qfeat")),
        weight_macs=float(effective_value(args, "weight_macs")),
        weight_params=float(effective_value(args, "weight_params")),
        weight_rtf=float(effective_value(args, "weight_rtf")),
    )


def apply_search_policy_overrides(search_space: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    min_n_filters = int(effective_value(args, "min_n_filters"))
    max_skip_blocks = int(effective_value(args, "max_skip_blocks"))
    n_filters = [value for value in search_space["macro_space"]["n_filters"] if int(value) >= min_n_filters]
    if not n_filters:
        raise SystemExit(f"selection policy leaves no n_filters values: min_n_filters={min_n_filters}")
    search_space["macro_space"]["n_filters"] = n_filters
    search_space["block_space"]["constraints"]["max_skip_blocks"] = max_skip_blocks
    search_space["search_policy"]["selection_mode"] = args.selection_mode
    search_space["search_policy"]["min_n_filters"] = min_n_filters
    search_space["search_policy"]["max_skip_blocks"] = max_skip_blocks
    return search_space


def completed_nas_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("encoder_type") == "nas-candidate" and row.get("status") == "completed" and row.get("valid_interface")
    ]


def rank_and_select(
    rows: List[Dict[str, Any]],
    hand_row: Dict[str, Any],
    top_k: int,
    score_fn: Callable[[Dict[str, Any], Dict[str, Any]], float],
    reason: str,
    breakdown_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, float]]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    candidates = completed_nas_rows(rows)
    for row in rows:
        row["stage_score"] = ""
        row["stage_rank"] = ""
        row["selected_for_next_stage"] = False
        row["stage_selection_reason"] = ""
        row["stage_semantic_penalty"] = ""
        row["stage_teacher_penalty"] = ""
        row["stage_rvq_penalty"] = ""
        row["stage_recon_penalty"] = ""
        row["stage_mel_penalty"] = ""
        row["stage_quality_penalty"] = ""
        row["stage_resource_score"] = ""

    ranked = sorted(candidates, key=lambda row: score_fn(row, hand_row))
    selected_ids = []
    for rank, row in enumerate(ranked, start=1):
        if breakdown_fn:
            row.update(breakdown_fn(row, hand_row))
        else:
            row["stage_score"] = score_fn(row, hand_row)
        row["stage_rank"] = rank
        row["selected_for_next_stage"] = rank <= min(top_k, len(ranked))
        row["stage_selection_reason"] = reason if row["selected_for_next_stage"] else "not in top-k"
        if row["selected_for_next_stage"]:
            selected_ids.append(row["candidate_id"])
    return ranked, selected_ids


def stage1_candidate_profile_row(run_dir: Path, candidate: Dict[str, Any], cfg: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    cid = candidate["candidate_id"]
    encoder = build_nas_encoder(candidate, cfg).to(device)
    profile_row = profile_encoder(encoder, device)
    return {
        "run_id": run_dir.name,
        "candidate_id": cid,
        "encoder_type": "nas-candidate",
        "architecture_json": f"artifacts/candidates/{cid}.json",
        "encoder_strides": candidate["encoder_strides"],
        "decoder_strides": candidate["decoder_strides"],
        "decoder_condition": candidate["decoder_condition"],
        "n_filters": candidate["n_filters"],
        "compress": candidate["compress"],
        "lstm": candidate["lstm"],
        "activation": candidate["activation"],
        "layer_ops_list": candidate["layer_ops_list"],
        "layer_se_list": candidate["layer_se_list"],
        "valid_interface": True,
        "status": "completed",
        "error": "",
        "stage": "stage1_profile",
        **profile_row,
    }


def disabled_distill_metrics() -> Dict[str, Any]:
    return {
        "distill_enabled": False,
        "distill_steps": 0,
        "train_mean_latent_smooth_l1": "",
        "train_mean_cosine_distance": "",
        "train_mean_temporal_delta_loss": "",
        "train_mean_total": "",
        "train_last_latent_smooth_l1": "",
        "train_last_cosine_distance": "",
        "train_last_temporal_delta_loss": "",
        "train_last_total": "",
    }


def candidate_row_with_optional_distill(
    run_dir: Path,
    stage: str,
    candidate: Dict[str, Any],
    cfg: Dict[str, Any],
    seed: int,
    device: torch.device,
    dl,
    max_batches: int,
    teacher_model,
    distill_steps: int,
    distill_lr: float,
    distill_weight_decay: float,
    distill_latent_weight: float,
    distill_cosine_weight: float,
    distill_delta_weight: float,
    distill_grad_clip: float,
    distill_log_every: int,
) -> Dict[str, Any]:
    cid = candidate["candidate_id"]
    torch.manual_seed(int(seed))
    model = build_candidate_model(candidate, cfg, seed, device, teacher_model=teacher_model, stage1_only=False)
    profile_row = profile_encoder(model.encoder, device)
    distill_metrics = disabled_distill_metrics()
    if distill_steps > 0:
        if teacher_model is None:
            raise SystemExit("short distillation requires --teacher-config and --teacher-checkpoint")
        train_stats = train_encoder(
            encoder=model.encoder,
            teacher_model=teacher_model,
            loader=dl,
            device=device,
            steps=distill_steps,
            lr=distill_lr,
            weight_decay=distill_weight_decay,
            latent_weight=distill_latent_weight,
            cosine_weight=distill_cosine_weight,
            delta_weight=distill_delta_weight,
            grad_clip=distill_grad_clip,
            log_every=distill_log_every,
            candidate_id=f"{stage}/{cid}",
        )
        distill_metrics = {
            "distill_enabled": True,
            "distill_steps": distill_steps,
            **train_stats,
        }
    metrics = evaluate_model(cid, model, dl, cfg, device, max_batches, teacher_model)
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
        "stage": stage,
        **profile_row,
        **distill_metrics,
        **metrics,
    }


def failed_candidate_row(run_dir: Path, candidate: Dict[str, Any], stage: str, exc: BaseException) -> Dict[str, Any]:
    return {
        "run_id": run_dir.name,
        "candidate_id": candidate.get("candidate_id"),
        "encoder_type": "nas-candidate",
        "architecture_json": f"artifacts/candidates/{candidate.get('candidate_id')}.json",
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
        "error": f"{exc.__class__.__name__}: {exc}",
        "stage": stage,
    }


def evaluate_candidates(
    run_dir: Path,
    stage: str,
    candidates: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    seed: int,
    device: torch.device,
    dl,
    max_batches: int,
    stage1_only: bool,
    teacher_model=None,
    distill_steps: int = 0,
    distill_lr: float = 1e-4,
    distill_weight_decay: float = 0.0,
    distill_latent_weight: float = 1.0,
    distill_cosine_weight: float = 1.0,
    distill_delta_weight: float = 0.5,
    distill_grad_clip: float = 1.0,
    distill_log_every: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = []
    errors = []
    repeated_errors: Dict[str, int] = {}
    for idx, candidate in enumerate(candidates, start=1):
        cid = candidate["candidate_id"]
        print(f"[{stage}] {idx}/{len(candidates)} {cid}", flush=True)
        try:
            if stage1_only:
                row = stage1_candidate_profile_row(run_dir, candidate, cfg, device)
                row.update(disabled_teacher_metrics())
                row.update(disabled_distill_metrics())
            else:
                row = candidate_row_with_optional_distill(
                    run_dir,
                    stage,
                    candidate,
                    cfg,
                    seed,
                    device,
                    dl,
                    max_batches,
                    teacher_model,
                    distill_steps,
                    distill_lr,
                    distill_weight_decay,
                    distill_latent_weight,
                    distill_cosine_weight,
                    distill_delta_weight,
                    distill_grad_clip,
                    distill_log_every,
                )
            rows.append(row)
        except Exception as exc:
            message = f"{exc.__class__.__name__}: {exc}"
            repeated_errors[message] = repeated_errors.get(message, 0) + 1
            errors.append({"stage": stage, "candidate_id": cid, "error": message, "traceback": traceback.format_exc(limit=8)})
            rows.append(failed_candidate_row(run_dir, candidate, stage, exc))
            if "out of memory" in message.lower() and repeated_errors[message] >= 2:
                raise
            if repeated_errors[message] >= 3:
                raise
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return rows, errors


def write_selected_candidates(run_dir: Path, stage_name: str, selected: List[Dict[str, Any]]) -> None:
    stage_dir = run_dir / "artifacts" / "stages"
    write_json(stage_dir / f"{stage_name}_selected_candidates.json", selected)
    write_text(
        stage_dir / f"{stage_name}_selected_candidates.txt",
        "\n".join(candidate["candidate_id"] for candidate in selected) + ("\n" if selected else ""),
    )


def prepare_stage_dataloader(run_dir: Path, stage: str, manifest: str, cfg: Dict[str, Any], max_samples: int):
    manifest_out = run_dir / "artifacts" / "stages" / f"{stage}_subset.txt"
    stats = normalize_manifest(manifest, manifest_out, PROJECT_ROOT, max_lines=max_samples)
    write_json(run_dir / "artifacts" / "stages" / f"{stage}_manifest_normalization.json", stats)
    if stats["written"] == 0:
        raise SystemExit(f"{stage}: no usable proxy samples after manifest normalization: {stats}")
    dl, sample_lines = make_dataloader(manifest_out, cfg)
    write_json(
        run_dir / "artifacts" / "stages" / f"{stage}_sample_manifest.json",
        {
            "source_manifest": manifest,
            "normalized_manifest": str(manifest_out),
            "sample_count": len(sample_lines),
            "samples": [line.strip() for line in sample_lines],
        },
    )
    return dl, sample_lines


def write_static_reports(run_dir: Path) -> None:
    decoder_lines = [
        "# Decoder Condition",
        "",
        "- Main NAS search target: transmitter-side encoder before latent Z.",
        "- Teacher-guided proxy condition: frozen pretrained SpeechTokenizer transform/RVQ/decoder.",
        "- Candidate decoder ops, width, depth, activation, and LSTM settings are not searched.",
        "- When teacher guidance is enabled, candidate encoders are evaluated by plugging into the frozen pretrained downstream components.",
        "",
    ]
    contamination_lines = [
        "# Decoder Contamination Check",
        "",
        "- The staged search space contains encoder macro/block choices only.",
        "- RVQ codebooks, M, K, L, payload accounting, and ChannelSim are not searched in this script.",
        "- Short distillation updates only the NAS candidate encoder.",
        "- Proxy reconstruction uses the frozen pretrained SpeechTokenizer decoder when teacher guidance is enabled.",
        "- Encoder complexity metrics report encoder-only params, MACs, and RTF.",
        "",
    ]
    write_text(run_dir / "reports" / "decoder_condition.md", "\n".join(decoder_lines))
    write_text(run_dir / "reports" / "decoder_contamination_check.md", "\n".join(contamination_lines))


def write_interface_report(run_dir: Path, rows_by_stage: Dict[str, List[Dict[str, Any]]]) -> None:
    lines = ["# Encoder Interface Check", ""]
    for stage, rows in rows_by_stage.items():
        lines.append(f"## {stage}")
        for row in rows:
            if row.get("status") != "completed":
                lines.append(f"- {row.get('candidate_id')}: failed, error={row.get('error')}")
                continue
            detail = f"- {row.get('candidate_id')}: shape_1s={row.get('interface_shape_1s')}"
            if row.get("encoder_strides"):
                detail += f", encoder_strides={row.get('encoder_strides')}"
            if row.get("codes_shape"):
                detail += f", codes_shape={row.get('codes_shape')}, output_shape={row.get('output_shape')}"
            lines.append(detail)
        lines.append("")
    write_text(run_dir / "reports" / "encoder_interface_check.md", "\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stage1-candidates", type=int, default=2048)
    parser.add_argument("--stage2-top-k", type=int, default=256)
    parser.add_argument("--stage3-top-k", type=int, default=32)
    parser.add_argument("--final-top-k", type=int, default=8)
    parser.add_argument("--stage2-max-samples", type=int, default=4)
    parser.add_argument("--stage2-max-batches", type=int, default=4)
    parser.add_argument("--stage3-max-samples", type=int, default=8)
    parser.add_argument("--stage3-max-batches", type=int, default=8)
    parser.add_argument("--final-max-samples", type=int, default=16)
    parser.add_argument("--final-max-batches", type=int, default=16)
    parser.add_argument("--selection-mode", choices=sorted(MODE_DEFAULTS), default="balanced")
    parser.add_argument("--min-n-filters", type=int, default=None)
    parser.add_argument("--max-skip-blocks", type=int, default=None)
    parser.add_argument("--quality-margin-sem", type=float, default=None)
    parser.add_argument("--quality-margin-recon", type=float, default=None)
    parser.add_argument("--quality-margin-mel", type=float, default=None)
    parser.add_argument("--quality-margin-latent", type=float, default=None)
    parser.add_argument("--quality-margin-delta", type=float, default=None)
    parser.add_argument("--quality-margin-cos", type=float, default=None)
    parser.add_argument("--weight-sem", type=float, default=None)
    parser.add_argument("--weight-recon", type=float, default=None)
    parser.add_argument("--weight-mel", type=float, default=None)
    parser.add_argument("--weight-latent", type=float, default=None)
    parser.add_argument("--weight-cos", type=float, default=None)
    parser.add_argument("--weight-delta", type=float, default=None)
    parser.add_argument("--weight-code", type=float, default=None)
    parser.add_argument("--weight-qfeat", type=float, default=None)
    parser.add_argument("--weight-macs", type=float, default=None)
    parser.add_argument("--weight-params", type=float, default=None)
    parser.add_argument("--weight-rtf", type=float, default=None)
    parser.add_argument("--teacher-config")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--teacher-target", choices=["pre_rvq"], default="pre_rvq")
    parser.add_argument("--teacher-cache-mode", choices=["none", "disk"], default="none")
    parser.add_argument("--distill-stage2-steps", type=int, default=0)
    parser.add_argument("--distill-stage3-steps", type=int, default=0)
    parser.add_argument("--distill-final-steps", type=int, default=0)
    parser.add_argument("--distill-lr", type=float, default=1e-4)
    parser.add_argument("--distill-weight-decay", type=float, default=0.0)
    parser.add_argument("--distill-latent-weight", type=float, default=1.0)
    parser.add_argument("--distill-cosine-weight", type=float, default=1.0)
    parser.add_argument("--distill-delta-weight", type=float, default=0.5)
    parser.add_argument("--distill-grad-clip", type=float, default=1.0)
    parser.add_argument("--distill-log-every", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> None:
    run_dir = ensure_run_layout(args.run_dir)
    install_tee_logging(run_dir)
    write_text(run_dir / "commands" / "run_command.txt", " ".join([sys.executable] + sys.argv) + "\n")

    cfg = load_json(args.config)
    shutil.copy2(args.config, run_dir / "configs" / Path(args.config).name)
    fixed_problems = validate_base_interface(cfg)
    if fixed_problems:
        raise SystemExit(f"fixed interface mismatch: {fixed_problems}")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    env = collect_environment_dict(PROJECT_ROOT, extra_tools=["ffmpeg", "opusenc", "opusdec"])
    write_json(run_dir / "reports" / "environment.json", env)
    write_text(run_dir / "reports" / "environment.md", format_environment_markdown(env))

    selection_cfg = build_selection_config(args)
    search_space = search_space_with_policy(args.seed, args.stage1_candidates, "random")
    search_space = apply_search_policy_overrides(search_space, args)
    search_space["search_policy"]["stage1_candidates"] = args.stage1_candidates
    search_space["search_policy"]["stage2_top_k"] = args.stage2_top_k
    search_space["search_policy"]["stage3_top_k"] = args.stage3_top_k
    search_space["search_policy"]["final_top_k"] = args.final_top_k
    write_json(run_dir / "configs" / "operator_library.json", OPERATOR_LIBRARY)
    write_json(run_dir / "configs" / "nas_search_space.json", search_space)
    write_json(
        run_dir / "configs" / "staged_nas_runtime_config.json",
        {
            "seed": args.seed,
            "device_requested": args.device,
            "device_used": str(device),
            "stage1_candidates": args.stage1_candidates,
            "stage2_top_k": args.stage2_top_k,
            "stage3_top_k": args.stage3_top_k,
            "final_top_k": args.final_top_k,
            "stage2_max_samples": args.stage2_max_samples,
            "stage2_max_batches": args.stage2_max_batches,
            "stage3_max_samples": args.stage3_max_samples,
            "stage3_max_batches": args.stage3_max_batches,
            "final_max_samples": args.final_max_samples,
            "final_max_batches": args.final_max_batches,
            "selection_mode": selection_cfg.mode,
            "min_n_filters": search_space["search_policy"]["min_n_filters"],
            "max_skip_blocks": search_space["search_policy"]["max_skip_blocks"],
            "quality_margin_sem": selection_cfg.quality_margin_sem,
            "quality_margin_recon": selection_cfg.quality_margin_recon,
            "quality_margin_mel": selection_cfg.quality_margin_mel,
            "quality_margin_latent": selection_cfg.quality_margin_latent,
            "quality_margin_delta": selection_cfg.quality_margin_delta,
            "quality_margin_cos": selection_cfg.quality_margin_cos,
            "weight_sem": selection_cfg.weight_sem,
            "weight_recon": selection_cfg.weight_recon,
            "weight_mel": selection_cfg.weight_mel,
            "weight_latent": selection_cfg.weight_latent,
            "weight_cos": selection_cfg.weight_cos,
            "weight_delta": selection_cfg.weight_delta,
            "weight_code": selection_cfg.weight_code,
            "weight_qfeat": selection_cfg.weight_qfeat,
            "weight_macs": selection_cfg.weight_macs,
            "weight_params": selection_cfg.weight_params,
            "weight_rtf": selection_cfg.weight_rtf,
            "teacher_enabled": bool(args.teacher_config or args.teacher_checkpoint),
            "teacher_target": args.teacher_target,
            "teacher_cache_mode": args.teacher_cache_mode,
            "distill_stage2_steps": args.distill_stage2_steps,
            "distill_stage3_steps": args.distill_stage3_steps,
            "distill_final_steps": args.distill_final_steps,
            "distill_lr": args.distill_lr,
            "distill_weight_decay": args.distill_weight_decay,
            "distill_latent_weight": args.distill_latent_weight,
            "distill_cosine_weight": args.distill_cosine_weight,
            "distill_delta_weight": args.distill_delta_weight,
            "distill_grad_clip": args.distill_grad_clip,
            "distill_log_every": args.distill_log_every,
            "pipeline": "stage1 profile -> stage2 short distillation + teacher alignment -> stage3 short distillation + RVQ compatibility -> stage4 final short distillation + Pareto",
        },
    )
    write_operator_report(run_dir)
    write_static_reports(run_dir)

    teacher_enabled = bool(args.teacher_config or args.teacher_checkpoint)
    if teacher_enabled and not (args.teacher_config and args.teacher_checkpoint):
        raise SystemExit("--teacher-config and --teacher-checkpoint must be provided together")
    distill_enabled = any(
        steps > 0
        for steps in [args.distill_stage2_steps, args.distill_stage3_steps, args.distill_final_steps]
    )
    if distill_enabled and not teacher_enabled:
        raise SystemExit("short distillation requires --teacher-config and --teacher-checkpoint")
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
            "selection_config": selection_cfg.__dict__,
            "short_distillation": {
                "enabled": distill_enabled,
                "stage2_steps": args.distill_stage2_steps,
                "stage3_steps": args.distill_stage3_steps,
                "final_steps": args.distill_final_steps,
                "lr": args.distill_lr,
                "latent_weight": args.distill_latent_weight,
                "cosine_weight": args.distill_cosine_weight,
                "delta_weight": args.distill_delta_weight,
                "optimized_module": "candidate encoder only",
                "frozen_modules": "pretrained SpeechTokenizer teacher, transform, RVQ, decoder",
            },
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
                f"- short_distillation_enabled: {distill_enabled}",
                f"- short_distillation_steps: stage2={args.distill_stage2_steps}, stage3={args.distill_stage3_steps}, final={args.distill_final_steps}",
                "- short_distillation_optimized_module: NAS candidate encoder only.",
                "- downstream_modules_for_proxy: frozen pretrained SpeechTokenizer transform/RVQ/decoder.",
                "- teacher_is_method_contribution: false",
                "",
            ]
        ),
    )

    candidates = sample_candidates(args.seed, args.stage1_candidates, search_space=search_space)
    if not candidates:
        raise SystemExit("no valid NAS candidates generated")
    for candidate in candidates:
        write_json(run_dir / "artifacts" / "candidates" / f"{candidate['candidate_id']}.json", candidate)

    all_errors = []
    rows_by_stage: Dict[str, List[Dict[str, Any]]] = {}

    print("[stage1_profile] hand encoder", flush=True)
    stage1_hand = base_row(run_dir, cfg, args.seed, device, None, 0, stage1_only=True, teacher_model=teacher_model)
    stage1_hand["stage"] = "stage1_profile"
    stage1_rows, errors = evaluate_candidates(
        run_dir, "stage1_profile", candidates, cfg, args.seed, device, None, 0, stage1_only=True
    )
    all_errors.extend(errors)
    stage1_rows = [stage1_hand] + stage1_rows
    ranked_stage1, stage2_ids = rank_and_select(
        stage1_rows,
        stage1_hand,
        args.stage2_top_k,
        profile_score,
        "top-k by encoder-only profile score",
    )
    stage2_candidates = [candidate for candidate in candidates if candidate["candidate_id"] in set(stage2_ids)]
    write_selected_candidates(run_dir, "stage1_to_stage2", stage2_candidates)
    write_stage_table(run_dir, "stage1_profile", stage1_rows)
    rows_by_stage["stage1_profile"] = stage1_rows

    if not stage2_candidates:
        raise SystemExit("stage1 selected no candidates for stage2")

    dl2, samples2 = prepare_stage_dataloader(run_dir, "stage2_proxy", args.manifest, cfg, args.stage2_max_samples)
    print("[stage2_proxy] hand encoder", flush=True)
    stage2_hand = base_row(run_dir, cfg, args.seed, device, dl2, args.stage2_max_batches, stage1_only=False, teacher_model=teacher_model)
    stage2_hand["stage"] = "stage2_proxy"
    stage2_rows, errors = evaluate_candidates(
        run_dir,
        "stage2_proxy",
        stage2_candidates,
        cfg,
        args.seed,
        device,
        dl2,
        args.stage2_max_batches,
        stage1_only=False,
        teacher_model=teacher_model,
        distill_steps=args.distill_stage2_steps,
        distill_lr=args.distill_lr,
        distill_weight_decay=args.distill_weight_decay,
        distill_latent_weight=args.distill_latent_weight,
        distill_cosine_weight=args.distill_cosine_weight,
        distill_delta_weight=args.distill_delta_weight,
        distill_grad_clip=args.distill_grad_clip,
        distill_log_every=args.distill_log_every,
    )
    all_errors.extend(errors)
    stage2_rows = [stage2_hand] + stage2_rows
    ranked_stage2, stage3_ids = rank_and_select(
        stage2_rows,
        stage2_hand,
        args.stage3_top_k,
        lambda row, hand: proxy_score(row, hand, selection_cfg),
        "top-k by stage2 quality-constrained proxy score",
        lambda row, hand: proxy_penalty_breakdown(row, hand, selection_cfg),
    )
    stage3_candidates = [candidate for candidate in stage2_candidates if candidate["candidate_id"] in set(stage3_ids)]
    write_selected_candidates(run_dir, "stage2_to_stage3", stage3_candidates)
    write_stage_table(run_dir, "stage2_proxy", stage2_rows)
    write_stage_table(run_dir, "stage2_teacher_alignment", stage2_rows)
    rows_by_stage["stage2_proxy"] = stage2_rows

    if not stage3_candidates:
        raise SystemExit("stage2 selected no candidates for stage3")

    dl3, samples3 = prepare_stage_dataloader(run_dir, "stage3_refined", args.manifest, cfg, args.stage3_max_samples)
    print("[stage3_refined] hand encoder", flush=True)
    stage3_hand = base_row(run_dir, cfg, args.seed, device, dl3, args.stage3_max_batches, stage1_only=False, teacher_model=teacher_model)
    stage3_hand["stage"] = "stage3_refined"
    stage3_rows, errors = evaluate_candidates(
        run_dir,
        "stage3_refined",
        stage3_candidates,
        cfg,
        args.seed,
        device,
        dl3,
        args.stage3_max_batches,
        stage1_only=False,
        teacher_model=teacher_model,
        distill_steps=args.distill_stage3_steps,
        distill_lr=args.distill_lr,
        distill_weight_decay=args.distill_weight_decay,
        distill_latent_weight=args.distill_latent_weight,
        distill_cosine_weight=args.distill_cosine_weight,
        distill_delta_weight=args.distill_delta_weight,
        distill_grad_clip=args.distill_grad_clip,
        distill_log_every=args.distill_log_every,
    )
    all_errors.extend(errors)
    stage3_rows = [stage3_hand] + stage3_rows
    ranked_stage3, final_ids = rank_and_select(
        stage3_rows,
        stage3_hand,
        args.final_top_k,
        lambda row, hand: proxy_score(row, hand, selection_cfg),
        "top-k by stage3 refined quality-constrained proxy score",
        lambda row, hand: proxy_penalty_breakdown(row, hand, selection_cfg),
    )
    final_candidates = [candidate for candidate in stage3_candidates if candidate["candidate_id"] in set(final_ids)]
    write_selected_candidates(run_dir, "stage3_to_final", final_candidates)
    write_stage_table(run_dir, "stage3_refined", stage3_rows)
    write_stage_table(run_dir, "stage3_rvq_compatibility", stage3_rows)
    rows_by_stage["stage3_refined"] = stage3_rows

    if not final_candidates:
        raise SystemExit("stage3 selected no candidates for final evaluation")

    dl4, samples4 = prepare_stage_dataloader(run_dir, "stage4_final", args.manifest, cfg, args.final_max_samples)
    print("[stage4_final] hand encoder", flush=True)
    final_hand = base_row(run_dir, cfg, args.seed, device, dl4, args.final_max_batches, stage1_only=False, teacher_model=teacher_model)
    final_hand["stage"] = "stage4_final"
    final_rows, errors = evaluate_candidates(
        run_dir,
        "stage4_final",
        final_candidates,
        cfg,
        args.seed,
        device,
        dl4,
        args.final_max_batches,
        stage1_only=False,
        teacher_model=teacher_model,
        distill_steps=args.distill_final_steps,
        distill_lr=args.distill_lr,
        distill_weight_decay=args.distill_weight_decay,
        distill_latent_weight=args.distill_latent_weight,
        distill_cosine_weight=args.distill_cosine_weight,
        distill_delta_weight=args.distill_delta_weight,
        distill_grad_clip=args.distill_grad_clip,
        distill_log_every=args.distill_log_every,
    )
    all_errors.extend(errors)
    final_rows = [final_hand] + final_rows
    rank_and_select(
        final_rows,
        final_hand,
        len(final_candidates),
        lambda row, hand: proxy_score(row, hand, selection_cfg),
        "final candidates ranked by final quality-constrained proxy score",
        lambda row, hand: proxy_penalty_breakdown(row, hand, selection_cfg),
    )
    write_stage_table(run_dir, "stage4_final", final_rows)
    rows_by_stage["stage4_final"] = final_rows

    final_completed = completed_nas_rows(final_rows)
    if not final_completed:
        raise SystemExit("no completed final NAS candidates")
    pareto_objectives = (
        [
            "teacher_latent_smooth_l1",
            "teacher_latent_cosine_distance",
            "teacher_temporal_delta_loss",
            "rvq_quantized_feature_l1",
        ]
        if teacher_enabled
        else []
    ) + [
        "semantic_proxy_loss",
        "proxy_recon_l1",
        "proxy_mel_loss",
        "encoder_macs",
        "encoder_params",
        "encoder_rtf_mean",
    ]
    frontier = pareto_frontier_by_objectives(final_completed, pareto_objectives)
    if not frontier:
        raise SystemExit("no final candidate on Pareto frontier")

    best = min(frontier, key=lambda row: proxy_score(row, final_hand, selection_cfg))
    best.update(proxy_penalty_breakdown(best, final_hand, selection_cfg))
    best["selected"] = True
    best["selection_reason"] = "minimum final quality-constrained proxy score among multi-objective Pareto candidates"
    for row in frontier:
        if row["candidate_id"] != best["candidate_id"]:
            row.update(proxy_penalty_breakdown(row, final_hand, selection_cfg))
            row["selected"] = False
            row["selection_reason"] = "final Pareto candidate, not minimum final quality-constrained proxy score"

    best_candidate = next(candidate for candidate in final_candidates if candidate["candidate_id"] == best["candidate_id"])
    best_dir = run_dir / "artifacts" / "best_architecture"
    best_handoff_config = build_encoder_only_handoff_config(best_candidate, best, cfg)
    write_json(best_dir / "best_seanet_config.json", best_handoff_config)
    write_json(best_dir / "best_candidate_raw.json", best_candidate)
    write_json(run_dir / "artifacts" / "candidate_errors.json", all_errors)

    all_stage_rows = stage1_rows + stage2_rows + stage3_rows + final_rows
    write_json(run_dir / "metrics" / "nas_records.json", all_stage_rows)
    write_csv(run_dir / "metrics" / "nas_records.csv", csv_rows(all_stage_rows, STAGE_FIELDNAMES), STAGE_FIELDNAMES)
    write_json(run_dir / "metrics" / "encoder_proxy_results.json", stage2_rows + stage3_rows + final_rows)
    write_csv(
        run_dir / "metrics" / "encoder_proxy_results.csv",
        csv_rows(stage2_rows + stage3_rows + final_rows, STAGE_FIELDNAMES),
        STAGE_FIELDNAMES,
    )
    write_json(run_dir / "metrics" / "teacher_alignment.json", stage2_rows + stage3_rows + final_rows)
    write_csv(
        run_dir / "metrics" / "teacher_alignment.csv",
        csv_rows(stage2_rows + stage3_rows + final_rows, STAGE_FIELDNAMES),
        STAGE_FIELDNAMES,
    )
    write_json(run_dir / "metrics" / "rvq_compatibility.json", stage2_rows + stage3_rows + final_rows)
    write_csv(
        run_dir / "metrics" / "rvq_compatibility.csv",
        csv_rows(stage2_rows + stage3_rows + final_rows, STAGE_FIELDNAMES),
        STAGE_FIELDNAMES,
    )
    write_json(run_dir / "metrics" / "pareto_frontier.json", frontier)
    write_csv(run_dir / "metrics" / "pareto_frontier.csv", csv_rows(frontier, PARETO_FIELDNAMES), PARETO_FIELDNAMES)
    write_json(
        run_dir / "metrics" / "results.json",
        {
            "selected": best,
            "stage1_rows": stage1_rows,
            "stage2_rows": stage2_rows,
            "stage3_rows": stage3_rows,
            "stage4_final_rows": final_rows,
            "candidate_errors": all_errors,
            "selection_config": selection_cfg.__dict__,
            "pareto_objectives": pareto_objectives,
        },
    )
    write_csv(run_dir / "metrics" / "results.csv", csv_rows(final_rows, STAGE_FIELDNAMES), STAGE_FIELDNAMES)

    write_interface_report(run_dir, rows_by_stage)

    selection_lines = [
        "# Staged NAS Selection",
        "",
        "- pipeline: stage1 profile -> stage2 short distillation + proxy -> stage3 short distillation + refined proxy -> stage4 final short distillation + Pareto",
        f"- selection_mode: {selection_cfg.mode}",
        f"- short_distillation_steps: stage2={args.distill_stage2_steps}, stage3={args.distill_stage3_steps}, final={args.distill_final_steps}",
        f"- short_distillation_loss: {args.distill_latent_weight}*SmoothL1(Z_B,Z_A) + {args.distill_cosine_weight}*cosine_distance + {args.distill_delta_weight}*temporal_delta_loss",
        "- short_distillation_updates: NAS candidate encoder only",
        "- frozen_proxy_components: pretrained SpeechTokenizer transform, RVQ, and decoder",
        f"- capacity_guard: min_n_filters={search_space['search_policy']['min_n_filters']}, max_skip_blocks={search_space['search_policy']['max_skip_blocks']}",
        f"- stage1_generated: {len(candidates)}",
        f"- stage1_to_stage2: {len(stage2_candidates)}",
        f"- stage2_to_stage3: {len(stage3_candidates)}",
        f"- stage3_to_final: {len(final_candidates)}",
        f"- selected_candidate: {best['candidate_id']}",
        "- selected_config: artifacts/best_architecture/best_seanet_config.json",
        "- raw_selected_candidate: artifacts/best_architecture/best_candidate_raw.json",
        "- stage1_score: 0.45*log(MAC_ratio)+0.45*log(param_ratio)+0.10*log(RTF_ratio)",
        "- proxy_quality_penalty: weighted positive log excess over hand-designed encoder for semantic_proxy_loss, proxy_recon_l1, and proxy_mel_loss",
        "- proxy_resource_score: weighted log ratios for encoder_macs, encoder_params, and encoder_rtf_mean",
        f"- quality_margins: semantic={selection_cfg.quality_margin_sem}, recon={selection_cfg.quality_margin_recon}, mel={selection_cfg.quality_margin_mel}",
        f"- weights: semantic={selection_cfg.weight_sem}, recon={selection_cfg.weight_recon}, mel={selection_cfg.weight_mel}, macs={selection_cfg.weight_macs}, params={selection_cfg.weight_params}, rtf={selection_cfg.weight_rtf}",
        f"- pareto_objectives: {', '.join(pareto_objectives)}",
        f"- selected_encoder_strides: {best['encoder_strides']}",
        f"- selected_decoder_strides: {best['decoder_strides']}",
        f"- decoder_condition: {best['decoder_condition']}",
        "",
        "This is a staged proxy NAS result. It must not be presented as a fully trained SCIT-Speech result.",
        "",
    ]
    write_text(run_dir / "reports" / "staged_selection.md", "\n".join(selection_lines))
    write_text(run_dir / "reports" / "pareto_selection.md", "\n".join(selection_lines))
    write_text(run_dir / "reports" / "teacher_guided_selection.md", "\n".join(selection_lines))

    failed_count = sum(1 for row in all_stage_rows if row.get("status") == "failed")
    summary_lines = [
        "# Experiment 1 Staged NAS Summary",
        "",
        "- status: completed",
        f"- run_id: {run_dir.name}",
        f"- seed: {args.seed}",
        f"- selection_mode: {selection_cfg.mode}",
        f"- capacity_guard: min_n_filters={search_space['search_policy']['min_n_filters']}, max_skip_blocks={search_space['search_policy']['max_skip_blocks']}",
        f"- stage1_generated: {len(candidates)}",
        f"- stage1_to_stage2: {len(stage2_candidates)}",
        f"- distill_stage2_steps: {args.distill_stage2_steps}",
        f"- stage2_proxy_samples: {len(samples2)}",
        f"- stage2_to_stage3: {len(stage3_candidates)}",
        f"- distill_stage3_steps: {args.distill_stage3_steps}",
        f"- stage3_refined_samples: {len(samples3)}",
        f"- stage3_to_final: {len(final_candidates)}",
        f"- distill_final_steps: {args.distill_final_steps}",
        f"- final_samples: {len(samples4)}",
        f"- failed_candidate_evaluations: {failed_count}",
        f"- selected_encoder: {best['candidate_id']}",
        f"- selected_encoder_strides: {best['encoder_strides']}",
        f"- selected_decoder_strides: {best['decoder_strides']}",
        f"- decoder_condition: {best['decoder_condition']}",
        "- fixed interface: sample_rate=16000, prod(encoder_strides)=320, latent_rate=50, latent_dimension=1024, M=3, K=1024.",
        "- decoder contamination: decoder ops/width/depth/activation are not searched; proxy reconstruction uses the frozen pretrained SpeechTokenizer decoder.",
        "- selection rule: short-distilled quality-constrained proxy score plus multi-objective Pareto over teacher alignment, RVQ compatibility, semantic, reconstruction, mel, MACs, params, and RTF.",
        "- limitation: all quality numbers are proxy metrics, not final SCIT-Speech training results.",
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
            "selection_mode": selection_cfg.mode,
            "selection_config": selection_cfg.__dict__,
            "pareto_objectives": pareto_objectives,
            "stage1_generated": len(candidates),
            "stage1_to_stage2": len(stage2_candidates),
            "stage2_to_stage3": len(stage3_candidates),
            "stage3_to_final": len(final_candidates),
            "distill_stage2_steps": args.distill_stage2_steps,
            "distill_stage3_steps": args.distill_stage3_steps,
            "distill_final_steps": args.distill_final_steps,
            "failed_candidate_evaluations": failed_count,
        },
    )
    write_text(
        run_dir / "reports" / "agent_handoff.md",
        "\n".join(
            [
                "# Exp1 Staged NAS Agent Handoff",
                "",
                "- status: completed",
                f"- selected_encoder: {best['candidate_id']}",
                f"- selection_mode: {selection_cfg.mode}",
                f"- short_distillation_steps: stage2={args.distill_stage2_steps}, stage3={args.distill_stage3_steps}, final={args.distill_final_steps}",
                "- selection rule: short-distilled quality-constrained proxy score plus multi-objective Pareto.",
                "- staged outputs are in metrics/stage1_profile.*, stage2_proxy.*, stage3_refined.*, and stage4_final.*.",
                "- downstream Exp2 may use artifacts/best_architecture/best_seanet_config.json as the NAS encoder route.",
                "- raw selected NAS search item is preserved in artifacts/best_architecture/best_candidate_raw.json.",
                "",
            ]
        ),
    )
    write_json(run_dir / "reports" / "agent_status.json", {"agent": "Exp1 Staged NAS Runner", "status": "completed", "run_id": run_dir.name})
    print(json.dumps({"status": "completed", "selected": best["candidate_id"], "run_dir": str(run_dir)}, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except SystemExit as exc:
        if exc.code not in (0, None):
            write_failure_artifacts(args.run_dir, exc, failed_step="run_staged_encoder_nas")
        raise
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        write_failure_artifacts(args.run_dir, exc, failed_step="run_staged_encoder_nas")
        raise


if __name__ == "__main__":
    main()
