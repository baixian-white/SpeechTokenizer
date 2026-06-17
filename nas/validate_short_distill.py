"""Short teacher-distillation validation for Exp1 NAS candidates.

This script is a diagnostic bridge between proxy NAS search and full Exp2
training. It trains only the candidate encoder for a small number of steps
against the frozen pretrained SpeechTokenizer encoder, then compares teacher
alignment and frozen-RVQ compatibility before and after training.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

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
from speechtokenizer.trainer.dataset import audioDataset, get_dataloader
from nas.evaluate_encoder_proxy import build_nas_encoder, load_json, profile_encoder
from nas.teacher_guided_proxy import (
    TEACHER_METRIC_KEYS,
    compute_teacher_guided_metrics,
    load_teacher_model,
)


DISTILL_LOSS_KEYS = [
    "latent_smooth_l1",
    "cosine_distance",
    "temporal_delta_loss",
    "total",
]


def _align_latents(student_latent: torch.Tensor, teacher_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if student_latent.ndim != 3 or teacher_latent.ndim != 3:
        raise ValueError(
            f"expected [B, D, T] latents, got {tuple(student_latent.shape)} and {tuple(teacher_latent.shape)}"
        )
    if student_latent.size(0) != teacher_latent.size(0):
        raise ValueError(
            f"batch mismatch: student={tuple(student_latent.shape)}, teacher={tuple(teacher_latent.shape)}"
        )
    if student_latent.size(1) != teacher_latent.size(1):
        raise ValueError(
            f"dimension mismatch: student={tuple(student_latent.shape)}, teacher={tuple(teacher_latent.shape)}"
        )
    length = min(student_latent.size(-1), teacher_latent.size(-1))
    if length <= 0:
        raise ValueError("cannot align empty latent tensors")
    return student_latent[..., :length], teacher_latent[..., :length]


def compute_distillation_losses(
    *,
    student_latent: torch.Tensor,
    teacher_latent: torch.Tensor,
    latent_weight: float = 1.0,
    cosine_weight: float = 1.0,
    delta_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    student_latent, teacher_latent = _align_latents(student_latent, teacher_latent.detach())
    student_btd = student_latent.transpose(1, 2).contiguous()
    teacher_btd = teacher_latent.transpose(1, 2).contiguous()

    latent_loss = F.smooth_l1_loss(student_btd, teacher_btd)
    cosine = F.cosine_similarity(student_btd, teacher_btd, dim=-1).mean()
    cosine_distance = torch.clamp(1.0 - cosine, min=0.0)
    if student_btd.size(1) > 1:
        student_delta = student_btd[:, 1:, :] - student_btd[:, :-1, :]
        teacher_delta = teacher_btd[:, 1:, :] - teacher_btd[:, :-1, :]
        delta_loss = F.smooth_l1_loss(student_delta, teacher_delta)
    else:
        delta_loss = student_btd.new_tensor(0.0)

    total = latent_weight * latent_loss + cosine_weight * cosine_distance + delta_weight * delta_loss
    return {
        "latent_smooth_l1": latent_loss,
        "cosine_distance": cosine_distance,
        "temporal_delta_loss": delta_loss,
        "total": total,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--teacher-config", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--candidate-json", nargs="*")
    parser.add_argument("--candidate-ids", nargs="*")
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--delta-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-encoders", action="store_true")
    return parser


def load_candidates(args: argparse.Namespace) -> List[Dict[str, object]]:
    if args.candidate_json:
        candidates = [load_json(path) for path in args.candidate_json]
    else:
        source = Path(args.source_run_dir) / "artifacts" / "stages" / "stage3_to_final_selected_candidates.json"
        if not source.exists():
            source = Path(args.source_run_dir) / "artifacts" / "best_architecture" / "best_seanet_config.json"
            if not source.exists():
                raise FileNotFoundError(f"candidate source not found under {args.source_run_dir}")
            candidates = [load_json(source)]
        else:
            candidates = load_json(source)

    if args.candidate_ids:
        wanted = set(args.candidate_ids)
        candidates = [candidate for candidate in candidates if candidate.get("candidate_id") in wanted]
    if args.candidate_limit is not None and args.candidate_limit > 0:
        candidates = candidates[: args.candidate_limit]
    if not candidates:
        raise SystemExit("no candidates selected for short distillation validation")
    return candidates


def make_loader(run_dir: Path, tag: str, manifest: str, cfg: Dict[str, object], max_samples: int, batch_size: int, train: bool):
    manifest_out = run_dir / "artifacts" / f"{tag}_subset.txt"
    stats = normalize_manifest(manifest, manifest_out, PROJECT_ROOT, max_lines=max_samples)
    write_json(run_dir / "artifacts" / f"{tag}_manifest_normalization.json", stats)
    if stats["written"] == 0:
        raise SystemExit(f"{tag}: no usable samples after manifest normalization: {stats}")
    with open(manifest_out, "r", encoding="utf-8") as f:
        lines = [line for line in f.readlines() if line.strip()]
    dataset = audioDataset(
        file_list=lines,
        segment_size=cfg["segment_size"],
        sample_rate=cfg["sample_rate"],
        downsample_rate=320,
        valid=not train,
    )
    loader = get_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=False,
        num_workers=0,
    )
    return loader, stats


def infinite_batches(loader: Iterable):
    while True:
        for batch in loader:
            yield batch


def train_encoder(
    *,
    encoder: torch.nn.Module,
    teacher_model: torch.nn.Module,
    loader,
    device: torch.device,
    steps: int,
    lr: float,
    weight_decay: float,
    latent_weight: float,
    cosine_weight: float,
    delta_weight: float,
    grad_clip: float,
    log_every: int,
    candidate_id: str,
) -> Dict[str, float]:
    encoder.train()
    teacher_model.eval()
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    step_losses = {key: 0.0 for key in DISTILL_LOSS_KEYS}
    batches = infinite_batches(loader)
    last = {key: 0.0 for key in DISTILL_LOSS_KEYS}

    for step in range(1, steps + 1):
        x, _ = next(batches)
        x = x.unsqueeze(1).to(device)
        with torch.no_grad():
            teacher_latent = teacher_model.encoder(x)
        student_latent = encoder(x)
        losses = compute_distillation_losses(
            student_latent=student_latent,
            teacher_latent=teacher_latent,
            latent_weight=latent_weight,
            cosine_weight=cosine_weight,
            delta_weight=delta_weight,
        )
        if not torch.isfinite(losses["total"]):
            raise RuntimeError(f"{candidate_id}: non-finite distillation loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
        optimizer.step()

        for key in DISTILL_LOSS_KEYS:
            value = float(losses[key].detach().item())
            step_losses[key] += value
            last[key] = value
        if log_every > 0 and (step == 1 or step == steps or step % log_every == 0):
            print(
                f"[distill] {candidate_id} step={step}/{steps} "
                f"loss={last['total']:.6f} latent={last['latent_smooth_l1']:.6f} "
                f"cos={last['cosine_distance']:.6f} delta={last['temporal_delta_loss']:.6f}",
                flush=True,
            )

    return {
        **{f"train_mean_{key}": step_losses[key] / max(steps, 1) for key in DISTILL_LOSS_KEYS},
        **{f"train_last_{key}": last[key] for key in DISTILL_LOSS_KEYS},
    }


def evaluate_encoder(
    *,
    encoder: torch.nn.Module,
    teacher_model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
    n_q: int,
) -> Dict[str, float]:
    encoder.eval()
    teacher_model.eval()
    totals = {key: 0.0 for key in TEACHER_METRIC_KEYS}
    totals.update({f"distill_{key}": 0.0 for key in DISTILL_LOSS_KEYS})
    count = 0

    with torch.inference_mode():
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            x, _ = batch
            x = x.unsqueeze(1).to(device)
            teacher_latent = teacher_model.encoder(x)
            student_latent = encoder(x)
            losses = compute_distillation_losses(student_latent=student_latent, teacher_latent=teacher_latent)
            teacher_quantized, teacher_codes, _, _ = teacher_model.quantizer(teacher_latent, n_q=n_q, layers=[0])
            student_quantized, student_codes, _, _ = teacher_model.quantizer(student_latent, n_q=n_q, layers=[0])
            metrics = compute_teacher_guided_metrics(
                teacher_latent=teacher_latent,
                student_latent=student_latent,
                teacher_codes=teacher_codes,
                student_codes=student_codes,
                teacher_quantized=teacher_quantized,
                student_quantized=student_quantized,
            )
            for key, value in metrics.items():
                totals[key] += float(value)
            for key, value in losses.items():
                totals[f"distill_{key}"] += float(value.item())
            count += 1
    if count == 0:
        raise RuntimeError("no eval batches")
    return {key: value / count for key, value in totals.items()} | {"evaluated_batches": count}


def row_from_metrics(candidate_id: str, prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def finite_delta(before: Dict[str, float], after: Dict[str, float], key: str) -> float:
    return float(after[key]) - float(before[key])


def improvement_fraction(before: Dict[str, float], after: Dict[str, float], key: str, lower_is_better: bool = True) -> float:
    base = max(abs(float(before[key])), 1e-12)
    delta = float(before[key]) - float(after[key]) if lower_is_better else float(after[key]) - float(before[key])
    return delta / base


def write_report(run_dir: Path, rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# Exp1 Short Distillation Validation",
        "",
        f"- source_run_dir: {args.source_run_dir}",
        f"- candidate_count: {len(rows)}",
        f"- train_steps_per_candidate: {args.train_steps}",
        f"- train_samples: {args.train_samples}",
        f"- eval_samples: {args.eval_samples}",
        f"- lr: {args.lr}",
        "- teacher: frozen pretrained SpeechTokenizer encoder",
        "- optimized_module: NAS candidate encoder only",
        "- RVQ metrics are monitored with the frozen teacher quantizer.",
        "",
        "## Results",
        "",
        "| candidate | latent before | latent after | latent improv | cosine before | cosine after | rvq agree before | rvq agree after |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {candidate_id} | {before_teacher_latent_smooth_l1:.6f} | {after_teacher_latent_smooth_l1:.6f} | "
            "{improve_teacher_latent_smooth_l1:.2%} | {before_teacher_latent_cosine_distance:.6f} | "
            "{after_teacher_latent_cosine_distance:.6f} | {before_rvq_code_agreement:.6f} | "
            "{after_rvq_code_agreement:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- A useful architecture should reduce latent/cosine/delta losses after short training.",
            "- RVQ code agreement may lag behind latent alignment, but it should not degrade sharply.",
            "- This validation is not full Exp2 training; it checks whether the NAS architecture can learn the teacher target.",
            "",
        ]
    )
    write_text(run_dir / "reports" / "short_distill_validation.md", "\n".join(lines))


def run(args: argparse.Namespace) -> None:
    run_dir = ensure_run_layout(args.run_dir)
    install_tee_logging(run_dir)
    write_text(run_dir / "commands" / "run_command.txt", " ".join([sys.executable] + sys.argv) + "\n")

    cfg = load_json(args.config)
    shutil.copy2(args.config, run_dir / "configs" / Path(args.config).name)
    candidates = load_candidates(args)
    write_json(run_dir / "configs" / "selected_candidates.json", candidates)
    write_json(run_dir / "configs" / "short_distill_runtime_config.json", vars(args))

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} candidates={len(candidates)} train_steps={args.train_steps}", flush=True)
    env = collect_environment_dict(PROJECT_ROOT, extra_tools=["ffmpeg"])
    write_json(run_dir / "reports" / "environment.json", env)
    write_text(run_dir / "reports" / "environment.md", format_environment_markdown(env))

    teacher_model = load_teacher_model(args.teacher_config, args.teacher_checkpoint, device)
    teacher_model.eval()
    train_loader, train_stats = make_loader(run_dir, "train", args.manifest, cfg, args.train_samples, args.batch_size, train=True)
    eval_loader, eval_stats = make_loader(run_dir, "eval", args.manifest, cfg, args.eval_samples, args.batch_size, train=False)

    rows = []
    errors = []
    for index, candidate in enumerate(candidates, start=1):
        cid = str(candidate["candidate_id"])
        print(f"[candidate] {index}/{len(candidates)} {cid}", flush=True)
        try:
            torch.manual_seed(int(args.seed))
            encoder = build_nas_encoder(candidate, cfg).to(device)
            profile = profile_encoder(encoder, device)
            before = evaluate_encoder(
                encoder=encoder,
                teacher_model=teacher_model,
                loader=eval_loader,
                device=device,
                max_batches=args.eval_samples,
                n_q=int(cfg["n_q"]),
            )
            train_stats_row = train_encoder(
                encoder=encoder,
                teacher_model=teacher_model,
                loader=train_loader,
                device=device,
                steps=args.train_steps,
                lr=args.lr,
                weight_decay=args.weight_decay,
                latent_weight=args.latent_weight,
                cosine_weight=args.cosine_weight,
                delta_weight=args.delta_weight,
                grad_clip=args.grad_clip,
                log_every=args.log_every,
                candidate_id=cid,
            )
            after = evaluate_encoder(
                encoder=encoder,
                teacher_model=teacher_model,
                loader=eval_loader,
                device=device,
                max_batches=args.eval_samples,
                n_q=int(cfg["n_q"]),
            )
            row = {
                "candidate_id": cid,
                "status": "completed",
                "error": "",
                "train_steps": args.train_steps,
                "train_samples": train_stats["written"],
                "eval_samples": eval_stats["written"],
                "encoder_strides": json.dumps(candidate.get("encoder_strides"), ensure_ascii=False),
                "n_filters": candidate.get("n_filters"),
                "compress": candidate.get("compress"),
                "lstm": candidate.get("lstm"),
                "activation": candidate.get("activation"),
                "layer_ops_list": json.dumps(candidate.get("layer_ops_list"), ensure_ascii=False),
                "layer_se_list": json.dumps(candidate.get("layer_se_list"), ensure_ascii=False),
                **profile,
                **row_from_metrics(cid, "before", before),
                **row_from_metrics(cid, "after", after),
                **train_stats_row,
            }
            for key in [
                "teacher_latent_smooth_l1",
                "teacher_latent_cosine_distance",
                "teacher_temporal_delta_loss",
                "rvq_quantized_feature_l1",
            ]:
                row[f"delta_{key}"] = finite_delta(before, after, key)
                row[f"improve_{key}"] = improvement_fraction(before, after, key, lower_is_better=True)
            row["delta_rvq_code_agreement"] = finite_delta(before, after, "rvq_code_agreement")
            row["improve_rvq_code_agreement"] = improvement_fraction(
                before, after, "rvq_code_agreement", lower_is_better=False
            )
            rows.append(row)
            if args.save_encoders:
                torch.save(encoder.state_dict(), run_dir / "checkpoints" / f"{cid}_encoder.pt")
            print(
                f"[result] {cid} latent {before['teacher_latent_smooth_l1']:.6f} -> "
                f"{after['teacher_latent_smooth_l1']:.6f}, rvq_agree "
                f"{before['rvq_code_agreement']:.6f} -> {after['rvq_code_agreement']:.6f}",
                flush=True,
            )
        except Exception as exc:
            message = f"{exc.__class__.__name__}: {exc}"
            errors.append({"candidate_id": cid, "error": message, "traceback": traceback.format_exc(limit=8)})
            rows.append({"candidate_id": cid, "status": "failed", "error": message})
            print(f"[error] {cid} {message}", flush=True)
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fieldnames = sorted({key for row in rows for key in row.keys()})
    write_csv(run_dir / "metrics" / "short_distill_results.csv", rows, fieldnames)
    write_json(run_dir / "metrics" / "short_distill_results.json", rows)
    write_json(run_dir / "artifacts" / "candidate_errors.json", errors)
    write_report(run_dir, [row for row in rows if row.get("status") == "completed"], args)
    completed = [row for row in rows if row.get("status") == "completed"]
    best = None
    if completed:
        best = max(completed, key=lambda row: float(row.get("improve_teacher_latent_smooth_l1", -math.inf)))
    status = {
        "status": "completed" if not errors else "completed_with_errors",
        "run_id": run_dir.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_run_dir": args.source_run_dir,
        "candidate_count": len(candidates),
        "completed_count": len(completed),
        "error_count": len(errors),
        "best_latent_improvement_candidate": best["candidate_id"] if best else None,
        "best_latent_improvement": best.get("improve_teacher_latent_smooth_l1") if best else None,
    }
    write_json(run_dir / "reports" / "status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
