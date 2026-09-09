"""Exp22-A: speaker identity preservation for SCIT-Speech reconstructions."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_lca_vs_base import build_model, load_audio_for_inference, load_json, load_state
from scripts.experiment_utils import ensure_run_layout, file_sha256, write_csv, write_json
from scripts.speaker_identity_utils import (
    SpeakerEmbeddingExtractor,
    compute_eer,
    enroll_profiles,
    filter_speaker_groups,
    group_paths_by_speaker,
    pairwise_verification_scores,
    predict_speaker,
    read_audio_paths,
    scan_audio_root,
    split_enrollment_and_test,
    tar_at_far,
)


DEFAULT_BASE_CONFIG = PROJECT_ROOT / "output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/configs/scit_speech_base_config.json"
DEFAULT_BASE_CKPT = PROJECT_ROOT / "output/experiments/exp2_scit_speech_distill30_retrain_20260529_seed42/checkpoints/SCIT-Speech-Base_best.pt"
DEFAULT_LCA_CONFIG = PROJECT_ROOT / "output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/configs/lca_finetune_config.json"
DEFAULT_LCA_CKPT = PROJECT_ROOT / "output/experiments/exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42/checkpoints/SCIT-Speech-LCA_v2_step30000_robust_optimum.pt"


def mean_or_nan(values: List[float]) -> float:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(sum(clean) / len(clean)) if clean else float("nan")


def model_reconstruct(model, audio_np: np.ndarray, L: int, n_q: int, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(audio_np).to(device).view(1, 1, -1)
    with torch.no_grad():
        codes = model.encode(x, n_q=int(n_q), st=0)
        recon = model.decode(codes[: int(L)].contiguous().long(), st=0)
    return recon[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)


def load_models(args: argparse.Namespace, device: torch.device):
    specs = {}
    if "base" in args.models:
        cfg = load_json(args.base_config)
        model = build_model(cfg).to(device).eval()
        load_state(model, args.base_checkpoint)
        specs["base"] = {
            "config": cfg,
            "model": model,
            "config_path": str(args.base_config),
            "checkpoint": str(args.base_checkpoint),
            "sha256": file_sha256(args.base_checkpoint),
        }
    if "lca" in args.models:
        cfg = load_json(args.lca_config)
        model = build_model(cfg).to(device).eval()
        load_state(model, args.lca_checkpoint)
        specs["lca"] = {
            "config": cfg,
            "model": model,
            "config_path": str(args.lca_config),
            "checkpoint": str(args.lca_checkpoint),
            "sha256": file_sha256(args.lca_checkpoint),
        }
    return specs


def build_groups(args: argparse.Namespace):
    if args.sample_list:
        paths = read_audio_paths(args.sample_list)
        source_desc = str(args.sample_list)
    elif args.audio_root:
        paths = scan_audio_root(args.audio_root)
        source_desc = str(args.audio_root)
    else:
        raise SystemExit("Provide --sample-list or --audio-root")
    groups = group_paths_by_speaker(paths, source=args.speaker_source)
    groups = filter_speaker_groups(
        groups,
        min_utterances=args.enroll_per_speaker + args.test_per_speaker,
        max_speakers=args.max_speakers,
        seed=args.seed,
    )
    enroll, test = split_enrollment_and_test(
        groups,
        enroll_per_speaker=args.enroll_per_speaker,
        test_per_speaker=args.test_per_speaker,
        seed=args.seed,
    )
    if not enroll:
        raise SystemExit("No speakers have enough utterances for the requested split")
    return source_desc, enroll, test


def evaluate(args: argparse.Namespace):
    run_dir = ensure_run_layout(args.run_dir)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if args.speaker_device == "auto":
        speaker_device = str(device)
    elif args.speaker_device == "cuda" and not torch.cuda.is_available():
        speaker_device = "cpu"
    else:
        speaker_device = args.speaker_device

    source_desc, enroll_paths, test_paths = build_groups(args)
    speaker_extractor = SpeakerEmbeddingExtractor(
        backend=args.speaker_backend,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        device=speaker_device,
        ecapa_source=args.ecapa_source,
        ecapa_savedir=args.ecapa_savedir,
    )
    profiles = enroll_profiles(
        enroll_paths,
        sample_rate=args.sample_rate,
        n_mfcc=args.n_mfcc,
        extractor=speaker_extractor,
    )
    if not profiles:
        raise SystemExit("No speaker profiles could be enrolled")

    models = load_models(args, device)
    n_q = int(next(iter(models.values()))["config"].get("n_q", 3)) if models else 3

    write_json(
        run_dir / "artifacts" / "speaker_split.json",
        {
            "source": source_desc,
            "speaker_source": args.speaker_source,
            "enroll": {speaker: [str(path) for path in paths] for speaker, paths in enroll_paths.items()},
            "test": {speaker: [str(path) for path in paths] for speaker, paths in test_paths.items()},
        },
    )

    rows = []
    total = sum(len(paths) for paths in test_paths.values())
    done = 0
    for speaker_id, paths in sorted(test_paths.items()):
        for audio_path in paths:
            done += 1
            audio_t = load_audio_for_inference(str(audio_path), args.sample_rate).squeeze(0)
            audio_np = audio_t.numpy().astype(np.float32, copy=False)
            sample_id = Path(audio_path).stem
            conditions = [("original", 0, audio_np)]
            for model_name, spec in models.items():
                for L in args.layers:
                    recon = model_reconstruct(spec["model"], audio_np, int(L), n_q=n_q, device=device)
                    length = min(len(audio_np), len(recon))
                    conditions.append((model_name, int(L), recon[:length]))

            for model_name, L, waveform in conditions:
                emb = speaker_extractor.from_waveform(waveform)
                pred = predict_speaker(emb, profiles, threshold=args.threshold)
                correct_score = pred.scores.get(speaker_id)
                impostor_scores = [score for sid, score in pred.scores.items() if sid != speaker_id]
                max_impostor = max(impostor_scores) if impostor_scores else None
                rows.append(
                    {
                        "model": model_name,
                        "L": int(L),
                        "sample_id": sample_id,
                        "speaker_id": speaker_id,
                        "audio_path": str(audio_path),
                        "predicted_speaker": pred.predicted_speaker,
                        "top1_correct": int(pred.predicted_speaker == speaker_id),
                        "verified": int(pred.verified),
                        "score": pred.score,
                        "correct_score": correct_score,
                        "max_impostor_score": max_impostor,
                        "margin": None if correct_score is None or max_impostor is None else float(correct_score - max_impostor),
                        "duration_sec": float(len(waveform) / args.sample_rate),
                    }
                )
            if done % args.progress_every == 0 or done == total:
                print(f"done {done}/{total}", flush=True)

    summary_rows = summarize(rows, far=args.tar_far)
    fields = [
        "model",
        "L",
        "sample_id",
        "speaker_id",
        "audio_path",
        "predicted_speaker",
        "top1_correct",
        "verified",
        "score",
        "correct_score",
        "max_impostor_score",
        "margin",
        "duration_sec",
    ]
    write_csv(run_dir / "metrics" / "speaker_identity_results.csv", rows, fields)
    write_csv(
        run_dir / "metrics" / "speaker_identity_summary.csv",
        summary_rows,
        [
            "model",
            "L",
            "n",
            "top1_accuracy",
            "verified_rate",
            "score_mean",
            "correct_score_mean",
            "max_impostor_score_mean",
            "margin_mean",
            "eer",
            "eer_threshold",
            "tar_at_far",
            "tar_threshold",
        ],
    )
    write_json(
        run_dir / "metrics" / "speaker_identity_results.json",
        {
            "args": vars(args),
            "device": str(device),
            "speaker_device": speaker_device,
            "speaker_backend": speaker_extractor.description,
            "speaker_count": len(profiles),
            "models": {name: {k: v for k, v in spec.items() if k != "model" and k != "config"} for name, spec in models.items()},
            "rows": rows,
            "summary": summary_rows,
        },
    )
    write_report(
        run_dir / "reports" / "speaker_identity_summary.md",
        summary_rows,
        len(profiles),
        args,
        speaker_extractor.description,
    )
    return run_dir


def summarize(rows: List[dict], far: float) -> List[dict]:
    buckets: Dict[tuple, List[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["model"], row["L"])].append(row)
    summary = []
    for (model_name, L), items in sorted(buckets.items(), key=lambda x: (str(x[0][0]), int(x[0][1]))):
        labels, scores = pairwise_verification_scores(items)
        eer, eer_threshold = compute_eer(labels, scores)
        tar, tar_threshold = tar_at_far(labels, scores, target_far=far)
        summary.append(
            {
                "model": model_name,
                "L": int(L),
                "n": len(items),
                "top1_accuracy": mean_or_nan([row["top1_correct"] for row in items]),
                "verified_rate": mean_or_nan([row["verified"] for row in items]),
                "score_mean": mean_or_nan([row["score"] for row in items]),
                "correct_score_mean": mean_or_nan([row["correct_score"] for row in items]),
                "max_impostor_score_mean": mean_or_nan([row["max_impostor_score"] for row in items]),
                "margin_mean": mean_or_nan([row["margin"] for row in items]),
                "eer": eer,
                "eer_threshold": eer_threshold,
                "tar_at_far": tar,
                "tar_threshold": tar_threshold,
            }
        )
    return summary


def write_report(path: Path, summary_rows: List[dict], speaker_count: int, args: argparse.Namespace, backend_desc: str) -> None:
    lines = [
        "# Exp22 Speaker Identity Preservation",
        "",
        f"- speakers: {speaker_count}",
        f"- enroll_per_speaker: {args.enroll_per_speaker}",
        f"- test_per_speaker: {args.test_per_speaker}",
        f"- backend: {backend_desc}",
        "",
        "| model | L | n | top1 | verified | correct cos | impostor cos | margin | EER | TAR@FAR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {L} | {n} | {top1_accuracy:.3f} | {verified_rate:.3f} | "
            "{correct_score_mean:.3f} | {max_impostor_score_mean:.3f} | {margin_mean:.3f} | "
            "{eer:.3f} | {tar_at_far:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Note: ECAPA is the preferred paper-grade backend for speaker-preservation claims. "
            "MFCC runs are retained as dependency-light sanity checks.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Exp22-A speaker identity preservation evaluation")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sample-list", default="")
    parser.add_argument("--audio-root", default="")
    parser.add_argument("--speaker-source", default="auto", choices=["auto", "vctk", "aishell", "librispeech"])
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--base-checkpoint", default=str(DEFAULT_BASE_CKPT))
    parser.add_argument("--lca-config", default=str(DEFAULT_LCA_CONFIG))
    parser.add_argument("--lca-checkpoint", default=str(DEFAULT_LCA_CKPT))
    parser.add_argument("--models", nargs="+", default=["base", "lca"], choices=["base", "lca"])
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-speakers", type=int, default=0, help="0 means all eligible speakers")
    parser.add_argument("--enroll-per-speaker", type=int, default=3)
    parser.add_argument("--test-per-speaker", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-mfcc", type=int, default=40)
    parser.add_argument("--speaker-backend", default="mfcc", choices=["mfcc", "ecapa"])
    parser.add_argument("--speaker-device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--ecapa-source", default="speechbrain/spkrec-ecapa-voxceleb")
    parser.add_argument("--ecapa-savedir", default=str(PROJECT_ROOT / "output/models/speechbrain_spkrec_ecapa_voxceleb"))
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--tar-far", type=float, default=0.01)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--force-exit-zero-after-success", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = evaluate(args)
    print(json.dumps({"status": "completed", "run_dir": str(run_dir)}, ensure_ascii=False, indent=2))
    if args.force_exit_zero_after_success:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
