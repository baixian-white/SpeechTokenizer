"""Exp22-B: speaker probes over SCIT-Speech codes and latent features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_lca_vs_base import build_model, load_audio_for_inference, load_json, load_state
from scripts.evaluate_speaker_identity import DEFAULT_BASE_CKPT, DEFAULT_BASE_CONFIG, DEFAULT_LCA_CKPT, DEFAULT_LCA_CONFIG
from scripts.experiment_utils import ensure_run_layout, file_sha256, write_csv, write_json
from scripts.speaker_identity_utils import (
    filter_speaker_groups,
    group_paths_by_speaker,
    read_audio_paths,
    scan_audio_root,
    split_enrollment_and_test,
)


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


def build_split(args: argparse.Namespace):
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
        min_utterances=args.train_per_speaker + args.test_per_speaker,
        max_speakers=args.max_speakers,
        seed=args.seed,
    )
    train, test = split_enrollment_and_test(
        groups,
        enroll_per_speaker=args.train_per_speaker,
        test_per_speaker=args.test_per_speaker,
        seed=args.seed,
    )
    if not train:
        raise SystemExit("No speakers have enough utterances for the requested probe split")
    return source_desc, train, test


def rows_from_split(train: Dict[str, List[Path]], test: Dict[str, List[Path]]) -> List[dict]:
    rows = []
    for split_name, split_map in [("train", train), ("test", test)]:
        for speaker_id, paths in sorted(split_map.items()):
            for path in paths:
                rows.append({"split": split_name, "speaker_id": speaker_id, "audio_path": path, "sample_id": path.stem})
    return rows


def extract_codes_hist(model, audio_np: np.ndarray, L: int, n_q: int, codebook_size: int, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(audio_np).to(device).view(1, 1, -1)
    with torch.no_grad():
        codes = model.encode(x, n_q=int(n_q), st=0)[: int(L), 0].detach().cpu().numpy()
    feats = []
    for layer_codes in codes:
        hist = np.bincount(layer_codes.astype(np.int64), minlength=codebook_size).astype(np.float32)
        denom = max(1.0, float(hist.sum()))
        feats.append(hist / denom)
    return np.concatenate(feats, axis=0).astype(np.float32, copy=False)


def extract_latent_stats(model, audio_np: np.ndarray, L: int, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(audio_np).to(device).view(1, 1, -1)
    with torch.no_grad():
        features = model.forward_feature(x, layers=list(range(int(L))))
    pooled = []
    for tensor in features:
        arr = tensor[0].detach().cpu().float().numpy()
        pooled.append(arr.mean(axis=1))
        pooled.append(arr.std(axis=1))
    return np.concatenate(pooled, axis=0).astype(np.float32, copy=False)


def extract_feature(model, audio_path: Path, feature_kind: str, L: int, cfg: dict, device: torch.device) -> np.ndarray:
    sample_rate = int(cfg.get("sample_rate", 16000))
    audio_np = load_audio_for_inference(str(audio_path), sample_rate).squeeze(0).numpy().astype(np.float32, copy=False)
    if feature_kind == "codes_hist":
        return extract_codes_hist(
            model,
            audio_np,
            L=L,
            n_q=int(cfg.get("n_q", 3)),
            codebook_size=int(cfg.get("codebook_size", 1024)),
            device=device,
        )
    if feature_kind == "latent_stats":
        return extract_latent_stats(model, audio_np, L=L, device=device)
    raise ValueError(f"unsupported feature_kind={feature_kind}")


def top_k_accuracy_from_proba(y_true, proba: np.ndarray, classes, k: int) -> float:
    classes_arr = np.asarray(list(classes))
    scores = np.asarray(proba)
    if scores.ndim != 2 or scores.shape[0] == 0 or classes_arr.size == 0:
        return float("nan")
    k = max(1, min(int(k), classes_arr.size))
    top_indices = np.argsort(scores, axis=1)[:, -k:]
    correct = [
        true_label in set(classes_arr[row_indices])
        for true_label, row_indices in zip(y_true, top_indices)
    ]
    return float(np.mean(correct)) if correct else float("nan")


def fit_and_score(X_train, y_train, X_test, y_test, seed: int) -> dict:
    labels = sorted(set(y_train))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=seed),
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    result = {
        "top1_accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "speaker_count": len(labels),
        "test_count": len(y_test),
        "top5_accuracy": float("nan"),
    }
    if len(labels) >= 2 and hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)
        k = min(5, len(labels))
        result["top5_accuracy"] = top_k_accuracy_from_proba(y_test, proba, clf.classes_, k=k)
    return result


def evaluate(args: argparse.Namespace):
    run_dir = ensure_run_layout(args.run_dir)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    source_desc, train, test = build_split(args)
    split_rows = rows_from_split(train, test)
    models = load_models(args, device)

    write_json(
        run_dir / "artifacts" / "speaker_probe_split.json",
        {
            "source": source_desc,
            "speaker_source": args.speaker_source,
            "train": {speaker: [str(path) for path in paths] for speaker, paths in train.items()},
            "test": {speaker: [str(path) for path in paths] for speaker, paths in test.items()},
        },
    )

    summary_rows = []
    per_sample_rows = []
    for model_name, spec in models.items():
        for feature_kind in args.feature_kinds:
            for L in args.layers:
                X_train = []
                y_train = []
                X_test = []
                y_test = []
                for idx, row in enumerate(split_rows, 1):
                    feature = extract_feature(spec["model"], row["audio_path"], feature_kind, int(L), spec["config"], device)
                    target = y_train if row["split"] == "train" else y_test
                    feats = X_train if row["split"] == "train" else X_test
                    feats.append(feature)
                    target.append(row["speaker_id"])
                    per_sample_rows.append(
                        {
                            "model": model_name,
                            "feature_kind": feature_kind,
                            "L": int(L),
                            "split": row["split"],
                            "speaker_id": row["speaker_id"],
                            "sample_id": row["sample_id"],
                            "audio_path": str(row["audio_path"]),
                            "feature_dim": int(feature.shape[0]),
                        }
                    )
                    if idx % args.progress_every == 0:
                        print(f"{model_name} {feature_kind} L={L}: extracted {idx}/{len(split_rows)}", flush=True)
                metrics = fit_and_score(np.stack(X_train), y_train, np.stack(X_test), y_test, seed=args.seed)
                summary_rows.append(
                    {
                        "model": model_name,
                        "feature_kind": feature_kind,
                        "L": int(L),
                        "feature_dim": int(X_train[0].shape[0]),
                        **metrics,
                    }
                )
                print(f"{model_name} {feature_kind} L={L}: top1={metrics['top1_accuracy']:.3f}", flush=True)

    write_csv(
        run_dir / "metrics" / "speaker_probe_results.csv",
        per_sample_rows,
        ["model", "feature_kind", "L", "split", "speaker_id", "sample_id", "audio_path", "feature_dim"],
    )
    write_csv(
        run_dir / "metrics" / "speaker_probe_summary.csv",
        summary_rows,
        ["model", "feature_kind", "L", "feature_dim", "speaker_count", "test_count", "top1_accuracy", "top5_accuracy", "macro_f1"],
    )
    write_json(
        run_dir / "metrics" / "speaker_probe_results.json",
        {
            "args": vars(args),
            "device": str(device),
            "models": {name: {k: v for k, v in spec.items() if k != "model" and k != "config"} for name, spec in models.items()},
            "summary": summary_rows,
        },
    )
    write_report(run_dir / "reports" / "speaker_probe_summary.md", summary_rows, args)
    return run_dir


def write_report(path: Path, summary_rows: List[dict], args: argparse.Namespace) -> None:
    lines = [
        "# Exp22 Speaker Probe",
        "",
        f"- train_per_speaker: {args.train_per_speaker}",
        f"- test_per_speaker: {args.test_per_speaker}",
        "",
        "| model | feature | L | speakers | test n | dim | top1 | top5 | macro-F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {feature_kind} | {L} | {speaker_count} | {test_count} | {feature_dim} | "
            "{top1_accuracy:.3f} | {top5_accuracy:.3f} | {macro_f1:.3f} |".format(**row)
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Exp22-B speaker probe evaluation")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sample-list", default="")
    parser.add_argument("--audio-root", default="")
    parser.add_argument("--speaker-source", default="auto", choices=["auto", "vctk", "aishell", "librispeech"])
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--base-checkpoint", default=str(DEFAULT_BASE_CKPT))
    parser.add_argument("--lca-config", default=str(DEFAULT_LCA_CONFIG))
    parser.add_argument("--lca-checkpoint", default=str(DEFAULT_LCA_CKPT))
    parser.add_argument("--models", nargs="+", default=["base", "lca"], choices=["base", "lca"])
    parser.add_argument("--feature-kinds", nargs="+", default=["codes_hist", "latent_stats"], choices=["codes_hist", "latent_stats"])
    parser.add_argument("--layers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--max-speakers", type=int, default=0, help="0 means all eligible speakers")
    parser.add_argument("--train-per-speaker", type=int, default=5)
    parser.add_argument("--test-per-speaker", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main():
    run_dir = evaluate(parse_args())
    print(json.dumps({"status": "completed", "run_dir": str(run_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
