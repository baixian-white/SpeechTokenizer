import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import write_csv, write_json, write_text, file_sha256
from speechtokenizer.model import SpeechTokenizer
from speechtokenizer.trainer.dataset import audioDataset, get_dataloader
from speechtokenizer.trainer.loss import recon_loss


def load_config(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_model(cfg):
    if cfg.get("nas_encoder_config"):
        from nas.encoder_only_model_variant import NASEncoderOnlySpeechTokenizer

        return NASEncoderOnlySpeechTokenizer(cfg, cfg["nas_encoder_config"])
    return SpeechTokenizer(cfg)


def write_wav(path, tensor, sample_rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().cpu().float().numpy()
    arr = np.squeeze(arr)
    sf.write(path, arr, sample_rate)


def safe_load_state_dict(model, ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    model.load_state_dict(state)


def codebook_stats(all_codes, codebook_size):
    rows = []
    for layer_idx in range(all_codes.shape[0]):
        flat = all_codes[layer_idx].reshape(-1)
        counts = np.bincount(flat, minlength=codebook_size)
        used = int((counts > 0).sum())
        probs = counts[counts > 0].astype(np.float64)
        probs = probs / probs.sum() if probs.sum() else probs
        entropy = float(-(probs * np.log(probs + 1e-12)).sum()) if probs.size else 0.0
        perplexity = float(np.exp(entropy))
        top1_frequency = float(counts.max() / counts.sum()) if counts.sum() else 0.0
        rows.append(
            {
                "layer": layer_idx + 1,
                "codebook_size": codebook_size,
                "used_codes": used,
                "usage_rate": used / codebook_size,
                "perplexity": perplexity,
                "dead_codes": codebook_size - used,
                "dead_code_ratio": (codebook_size - used) / codebook_size,
                "top1_frequency": top1_frequency,
            }
        )
    return rows


def parse_sample_id(line, idx):
    audio = Path(line.split("\t")[0].lstrip("\ufeff").strip())
    return audio.stem if audio.name else f"sample_{idx:03d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    best_ckpt = run_dir / "checkpoints" / "SCIT-Speech-Base_best.pt"
    if Path(args.checkpoint).resolve() != best_ckpt.resolve():
        shutil.copy2(args.checkpoint, best_ckpt)

    model = build_model(cfg)
    safe_load_state_dict(model, best_ckpt)
    model.to(device)
    model.eval()

    with open(args.sample_list, "r", encoding="utf-8-sig") as f:
        sample_lines = [line for line in f.readlines() if line.strip()][: args.max_samples]
    ds = audioDataset(sample_lines, segment_size=cfg["segment_size"], sample_rate=cfg["sample_rate"], downsample_rate=320, valid=True)
    dl = get_dataloader(ds, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    sample_rows = []
    metric_rows = []
    all_code_chunks = []
    with torch.inference_mode():
        for idx, batch in enumerate(dl):
            x, semantic = batch
            sample_id = parse_sample_id(sample_lines[idx], idx)
            x = x.unsqueeze(1).to(device)
            codes = model.encode(x, n_q=cfg["n_q"], st=0)
            all_code_chunks.append(codes.detach().cpu().numpy())
            original_path = run_dir / "samples" / "fixed" / "original" / f"{sample_id}.wav"
            write_wav(original_path, x[0, 0], cfg["sample_rate"])

            row = {
                "sample_id": sample_id,
                "duration_sec": x.shape[-1] / cfg["sample_rate"],
                "original_path": str(original_path),
            }
            for L in (1, 2, 3):
                recon = model.decode(codes[:L], st=0)
                recon_path = run_dir / "samples" / "fixed" / f"recon_L{L}" / f"{sample_id}.wav"
                write_wav(recon_path, recon[0, 0], cfg["sample_rate"])
                loss = float(recon_loss(x, recon).item())
                row[f"recon_L{L}_path"] = str(recon_path)
                metric_rows.append(
                    {
                        "sample_id": sample_id,
                        "L": L,
                        "ideal_bitrate_bps": L * 50 * 10,
                        "recon_l1": loss,
                        "codes_shape": json.dumps(list(codes[:L].shape)),
                        "output_shape": json.dumps(list(recon.shape)),
                        "duration_sec": x.shape[-1] / cfg["sample_rate"],
                    }
                )
            sample_rows.append(row)

    all_codes = np.concatenate(all_code_chunks, axis=1)
    usage_rows = codebook_stats(all_codes, cfg["codebook_size"])

    write_csv(
        run_dir / "metrics" / "layer_reconstruction.csv",
        metric_rows,
        ["sample_id", "L", "ideal_bitrate_bps", "recon_l1", "codes_shape", "output_shape", "duration_sec"],
    )
    write_json(run_dir / "metrics" / "layer_reconstruction.json", metric_rows)
    write_csv(
        run_dir / "metrics" / "codebook_usage.csv",
        usage_rows,
        ["layer", "codebook_size", "used_codes", "usage_rate", "perplexity", "dead_codes", "dead_code_ratio", "top1_frequency"],
    )
    write_json(run_dir / "metrics" / "codebook_usage.json", usage_rows)
    write_csv(
        run_dir / "metrics" / "results.csv",
        metric_rows,
        ["sample_id", "L", "ideal_bitrate_bps", "recon_l1", "codes_shape", "output_shape", "duration_sec"],
    )
    write_json(
        run_dir / "metrics" / "results.json",
        {
            "layer_reconstruction": metric_rows,
            "codebook_usage": usage_rows,
            "checkpoint": str(best_ckpt),
            "checkpoint_sha256": file_sha256(best_ckpt),
        },
    )
    write_csv(
        run_dir / "samples" / "sample_manifest.csv",
        sample_rows,
        ["sample_id", "duration_sec", "original_path", "recon_L1_path", "recon_L2_path", "recon_L3_path"],
    )
    write_json(
        run_dir / "checkpoints" / "checkpoint_manifest.json",
        {
            "best_checkpoint": str(best_ckpt),
            "sha256": file_sha256(best_ckpt),
            "source_checkpoint": str(args.checkpoint),
            "config": str(args.config),
            "note": "Short debug/tracer training checkpoint; not a full 60 epoch SCIT-Speech-Base run.",
        },
    )

    usage_md = ["# Codebook Usage", ""]
    for row in usage_rows:
        usage_md.append(
            f"- layer {row['layer']}: usage_rate={row['usage_rate']:.6f}, perplexity={row['perplexity']:.6f}, dead_code_ratio={row['dead_code_ratio']:.6f}"
        )
    write_text(run_dir / "reports" / "codebook_usage.md", "\n".join(usage_md) + "\n")
    write_text(
        run_dir / "reports" / "layer_reconstruction.md",
        "# Layer Reconstruction\n\nGenerated original and L=1/2/3 reconstruction samples for the fixed debug sample list.\n",
    )
    write_text(
        run_dir / "reports" / "summary.md",
        "\n".join(
            [
                "# Experiment 2 Summary",
                "",
                "- status: partial",
                f"- run_id: {run_dir.name}",
                "- route: NAS encoder-only proxy route from Exp1; decoder fixed.",
                "- training scope: short monitored debug/tracer run with max_train_steps=2.",
                "- warning: this is not the full planned 60 epoch SCIT-Speech-Base training run and must not be reported as the final Base conclusion.",
                f"- checkpoint: checkpoints/SCIT-Speech-Base_best.pt",
                f"- layer samples: samples/fixed/recon_L1, recon_L2, recon_L3",
                f"- codebook usage: metrics/codebook_usage.csv",
                "",
            ]
        ),
    )
    write_json(
        run_dir / "reports" / "status.json",
        {
            "status": "partial",
            "run_id": run_dir.name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "checkpoint": "checkpoints/SCIT-Speech-Base_best.pt",
            "route": "short_debug_tracer_nas_encoder_only",
            "warning": "debug/tracer packaging output, not completed formal Exp2 training",
        },
    )
    write_text(
        run_dir / "reports" / "agent_handoff.md",
        "# Exp2 Base Training Agent Handoff\n\n- status: partial\n- checkpoint: checkpoints/SCIT-Speech-Base_best.pt\n- warning: short debug/tracer run, not full training conclusion.\n",
    )
    write_json(run_dir / "reports" / "agent_status.json", {"agent": "Exp2 Base Training Agent (simulated)", "status": "partial"})
    print(json.dumps({"status": "partial", "checkpoint": str(best_ckpt)}, indent=2))


if __name__ == "__main__":
    main()
