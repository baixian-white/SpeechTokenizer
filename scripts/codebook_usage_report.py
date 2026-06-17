import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import file_sha256, write_csv, write_json, write_text
from scripts.package_exp2_outputs import build_model, codebook_stats, safe_load_state_dict
from speechtokenizer.trainer.dataset import audioDataset, get_dataloader


def load_config(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def codebook_usage_report(run_dir, config, checkpoint, sample_list, max_samples=32, device="cuda"):
    run_dir = Path(run_dir)
    cfg = load_config(config)
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not cfg.get("nas_encoder_config"):
        raise ValueError("Exp2 NAS route requires config['nas_encoder_config'] for NASEncoderOnlySpeechTokenizer")

    actual_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    safe_load_state_dict(model, checkpoint)
    model.to(actual_device)
    model.eval()

    with open(sample_list, "r", encoding="utf-8-sig") as f:
        sample_lines = [line for line in f.readlines() if line.strip()][: int(max_samples)]
    if not sample_lines:
        raise ValueError(f"Sample list is empty: {sample_list}")

    ds = audioDataset(
        sample_lines,
        segment_size=cfg["segment_size"],
        sample_rate=cfg["sample_rate"],
        downsample_rate=320,
        valid=True,
    )
    dl = get_dataloader(ds, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    code_chunks = []
    with torch.inference_mode():
        for batch in dl:
            x, _semantic = batch
            x = x.unsqueeze(1).to(actual_device)
            codes = model.encode(x, n_q=cfg["n_q"], st=0)
            code_chunks.append(codes.detach().cpu().numpy())
    if not code_chunks:
        raise ValueError("No codes were produced from the sample list")

    all_codes = np.concatenate(code_chunks, axis=1)
    rows = codebook_stats(all_codes, int(cfg["codebook_size"]))
    for row in rows:
        row["sample_count"] = len(sample_lines)
        row["note"] = "sanity/proxy codebook usage, not final communication quality"

    fields = [
        "layer",
        "codebook_size",
        "used_codes",
        "usage_rate",
        "perplexity",
        "dead_codes",
        "dead_code_ratio",
        "top1_frequency",
        "sample_count",
        "note",
    ]
    write_csv(run_dir / "metrics" / "codebook_usage.csv", rows, fields)
    write_json(
        run_dir / "metrics" / "codebook_usage.json",
        {
            "run_id": run_dir.name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "sample_count": len(sample_lines),
            "rows": rows,
            "scope_note": "Sanity/proxy codebook usage only; not a final communication quality conclusion.",
        },
    )

    md = ["# Codebook Usage", "", f"- run_id: {run_dir.name}", f"- samples: {len(sample_lines)}", ""]
    for row in rows:
        md.append(
            f"- layer {row['layer']}: usage_rate={row['usage_rate']:.6f}, "
            f"perplexity={row['perplexity']:.6f}, dead_code_ratio={row['dead_code_ratio']:.6f}, "
            f"top1_frequency={row['top1_frequency']:.6f}"
        )
    md.extend(["", "Scope: sanity/proxy diagnostic only; do not report as final channel or codec quality.", ""])
    write_text(run_dir / "reports" / "codebook_usage.md", "\n".join(md))
    return rows


def build_parser():
    parser = argparse.ArgumentParser(description="Compute Exp2 RVQ codebook usage diagnostics.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser


def main():
    args = build_parser().parse_args()
    rows = codebook_usage_report(args.run_dir, args.config, args.checkpoint, args.sample_list, args.max_samples, args.device)
    print(json.dumps({"status": "completed", "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
