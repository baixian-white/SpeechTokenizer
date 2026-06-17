import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import file_sha256, write_csv, write_json, write_text
from scripts.package_exp2_outputs import build_model, parse_sample_id, safe_load_state_dict
from speechtokenizer.trainer.dataset import audioDataset, get_dataloader
from speechtokenizer.trainer.loss import recon_loss


def load_config(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_wav(path, tensor, sample_rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.squeeze(tensor.detach().cpu().float().numpy())
    sf.write(path, arr, sample_rate)


def evaluate_layer_reconstruction(run_dir, config, checkpoint, sample_list, max_samples=3, device="cuda"):
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

    sample_rows = []
    metric_rows = []
    with torch.inference_mode():
        for idx, batch in enumerate(dl):
            x, _semantic = batch
            sample_id = parse_sample_id(sample_lines[idx], idx)
            x = x.unsqueeze(1).to(actual_device)
            codes = model.encode(x, n_q=cfg["n_q"], st=0)
            original_path = run_dir / "samples" / "fixed" / "original" / f"{sample_id}.wav"
            write_wav(original_path, x[0, 0], cfg["sample_rate"])
            sample_row = {
                "sample_id": sample_id,
                "duration_sec": x.shape[-1] / cfg["sample_rate"],
                "original_path": str(original_path),
            }
            for layer_count in range(1, int(cfg["n_q"]) + 1):
                recon = model.decode(codes[:layer_count], st=0)
                recon_path = run_dir / "samples" / "fixed" / f"recon_L{layer_count}" / f"{sample_id}.wav"
                write_wav(recon_path, recon[0, 0], cfg["sample_rate"])
                sample_row[f"recon_L{layer_count}_path"] = str(recon_path)
                metric_rows.append(
                    {
                        "sample_id": sample_id,
                        "L": layer_count,
                        "ideal_bitrate_bps": layer_count * 50 * 10,
                        "recon_l1": float(recon_loss(x, recon).item()),
                        "codes_shape": json.dumps(list(codes[:layer_count].shape)),
                        "output_shape": json.dumps(list(recon.shape)),
                        "duration_sec": x.shape[-1] / cfg["sample_rate"],
                        "note": "sanity/proxy reconstruction, not final communication quality",
                    }
                )
            sample_rows.append(sample_row)

    metric_fields = [
        "sample_id",
        "L",
        "ideal_bitrate_bps",
        "recon_l1",
        "codes_shape",
        "output_shape",
        "duration_sec",
        "note",
    ]
    sample_fields = ["sample_id", "duration_sec", "original_path"] + [
        f"recon_L{i}_path" for i in range(1, int(cfg["n_q"]) + 1)
    ]
    write_csv(run_dir / "metrics" / "layer_reconstruction.csv", metric_rows, metric_fields)
    write_json(
        run_dir / "metrics" / "layer_reconstruction.json",
        {
            "run_id": run_dir.name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "sample_count": len(sample_rows),
            "metrics": metric_rows,
            "scope_note": "Sanity/proxy reconstruction only; not a final communication quality conclusion.",
        },
    )
    write_csv(run_dir / "samples" / "fixed" / "sample_manifest.csv", sample_rows, sample_fields)
    write_text(
        run_dir / "reports" / "layer_reconstruction.md",
        "\n".join(
            [
                "# Layer Reconstruction",
                "",
                f"- run_id: {run_dir.name}",
                f"- checkpoint: {checkpoint}",
                f"- samples: {len(sample_rows)}",
                "- outputs: samples/fixed/original and samples/fixed/recon_L1..recon_L3",
                "- scope: sanity/proxy reconstruction only; do not report as final WER/PESQ/STOI or channel quality.",
                "",
            ]
        ),
    )
    return metric_rows


def build_parser():
    parser = argparse.ArgumentParser(description="Export Exp2 fixed-sample L=1/2/3 reconstructions.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser


def main():
    args = build_parser().parse_args()
    rows = evaluate_layer_reconstruction(
        args.run_dir, args.config, args.checkpoint, args.sample_list, args.max_samples, args.device
    )
    print(json.dumps({"status": "completed", "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
