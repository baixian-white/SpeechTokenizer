import argparse
import json
import math
from pathlib import Path

import torch

from speechtokenizer import SpeechTokenizer
from speechtokenizer.trainer.dataset import audioDataset, get_dataloader


def _load_filelist(path: Path, limit: int) -> list[str]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line + "\n")
            if len(rows) >= limit:
                break
    return rows


def _check_fixed_config(cfg: dict) -> dict:
    expected = {
        "sample_rate": 16000,
        "strides": [8, 5, 4, 2],
        "dimension": 1024,
        "n_q": 3,
        "codebook_size": 1024,
    }
    actual = {key: cfg.get(key) for key in expected}
    return {
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
    }


def run_smoke(config_path: Path, train_file: Path, valid_file: Path, device_name: str) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    fixed = _check_fixed_config(cfg)
    if not fixed["passed"]:
        raise RuntimeError(f"fixed config mismatch: {fixed}")

    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = SpeechTokenizer(cfg).to(device)
    model.eval()

    train_rows = _load_filelist(train_file, limit=max(2, int(cfg.get("batch_size", 1))))
    valid_rows = _load_filelist(valid_file, limit=2)
    if not train_rows:
        raise RuntimeError("empty train file list")
    if not valid_rows:
        raise RuntimeError("empty valid file list")

    ds = audioDataset(
        file_list=train_rows,
        segment_size=int(cfg.get("segment_size", 16000)),
        sample_rate=int(cfg["sample_rate"]),
        downsample_rate=int(math.prod(cfg["strides"])),
    )
    dl = get_dataloader(ds, batch_size=1, shuffle=False, num_workers=0)
    x, semantic = next(iter(dl))
    x = x.unsqueeze(1).to(device)
    semantic = semantic.to(device)

    with torch.inference_mode():
        out, commit_loss, feature = model(x)
        codes_all = model.encode(x, n_q=cfg["n_q"])
        layer_decode = {}
        for L in (1, 2, 3):
            decoded = model.decode(codes_all[:L])
            layer_decode[f"L{L}"] = {
                "codes_shape": list(codes_all[:L].shape),
                "decoded_shape": list(decoded.shape),
                "decoded_finite": bool(torch.isfinite(decoded).all().item()),
                "decoded_abs_mean": float(decoded.abs().mean().detach().cpu().item()),
            }

    duration_sec = x.shape[-1] / cfg["sample_rate"]
    latent_steps = codes_all.shape[-1]
    payload = {}
    for L in (1, 2, 3):
        bits_per_index = math.ceil(math.log2(cfg["codebook_size"]))
        total_indices = L * latent_steps
        ideal_bits = total_indices * bits_per_index
        payload[f"L{L}"] = {
            "bits_per_index": bits_per_index,
            "latent_steps": int(latent_steps),
            "duration_sec": duration_sec,
            "ideal_bits": int(ideal_bits),
            "ideal_bitrate_bps": ideal_bits / duration_sec,
            "expected_manual_bps": L * 50 * bits_per_index,
        }

    return {
        "status": "passed",
        "config_path": str(config_path),
        "train_file": str(train_file),
        "valid_file": str(valid_file),
        "device": str(device),
        "fixed_config": fixed,
        "train_rows_checked": len(train_rows),
        "valid_rows_checked": len(valid_rows),
        "batch_audio_shape": list(x.shape),
        "batch_semantic_shape": list(semantic.shape),
        "forward": {
            "out_shape": list(out.shape),
            "commit_loss_finite": bool(torch.isfinite(commit_loss).all().item()),
            "feature_shape": list(feature.shape),
            "output_finite": bool(torch.isfinite(out).all().item()),
        },
        "codes_all_shape": list(codes_all.shape),
        "layer_decode": layer_decode,
        "payload_toy": payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal non-training smoke checks for SCIT experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--json_out", required=True)
    parser.add_argument("--md_out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        result = run_smoke(
            config_path=Path(args.config),
            train_file=Path(args.train_file),
            valid_file=Path(args.valid_file),
            device_name=args.device,
        )
    except Exception as exc:
        result = {"status": "failed", "error": repr(exc)}

    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = ["# Preflight Smoke Test", "", f"- status: {result['status']}"]
    if result["status"] == "failed":
        lines.append(f"- error: `{result['error']}`")
    else:
        lines.extend(
            [
                f"- device: `{result['device']}`",
                f"- audio_shape: `{result['batch_audio_shape']}`",
                f"- semantic_shape: `{result['batch_semantic_shape']}`",
                f"- codes_all_shape: `{result['codes_all_shape']}`",
                "",
                "## Layer Decode",
            ]
        )
        for key, value in result["layer_decode"].items():
            lines.append(f"- {key}: codes `{value['codes_shape']}`, decoded `{value['decoded_shape']}`, finite `{value['decoded_finite']}`")
        lines.extend(["", "## Payload Toy"])
        for key, value in result["payload_toy"].items():
            lines.append(f"- {key}: ideal_bitrate_bps `{value['ideal_bitrate_bps']}`, expected_manual_bps `{value['expected_manual_bps']}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
