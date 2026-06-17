import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import write_csv, write_json, write_text
from scripts.package_exp2_outputs import build_model, safe_load_state_dict
from speechtokenizer.trainer.loss import recon_loss


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_audio(path, sample_rate):
    try:
        audio, sr = torchaudio.load(path)
    except ImportError as exc:
        if "TorchCodec" not in str(exc):
            raise
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        audio = torch.from_numpy(data.T).contiguous()
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
    return audio.float()


def write_wav(path, tensor, sample_rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.squeeze(tensor.detach().cpu().float().numpy())
    sf.write(path, arr, sample_rate)


def read_sample_rows(sample_list, max_samples):
    rows = []
    with open(sample_list, "r", encoding="utf-8-sig") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            fields = raw.split("\t")
            rows.append({"audio": fields[0].lstrip("\ufeff").strip(), "feature": fields[1].strip() if len(fields) > 1 else ""})
            if len(rows) >= max_samples:
                break
    return rows


def _rel(path, base):
    return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()


def build_listen_html(rows, output_path):
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        "  <title>Exp2 Full Utterance Samples</title>",
        "  <style>",
        "    body { font-family: Arial, sans-serif; margin: 24px; background: #f7f7f5; color: #202124; }",
        "    h1 { font-size: 22px; margin-bottom: 8px; }",
        "    p { max-width: 980px; line-height: 1.45; }",
        "    table { border-collapse: collapse; width: 100%; max-width: 1200px; background: #fff; margin-top: 18px; }",
        "    th, td { border: 1px solid #d6d6d6; padding: 10px; vertical-align: top; }",
        "    th { background: #eceff1; text-align: left; }",
        "    audio { width: 230px; display: block; }",
        "    .metric { color: #3c4043; font-size: 13px; margin-top: 6px; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h1>Exp2 SCIT-Speech-Base Full Utterance Reconstructions</h1>",
        "  <p>These are complete utterance forward passes from the packaged Exp2 checkpoint. They are listening samples only, not WER/PESQ/STOI or channel-quality conclusions.</p>",
        "  <table>",
        "    <thead>",
        "      <tr><th>Sample</th><th>Original</th><th>L1 / 500 bps</th><th>L2 / 1000 bps</th><th>L3 / 1500 bps</th></tr>",
        "    </thead>",
        "    <tbody>",
    ]
    base = Path(output_path).parent
    for row in rows:
        lines.extend(
            [
                "      <tr>",
                f"        <td>{row['sample_id']}<div class=\"metric\">duration={row['duration_sec']:.2f}s</div></td>",
                f"        <td><audio controls src=\"{_rel(row['original_path'], base)}\"></audio></td>",
                f"        <td><audio controls src=\"{_rel(row['recon_L1_path'], base)}\"></audio><div class=\"metric\">recon_l1={row['recon_L1_l1']:.6f}</div></td>",
                f"        <td><audio controls src=\"{_rel(row['recon_L2_path'], base)}\"></audio><div class=\"metric\">recon_l1={row['recon_L2_l1']:.6f}</div></td>",
                f"        <td><audio controls src=\"{_rel(row['recon_L3_path'], base)}\"></audio><div class=\"metric\">recon_l1={row['recon_L3_l1']:.6f}</div></td>",
                "      </tr>",
            ]
        )
    lines.extend(["    </tbody>", "  </table>", "</body>", "</html>", ""])
    write_text(output_path, "\n".join(lines))


def export_full_utterance_samples(run_dir, config, checkpoint, sample_list, max_samples=4, device="cuda"):
    run_dir = Path(run_dir)
    cfg = load_json(config)
    sample_rate = int(cfg["sample_rate"])
    actual_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    safe_load_state_dict(model, checkpoint)
    model.to(actual_device)
    model.eval()

    rows = []
    sample_rows = read_sample_rows(sample_list, int(max_samples))
    if not sample_rows:
        raise ValueError(f"No samples found in {sample_list}")

    with torch.inference_mode():
        for row in sample_rows:
            audio_path = Path(row["audio"])
            sample_id = audio_path.stem
            audio = load_audio(str(audio_path), sample_rate)
            x = audio.unsqueeze(0).to(actual_device)
            codes = model.encode(x, n_q=int(cfg["n_q"]), st=0)

            out_row = {
                "sample_id": sample_id,
                "source_audio": str(audio_path),
                "duration_sec": float(x.shape[-1] / sample_rate),
                "original_path": str(run_dir / "samples" / "full_utterance" / "original" / f"{sample_id}.wav"),
            }
            write_wav(out_row["original_path"], audio[0], sample_rate)
            for layer_count in range(1, int(cfg["n_q"]) + 1):
                recon = model.decode(codes[:layer_count], st=0)
                recon_path = run_dir / "samples" / "full_utterance" / f"recon_L{layer_count}" / f"{sample_id}.wav"
                write_wav(recon_path, recon[0, 0], sample_rate)
                out_row[f"recon_L{layer_count}_path"] = str(recon_path)
                out_row[f"recon_L{layer_count}_l1"] = float(recon_loss(x, recon).item())
                out_row[f"recon_L{layer_count}_shape"] = json.dumps(list(recon.shape))
            rows.append(out_row)

    fields = [
        "sample_id",
        "source_audio",
        "duration_sec",
        "original_path",
        "recon_L1_path",
        "recon_L1_l1",
        "recon_L1_shape",
        "recon_L2_path",
        "recon_L2_l1",
        "recon_L2_shape",
        "recon_L3_path",
        "recon_L3_l1",
        "recon_L3_shape",
    ]
    write_csv(run_dir / "metrics" / "full_utterance_reconstruction.csv", rows, fields)
    write_json(
        run_dir / "metrics" / "full_utterance_reconstruction.json",
        {
            "run_id": run_dir.name,
            "checkpoint": str(checkpoint),
            "sample_count": len(rows),
            "rows": rows,
            "scope_note": "Complete utterance listening/proxy reconstruction only; not final communication quality.",
        },
    )
    html_path = run_dir / "samples" / "full_utterance" / "listen_full.html"
    build_listen_html(rows, html_path)
    return rows, html_path


def build_parser():
    parser = argparse.ArgumentParser(description="Export complete utterance Exp2 listening samples.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser


def main():
    args = build_parser().parse_args()
    rows, html_path = export_full_utterance_samples(
        args.run_dir,
        args.config,
        args.checkpoint,
        args.sample_list,
        args.max_samples,
        args.device,
    )
    print(json.dumps({"status": "completed", "rows": len(rows), "listen_html": str(html_path)}, indent=2))


if __name__ == "__main__":
    main()
