import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from thop import profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import write_csv, write_json, write_text
from scripts.package_exp2_outputs import build_model, safe_load_state_dict, codebook_stats
from speechtokenizer.trainer.dataset import audioDataset, get_dataloader
from speechtokenizer.trainer.loss import recon_loss


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_wav(path, tensor, sample_rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.squeeze(tensor.detach().cpu().float().numpy()), sample_rate)


def sample_id(line, idx):
    return Path(line.split("\t")[0].lstrip("\ufeff").strip()).stem or f"sample_{idx:03d}"


def load_variant(variant, device):
    cfg = load_json(variant["config_path"])
    model = build_model(cfg)
    safe_load_state_dict(model, variant["checkpoint_path"])
    model.to(device)
    model.eval()
    return cfg, model


def profile_model(model, device):
    dummy = torch.randn(1, 1, 16000, device=device)
    with torch.inference_mode():
        enc_flops, _ = profile(model.encoder, inputs=(dummy,), verbose=False)
        z = model.encoder(dummy)
        total_params = sum(p.numel() for p in model.parameters())
        encoder_params = sum(p.numel() for p in model.encoder.parameters())
        if device.type == "cuda":
            torch.cuda.synchronize()
        tic = time.perf_counter()
        out = model.decode(model.encode(dummy, n_q=3, st=0), st=0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        rtf = time.perf_counter() - tic
    return {
        "encoder_params": int(encoder_params),
        "encoder_macs": int(enc_flops),
        "encoder_rtf": "",
        "total_params": int(total_params),
        "total_macs": "",
        "total_rtf": rtf,
        "profile_sample_sec": 1.0,
        "notes": "encoder_macs computed with THOP for 1s input; total_macs not computed in this debug run",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    matrix = load_json(args.matrix)
    variants = [v for v in matrix["variants"] if v["status"] == "ready"]
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    with open(args.sample_list, "r", encoding="utf-8-sig") as f:
        lines = [line for line in f.readlines() if line.strip()][: args.max_samples]

    results = []
    compute = []
    diagnostics = []
    manifest_rows = []
    for variant in variants:
        cfg, model = load_variant(variant, device)
        compute_row = {"variant_id": variant["variant_id"], **profile_model(model, device)}
        compute.append(compute_row)
        ds = audioDataset(lines, segment_size=cfg["segment_size"], sample_rate=cfg["sample_rate"], downsample_rate=320, valid=True)
        dl = get_dataloader(ds, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
        code_chunks = []
        for idx, batch in enumerate(dl):
            sid = sample_id(lines[idx], idx)
            x, semantic = batch
            x = x.unsqueeze(1).to(device)
            with torch.inference_mode():
                codes = model.encode(x, n_q=cfg["n_q"], st=0)
                code_chunks.append(codes.detach().cpu().numpy())
                for L in (1, 2, 3):
                    tic = time.perf_counter()
                    out = model.decode(codes[:L], st=0)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - tic
                    out_path = run_dir / "samples" / variant["variant_id"] / f"L{L}" / f"{sid}.wav"
                    write_wav(out_path, out[0, 0], cfg["sample_rate"])
                    duration = x.shape[-1] / cfg["sample_rate"]
                    results.append(
                        {
                            "variant_id": variant["variant_id"],
                            "ablation_group": variant["ablation_group"],
                            "L": L,
                            "channel": "clean",
                            "ideal_bitrate_bps": L * 50 * 10,
                            "WER": "",
                            "CER": "",
                            "STOI": "",
                            "PESQ": "",
                            "ViSQOL": "",
                            "semantic_similarity": "",
                            "recon_l1": float(recon_loss(x, out).item()),
                            "RTF": elapsed / duration if duration > 0 else None,
                            "sample_count": 1,
                            "notes": "Objective audio metrics not executed in this debug run",
                        }
                    )
        all_codes = np.concatenate(code_chunks, axis=1)
        for row in codebook_stats(all_codes, cfg["codebook_size"]):
            diagnostics.append({"variant_id": variant["variant_id"], **row})
        manifest_rows.append(
            {
                "variant_id": variant["variant_id"],
                "ablation_group": variant["ablation_group"],
                "parent_run_id": variant.get("parent_run_id", ""),
                "changed_factor": variant["changed_factor"],
                "fixed_factors": variant["fixed_factors"],
                "config_path": variant["config_path"],
                "checkpoint_path": variant["checkpoint_path"],
                "status": "completed",
                "notes": variant.get("notes", ""),
            }
        )

    result_fields = ["variant_id", "ablation_group", "L", "channel", "ideal_bitrate_bps", "WER", "CER", "STOI", "PESQ", "ViSQOL", "semantic_similarity", "recon_l1", "RTF", "sample_count", "notes"]
    compute_fields = ["variant_id", "encoder_params", "encoder_macs", "encoder_rtf", "total_params", "total_macs", "total_rtf", "profile_sample_sec", "notes"]
    diag_fields = ["variant_id", "layer", "codebook_size", "used_codes", "usage_rate", "perplexity", "dead_codes", "dead_code_ratio", "top1_frequency"]
    manifest_fields = ["variant_id", "ablation_group", "parent_run_id", "changed_factor", "fixed_factors", "config_path", "checkpoint_path", "status", "notes"]
    write_csv(run_dir / "metrics" / "ablation_results.csv", results, result_fields)
    write_json(run_dir / "metrics" / "ablation_results.json", results)
    write_csv(run_dir / "metrics" / "results.csv", results, result_fields)
    write_json(run_dir / "metrics" / "results.json", {"ablation_results": results, "compute_profile": compute, "codebook_diagnostics": diagnostics})
    write_csv(run_dir / "metrics" / "compute_profile.csv", compute, compute_fields)
    write_json(run_dir / "metrics" / "compute_profile.json", compute)
    write_csv(run_dir / "metrics" / "codebook_diagnostics.csv", diagnostics, diag_fields)
    write_json(run_dir / "metrics" / "codebook_diagnostics.json", diagnostics)
    write_csv(run_dir / "metrics" / "variant_manifest.csv", manifest_rows, manifest_fields)

    md = ["# Variant Manifest", ""]
    for row in manifest_rows:
        md.append(f"- {row['variant_id']} ({row['ablation_group']}): changed={row['changed_factor']}; status={row['status']}; notes={row['notes']}")
    write_text(run_dir / "reports" / "variant_manifest.md", "\n".join(md) + "\n")
    write_text(run_dir / "reports" / "ablation_comparison.md", "# Ablation Comparison\n\nShort debug/tracer ablation evaluation completed. Objective WER/STOI/PESQ metrics were not executed; recon_l1, compute profile, and codebook diagnostics are provided for inspection.\n")
    write_text(run_dir / "reports" / "summary.md", "# Experiment 5 Summary\n\n- status: completed\n- scope: minimum debug/tracer ablation matrix.\n- warning: not a full paper-grade ablation result.\n- metrics: metrics/ablation_results.csv, metrics/compute_profile.csv, metrics/codebook_diagnostics.csv\n")
    write_json(run_dir / "reports" / "status.json", {"status": "completed", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "variant_count": len(variants)})
    write_text(run_dir / "reports" / "agent_handoff.md", "# Exp5 Ablation Agent Handoff\n\n- status: completed\n- variants evaluated with shared sample list and L=1/2/3 clean decode.\n")
    write_json(run_dir / "reports" / "agent_status.json", {"agent": "Exp5 Ablation Agent (simulated)", "status": "completed"})
    print(json.dumps({"status": "completed", "variants": len(variants), "rows": len(results)}, indent=2))


if __name__ == "__main__":
    main()

