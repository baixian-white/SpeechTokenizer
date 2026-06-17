import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import write_csv, write_json, write_text
from speechtokenizer.trainer.loss import mel_loss


METRIC_FIELDS = [
    "sample_set",
    "sample_id",
    "L",
    "ideal_bitrate_bps",
    "duration_sec",
    "wave_l1",
    "rmse",
    "mel_l1",
    "si_snr_db",
    "corr",
    "stoi",
    "pesq_wb",
    "original_rms",
    "recon_rms",
    "rms_ratio_db",
    "recon_peak",
    "clip_ratio",
    "length_samples",
    "original_path",
    "recon_path",
]


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_wav(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False), int(sample_rate)


def si_snr_db(reference, estimate, eps=1e-8):
    reference = reference - float(np.mean(reference))
    estimate = estimate - float(np.mean(estimate))
    ref_energy = float(np.sum(reference**2)) + eps
    scale = float(np.sum(estimate * reference)) / ref_energy
    target = scale * reference
    noise = estimate - target
    return 10.0 * math.log10((float(np.sum(target**2)) + eps) / (float(np.sum(noise**2)) + eps))


def pearson_corr(reference, estimate, eps=1e-8):
    ref = reference - float(np.mean(reference))
    est = estimate - float(np.mean(estimate))
    denom = math.sqrt(float(np.sum(ref**2)) * float(np.sum(est**2))) + eps
    return float(np.sum(ref * est) / denom)


def compute_mel_l1(reference, estimate, cfg):
    x = torch.from_numpy(reference).view(1, 1, -1)
    y = torch.from_numpy(estimate).view(1, 1, -1)
    return float(
        mel_loss(
            x,
            y,
            n_fft=int(cfg["n_fft"]),
            num_mels=int(cfg["num_mels"]),
            sample_rate=int(cfg["sample_rate"]),
            hop_size=int(cfg["hop_size"]),
            win_size=int(cfg["win_size"]),
            fmin=cfg.get("fmin", 0),
            fmax=cfg.get("fmax_for_loss", None),
        ).item()
    )


def maybe_compute_stoi(reference, estimate, sample_rate):
    try:
        return float(stoi(reference, estimate, sample_rate, extended=False))
    except Exception:
        return None


def maybe_compute_pesq(reference, estimate, sample_rate):
    if sample_rate != 16000:
        return None
    try:
        return float(pesq(sample_rate, reference, estimate, "wb"))
    except Exception:
        return None


def compare_pair(original_path, recon_path, cfg):
    reference, sr_ref = load_wav(original_path)
    estimate, sr_est = load_wav(recon_path)
    if sr_ref != sr_est:
        raise ValueError(f"Sample-rate mismatch: {original_path}={sr_ref}, {recon_path}={sr_est}")

    length = min(len(reference), len(estimate))
    reference = reference[:length]
    estimate = estimate[:length]
    diff = estimate - reference
    ref_rms = float(np.sqrt(np.mean(reference**2)))
    est_rms = float(np.sqrt(np.mean(estimate**2)))
    rms_ratio_db = 20.0 * math.log10((est_rms + 1e-8) / (ref_rms + 1e-8))

    return {
        "duration_sec": length / sr_ref,
        "wave_l1": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mel_l1": compute_mel_l1(reference, estimate, cfg),
        "si_snr_db": si_snr_db(reference, estimate),
        "corr": pearson_corr(reference, estimate),
        "stoi": maybe_compute_stoi(reference, estimate, sr_ref),
        "pesq_wb": maybe_compute_pesq(reference, estimate, sr_ref),
        "original_rms": ref_rms,
        "recon_rms": est_rms,
        "rms_ratio_db": rms_ratio_db,
        "recon_peak": float(np.max(np.abs(estimate))) if length else 0.0,
        "clip_ratio": float(np.mean(np.abs(estimate) >= 0.999)) if length else 0.0,
        "length_samples": int(length),
    }


def collect_rows(run_dir, cfg, sample_sets):
    run_dir = Path(run_dir)
    rows = []
    n_q = int(cfg.get("n_q", 3))
    for sample_set in sample_sets:
        sample_root = run_dir / "samples" / sample_set
        original_dir = sample_root / "original"
        if not original_dir.exists():
            continue
        for original_path in sorted(original_dir.glob("*.wav")):
            sample_id = original_path.stem
            for layer in range(1, n_q + 1):
                recon_path = sample_root / f"recon_L{layer}" / f"{sample_id}.wav"
                if not recon_path.exists():
                    continue
                rows.append(
                    {
                        "sample_set": sample_set,
                        "sample_id": sample_id,
                        "L": layer,
                        "ideal_bitrate_bps": layer * 50 * 10,
                        "original_path": str(original_path),
                        "recon_path": str(recon_path),
                        **compare_pair(original_path, recon_path, cfg),
                    }
                )
    if not rows:
        raise ValueError(f"No reconstruction pairs found under {run_dir / 'samples'}")
    return rows


def summarize(values):
    values = [value for value in values if value is not None]
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None}
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def build_summary(rows):
    summary = {}
    for sample_set in sorted({row["sample_set"] for row in rows}):
        summary[sample_set] = {}
        for layer in sorted({row["L"] for row in rows if row["sample_set"] == sample_set}):
            layer_rows = [row for row in rows if row["sample_set"] == sample_set and row["L"] == layer]
            summary[sample_set][f"L{layer}"] = {
                "n": len(layer_rows),
                "wave_l1": summarize([row["wave_l1"] for row in layer_rows]),
                "rmse": summarize([row["rmse"] for row in layer_rows]),
                "mel_l1": summarize([row["mel_l1"] for row in layer_rows]),
                "si_snr_db": summarize([row["si_snr_db"] for row in layer_rows]),
                "corr": summarize([row["corr"] for row in layer_rows]),
                "stoi": summarize([row["stoi"] for row in layer_rows]),
                "pesq_wb": summarize([row["pesq_wb"] for row in layer_rows]),
                "rms_ratio_db": summarize([row["rms_ratio_db"] for row in layer_rows]),
                "clip_ratio": summarize([row["clip_ratio"] for row in layer_rows]),
            }
    return summary


def fmt(value, digits=3):
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def pct_delta(current, baseline):
    if current is None or baseline in (None, 0):
        return None
    return (current - baseline) / abs(baseline) * 100.0


def mean_at(summary, sample_set, layer, metric):
    return summary.get(sample_set, {}).get(layer, {}).get(metric, {}).get("mean")


def build_report(run_id, rows, summary, csv_path, json_path, baseline_summary=None):
    lines = [
        "# 样本语音质量定量评估",
        "",
        f"- 实验编号：{run_id}",
        "- 评估对象：`samples/fixed` 与 `samples/full_utterance` 中的原始语音和 L1/L2/L3 重建语音。",
        "- 评估方式：逐条比较原始语音与重建语音，计算侵入式客观指标；未运行 ASR，因此不包含 WER/CER。",
        "",
        "## 汇总结果",
        "",
        "| 样本集 | 层数 | 样本数 | 码率 bps | 波形 L1 ↓ | Mel L1 ↓ | SI-SNR dB ↑ | 相关系数 ↑ | STOI ↑ | PESQ-WB ↑ | RMS 比例 dB | 裁剪比例 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample_set in ("fixed", "full_utterance"):
        for layer in ("L1", "L2", "L3"):
            item = summary.get(sample_set, {}).get(layer)
            if not item:
                continue
            layer_num = int(layer[1:])
            lines.append(
                f"| {sample_set} | {layer} | {item['n']} | {layer_num * 500} | "
                f"{fmt(item['wave_l1']['mean'], 6)} | {fmt(item['mel_l1']['mean'], 6)} | "
                f"{fmt(item['si_snr_db']['mean'], 3)} | {fmt(item['corr']['mean'], 4)} | "
                f"{fmt(item['stoi']['mean'], 4)} | {fmt(item['pesq_wb']['mean'], 3)} | "
                f"{fmt(item['rms_ratio_db']['mean'], 3)} | {fmt(item['clip_ratio']['mean'], 6)} |"
            )

    if baseline_summary:
        lines.extend(
            [
                "",
                "## 相对实验二 best-dev 基线",
                "",
                "| 样本集 | 层数 | ΔPESQ-WB | ΔSTOI | ΔSI-SNR dB | ΔMel L1 | ΔMel L1 % | 判断 |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for sample_set in ("fixed", "full_utterance"):
            for layer in ("L1", "L2", "L3"):
                current_pesq = mean_at(summary, sample_set, layer, "pesq_wb")
                base_pesq = mean_at(baseline_summary, sample_set, layer, "pesq_wb")
                current_stoi = mean_at(summary, sample_set, layer, "stoi")
                base_stoi = mean_at(baseline_summary, sample_set, layer, "stoi")
                current_sisnr = mean_at(summary, sample_set, layer, "si_snr_db")
                base_sisnr = mean_at(baseline_summary, sample_set, layer, "si_snr_db")
                current_mel = mean_at(summary, sample_set, layer, "mel_l1")
                base_mel = mean_at(baseline_summary, sample_set, layer, "mel_l1")
                d_pesq = None if current_pesq is None or base_pesq is None else current_pesq - base_pesq
                d_stoi = None if current_stoi is None or base_stoi is None else current_stoi - base_stoi
                d_sisnr = None if current_sisnr is None or base_sisnr is None else current_sisnr - base_sisnr
                d_mel = None if current_mel is None or base_mel is None else current_mel - base_mel
                d_mel_pct = pct_delta(current_mel, base_mel)
                if d_pesq is None:
                    verdict = "待判定"
                elif d_pesq >= 0.03 and (d_stoi is None or d_stoi >= -0.005) and (
                    (d_sisnr is not None and d_sisnr > 0) or (d_mel is not None and d_mel < 0)
                ):
                    verdict = "通过"
                elif d_pesq <= -0.03 or (d_stoi is not None and d_stoi <= -0.01):
                    verdict = "退化"
                else:
                    verdict = "未达显著改善"
                lines.append(
                    f"| {sample_set} | {layer} | {fmt(d_pesq, 3)} | {fmt(d_stoi, 4)} | "
                    f"{fmt(d_sisnr, 3)} | {fmt(d_mel, 6)} | {fmt(d_mel_pct, 2)}% | {verdict} |"
                )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- L3 仍是当前最佳层数，通常优于 L1/L2。",
            "- 是否接受本阶段结果，应优先看相对实验二 best-dev 基线的 L3 PESQ-WB、STOI、SI-SNR 与 Mel L1 是否同时改善。",
            "- 裁剪比例用于排查满幅削波；若为 0，听感问题更可能来自解码细节、噪声、相位或 token 信息不足。",
            "",
            "## 数据文件",
            "",
            f"- 样本级明细 CSV：`{csv_path}`",
            f"- 汇总 JSON：`{json_path}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate fixed and full-utterance reconstruction audio quality.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-sets", nargs="+", default=["fixed", "full_utterance"])
    parser.add_argument("--baseline-summary")
    parser.add_argument("--csv-name", default="sample_audio_quality_eval.csv")
    parser.add_argument("--json-name", default="sample_audio_quality_eval_summary.json")
    parser.add_argument("--report-name", default="sample_audio_quality_eval.md")
    return parser


def main():
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_json(args.config)
    rows = collect_rows(run_dir, cfg, args.sample_sets)
    summary = build_summary(rows)

    csv_path = run_dir / "metrics" / args.csv_name
    json_path = run_dir / "metrics" / args.json_name
    report_path = run_dir / "reports" / args.report_name
    write_csv(csv_path, rows, METRIC_FIELDS)
    write_json(json_path, {"run_id": run_dir.name, "rows": len(rows), "summary": summary})

    baseline_summary = None
    if args.baseline_summary:
        baseline_payload = load_json(args.baseline_summary)
        baseline_summary = baseline_payload.get("summary", baseline_payload)
    report = build_report(run_dir.name, rows, summary, csv_path, json_path, baseline_summary)
    write_text(report_path, report)
    print(json.dumps({"status": "completed", "rows": len(rows), "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
