"""On-the-fly ASR/WER evaluation for SCIT Base vs LCA.

Reconstructs audio for Base/LCA at L=1/2/3 and immediately runs Whisper ASR,
without saving all reconstructed wavs by default. Designed for LibriSpeech
subset/full intelligibility checks after no-save objective evaluations.
"""
import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper

_FFMPEG_DIR = r"C:\Users\Windows11\.conda\envs\speechtokenizer\Library\bin"
if _FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_lca_vs_base import load_json, build_model, load_state, load_audio_for_inference, read_sample_rows, hash_file, write_wav


def load_libri_transcripts(libri_root):
    transcripts = {}
    for trans_file in Path(libri_root).rglob("*.trans.txt"):
        with trans_file.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1].strip()
    return transcripts


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def reconstruct_clean(model, audio_np, L, n_q, device):
    x = torch.from_numpy(audio_np).to(device).view(1, 1, -1)
    with torch.no_grad():
        codes = model.encode(x, n_q=int(n_q), st=0)
        recon = model.decode(codes[:int(L)].contiguous().long(), st=0)
    return recon[0, 0].detach().cpu().numpy().astype(np.float32)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not np.isnan(value):
            values.append(value)
    return sum(values) / len(values) if values else float("nan")


def write_summary(run_dir, rows, base_ckpt, lca_ckpt, whisper_model):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["model"], row["L"])].append(row)
    lines = [
        "# On-the-fly ASR/WER summary",
        "",
        f"- Whisper: `{whisper_model}`",
        f"- Base: `{base_ckpt}`",
        f"- LCA: `{lca_ckpt}`",
        "",
        "| model | L | n | WER vs GT ↓ | CER vs GT ↓ | WER vs original Whisper ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in ["base", "lca"]:
        for L in [1, 2, 3]:
            rows_here = buckets[(model, L)]
            if not rows_here:
                continue
            lines.append(
                f"| {model} | {L} | {len(rows_here)} | "
                f"{mean(rows_here, 'wer_vs_gt'):.4f} | {mean(rows_here, 'cer_vs_gt'):.4f} | "
                f"{mean(rows_here, 'wer_vs_original_whisper'):.4f} |"
            )
    lines += ["", "## LCA - Base deltas", "", "| L | ΔWER vs GT | ΔCER vs GT | ΔWER vs original Whisper |", "|---:|---:|---:|---:|"]
    for L in [1, 2, 3]:
        base_rows = buckets[("base", L)]
        lca_rows = buckets[("lca", L)]
        if base_rows and lca_rows:
            lines.append(
                f"| {L} | "
                f"{mean(lca_rows, 'wer_vs_gt') - mean(base_rows, 'wer_vs_gt'):+.4f} | "
                f"{mean(lca_rows, 'cer_vs_gt') - mean(base_rows, 'cer_vs_gt'):+.4f} | "
                f"{mean(lca_rows, 'wer_vs_original_whisper') - mean(base_rows, 'wer_vs_original_whisper'):+.4f} |"
            )
    out = run_dir / "reports" / "asr_wer_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--lca-config", required=True)
    parser.add_argument("--lca-checkpoint", required=True)
    parser.add_argument("--sample-list", required=True)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--libri-root", default="data/SpeechPretrain/LibriSpeech")
    parser.add_argument("--whisper-model", default="base.en")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-sample-count", type=int, default=5)
    args = parser.parse_args()

    import jiwer
    word_transform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
    char_transform = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(), jiwer.ReduceToListOfListOfChars()])

    run_dir = Path(args.run_dir)
    base_cfg = load_json(args.base_config)
    lca_cfg = load_json(args.lca_config)
    sample_rate = int(base_cfg.get("sample_rate", 16000))
    n_q = int(base_cfg.get("n_q", 3))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"loading transcripts from {args.libri_root}", flush=True)
    transcripts = load_libri_transcripts(args.libri_root)
    print(f"loaded {len(transcripts)} transcripts", flush=True)

    print(f"loading Base {args.base_checkpoint}", flush=True)
    base_model = build_model(base_cfg).to(device).eval()
    load_state(base_model, args.base_checkpoint)
    print(f"loading LCA {args.lca_checkpoint}", flush=True)
    lca_model = build_model(lca_cfg).to(device).eval()
    load_state(lca_model, args.lca_checkpoint)

    print(f"loading Whisper {args.whisper_model}", flush=True)
    asr_device = args.device if torch.cuda.is_available() else "cpu"
    asr_model = whisper.load_model(args.whisper_model, device=asr_device)

    samples = read_sample_rows(args.sample_list, args.max_samples)
    print(f"evaluating {len(samples)} samples x 3 L x 2 models", flush=True)
    rows = []
    original_hyp = {}
    transcript_dir = run_dir / "artifacts" / "asr_transcripts"
    tic = time.time()

    with tempfile.TemporaryDirectory(prefix="scit_asr_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        for idx, sample in enumerate(samples, 1):
            sample_id = sample["sample_id"]
            gt = transcripts.get(sample_id, "")
            if not gt:
                print(f"skip {sample_id}: missing transcript", flush=True)
                continue
            audio = load_audio_for_inference(sample["audio"], sample_rate).squeeze(0).numpy().astype(np.float32)
            orig_path = tmp_dir / f"{sample_id}_original.wav"
            sf.write(orig_path, audio, sample_rate)
            orig_result = asr_model.transcribe(str(orig_path), language="en", verbose=False, condition_on_previous_text=False)
            original_hyp[sample_id] = orig_result["text"].strip()
            if idx <= max(0, args.save_sample_count):
                write_wav(run_dir / "samples" / "original" / f"{sample_id}.wav", torch.from_numpy(audio), sample_rate)

            for model_name, model in [("base", base_model), ("lca", lca_model)]:
                for L in [1, 2, 3]:
                    recon = reconstruct_clean(model, audio, L, n_q, device)
                    wav_path = tmp_dir / f"{sample_id}_{model_name}_L{L}.wav"
                    sf.write(wav_path, recon, sample_rate)
                    if idx <= max(0, args.save_sample_count):
                        write_wav(run_dir / "samples" / model_name / f"L{L}" / f"{sample_id}.wav", torch.from_numpy(recon), sample_rate)
                    result = asr_model.transcribe(str(wav_path), language="en", verbose=False, condition_on_previous_text=False)
                    hyp = result["text"].strip()
                    wer_gt = jiwer.wer([gt], [hyp], reference_transform=word_transform, hypothesis_transform=word_transform)
                    cer_gt = jiwer.cer([gt], [hyp], reference_transform=char_transform, hypothesis_transform=char_transform)
                    ref_hyp = original_hyp[sample_id]
                    wer_orig = jiwer.wer([ref_hyp], [hyp], reference_transform=word_transform, hypothesis_transform=word_transform)
                    cer_orig = jiwer.cer([ref_hyp], [hyp], reference_transform=char_transform, hypothesis_transform=char_transform)
                    out_txt = transcript_dir / model_name / f"L{L}" / f"{sample_id}.txt"
                    out_txt.parent.mkdir(parents=True, exist_ok=True)
                    out_txt.write_text(f"# sample_id={sample_id}\n# model={model_name}\n# L={L}\n\nGT:\n{gt}\n\nWhisper(original):\n{ref_hyp}\n\nWhisper(decoded):\n{hyp}\n", encoding="utf-8")
                    rows.append({
                        "model": model_name,
                        "L": int(L),
                        "sample_id": sample_id,
                        "wer_vs_gt": float(wer_gt),
                        "cer_vs_gt": float(cer_gt),
                        "wer_vs_original_whisper": float(wer_orig),
                        "cer_vs_original_whisper": float(cer_orig),
                        "ground_truth": gt,
                        "original_whisper": ref_hyp,
                        "hypothesis_text": hyp,
                        "transcript_path": str(out_txt),
                    })
            if idx % 10 == 0 or idx == len(samples):
                print(f"done {idx}/{len(samples)} elapsed={time.time() - tic:.1f}s", flush=True)

    fields = ["model", "L", "sample_id", "wer_vs_gt", "cer_vs_gt", "wer_vs_original_whisper", "cer_vs_original_whisper", "ground_truth", "original_whisper", "hypothesis_text", "transcript_path"]
    csv_path = run_dir / "metrics" / "asr_wer_results.csv"
    json_path = run_dir / "metrics" / "asr_wer_results.json"
    write_csv(csv_path, rows, fields)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "rows": rows,
        "sample_count": len(samples),
        "whisper_model": args.whisper_model,
        "base_checkpoint": args.base_checkpoint,
        "base_sha256": hash_file(args.base_checkpoint),
        "lca_checkpoint": args.lca_checkpoint,
        "lca_sha256": hash_file(args.lca_checkpoint),
        "original_whisper": original_hyp,
    }, ensure_ascii=False), encoding="utf-8")
    summary = write_summary(run_dir, rows, args.base_checkpoint, args.lca_checkpoint, args.whisper_model)
    print(json.dumps({"status": "completed", "rows": len(rows), "csv": str(csv_path), "summary": str(summary)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
