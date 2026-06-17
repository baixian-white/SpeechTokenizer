"""Compute WER/CER for all decoded wavs in exp4 using Whisper.

For each decoded wav (across all method × setting × sample combinations),
load the ground-truth transcript from LibriSpeech `*.trans.txt`, run Whisper
ASR on the decoded wav, compute WER/CER via jiwer.

Outputs:
  metrics/asr_results.csv / asr_results.json
  per-wav transcripts saved as artifacts/asr_transcripts/{method}/{setting}/{sample_id}.txt
  reports/asr_summary.md
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper

# Ensure Whisper's internal ffmpeg lookup can find the conda-installed ffmpeg.
# whisper.audio.load_audio uses subprocess.run(["ffmpeg", ...]) which requires
# ffmpeg to be on PATH. We prepend the conda env Library/bin so it's found.
_FFMPEG_DIR = r"C:\Users\Windows11\.conda\envs\speechtokenizer\Library\bin"
if _FFMPEG_DIR not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_utils import write_csv, write_json


def load_json(p):
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def load_libri_transcripts(libri_root):
    """Walk LibriSpeech and build {sample_id: transcript} dict from all *.trans.txt."""
    libri_root = Path(libri_root)
    transcripts = {}
    for trans_file in libri_root.rglob("*.trans.txt"):
        with open(trans_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    sid, text = parts
                    transcripts[sid] = text.strip()
    return transcripts


def normalize_text(s):
    """Whisper-style normalization for fair WER comparison: lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--libri-root", default="h:/H-CODE/speechtokenizer/data/SpeechPretrain/LibriSpeech")
    ap.add_argument("--whisper-model", default="base.en", help="base.en is fast and English-only")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    # Load existing audio_quality_results.json to find all (method, setting, sample, decoded_path)
    aq = load_json(run_dir / "metrics" / "audio_quality_results.json")
    rows = aq["rows"]
    print(f"Total wavs to ASR: {len(rows)}", flush=True)

    # Load ground-truth transcripts
    transcripts = load_libri_transcripts(args.libri_root)
    print(f"Loaded {len(transcripts)} ground-truth transcripts from LibriSpeech")

    # Load Whisper
    print(f"Loading Whisper model: {args.whisper_model}")
    device = args.device if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(args.whisper_model, device=device)

    # Also ASR the originals (for "Whisper-vs-original WER" reference baseline)
    sample_ids = sorted({r["sample_id"] for r in rows})
    original_dir = run_dir / "samples" / "original"

    # Pre-load originals' transcripts via Whisper
    original_hyp = {}
    print("\n--- ASR on originals (reference for codec-induced WER delta) ---")
    for sid in sample_ids:
        wav = original_dir / f"{sid}.wav"
        if not wav.exists():
            continue
        result = model.transcribe(str(wav), language="en", verbose=False, condition_on_previous_text=False)
        original_hyp[sid] = result["text"].strip()
        print(f"  {sid}: {original_hyp[sid][:80]}...")

    # Now ASR each decoded wav
    print("\n--- ASR on decoded wavs ---")
    import jiwer
    transformations = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    cer_transforms = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ])

    asr_dir = run_dir / "artifacts" / "asr_transcripts"
    asr_dir.mkdir(parents=True, exist_ok=True)

    asr_rows = []
    for i, r in enumerate(rows):
        wav = Path(r["decoded_path"])
        if not wav.exists():
            print(f"  [{i+1}/{len(rows)}] missing: {wav}")
            continue
        method = r["method"]
        setting = r["codec_setting"]
        sid = r["sample_id"]
        gt = transcripts.get(sid)
        if not gt:
            print(f"  [{i+1}/{len(rows)}] no GT for {sid}")
            continue

        # ASR
        result = model.transcribe(str(wav), language="en", verbose=False, condition_on_previous_text=False)
        hyp = result["text"].strip()

        # WER/CER vs ground truth
        try:
            wer_gt = jiwer.wer([gt], [hyp], reference_transform=transformations, hypothesis_transform=transformations)
            cer_gt = jiwer.cer([gt], [hyp], reference_transform=cer_transforms, hypothesis_transform=cer_transforms)
        except Exception as e:
            wer_gt = float("nan"); cer_gt = float("nan")

        # WER/CER vs original-Whisper (codec-induced delta)
        ref_hyp = original_hyp.get(sid, gt)
        try:
            wer_codec = jiwer.wer([ref_hyp], [hyp], reference_transform=transformations, hypothesis_transform=transformations)
            cer_codec = jiwer.cer([ref_hyp], [hyp], reference_transform=cer_transforms, hypothesis_transform=cer_transforms)
        except Exception:
            wer_codec = float("nan"); cer_codec = float("nan")

        # Save transcripts
        out_path = asr_dir / method / setting / f"{sid}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            f"# method={method} setting={setting} sample_id={sid}\n"
            f"# GT (LibriSpeech):\n{gt}\n\n"
            f"# Whisper(original):\n{ref_hyp}\n\n"
            f"# Whisper(decoded):\n{hyp}\n",
            encoding="utf-8",
        )

        asr_row = {
            "method": method,
            "model": r.get("model", ""),
            "L": r.get("L", ""),
            "codec_setting": setting,
            "sample_id": sid,
            "decoded_path": str(wav),
            "wer_vs_gt": float(wer_gt),
            "cer_vs_gt": float(cer_gt),
            "wer_vs_original_whisper": float(wer_codec),
            "cer_vs_original_whisper": float(cer_codec),
            "hypothesis_text": hyp,
            "transcript_path": str(out_path),
        }
        asr_rows.append(asr_row)
        if (i + 1) % 20 == 0 or i == len(rows) - 1:
            print(f"  [{i+1}/{len(rows)}] {method} {setting} {sid}: WER {wer_gt:.3f} CER {cer_gt:.3f}", flush=True)

    # Save outputs
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fields = ["method", "model", "L", "codec_setting", "sample_id",
              "wer_vs_gt", "cer_vs_gt", "wer_vs_original_whisper", "cer_vs_original_whisper",
              "hypothesis_text",
              "decoded_path", "transcript_path"]
    write_csv(metrics_dir / "asr_results.csv", asr_rows, fields)
    write_json(metrics_dir / "asr_results.json", {
        "run_id": run_dir.name,
        "whisper_model": args.whisper_model,
        "ground_truth_source": "LibriSpeech *.trans.txt",
        "original_whisper_transcripts": original_hyp,
        "rows": asr_rows,
    })
    print(json.dumps({"status": "completed", "rows": len(asr_rows), "csv": str(metrics_dir / "asr_results.csv")}, indent=2))


if __name__ == "__main__":
    main()
