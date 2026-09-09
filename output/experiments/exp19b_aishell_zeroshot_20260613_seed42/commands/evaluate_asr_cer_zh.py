#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp19b: Chinese ASR/CER eval on AISHELL — Base vs LCA zero-shot.
Adapted from evaluate_asr_wer_onthefly.py: multilingual Whisper, language='zh', CER primary
(Chinese is character-based; we strip whitespace and compute char error rate vs AISHELL
ground-truth transcript). Also CER vs original-audio Whisper hyp (intelligibility-preservation).

Run: conda run -n speechtokenizer python <this file> --whisper-model medium ...
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_lca_vs_base import load_json, build_model, load_state, load_audio_for_inference, read_sample_rows  # noqa: E402


def load_aishell_transcripts(path):
    """aishell_transcript_v0.8.txt: '<utt_id> 汉字 汉字 ...' per line. Return {utt_id: '汉字汉字...'} (no spaces)."""
    tr = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                tr[parts[0]] = parts[1].replace(" ", "")
    return tr


def reconstruct_clean(model, audio_np, L, n_q, device):
    x = torch.from_numpy(audio_np).to(device).view(1, 1, -1)
    with torch.no_grad():
        codes = model.encode(x, n_q=int(n_q), st=0)
        recon = model.decode(codes[: int(L)].contiguous().long(), st=0)
    return recon[0, 0].detach().cpu().numpy().astype(np.float32)


def cer_zh(ref, hyp):
    """Character error rate for Chinese: compare as char sequences, ignore spaces/punct-lite."""
    import jiwer
    r = ref.replace(" ", "")
    h = hyp.replace(" ", "")
    if not r:
        return float("nan")
    return jiwer.cer(r, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--lca-config", required=True)
    ap.add_argument("--lca-checkpoint", required=True)
    ap.add_argument("--sample-list", required=True)
    ap.add_argument("--transcript", required=True, help="aishell_transcript_v0.8.txt")
    ap.add_argument("--whisper-model", default="medium")
    ap.add_argument("--max-samples", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    base_cfg = load_json(args.base_config); lca_cfg = load_json(args.lca_config)
    sr = int(base_cfg.get("sample_rate", 16000)); n_q = int(base_cfg.get("n_q", 3))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tr = load_aishell_transcripts(args.transcript)
    print(f"loaded {len(tr)} transcripts", flush=True)
    base_model = build_model(base_cfg).to(device).eval(); load_state(base_model, args.base_checkpoint)
    lca_model = build_model(lca_cfg).to(device).eval(); load_state(lca_model, args.lca_checkpoint)
    print(f"loading whisper {args.whisper_model}", flush=True)
    asr = whisper.load_model(args.whisper_model, device=(args.device if torch.cuda.is_available() else "cpu"))

    samples = read_sample_rows(args.sample_list, args.max_samples)
    print(f"evaluating {len(samples)} samples x 3 L x 2 models (zh CER)", flush=True)
    rows = []
    tic = time.time()
    skipped = 0
    with tempfile.TemporaryDirectory(prefix="scit_zh_") as tmp:
        tmp = Path(tmp)
        for idx, s in enumerate(samples, 1):
            sid = s["sample_id"]
            gt = tr.get(sid, "")
            if not gt:
                skipped += 1
                continue
            audio = load_audio_for_inference(s["audio"], sr).squeeze(0).numpy().astype(np.float32)
            op = tmp / f"{sid}_o.wav"; sf.write(op, audio, sr)
            oh = asr.transcribe(str(op), language="zh", verbose=False, condition_on_previous_text=False)["text"].strip()
            for mn, model in [("base", base_model), ("lca", lca_model)]:
                for L in (1, 2, 3):
                    rec = reconstruct_clean(model, audio, L, n_q, device)
                    wp = tmp / f"{sid}_{mn}_L{L}.wav"; sf.write(wp, rec, sr)
                    hyp = asr.transcribe(str(wp), language="zh", verbose=False, condition_on_previous_text=False)["text"].strip()
                    rows.append({"model": mn, "sample_id": sid, "L": L,
                                 "cer_vs_gt": cer_zh(gt, hyp),
                                 "cer_vs_original_whisper": cer_zh(oh, hyp),
                                 "gt_len": len(gt.replace(" ", "")), "hyp_len": len(hyp.replace(" ", ""))})
            if idx % 50 == 0 or idx == len(samples):
                print(f"done {idx}/{len(samples)} elapsed={time.time()-tic:.1f}s skipped={skipped}", flush=True)

    import csv
    fields = ["model", "sample_id", "L", "cer_vs_gt", "cer_vs_original_whisper", "gt_len", "hyp_len"]
    with open(run_dir / "metrics" / "asr_cer_zh_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(json.dumps({"status": "completed", "rows": len(rows), "skipped": skipped,
                      "whisper": args.whisper_model}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

