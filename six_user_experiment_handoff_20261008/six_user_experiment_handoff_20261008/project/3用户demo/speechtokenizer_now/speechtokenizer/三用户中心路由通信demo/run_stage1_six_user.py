#!/usr/bin/env python3
"""Run one-sender-at-a-time A-F VCTK replay through six real client processes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import secrets
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from stage1_protocol import CHUNKS_PER_TRIAL, USERS


DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(parent for parent in DEMO_DIR.parents if (parent / "scripts" / "speaker_identity_utils.py").is_file())
DEFAULT_ASSET_ROOT = (
    REPO_ROOT
    if (REPO_ROOT / "data/VCTK/wav48_silence_trimmed").is_dir()
    else REPO_ROOT.parent / "six_user_demo_handoff_20260920" / "project"
)
MIN_FREE_RAM_BYTES = 8 * 1024 ** 3
LCA_SHA256 = "af60223b2f44733f5a43d5a022871dad061dfe2421473c96f5d350c3d4b49a5a"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def available_ram_bytes():
    import psutil

    return int(psutil.virtual_memory().available)


def asset_paths(asset_root):
    return {
        "split": asset_root / "tmp/exp23_formal/cache_lca_l3/cache_manifest/split_manifest.json",
        "config": asset_root / "3用户demo/Log/spt_base/config.json",
        "checkpoint": asset_root / "3用户demo/Log/spt_base/SpeechTokenizer_best_dev.pt",
        "bundle": asset_root / "tmp/exp23_formal/export/seed42_bundle",
        "vctk": asset_root / "data/VCTK/wav48_silence_trimmed",
    }


def select_manifest_rows(split_manifest, split):
    payload = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    rows = payload["splits"][split]
    selected = {}
    for user, speaker in USERS.items():
        matches = [row for row in rows if row["speaker_id"] == speaker]
        if len(matches) != 15:
            raise ValueError(f"locked {split} split must have 15 rows for {speaker}, found {len(matches)}")
        selected[user] = matches[:1] if split == "validation" else matches
    return selected


def preflight(asset_root):
    paths = asset_paths(asset_root)
    missing = [str(path) for key, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Stage 1 assets: " + "; ".join(missing))
    required = ("numpy", "torch", "torchaudio", "soundfile", "sounddevice", "scipy", "psutil", "beartype")
    absent = [name for name in required if importlib.util.find_spec(name) is None]
    if absent:
        raise RuntimeError("Current Python lacks Stage 1 packages: " + ", ".join(absent))
    if file_sha256(paths["checkpoint"]) != LCA_SHA256:
        raise ValueError("LCA checkpoint SHA-256 does not match the Exp23 frozen model")
    manifest = json.loads((paths["bundle"] / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if file_sha256(paths["bundle"] / name) != expected:
            raise ValueError(f"Classifier bundle hash mismatch: {name}")
    labels = json.loads((paths["bundle"] / "speaker_labels.json").read_text(encoding="utf-8"))
    if not all(speaker in labels for speaker in USERS.values()):
        raise ValueError("Classifier bundle does not contain all A-F VCTK labels")
    selected = {split: select_manifest_rows(paths["split"], split) for split in ("validation", "test")}
    for by_user in selected.values():
        for user, rows in by_user.items():
            for row in rows:
                source = paths["vctk"] / USERS[user] / Path(row["audio_path"]).name
                if not source.is_file():
                    raise FileNotFoundError(source)
    free_ram = available_ram_bytes()
    report = {
        "python": sys.executable,
        "asset_root": str(asset_root),
        "free_ram_gb": round(free_ram / 1024 ** 3, 2),
        "required_free_ram_gb": 8,
        "resource_ready": free_ram >= MIN_FREE_RAM_BYTES,
        "lca_sha256": LCA_SHA256,
        "classifier_sha256": manifest["files"]["classifier.pt"],
        "mapping": USERS,
        "validation_utterances_per_user": 1,
        "test_utterances_per_user": 15,
    }
    return paths, selected, report


def prepare_wav(source, destination):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(str(source), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != 16000:
        factor = math.gcd(int(sr), 16000)
        mono = resample_poly(mono, 16000 // factor, int(sr) // factor)
    if len(mono) >= 48000:
        start = (len(mono) - 48000) // 2
        mono = mono[start:start + 48000]
    else:
        mono = np.pad(mono, (0, 48000 - len(mono)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), np.asarray(mono, dtype=np.float32), 16000, subtype="FLOAT")
    return destination


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def launch(args, log_path, cwd):
    handle = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen([sys.executable, "-u", *args], cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT)
    except Exception:
        handle.close()
        raise
    return proc, handle


def wait_for_log(proc, path, phrase, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and phrase in path.read_text(encoding="utf-8", errors="replace"):
            return
        if proc.poll() is not None:
            raise RuntimeError(f"Process exited while waiting for {phrase}: {path}")
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for {phrase}: {path}")


def stop_process(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run_round(paths, rows, split, sender, output_dir, key_path):
    round_dir = output_dir / split / sender
    round_dir.mkdir(parents=True, exist_ok=False)
    trials = []
    for index, row in enumerate(rows):
        original = paths["vctk"] / USERS[sender] / Path(row["audio_path"]).name
        wav = prepare_wav(original, round_dir / "wav" / f"{index:02d}_{row['utterance_group']}.wav")
        trials.append({"utterance_group": row["utterance_group"], "wav_path": str(wav), "source_audio_path": str(original)})
    config = {"users": USERS, "sender": sender, "split": split, "chunks_per_trial": CHUNKS_PER_TRIAL, "trials": trials}
    config_path = round_dir / "round_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    port = free_port()
    room = f"stage1_{split}_{sender}_{port}"
    # Receivers start one at a time before the sender. Their watchdog must
    # allow for all six model loads as well as the playlist duration; healthy
    # clients exit as soon as their expected packets have been decoded.
    seconds = len(trials) * 3 + 90
    router_log = round_dir / "router.log"
    processes = []
    handles = []
    try:
        router, handle = launch([str(DEMO_DIR / "router_server.py"), "--listen_ip", "127.0.0.1", "--listen_port", str(port), "--room_id", room, "--max_queue", "8"], router_log, REPO_ROOT)
        processes.append(router); handles.append(handle)
        wait_for_log(router, router_log, "[ROUTER] listening", 20)
        clients = {}
        for user in USERS:
            if user == sender:
                continue
            log = round_dir / f"client_{user}.log"
            argv = [str(DEMO_DIR / "group_client.py"), "--user_id", user, "--room_id", room, "--router_ip", "127.0.0.1", "--router_port", str(port), "--config_path", str(paths["config"]), "--ckpt_path", str(paths["checkpoint"]), "--key_path", str(key_path), "--device", "cpu", "--rvq_layers", "3", "--chunk_seconds", "0.5", "--speaker_classifier_candidates", "all", "--no_mic", "--no_play", "--channel", "clean", "--torch_threads", "1", "--stage1_config", str(config_path), "--stage1_results_csv", str(round_dir / f"results_{user}.csv"), "--summary_csv", str(round_dir / f"summary_{user}.csv"), "--speaker_classifier_bundle", str(paths["bundle"]), "--speaker_classifier_device", "cpu", "--speaker_window_sec", "3", "--speaker_hop_sec", "1", "--run_seconds", str(seconds)]
            proc, handle = launch(argv, log, REPO_ROOT)
            processes.append(proc); handles.append(handle); clients[user] = proc
            wait_for_log(router, router_log, f"registered user={user} ", 45)
        sender_log = round_dir / f"client_{sender}.log"
        argv = [str(DEMO_DIR / "group_client.py"), "--user_id", sender, "--room_id", room, "--router_ip", "127.0.0.1", "--router_port", str(port), "--config_path", str(paths["config"]), "--ckpt_path", str(paths["checkpoint"]), "--key_path", str(key_path), "--device", "cpu", "--rvq_layers", "3", "--chunk_seconds", "0.5", "--no_mic", "--no_play", "--channel", "clean", "--torch_threads", "1", "--stage1_config", str(config_path), "--summary_csv", str(round_dir / f"summary_{sender}.csv"), "--run_seconds", str(seconds)]
        source, handle = launch(argv, sender_log, REPO_ROOT)
        processes.append(source); handles.append(handle); clients[sender] = source
        wait_for_log(router, router_log, f"registered user={sender} ", 45)
        deadline = time.monotonic() + seconds + 30
        while any(proc.poll() is None for proc in clients.values()) and time.monotonic() < deadline:
            time.sleep(0.5)
        exit_codes = {user: proc.poll() for user, proc in clients.items()}
        if any(code is None for code in exit_codes.values()):
            raise TimeoutError(f"Stage 1 client timeout: {exit_codes}")
    finally:
        for proc in reversed(processes):
            stop_process(proc)
        for handle in handles:
            handle.close()
    results = []
    for user in USERS:
        if user == sender:
            continue
        path = round_dir / f"results_{user}.csv"
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as handle:
                results.extend(csv.DictReader(handle))
    metrics = {}
    for user in USERS:
        if user == sender:
            continue
        path = round_dir / f"summary_{user}.csv"
        if path.is_file():
            with path.open(encoding="utf-8", newline="") as handle:
                data = list(csv.DictReader(handle))
            if data:
                metrics[(sender, user)] = data[-1]
    receipt = {"split": split, "sender": sender, "port": port, "room": room, "exit_codes": exit_codes, "expected_packets_per_receiver": len(trials) * CHUNKS_PER_TRIAL, "result_rows": len(results), "receiver_summaries": len(metrics)}
    (round_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"{split} sender={sender} rows={len(results)}/{len(trials) * 5} exits={exit_codes}", flush=True)
    if any(code != 0 for code in exit_codes.values()) or len(results) != len(trials) * 5 or len(metrics) != 5:
        raise RuntimeError(f"Stage 1 round failed; inspect {round_dir}")
    if split == "validation" and any(int(row["decoded_packets"]) != 6 or int(row["evaluated"]) != 1 for row in results):
        raise RuntimeError(f"Stage 1 smoke did not decode and classify all five streams; inspect {round_dir}")
    return results, metrics


def aggregate(rows, client_metrics=None):
    client_metrics = client_metrics or {}
    groups = {}
    for row in rows:
        key = (row["sender_id"], row["receiver_id"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (sender, receiver), items in sorted(groups.items()):
        count = len(items)
        evaluated = sum(int(row["evaluated"]) for row in items)
        correct = sum(int(row["correct"]) for row in items)
        e2e = sum(int(row["end_to_end_correct"]) for row in items)
        missing = sum(int(row["missing_decoded_packets"]) for row in items)
        verified = sum(int(row["verified"] or 0) for row in items)
        metrics = client_metrics.get((sender, receiver), {})
        summary.append({"sender_id": sender, "receiver_id": receiver, "utterances": count, "evaluated": evaluated, "correct": correct, "end_to_end_correct": e2e, "recognition_coverage": evaluated / count, "accuracy_when_evaluated": correct / evaluated if evaluated else None, "end_to_end_accuracy": e2e / count, "verified_rate_when_evaluated": verified / evaluated if evaluated else None, "received_packets": sum(int(row["received_packets"]) for row in items), "decoded_packets": sum(int(row["decoded_packets"]) for row in items), "missing_decoded_packets": missing, "decode_ms_p95": metrics.get("decode_ms_p95"), "decode_rtf_p95": metrics.get("decode_rtf_p95"), "jitter_s_p95": metrics.get("jitter_s_p95"), "body_kbps": metrics.get("body_kbps"), "total_kbps": metrics.get("total_kbps")})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Single-machine six-client VCTK/Exp23 Stage 1 replay")
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--mode", choices=("smoke", "formal", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tmp" / ("stage1_six_user_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    asset_root = args.asset_root.resolve()
    paths, selected, report = preflight(asset_root)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not report["resource_ready"]:
        print(
            f"BLOCKED: Six independent clients need 8 GiB available RAM; currently {report['free_ram_gb']} GiB. "
            "Close other memory-heavy programs and rerun.",
            file=sys.stderr,
        )
        return 2
    if args.preflight_only:
        return 0
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Stage 1 output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "preflight.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "run_info.json").write_text(json.dumps({"mode": args.mode, "started_at": datetime.now().isoformat(), "code_root": str(REPO_ROOT), "asset_root": str(asset_root), "python": sys.executable, "stage1_source": "A-F single sender, clean, 16 kHz, 3 s WAV, L3"}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    key_path = output_dir / "demo.key"
    key_path.write_text("".join(f"{byte:08b}" for byte in secrets.token_bytes(512)), encoding="ascii")
    modes = ("validation", "test") if args.mode == "both" else (("validation",) if args.mode == "smoke" else ("test",))
    failures = []
    try:
        for split in modes:
            all_rows = []
            all_metrics = {}
            for sender in USERS:
                if available_ram_bytes() < MIN_FREE_RAM_BYTES:
                    raise MemoryError("Available RAM fell below 8 GiB before the next round")
                try:
                    rows, metrics = run_round(paths, selected[split][sender], split, sender, output_dir, key_path)
                    all_rows.extend(rows)
                    all_metrics.update(metrics)
                except Exception as error:
                    failures.append({"split": split, "sender": sender, "error": f"{type(error).__name__}: {error}"})
                    raise
            write_csv(output_dir / f"{split}_utterance_results.csv", all_rows)
            write_csv(output_dir / f"{split}_link_summary.csv", aggregate(all_rows, all_metrics))
            expected_rows = 30 if split == "validation" else 450
            if len(all_rows) != expected_rows or len(aggregate(all_rows)) != 30:
                raise RuntimeError(f"Stage 1 {split} result shape is incomplete")
    finally:
        (output_dir / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PASS: Stage 1 results at {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
