"""Bounded local replay checks. Never opens a microphone or playback device."""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def verify_bundle(exp23, output):
    import numpy as np
    import soundfile as sf
    import torch

    sys.path.insert(0, str(exp23))
    from speechtokenizer.speaker_identity.cache import aligned_crop_bounds
    from speechtokenizer.speaker_identity.inference import SpeakerClassifierBundle

    torch.set_num_threads(4)
    formal = exp23 / "tmp" / "exp23_formal"
    records = json.loads(
        (formal / "cache_lca_l3" / "cache_manifest" / "cache_records.json").read_text(encoding="utf-8")
    )
    row = next(
        row for row in records
        if row["split"] == "validation" and row["valid_sample_count"] >= 48000
    )
    waveform, sample_rate = sf.read(row["reconstruction_path"], dtype="float32")
    with np.load(row["codes_path"], allow_pickle=False) as archive:
        codes = archive["codes"]
    length = min(len(waveform), row["valid_sample_count"])
    bounds = aligned_crop_bounds(length, codes.shape[-1], 0, 48000)
    clip = waveform[bounds.audio_start:bounds.audio_end]
    clip_codes = codes[:, bounds.token_start:bounds.token_end]
    assert len(clip) == 48000 and sample_rate == 16000
    bundle = SpeakerClassifierBundle(formal / "export" / "seed42_bundle", device="cpu")
    prediction = bundle.predict(clip, clip_codes)
    result = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_loaded": True,
        "speaker_count": bundle.speaker_count,
        "source_split": "validation",
        "expected_speaker": row["speaker_id"],
        "predicted_speaker": prediction.predicted_speaker,
        "score": prediction.score,
        "waveform_samples": len(clip),
        "codes_shape": list(clip_codes.shape),
        "scope": "One existing validation sample, not a rerun of the locked formal test.",
    }
    assert prediction.predicted_speaker == row["speaker_id"]
    write_json(output / "bundle_load_smoke.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


def run_replay(source, exp23, output, mode, duration, device):
    code_root = source if mode == "profile" else exp23
    demo = next(path for path in code_root.glob("3*demo") if path.is_dir())
    demo_code = next(demo.rglob("group_client.py")).parent
    source_demo = next(path for path in source.glob("3*demo") if path.is_dir())
    key = next(source_demo.rglob("userA/46.txt"))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    room = f"handoff_{mode}_{port}"
    environment = dict(os.environ)
    environment.update({
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    commands = []
    processes = []
    handles = []
    result = {"mode": mode, "started_at_utc": datetime.now(timezone.utc).isoformat()}

    def launch(name, command):
        log = output / f"{name}.log"
        handle = log.open("w", encoding="utf-8")
        handles.append(handle)
        process = subprocess.Popen(
            command,
            cwd=str(code_root),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=flags,
        )
        processes.append((name, process))
        commands.append({
            "name": name,
            "argv": command,
            "cwd": str(code_root),
            "windows_command_line": subprocess.list2cmdline(command),
            "log": str(log),
        })
        return process

    try:
        router = launch("router", [
            sys.executable, "-u", str(demo_code / "router_server.py"),
            "--listen_ip", "127.0.0.1", "--listen_port", str(port),
            "--room_id", room, "--max_queue", "8",
        ])
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if router.poll() is not None:
                raise RuntimeError("Router exited before listening")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            raise TimeoutError("Router did not listen within 15 seconds")
        clients = []
        for speaker in ("p225", "p226", "p227"):
            command = [
                sys.executable, "-u", str(demo_code / "group_client.py"),
                "--user_id", speaker, "--room_id", room,
                "--router_ip", "127.0.0.1", "--router_port", str(port),
                "--config_path", str(source_demo / "Log" / "spt_base" / "config.json"),
                "--ckpt_path", str(source_demo / "Log" / "spt_base" / "SpeechTokenizer_best_dev.pt"),
                "--key_path", str(key), "--device", device,
                "--rvq_layers", "3", "--chunk_seconds", "0.5",
                "--wav_input", str(source / "data" / "VCTK" / "wav48_silence_trimmed" / speaker / f"{speaker}_004_mic1.flac"),
                "--loop_wav", "--no_mic", "--no_play", "--monitor",
                "--run_seconds", str(duration),
                "--summary_csv", str(output / f"{speaker}_summary.csv"),
                "--speaker_hop_sec", "1",
            ]
            if mode == "profile":
                command += [
                    "--speaker_id_enable",
                    "--speaker_profile_dir", str(source / "output" / "experiments" / "exp22_three_user_speaker_demo_smoke_20260708_seed42" / "speaker_profiles"),
                    "--speaker_backend", "ecapa", "--speaker_device", "cpu",
                    "--ecapa_savedir", str(source / "output" / "models" / "speechbrain_spkrec_ecapa_voxceleb"),
                    "--speaker_window_sec", "2", "--speaker_threshold", "0.25",
                ]
            else:
                command += [
                    "--speaker_classifier_bundle", str(exp23 / "tmp" / "exp23_formal" / "export" / "seed42_bundle"),
                    "--speaker_classifier_device", "cpu", "--speaker_window_sec", "3",
                ]
            clients.append((speaker, launch(speaker, command)))
        write_json(output / "commands.json", {
            "environment_overrides": {key: environment[key] for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONIOENCODING", "PYTHONUNBUFFERED")},
            "commands": commands,
        })
        deadline = time.monotonic() + duration + 100
        while any(process.poll() is None for _, process in clients):
            if time.monotonic() > deadline:
                raise TimeoutError("Client replay exceeded bounded timeout")
            time.sleep(0.25)
        result["exit_codes"] = {speaker: process.returncode for speaker, process in clients}
        summaries = {}
        for speaker, _ in clients:
            path = output / f"{speaker}_summary.csv"
            if path.exists():
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    summaries[speaker] = list(csv.DictReader(handle))
        result["summaries"] = summaries
        result["log_errors"] = {}
        for speaker, _ in clients:
            text = (output / f"{speaker}.log").read_text(encoding="utf-8", errors="replace")
            result["log_errors"][speaker] = [
                line for line in text.splitlines()
                if any(marker in line for marker in ("Traceback", "ModuleNotFoundError", "RuntimeError", "[RX] error", "[SPKID] update failed", "CUFFT_INTERNAL_ERROR"))
            ]
        result["passed"] = (
            all(code == 0 for code in result["exit_codes"].values())
            and len(summaries) == 3
            and not any(result["log_errors"].values())
            and all(
                len(rows) == 1
                and int(rows[0]["sent_packets"]) > 0
                and int(rows[0]["recv_packets"]) > 0
                and int(rows[0]["speaker_eval_count"]) > 0
                for rows in summaries.values()
            )
        )
    except Exception as error:
        result["passed"] = False
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for handle in handles:
            handle.close()
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["scope"] = "Three local replay clients only; no microphone, playback, six-user or open-set acceptance."
        write_json(output / "replay_result.json", result)
    print(json.dumps(result, ensure_ascii=True, indent=2), flush=True)
    return 0 if result.get("passed") else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exp23-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("profile", "bundle", "bundle-load"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Output directory must be empty to preserve previous verification receipts.")
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "bundle-load":
        return verify_bundle(args.exp23_root.resolve(), output)
    return run_replay(
        args.source_root.resolve(), args.exp23_root.resolve(), output,
        args.mode, args.duration, args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
