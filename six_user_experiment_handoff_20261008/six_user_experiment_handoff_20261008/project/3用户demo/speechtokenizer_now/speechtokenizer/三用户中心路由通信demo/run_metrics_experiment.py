"""Prepare and execute the A/B/C/D system measurement protocol on one host."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import math
import os
import platform
import secrets
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from metrics_protocol import (SR, USERS, expected_links, read_json, sha256, trial_jobs, write_csv, write_json)
from run_stage1_six_user import (DEMO_DIR, REPO_ROOT, DEFAULT_ASSET_ROOT, free_port,
                                preflight, prepare_wav, select_manifest_rows, stop_process)


RAM_REQUIREMENTS_GIB = {
    "strict": {"A": 2.0, "B": 4.0, "C": 8.0, "D": 8.0},
    "balanced": {"A": 1.5, "B": 3.0, "C": 6.0, "D": 6.0},
}
GPU_RAM_REQUIREMENTS_GIB = {"A": 1.0, "B": 1.5, "C": 3.0, "D": 3.0}


def parse_classifier_candidates(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")
    labels = list(dict.fromkeys(str(label).strip() for label in value if str(label).strip()))
    unknown = sorted(set(labels) - set(USERS.values()))
    if unknown:
        raise ValueError("classifier candidates must be among the six demo speakers: " + ", ".join(unknown))
    return labels


def versions():
    result = {"python": platform.python_version()}
    for name in ("torch", "torchaudio", "numpy", "soundfile", "sounddevice", "scipy", "psutil", "pystoi", "pesq", "matplotlib"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def code_identity():
    commands = {"head": ["git", "rev-parse", "HEAD"], "branch": ["git", "branch", "--show-current"],
                "dirty_files": ["git", "status", "--porcelain"]}
    identity = {}
    for key, command in commands.items():
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        identity[key] = result.stdout.strip() if result.returncode == 0 else None
    identity["source_hashes"] = {p.name: sha256(p) for p in DEMO_DIR.glob("*.py")}
    for relative in ("speechtokenizer/speaker_identity/inference.py", "speechtokenizer/trainer/loss.py"):
        identity["source_hashes"][relative] = sha256(REPO_ROOT / relative)
    identity["source_type"] = "git_checkout" if identity["head"] else "standalone_file_hashes"
    if not identity["head"]:
        identity["git_metadata_status"] = "runtime_folder_has_no_git_repository"
    return identity


def verify(args):
    import psutil
    import torch
    paths, _, report = preflight(args.asset_root.resolve())
    preprocessing = read_json(paths["bundle"] / "preprocessing.json")
    model_config = read_json(paths["bundle"] / "model_config.json")
    if read_json(paths["config"])["sample_rate"] != SR or any(preprocessing[k] != v for k, v in
            (("sample_rate", SR), ("rvq_layers", 3), ("window_samples", 48000))):
        raise ValueError("Frozen codec and classifier preprocessing must match 16 kHz, L3, three seconds")
    if model_config["input_mode"] != "codes_waveform" or model_config["speaker_count"] != 110:
        raise ValueError("System measurement requires the fixed 110-class codes_waveform bundle")
    dependencies = ("pystoi", "matplotlib")
    missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
    quality_python = args.quality_python
    if not quality_python:
        quality_python = sys.executable
    try:
        check = subprocess.run([quality_python, "-c", "import pesq,soundfile,json,platform,importlib.metadata; print(json.dumps({'python': platform.python_version(), 'pesq': importlib.metadata.version('pesq'), 'soundfile': soundfile.__version__}))"],
                               capture_output=True, text=True, timeout=15)
        if check.returncode:
            missing.append("pesq_worker")
        else:
            import json
            report["quality_runtime"] = json.loads(check.stdout)
    except (OSError, subprocess.TimeoutExpired):
        missing.append("pesq_worker")
    args.quality_python = quality_python
    report["quality_python"] = quality_python
    report["missing_measurement_dependencies"] = missing
    ram_table = GPU_RAM_REQUIREMENTS_GIB if args.compute_backend == "gpu_worker" else RAM_REQUIREMENTS_GIB
    needed = max((ram_table[c] if args.compute_backend == "gpu_worker" else RAM_REQUIREMENTS_GIB[args.ram_policy][c]
                  for c in args.conditions), default=1)
    report["required_free_ram_gb"] = needed
    report["ram_policy"] = args.ram_policy
    report["ram_policy_warning"] = (
        "balanced lowers the six-client preflight guard to 6 GiB; monitor swap, queue growth and client exits"
        if args.ram_policy == "balanced" else None
    )
    report["resource_ready"] = psutil.virtual_memory().available >= needed * 1024**3
    report["deployment"] = "single_host_cpu_loopback"
    report["compute_backend"] = args.compute_backend
    report["gpu_ready"] = True
    if args.compute_backend == "gpu_worker":
        if not torch.cuda.is_available():
            report["gpu_ready"] = False
            report["gpu_reason"] = "PyTorch CUDA is unavailable"
        else:
            free_gpu, total_gpu = torch.cuda.mem_get_info()
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["gpu_free_gib"] = round(free_gpu / 1024**3, 2)
            report["gpu_total_gib"] = round(total_gpu / 1024**3, 2)
            report["gpu_min_free_gib"] = 1.5
            report["gpu_ready"] = free_gpu >= 1.5 * 1024**3
            if not report["gpu_ready"]:
                report["gpu_reason"] = "GPU worker requires at least 1.5 GiB free VRAM"
    target = args.output_dir.resolve().parent
    while not target.exists():
        target = target.parent
    free_disk = shutil.disk_usage(target).free
    if hasattr(args, "estimated_audio_bytes"):
        audio_bytes = args.estimated_audio_bytes
    else:
        audio_bytes = 0
        scenarios = args.scenarios.split(",") if isinstance(args.scenarios, str) else args.scenarios
        for condition in args.conditions:
            for scenario in scenarios:
                seconds = 36 if scenario == "switch" else math.ceil(args.duration_seconds / 3) * 3 if scenario == "stability" else args.utterances * 3
                audio_bytes += sum(len(expected_links(j)) * seconds * SR * 4 for j in trial_jobs(condition, scenario)) * args.repeats
    estimated_disk = round(audio_bytes * 1.3 + 64 * 1024**2)
    report["disk_free_gib"] = round(free_disk / 1024**3, 2)
    report["estimated_output_gib"] = round(estimated_disk / 1024**3, 2)
    report["disk_ready"] = free_disk >= estimated_disk
    report["measurement_ready"] = report["resource_ready"] and report["disk_ready"] and report["gpu_ready"] and not missing
    report["dependencies"] = versions()
    return paths, report


def prepare(args, paths, report):
    import soundfile as sf
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    classifier_candidates = parse_classifier_candidates(args.classifier_candidates)
    selected = select_manifest_rows(paths["split"], "validation" if args.mode == "smoke" else "test")
    source_rows = []
    prepared = {}
    for user, rows in selected.items():
        prepared[user] = []
        for row in rows[:args.utterances]:
            original = paths["vctk"] / USERS[user] / Path(row["audio_path"]).name
            target = prepare_wav(original, output / "audio" / "source" / (row["utterance_group"] + ".wav"))
            info = sf.info(str(original))
            part = {"speaker_id": USERS[user], "utterance_group": row["utterance_group"],
                    "wav_path": str(target), "samples": 48000, "source_audio_path": str(original)}
            prepared[user].append(part)
            source_rows.append({"source_user": user, "speaker_id": USERS[user], "utterance_group": row["utterance_group"],
                                "split": "validation" if args.mode == "smoke" else "test",
                                "original_path": str(original), "original_sha256": sha256(original),
                                "original_sample_rate": info.samplerate, "original_samples": info.frames,
                                "original_duration_seconds": info.duration, "prepared_path": str(target),
                                "prepared_sha256": sha256(target), "sample_rate": SR, "samples": 48000,
                                "preparation": "mono_mean; scipy_resample_poly; center_crop_or_end_zero_pad_48000"})
    write_csv(output / "audio_manifest.csv", source_rows)
    key = output / "demo.key"
    key.write_text("".join(f"{b:08b}" for b in secrets.token_bytes(512)), encoding="ascii")
    jobs = []
    for repeat in range(args.repeats):
        for condition in args.conditions:
            for scenario in args.scenarios:
                for spec in trial_jobs(condition, scenario):
                    streams = {}
                    for sender in spec["active_senders"]:
                        segments = []
                        if scenario == "switch":
                            sender_index = list(USERS).index(sender)
                            for step in range(6):
                                source_user = list(USERS)[(sender_index + step) % 6]
                                # Six-second dwell replays the fixed three-second segment twice.
                                segments.extend(dict(prepared[source_user][0]) for _ in range(2))
                        elif scenario == "stability":
                            count = math.ceil(args.duration_seconds / 3)
                            segments = [dict(prepared[sender][i % len(prepared[sender])]) for i in range(count)]
                        else:
                            segments = [dict(p) for p in prepared[sender]]
                        streams[sender] = segments
                    job_id = f"{condition}_{scenario}_r{repeat + 1:02d}_j{len(jobs):03d}"
                    required_table = GPU_RAM_REQUIREMENTS_GIB if args.compute_backend == "gpu_worker" else RAM_REQUIREMENTS_GIB
                    required_ram = required_table[condition] if args.compute_backend == "gpu_worker" else required_table[args.ram_policy][condition]
                    job = {**spec, "job_id": job_id, "repeat": repeat + 1, "streams": streams,
                           "continuous": scenario in ("switch", "stability"), "playback": args.playback,
                           "playback_max_seconds": .8, "queue_blocks": args.queue_blocks,
                           "block_deadline_seconds": args.block_deadline_seconds,
                           "prediction_deadline_seconds": args.prediction_deadline_seconds,
                           "drain_seconds": args.drain_seconds, "startup_timeout_seconds": 240,
                           "resource_interval_seconds": 1.0, "torch_threads": args.torch_threads,
                           "model_config": str(paths["config"]), "checkpoint": str(paths["checkpoint"]),
                           "bundle": str(paths["bundle"]), "key_path": str(key),
                           "source_sample_rate": SR, "chunk_samples": 8000,
                           "window_samples": 48000, "hop_samples": 16000,
                           "compute_backend": args.compute_backend,
                           "classifier_candidates": classifier_candidates,
                           "ram_policy": args.ram_policy,
                           "ram_required_gib": required_ram}
                    jobs.append(job)
    config = {"schema": "six_user_system_metrics_v1", "mode": args.mode, "jobs": jobs,
              "model_hashes": {"checkpoint": sha256(paths["checkpoint"]), "model_config": sha256(paths["config"]),
                               "bundle_manifest": sha256(paths["bundle"] / "manifest.json")},
              "audio_manifest_sha256": sha256(output / "audio_manifest.csv"), "source_rows": source_rows,
              "policies": {"short_audio": "end_zero_padding", "silence": "included",
                           "cross_speaker_windows": "recorded; excluded_from_steady_accuracy; target_is_latest_sample_identity",
                           "initial_context": "3_seconds_included_in_first_response_time",
                           "first_correct": "first_single_on_time_correct_window_output; continuous_response_success_before_identity_dwell_end",
                           "missing_audio": "zero_fill_at_original_block_positions; no_time_compression",
                           "quality_alignment": "each_block_trim_or_end_pad_to_8000; no_cross_correlation_shift",
                           "sample_unit": "source_utterance; receiver_copies_and_replays_are_dependent",
                           "clock": "time.perf_counter_ns shared by processes on this host; nanoseconds",
                           "latency_reference": "source_block_first_sample_scheduled_WAV_replay",
                           "latency_boundary": "decoder_completion; optional playback_callback; acoustic_output_not_measured",
                           "network_measurement": "application_layer_only; no_transport_or_link_capture",
                            "classifier": "unchanged_110_class_seed42_temperature_calibrated",
                            "classifier_candidates": classifier_candidates or "all_110_bundle_labels",
                           "mel_config": read_json(paths["config"]), "inference_device": "cpu"}}
    write_json(output / "experiment_plan.json", config)
    config["policies"]["ram_policy"] = args.ram_policy
    config["policies"]["ram_policy_warning"] = report.get("ram_policy_warning")
    write_json(output / "experiment_plan.json", config)
    manifest = {"schema": config["schema"], "created_at": datetime.now().isoformat(), "mode": args.mode,
                "status": "prepared", "code": code_identity(), "python": sys.executable,
                "dependencies": versions(), "host": platform.node(), "os": platform.platform(),
                "deployment": "single_host_cpu_loopback", "model_hashes": config["model_hashes"],
                "plan_sha256": sha256(output / "experiment_plan.json"), "preflight": report,
                "policies": config["policies"], "conditions": args.conditions, "scenarios": args.scenarios,
                "repeats": args.repeats, "fixture": False}
    manifest["ram_policy"] = args.ram_policy
    manifest["ram_required_gib"] = report["required_free_ram_gb"]
    manifest["quality_python"] = args.quality_python
    manifest["quality_runtime"] = report.get("quality_runtime")
    import psutil
    clock = time.get_clock_info("perf_counter")
    manifest["hardware"] = {"processor": platform.processor(), "logical_cpu_count": psutil.cpu_count(),
                            "physical_cpu_count": psutil.cpu_count(logical=False), "total_ram_bytes": psutil.virtual_memory().total}
    manifest["clock_info"] = {"implementation": clock.implementation, "resolution_seconds": clock.resolution,
                              "monotonic": clock.monotonic, "adjustable": clock.adjustable}
    manifest["wall_timezone"] = str(datetime.now().astimezone().tzinfo)
    write_json(output / "run_manifest.json", manifest)
    (output / "logs").mkdir()
    write_json(output / "failures.json", [])
    return output, config


def start_process(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w", encoding="utf-8")
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", MKL_THREADING_LAYER="sequential")
    try:
        process = subprocess.Popen([sys.executable, "-u", *command], cwd=REPO_ROOT,
                                   stdout=handle, stderr=subprocess.STDOUT, env=environment)
    except Exception:
        handle.close()
        raise
    return process, handle


def run_job(output, job, fixture=False):
    import psutil
    directory = output / "jobs" / job["job_id"]
    directory.mkdir(parents=True, exist_ok=False)
    events = directory / "events"
    events.mkdir()
    job = dict(job, port=free_port(), worker_port=free_port(), worker_host="127.0.0.1",
               gate_path=str(directory / "gate.json"))
    config_path = directory / "job.json"
    write_json(config_path, job)
    processes, handles, clients = [], [], {}
    router = None
    worker = None
    receipt = {"job_id": job["job_id"], "condition": job["condition"], "scenario": job["scenario"],
               "repeat": job["repeat"], "connected_users_planned": len(job["users"]),
               "active_senders_planned": len(job["active_senders"]), "status": "starting", "exit_codes": {},
               "process_start_ns": time.perf_counter_ns(), "started_at": datetime.now().isoformat()}
    error = None
    print(f"START {job['job_id']}: connected={len(job['users'])}, active_sources={len(job['active_senders'])}, logs={directory / 'logs'}", flush=True)
    try:
        if job.get("compute_backend") == "gpu_worker":
            worker_command = [
                str(DEMO_DIR / "gpu_worker.py"), "--port", str(job["worker_port"]),
                "--output", str(events), "--config", job["model_config"],
                "--checkpoint", job["checkpoint"], "--bundle", job["bundle"],
                "--stop-file", str(directory / "worker.stop"),
            ]
            if job.get("classifier_candidates"):
                worker_command.extend(["--candidate-labels", ",".join(job["classifier_candidates"])])
            worker, log = start_process(
                worker_command,
                directory / "logs" / "gpu_worker.log")
            processes.append(worker)
            handles.append(log)
            deadline = time.monotonic() + 120
            while not (events / "worker_ready.json").is_file():
                if worker.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("GPU worker failed to become ready; inspect gpu_worker.log")
                time.sleep(.2)
        if job["transport"] != "local":
            router, log = start_process([str(DEMO_DIR / "router_server.py"), "--listen_ip", "127.0.0.1",
                                        "--listen_port", str(job["port"]), "--room_id", job["job_id"],
                                        "--max_queue", str(job["queue_blocks"]), "--metrics_dir", str(events),
                                        "--stop_file", str(directory / "router.stop")], directory / "logs" / "router.log")
            processes.append(router); handles.append(log)
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                text = (directory / "logs" / "router.log").read_text(encoding="utf-8", errors="replace")
                if "[ROUTER] listening" in text:
                    break
                if router.poll() is not None:
                    raise RuntimeError("Router failed before registration")
                time.sleep(.1)
            else:
                raise TimeoutError("Router did not listen")
        for user in job["users"]:
            print(f"  loading/registering client {user}", flush=True)
            command = [str(DEMO_DIR / "metrics_client.py"), "--job", str(config_path), "--user", user, "--output", str(events)]
            if fixture:
                command.append("--fixture")
            process, log = start_process(command, directory / "logs" / (user + ".log"))
            processes.append(process); handles.append(log); clients[user] = process
            deadline = time.monotonic() + 90
            while not (events / (user + ".ready.json")).is_file():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"Client {user} failed to load/register; inspect {directory / 'logs'}")
                time.sleep(.2)
            print(f"  client {user} ready", flush=True)
        if router:
            # Registration is acknowledged by the router log before opening the barrier.
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                text = (directory / "logs" / "router.log").read_text(encoding="utf-8", errors="replace")
                if all(f"registered user={u} " in text for u in job["users"]):
                    break
                time.sleep(.1)
            else:
                raise TimeoutError("Router registration barrier incomplete")
        origin = time.perf_counter_ns() + 1_000_000_000
        write_json(job["gate_path"], {"origin_ns": origin, "clock": "host_perf_counter_ns"})
        duration = max(sum(p["samples"] for p in parts) / SR for parts in job["streams"].values())
        receipt.update(status="running", origin_ns=origin, source_duration_seconds=duration,
                       source_audio_seconds=sum(sum(p["samples"] for p in parts) / SR for parts in job["streams"].values()))
        write_json(directory / "receipt.json", receipt)
        print(f"  replay started: source_duration={duration:.1f}s, drain_limit={job['drain_seconds']:.1f}s", flush=True)
        deadline = time.monotonic() + duration + job["drain_seconds"] + 35
        while any(p.poll() is None for p in clients.values()):
            if router and router.poll() is not None:
                raise RuntimeError("Router exited during replay")
            if time.monotonic() > deadline:
                raise TimeoutError("Parent watchdog expired")
            time.sleep(.2)
        receipt["exit_codes"] = {u: p.returncode for u, p in clients.items()}
        if any(code != 0 for code in receipt["exit_codes"].values()):
            raise RuntimeError("Client failure: " + str(receipt["exit_codes"]))
        receipt["status"] = "complete"
    except KeyboardInterrupt:
        receipt.update(status="interrupted", error="user interrupted the run")
        raise
    except Exception as exc:
        error = repr(exc)
        receipt.update(status="failed", error=error)
    finally:
        for user, process in clients.items():
            if process.poll() is None:
                stop_process(process)
            receipt["exit_codes"][user] = process.returncode
        if router:
            (directory / "router.stop").touch()
            try:
                router.wait(timeout=5)
            except subprocess.TimeoutExpired:
                stop_process(router)
            receipt["router_exit_code"] = router.returncode
        if worker:
            (directory / "worker.stop").touch()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                stop_process(worker)
            receipt["worker_exit_code"] = worker.returncode
        for handle in handles:
            handle.close()
        receipt["finished_at"] = datetime.now().isoformat()
        receipt["finished_ns"] = time.perf_counter_ns()
        receipt["elapsed_seconds"] = (receipt["finished_ns"] - receipt["process_start_ns"]) / 1e9
        write_json(directory / "receipt.json", receipt)
    print(f"{job['job_id']}: {receipt['status']} exits={receipt['exit_codes']}", flush=True)
    return error


def execute(output, config, fixture=False):
    import psutil
    from summarize_metrics_experiment import summarize
    manifest = read_json(output / "run_manifest.json")
    if (output / "jobs").exists() and any((output / "jobs").iterdir()):
        raise FileExistsError("This prepared run already has job records; create a new plan for another run")
    if manifest["plan_sha256"] != sha256(output / "experiment_plan.json"):
        raise ValueError("Frozen experiment plan was changed")
    if config["audio_manifest_sha256"] != sha256(output / "audio_manifest.csv"):
        raise ValueError("Frozen audio manifest was changed")
    for row in config["source_rows"]:
        if sha256(row["prepared_path"]) != row["prepared_sha256"] or sha256(row["original_path"]) != row["original_sha256"]:
            raise ValueError("Frozen source audio was changed: " + row["utterance_group"])
    first_job = config["jobs"][0]
    expected_model_hashes = config["model_hashes"]
    for name, path in (("checkpoint", first_job["checkpoint"]), ("model_config", first_job["model_config"]),
                       ("bundle_manifest", Path(first_job["bundle"]) / "manifest.json")):
        if sha256(path) != expected_model_hashes[name]:
            raise ValueError("Frozen model/config changed: " + name)
    manifest.update(status="running", started_at=datetime.now().isoformat(), fixture=fixture,
                    execution_code=code_identity(), execution_dependencies=versions())
    write_json(output / "run_manifest.json", manifest)
    failures = []
    try:
        for job in config["jobs"]:
            required = float(job.get("ram_required_gib", {1: 2, 3: 4, 6: 8}[len(job["users"])])) * 1024**3
            if not fixture and psutil.virtual_memory().available < required:
                failures.append({"job_id": job["job_id"], "reason": "insufficient free RAM before startup"})
                break
            error = run_job(output, job, fixture)
            if error:
                failures.append({"job_id": job["job_id"], "reason": error})
            write_json(output / "failures.json", failures)
    except KeyboardInterrupt:
        failures.append({"reason": "interrupted"})
    except Exception as error:
        failures.append({"reason": repr(error)})
    finally:
        write_json(output / "failures.json", failures)
        manifest.update(status="failed" if failures else "complete", finished_at=datetime.now().isoformat())
        write_json(output / "run_manifest.json", manifest)
        summarize(output)
    print(f"RESULTS: {output}", flush=True)
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--quality-python", help="Python with PESQ and soundfile; recorded separately from the codec environment")
    parser.add_argument("--ram-policy", choices=tuple(RAM_REQUIREMENTS_GIB), default="strict",
                        help="strict: 8 GiB for six clients; balanced: 6 GiB with explicit resource warning")
    parser.add_argument("--compute-backend", choices=("local_cpu", "gpu_worker"), default="local_cpu")
    parser.add_argument("--classifier-candidates", default="",
                        help="Optional comma-separated known-roster labels; empty keeps the original 110-class output")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tmp" / ("system_metrics_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--conditions", default="A,B,C,D")
    parser.add_argument("--scenarios", default="sequential,concurrent")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--utterances", type=int)
    parser.add_argument("--duration-seconds", type=float, default=1800)
    parser.add_argument("--block-deadline-seconds", type=float, default=10)
    parser.add_argument("--prediction-deadline-seconds", type=float, default=15)
    parser.add_argument("--drain-seconds", type=float, default=45)
    parser.add_argument("--queue-blocks", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--playback", choices=("off", "device"), default="off")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-plan", type=Path, help="Execute a prepared experiment_plan.json")
    args = parser.parse_args()
    if args.run_plan:
        config = read_json(args.run_plan)
        args.conditions = sorted({j["condition"] for j in config["jobs"]})
        args.asset_root = Path(config["jobs"][0]["checkpoint"]).parents[3]
        args.output_dir = args.run_plan.resolve().parent
        args.estimated_audio_bytes = sum(sum(sum(p["samples"] for p in j["streams"][s]) * 4 for s, _ in expected_links(j)) for j in config["jobs"])
        args.quality_python = read_json(args.run_plan.parent / "run_manifest.json")["quality_python"]
        args.ram_policy = read_json(args.run_plan.parent / "run_manifest.json").get("ram_policy", "strict")
        args.compute_backend = read_json(args.run_plan.parent / "run_manifest.json").get("compute_backend", "local_cpu")
        paths, report = verify(args)
        if not report["measurement_ready"]:
            print(report)
            return 2
        return execute(args.run_plan.resolve().parent, config)
    args.conditions = args.conditions.split(",")
    args.scenarios = args.scenarios.split(",")
    if not set(args.conditions) <= {"A", "B", "C", "D"} or not set(args.scenarios) <= {"sequential", "concurrent", "switch", "stability"}:
        parser.error("Use A/B/C/D and sequential/concurrent/switch/stability")
    args.repeats = args.repeats if args.repeats is not None else (1 if args.mode == "smoke" else 3)
    args.utterances = args.utterances if args.utterances is not None else (1 if args.mode == "smoke" else 15)
    if args.repeats < 1 or not 1 <= args.utterances <= (1 if args.mode == "smoke" else 15):
        parser.error("Invalid repeat or utterance count for the selected split")
    if min(args.duration_seconds, args.block_deadline_seconds, args.prediction_deadline_seconds,
           args.drain_seconds, args.queue_blocks, args.torch_threads) <= 0:
        parser.error("Durations, deadlines, queue size and thread count must be positive")
    paths, report = verify(args)
    import json
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        if not report["measurement_ready"]:
            print(f"BLOCKED: free RAM={report['free_ram_gb']} GiB, required={report['required_free_ram_gb']} GiB; "
                  f"disk_ready={report['disk_ready']}; missing packages={report['missing_measurement_dependencies']}", file=sys.stderr)
        return 0 if report["measurement_ready"] else 2
    if not args.prepare_only and not report["measurement_ready"]:
        print("BLOCKED: Resolve missing measurement packages, free RAM or output disk space before starting clients.", file=sys.stderr)
        return 2
    output, config = prepare(args, paths, report)
    if args.prepare_only:
        print(f"PREPARED: {output / 'experiment_plan.json'}")
        return 0
    return execute(output, config)


if __name__ == "__main__":
    raise SystemExit(main())
