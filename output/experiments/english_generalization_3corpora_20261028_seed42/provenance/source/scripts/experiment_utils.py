import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


STANDARD_SUBDIRS = (
    "configs",
    "commands",
    "logs",
    "checkpoints",
    "metrics",
    "samples",
    "reports",
    "artifacts",
    "cache_manifest",
)


FIXED_CONDITIONS = {
    "sample_rate": 16000,
    "strides": [8, 5, 4, 2],
    "dimension": 1024,
    "n_q": 3,
    "codebook_size": 1024,
}


def iso_now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_run_layout(run_dir):
    run_dir = Path(run_dir)
    for name in STANDARD_SUBDIRS:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    return run_dir


class _TeeStream:
    def __init__(self, stream, log_path):
        self.stream = stream
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "a", encoding="utf-8", buffering=1)

    def write(self, text):
        written = self.stream.write(text)
        self.log_file.write(text)
        return written

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def close_log(self):
        self.log_file.flush()
        self.log_file.close()

    def isatty(self):
        return self.stream.isatty()

    def writable(self):
        return True

    def __getattr__(self, name):
        return getattr(self.stream, name)


def install_tee_logging(run_dir):
    run_dir = ensure_run_layout(run_dir)

    if isinstance(sys.stdout, _TeeStream):
        sys.stdout.close_log()
        sys.stdout = sys.stdout.stream
    if isinstance(sys.stderr, _TeeStream):
        sys.stderr.close_log()
        sys.stderr = sys.stderr.stream

    sys.stdout = _TeeStream(sys.stdout, run_dir / "logs" / "stdout.log")
    sys.stderr = _TeeStream(sys.stderr, run_dir / "logs" / "stderr.log")
    return run_dir


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def append_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def file_sha256(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def run_command(args, cwd=None, timeout=60):
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            timeout=timeout,
            text=True,
            capture_output=True,
            check=False,
        )
        return {
            "args": list(args),
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {
            "args": list(args),
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
        }


def collect_environment_metadata(project_root=None):
    project_root = Path(project_root or Path.cwd()).resolve()
    git_revision_command = ["git", "rev-parse", "HEAD"]
    git_status_command = ["git", "status", "--porcelain"]
    git_metadata = {
        "revision": None,
        "dirty": None,
        "error": None,
    }

    git_revision_result = run_command(
        git_revision_command,
        cwd=project_root,
        timeout=20,
    )
    if git_revision_result["returncode"] == 0 and git_revision_result["stdout"]:
        git_metadata["revision"] = git_revision_result["stdout"]
        git_status_result = run_command(
            git_status_command,
            cwd=project_root,
            timeout=20,
        )
        if git_status_result["returncode"] == 0:
            git_metadata["dirty"] = bool(git_status_result["stdout"])
        else:
            returncode = git_status_result["returncode"]
            if returncode is None:
                returncode = "unavailable"
            git_metadata["error"] = (
                "git status --porcelain failed "
                f"(returncode={returncode})"
            )
    else:
        returncode = git_revision_result["returncode"]
        if returncode is None:
            returncode = "unavailable"
        git_metadata["error"] = (
            "git rev-parse HEAD failed "
            f"(returncode={returncode})"
        )

    packages = {}
    distributions = {
        "torch": "torch",
        "torchaudio": "torchaudio",
        "speechbrain": "speechbrain",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
    }
    for package_name, distribution_name in distributions.items():
        try:
            packages[package_name] = {
                "version": importlib.metadata.version(distribution_name),
                "error": None,
            }
        except Exception as exc:
            packages[package_name] = {
                "version": None,
                "error": exc.__class__.__name__,
            }

    return {
        "python": {
            "version": sys.version.replace("\n", " "),
        },
        "platform": platform.platform(),
        "git": git_metadata,
        "packages": packages,
    }


def validate_fixed_conditions(config):
    problems = []
    for key, expected in FIXED_CONDITIONS.items():
        actual = config.get(key)
        if actual != expected:
            problems.append({"key": key, "expected": expected, "actual": actual})
    downsample = 1
    for stride in config.get("strides", []):
        downsample *= int(stride)
    if downsample != 320:
        problems.append({"key": "encoder_downsample_rate", "expected": 320, "actual": downsample})
    latent_rate = None
    if config.get("sample_rate") and downsample:
        latent_rate = float(config["sample_rate"]) / float(downsample)
    if latent_rate != 50.0:
        problems.append({"key": "latent_rate", "expected": 50.0, "actual": latent_rate})
    return problems


def map_manifest_path(path_value, project_root):
    raw = Path(path_value)
    if raw.exists():
        return raw.resolve(), "as_listed"

    normalized = path_value.replace("\\", "/")
    marker = "/data/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        candidate = Path(project_root) / "data" / Path(suffix)
        if candidate.exists():
            return candidate.resolve(), "mapped_from_data_suffix"

    data_marker = "data/"
    if data_marker in normalized:
        suffix = normalized.split(data_marker, 1)[1]
        candidate = Path(project_root) / "data" / Path(suffix)
        if candidate.exists():
            return candidate.resolve(), "mapped_from_data_suffix"

    return raw, "missing"


def normalize_manifest(source_path, output_path, project_root, max_lines=None):
    source_path = Path(source_path)
    output_path = Path(output_path)
    rows = []
    stats = {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "total_read": 0,
        "written": 0,
        "missing_audio": 0,
        "missing_feature": 0,
        "mapped_paths": 0,
        "bad_rows": 0,
        "truncated_to": max_lines,
    }
    missing_examples = []

    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_lines is not None and stats["total_read"] >= max_lines:
                break
            stats["total_read"] += 1
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split("\t")
            if len(parts) < 2:
                stats["bad_rows"] += 1
                if len(missing_examples) < 10:
                    missing_examples.append({"line": raw, "reason": "expected audio<TAB>feature"})
                continue
            audio, feature = parts[0], parts[1]
            audio_path, audio_mode = map_manifest_path(audio, project_root)
            feature_path, feature_mode = map_manifest_path(feature, project_root)
            if audio_mode.startswith("mapped") or feature_mode.startswith("mapped"):
                stats["mapped_paths"] += 1
            if not audio_path.exists():
                stats["missing_audio"] += 1
                if len(missing_examples) < 10:
                    missing_examples.append({"audio": audio, "feature": feature, "reason": "missing_audio"})
                continue
            if not feature_path.exists():
                stats["missing_feature"] += 1
                if len(missing_examples) < 10:
                    missing_examples.append({"audio": audio, "feature": feature, "reason": "missing_feature"})
                continue
            rows.append(f"{audio_path}\t{feature_path}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(rows)
    stats["written"] = len(rows)
    stats["missing_examples"] = missing_examples
    return stats


def disk_free_gb(path):
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def collect_environment_dict(project_root, extra_tools=None):
    extra_tools = extra_tools or []
    env = {
        "timestamp": iso_now(),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": str(Path(project_root).resolve()),
        "disk_free_gb": disk_free_gb(project_root),
        "packages": {},
        "cuda": {},
        "git": {},
        "tools": {},
    }

    for mod_name in ["torch", "torchaudio", "numpy", "accelerate", "beartype", "einops", "optuna", "thop", "filelock", "matplotlib", "soundfile"]:
        try:
            mod = __import__(mod_name)
            env["packages"][mod_name] = getattr(mod, "__version__", "installed")
        except Exception as exc:
            env["packages"][mod_name] = f"missing: {exc.__class__.__name__}: {exc}"

    try:
        import torch

        env["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()),
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        env["cuda"] = {"error": repr(exc)}

    for git_args, key in [(["git", "rev-parse", "HEAD"], "commit"), (["git", "status", "--short"], "status_short")]:
        result = run_command(git_args, cwd=project_root, timeout=20)
        env["git"][key] = result["stdout"] if result["returncode"] == 0 else result["stderr"]

    for tool in extra_tools:
        located = shutil.which(tool)
        if located:
            version = run_command([tool, "-version"], cwd=project_root, timeout=20)
            if version["returncode"] != 0:
                version = run_command([tool, "--version"], cwd=project_root, timeout=20)
            env["tools"][tool] = {
                "path": located,
                "version": version["stdout"].splitlines()[:3],
                "stderr": version["stderr"].splitlines()[:3],
            }
        else:
            env["tools"][tool] = {"path": None, "version": None, "stderr": "not found"}

    return env


def format_environment_markdown(env):
    lines = [
        "# Environment",
        "",
        f"- timestamp: {env.get('timestamp')}",
        f"- python_executable: `{env.get('python_executable')}`",
        f"- python_version: `{env.get('python_version')}`",
        f"- platform: `{env.get('platform')}`",
        f"- cwd: `{env.get('cwd')}`",
        f"- disk_free_gb: {env.get('disk_free_gb'):.2f}",
        "",
        "## Packages",
    ]
    for key, value in env.get("packages", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## CUDA"])
    for key, value in env.get("cuda", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Git"])
    lines.append(f"- commit: `{env.get('git', {}).get('commit')}`")
    lines.append("- status_short:")
    status = env.get("git", {}).get("status_short") or ""
    if status:
        for line in status.splitlines():
            lines.append(f"  - `{line}`")
    else:
        lines.append("  - clean")
    lines.extend(["", "## Tools"])
    for tool, info in env.get("tools", {}).items():
        lines.append(f"- {tool}: `{info.get('path')}`")
        version = info.get("version")
        if version:
            for line in version:
                lines.append(f"  - `{line}`")
        elif info.get("stderr"):
            lines.append(f"  - `{info.get('stderr')}`")
    lines.append("")
    return "\n".join(lines)
