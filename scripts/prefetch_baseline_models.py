import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPOS = (
    "facebook/encodec_24khz",
    "descript/dac_16khz",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root):
    rows = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prefetch optional neural codec baseline model weights.")
    parser.add_argument(
        "--output-dir",
        default="output/baseline_model_cache",
        help="Directory for local model snapshots.",
    )
    parser.add_argument(
        "--repo",
        action="append",
        dest="repos",
        help="Hugging Face repo id to download. Can be passed multiple times.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision for every requested repo.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repos = tuple(args.repos or DEFAULT_REPOS)

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "note": (
            "These are optional neural codec baseline snapshots. Traditional "
            "PCM/Opus/AMR-WB/Codec2 baselines require codec tools, not checkpoints."
        ),
        "models": [],
    }

    for repo_id in repos:
        local_dir = output_dir / repo_id.replace("/", "__")
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            revision=args.revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        files = collect_files(snapshot_path)
        manifest["models"].append(
            {
                "repo_id": repo_id,
                "revision": args.revision or "default",
                "local_dir": str(Path(snapshot_path)),
                "file_count": len(files),
                "total_bytes": sum(row["bytes"] for row in files),
                "files": files,
            }
        )

    manifest_path = output_dir / "baseline_model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "models": manifest["models"]}, indent=2))


if __name__ == "__main__":
    main()
