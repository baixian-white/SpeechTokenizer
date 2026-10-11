"""Build and verify the explicitly scoped demo handoff ZIP64 package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ARTIFACT_CATEGORIES = {
    "lca", "base", "ecapa", "exp23_cache", "exp23_cache_metadata",
    "exp23_seed", "exp23_aggregate", "exp23_bundle",
    "demo_enrollment", "historical_demo_receipt",
}
CODE_SCOPES = (
    "speechtokenizer", "nas", "scripts", "config", "tests", "docs",
    "3\u7528\u6237demo", "setup.py", "README.md", "LICENSE",
    "PROJECT_STRUCTURE.md", "REPRODUCIBILITY.md",
)
NO_COMPRESS = {".flac", ".npz", ".zip", ".png", ".jpg", ".jpeg", ".pdf"}
PACKAGE_ROOT = "six_user_demo_handoff_20260920"


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json_text(value), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def code_files(root):
    raw = git(root, "ls-files", "-z", "--", *CODE_SCOPES)
    return [part.decode("utf-8") for part in raw.split(b"\0") if part]


def walk_files(root, skip_dirs=()):
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name not in skip_dirs)
        for name in sorted(files):
            if name.startswith("~$") or name.endswith((".pyc", ".pyo")):
                continue
            yield Path(folder) / name


def is_old_key(path):
    return path.name == "46.txt" and "userA" in path.parts


def validate_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name:
        raise ValueError("Unsafe archive path: " + name)
    return name


def create_plan(source, exp23, handoff):
    entries = {}
    overrides = []

    def add(path, destination, category, expected_hash="", commit="", run_id=""):
        path = Path(path)
        if is_old_key(path):
            return
        destination = validate_name(destination)
        stat = path.stat()
        row = {
            "source": str(path),
            "archive_path": destination,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "category": category,
            "expected_sha256": expected_hash,
            "git_commit": commit,
            "run_id": run_id,
        }
        if destination in entries and entries[destination]["source"] != str(path):
            overrides.append({
                "archive_path": destination,
                "previous_source": entries[destination]["source"],
                "selected_source": str(path),
            })
        entries[destination] = row

    commits = {}
    for root, prefix, category in (
        (exp23, "project", "exp23_source"),
        (source, "reference_legacy_source", "legacy_source_reference"),
    ):
        commit = git(root, "rev-parse", "HEAD").decode().strip()
        commits[category] = commit
        for relative in code_files(root):
            add(root / relative, prefix + "/" + relative, category, commit=commit)

    with (handoff / "SHA256SUMS.csv").open(encoding="utf-8-sig", newline="") as handle:
        old_inventory = list(csv.DictReader(handle))
    for row in old_inventory:
        if row["category"] not in ARTIFACT_CATEGORIES:
            continue
        path = Path(row["path"])
        try:
            relative = path.relative_to(source)
        except ValueError:
            relative = path.relative_to(exp23)
        add(path, "project/" + relative.as_posix(), row["category"],
            row["sha256"], row["git_commit"], row["run_id"])

    for path in walk_files(source / "data" / "VCTK"):
        add(path, "project/" + path.relative_to(source).as_posix(), "vctk")
    demo = next(path for path in source.glob("3*demo") if path.is_dir())
    add(demo / "example_input.wav", "project/" + (demo / "example_input.wav").relative_to(source).as_posix(), "demo_example")
    for path in walk_files(handoff, skip_dirs=("__pycache__", "zip_support")):
        add(path, "handoff/" + path.relative_to(handoff).as_posix(), "handoff_documentation")

    support = handoff / "zip_support"
    add(support / "START_HERE_zh.md", "START_HERE_zh.md", "package_documentation")
    add(support / "package_tools.py", "package_tools.py", "package_tools")
    add(support / "build_handoff_zip.py", "packaging/build_handoff_zip.py", "packaging_source")
    snapshot = (
        Path.home() / ".cache" / "huggingface" / "hub" /
        "models--speechbrain--spkrec-ecapa-voxceleb" / "snapshots" /
        "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
    )
    for path in walk_files(snapshot):
        add(path, "reference_hf_snapshot/" + path.relative_to(snapshot).as_posix(), "ecapa_snapshot")

    planned = sorted(entries.values(), key=lambda item: item["archive_path"])
    names = [item["archive_path"] for item in planned]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Case-insensitive archive path collision")
    assert not any("/userA/46.txt" in name for name in names)
    assert not any("/.git/" in name for name in names)
    category_counts = dict(Counter(item["category"] for item in planned))
    vctk = [item for item in planned if item["category"] == "vctk"]
    assert sum(item["archive_path"].endswith(".flac") for item in vctk) == 88328
    assert category_counts["exp23_cache"] == 24200
    dataset_dirs = {
        PurePosixPath(item["archive_path"]).parts[4]
        for item in vctk if "/wav48_silence_trimmed/" in item["archive_path"]
        and item["archive_path"].endswith(".flac")
    }
    assert len(dataset_dirs) == 110
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source),
        "exp23_root": str(exp23),
        "commits": commits,
        "files": planned,
        "files_before_generated_metadata": len(planned),
        "total_input_bytes": sum(item["bytes"] for item in planned),
        "category_counts": category_counts,
        "explicitly_excluded": [
            "Old demo key userA/46.txt per latest user instruction",
            ".git", "unrelated datasets and experiments",
            "existing project archives", "__pycache__", "manuscript",
        ],
        "source_overrides": overrides,
        "zip_password": None,
        "key_included": False,
    }


def zip_info(name, size, suffix):
    info = zipfile.ZipInfo(PACKAGE_ROOT + "/" + name, date_time=(2026, 9, 20, 0, 0, 0))
    info.file_size = size
    info.compress_type = zipfile.ZIP_STORED if suffix in NO_COMPRESS else zipfile.ZIP_DEFLATED
    info._compresslevel = 1
    info.external_attr = 0o100644 << 16
    return info


def write_payload(archive, row):
    path = Path(row["source"])
    before = path.stat()
    if before.st_size != row["bytes"] or before.st_mtime_ns != row["mtime_ns"]:
        raise RuntimeError("Source changed since preflight: " + str(path))
    digest = hashlib.sha256()
    info = zip_info(row["archive_path"], row["bytes"], path.suffix.lower())
    with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            target.write(chunk)
            digest.update(chunk)
    after = path.stat()
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise RuntimeError("Source changed during packing: " + str(path))
    value = digest.hexdigest()
    if row["expected_sha256"] and row["expected_sha256"] != value:
        raise RuntimeError("Previously audited SHA-256 mismatch: " + str(path))
    return {
        "archive_path": row["archive_path"],
        "bytes": row["bytes"],
        "sha256": value,
        "category": row["category"],
        "git_commit": row["git_commit"],
        "run_id": row["run_id"],
    }


def manifest_bytes(rows):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=(
        "archive_path", "bytes", "sha256", "category", "git_commit", "run_id"
    ))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exp23-root", type=Path, required=True)
    parser.add_argument("--handoff-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    final = args.zip_path.absolute()
    partial = final.with_name(final.name + ".partial")
    progress = final.with_name(final.name + ".progress.json")
    verification = final.with_name(final.name + ".verification.json")
    if final.exists() or partial.exists():
        raise FileExistsError("Existing archive or partial preserved; inspect it before resuming: " + str(final))
    plan = create_plan(args.source_root.absolute(), args.exp23_root.absolute(), args.handoff_dir.absolute())
    final.parent.mkdir(parents=True, exist_ok=True)
    write_json(final.with_name(final.name + ".plan.json"), plan)
    if shutil.disk_usage(final.parent).free < plan["total_input_bytes"] * 1.15 + 1024 ** 3:
        raise RuntimeError("Insufficient free space for ZIP and verification metadata")
    print(json_text({key: value for key, value in plan.items() if key not in {"files", "source_overrides"}}), flush=True)
    if args.preflight_only:
        return 0

    started = time.monotonic()
    last_report = 0.0
    processed_bytes = 0
    rows = []

    def report(stage, count, total, force=False):
        nonlocal last_report
        now = time.monotonic()
        if not force and now - last_report < 10:
            return
        last_report = now
        status = {
            "stage": stage, "files_done": count, "files_total": total,
            "payload_bytes_processed": processed_bytes,
            "elapsed_seconds": round(now - started, 1),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(progress, status)
        print(json.dumps(status), flush=True)

    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for index, row in enumerate(plan["files"], 1):
                rows.append(write_payload(archive, row))
                processed_bytes += row["bytes"]
                report("packing", index, len(plan["files"]))
            package_metadata = {
                key: value for key, value in plan.items()
                if key not in {"files", "source_overrides"}
            }
            package_metadata.update({
                "package_root": PACKAGE_ROOT,
                "cache_scope": "Canonical 110-class LCA L3 only",
                "text_controlled_manifest": "Newly generated on 2026-09-20, not a historical formal experiment",
                "known_runtime_issues": "See START_HERE_zh.md and handoff/verification/verification_summary.json",
                "manifest_scope": "All payload files except PACKAGE_SHA256SUMS.csv itself",
            })
            metadata_bytes = json_text(package_metadata).encode("utf-8")
            metadata_name = "PACKAGE_CONTENTS.json"
            archive.writestr(zip_info(metadata_name, len(metadata_bytes), ".json"), metadata_bytes)
            rows.append({
                "archive_path": metadata_name, "bytes": len(metadata_bytes),
                "sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                "category": "package_metadata", "git_commit": "", "run_id": "",
            })
            manifest = manifest_bytes(rows)
            archive.writestr(zip_info("PACKAGE_SHA256SUMS.csv", len(manifest), ".csv"), manifest)
        report("verifying_archive", 0, len(rows), force=True)
        processed_bytes = 0
        with zipfile.ZipFile(partial, "r") as archive:
            infos = archive.infolist()
            assert len(infos) == len(rows) + 1
            assert len({info.filename for info in infos}) == len(infos)
            assert all(not (info.flag_bits & 1) for info in infos)
            assert not any(info.filename.endswith("/userA/46.txt") for info in infos)
            stored_manifest = archive.read(PACKAGE_ROOT + "/PACKAGE_SHA256SUMS.csv")
            assert stored_manifest == manifest
            for index, row in enumerate(rows, 1):
                digest = hashlib.sha256()
                size = 0
                with archive.open(PACKAGE_ROOT + "/" + row["archive_path"]) as handle:
                    for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                assert size == row["bytes"], row["archive_path"]
                assert digest.hexdigest() == row["sha256"], row["archive_path"]
                processed_bytes += size
                report("verifying_archive", index, len(rows))
        report("hashing_zip", len(rows), len(rows), force=True)
        archive_hash = file_hash(partial)
        if final.exists():
            raise FileExistsError("Refusing to replace an existing final archive")
        partial.rename(final)
        final.with_name(final.name + ".sha256").write_text(
            archive_hash + "  " + final.name + "\n", encoding="ascii"
        )
        receipt = {
            "success": True,
            "zip_path": str(final),
            "zip_bytes": final.stat().st_size,
            "zip_sha256": archive_hash,
            "verified_payload_files": len(rows),
            "zip_entries": len(rows) + 1,
            "payload_bytes": sum(row["bytes"] for row in rows),
            "zip_crc_and_all_payload_sha256": "passed",
            "key_included": False,
            "encrypted": False,
            "partial_exists": partial.exists(),
            "elapsed_seconds": round(time.monotonic() - started, 1),
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(verification, receipt)
        report("complete", len(rows), len(rows), force=True)
        print(json_text(receipt), flush=True)
    except BaseException as error:
        write_json(verification, {
            "success": False, "error": str(error),
            "partial_path": str(partial), "partial_preserved": partial.exists(),
        })
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
