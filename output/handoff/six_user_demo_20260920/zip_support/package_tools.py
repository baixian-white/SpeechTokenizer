"""Verify extracted package files or prepare relocated Exp23 manifest copies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


OLD_ROOTS = (
    "D:/tmp/speechtokenizer-exp23",
    "H:/H-CODE/speechtokenizer",
    "D:/H-CODE/speechtokenizer",
)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or ":" in name:
        raise ValueError("Unsafe package member: " + name)
    destination = root.joinpath(*relative.parts).resolve()
    destination.relative_to(root.resolve())
    return destination


def verify(root):
    with (root / "PACKAGE_SHA256SUMS.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows, 1):
        path = safe_member(root, row["archive_path"])
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise ValueError("Missing file or size mismatch: " + row["archive_path"])
        if digest_file(path) != row["sha256"]:
            raise ValueError("SHA-256 mismatch: " + row["archive_path"])
        if index % 5000 == 0:
            print(f"Verified {index}/{len(rows)}", flush=True)
    print(f"PASS: {len(rows)} packaged files match their SHA-256 manifest.")


def rewrite_paths(value, project):
    if isinstance(value, dict):
        return {key: rewrite_paths(item, project) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_paths(item, project) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        for original in OLD_ROOTS:
            if normalized.casefold().startswith(original.casefold() + "/"):
                relative = normalized[len(original) + 1:]
                return str(safe_member(project, relative))
    return value


def validate_manifest_paths(value):
    count = 0
    if isinstance(value, list):
        return sum(validate_manifest_paths(item) for item in value)
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"audio_path", "source_audio_path", "codes_path", "reconstruction_path"}:
                if not Path(item).is_file():
                    raise FileNotFoundError(item)
                count += 1
            else:
                count += validate_manifest_paths(item)
    return count


def prepare_paths(root):
    project = root / "project"
    formal = project / "tmp" / "exp23_formal"
    target = project / "tmp" / "exp23_portable"
    inputs = {
        "cache_records.json": formal / "cache_lca_l3" / "cache_manifest" / "cache_records.json",
        "canonical_110_split.json": formal / "cache_lca_l3" / "cache_manifest" / "split_manifest.json",
        "text_controlled_109_split.json": root / "handoff" / "text_controlled_109_split.generated.json",
    }
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Keep previous relocated manifests intact; output is not empty: " + str(target))
    prepared = {}
    checked = 0
    for name, source in inputs.items():
        original = json.loads(source.read_text(encoding="utf-8-sig"))
        relocated = rewrite_paths(original, project)
        checked += validate_manifest_paths(relocated)
        prepared[name] = relocated
    target.mkdir(parents=True, exist_ok=True)
    for name, value in prepared.items():
        (target / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    receipt = {
        "package_root": str(root.resolve()),
        "project_root": str(project.resolve()),
        "validated_path_references": checked,
        "output_files": list(prepared),
        "original_manifests_unchanged": True,
        "note": "Only path strings were relocated. This is not a new split or experiment.",
    }
    (target / "relocation_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=True, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--prepare-paths", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not args.verify and not args.prepare_paths:
        parser.error("Choose --verify and/or --prepare-paths")
    root = args.root.resolve()
    if args.verify:
        verify(root)
    if args.prepare_paths:
        prepare_paths(root)


if __name__ == "__main__":
    main()
