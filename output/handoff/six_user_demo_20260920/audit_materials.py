"""Audit local demo handoff inputs without modifying experimental artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


LCA_RUN = "exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42"
BASE_RUN = "exp2_scit_speech_distill30_retrain_20260529_seed42"
ECAPA_DIR = "speechbrain_spkrec_ecapa_voxceleb"
DEMO_RUN = "exp22_three_user_speaker_demo_ecapa_smoke_20260708_seed42_rerun2_cpu"
PROFILE_RUN = "exp22_three_user_speaker_demo_smoke_20260708_seed42"
ROWS = {}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path, category, run_id="", git_commit="", provenance=""):
    path = Path(path).absolute()
    key = str(path)
    if key not in ROWS:
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("File changed during hashing: " + key)
        ROWS[key] = {
            "path": key,
            "bytes": after.st_size,
            "sha256": digest,
            "category": category,
            "run_id": run_id,
            "git_commit": git_commit,
            "provenance": provenance,
        }
    return ROWS[key]


def record_tree(root, category, run_id="", git_commit="", provenance=""):
    files = sorted(path for path in Path(root).rglob("*") if path.is_file())
    for index, path in enumerate(files, 1):
        record(path, category, run_id, git_commit, provenance)
        if index % 5000 == 0:
            print(f"Hashed {category}: {index}/{len(files)}", flush=True)
    return {"files": len(files), "bytes": sum(path.stat().st_size for path in files)}


def git_state(root):
    def run(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, encoding="utf-8"
        ).strip()

    return {
        "root": str(root),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status": run("status", "--short"),
        "note": "Current checkout revision is not necessarily the training revision.",
    }


def tree_inventory(root):
    root = Path(root)
    dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "path": str(root),
        "speaker_directories": dirs,
        "directory_count": len(dirs),
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "extensions": dict(Counter(path.suffix for path in files)),
    }


def split_audit(manifest):
    speakers = manifest["speakers"]
    group_sets = {}
    text_sets = {}
    missing = []
    per_speaker = {}
    for split, rows in manifest["splits"].items():
        groups = {(row["speaker_id"], row["utterance_group"]) for row in rows}
        if len(groups) != len(rows):
            raise AssertionError(f"Duplicate groups in {split}")
        group_sets[split] = groups
        text_sets[split] = {
            row["transcript_hash"] for row in rows if row.get("transcript_hash")
        }
        counts = Counter(row["speaker_id"] for row in rows)
        expected = manifest["counts"]["per_speaker"][split]
        assert set(counts) == set(speakers)
        assert set(counts.values()) == {expected}
        per_speaker[split] = {"minimum": min(counts.values()), "maximum": max(counts.values())}
        missing.extend(row["audio_path"] for row in rows if not Path(row["audio_path"]).is_file())
    overlap = {}
    text_overlap = {}
    names = list(group_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            pair = left + "/" + right
            overlap[pair] = len(group_sets[left] & group_sets[right])
            text_overlap[pair] = len(text_sets[left] & text_sets[right])
    assert not missing
    assert not any(overlap.values())
    if manifest["protocol"]["name"] == "exp23_text_controlled_109_class":
        assert not any(text_overlap.values())
    return {
        "protocol": manifest["protocol"],
        "seed": manifest["seed"],
        "speakers": len(speakers),
        "has_s5": "s5" in speakers,
        "has_p315": "p315" in speakers,
        "excluded_speakers": manifest["excluded_speakers"],
        "totals": {key: len(value) for key, value in manifest["splits"].items()},
        "per_speaker": per_speaker,
        "group_overlap": overlap,
        "text_overlap": text_overlap if any(text_sets.values()) else "not applicable",
        "missing_audio_paths": missing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--exp23-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source = args.source_root.resolve()
    exp23 = args.exp23_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    formal = exp23 / "tmp" / "exp23_formal"
    lca = source / "output" / "experiments" / LCA_RUN
    base = source / "output" / "experiments" / BASE_RUN
    demo = next(path for path in source.glob("3*demo") if path.is_dir())
    demo_code = next(demo.rglob("group_client.py")).parent
    exp23_demo = exp23 / demo_code.relative_to(source)
    key = next(demo.rglob("userA/46.txt"))
    source_git = git_state(source)
    exp23_git = git_state(exp23)
    generated_at = datetime.now(timezone.utc).isoformat()

    print("Auditing codec checkpoints and configs", flush=True)
    demo_ckpt = demo / "Log" / "spt_base" / "SpeechTokenizer_best_dev.pt"
    lca_ckpt = lca / "checkpoints" / "SCIT-Speech-LCA_v2_step30000_robust_optimum.pt"
    base_ckpt = base / "checkpoints" / "SCIT-Speech-Base_best.pt"
    lca_files = [
        demo_ckpt,
        demo / "Log" / "spt_base" / "config.json",
        *demo.glob("exp3*/configs/best_seanet_config.json"),
        lca_ckpt,
        lca / "checkpoints" / "SCIT-Speech-LCA_best.pt",
        lca / "checkpoints" / "checkpoint_manifest.json",
        lca / "configs" / "lca_finetune_config.json",
        lca / "configs" / "best_seanet_config.json",
    ]
    for path in lca_files:
        record(path, "lca", LCA_RUN, provenance="Training commit not recorded; run manifest available.")
    for path in [
        base_ckpt,
        base / "configs" / "scit_speech_base_config.json",
        base / "configs" / "best_seanet_config.json",
        base / "checkpoints" / "checkpoint_manifest.json",
    ]:
        record(path, "base", BASE_RUN, provenance="Training commit not recorded; run manifest available.")
    assert record(demo_ckpt, "lca")["sha256"] == record(lca_ckpt, "lca")["sha256"]
    assert record(lca_ckpt, "lca")["sha256"] == read_json(lca / "checkpoints" / "checkpoint_manifest.json")["sha256"]
    assert record(base_ckpt, "base")["sha256"] == read_json(base / "checkpoints" / "checkpoint_manifest.json")["sha256"]
    demo_config = read_json(demo / "Log" / "spt_base" / "config.json")
    training_config = read_json(lca / "configs" / "lca_finetune_config.json")
    assert demo_config == training_config

    print("Auditing dataset and canonical split", flush=True)
    audio_root = source / "data" / "VCTK" / "wav48_silence_trimmed"
    transcript_root = source / "data" / "VCTK" / "txt"
    audio_inventory = tree_inventory(audio_root)
    transcript_inventory = tree_inventory(transcript_root)
    missing_transcript_speakers = sorted(
        set(audio_inventory["speaker_directories"]) - set(transcript_inventory["speaker_directories"])
    )
    assert audio_inventory["directory_count"] == 110
    assert transcript_inventory["directory_count"] == 109
    assert missing_transcript_speakers == ["p315"]
    canonical_path = formal / "cache_lca_l3" / "cache_manifest" / "split_manifest.json"
    canonical = split_audit(read_json(canonical_path))

    # Use the repository's pure-data implementation without importing GPU/model modules.
    module_path = exp23 / "speechtokenizer" / "speaker_identity" / "data.py"
    spec = importlib.util.spec_from_file_location("handoff_exp23_split_data", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    print("Generating new 109-class text-controlled manifest using seed 1", flush=True)
    controlled_manifest = module.build_text_controlled_split(
        audio_root,
        train_count=80,
        validation_count=10,
        test_count=10,
        seed=1,
        transcript_root=transcript_root,
    )
    controlled = split_audit(controlled_manifest)
    controlled_path = out / "text_controlled_109_split.generated.json"
    write_json(controlled_path, controlled_manifest)
    record(
        controlled_path,
        "generated_manifest",
        "handoff_20260920",
        exp23_git["commit"],
        "Newly generated from current dataset and data.py; not an existing formal experiment result.",
    )
    record(module_path, "exp23_source", git_commit=exp23_git["commit"])
    controlled_receipt = {
        "generated_at_utc": generated_at,
        "status": "newly_generated_not_formal_training_or_test",
        "implementation": record(module_path, "exp23_source"),
        "manifest": record(controlled_path, "generated_manifest"),
        "audio_root": str(audio_root),
        "transcript_root": str(transcript_root),
        "validation": controlled,
    }
    write_json(out / "text_controlled_generation_receipt.json", controlled_receipt)

    print("Verifying cached file references and hashing all cache payloads", flush=True)
    records_path = canonical_path.with_name("cache_records.json")
    cache_records = read_json(records_path)
    missing = []
    referenced_cache = set()
    for row in cache_records:
        assert row["checkpoint_sha256"] == record(lca_ckpt, "lca")["sha256"]
        assert row["model_name"] == "lca" and row["rvq_layers"] == 3
        for name in ("source_audio_path", "codes_path", "reconstruction_path"):
            path = Path(row[name])
            if not path.is_file():
                missing.append(str(path))
            if name != "source_audio_path":
                referenced_cache.add(str(path.resolve()))
    assert not missing, missing[:5]
    actual_cache = {
        str(path.resolve()) for path in (formal / "cache").rglob("*") if path.is_file()
    }
    assert actual_cache == referenced_cache
    assert len(cache_records) == 12100
    cache_stats = record_tree(formal / "cache", "exp23_cache", "exp23_formal/cache", provenance="Canonical LCA L3 only.")
    record_tree(formal / "cache_lca_l3", "exp23_cache_metadata", "exp23_formal/cache_lca_l3")
    cache_audit = {
        **cache_stats,
        "path": str(formal / "cache"),
        "records": len(cache_records),
        "counts": dict(Counter(row["split"] for row in cache_records)),
        "conditions": sorted({(row["model_name"], row["rvq_layers"]) for row in cache_records}),
        "missing_paths": missing,
        "unreferenced_payload_files": sorted(actual_cache - referenced_cache),
        "recorded_failures": read_json(canonical_path.with_name("cache_failures.json")),
        "absolute_paths_require_rebasing_on_recipient_machine": True,
        "checkpoint_hashes": sorted({row["checkpoint_sha256"] for row in cache_records}),
        "config_hashes": sorted({row["config_sha256"] for row in cache_records}),
    }

    print("Auditing formal seed artifacts and export bundle", flush=True)
    seeds = []
    for seed in (41, 42, 43):
        root = formal / f"seed{seed}"
        environment = read_json(root / "environment.json")
        revision = environment["git"]["revision"]
        record_tree(root, "exp23_seed", f"exp23_formal/seed{seed}", revision, "Historical environment records dirty=true; commit alone is not a complete source snapshot.")
        validation = read_json(root / "metrics" / "validation_metrics.json")
        test = read_json(root / "metrics" / "test_metrics.json")
        for split in ("validation", "test"):
            with (root / "metrics" / f"{split}_predictions.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 1650
        seeds.append({
            "seed": seed,
            "root": str(root),
            "checkpoint": record(root / "checkpoints" / "best_macro_f1.pt", "exp23_seed"),
            "validation": {key: value for key, value in validation.items() if key != "per_speaker_accuracy"},
            "test": {key: value for key, value in test.items() if key != "per_speaker_accuracy"},
            "temperature": read_json(root / "metrics" / "temperature.json")["temperature"],
            "historical_environment": environment,
        })
    record_tree(formal / "aggregate", "exp23_aggregate", "exp23_formal/aggregate")
    bundle = formal / "export" / "seed42_bundle"
    bundle_manifest = read_json(bundle / "manifest.json")
    record_tree(bundle, "exp23_bundle", "exp23_formal/export/seed42_bundle")
    for name, expected in bundle_manifest["files"].items():
        assert record(bundle / name, "exp23_bundle")["sha256"] == expected
    labels = read_json(bundle / "speaker_labels.json")
    assert len(labels) == 110

    print("Auditing ECAPA files and local Hugging Face revision", flush=True)
    ecapa = source / "output" / "models" / ECAPA_DIR
    record_tree(ecapa, "ecapa", "speechbrain/spkrec-ecapa-voxceleb")
    hf_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--speechbrain--spkrec-ecapa-voxceleb"
    revision = (hf_root / "refs" / "main").read_text(encoding="ascii").strip()
    snapshot = hf_root / "snapshots" / revision
    ecapa_matches = {}
    for path in sorted(ecapa.iterdir()):
        if not path.is_file():
            continue
        snapshot_name = "label_encoder.txt" if path.name == "label_encoder.ckpt" else path.name
        snapshot_path = snapshot / snapshot_name
        exp23_path = exp23 / "output" / "models" / ECAPA_DIR / path.name
        source_hash = record(path, "ecapa")["sha256"]
        ecapa_matches[path.name] = {
            "snapshot_path": str(snapshot_path),
            "snapshot_match": sha256(snapshot_path) == source_hash,
            "exp23_copy_match": sha256(exp23_path) == source_hash,
        }
        assert all(ecapa_matches[path.name][key] for key in ("snapshot_match", "exp23_copy_match"))

    record(key, "demo_key_private", "legacy_demo", provenance="Hash and path only; do not publish key contents.")
    for root, category, commit in (
        (demo_code, "legacy_demo_source", source_git["commit"]),
        (exp23_demo, "exp23_demo_source", exp23_git["commit"]),
    ):
        for path in root.iterdir():
            if path.is_file() and path.suffix in {".py", ".md", ".yml", ".txt"}:
                record(path, category, git_commit=commit, provenance="Current source snapshot, not proof of historical training source.")
    record_tree(source / "output" / "experiments" / DEMO_RUN, "historical_demo_receipt", DEMO_RUN)
    record_tree(source / "output" / "experiments" / PROFILE_RUN / "speaker_profiles", "demo_enrollment", PROFILE_RUN)
    for path in (exp23 / "speechtokenizer" / "speaker_identity").glob("*.py"):
        record(path, "exp23_source", git_commit=exp23_git["commit"])
    for path in (exp23 / "scripts").glob("*exp23*.py"):
        record(path, "exp23_source", git_commit=exp23_git["commit"])
    record(exp23 / "config" / "exp23_speaker_classifier.json", "exp23_config", git_commit=exp23_git["commit"])

    packages = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        text=True,
        encoding="utf-8",
    )
    environment = {
        "captured_at_utc": generated_at,
        "executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": json.loads(packages),
    }
    write_json(out / "environment_current.json", environment)
    audit = {
        "audited_at_utc": generated_at,
        "source_checkout": source_git,
        "exp23_checkout": exp23_git,
        "codec": {
            "lca_demo_and_formal_file_hash_identical": True,
            "lca_demo_and_training_config_json_semantically_equal": True,
            "lca": record(lca_ckpt, "lca"),
            "base": record(base_ckpt, "base"),
        },
        "dataset": {
            "audio": audio_inventory,
            "transcripts": transcript_inventory,
            "audio_speakers_without_transcript_dir": missing_transcript_speakers,
            "scope": "Local counts and split input checks; no comparison with an official full-dataset checksum inventory.",
        },
        "canonical_split": {"path": str(canonical_path), **canonical},
        "text_controlled_split": {"path": str(controlled_path), "newly_generated": True, **controlled},
        "cache": cache_audit,
        "seeds": seeds,
        "bundle": {
            "path": str(bundle),
            "manifest_hash_check_passed": True,
            "model_config": read_json(bundle / "model_config.json"),
            "preprocessing": read_json(bundle / "preprocessing.json"),
            "historical_smoke": read_json(bundle / "validation_smoke.json"),
        },
        "ecapa": {
            "path": str(ecapa),
            "historical_speechbrain_version": seeds[1]["historical_environment"]["packages"]["speechbrain"]["version"],
            "local_hf_snapshot_revision": revision,
            "local_snapshot_hash_matches": ecapa_matches,
            "scope": "Snapshot identified by content hashes, not a newly downloaded or independently recorded training-time revision.",
        },
        "key": {"path": str(key), "contents_not_exported": True},
        "gaps": [
            "No pre-existing formal text-controlled 109-class manifest found in audited workspaces; generated one separately.",
            "No formal 109-class or two-second robustness results, or complete Base-condition cache, in exp23_formal.",
            "Historical Exp23 environment says git dirty=true; full historical source provenance is incomplete.",
            "Exp2/Exp3 checkpoint manifests do not record a training-time Git commit.",
            "Old demo environment.yml differs from the verified local environment and omits SpeechBrain.",
            "Six-user live-microphone acceptance has not been performed by this materials audit.",
        ],
    }
    write_json(out / "materials_audit.json", audit)
    with (out / "SHA256SUMS.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "path", "bytes", "sha256", "category", "run_id", "git_commit", "provenance"
        ])
        writer.writeheader()
        writer.writerows(ROWS[key] for key in sorted(ROWS))
    summary = {
        "files_hashed": len(ROWS),
        "bytes_hashed": sum(row["bytes"] for row in ROWS.values()),
        "cache_files_hashed": cache_stats["files"],
        "canonical_counts": canonical["totals"],
        "text_controlled_counts": controlled["totals"],
        "bundle_hash_check": True,
        "output_dir": str(out),
    }
    write_json(out / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
