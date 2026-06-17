"""One-shot migration: rewrite E:\\CODE\\speechtokenizer references to H:\\H-CODE\\speechtokenizer.

Usage:
  python scripts/migrate_paths_e_to_h.py --files /tmp/migration_verify/files_to_rewrite.txt \
                                         --backup-dir /tmp/migration_verify/backups \
                                         [--dry-run]

The script replaces literal substrings (no regex). Both single-backslash (`E:\\CODE\\speechtokenizer`)
and double-backslash (`E:\\\\CODE\\\\speechtokenizer`) forms are handled because the latter is just
the former encoded twice on disk.
"""
import argparse
import shutil
import sys
from pathlib import Path

OLD_SINGLE = r"E:\CODE\speechtokenizer"
NEW_SINGLE = r"H:\H-CODE\speechtokenizer"
OLD_DOUBLE = r"E:\\CODE\\speechtokenizer"
NEW_DOUBLE = r"H:\\H-CODE\\speechtokenizer"


def rewrite_file(path: Path, backup_dir: Path | None, dry_run: bool) -> dict:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        text = raw.decode("utf-8-sig")
        encoding = "utf-8-sig"

    n_double = text.count(OLD_DOUBLE)
    text_after_double = text.replace(OLD_DOUBLE, NEW_DOUBLE)
    n_single = text_after_double.count(OLD_SINGLE)
    new_text = text_after_double.replace(OLD_SINGLE, NEW_SINGLE)
    changed = new_text != text

    result = {
        "path": str(path),
        "encoding": encoding,
        "double_backslash_replacements": n_double,
        "single_backslash_replacements": n_single,
        "changed": changed,
    }

    if not changed:
        return result

    if dry_run:
        result["status"] = "dry-run"
        return result

    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        rel = path.as_posix().replace(":", "_").replace("/", "__")
        shutil.copy2(path, backup_dir / rel)

    path.write_bytes(new_text.encode(encoding))
    result["status"] = "rewritten"
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True, help="text file listing target paths, one per line")
    ap.add_argument("--backup-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = [Path(p.strip()) for p in Path(args.files).read_text(encoding="utf-8").splitlines() if p.strip()]
    backup_dir = Path(args.backup_dir) if args.backup_dir else None

    summary = {"total": len(paths), "rewritten": 0, "dry_run": 0, "unchanged": 0, "missing": 0}
    rows = []
    for p in paths:
        if not p.exists():
            summary["missing"] += 1
            rows.append({"path": str(p), "status": "missing"})
            continue
        r = rewrite_file(p, backup_dir, args.dry_run)
        if not r["changed"]:
            summary["unchanged"] += 1
        elif args.dry_run:
            summary["dry_run"] += 1
        else:
            summary["rewritten"] += 1
        rows.append(r)

    import json
    print(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
