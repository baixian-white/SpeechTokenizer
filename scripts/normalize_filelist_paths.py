import argparse
import json
from pathlib import Path


def _resolve_path(raw_path: str, repo_root: Path) -> Path | None:
    raw = Path(raw_path)
    if raw.exists():
        return raw

    parts = list(raw.parts)
    if "data" in parts:
        idx = parts.index("data")
        candidate = repo_root.joinpath(*parts[idx:])
        if candidate.exists():
            return candidate

    candidate = repo_root / raw_path
    if candidate.exists():
        return candidate

    return None


def normalize_filelist(input_path: Path, output_path: Path, repo_root: Path, limit: int | None = None) -> dict:
    total = 0
    written = 0
    missing = []
    rows = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            fields = line.split("\t")
            if len(fields) != 2:
                missing.append({"line": line, "reason": "expected two tab-separated fields"})
                continue
            audio = _resolve_path(fields[0], repo_root)
            feature = _resolve_path(fields[1], repo_root)
            if audio is None or feature is None:
                missing.append(
                    {
                        "line": line,
                        "audio_found": audio is not None,
                        "feature_found": feature is not None,
                    }
                )
                continue
            rows.append(f"{audio}\t{feature}\n")
            written += 1
            if limit is not None and written >= limit:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(rows), encoding="utf-8")
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_rows_seen": total,
        "rows_written": written,
        "limit": limit,
        "missing_count": len(missing),
        "missing_examples": missing[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a run-local filelist with paths resolved to this repository.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    report = normalize_filelist(
        input_path=Path(args.input),
        output_path=Path(args.output),
        repo_root=repo_root,
        limit=args.limit,
    )
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
