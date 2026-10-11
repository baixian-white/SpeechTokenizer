"""Check the expanded supplement against retained records and its rendered PDF."""

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import pdfplumber
from PIL import Image, ImageDraw, ImageOps


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / "supplementary_methods_v2_assets"
BUILD = ROOT / "tmp/pdfs/supplementary_methods_v2"
REVIEW = ROOT / "tmp/pdfs/supplementary_methods_v2_review"
DELIVERY = ROOT / "output/pdf/supplementary_methods_v2.pdf"
SPEAKER = Path("D:/tmp/speechtokenizer-exp23")
FORMAL = SPEAKER / "tmp/exp23_formal"
EXPERIMENTS = ROOT / "output/experiments"
NAS = EXPERIMENTS / "exp1_nas_distill_run1_seed42"
BASE = EXPERIMENTS / "exp2_scit_speech_distill30_retrain_20260529_seed42"
LCA = EXPERIMENTS / "exp3_low_load_channel_aware_adaptation_v2_strong_perturb_20260531_seed42"
QUALITY = EXPERIMENTS / "exp12_baseline_comparison_test300_20260610_seed42"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_paths():
    paths = [
        NAS / "configs/nas_search_space.json",
        NAS / "configs/staged_nas_runtime_config.json",
        NAS / "metrics/stage1_profile.csv",
        NAS / "artifacts/best_architecture/best_seanet_config.json",
        NAS / "reports/environment.json",
        BASE / "configs/scit_speech_base_config.json",
        BASE / "artifacts/train_files.txt",
        BASE / "artifacts/valid_files.txt",
        LCA / "configs/lca_finetune_config.json",
        QUALITY / "commands/analyze_baseline_300_stats.py",
        QUALITY / "runs/test-clean_300/metrics/per_method_summary.csv",
        FORMAL / "cache_lca_l3/cache_manifest/split_manifest.json",
        FORMAL / "cache_lca_l3/cache_manifest/tokenizer_state_hashes.json",
        FORMAL / "cache_lca_l3/cache_manifest/cache_failures.json",
        FORMAL / "aggregate/bootstrap_summary.json",
        FORMAL / "export/seed42_bundle/preprocessing.json",
        FORMAL / "export/seed42_bundle/calibration.json",
        FORMAL / "seed42/environment.json",
    ]
    paths.extend(FORMAL / f"seed{seed}/training_config.json" for seed in (41, 42, 43))
    paths.extend(ROOT / name for name in (
        "speechtokenizer/trainer/loss.py",
        "speechtokenizer/trainer/lca_trainer.py",
        "speechtokenizer/quantization/core_vq.py",
        "speechtokenizer/model.py",
        "nas/run_staged_encoder_nas.py",
        "scripts/evaluate_packet_burst_loss.py",
        "scripts/evaluate_lca_vs_base.py",
    ))
    paths.extend(SPEAKER / "speechtokenizer/speaker_identity" / f"{name}.py"
                 for name in ("data", "cache", "token_encoder", "audio_encoder",
                              "fusion", "losses", "training", "metrics"))
    paths.append(SPEAKER / "scripts/train_exp23_speaker_classifier.py")
    return paths


def snapshot():
    ASSETS.mkdir(exist_ok=True)
    destination = ASSETS / "protected_snapshot.json"
    if destination.exists():
        raise FileExistsError("Keep the original snapshot; do not overwrite it.")
    protected = []
    for path in HERE.iterdir():
        if path.is_file() and (
            path.name.startswith(("results_", "methods_", "manuscript."))
            or path.name in (
                "supplementary_methods.tex", "supplementary_methods_body.tex",
                "supplementary_methods_references.tex", "supplementary_methods.pdf",
            )
        ):
            protected.append(path)
    for directory in ("results_v2_assets", "results_v3_assets"):
        protected.extend(path for path in (HERE / directory).rglob("*") if path.is_file())
    sources = source_paths()
    missing = [str(path) for path in protected + sources if not path.is_file()]
    assert not missing, missing
    record = {
        "captured_on": "2026-09-22",
        "scope": "Existing manuscript and Results files are read-only in this revision.",
        "protected": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(protected)},
        "sources": {str(path): sha256(path) for path in sources},
    }
    write_json(destination, record)
    print(json.dumps({"protected_files": len(protected), "source_files": len(sources)}))


def check_records():
    search = load_json(NAS / "configs/staged_nas_runtime_config.json")
    fixed = load_json(NAS / "configs/nas_search_space.json")["fixed"]
    assert [search[key] for key in (
        "stage1_candidates", "stage2_top_k", "stage3_top_k", "final_top_k"
    )] == [8192, 512, 64, 8]
    assert [search[key] for key in (
        "distill_stage2_steps", "distill_stage3_steps", "distill_final_steps"
    )] == [100, 500, 1500]
    assert (fixed["latent_dimension"], fixed["n_q"], fixed["codebook_size"]) == (1024, 3, 1024)
    rate = fixed["latent_rate"] * fixed["n_q"] * math.ceil(math.log2(fixed["codebook_size"]))
    assert rate == 1500 and math.ceil(rate * 0.5 / 8) == 94

    base = load_json(BASE / "configs/scit_speech_base_config.json")
    lca = load_json(LCA / "configs/lca_finetune_config.json")
    assert base["batch_size"] == lca["batch_size"] == 8
    assert base["distill_loss_lambda"] == lca["distill_loss_lambda"] == 30
    assert base["mel_loss_lambdas"] == [45, 1, 1, 1]
    assert (base["learning_rate"], lca["learning_rate"]) == (1e-4, 1e-5)
    assert (lca["lambda_full"], lca["lambda_comm"], lca["lambda_consistency"]) == (1, 1.5, 0.5)
    assert [(row["p_drop"], row["p_sub"]) for row in lca["channel_sim"]["conditions"]] == [
        (0, 0), (0.05, 0), (0.10, 0), (0, 0.01), (0, 0.03)
    ]
    manifests = [
        [line.strip() for line in (BASE / f"artifacts/{name}_files.txt").read_text(
            encoding="utf-8-sig").splitlines() if line.strip()]
        for name in ("train", "valid")
    ]
    assert list(map(len, manifests)) == [27039, 1500]
    assert not set(manifests[0]) & set(manifests[1])

    split = load_json(FORMAL / "cache_lca_l3/cache_manifest/split_manifest.json")
    assert len(split["speakers"]) == 110 and "s5" in split["speakers"]
    group_sets = {}
    for name, count in (("train", 8800), ("validation", 1650), ("test", 1650)):
        rows = split["splits"][name]
        assert len(rows) == count
        group_sets[name] = {(row["speaker_id"], row["utterance_group"]) for row in rows}
        assert len(group_sets[name]) == count
        for speaker in split["speakers"]:
            assert sum(row["speaker_id"] == speaker for row in rows) == count // 110
    assert not group_sets["train"] & (group_sets["validation"] | group_sets["test"])
    assert not group_sets["validation"] & group_sets["test"]
    states = load_json(FORMAL / "cache_lca_l3/cache_manifest/tokenizer_state_hashes.json")
    assert all(row["before_sha256"] == row["after_sha256"] for row in states)
    failures = load_json(FORMAL / "cache_lca_l3/cache_manifest/cache_failures.json")
    assert not failures, failures
    for seed in (41, 42, 43):
        run = load_json(FORMAL / f"seed{seed}/training_config.json")["args"]
        assert (run["seed"], run["epochs"], run["batch_size"], run["gradient_accumulation"]) == (
            seed, 30, 16, 8)
        assert run["ecapa_stage"] == "A"
    calibration = load_json(FORMAL / "export/seed42_bundle/calibration.json")
    assert calibration["temperature"] == 1.01877269160619
    archived_ci = load_json(FORMAL / "aggregate/bootstrap_summary.json")
    assert not any(key in archived_ci for key in ("replicates", "bootstrap_seed", "n_boot"))

    with (QUALITY / "runs/test-clean_300/metrics/per_method_summary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        summaries = list(csv.DictReader(stream))
    for method, setting in (
        ("scit_base", "L=3"), ("dac", "n_q_3"), ("encodec", "bw1.5kbps_n_cb2"),
        ("opus", "opus_6000bps"), ("pcm", "16bit_16khz_passthrough"),
    ):
        matches = [row for row in summaries if row["method"] == method and row["codec_setting"] == setting]
        assert len(matches) == 1 and int(matches[0]["n_samples"]) == 300
    return {
        "rate_bps": rate,
        "packed_half_second_bytes": 94,
        "codec_train_validation_counts": [27039, 1500],
        "speaker_split_counts": [8800, 1650, 1650],
        "speaker_count": 110,
        "codec_state_unchanged_during_cache": True,
        "cache_failures": 0,
        "speaker_seeds": [41, 42, 43],
        "speaker_temperature_seed42": calibration["temperature"],
        "speaker_bootstrap_execution_metadata": {
            "replicate_count": None,
            "rng_seed": None,
            "handling": "Unresolved author check; no executed values invented in the supplement.",
        },
    }


def check_text():
    body = (HERE / "supplementary_methods_v2_body.tex").read_text(encoding="utf-8")
    chinese = (HERE / "supplementary_methods_v2_zh.md").read_text(encoding="utf-8")
    references = (HERE / "supplementary_methods_references.tex").read_text(encoding="utf-8")
    sections = re.findall(r"\\section\{([^}]+)\}", body)
    subsections = re.findall(r"\\subsection\{([^}]+)\}", body)
    assert len(sections) == 9
    assert len(re.findall(r"(?m)^## S[1-9] ", chinese)) == 9
    assert len(re.findall(r"(?m)^### S\d+\.\d+ ", chinese)) == len(subsections)
    labels = re.findall(r"\\label\{([^}]+)\}", body)
    assert len(labels) == len(set(labels))
    used = set(re.findall(r"\\ref\{([^}]+)\}", body))
    assert used <= set(labels), used - set(labels)
    cites = set(re.findall(r"\\cite\{([^}]+)\}", body))
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", references))
    assert cites <= bibitems
    tables = body.count(r"\begin{table}") + body.count(r"\begin{table*}")
    assert tables == len(re.findall(r"(?m)^\*\*表 S\d+\.", chinese)) == 8
    for value in ("27,039", "1,500", "2,620", "8,800", "1,650", "107,500", "30,000",
                  "122,500", "32,500", "94", "1504", "1.01877269160619"):
        assert value in body and value in chinese, value
    assert not re.search(r"(?i)\b(?:TODO|TBD|placeholder)\b|Author confirmation required", body)
    assert "closed-set" in body and "闭集" in chinese
    assert "previous original index" in body.lower() and "前一原始索引" in chinese
    assert "archived endpoints" in body and "归档端点" in chinese
    assert body.isascii()
    prose = re.sub(r"\\begin\{(equation|align|table\*?)\}.*?\\end\{\1\}", "", body, flags=re.S)
    prose = re.sub(r"\$[^$]*\$", "", prose)
    prose = re.sub(r"\\(?:label|ref|cite)\{[^}]*\}", "", prose)
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", prose)
    return {
        "sections": len(sections),
        "subsections": len(subsections),
        "tables": tables,
        "numbered_equation_labels": len([label for label in labels if label.startswith("eq:")]),
        "approximate_english_prose_words_including_headings": len(words),
        "word_count_note": "Alphabetic groups; excludes tables, displayed math, inline math and citation keys.",
        "chinese_characters": len(re.findall(r"[\u4e00-\u9fff]", chinese)),
        "citations_resolve_to_retained_bibliography": True,
        "section_table_and_key_value_parity": True,
    }


def check_pdf():
    pdf = BUILD / "supplementary_methods_v2.pdf"
    pages = []
    all_text = []
    with pdfplumber.open(pdf) as document:
        for number, page in enumerate(document.pages, 1):
            text = page.extract_text(x_tolerance=1) or ""
            assert "??" not in text, number
            assert len(text) > 250, (number, len(text))
            outside = [
                char for char in page.chars if char["text"].strip()
                and (char["x0"] < 18 or char["x1"] > page.width - 18
                     or char["top"] < 16 or char["bottom"] > page.height - 16)
            ]
            assert not outside, f"Page bounds: {number}"
            all_text.append(text)
            pages.append({"page": number, "text_characters": len(text), "within_page_bounds": True})
    compact = re.sub(r"\s+", "", "\n".join(all_text))
    for number in range(1, 9):
        assert f"TableS{number}:" in compact, number
    for phrase in ("1500", "1504", "1.01877269160619", "110"):
        assert phrase in compact
    log = (BUILD / "supplementary_methods_v2.log").read_text(encoding="utf-8", errors="replace")
    errors = re.findall(
        r"(?im)^.*(?:Overfull|undefined references|undefined on input|"
        r"Undefined control|Fatal error|Label\(s\) may have changed|"
        r"referenced but does not exist).*$", log,
    )
    assert not errors, errors
    return {
        "page_count": len(pages), "pages": pages, "unresolved_references_or_overflows": errors,
        "underfull_messages": len(re.findall("Underfull", log)), "sha256": sha256(pdf),
    }


def contact_sheets(page_count):
    renders = list(REVIEW.glob("page-*.png"))
    if len(renders) != page_count:
        raise ValueError(f"Expected {page_count} rendered pages, found {len(renders)}.")
    images = {int(path.stem.split("-")[-1]): path for path in renders}
    outputs = []
    for start in range(1, page_count + 1, 4):
        contact = Image.new("RGB", (1500, 2150), "#eeeeee")
        draw = ImageDraw.Draw(contact)
        for offset, number in enumerate(range(start, min(start + 4, page_count + 1))):
            image = Image.open(images[number]).convert("RGB")
            x, y = (offset % 2) * 750, (offset // 2) * 1075
            draw.text((x + 15, y + 10), f"Page {number}", fill="black")
            thumb = ImageOps.contain(image, (720, 1025))
            contact.paste(thumb, (x + 15, y + 35))
        destination = REVIEW / f"contact-{start:02d}.png"
        contact.save(destination)
        outputs.append(str(destination))
    return outputs


def verify():
    record = load_json(ASSETS / "protected_snapshot.json")
    for relative, expected in record["protected"].items():
        assert sha256(ROOT / relative) == expected, f"Protected file changed: {relative}"
    for path, expected in record["sources"].items():
        assert sha256(Path(path)) == expected, f"Evidence changed: {path}"
    report = {
        "status": "passed",
        "checked_on": "2026-09-22",
        "records": check_records(),
        "text": check_text(),
        "pdf": check_pdf(),
        "protected_original_count": len(record["protected"]),
        "source_record_count": len(record["sources"]),
        "new_experiments_run": False,
        "original_results_reestimated": False,
    }
    if DELIVERY.exists():
        assert sha256(DELIVERY) == report["pdf"]["sha256"], "Export differs from reviewed build."
        report["delivery"] = {
            "path": str(DELIVERY),
            "matches_reviewed_build": True,
            "sha256": sha256(DELIVERY),
        }
    report["rendered_contact_sheets"] = contact_sheets(report["pdf"]["page_count"])
    write_json(ASSETS / "verification.json", report)
    print(json.dumps({key: report[key] for key in (
        "status", "text", "protected_original_count", "source_record_count"
    )}, ensure_ascii=False, indent=2))
    print(json.dumps({"pdf_pages": report["pdf"]["page_count"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    snapshot() if args.snapshot else verify()
