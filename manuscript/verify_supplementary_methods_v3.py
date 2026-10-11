"""Validate supplement additions, protected originals, figures and rendered pages."""

import argparse
import json
import re
from pathlib import Path

import pdfplumber
from PIL import Image

import verify_supplementary_methods_v2 as prior


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / "supplementary_methods_v3_assets"
BUILD = ROOT / "tmp/pdfs/supplementary_methods_v3"
REVIEW = ROOT / "tmp/pdfs/supplementary_methods_v3_review_20261008"
DELIVERY = ROOT / "output/pdf/supplementary_methods_v3.pdf"


def snapshot():
    ASSETS.mkdir(exist_ok=True)
    target = ASSETS / "protected_snapshot.json"
    if target.exists():
        raise FileExistsError("The pre-edit protection snapshot already exists.")
    record = prior.load_json(prior.ASSETS / "protected_snapshot.json")
    for relative, expected in record["protected"].items():
        assert prior.sha256(ROOT / relative) == expected, relative
    for path, expected in record["sources"].items():
        assert prior.sha256(Path(path)) == expected, path
    keep = list(HERE.glob("supplementary_methods_v2*"))
    keep += [HERE / "verify_supplementary_methods_v2.py", prior.DELIVERY]
    for path in keep:
        files = path.rglob("*") if path.is_dir() else [path]
        for item in files:
            if item.is_file():
                record["protected"][str(item.relative_to(ROOT))] = prior.sha256(item)
    for relative in ("nas/SeaNet.py", "nas/model_components.py", "nas/encoder_only_model_variant.py"):
        path = ROOT / relative
        record["sources"][str(path)] = prior.sha256(path)
    record["captured_on"] = "2026-09-22"
    record["scope"] = "Preserve v2, the main manuscript, Results and experiment sources."
    prior.write_json(target, record)
    print(json.dumps({"protected_files": len(record["protected"]),
                      "evidence_files": len(record["sources"])}))


def check_geometry(body):
    architecture = prior.load_json(prior.NAS / "artifacts/best_architecture/best_seanet_config.json")
    assert architecture["encoder_strides"] == [5, 4, 4, 4]
    assert architecture["n_filters"] == 24 and architecture["lstm"] == 1
    assert architecture["layer_ops_list"] == ["skip", "std_k7", "sep_k9", "dil_k5"]
    assert not any(architecture["layer_se_list"])
    records = []
    for samples in (8000, 16000):
        length, channels = samples, architecture["n_filters"]
        shapes = [[1, channels, length]]
        for stride in architecture["encoder_strides"]:
            length = (length + stride - 1) // stride
            channels *= 2
            shapes.append([1, channels, length])
        shapes.extend([[1, 2 * channels, length], [1, architecture["dimension"], length]])
        assert shapes[-1] == [1, 1024, samples // 320]
        records.append({"samples": samples, "stage_shapes": shapes})
    for expression in (
        r"$24\rightarrow48$", r"$48\rightarrow96$", r"$96\rightarrow192$",
        r"$192\rightarrow384$", r"$384\rightarrow768$", r"$768\rightarrow1024$",
        r"$B\times3\times T_c$", r"$B\times256\times T_c$", r"$B\times512$",
        r"$B\times192$", r"$B\times110$",
    ):
        assert expression in body, expression
    source = (ROOT / "nas/model_components.py").read_text(encoding="utf-8")
    assert "'dil_k5'" in source and "dilation=D * 2" in source
    assert "nn.GroupNorm(1, channels)" in source
    return {"verified_from_retained_configuration_and_implementation": True,
            "shape_derivation": records, "model_benchmark_run": False}


def check_text():
    body = (HERE / "supplementary_methods_v3_body.tex").read_text(encoding="utf-8")
    chinese = (HERE / "supplementary_methods_v3_zh.md").read_text(encoding="utf-8")
    old = (HERE / "supplementary_methods_v2_body.tex").read_text(encoding="utf-8")
    terminology_changes = (
        (
            ". It follows the SpeechTokenizer representation framework~\\cite{speechtokenizer}, with an encoder selected by the procedure",
            "~\\cite{speechtokenizer}. The encoder was selected by the procedure",
        ),
        (
            "came from the fixed pretrained SpeechTokenizer teacher.",
            "came from the fixed pretrained SpeechTokenizer teacher~\\cite{speechtokenizer}.",
        ),
        (
            "The teacher was the pretrained HuBERT-average SpeechTokenizer checkpoint.",
            "The teacher was the pretrained HuBERT-average SpeechTokenizer checkpoint~\\cite{speechtokenizer}.",
        ),
    )
    for before, after in terminology_changes:
        assert old.count(before) == 1, before
        old = old.replace(before, after)
    stripped = re.sub(r"\n% BEGIN V[34]-[^\n]+\n.*?% END V[34]-[^\n]+\n", "", body, flags=re.S)
    assert stripped.strip() == old.strip(), "An existing English passage changed."
    sections = re.findall(r"\\section\{([^}]+)\}", body)
    subsections = re.findall(r"\\subsection\{([^}]+)\}", body)
    assert len(sections) == len(re.findall(r"(?m)^## S[1-9] ", chinese)) == 9
    assert len(subsections) == len(re.findall(r"(?m)^### S\d+\.\d+ ", chinese)) == 48
    tables = len(re.findall(r"\\begin\{table\*?\}", body))
    figures = len(re.findall(r"\\begin\{figure\*?\}", body))
    assert tables == len(re.findall(r"(?m)^\*\*表 S\d+\.", chinese)) == 11
    figure_numbers = [int(n) for n in re.findall(r"(?m)^\*\*图 S(\d+)\.", chinese)]
    assert figures == 5 and figure_numbers == list(range(1, figures + 1))
    assert body.count(r"\begin{algorithm}") == chinese.count("**算法 S1.") == 1
    labels = re.findall(r"\\label\{([^}]+)\}", body)
    assert len(labels) == len(set(labels))
    assert set(re.findall(r"\\ref\{([^}]+)\}", body)) <= set(labels)
    refs = (HERE / "supplementary_methods_references.tex").read_text(encoding="utf-8")
    assert set(re.findall(r"\\cite\{([^}]+)\}", body)) <= set(re.findall(r"\\bibitem\{([^}]+)\}", refs))
    assert body.isascii()
    assert not re.search(r"(?i)\b(TODO|TBD|placeholder)\b", body)
    for value in ("27,039", "1,500", "2,620", "8,800", "1,650", "107,500", "30,000",
                  "122,500", "32,500", "1.01877269160619"):
        assert value in body and value in chinese, value
    for condition in ("closed-set", "archived endpoints", "previous original index"):
        assert condition in body.lower(), condition
    for path in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", body):
        assert (HERE / path).is_file(), path
    prose = re.sub(r"(?m)(?<!\\)%.*$", "", body)
    prose = re.sub(r"\\begin\{(equation|align|table\*?|figure\*?|algorithm)\}.*?\\end\{\1\}",
                   "", prose, flags=re.S)
    prose = re.sub(r"\$[^$]*\$|\\(?:label|ref|cite)\{[^}]*\}", "", prose)
    return {
        "sections": len(sections), "subsections": len(subsections),
        "tables": tables, "figures": figures, "algorithms": 1,
        "numbered_equation_labels": sum(label.startswith("eq:") for label in labels),
        "original_english_passages_preserved": True,
        "english_chinese_structure_matches": True,
        "approximate_english_prose_words": len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", prose)),
        "chinese_characters": len(re.findall(r"[\u4e00-\u9fff]", chinese)),
        "geometry": check_geometry(body),
    }


def check_diagrams():
    evidence = prior.load_json(ASSETS / "diagram_checks.json")
    example = evidence["perturbation_example"]
    original = example["original"]
    expected = [original[index - 1] if selected and index else original[index]
                for index, selected in enumerate(example["erasure_mask"])]
    assert example["erased"] == expected
    assert len(example["packet_frames"]) * example["frame_ms"] == 100
    assert len(example["burst_frames"]) * example["frame_ms"] == 200
    receiver = evidence["receiver"]
    assert receiver["block_frames"] * receiver["layers"] * 10 == receiver["informative_bits"]
    assert receiver["packed_bytes"] * 8 - receiver["informative_bits"] == receiver["padding_bits"]
    for entry in evidence["layout_checks"]:
        path = ASSETS / f"{entry['figure']}.pdf"
        with pdfplumber.open(path) as document:
            assert len(document.pages) == 1
            page = document.pages[0]
            assert page.width == 510
            assert len(page.extract_text()) > 250
            for char in page.chars:
                assert 0 <= char["x0"] <= char["x1"] <= page.width
                assert 0 <= char["top"] <= char["bottom"] <= page.height
    return {"count": 2, "examples_match_protocol": True, "label_bounds_passed": True}


def check_data_figures():
    figures = ("nas_finalists.png", "paired_robustness.png", "speaker_accuracy.png")
    sizes = {}
    for name in figures:
        with Image.open(ASSETS / name) as image:
            assert image.format == "PNG" and min(image.size) >= 900, (name, image.size)
            sizes[name] = list(image.size)
    results_evidence = prior.load_json(HERE / "results_v3_assets/verification.json")
    hashes = results_evidence["speaker"]["sources_sha256"]
    for seed in (41, 42, 43):
        suffix = f"seed{seed}\\metrics\\test_metrics.json"
        expected = next(value for path, value in hashes.items() if path.endswith(suffix))
        snapshot = ASSETS / f"speaker_formal_metrics/seed{seed}/metrics/test_metrics.json"
        assert prior.sha256(snapshot) == expected, snapshot
    return {
        "count": len(figures), "pixel_dimensions": sizes,
        "formal_speaker_metrics_match_archived_sources": True,
    }


def check_pdf():
    pdf = BUILD / "supplementary_methods_v3.pdf"
    texts, pages = [], []
    with pdfplumber.open(pdf) as document:
        for number, page in enumerate(document.pages, 1):
            text = page.extract_text(x_tolerance=1) or ""
            assert "??" not in text and len(text) > 250, (number, text[:100])
            for char in page.chars:
                if char["text"].strip():
                    assert char["x0"] >= 18 and char["x1"] <= page.width - 18, number
                    assert char["top"] >= 16 and char["bottom"] <= page.height - 16, number
            texts.append(text)
            pages.append({"page": number, "characters": len(text), "within_bounds": True})
    assert len(pages) > 11
    compact = re.sub(r"\s+", "", "\n".join(texts))
    for count, name in ((11, "Table"), (5, "Figure"), (1, "Algorithm")):
        for number in range(1, count + 1):
            assert f"{name}S{number}" in compact, (name, number)
    log = (BUILD / "supplementary_methods_v3.log").read_text(encoding="utf-8", errors="replace")
    errors = re.findall(
        r"(?im)^.*(?:Overfull|undefined references|undefined on input|Undefined control|"
        r"Fatal error|Label\(s\) may have changed|referenced but does not exist|"
        r"Float too large|Too many unprocessed floats).*$", log)
    assert not errors, errors
    return {"page_count": len(pages), "pages": pages, "layout_or_reference_errors": errors,
            "sha256": prior.sha256(pdf), "underfull_messages": log.count("Underfull")}


def verify():
    snapshot_record = prior.load_json(ASSETS / "protected_snapshot.json")
    terminology_record = prior.load_json(HERE / "codec_terminology_verification.json")
    authorized_changes = set(terminology_record["authorized_changes_to_prior_snapshot"])
    changed_protected = {
        relative for relative, expected in snapshot_record["protected"].items()
        if prior.sha256(ROOT / relative) != expected
    }
    assert changed_protected == authorized_changes, (
        "Protected-file changes differ from the documented September terminology revision.",
        changed_protected ^ authorized_changes,
    )
    for document in terminology_record["documents"]:
        path = Path(document["file"])
        if path.name in {"results_v3.pdf", "manuscript.pdf"}:
            assert prior.sha256(path) == document["sha256"], path
    for path, expected in snapshot_record["sources"].items():
        assert prior.sha256(Path(path)) == expected, path
    shapes = prior.load_json(ASSETS / "shape_checks.json")
    assert shapes["status"] == "passed" and shapes["encoder_parameter_count"] == 8873360
    report = {
        "status": "passed", "checked_on": "2026-10-08",
        "records": prior.check_records(), "text": check_text(), "diagrams": check_diagrams(),
        "data_figures": check_data_figures(),
        "pdf": check_pdf(), "protected_original_count": len(snapshot_record["protected"]),
        "documented_prior_terminology_changes": sorted(changed_protected),
        "source_record_count": len(snapshot_record["sources"]),
        "new_experiments_run": False, "original_results_reestimated": False,
        "synthetic_shape_checks": shapes,
    }
    prior.REVIEW = REVIEW
    report["contact_sheets"] = prior.contact_sheets(report["pdf"]["page_count"])
    if DELIVERY.exists():
        assert prior.sha256(DELIVERY) == report["pdf"]["sha256"]
        report["delivery"] = {"path": str(DELIVERY), "matches_build": True}
    prior.write_json(ASSETS / "verification.json", report)
    print(json.dumps({name: report[name] for name in
                      ("status", "text", "diagrams", "protected_original_count", "source_record_count")},
                     ensure_ascii=False, indent=2))
    print(json.dumps({"pdf_pages": report["pdf"]["page_count"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", action="store_true")
    options = parser.parse_args()
    snapshot() if options.snapshot else verify()
