"""Check the standalone Results against saved measurements and rendered pages."""

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import pdfplumber
from PIL import Image, ImageDraw, ImageOps


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / "results_v2_assets"
REVIEW = ROOT / "tmp" / "results_v2_review"
SPEAKER = Path("D:/tmp/speechtokenizer-exp23/tmp/exp23_formal")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_close(actual, expected, message):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError(f"{message}: {actual} != {expected}")


def prose_count(text):
    text = re.sub(r"(?m)^\s*%.*$", "", text)
    text = re.sub(
        r"\\begin\{(figure\*?|table\*?|equation\*?|align\*?)\}.*?\\end\{\1\}",
        " ", text, flags=re.S,
    )
    text = re.sub(r"\$[^$]*\$", " ", text, flags=re.S)
    text = re.sub(r"\\(?:label|ref|cite|input|section|subsection)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z]+\*?", " ", text)
    return len(re.findall(r"\b[A-Za-z]+(?:[-'][A-Za-z]+)*\b", text))


def section_counts(text):
    parts = re.split(r"\\subsection\{([^}]+)\}", text)
    rows = [{"section": "Opening", "words": prose_count(parts[0])}]
    for i in range(1, len(parts), 2):
        rows.append({"section": parts[i], "words": prose_count(parts[i + 1])})
    return rows


def verify_chinese_tables(evidence, chinese):
    blocks = []
    for block in re.findall(r"(?m)(?:^\|.*\|\n?)+", chinese):
        rows = [
            [cell.strip() for cell in line.strip("|").split("|")]
            for line in block.strip().splitlines()
        ]
        blocks.append(rows[2:])
    assert [len(rows) for rows in blocks] == [5, 5, 4]

    def numbers(cell):
        return [float(value) for value in re.findall(r"\d+(?:\.\d+)?", cell)]

    for row, values in zip(blocks[0], evidence["quality"]):
        for i, metric in enumerate(("mel_l1", "stoi", "pesq_wb"), 2):
            expected = [
                round(float(values[f"{metric}_{key}"]), 3)
                for key in ("mean", "ci_low", "ci_high")
            ]
            assert numbers(row[i]) == expected, ("Chinese quality row", row)
    for row, values in zip(blocks[1], evidence["robustness"]):
        expected = [
            round(values[model][metric], 4)
            for metric in ("mel_l1", "stoi", "pesq_wb")
            for model in ("base", "lca")
        ]
        assert [numbers(cell)[0] for cell in row[1:]] == expected
    metrics = evidence["speaker"]["test_metrics"]
    intervals = evidence["speaker"]["bootstrap"]["bootstrap_by_seed"]
    keys = ("top1", "macro_f1", "top5", "ece_15bin", "worst_decile_accuracy")
    for row, seed, interval in zip(blocks[2][:3], ("41", "42", "43"), intervals):
        assert [numbers(cell)[0] for cell in row[1:6]] == [
            round(100 * metrics[seed][key], 2) for key in keys
        ]
        assert numbers(row[6]) == [
            round(100 * interval[key], 2) for key in ("ci_low", "ci_high")
        ]
    for cell, key in zip(blocks[2][3][1:6], keys):
        values = [100 * metrics[seed][key] for seed in ("41", "42", "43")]
        expected = [round(statistics.mean(values), 2)]
        if key in ("top1", "macro_f1"):
            expected.append(round(statistics.stdev(values), 2))
        assert numbers(cell) == expected
    return {"table_count": 3, "all_numeric_cells_match_english_sources": True}


def verify_quality(evidence, chinese):
    path = ROOT / (
        "output/experiments/exp12_baseline_comparison_test300_20260610_seed42/"
        "runs/test-clean_300/metrics/audio_quality_results.csv"
    )
    rows = read_csv(path)
    sample_sets = []
    for summary in evidence["quality"]:
        matched = [
            row for row in rows
            if row["method"] == summary["method"]
            and row["codec_setting"] == summary["codec_setting"]
        ]
        samples = {row["sample_id"] for row in matched}
        assert len(matched) == len(samples) == 300
        sample_sets.append(samples)
        for metric in ("mel_l1", "stoi", "pesq_wb"):
            value = statistics.mean(float(row[metric]) for row in matched)
            check_close(value, float(summary[f"{metric}_mean"]), f"Quality {metric}")
            for key in ("mean", "ci_low", "ci_high"):
                assert f"{float(summary[f'{metric}_{key}']):.3f}" in chinese
    assert all(samples == sample_sets[0] for samples in sample_sets)
    return {
        "raw_means_match_summary": True,
        "same_300_utterances_across_five_codecs": True,
        "raw_source": str(path),
        "raw_source_sha256": sha256(path),
    }


def verify_speaker():
    results = []
    sources = {}
    for seed in (41, 42, 43):
        prediction_path = SPEAKER / f"seed{seed}/metrics/test_predictions.csv"
        metric_path = SPEAKER / f"seed{seed}/metrics/test_metrics.json"
        rows = read_csv(prediction_path)
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        assert len(rows) == 1650
        assert len({row["utterance_group"] for row in rows}) == 1650
        true_counts = Counter(int(row["label"]) for row in rows)
        predicted_counts = Counter(int(row["prediction"]) for row in rows)
        correct = Counter()
        for row in rows:
            observed = int(row["label"]) == int(row["prediction"])
            assert int(row["correct"]) == observed
            correct[int(row["label"])] += observed
        assert len(true_counts) == 110 and set(true_counts.values()) == {15}
        top1 = sum(correct.values()) / len(rows)
        macro_f1 = statistics.mean(
            2 * correct[label] / (true_counts[label] + predicted_counts[label])
            for label in true_counts
        )
        per_speaker = {label: correct[label] / true_counts[label] for label in true_counts}
        worst_decile = statistics.mean(sorted(per_speaker.values())[:11])
        for key, value in (("top1", top1), ("macro_f1", macro_f1),
                           ("worst_decile_accuracy", worst_decile)):
            check_close(value, metrics[key], f"Seed {seed}: {key}")
        for label, value in per_speaker.items():
            check_close(value, metrics["per_speaker_accuracy"][str(label)],
                        f"Seed {seed}: speaker {label}")
        results.append({
            "seed": seed, "top1": top1, "macro_f1": macro_f1,
            "worst_decile_accuracy": worst_decile, "test_items": len(rows),
        })
        for path in (prediction_path, metric_path, SPEAKER / f"seed{seed}/training_config.json"):
            sources[str(path)] = sha256(path)
    state_path = SPEAKER / "cache_lca_l3/cache_manifest/tokenizer_state_hashes.json"
    states = json.loads(state_path.read_text(encoding="utf-8"))
    assert all(row["before_sha256"] == row["after_sha256"] for row in states)
    for path in (state_path, SPEAKER / "aggregate/bootstrap_summary.json"):
        sources[str(path)] = sha256(path)
    return {
        "recomputed_from_locked_predictions": results,
        "mean_top1": statistics.mean(row["top1"] for row in results),
        "sample_std_top1": statistics.stdev(row["top1"] for row in results),
        "codec_cache_hashes_unchanged": True,
        "bootstrap_intervals_not_reestimated": True,
        "top5_and_ece_retained_from_saved_metrics": True,
        "sources_sha256": sources,
    }


def verify_pdf():
    pages = []
    with pdfplumber.open(HERE / "results_v2.pdf") as document:
        assert 5 <= len(document.pages) <= 6
        all_text = ""
        for number, page in enumerate(document.pages, 1):
            text = page.extract_text(x_tolerance=1) or ""
            all_text += text
            assert "??" not in text
            assert len(text) > 1200, f"Potentially isolated float/empty page {number}"
            outside = [
                char for char in page.chars
                if char["text"].strip()
                and (char["x0"] < 25 or char["x1"] > page.width - 25
                     or char["top"] < 20 or char["bottom"] > page.height - 20)
            ]
            assert not outside, f"Text outside safe page bounds on page {number}"
            body_chars = [char for char in page.chars if 9.8 < char["size"] < 10.2]
            left = sum(char["x0"] < page.width / 2 for char in body_chars)
            right = sum(char["x0"] >= page.width / 2 for char in body_chars)
            assert left > 100 and right > 100, f"Missing text column on page {number}"
            pages.append({
                "page": number, "text_characters": len(text),
                "left_column_body_characters": left,
                "right_column_body_characters": right,
                "text_within_page_bounds": True,
            })
        compact = re.sub(r"\s", "", all_text)
        for label in ("Figure1:", "Figure2:", "Table1:", "Table2:", "Table3:"):
            assert label in compact, f"Missing rendered label {label}"
        diagram_page = document.pages[1]
        words = diagram_page.extract_words(x_tolerance=1)
        caption = next(word for word in words if word["text"].startswith("Figure"))
        diagram_image = Image.open(REVIEW / "page-2.png").convert("RGB")
        scale = diagram_image.height / diagram_page.height
        bounds = (32 * scale, 78 * scale,
                  (diagram_page.width - 32) * scale, (caption["top"] - 7) * scale)
        diagram_image.crop(tuple(round(value) for value in bounds)).save(
            ASSETS / "components.png"
        )
    contact = Image.new("RGB", (1200, 1740), "#eeeeee")
    draw = ImageDraw.Draw(contact)
    for i in range(len(pages)):
        image = Image.open(REVIEW / f"page-{i + 1}.png").convert("RGB")
        x, y = (i % 3) * 400, (i // 3) * 870
        draw.text((x + 12, y + 12), f"Page {i + 1}", fill="black")
        thumbnail = ImageOps.contain(image, (380, 820))
        contact.paste(thumbnail, (x + 10, y + 35))
    contact.save(REVIEW / "contact.png")
    return {"page_count": len(pages), "pages": pages}


def main():
    evidence = json.loads((ASSETS / "evidence.json").read_text(encoding="utf-8"))
    body = (HERE / "results_v2_body.tex").read_text(encoding="utf-8")
    wrapper = (HERE / "results_v2.tex").read_text(encoding="utf-8")
    chinese = (HERE / "results_v2_zh.md").read_text(encoding="utf-8")
    original = (HERE / "manuscript.tex").read_text(encoding="utf-8")
    old_results = original.split(r"\section{Results}", 1)[1].split(
        r"\section{Discussion}", 1
    )[0]
    original_hashes = {}
    for relative, expected in evidence["protected_original_hashes"].items():
        actual = sha256(ROOT / relative)
        assert actual == expected, f"Protected original changed: {relative}"
        original_hashes[relative] = actual
    for path, expected in evidence["sources"].items():
        assert sha256(Path(path)) == expected, f"Source changed: {path}"
    tables = "\n".join(
        (ASSETS / name).read_text(encoding="utf-8")
        for name in ("quality_table.tex", "robustness_table.tex", "speaker_table.tex")
    )
    for result in evidence["robustness"]:
        for model in ("base", "lca"):
            for value in result[model].values():
                assert f"{value:.4f}" in chinese
    assert body.count(r"\subsection{") == 5
    assert body.count(r"\begin{figure*}") == 2
    assert tables.count(r"\begin{table*}") == 3
    assert body.count(r"\begin{equation}") == 2
    assert len(re.findall(r"(?m)^### 2\.[1-5] ", chinese)) == 5
    assert not any(command in body for command in (r"\newpage", r"\clearpage", r"\vspace"))
    defined = set(re.findall(r"\\label\{([^}]+)\}", body + tables))
    references = set(re.findall(r"\\ref\{([^}]+)\}", body + tables))
    assert references <= defined
    cited = set(re.findall(r"\\cite\{([^}]+)\}", body))
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", original))
    standalone_cites = set(re.findall(r"\\bibcite\{([^}]+)\}", wrapper))
    assert cited <= bibliography and cited <= standalone_cites
    log = (HERE / "results_v2.log").read_text(encoding="utf-8", errors="replace")
    bad = re.findall(
        r"(?im)^.*(?:Overfull|undefined references|undefined on input|"
        r"Undefined control|Fatal error|Label\(s\) may have changed|"
        r"referenced but does not exist).*$", log,
    )
    assert not bad, bad
    report = {
        "status": "passed",
        "prose_word_count_convention": (
            "Alphabetic English word groups; hyphenated words count as one; "
            "exclude headings, tables, figure captions, displayed and inline math, "
            "reference commands and numeric-only tokens. Approximate editorial count."
        ),
        "before_prose_words": prose_count(old_results),
        "after_prose_words": prose_count(body),
        "before_sections": section_counts(old_results),
        "after_sections": section_counts(body),
        "figures": 2, "tables": 3, "displayed_equations": 2, "subsections": 5,
        "quality": verify_quality(evidence, chinese),
        "speaker": verify_speaker(),
        "chinese_tables": verify_chinese_tables(evidence, chinese),
        "pdf": verify_pdf(),
        "protected_originals_unchanged": original_hashes,
        "chinese_numeric_checks_passed": True,
        "unresolved_references_or_overfull_boxes": bad,
        "underfull_box_messages": len(re.findall(r"Underfull", log)),
        "pdf_sha256": sha256(HERE / "results_v2.pdf"),
    }
    (ASSETS / "verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"],
        "pages": report["pdf"]["page_count"],
        "before_words": report["before_prose_words"],
        "after_words": report["after_prose_words"],
        "after_sections": report["after_sections"],
        "preserved_original_files": len(original_hashes),
        "quality_means_and_speaker_predictions": "verified",
    }, indent=2))


if __name__ == "__main__":
    main()
