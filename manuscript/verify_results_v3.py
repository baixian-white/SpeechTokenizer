"""Verify v3 wording changes, source data, translation tables and PDF layout."""

import json
import re
from pathlib import Path

import pdfplumber
from PIL import Image, ImageDraw, ImageOps

from verify_results_v2 import (
    prose_count,
    section_counts,
    sha256,
    verify_chinese_tables,
    verify_quality,
    verify_speaker,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = HERE / "results_v3_assets"
REVIEW = ROOT / "tmp/pdfs/results_v3_review"


def table_rows(text):
    return text.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0].strip()


def verify_pdf():
    pages = []
    with pdfplumber.open(HERE / "results_v3.pdf") as document:
        assert 5 <= len(document.pages) <= 6, len(document.pages)
        all_text = ""
        for number, page in enumerate(document.pages, 1):
            text = page.extract_text(x_tolerance=1) or ""
            all_text += text
            assert "??" not in text
            assert len(text) > 1000, f"Sparse page {number}"
            outside = [
                char for char in page.chars
                if char["text"].strip()
                and (char["x0"] < 20 or char["x1"] > page.width - 20
                     or char["top"] < 18 or char["bottom"] > page.height - 18)
            ]
            assert not outside, f"Text outside page bounds on page {number}"
            body_chars = [char for char in page.chars if 9.8 < char["size"] < 10.2]
            left = sum(char["x0"] < page.width / 2 for char in body_chars)
            right = sum(char["x0"] >= page.width / 2 for char in body_chars)
            assert left > 100 and right > 100, (number, left, right)
            pages.append({
                "page": number, "text_characters": len(text),
                "left_column_body_characters": left,
                "right_column_body_characters": right,
                "text_within_page_bounds": True,
            })
        compact = re.sub(r"\s", "", all_text)
        for label in ("Figure1:", "Figure2:", "Table1:", "Table2:", "Table3:"):
            assert label in compact, label
    contact = Image.new("RGB", (1500, 1450), "#eeeeee")
    draw = ImageDraw.Draw(contact)
    for i in range(len(pages)):
        image = Image.open(REVIEW / f"page-{i + 1}.png").convert("RGB")
        x, y = (i % 3) * 500, (i // 3) * 725
        draw.text((x + 12, y + 10), f"Page {i + 1}", fill="black")
        thumbnail = ImageOps.contain(image, (480, 680))
        contact.paste(thumbnail, (x + 10, y + 30))
    contact.save(REVIEW / "contact.png")
    return {"page_count": len(pages), "pages": pages}


def main():
    evidence = json.loads((ASSETS / "evidence.json").read_text(encoding="utf-8"))
    body = (HERE / "results_v3_body.tex").read_text(encoding="utf-8")
    wrapper = (HERE / "results_v3.tex").read_text(encoding="utf-8")
    before = (HERE / "results_v2_body.tex").read_text(encoding="utf-8")
    chinese = (HERE / "results_v3_zh.md").read_text(encoding="utf-8")
    original = (HERE / "manuscript.tex").read_text(encoding="utf-8")
    for relative, expected in evidence["protected_original_hashes"].items():
        assert sha256(ROOT / relative) == expected, f"Protected file changed: {relative}"
    for path, expected in evidence["sources"].items():
        assert sha256(Path(path)) == expected, f"Source changed: {path}"
    old_evidence = json.loads(
        (HERE / "results_v2_assets/evidence.json").read_text(encoding="utf-8")
    )
    for key in ("nas", "quality", "robustness", "speaker"):
        assert evidence[key] == old_evidence[key], f"Evidence changed: {key}"
    tables = []
    for name in ("quality_table.tex", "robustness_table.tex", "speaker_table.tex"):
        new = (ASSETS / name).read_text(encoding="utf-8")
        old = (HERE / "results_v2_assets" / name).read_text(encoding="utf-8")
        assert table_rows(new) == table_rows(old), name
        tables.append(new)
    displays = "\n".join(tables)
    assert body.count(r"\subsection{") == 5
    assert body.count(r"\begin{figure*}") == 2
    assert displays.count(r"\begin{table*}") == 3
    assert body.count(r"\begin{equation}") == 3
    assert len(re.findall(r"(?m)^### 2\.[1-5] ", chinese)) == 5
    assert not any(cmd in body for cmd in (r"\newpage", r"\clearpage", r"\vspace"))
    defined = set(re.findall(r"\\label\{([^}]+)\}", body + displays))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", body + displays))
    assert refs <= defined
    cites = set(re.findall(r"\\cite\{([^}]+)\}", body))
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", original))
    standalone = set(re.findall(r"\\bibcite\{([^}]+)\}", wrapper))
    assert cites <= bibliography and cites <= standalone
    assert not any(phrase in body for phrase in (
        "could not establish final reconstruction quality",
        "RTF is not end-to-end latency",
        "did not complete token-only",
        "transcript-controlled evaluation",
        "branch-specific mechanisms unresolved",
    ))
    assert "closed-set" in body
    assert "110 enrolled identities" in body
    assert "1.5~kbit/s" in body and "$L=3$" in body
    assert "PESQ-WB" in body and "decreased by 0.0141" in body
    assert "77.78" in body and "frozen" in body
    log = (HERE / "results_v3.log").read_text(encoding="utf-8", errors="replace")
    bad = re.findall(
        r"(?im)^.*(?:Overfull|undefined references|undefined on input|"
        r"Undefined control|Fatal error|Label\(s\) may have changed|"
        r"referenced but does not exist).*$", log,
    )
    assert not bad, bad
    report = {
        "status": "passed",
        "before_prose_words": prose_count(before),
        "after_prose_words": prose_count(body),
        "word_count_convention": (
            "Approximate English prose count; alphabetic groups, hyphenated words "
            "count as one; exclude headings, displays, captions, math and numeric tokens."
        ),
        "before_sections": section_counts(before),
        "after_sections": section_counts(body),
        "figures": 2, "tables": 3, "displayed_equations": 3, "subsections": 5,
        "archived_evidence_identical_to_v2": True,
        "all_table_rows_identical_to_v2": True,
        "requested_concluding_caveats_removed": True,
        "quality": verify_quality(evidence, chinese),
        "speaker": verify_speaker(),
        "chinese_tables": verify_chinese_tables(evidence, chinese),
        "pdf": verify_pdf(),
        "protected_originals_unchanged": evidence["protected_original_hashes"],
        "unresolved_references_or_overfull_boxes": bad,
        "underfull_box_messages": len(re.findall(r"Underfull", log)),
        "pdf_sha256": sha256(HERE / "results_v3.pdf"),
    }
    (ASSETS / "verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": report["status"], "pages": report["pdf"]["page_count"],
        "before_words": report["before_prose_words"],
        "after_words": report["after_prose_words"],
        "after_sections": report["after_sections"],
        "protected_files": len(report["protected_originals_unchanged"]),
        "tables_and_translation": "verified",
    }, indent=2))


if __name__ == "__main__":
    main()
