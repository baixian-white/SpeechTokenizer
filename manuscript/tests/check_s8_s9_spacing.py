"""Catch large blank regions around S8 figures and the S9 crosswalk table."""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


def pdf_lines(pdf: Path) -> list[list[tuple[float, float, str]]]:
    raw = subprocess.run(
        ["pdftotext", "-bbox-layout", str(pdf), "-"],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
    root = ET.fromstring(raw)
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    pages = []
    for page in root.findall(".//x:page", ns):
        lines = []
        for line in page.findall(".//x:line", ns):
            words = [word.text or "" for word in line.findall("x:word", ns)]
            lines.append(
                (
                    float(line.attrib["xMin"]),
                    float(line.attrib["yMin"]),
                    " ".join(words),
                )
            )
        pages.append(lines)
    return pages


def check_s8_page_16(pdf: Path) -> None:
    # This is a visual check, so figures count as useful page content too.
    raster = subprocess.run(
        [
            "pdftoppm",
            "-f",
            "16",
            "-l",
            "16",
            "-gray",
            "-r",
            "72",
            "-singlefile",
            str(pdf),
        ],
        check=True,
        capture_output=True,
    ).stdout
    with Image.open(io.BytesIO(raster)) as image:
        region = image.crop((320, 260, 550, 700))
        ink = sum(pixel < 220 for pixel in region.get_flattened_data())
        density = ink / (region.width * region.height)
    print(f"S8 page 16 lower-right ink: {density:.4%} (minimum 1.00%)")
    if density < 0.01:
        raise AssertionError("S8 leaves most of page 16's right column empty")


def check_s9_crosswalk(pages: list[list[tuple[float, float, str]]]) -> None:
    headings = [
        (index, line)
        for index, lines in enumerate(pages)
        for line in lines
        if line[2].startswith("S9. Experimental correspondence")
    ]
    captions = [
        (index, line)
        for index, lines in enumerate(pages)
        for line in lines
        if line[2].startswith("Table S11:")
    ]
    if len(headings) != 1 or len(captions) != 1:
        raise AssertionError("Expected one S9 heading and one Table S11 caption")
    section_page, heading = headings[0]
    table_page, caption = captions[0]
    if section_page == table_page and heading[0] < 300 and caption[0] < 300:
        body_before_table = [
            line
            for line in pages[section_page]
            if line[0] < 290
            and heading[1] < line[1] < caption[1]
            and not re.fullmatch(r"\d+", line[2])
        ]
        if not body_before_table:
            raise AssertionError("No S9 body text before Table S11")
        gap = caption[1] - max(line[1] for line in body_before_table)
        print(f"S9 body to Table S11: {gap:.1f} pt (maximum 85 pt)")
        if gap > 85:
            raise AssertionError("Table S11 is separated from S9 text by a large blank")
    else:
        # A table on the next page is acceptable only if the heading page
        # does not abandon its left column near the top.
        last_left = max(
            line[1] for line in pages[section_page] if line[0] < 290
        )
        print(
            f"S9 heading page {section_page + 1} left-column bottom: "
            f"{last_left:.1f} pt (minimum 550 pt)"
        )
        if heading[1] < 250 and last_left < 550:
            raise AssertionError("S9 leaves most of its opening column empty")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "supplementary_methods_v3.pdf",
    )
    args = parser.parse_args()
    pdf = args.pdf.resolve()
    pages = pdf_lines(pdf)
    failures = []
    for name, check in (
        ("S8", lambda: check_s8_page_16(pdf)),
        ("S9", lambda: check_s9_crosswalk(pages)),
    ):
        try:
            check()
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("S8/S9 spacing checks passed")


if __name__ == "__main__":
    main()
