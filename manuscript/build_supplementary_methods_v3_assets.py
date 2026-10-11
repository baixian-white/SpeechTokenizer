"""Generate exact, vector-based protocol diagrams from the retained definitions."""

import json
import math
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "supplementary_methods_v3_assets"
INK = "#202629"
MUTED = "#556068"
LINE = "#aab4ba"
TEAL = "#007c75"
BLUE = "#0072b2"
ORANGE = "#c66b00"
PALE = "#f2f5f6"
CHECKS = []


class Diagram:
    def __init__(self, name, width, height):
        self.name, self.width, self.height = name, width, height
        self.pdf = canvas.Canvas(str(ASSETS / f"{name}.pdf"), pagesize=(width, height))
        self.pdf.setTitle(name.replace("_", " "))
        self.pdf.setAuthor("")
        self.text_checks = []

    def text(self, x, y, value, size=8, bold=False, color=INK, align="left", limit=None):
        font = "Helvetica-Bold" if bold else "Helvetica"
        width = stringWidth(value, font, size)
        left = x - width / 2 if align == "center" else x - width if align == "right" else x
        assert left >= 0 and left + width <= self.width, (self.name, value, left, width)
        assert 0 <= y <= self.height - size, (self.name, value, y)
        if limit is not None:
            assert width <= limit, (self.name, value, width, limit)
        self.pdf.setFont(font, size)
        self.pdf.setFillColor(HexColor(color))
        self.pdf.drawString(left, y, value)
        self.text_checks.append({"text": value, "width": width, "limit": limit})

    def box(self, x, y, width, height, lines, accent=TEAL, size=8):
        self.pdf.setFillColor(HexColor(PALE))
        self.pdf.setStrokeColor(HexColor(LINE))
        self.pdf.setLineWidth(0.55)
        self.pdf.rect(x, y, width, height, fill=1, stroke=1)
        self.pdf.setFillColor(HexColor(accent))
        self.pdf.rect(x, y, 2, height, fill=1, stroke=0)
        line_height = size + 3
        baseline = y + height / 2 + (len(lines) - 1) * line_height / 2 - size / 3
        for index, line in enumerate(lines):
            self.text(x + width / 2, baseline - index * line_height, line, size,
                      bold=index == 0, align="center", limit=width - 10)

    def arrow(self, points, color=MUTED):
        self.pdf.setStrokeColor(HexColor(color))
        self.pdf.setFillColor(HexColor(color))
        self.pdf.setLineWidth(0.8)
        path = self.pdf.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        self.pdf.drawPath(path)
        x, y = points[-1]
        previous_x, previous_y = points[-2]
        angle = math.atan2(y - previous_y, x - previous_x)
        triangle = self.pdf.beginPath()
        triangle.moveTo(x, y)
        for offset in (-0.45, 0.45):
            triangle.lineTo(x - 4.8 * math.cos(angle + offset),
                            y - 4.8 * math.sin(angle + offset))
        triangle.close()
        self.pdf.drawPath(triangle, fill=1, stroke=0)

    def finish(self):
        self.pdf.showPage()
        self.pdf.save()
        CHECKS.append({"figure": self.name, "text_bounds_checked": True,
                       "labels": self.text_checks})


def perturbations():
    diagram = Diagram("index_perturbations", 510, 244)
    diagram.text(5, 231, "a", size=10, bold=True)
    diagram.text(21, 231, "Index values and replacement rule", size=9, bold=True)
    diagram.text(366, 231, "Illustrative indices, not test data", size=7, color=MUTED)
    values = [13, 44, 72, 31, 86, 9]
    erased = [13, 44, 44, 72, 86, 9]
    substituted = [13, 801, 72, 31, 206, 9]
    rows = [
        ("Original", values, set(), TEAL),
        ("Erasure replacement", erased, {2, 3}, ORANGE),
        ("Random substitution", substituted, {1, 4}, BLUE),
    ]
    for row, (label, codes, affected, accent) in enumerate(rows):
        y = 195 - 25 * row
        diagram.text(5, y + 6, label, size=8)
        for index, value in enumerate(codes):
            x = 143 + index * 54
            diagram.pdf.setFillColor(HexColor("#f8eddc" if index in affected else PALE))
            diagram.pdf.setStrokeColor(HexColor(accent if index in affected else LINE))
            diagram.pdf.rect(x, y, 44, 20, fill=1, stroke=1)
            diagram.text(x + 22, y + 6, str(value), size=8, align="center")
    diagram.text(143, 128, "Consecutive erasures read the preceding ORIGINAL values.", size=8)

    def grid(x, title, selected, panel):
        diagram.text(x, 104, panel, size=10, bold=True)
        diagram.text(x + 16, 104, title, size=8.5, bold=True)
        cell = 9.6
        origin = x + 34
        for layer in range(3):
            y = 78 - layer * 13
            diagram.text(x + 1, y + 3, f"L{layer + 1}", size=7.5)
            for frame in range(20):
                diagram.pdf.setFillColor(HexColor("#f8eddc" if frame in selected else PALE))
                diagram.pdf.setStrokeColor(HexColor(ORANGE if frame in selected else LINE))
                diagram.pdf.setLineWidth(0.45)
                diagram.pdf.rect(origin + frame * cell, y, cell - 1.4, 10, fill=1, stroke=1)
        for frame in (0, 5, 10, 15, 19):
            diagram.text(origin + frame * cell + 3.8, 37, str(frame), size=6.8, align="center")
        diagram.text(origin + 191, 24, "frame index", size=7, align="right", color=MUTED)

    grid(5, "Five-frame packet mask (100 ms)", set(range(5, 10)), "b")
    grid(269, "Ten-frame burst mask (200 ms)", set(range(7, 17)), "c")
    diagram.text(5, 6, "Shaded cells: selected positions. Grouped masks share the same time span across all three layers.",
                 size=7.5, color=MUTED)
    diagram.finish()
    return {
        "original": values, "erasure_mask": [False, False, True, True, False, False],
        "erased": erased, "substitution_mask": [False, True, False, False, True, False],
        "substituted": substituted, "packet_frames": list(range(5, 10)),
        "burst_frames": list(range(7, 17)), "frame_ms": 20,
    }


def receiver():
    diagram = Diagram("receiver_pairing", 510, 270)
    diagram.text(5, 256, "a", size=10, bold=True)
    diagram.text(21, 256, "One received block: index recovery and local decoding", size=9, bold=True)
    diagram.box(5, 198, 102, 44, ["Packed payload", "94 bytes / 0.5 s"], BLUE)
    diagram.box(132, 198, 102, 44, ["Unpack", "indices [3, 25]"], BLUE)
    diagram.box(260, 198, 113, 44, ["Lookup + decoder", "local model weights"])
    diagram.box(399, 198, 106, 44, ["Decoded waveform", "8,000 samples"])
    for start, end in ((107, 132), (234, 260), (373, 399)):
        diagram.arrow([(start + 2, 220), (end - 3, 220)])
    diagram.text(5, 182, "Array shape and index count remove padding; both outputs retain the same stream and interval.",
                 size=7.8, color=MUTED)

    diagram.text(5, 158, "b", size=10, bold=True)
    diagram.text(21, 158, "Paired crop from stream-specific buffers", size=9, bold=True)
    diagram.box(5, 100, 103, 43, ["Token crop", "[1, 3, T_c]"], BLUE)
    diagram.box(136, 100, 118, 43, ["Token encoder", "masked time pooling"], BLUE)
    diagram.box(279, 100, 100, 43, ["Token embedding", "[1, 256]"], BLUE)
    diagram.box(5, 33, 103, 43, ["Audio crop", "[1, N_c]"])
    diagram.box(136, 33, 118, 43, ["ECAPA + projection", "192 to 256"])
    diagram.box(279, 33, 100, 43, ["Audio embedding", "[1, 256]"])
    for y in (121.5, 54.5):
        diagram.arrow([(110, y), (133, y)])
        diagram.arrow([(256, y), (276, y)])
    diagram.box(418, 65, 87, 50, ["Fusion + head", "110 logits"], ORANGE)
    diagram.arrow([(381, 121.5), (404, 121.5), (404, 102), (415, 102)], BLUE)
    diagram.arrow([(381, 54.5), (404, 54.5), (404, 78), (415, 78)], TEAL)
    diagram.arrow([(461.5, 63), (461.5, 47)], ORANGE)
    diagram.text(461.5, 35, "Calibrated", size=7.8, align="center")
    diagram.text(461.5, 24, "probabilities", size=7.8, align="center")
    diagram.text(5, 7, "N_c: actual cropped audio length (up to 48,000). T_c: matched code length from the cache or buffer.",
                 size=7.5, color=MUTED)
    diagram.finish()


def main():
    ASSETS.mkdir(exist_ok=True)
    examples = perturbations()
    receiver()
    evidence = {
        "figure_type": "Method schematics, not additional experimental observations.",
        "perturbation_example": examples,
        "receiver": {
            "layers": 3, "nominal_frames_per_second": 50, "block_samples": 8000,
            "block_frames": 25, "informative_bits": 750, "packed_bytes": 94,
            "padding_bits": 2, "classification_max_samples": 48000,
            "code_crop_uses_actual_length": True,
        },
        "layout_checks": CHECKS,
    }
    (ASSETS / "diagram_checks.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": 2, "text_bounds_checked": True}))


if __name__ == "__main__":
    main()
