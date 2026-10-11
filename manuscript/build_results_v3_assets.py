"""Build the v3 overview and regenerate numeric displays from archived evidence."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

import build_results_v2_assets as source


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results_v3_assets"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_protected():
    paths = [
        HERE / "manuscript.tex", HERE / "manuscript.pdf",
        HERE / "results_zh.md", HERE / "methods_main_zh.md",
        HERE / "supplementary_methods_body.tex",
    ]
    paths += sorted(HERE.glob("results_v2*"))
    paths += sorted((HERE / "results_v2_assets").glob("*"))
    return {str(path.relative_to(ROOT)): digest(path)
            for path in paths if path.is_file()}


def build_components():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.15, 4.65))
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.018, top=0.988)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    ink, grey = "#20262B", "#69737B"
    blue, green, orange = "#DDECF4", "#E1F0E8", "#F8EAD9"

    def text(x, y, value, size=8, weight="normal", ha="center", color=ink):
        return ax.text(x, y, value, ha=ha, va="center", fontsize=size,
                       weight=weight, color=color, linespacing=1.35)

    def box(x, y, width, height, label, fill="white", size=8):
        ax.add_patch(Rectangle((x, y - height / 2), width, height,
                               facecolor=fill, edgecolor="#89939A", linewidth=0.65))
        text(x + width / 2, y, label, size)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
            color=grey, linewidth=0.85, shrinkA=1, shrinkB=1,
        ))

    text(0, 97, "a  Shared index interface", 10, "bold", "left")
    boxes = [
        (0, 13, "Speech\n16 kHz", "white"),
        (17, 18, "NAS encoder\n1024-D, 50 Hz", blue),
        (39, 18, "RVQ\n3 codebooks", blue),
        (61, 16, "Indices\n10 bits each", orange),
        (81, 19, "Local lookup\n+ decoder", green),
    ]
    for x, width, label, color in boxes:
        box(x, 83, width, 13, label, color)
    for (x, width, *_), (next_x, *_) in zip(boxes, boxes[1:]):
        arrow(x + width, 83, next_x, 83)
    text(48, 72, "1024 entries / codebook", 7.5, color=grey)
    text(69, 72, "1.5 kbit/s", 8, "bold")
    text(90.5, 72, "Reconstructed speech", 7.5, color=grey)
    ax.plot([0, 100], [67.5, 67.5], color="#CED4D8", linewidth=0.55)

    text(0, 62, "b  Successive learning stages", 10, "bold", "left")
    stages = [
        (0, 22, "Teacher-aligned\narchitecture search", blue),
        (26, 22, "Joint codec training\nSCIT", blue),
        (52, 22, "Channel adaptation\nSCIT-LCA", orange),
        (78, 22, "Freeze codec\nTrain identification", green),
    ]
    for x, width, label, color in stages:
        box(x, 50, width, 12, label, color)
    for (x, width, *_), (next_x, *_) in zip(stages, stages[1:]):
        arrow(x + width, 50, next_x, 50)
    text(50, 39.8, "Adaptation: perturbed-speech reconstruction + clean / perturbed consistency",
         7.8, color=grey)
    ax.plot([0, 100], [35.3, 35.3], color="#CED4D8", linewidth=0.55)

    text(0, 30.4, "c  Identification from the frozen representation", 10, "bold", "left")
    box(0, 19, 19, 9.5, "Received indices", orange, 7.8)
    box(0, 5.5, 19, 10, "Reconstructed\nspeech", green)
    box(25, 19, 29, 9.5, "Token encoder (trainable)", green, 7.8)
    box(25, 5.5, 29, 10, "ECAPA-TDNN (frozen)\nAudio projection (trainable)", green, 7.6)
    arrow(19, 19, 25, 19)
    arrow(19, 5.5, 25, 5.5)
    ax.plot([54, 62, 62], [19, 19, 12.25], color=grey, linewidth=0.85)
    ax.plot([54, 62, 62], [5.5, 5.5, 12.25], color=grey, linewidth=0.85)
    arrow(62, 12.25, 67, 12.25)
    box(67, 12.25, 13, 11, "Gated\nfusion", green)
    arrow(80, 12.25, 84, 12.25)
    box(84, 12.25, 16, 11, "110-speaker\nclassifier", green, 7.8)
    text(58, 23, "256-D", 7.2, color=grey)
    text(58, 1.8, "256-D", 7.2, color=grey)
    fig.savefig(OUT / "components.pdf")
    fig.savefig(OUT / "components.png", dpi=240)
    plt.close(fig)


def adapt_tables():
    for name in ("quality_table.tex", "robustness_table.tex", "speaker_table.tex"):
        path = OUT / name
        text = path.read_text(encoding="utf-8").replace("tab:rv2-", "tab:rv3-")
        if name == "robustness_table.tex":
            text = text.replace(
                "The burst condition removes one randomly placed span of ten latent frames "
                "per utterance; it is not a fixed percentage of packet loss. These "
                "token-domain perturbations do not emulate a complete packet-loss "
                "concealment receiver.",
                "The burst condition masks one randomly placed span of ten latent frames "
                "per utterance. All conditions use the token-domain protocol in "
                "Supplementary Methods S5.",
            )
        if name == "speaker_table.tex":
            text = text.replace(
                "The last column reports within-seed hierarchical-bootstrap intervals, "
                "not uncertainty on the three-seed mean.",
                "The last column reports within-seed hierarchical-bootstrap intervals.",
            ).replace("42 (delivery)", "42").replace(
                "Seed 42 was predeclared for delivery. Intervals are retained from the "
                "archived analysis; its bootstrap replicate count and RNG seed remain "
                "an author-confirmation item in Methods and Supplementary Methods S7.",
                "The intervals are within-run hierarchical-bootstrap estimates and do "
                "not represent uncertainty on the three-run mean.",
            )
        path.write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    snapshot = OUT / "protected_snapshot.json"
    if not snapshot.exists():
        snapshot.write_text(json.dumps(capture_protected(), indent=2) + "\n", encoding="utf-8")
    protected = json.loads(snapshot.read_text(encoding="utf-8"))
    for relative, expected in protected.items():
        assert digest(ROOT / relative) == expected, relative
    source.OUT = OUT
    evidence = {
        "generated_from_archived_records": True,
        "nas": source.build_nas(),
        "quality": source.build_quality(),
        "robustness": source.build_robustness(),
        "speaker": source.build_speaker(),
        "sources": {str(path): digest(path) for path in
                    (source.NAS, source.QUALITY, source.INDEX, source.PACKET)},
        "protected_original_hashes": protected,
    }
    adapt_tables()
    build_components()
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "built", "figures": 2, "tables": 3,
        "protected_files": len(protected), "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
