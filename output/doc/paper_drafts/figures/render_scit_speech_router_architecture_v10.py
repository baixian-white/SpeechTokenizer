from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
PNG_OUT = OUT_DIR / "scit_speech_three_user_router_architecture_v10_top_text_fixed_20260630.png"

W, H = 1920, 1080

BLUE = "#1f5ea8"
BLUE_DARK = "#143f7a"
BLUE_LIGHT = "#e9f3ff"
BLUE_FILL = "#bfe0ff"
BLUE_MODULE = "#2f80d5"
GREEN = "#188a3b"
GREEN_DARK = "#0f6e2d"
GREEN_LIGHT = "#dff4df"
GRAY = "#6a7178"
GRAY_LIGHT = "#f2f4f7"
GRAY_MID = "#d9dee5"
INK = "#1f2933"


def font(size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if bold and italic:
        candidates.extend(
            [
                r"C:\Windows\Fonts\arialbi.ttf",
                r"C:\Windows\Fonts\calibriz.ttf",
            ]
        )
    elif bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\arialbd.ttf",
                r"C:\Windows\Fonts\calibrib.ttf",
            ]
        )
    elif italic:
        candidates.extend(
            [
                r"C:\Windows\Fonts\ariali.ttf",
                r"C:\Windows\Fonts\calibrii.ttf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font(30, True)
F_SECTION = font(24, True)
F_LABEL = font(20, True)
F_LABEL_SMALL = font(17, True)
F_TEXT = font(18)
F_TEXT_SMALL = font(15)
F_TEXT_XSMALL = font(13)
F_TINY = font(13, True)
F_NOTE = font(26, True)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=4)
    return int(box[2] - box[0]), int(box[3] - box[1])


def center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str = INK,
    spacing: int = 4,
) -> None:
    x1, y1, x2, y2 = box
    tw, th = text_size(draw, text, fnt)
    draw.multiline_text(
        (x1 + (x2 - x1 - tw) / 2, y1 + (y2 - y1 - th) / 2),
        text,
        font=fnt,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def round_rect(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str,
    width: int = 2,
    radius: int = 14,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    p1: tuple[int, int],
    p2: tuple[int, int],
    fill: str,
    width: int = 2,
    dash: int = 12,
    gap: int = 8,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    while pos < length:
        end = min(pos + dash, length)
        draw.line(
            [(x1 + ux * pos, y1 + uy * pos), (x1 + ux * end, y1 + uy * end)],
            fill=fill,
            width=width,
        )
        pos += dash + gap


def polyline(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], fill: str, width: int) -> None:
    for a, b in zip(pts, pts[1:]):
        draw.line([a, b], fill=fill, width=width)


def arrowhead(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str, size: int = 18) -> None:
    sx, sy = start
    ex, ey = end
    angle = math.atan2(ey - sy, ex - sx)
    left = (ex - size * math.cos(angle - math.pi / 6), ey - size * math.sin(angle - math.pi / 6))
    right = (ex - size * math.cos(angle + math.pi / 6), ey - size * math.sin(angle + math.pi / 6))
    draw.polygon([end, left, right], fill=fill)


def arrow(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], fill: str, width: int = 5, size: int = 18) -> None:
    polyline(draw, pts, fill, width)
    arrowhead(draw, pts[-2], pts[-1], fill, size)


def double_arrow(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], fill: str, width: int = 4, size: int = 14) -> None:
    polyline(draw, pts, fill, width)
    arrowhead(draw, pts[-2], pts[-1], fill, size)
    arrowhead(draw, pts[1], pts[0], fill, size)


def vertical_double_arrow_segment(
    draw: ImageDraw.ImageDraw,
    x: int,
    y1: int,
    y2: int,
    fill: str,
    width: int = 4,
    size: int = 14,
) -> None:
    draw.line([(x, y1), (x, y2)], fill=fill, width=width)
    arrowhead(draw, (x, y1 + 20), (x, y1), fill, size)
    arrowhead(draw, (x, y2 - 20), (x, y2), fill, size)


def token_chip(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, w: int = 54) -> None:
    box = (x, y, x + w, y + 22)
    round_rect(draw, box, "#edf8ee", GREEN, width=1, radius=8)
    center_text(draw, box, label, font(12, True), GREEN_DARK)


def codebook_grid(draw: ImageDraw.ImageDraw, x: int, y: int, scale: int = 16) -> None:
    colors = ["#e8f6e8", "#bde8bd", "#77c877", "#2c9a48"]
    for r in range(3):
        for c in range(4):
            idx = (r * 4 + c) % len(colors)
            box = (x + c * scale, y + r * scale, x + c * scale + scale - 3, y + r * scale + scale - 3)
            draw.rectangle(box, fill=colors[idx], outline=GREEN_DARK, width=1)


def draw_module(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fill: str,
    outline: str,
    text_fill: str = INK,
) -> None:
    round_rect(draw, box, fill, outline, width=2, radius=12)
    center_text(draw, box, text, F_TEXT_SMALL, text_fill)


def draw_local_arrow(draw: ImageDraw.ImageDraw, p1: tuple[int, int], p2: tuple[int, int]) -> None:
    arrow(draw, [p1, p2], "#6b7c8f", width=3, size=11)


def draw_client(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int = 520,
    h: int = 306,
    title: str = "Client A",
    send_mirror: bool = False,
    recv_mirror: bool = False,
) -> dict[str, tuple[int, int]]:
    round_rect(draw, (x, y, x + w, y + h), "white", BLUE, width=3, radius=16)
    center_text(draw, (x, y + 8, x + w, y + 42), title, F_SECTION, BLUE_DARK)

    draw.rectangle((x + 18, y + 52, x + w - 18, y + 82), fill="#e6edf5")
    center_text(draw, (x + 18, y + 52, x + w - 18, y + 82), "Send pipeline", F_LABEL_SMALL, BLUE_DARK)
    draw.rectangle((x + 18, y + 188, x + w - 18, y + 214), fill="#e6edf5")
    center_text(draw, (x + 18, y + 188, x + w - 18, y + 214), "Receive pipeline", F_LABEL_SMALL, BLUE_DARK)

    mw, mh = 92, 54
    left_to_right = [x + 30, x + 150, x + 270, x + 390]
    right_to_left = [x + 398, x + 278, x + 158, x + 38]
    send_x = right_to_left if send_mirror else left_to_right
    recv_x = right_to_left if recv_mirror else left_to_right
    send_y = y + 94
    recv_y = y + 226
    send_labels = (
        ["Audio\nInput", "SCIT\nEncoder", "RVQ\nIndex Pack", "Index\nPacket"]
        if send_mirror
        else ["Audio\nInput", "SCIT\nEncoder", "RVQ\nIndex Pack", "Index\nPacket"]
    )
    recv_labels = (
        ["Packet\nUnpack", "Jitter\nBuffer", "SCIT\nDecoder", "Audio\nOutput"]
        if recv_mirror
        else ["Packet\nUnpack", "Jitter\nBuffer", "SCIT\nDecoder", "Audio\nOutput"]
    )
    send_fills = [BLUE_LIGHT, BLUE_MODULE, GREEN_LIGHT, GREEN_LIGHT]
    recv_fills = [GREEN_LIGHT, BLUE_LIGHT, BLUE_MODULE, BLUE_LIGHT]
    send_text = [INK, "white", INK, INK]
    recv_text = [INK, INK, "white", INK]

    centers = {}
    for i, label in enumerate(send_labels):
        box = (send_x[i], send_y, send_x[i] + mw, send_y + mh)
        outline = GREEN if ("Index" in label or "RVQ" in label) else BLUE
        draw_module(draw, box, label, send_fills[i], outline, send_text[i])
        centers[f"send{i}"] = (send_x[i] + mw // 2, send_y + mh // 2)
        if i:
            if send_mirror:
                draw_local_arrow(draw, (send_x[i - 1], send_y + mh // 2), (send_x[i] + mw + 6, send_y + mh // 2))
            else:
                draw_local_arrow(draw, (send_x[i - 1] + mw, send_y + mh // 2), (send_x[i] - 6, send_y + mh // 2))

    for i, label in enumerate(recv_labels):
        box = (recv_x[i], recv_y, recv_x[i] + mw, recv_y + mh)
        outline = GREEN if label.startswith("Packet") else BLUE
        draw_module(draw, box, label, recv_fills[i], outline, recv_text[i])
        centers[f"recv{i}"] = (recv_x[i] + mw // 2, recv_y + mh // 2)
        if i:
            if recv_mirror:
                draw_local_arrow(draw, (recv_x[i - 1], recv_y + mh // 2), (recv_x[i] + mw + 6, recv_y + mh // 2))
            else:
                draw_local_arrow(draw, (recv_x[i - 1] + mw, recv_y + mh // 2), (recv_x[i] - 6, recv_y + mh // 2))

    codebook_grid(draw, x + w // 2 - 88, y + 153, 14)
    center_text(draw, (x + w // 2 - 20, y + 150, x + w // 2 + 140, y + 184), "same RVQ\ncodebook", font(13, True), GREEN_DARK)

    if send_mirror:
        centers["index_out"] = (send_x[3], send_y + mh // 2)
    else:
        centers["index_out"] = (send_x[3] + mw, send_y + mh // 2)
    if recv_mirror:
        centers["packet_in"] = (recv_x[0] + mw, recv_y + mh // 2)
    else:
        centers["packet_in"] = (recv_x[0], recv_y + mh // 2)
    centers["top"] = (x + w // 2, y)
    return centers


def draw_shared_block(draw: ImageDraw.ImageDraw) -> dict[str, tuple[int, int]]:
    x, y, w, h = 500, 56, 920, 210
    draw.rectangle((470, 18, 1450, 282), fill="#f4f8fc")
    center_text(draw, (470, 18, 1450, 52), "Offline shared artifacts", F_SECTION, BLUE_DARK)
    # dashed border
    for p1, p2 in [((x, y), (x + w, y)), ((x + w, y), (x + w, y + h)), ((x + w, y + h), (x, y + h)), ((x, y + h), (x, y))]:
        dashed_line(draw, p1, p2, BLUE_DARK, width=2, dash=12, gap=8)
    center_text(draw, (x, y + 8, x + w, y + 42), "Shared offline deployment", F_SECTION, BLUE_DARK)

    tile_w = 245
    tiles = [
        (x + 35, y + 62, "Pre-shared\nSCIT-Speech-\nLCA weights", "weights"),
        (x + 337, y + 62, "Pre-shared\nRVQ codebooks", "codebooks"),
        (x + 640, y + 62, "Pre-shared\ndecoder", "decoder"),
    ]
    for tx, ty, label, kind in tiles:
        tile = (tx, ty, tx + tile_w, ty + 100)
        round_rect(draw, tile, "white", GRAY_MID, width=2, radius=10)
        if kind == "codebooks":
            codebook_grid(draw, tx + 96, ty + 14, 15)
            center_text(draw, (tx + 14, ty + 58, tx + tile_w - 14, ty + 94), label, F_TEXT_XSMALL, INK)
        elif kind == "weights":
            draw.ellipse((tx + 96, ty + 11, tx + 108, ty + 23), fill=BLUE_FILL, outline=BLUE)
            draw.ellipse((tx + 138, ty + 11, tx + 150, ty + 23), fill=BLUE_FILL, outline=BLUE)
            draw.ellipse((tx + 117, ty + 38, tx + 129, ty + 50), fill=BLUE_FILL, outline=BLUE)
            draw.line((tx + 102, ty + 17, tx + 144, ty + 17), fill=BLUE, width=2)
            draw.line((tx + 102, ty + 17, tx + 123, ty + 44), fill=BLUE, width=2)
            draw.line((tx + 144, ty + 17, tx + 123, ty + 44), fill=BLUE, width=2)
            center_text(draw, (tx + 14, ty + 55, tx + tile_w - 14, ty + 98), label, F_TEXT_XSMALL, INK)
        else:
            draw.rectangle((tx + 96, ty + 20, tx + 114, ty + 64), fill=BLUE_FILL, outline=BLUE, width=2)
            draw.polygon([(tx + 114, ty + 42), (tx + 150, ty + 22), (tx + 150, ty + 62)], fill=BLUE_FILL, outline=BLUE)
            center_text(draw, (tx + 14, ty + 62, tx + tile_w - 14, ty + 94), label, F_TEXT_XSMALL, INK)

    center_text(draw, (x, y + 160, x + w, y + 200), "same codebooks at all clients", font(18, True, True), GREEN_DARK)
    return {"bottom_left": (x + 210, y + h), "bottom_mid": (x + w // 2, y + h), "bottom_right": (x + w - 210, y + h)}


def main() -> None:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 282, W, 676), fill="#fbfdff")
    center_text(draw, (690, 274, 1230, 306), "Online index forwarding", F_SECTION, GRAY)

    shared = draw_shared_block(draw)

    # Offline dashed links, kept behind the online green flow.
    dashed_line(draw, shared["bottom_left"], (332, 318), "#7992ad", width=3, dash=13, gap=10)
    dashed_line(draw, shared["bottom_right"], (1588, 318), "#7992ad", width=3, dash=13, gap=10)
    dashed_line(draw, shared["bottom_mid"], (960, 708), "#7992ad", width=3, dash=13, gap=10)

    a = draw_client(draw, 72, 318, title="Client A")
    b = draw_client(draw, 1328, 318, title="Client B", send_mirror=True, recv_mirror=True)
    c = draw_client(draw, 700, 708, title="Client C")

    router = (810, 362, 1110, 520)
    round_rect(draw, router, GRAY_LIGHT, "#6e747a", width=3, radius=18)
    center_text(draw, (router[0], router[1] + 10, router[2], router[1] + 52), "Central Router", F_SECTION, INK)
    center_text(
        draw,
        (router[0] + 20, router[1] + 58, router[2] - 20, router[3] - 12),
        "packet forwarding only\nno decoding\nno codebook access",
        F_TEXT,
        INK,
    )

    callout = (770, 544, 1150, 662)
    round_rect(draw, callout, "#f3fff3", GREEN, width=2, radius=14)
    center_text(
        draw,
        callout,
        "Index-only network payload\n10-bit RVQ indices\nL = 1 / 2 / 3\nbody payload",
        font(22, True),
        GREEN_DARK,
        spacing=7,
    )

    # Simplified bidirectional index-only links. The client panels describe local
    # send/receive endpoints; these links keep the network layer readable.
    double_arrow(draw, [(592, 470), (810, 470)], GREEN, width=4, size=14)
    center_text(draw, (618, 480, 790, 508), "RVQ index\npackets only", F_TEXT_SMALL, GREEN_DARK)

    double_arrow(draw, [(1110, 470), (1328, 470)], GREEN, width=4, size=14)
    center_text(draw, (1130, 480, 1304, 508), "RVQ index\npackets only", F_TEXT_SMALL, GREEN_DARK)

    vertical_double_arrow_segment(draw, 960, 520, callout[1], GREEN, width=4, size=14)
    vertical_double_arrow_segment(draw, 960, callout[3], 708, GREEN, width=4, size=14)

    # Repaint client borders and titles lightly where dashed links pass over them.
    for x, y, title in [(72, 318, "Client A"), (1328, 318, "Client B"), (700, 708, "Client C")]:
        draw.rounded_rectangle((x, y, x + 520, y + 306), radius=16, outline=BLUE, width=3)
        center_text(draw, (x, y + 8, x + 520, y + 42), title, F_SECTION, BLUE_DARK)

    img.save(PNG_OUT, "PNG", optimize=True)
    print(PNG_OUT)


if __name__ == "__main__":
    main()
