#!/usr/bin/env python3
"""Generate a defocus-robust sensor measurement pattern.

The pattern keeps the existing 11x7 measurement patch count but adds a
low-frequency detection scaffold:
- neutral gray outside background
- thick black outer frame
- thick L corners
- large asymmetric orientation markers
- coarse timing bars outside the measurement grid
- clean measurement patches in the center
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


WIDTH = 3840
HEIGHT = 2160
COLS = 11
ROWS = 7
TOTAL_PATCHES = COLS * ROWS

OUTPUT_IMAGE = Path("DR_Grid_4.0_Defocus_Robust.png")
OUTPUT_GEOMETRY = Path("DR_Grid_4.0_Defocus_Robust.geometry.json")

BG_COLOR = (185, 185, 185)
FRAME_COLOR = (0, 0, 0)
PATCH_START_COLOR = (26, 11, 18)
PATCH_END_COLOR = (254, 151, 223)

FRAME = {
    "x": 170,
    "y": 150,
    "w": 3500,
    "h": 1720,
    "thickness": 70,
}

GUARD_MARGIN = 92
PATCH_SPACING = 34
INNER_PATCH_SCALE = 0.66


def interpolate_color(start_rgb, end_rgb, index, total_steps):
    if total_steps <= 1:
        return start_rgb

    # Slightly non-linear distribution: more separation in darker/mid tones.
    ratio = index / (total_steps - 1)
    ratio = ratio**0.82

    return tuple(
        int(round(start + (end - start) * ratio))
        for start, end in zip(start_rgb, end_rgb)
    )


def rect_from_xywh(x, y, w, h):
    return [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))]


def draw_thick_frame(draw):
    x = FRAME["x"]
    y = FRAME["y"]
    w = FRAME["w"]
    h = FRAME["h"]
    t = FRAME["thickness"]

    outer = rect_from_xywh(x, y, w, h)
    inner = rect_from_xywh(x + t, y + t, w - 2 * t, h - 2 * t)

    draw.rectangle(outer, fill=FRAME_COLOR)
    draw.rectangle(inner, fill=BG_COLOR)

    return outer, inner


def draw_l_corners(draw, inner_frame):
    x1, y1, x2, y2 = inner_frame
    arm = 260
    thick = 78

    # Top-left
    draw.rectangle([x1, y1, x1 + arm, y1 + thick], fill=FRAME_COLOR)
    draw.rectangle([x1, y1, x1 + thick, y1 + arm], fill=FRAME_COLOR)

    # Top-right
    draw.rectangle([x2 - arm, y1, x2, y1 + thick], fill=FRAME_COLOR)
    draw.rectangle([x2 - thick, y1, x2, y1 + arm], fill=FRAME_COLOR)

    # Bottom-right
    draw.rectangle([x2 - arm, y2 - thick, x2, y2], fill=FRAME_COLOR)
    draw.rectangle([x2 - thick, y2 - arm, x2, y2], fill=FRAME_COLOR)

    # Bottom-left
    draw.rectangle([x1, y2 - thick, x1 + arm, y2], fill=FRAME_COLOR)
    draw.rectangle([x1, y2 - arm, x1 + thick, y2], fill=FRAME_COLOR)


def draw_orientation_markers(draw):
    # Large, asymmetric, low-frequency shapes outside the main frame.
    # TL: square
    draw.rectangle([42, 42, 132, 132], fill=FRAME_COLOR)

    # TR: triangle
    draw.polygon([(3710, 42), (3820, 132), (3600, 132)], fill=FRAME_COLOR)

    # BR: filled circle
    draw.ellipse([3650, 1986, 3820, 2156], fill=FRAME_COLOR)

    # BL: ring
    draw.ellipse([40, 1986, 210, 2156], fill=FRAME_COLOR)
    draw.ellipse([83, 2029, 167, 2113], fill=BG_COLOR)

    # Top-center small orientation dot, deliberately larger than v3.
    draw.ellipse([1888, 46, 1952, 110], fill=FRAME_COLOR)


def draw_timing_bars(draw, inner_frame, grid_rect):
    x1, y1, x2, y2 = inner_frame
    gx1, gy1, gx2, gy2 = grid_rect

    # Coarse bars between the frame and grid. Wide enough to survive defocus.
    bar_h = 34
    top_y = y1 + 28
    bottom_y = y2 - 28 - bar_h
    left_x = x1 + 28
    right_x = x2 - 28 - bar_h
    bar_w = 78
    gap = 42

    x = gx1
    idx = 0
    while x + bar_w <= gx2:
        if idx % 2 == 0:
            draw.rectangle([x, top_y, x + bar_w, top_y + bar_h], fill=FRAME_COLOR)
        if idx % 3 != 1:
            draw.rectangle([x, bottom_y, x + bar_w, bottom_y + bar_h], fill=FRAME_COLOR)
        x += bar_w + gap
        idx += 1

    bar_w_vertical = 34
    bar_h_vertical = 78
    y = gy1
    idx = 0
    while y + bar_h_vertical <= gy2:
        if idx % 2 == 0:
            draw.rectangle([left_x, y, left_x + bar_w_vertical, y + bar_h_vertical], fill=FRAME_COLOR)
        if idx % 3 != 1:
            draw.rectangle([right_x, y, right_x + bar_w_vertical, y + bar_h_vertical], fill=FRAME_COLOR)
        y += bar_h_vertical + gap
        idx += 1


def draw_measurement_grid(draw, inner_frame):
    x1, y1, x2, y2 = inner_frame
    gx1 = x1 + GUARD_MARGIN
    gy1 = y1 + GUARD_MARGIN + 82
    gx2 = x2 - GUARD_MARGIN
    gy2 = y2 - GUARD_MARGIN - 82

    grid_w = gx2 - gx1
    grid_h = gy2 - gy1
    patch_w = (grid_w - (COLS - 1) * PATCH_SPACING) / COLS
    patch_h = (grid_h - (ROWS - 1) * PATCH_SPACING) / ROWS

    draw.rectangle([gx1, gy1, gx2, gy2], fill=FRAME_COLOR)

    patches = []
    patch_index = 0
    for row in range(ROWS):
        for col in range(COLS):
            x = gx1 + col * (patch_w + PATCH_SPACING)
            y = gy1 + row * (patch_h + PATCH_SPACING)
            color = interpolate_color(PATCH_START_COLOR, PATCH_END_COLOR, patch_index, TOTAL_PATCHES)
            patch_rect = [x, y, x + patch_w, y + patch_h]
            draw.rectangle([int(round(v)) for v in patch_rect], fill=color)

            cx = x + patch_w / 2
            cy = y + patch_h / 2
            iw = patch_w * INNER_PATCH_SCALE
            ih = patch_h * INNER_PATCH_SCALE
            inner_rect = [cx - iw / 2, cy - ih / 2, cx + iw / 2, cy + ih / 2]

            patches.append(
                {
                    "index": patch_index + 1,
                    "row": row,
                    "col": col,
                    "rect": [round(v, 3) for v in patch_rect],
                    "inner_rect": [round(v, 3) for v in inner_rect],
                    "center": [round(cx, 3), round(cy, 3)],
                    "rgb": color,
                }
            )
            patch_index += 1

    return [gx1, gy1, gx2, gy2], patches


def main():
    image = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(image)

    outer_frame, inner_frame = draw_thick_frame(draw)
    grid_rect, patches = draw_measurement_grid(draw, inner_frame)
    draw_timing_bars(draw, inner_frame, grid_rect)
    draw_l_corners(draw, inner_frame)
    draw_orientation_markers(draw)

    image.save(OUTPUT_IMAGE)

    geometry = {
        "pattern_version": "DR_Grid_4.0_Defocus_Robust",
        "canvas": {"width": WIDTH, "height": HEIGHT},
        "patch_grid": {
            "cols": COLS,
            "rows": ROWS,
            "patch_count": TOTAL_PATCHES,
            "grid_rect": [round(v, 3) for v in grid_rect],
            "patch_spacing": PATCH_SPACING,
            "inner_patch_scale": INNER_PATCH_SCALE,
            "patches": patches,
        },
        "detection": {
            "background_rgb": BG_COLOR,
            "outer_frame": outer_frame,
            "inner_frame": inner_frame,
            "frame_thickness": FRAME["thickness"],
            "marker_types": {
                "top_left": "square",
                "top_right": "triangle",
                "bottom_right": "filled_circle",
                "bottom_left": "ring",
                "top_center": "filled_circle",
            },
        },
    }

    OUTPUT_GEOMETRY.write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    print(f"Generated {OUTPUT_IMAGE}")
    print(f"Generated {OUTPUT_GEOMETRY}")


if __name__ == "__main__":
    main()
