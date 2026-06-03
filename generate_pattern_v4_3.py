#!/usr/bin/env python3
"""Generate a color-ring marker measurement pattern.

Variant 4.3 keeps the same 11x7 measurement patch count. Each corner marker is
a large colored ring with a black center. The colored ring identifies the
corner, while the black center gives a stable geometric anchor that survives
overexposure of the ring better than a filled colored dot.
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

OUTPUT_IMAGE = Path("DR_Grid_4.3_ColorRings.png")
OUTPUT_GEOMETRY = Path("DR_Grid_4.3_ColorRings.geometry.json")

BG_COLOR = (0, 0, 0)
BORDER_COLOR = (255, 255, 255)
GRID_BG_COLOR = (0, 0, 0)
PATCH_START_COLOR = (26, 11, 18)
PATCH_END_COLOR = (254, 151, 223)

BORDER_RECT = [80, 70, 3760, 2090]
BORDER_WIDTH = 16
GRID_RECT = [330, 285, 3510, 1895]
PATCH_SPACING = 28
INNER_PATCH_SCALE = 0.70

MARKERS = {
    "top_left": {
        "name": "red",
        "rgb": (190, 0, 0),
        "center": [180, 170],
        "outer_radius": 112,
        "inner_radius": 46,
    },
    "top_right": {
        "name": "green",
        "rgb": (0, 170, 0),
        "center": [3660, 170],
        "outer_radius": 112,
        "inner_radius": 46,
    },
    "bottom_right": {
        "name": "blue",
        "rgb": (0, 70, 210),
        "center": [3660, 1990],
        "outer_radius": 112,
        "inner_radius": 46,
    },
    "bottom_left": {
        "name": "yellow",
        "rgb": (190, 150, 0),
        "center": [180, 1990],
        "outer_radius": 112,
        "inner_radius": 46,
    },
}


def interpolate_color(start_rgb, end_rgb, index, total_steps):
    if total_steps <= 1:
        return start_rgb

    ratio = index / (total_steps - 1)
    ratio = ratio**0.86
    return tuple(
        int(round(start + (end - start) * ratio))
        for start, end in zip(start_rgb, end_rgb)
    )


def draw_white_border(draw):
    x1, y1, x2, y2 = BORDER_RECT
    for i in range(BORDER_WIDTH):
        draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=BORDER_COLOR)


def draw_color_ring_markers(draw):
    for marker in MARKERS.values():
        cx, cy = marker["center"]
        outer = marker["outer_radius"]
        inner = marker["inner_radius"]
        draw.ellipse([cx - outer, cy - outer, cx + outer, cy + outer], fill=marker["rgb"])
        draw.ellipse([cx - inner, cy - inner, cx + inner, cy + inner], fill=BG_COLOR)


def draw_measurement_grid(draw):
    gx1, gy1, gx2, gy2 = GRID_RECT
    grid_w = gx2 - gx1
    grid_h = gy2 - gy1
    patch_w = (grid_w - (COLS - 1) * PATCH_SPACING) / COLS
    patch_h = (grid_h - (ROWS - 1) * PATCH_SPACING) / ROWS

    draw.rectangle(GRID_RECT, fill=GRID_BG_COLOR)

    patches = []
    index = 0
    for row in range(ROWS):
        for col in range(COLS):
            x = gx1 + col * (patch_w + PATCH_SPACING)
            y = gy1 + row * (patch_h + PATCH_SPACING)
            color = interpolate_color(PATCH_START_COLOR, PATCH_END_COLOR, index, TOTAL_PATCHES)

            rect = [x, y, x + patch_w, y + patch_h]
            draw.rectangle([int(round(v)) for v in rect], fill=color)

            cx = x + patch_w / 2
            cy = y + patch_h / 2
            iw = patch_w * INNER_PATCH_SCALE
            ih = patch_h * INNER_PATCH_SCALE
            inner_rect = [cx - iw / 2, cy - ih / 2, cx + iw / 2, cy + ih / 2]

            patches.append(
                {
                    "index": index + 1,
                    "row": row,
                    "col": col,
                    "rect": [round(v, 3) for v in rect],
                    "inner_rect": [round(v, 3) for v in inner_rect],
                    "center": [round(cx, 3), round(cy, 3)],
                    "rgb": color,
                }
            )
            index += 1

    return patches


def main():
    image = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(image)

    draw_white_border(draw)
    patches = draw_measurement_grid(draw)
    draw_color_ring_markers(draw)
    image.save(OUTPUT_IMAGE)

    geometry = {
        "pattern_version": "DR_Grid_4.3_ColorRings",
        "canvas": {"width": WIDTH, "height": HEIGHT},
        "patch_grid": {
            "cols": COLS,
            "rows": ROWS,
            "patch_count": TOTAL_PATCHES,
            "grid_rect": GRID_RECT,
            "patch_spacing": PATCH_SPACING,
            "inner_patch_scale": INNER_PATCH_SCALE,
            "patches": patches,
        },
        "detection": {
            "background_rgb": BG_COLOR,
            "white_border_rect": BORDER_RECT,
            "white_border_width": BORDER_WIDTH,
            "markers": MARKERS,
            "marker_order": ["top_left", "top_right", "bottom_right", "bottom_left"],
            "marker_color_order": ["red", "green", "blue", "yellow"],
            "anchor_strategy": "color_ring_identity_black_center_anchor",
        },
    }

    OUTPUT_GEOMETRY.write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    print(f"Generated {OUTPUT_IMAGE}")
    print(f"Generated {OUTPUT_GEOMETRY}")


if __name__ == "__main__":
    main()
