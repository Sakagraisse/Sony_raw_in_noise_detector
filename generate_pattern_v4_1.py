#!/usr/bin/env python3
"""Generate a measurement-maximized defocus-friendly pattern.

Variant 4.1 keeps the same 11x7 patch count but removes the large external
markers from v4. The screen is expected to be used in a dark room, so the
pattern uses a black surround and a thin white border around the measurement
area. This maximizes patch size while still giving the detector a simple,
high-contrast rectangle for homography.
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

OUTPUT_IMAGE = Path("DR_Grid_4.1_MaxMeasure_WhiteBorder.png")
OUTPUT_GEOMETRY = Path("DR_Grid_4.1_MaxMeasure_WhiteBorder.geometry.json")

BG_COLOR = (0, 0, 0)
BORDER_COLOR = (255, 255, 255)
GRID_BG_COLOR = (0, 0, 0)
PATCH_START_COLOR = (26, 11, 18)
PATCH_END_COLOR = (254, 151, 223)

BORDER_RECT = [105, 95, 3735, 2055]
BORDER_WIDTH = 18
GRID_RECT = [170, 210, 3670, 1940]
PATCH_SPACING = 26
INNER_PATCH_SCALE = 0.70


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
    # Draw multiple outlines because Pillow outline width support varies by version.
    x1, y1, x2, y2 = BORDER_RECT
    for i in range(BORDER_WIDTH):
        draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=BORDER_COLOR)

    # Minimal orientation cue integrated into the border, not an external marker.
    # It helps distinguish top from bottom after homography.
    notch_w = 190
    notch_h = 34
    cx = WIDTH // 2
    draw.rectangle(
        [cx - notch_w // 2, y1, cx + notch_w // 2, y1 + notch_h],
        fill=BORDER_COLOR,
    )


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

    image.save(OUTPUT_IMAGE)

    geometry = {
        "pattern_version": "DR_Grid_4.1_MaxMeasure_WhiteBorder",
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
            "orientation_cue": "small_top_center_border_notch",
        },
    }

    OUTPUT_GEOMETRY.write_text(json.dumps(geometry, indent=2), encoding="utf-8")
    print(f"Generated {OUTPUT_IMAGE}")
    print(f"Generated {OUTPUT_GEOMETRY}")


if __name__ == "__main__":
    main()
