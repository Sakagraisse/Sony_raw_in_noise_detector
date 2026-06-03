#!/usr/bin/env python3
"""ISO stress test for the v4.3 color-ring pattern detector.

Assumption: ISO 100 is correctly exposed. Higher ISO values are simulated by
applying an exposure gain of ISO / 100 after perspective and optical blur, then
clipping to 8-bit. The goal is not to model RAW physics perfectly, but to stress
the marker detector when patches and marker rings become saturated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import generate_pattern_v4_3
from test_pattern_v4_3_ring_detection import detect_pattern, draw_overlay


PATTERN_IMAGE = Path("DR_Grid_4.3_ColorRings.png")
PATTERN_GEOMETRY = Path("DR_Grid_4.3_ColorRings.geometry.json")
OUTPUT_DIR = Path("output/v4_3_iso_100_to_204800_detection_overlays")
DIRECT_ONLY_OUTPUT_DIR = Path("output/v4_3_iso_100_to_204800_direct_detection_only")

ISO_VALUES = [
    100,
    200,
    400,
    800,
    1600,
    3200,
    6400,
    12800,
    25600,
    51200,
    102400,
    204800,
]


def ensure_pattern_exists() -> None:
    if PATTERN_IMAGE.exists() and PATTERN_GEOMETRY.exists():
        return
    generate_pattern_v4_3.main()


def make_quads(width: int, height: int) -> dict[str, np.ndarray]:
    flat = np.array(
        [
            (120, 70),
            (width - 120, 70),
            (width - 120, height - 70),
            (120, height - 70),
        ],
        dtype=np.float32,
    )

    # About 20% perspective/parallax stress: one side appears larger/closer,
    # the opposite side compressed and shifted.
    parallax_20pct = np.array(
        [
            (width * 0.17, height * 0.10),
            (width * 0.96, height * 0.06),
            (width * 0.89, height * 0.94),
            (width * 0.04, height * 0.82),
        ],
        dtype=np.float32,
    )

    return {
        "flat": flat,
        "parallax_20pct": parallax_20pct,
    }


def generate_capture(
    source_bgr: np.ndarray,
    dst_quad: np.ndarray,
    output_size: tuple[int, int],
    blur_radius: int,
    exposure_gain: float,
) -> np.ndarray:
    src_h, src_w = source_bgr.shape[:2]
    src_quad = np.array(
        [[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src_quad, dst_quad)
    warped = cv2.warpPerspective(
        source_bgr,
        homography,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )

    if blur_radius > 0:
        kernel_size = blur_radius * 2 + 1
        warped = cv2.GaussianBlur(warped, (kernel_size, kernel_size), blur_radius / 2.0)

    if exposure_gain != 1.0:
        warped = np.clip(warped.astype(np.float32) * exposure_gain, 0, 255).astype(np.uint8)

    return warped


def make_montage(summary: list[dict], output_path: Path) -> None:
    thumbs = []
    for row in summary:
        image = Image.open(row["overlay"]).convert("RGB")
        image.thumbnail((480, 270))
        tile = Image.new("RGB", (480, 304), (18, 18, 18))
        tile.paste(image, (0, 0))
        draw = ImageDraw.Draw(tile)
        label = f"{row['view']} | ISO {row['iso']} | {row['overlay_mode']}"
        draw.rectangle([0, 270, 480, 304], fill=(18, 18, 18))
        draw.text((8, 279), label, fill=(235, 235, 235))
        thumbs.append(tile)

    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    montage = Image.new("RGB", (cols * 480, rows * 304), (0, 0, 0))
    for idx, tile in enumerate(thumbs):
        x = (idx % cols) * 480
        y = (idx // cols) * 304
        montage.paste(tile, (x, y))
    montage.save(output_path)


def run_stress(output_dir: Path, propagate_from_iso100: bool) -> tuple[int, int, Path, Path]:
    ensure_pattern_exists()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = cv2.imread(str(PATTERN_IMAGE), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Could not read {PATTERN_IMAGE}")
    geometry = json.loads(PATTERN_GEOMETRY.read_text(encoding="utf-8"))

    output_size = (1920, 1080)
    quads = make_quads(*output_size)
    blur_radius = 10
    summary = []

    for view_name, dst_quad in quads.items():
        view_dir = output_dir / view_name
        view_dir.mkdir(exist_ok=True)
        reference_detection = None

        for iso in ISO_VALUES:
            gain = iso / 100.0
            capture = generate_capture(source, dst_quad, output_size, blur_radius, gain)
            detection = detect_pattern(capture, geometry)

            if iso == 100 and detection["ok"]:
                reference_detection = detection

            stem = f"{view_name}_iso_{iso:06d}_gain_{gain:g}"
            input_path = view_dir / f"{stem}_input.png"
            overlay_path = view_dir / f"{stem}_overlay.png"
            cv2.imwrite(str(input_path), capture)

            if propagate_from_iso100:
                if iso == 100 and detection["ok"]:
                    overlay_detection = detection
                    overlay_mode = "direct_iso100"
                else:
                    overlay_detection = reference_detection
                    overlay_mode = "propagated_from_iso100" if reference_detection is not None else "none"
            else:
                overlay_detection = detection if detection["ok"] else None
                overlay_mode = "direct" if detection["ok"] else "direct_failed"

            row = {
                "view": view_name,
                "iso": iso,
                "exposure_gain": gain,
                "blur_radius": blur_radius,
                "direct_detection_ok": bool(detection["ok"]),
                "overlay_ok": bool(overlay_detection is not None),
                "overlay_mode": overlay_mode if overlay_detection is not None else "none",
                "input": str(input_path),
                "overlay": str(overlay_path),
            }

            if overlay_detection is not None:
                overlay = draw_overlay(capture, overlay_detection, geometry)
                cv2.imwrite(str(overlay_path), overlay)
                row.update(
                    {
                        "detected_markers": detection.get("detected_markers"),
                        "anchor_methods": detection.get("anchor_methods", {}),
                        "areas": {k: round(float(v), 3) for k, v in detection.get("areas", {}).items()},
                        "direct_reason": detection.get("reason", ""),
                    }
                )
            else:
                cv2.imwrite(str(overlay_path), capture)
                row.update(
                    {
                        "reason": detection.get("reason", "unknown"),
                        "anchor_methods": detection.get("anchor_methods", {}),
                        "areas": {k: round(float(v), 3) for k, v in detection.get("areas", {}).items()},
                    }
                )

            summary.append(row)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    montage_path = output_dir / "overlay_montage.png"
    make_montage(summary, montage_path)

    direct_ok_count = sum(1 for row in summary if row["direct_detection_ok"])
    overlay_ok_count = sum(1 for row in summary if row["overlay_ok"])
    print(f"Generated {len(summary)} ISO stress overlays in {output_dir}")
    print(f"Direct detection success: {direct_ok_count}/{len(summary)}")
    print(f"Overlay success with ISO100 propagation: {overlay_ok_count}/{len(summary)}")
    print(f"Summary: {summary_path}")
    print(f"Overlay montage: {montage_path}")
    return direct_ok_count, overlay_ok_count, summary_path, montage_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Generate overlays using only direct per-ISO detection, without ISO100 propagation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.direct_only:
        run_stress(DIRECT_ONLY_OUTPUT_DIR, propagate_from_iso100=False)
    else:
        run_stress(OUTPUT_DIR, propagate_from_iso100=True)


if __name__ == "__main__":
    main()
