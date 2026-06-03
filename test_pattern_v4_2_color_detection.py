#!/usr/bin/env python3
"""Synthetic test bench for v4.2 color-corner detection.

Detection strategy:
1. Find the red/green/blue/yellow blurred blobs by color.
2. Compute each blob centroid.
3. Use known marker colors to orient the pattern.
4. Compute homography from source marker centers to detected marker centers.
5. Project the known patch inner rectangles and save overlays.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


PATTERN_IMAGE = Path("DR_Grid_4.2_ColorCorners.png")
PATTERN_GEOMETRY = Path("DR_Grid_4.2_ColorCorners.geometry.json")
OUTPUT_DIR = Path("output/pattern_v4_2_color_detection_tests")


@dataclass(frozen=True)
class TestCase:
    name: str
    dst_quad: np.ndarray
    blur_radius: int


def ensure_pattern_exists() -> None:
    if PATTERN_IMAGE.exists() and PATTERN_GEOMETRY.exists():
        return

    import generate_pattern_v4_2

    generate_pattern_v4_2.main()


def marker_mask(image_bgr: np.ndarray, color_name: str) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    if color_name == "red":
        mask = (((h <= 12) | (h >= 168)) & (s > 55) & (v > 35))
    elif color_name == "green":
        mask = ((h >= 38) & (h <= 92) & (s > 45) & (v > 35))
    elif color_name == "blue":
        mask = ((h >= 96) & (h <= 138) & (s > 45) & (v > 35))
    elif color_name == "yellow":
        mask = ((h >= 18) & (h <= 38) & (s > 45) & (v > 45))
    else:
        raise ValueError(f"Unsupported color {color_name}")

    mask_u8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask_u8


def detect_color_centroid(image_bgr: np.ndarray, color_name: str) -> tuple[np.ndarray | None, float, np.ndarray]:
    mask = marker_mask(image_bgr, color_name)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0, mask

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < 40:
        return None, area, mask

    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-9:
        return None, area, mask

    center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)
    return center, area, mask


def detect_pattern(image_bgr: np.ndarray, geometry: dict) -> dict:
    detected = {}
    masks = {}
    areas = {}

    for color_name in geometry["detection"]["marker_color_order"]:
        center, area, mask = detect_color_centroid(image_bgr, color_name)
        masks[color_name] = mask
        areas[color_name] = area
        if center is None:
            return {
                "ok": False,
                "reason": f"{color_name} marker not found",
                "masks": masks,
                "areas": areas,
            }
        detected[color_name] = center

    src_points = []
    dst_points = []
    for corner_name in geometry["detection"]["marker_order"]:
        marker = geometry["detection"]["markers"][corner_name]
        color_name = marker["name"]
        src_points.append(marker["center"])
        dst_points.append(detected[color_name])

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    homography, _ = cv2.findHomography(src_points, dst_points, method=0)
    if homography is None:
        return {"ok": False, "reason": "homography failed", "masks": masks, "areas": areas}

    return {
        "ok": True,
        "homography": homography,
        "detected_markers": {k: np.round(v, 3).tolist() for k, v in detected.items()},
        "areas": areas,
        "masks": masks,
    }


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, homography).reshape(-1, 2)


def draw_overlay(image_bgr: np.ndarray, detection: dict, geometry: dict) -> np.ndarray:
    overlay = image_bgr.copy()
    homography = detection["homography"]

    grid_rect = geometry["patch_grid"]["grid_rect"]
    gx1, gy1, gx2, gy2 = grid_rect
    grid_src = np.array([[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]], dtype=np.float32)
    grid_dst = project_points(grid_src, homography).astype(np.int32)
    cv2.polylines(overlay, [grid_dst], True, (0, 255, 255), 3, cv2.LINE_AA)

    for patch in geometry["patch_grid"]["patches"]:
        x1, y1, x2, y2 = patch["inner_rect"]
        rect_src = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        rect_dst = project_points(rect_src, homography).astype(np.int32)
        cv2.polylines(overlay, [rect_dst], True, (0, 255, 0), 1, cv2.LINE_AA)

    marker_colors_bgr = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 80, 0),
        "yellow": (0, 255, 255),
    }
    for color_name, center in detection["detected_markers"].items():
        pt = tuple(np.round(center).astype(int))
        cv2.circle(overlay, pt, 9, marker_colors_bgr[color_name], -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            color_name,
            (pt[0] + 12, pt[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            marker_colors_bgr[color_name],
            2,
            cv2.LINE_AA,
        )

    return overlay


def make_test_cases(width: int, height: int) -> list[TestCase]:
    def quad(points):
        return np.array(points, dtype=np.float32)

    return [
        TestCase("00_flat_sharp", quad([(120, 70), (width - 120, 70), (width - 120, height - 70), (120, height - 70)]), 0),
        TestCase("01_flat_blur6", quad([(120, 70), (width - 120, 70), (width - 120, height - 70), (120, height - 70)]), 6),
        TestCase("02_flat_blur14", quad([(120, 70), (width - 120, 70), (width - 120, height - 70), (120, height - 70)]), 14),
        TestCase("03_flat_blur22", quad([(120, 70), (width - 120, 70), (width - 120, height - 70), (120, height - 70)]), 22),
        TestCase("04_yaw_left_blur10", quad([(235, 95), (width - 80, 35), (width - 135, height - 45), (115, height - 135)]), 10),
        TestCase("05_yaw_right_blur10", quad([(80, 35), (width - 235, 95), (width - 115, height - 135), (135, height - 45)]), 10),
        TestCase("06_pitch_up_blur12", quad([(185, 150), (width - 185, 150), (width - 80, height - 55), (80, height - 55)]), 12),
        TestCase("07_pitch_down_blur12", quad([(80, 55), (width - 80, 55), (width - 185, height - 150), (185, height - 150)]), 12),
        TestCase("08_skew_combo_blur16", quad([(260, 80), (width - 115, 145), (width - 250, height - 70), (95, height - 180)]), 16),
        TestCase("09_severe_pivot_blur18", quad([(310, 135), (width - 60, 75), (width - 210, height - 100), (70, height - 235)]), 18),
    ]


def generate_capture(source_bgr: np.ndarray, test_case: TestCase, output_size: tuple[int, int]) -> np.ndarray:
    src_h, src_w = source_bgr.shape[:2]
    src_quad = np.array([[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src_quad, test_case.dst_quad)
    warped = cv2.warpPerspective(source_bgr, homography, output_size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    if test_case.blur_radius > 0:
        kernel_size = test_case.blur_radius * 2 + 1
        warped = cv2.GaussianBlur(warped, (kernel_size, kernel_size), test_case.blur_radius / 2.0)

    return warped


def save_combined_mask(masks: dict[str, np.ndarray], path: Path) -> None:
    if not masks:
        return
    combined = np.zeros((*next(iter(masks.values())).shape, 3), dtype=np.uint8)
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 80, 0),
        "yellow": (0, 255, 255),
    }
    for color_name, mask in masks.items():
        combined[mask > 0] = color_map[color_name]
    cv2.imwrite(str(path), combined)


def make_overlay_montage(summary: list[dict], output_path: Path) -> None:
    thumbs = []
    for row in summary:
        image = Image.open(row["overlay"]).convert("RGB")
        image.thumbnail((480, 270))
        tile = Image.new("RGB", (480, 300), (20, 20, 20))
        tile.paste(image, (0, 0))

        draw = ImageDraw.Draw(tile)
        label = f"{row['case']} | ok={row['ok']} | blur={row['blur_radius']}"
        draw.rectangle([0, 270, 480, 300], fill=(20, 20, 20))
        draw.text((8, 278), label, fill=(235, 235, 235))
        thumbs.append(tile)

    cols = 2
    rows = int(np.ceil(len(thumbs) / cols))
    montage = Image.new("RGB", (cols * 480, rows * 300), (0, 0, 0))
    for idx, tile in enumerate(thumbs):
        x = (idx % cols) * 480
        y = (idx // cols) * 300
        montage.paste(tile, (x, y))

    montage.save(output_path)


def main() -> None:
    ensure_pattern_exists()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = cv2.imread(str(PATTERN_IMAGE), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Could not read {PATTERN_IMAGE}")

    geometry = json.loads(PATTERN_GEOMETRY.read_text(encoding="utf-8"))
    output_size = (1920, 1080)
    cases = make_test_cases(*output_size)

    summary = []
    for case in cases:
        capture = generate_capture(source, case, output_size)
        detection = detect_pattern(capture, geometry)

        input_path = OUTPUT_DIR / f"{case.name}_input.png"
        mask_path = OUTPUT_DIR / f"{case.name}_color_mask.png"
        overlay_path = OUTPUT_DIR / f"{case.name}_overlay.png"

        cv2.imwrite(str(input_path), capture)
        save_combined_mask(detection.get("masks", {}), mask_path)

        row = {
            "case": case.name,
            "blur_radius": case.blur_radius,
            "ok": bool(detection["ok"]),
            "input": str(input_path),
            "mask": str(mask_path),
            "overlay": str(overlay_path),
        }

        if detection["ok"]:
            overlay = draw_overlay(capture, detection, geometry)
            cv2.imwrite(str(overlay_path), overlay)
            row.update(
                {
                    "detected_markers": detection["detected_markers"],
                    "areas": {k: round(float(v), 3) for k, v in detection["areas"].items()},
                }
            )
        else:
            cv2.imwrite(str(overlay_path), capture)
            row.update(
                {
                    "reason": detection.get("reason", "unknown"),
                    "areas": {k: round(float(v), 3) for k, v in detection.get("areas", {}).items()},
                }
            )

        summary.append(row)

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    montage_path = OUTPUT_DIR / "overlay_montage.png"
    make_overlay_montage(summary, montage_path)

    ok_count = sum(1 for row in summary if row["ok"])
    print(f"Generated {len(cases)} synthetic tests in {OUTPUT_DIR}")
    print(f"Detection success: {ok_count}/{len(cases)}")
    print(f"Summary: {summary_path}")
    print(f"Overlay montage: {montage_path}")


if __name__ == "__main__":
    main()
