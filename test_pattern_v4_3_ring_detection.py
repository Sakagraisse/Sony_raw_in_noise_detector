#!/usr/bin/env python3
"""Synthetic test bench for v4.3 color-ring marker detection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


PATTERN_IMAGE = Path("DR_Grid_4.3_ColorRings.png")
PATTERN_GEOMETRY = Path("DR_Grid_4.3_ColorRings.geometry.json")
OUTPUT_DIR = Path("output/pattern_v4_3_ring_detection_tests")


@dataclass(frozen=True)
class TestCase:
    name: str
    dst_quad: np.ndarray
    blur_radius: int
    exposure_gain: float = 1.0


def ensure_pattern_exists() -> None:
    if PATTERN_IMAGE.exists() and PATTERN_GEOMETRY.exists():
        return

    import generate_pattern_v4_3

    generate_pattern_v4_3.main()


def marker_mask(image_bgr: np.ndarray, color_name: str) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    if color_name == "red":
        mask = (((h <= 12) | (h >= 168)) & (s > 42) & (v > 24))
    elif color_name == "green":
        mask = ((h >= 38) & (h <= 94) & (s > 38) & (v > 24))
    elif color_name == "blue":
        mask = ((h >= 96) & (h <= 140) & (s > 38) & (v > 24))
    elif color_name == "yellow":
        mask = ((h >= 17) & (h <= 42) & (s > 38) & (v > 28))
    else:
        raise ValueError(f"Unsupported color {color_name}")

    mask_u8 = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask_u8


def contour_centroid(contour: np.ndarray) -> np.ndarray | None:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-9:
        return None
    return np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)


def detect_black_center(image_bgr: np.ndarray, color_contour: np.ndarray, ring_center: np.ndarray) -> np.ndarray | None:
    x, y, w, h = cv2.boundingRect(color_contour)
    pad = int(max(w, h) * 0.35)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image_bgr.shape[1], x + w + pad)
    y2 = min(image_bgr.shape[0], y + h + pad)
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_mask = (gray < max(45, int(np.percentile(gray, 18)))).astype(np.uint8) * 255

    # Limit to a circular-ish region around the colored ring centroid so the
    # black background outside the marker does not dominate.
    yy, xx = np.indices(dark_mask.shape)
    local_center = ring_center - np.array([x1, y1], dtype=np.float32)
    radius = max(w, h) * 0.40
    local_region = ((xx - local_center[0]) ** 2 + (yy - local_center[1]) ** 2) <= radius**2
    dark_mask[~local_region] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 20:
        return None
    center = contour_centroid(contour)
    if center is None:
        return None
    return center + np.array([x1, y1], dtype=np.float32)


def detect_marker_anchor(image_bgr: np.ndarray, color_name: str) -> tuple[np.ndarray | None, float, str, np.ndarray]:
    mask = marker_mask(image_bgr, color_name)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0, "missing", mask

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < 40:
        return None, area, "too_small", mask

    ring_center = contour_centroid(contour)
    if ring_center is None:
        return None, area, "bad_moments", mask

    black_center = detect_black_center(image_bgr, contour, ring_center)
    if black_center is not None:
        return black_center, area, "black_center", mask
    return ring_center, area, "color_centroid", mask


def detect_pattern(image_bgr: np.ndarray, geometry: dict) -> dict:
    detected = {}
    masks = {}
    areas = {}
    anchor_methods = {}

    for color_name in geometry["detection"]["marker_color_order"]:
        center, area, method, mask = detect_marker_anchor(image_bgr, color_name)
        masks[color_name] = mask
        areas[color_name] = area
        anchor_methods[color_name] = method
        if center is None:
            return {
                "ok": False,
                "reason": f"{color_name} marker not found",
                "masks": masks,
                "areas": areas,
                "anchor_methods": anchor_methods,
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
        return {
            "ok": False,
            "reason": "homography failed",
            "masks": masks,
            "areas": areas,
            "anchor_methods": anchor_methods,
        }

    return {
        "ok": True,
        "homography": homography,
        "detected_markers": {k: np.round(v, 3).tolist() for k, v in detected.items()},
        "areas": areas,
        "anchor_methods": anchor_methods,
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
            f"{color_name}:{detection['anchor_methods'][color_name]}",
            (pt[0] + 12, pt[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            marker_colors_bgr[color_name],
            2,
            cv2.LINE_AA,
        )

    return overlay


def make_test_cases(width: int, height: int) -> list[TestCase]:
    def quad(points):
        return np.array(points, dtype=np.float32)

    base = [
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
    exposure = [
        TestCase("10_flat_blur10_gain1_8", base[1].dst_quad, 10, 1.8),
        TestCase("11_yaw_blur10_gain1_8", base[4].dst_quad, 10, 1.8),
    ]
    return base + exposure


def generate_capture(source_bgr: np.ndarray, test_case: TestCase, output_size: tuple[int, int]) -> np.ndarray:
    src_h, src_w = source_bgr.shape[:2]
    src_quad = np.array([[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src_quad, test_case.dst_quad)
    warped = cv2.warpPerspective(source_bgr, homography, output_size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    if test_case.exposure_gain != 1.0:
        warped = np.clip(warped.astype(np.float32) * test_case.exposure_gain, 0, 255).astype(np.uint8)

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
            "exposure_gain": case.exposure_gain,
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
                    "anchor_methods": detection["anchor_methods"],
                }
            )
        else:
            cv2.imwrite(str(overlay_path), capture)
            row.update(
                {
                    "reason": detection.get("reason", "unknown"),
                    "areas": {k: round(float(v), 3) for k, v in detection.get("areas", {}).items()},
                    "anchor_methods": detection.get("anchor_methods", {}),
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
