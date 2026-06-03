#!/usr/bin/env python3
"""Synthetic test bench for the v4.1 white-border pattern detector.

It generates perspective/blurred captures from
DR_Grid_4.1_MaxMeasure_WhiteBorder.png, detects the white neutral border,
projects the 11x7 patch geometry, and saves overlay images.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PATTERN_IMAGE = Path("DR_Grid_4.1_MaxMeasure_WhiteBorder.png")
PATTERN_GEOMETRY = Path("DR_Grid_4.1_MaxMeasure_WhiteBorder.geometry.json")
OUTPUT_DIR = Path("output/pattern_detection_tests")


@dataclass(frozen=True)
class TestCase:
    name: str
    dst_quad: np.ndarray
    blur_radius: int


def ensure_pattern_exists() -> None:
    if PATTERN_IMAGE.exists() and PATTERN_GEOMETRY.exists():
        return

    import generate_pattern_v4_1

    generate_pattern_v4_1.main()


def order_quad(points: np.ndarray) -> np.ndarray:
    """Return points ordered as TL, TR, BR, BL."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmax(diffs)]
    ordered[3] = pts[np.argmin(diffs)]
    return ordered


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def line_from_y_fit(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 30:
        return None
    x = points[:, 0]
    y = points[:, 1]
    if np.std(x) < 1e-3:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return np.array([-slope, 1.0, -intercept], dtype=np.float64)


def line_from_x_fit(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 30:
        return None
    x = points[:, 0]
    y = points[:, 1]
    if np.std(y) < 1e-3:
        return None
    slope, intercept = np.polyfit(y, x, 1)
    return np.array([1.0, -slope, -intercept], dtype=np.float64)


def intersect_lines(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
    cross = np.cross(line_a, line_b)
    if abs(cross[2]) < 1e-9:
        return None
    return (cross[:2] / cross[2]).astype(np.float32)


def fit_quad_from_mask_bands(mask: np.ndarray) -> tuple[np.ndarray | None, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) < 80:
        return None, 0.0

    points = np.column_stack([xs, ys]).astype(np.float32)
    x_min, x_max = np.percentile(xs, [1, 99])
    y_min, y_max = np.percentile(ys, [1, 99])
    width = x_max - x_min
    height = y_max - y_min
    if width < 50 or height < 50:
        return None, 0.0

    y_band = max(28.0, height * 0.10)
    x_band = max(28.0, width * 0.10)

    top_pts = points[points[:, 1] <= y_min + y_band]
    bottom_pts = points[points[:, 1] >= y_max - y_band]
    left_pts = points[points[:, 0] <= x_min + x_band]
    right_pts = points[points[:, 0] >= x_max - x_band]

    top_line = line_from_y_fit(top_pts)
    bottom_line = line_from_y_fit(bottom_pts)
    left_line = line_from_x_fit(left_pts)
    right_line = line_from_x_fit(right_pts)
    if any(line is None for line in (top_line, bottom_line, left_line, right_line)):
        return None, 0.0

    corners = [
        intersect_lines(top_line, left_line),
        intersect_lines(top_line, right_line),
        intersect_lines(bottom_line, right_line),
        intersect_lines(bottom_line, left_line),
    ]
    if any(corner is None for corner in corners):
        return None, 0.0

    quad = order_quad(np.array(corners, dtype=np.float32))
    area_ratio = polygon_area(quad) / float(mask.shape[0] * mask.shape[1])
    if area_ratio < 0.05:
        return None, area_ratio
    return quad, area_ratio


def white_neutral_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Mask bright neutral pixels, excluding bright magenta patches."""
    image = image_bgr.astype(np.float32)
    max_channel = np.max(image, axis=2)
    min_channel = np.min(image, axis=2)
    mean_channel = np.mean(image, axis=2)
    neutral_delta = max_channel - min_channel

    # Defocus turns the white border into a gray, low-chroma band. A pure
    # "brightest pixels only" threshold breaks too quickly, so keep the luminance
    # threshold intentionally permissive and let RGB neutrality reject the pink
    # measurement patches.
    brightness_threshold = max(62.0, min(145.0, float(np.percentile(mean_channel, 86))))
    mask = ((mean_channel > brightness_threshold) & (neutral_delta < 38.0)).astype(np.uint8) * 255

    h, w = mask.shape
    close_k = max(9, min(h, w) // 80)
    if close_k % 2 == 0:
        close_k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def find_border_quad(mask: np.ndarray) -> tuple[np.ndarray | None, float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(mask.shape[0] * mask.shape[1])

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:8]:
            area = cv2.contourArea(contour)
            if area < image_area * 0.0002:
                continue

            hull = cv2.convexHull(contour)
            for eps_factor in (0.012, 0.018, 0.026, 0.038, 0.055):
                approx = cv2.approxPolyDP(hull, eps_factor * cv2.arcLength(hull, True), True)
                if len(approx) == 4:
                    quad = order_quad(approx.reshape(4, 2))
                    quad_area_ratio = polygon_area(quad) / image_area
                    if quad_area_ratio >= 0.05:
                        return quad, quad_area_ratio

    band_quad, band_area_ratio = fit_quad_from_mask_bands(mask)
    if band_quad is not None:
        return band_quad, band_area_ratio

    # Fallback for fragmented blurred borders: use all neutral-white pixels as
    # one point cloud and fit the convex quadrilateral around them.
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None, 0.0

    points = np.column_stack([xs, ys]).astype(np.float32)
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    for eps_factor in (0.012, 0.018, 0.026, 0.038, 0.055, 0.075):
        approx = cv2.approxPolyDP(hull, eps_factor * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            quad = order_quad(approx.reshape(4, 2))
            return quad, float(len(xs)) / image_area

    rect = cv2.minAreaRect(points)
    quad = order_quad(cv2.boxPoints(rect))
    return quad, float(len(xs)) / image_area


def orient_quad_with_notch(mask: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Rotate quad order so the border notch is on the canonical top edge."""
    canonical_w, canonical_h = 1000, 540
    dst = np.array(
        [[0, 0], [canonical_w - 1, 0], [canonical_w - 1, canonical_h - 1], [0, canonical_h - 1]],
        dtype=np.float32,
    )

    best_score = -1.0
    best_quad = quad
    best_rotation = 0

    for rotation in range(4):
        candidate = np.roll(quad, -rotation, axis=0).astype(np.float32)

        top_len = np.linalg.norm(candidate[1] - candidate[0])
        bottom_len = np.linalg.norm(candidate[2] - candidate[3])
        left_len = np.linalg.norm(candidate[3] - candidate[0])
        right_len = np.linalg.norm(candidate[2] - candidate[1])
        if min(top_len, bottom_len) < max(left_len, right_len) * 1.25:
            continue

        h_mat = cv2.getPerspectiveTransform(candidate, dst)
        rectified = cv2.warpPerspective(mask, h_mat, (canonical_w, canonical_h))

        # The v4.1 top notch is a thick white area centered just inside the top border.
        notch_region = rectified[16:42, canonical_w // 2 - 70 : canonical_w // 2 + 70]
        side_left = rectified[16:42, 90:230]
        side_right = rectified[16:42, canonical_w - 230 : canonical_w - 90]

        notch_score = float(np.mean(notch_region > 0))
        side_score = 0.5 * float(np.mean(side_left > 0) + np.mean(side_right > 0))
        score = notch_score - side_score

        if score > best_score:
            best_score = score
            best_quad = candidate
            best_rotation = rotation

    return best_quad, best_rotation, best_score


def detect_pattern(image_bgr: np.ndarray, geometry: dict) -> dict:
    mask = white_neutral_mask(image_bgr)
    quad, area_ratio = find_border_quad(mask)
    if quad is None:
        return {"ok": False, "mask": mask, "reason": "white border not found"}

    oriented_quad, rotation, notch_score = orient_quad_with_notch(mask, quad)
    source_rect = geometry["detection"]["white_border_rect"]
    x1, y1, x2, y2 = source_rect
    src_quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(src_quad, oriented_quad.astype(np.float32))

    return {
        "ok": True,
        "mask": mask,
        "quad": oriented_quad,
        "homography": homography,
        "area_ratio": area_ratio,
        "notch_rotation": rotation,
        "notch_score": notch_score,
    }


def project_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, homography).reshape(-1, 2)


def draw_overlay(image_bgr: np.ndarray, detection: dict, geometry: dict) -> np.ndarray:
    overlay = image_bgr.copy()
    homography = detection["homography"]

    border_quad = detection["quad"].astype(np.int32)
    cv2.polylines(overlay, [border_quad], True, (255, 255, 255), 4, cv2.LINE_AA)

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

    for idx, point in enumerate(border_quad):
        cv2.circle(overlay, tuple(point), 7, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            overlay,
            str(idx),
            tuple(point + np.array([10, -10])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
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
        mask_path = OUTPUT_DIR / f"{case.name}_mask.png"
        overlay_path = OUTPUT_DIR / f"{case.name}_overlay.png"

        cv2.imwrite(str(input_path), capture)
        cv2.imwrite(str(mask_path), detection["mask"])

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
                    "area_ratio": round(float(detection["area_ratio"]), 6),
                    "notch_rotation": int(detection["notch_rotation"]),
                    "notch_score": round(float(detection["notch_score"]), 6),
                    "quad": np.round(detection["quad"], 3).tolist(),
                }
            )
        else:
            cv2.imwrite(str(overlay_path), capture)
            row["reason"] = detection.get("reason", "unknown")

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
