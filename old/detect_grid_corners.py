#!/usr/bin/env python3
"""
detect_grid_corners.py

Detect grid intersections/corners for a displayed patch grid.

Features:
- Accepts RAW (.dng) or image files.
- Morphological extraction of horizontal and vertical lines.
- Intersection detection and clustering into expected `cols x rows` grid.
- Sorting into a matrix ordered TL->TR->BR->BL.
- Outputs corner coordinates (4 corners as floats), grid point CSV, and debug overlay image.
- Fallback to Hough line detection for difficult images.

Usage:
    python3 detect_grid_corners.py ip_test_chart.dng --cols 11 --rows 7 --out ip_test_chart

"""

import argparse
import os
import json
import math
from typing import List, Tuple, Optional

import cv2
import numpy as np
import rawpy


def read_image_rgb(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.dng', '.nef', '.cr2'):
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess(output_bps=8, use_camera_wb=True, no_auto_bright=True)
            # rawpy returns BGR? It returns RGB usually; ensure shape
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # consistent with cv2 BGR representation
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def enhance_gray(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def morphological_lines(gray: np.ndarray, cols: int, rows: int,
                        kernel_scale: float = 0.4, iterations: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    h, w = gray.shape
    approx_pitch_x = w / cols
    approx_pitch_y = h / rows

    horiz_len = max(3, int(max(3, approx_pitch_x * kernel_scale)))
    vert_len = max(3, int(max(3, approx_pitch_y * kernel_scale)))

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))

    # Threshold: Otsu on blurred image
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close then open to emphasize long lines
    horiz = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, horiz_kernel, iterations=iterations)
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_OPEN, horiz_kernel, iterations=iterations)

    vert = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, vert_kernel, iterations=iterations)
    vert = cv2.morphologyEx(vert, cv2.MORPH_OPEN, vert_kernel, iterations=iterations)

    return horiz, vert


def intersection_points(horiz_mask: np.ndarray, vert_mask: np.ndarray, min_count: int = 10):
    intersect = cv2.bitwise_and(horiz_mask, vert_mask)
    # small opening
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    intersect = cv2.morphologyEx(intersect, cv2.MORPH_OPEN, kernel_small)

    pts = cv2.findNonZero(intersect)
    if pts is None:
        return []
    pts = pts.reshape(-1, 2)
    return pts.tolist()


def reduce_to_expected_positions(coords: np.ndarray, expected: int) -> Optional[np.ndarray]:
    # coords: 1D array of ints
    if len(coords) < expected:
        return None
    # Quick path: if equal, return
    uq = np.unique(coords)
    if len(uq) == expected:
        return np.sort(uq)
    if len(uq) < expected:
        # Not enough unique positions: group by histogram bins
        hist, bin_edges = np.histogram(coords, bins=expected)
        centers = ((bin_edges[:-1] + bin_edges[1:]) / 2.0).astype(int)
        return np.sort(centers)
    # More unique than expected: histogram-based merge
    hist, edges = np.histogram(coords, bins=expected)
    centers = ((edges[:-1] + edges[1:]) / 2.0).astype(int)
    return np.sort(centers)


def grid_from_intersections(points: List[Tuple[int, int]], cols: int, rows: int) -> Optional[np.ndarray]:
    if len(points) == 0:
        return None
    pts = np.array(points)
    xs = pts[:, 0]
    ys = pts[:, 1]

    ux = reduce_to_expected_positions(xs, cols)
    uy = reduce_to_expected_positions(ys, rows)

    if ux is None or uy is None:
        return None

    # For each expected grid (col, row), find the closest intersection
    grid = np.zeros((rows, cols, 2), dtype=float)
    for r in range(rows):
        for c in range(cols):
            # expected pos
            ex = ux[c]
            ey = uy[r]
            # find closest point
            dists = np.abs(xs - ex) + np.abs(ys - ey)
            idx = np.argmin(dists)
            grid[r, c, 0] = xs[idx]
            grid[r, c, 1] = ys[idx]

    return grid


def corners_from_grid(grid: np.ndarray) -> List[Tuple[float, float]]:
    # Return TL, TR, BR, BL as float coords
    rows, cols, _ = grid.shape
    tl = tuple(grid[0, 0].tolist())
    tr = tuple(grid[0, cols-1].tolist())
    br = tuple(grid[rows-1, cols-1].tolist())
    bl = tuple(grid[rows-1, 0].tolist())
    return [tl, tr, br, bl]


def draw_debug_overlay(img: np.ndarray, grid: np.ndarray, points: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
    out = img.copy()
    if grid is not None:
        rows, cols, _ = grid.shape
        for r in range(rows):
            for c in range(cols):
                x, y = int(grid[r, c, 0]), int(grid[r, c, 1])
                cv2.circle(out, (x, y), 6, (0, 255, 0), 2)
        # Draw cells
        for r in range(rows):
            for c in range(cols - 1):
                p1 = tuple(grid[r, c].astype(int))
                p2 = tuple(grid[r, c+1].astype(int))
                cv2.line(out, p1, p2, (255, 0, 0), 1)
        for c in range(cols):
            for r in range(rows - 1):
                p1 = tuple(grid[r, c].astype(int))
                p2 = tuple(grid[r+1, c].astype(int))
                cv2.line(out, p1, p2, (255, 0, 0), 1)
    if points is not None:
        for (x, y) in points:
            cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)
    return out


def detect_corners_pipeline(path: str, output_prefix: str, cols: int = 11, rows: int = 7,
                            kernel_scale: float = 0.4, iterations: int = 1, fallback_hough: bool = True) -> Optional[dict]:
    img = read_image_rgb(path)
    if img is None:
        print(f"Cannot read {path}")
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = enhance_gray(gray)

    horiz, vert = morphological_lines(gray, cols, rows, kernel_scale, iterations)
    pts = intersection_points(horiz, vert)

    grid = None
    if len(pts) >= int(cols * rows * 0.5):
        grid = grid_from_intersections(pts, cols, rows)

    if grid is None and fallback_hough:
        # Try Hough line detection fallback
        print("Falling back to Hough line detection")
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, threshold=50, minLineLength=min(w, h)/5.0, maxLineGap=int(max(w, h)/100.0))
        if lines is not None:
            # Separate vertical and horizontal lines
            vert_lines = []
            horiz_lines = []
            for l in lines.reshape(-1, 4):
                x1, y1, x2, y2 = l
                dx = x2 - x1
                dy = y2 - y1
                if abs(dx) > abs(dy) * 2:  # approx horizontal
                    horiz_lines.append(l)
                elif abs(dy) > abs(dx) * 2:  # approx vertical
                    vert_lines.append(l)
            # Convert to masks
            mask_h = np.zeros_like(gray)
            for l in horiz_lines:
                cv2.line(mask_h, (l[0], l[1]), (l[2], l[3]), 255, 2)
            mask_v = np.zeros_like(gray)
            for l in vert_lines:
                cv2.line(mask_v, (l[0], l[1]), (l[2], l[3]), 255, 2)
            pts2 = intersection_points(mask_h, mask_v)
            if len(pts2) > len(pts):
                pts = pts2
                grid = grid_from_intersections(pts, cols, rows)

    if grid is None:
        print("Failed to detect grid intersections.")
        return None

    # get corners
    corners = corners_from_grid(grid)

    # Save outputs
    os.makedirs(os.path.dirname(output_prefix) or '.', exist_ok=True)
    cv2.imwrite(output_prefix + '.grid_debug.jpg', draw_debug_overlay(img, grid, pts))

    # Save grid points CSV
    pts_csv = []
    rows_g, cols_g, _ = grid.shape
    for r in range(rows_g):
        for c in range(cols_g):
            x, y = float(grid[r, c, 0]), float(grid[r, c, 1])
            pts_csv.append({'r': r, 'c': c, 'x': x, 'y': y})
    with open(output_prefix + '.grid_points.json', 'w') as fp:
        json.dump({'cols': cols, 'rows': rows, 'corners': corners, 'points': pts_csv}, fp, indent=2)

    return {'corners': corners, 'grid': grid, 'image_debug': output_prefix + '.grid_debug.jpg', 'json': output_prefix + '.grid_points.json'}


def main():
    parser = argparse.ArgumentParser(description='Detect grid corners via morphological line intersection')
    parser.add_argument('input', help='Input image (supports DNG/raw and standard images)')
    parser.add_argument('--cols', type=int, default=11, help='Number of columns in grid')
    parser.add_argument('--rows', type=int, default=7, help='Number of rows in grid')
    parser.add_argument('--out', default=None, help='Output prefix (defaults to <input>.grid)')
    parser.add_argument('--kernel-scale', default=0.35, type=float, help='Scale factor for morphological kernel relative to pitch')
    parser.add_argument('--iter', default=1, type=int, help='Morphology iterations')
    parser.add_argument('--no-hough', dest='no_hough', action='store_true', help='Disable Hough fallback')
    args = parser.parse_args()

    out_prefix = args.out if args.out else (os.path.splitext(args.input)[0] + '.grid')
    res = detect_corners_pipeline(args.input, out_prefix, args.cols, args.rows, args.kernel_scale, args.iter, not args.no_hough)
    if res is None:
        print('Detection failed')
        exit(1)
    print('Corners:', res['corners'])
    print('Saved:', res['image_debug'], res['json'])


if __name__ == '__main__':
    main()
