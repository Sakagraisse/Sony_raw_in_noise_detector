import rawpy
import numpy as np
import cv2
import tifffile
# sys and os not required here
import argparse
import os
import shutil
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import linregress
import matplotlib.pyplot as plt
import json

def get_theoretical_patch_centers(target_w, target_h, cols=11, rows=7):
    """
    Renvoie les coordonnées théoriques (pixels) des 4 coins de la grille (centres des patchs).
    Basé sur la géométrie définie dans display_grid_dynamic.py
    """
    W_render = 3840.0
    H_render = 2160.0
    
    margin_x = W_render * 0.052
    margin_y = H_render * 0.092
    patch_spacing = W_render * 0.0052
    
    grid_area_w = W_render - (2 * margin_x)
    grid_area_h = H_render - (2 * margin_y) - (H_render * 0.07)
    
    COLS = cols
    ROWS = rows
    
    patch_w = (grid_area_w - ((COLS - 1) * patch_spacing)) / COLS
    patch_h = (grid_area_h - ((ROWS - 1) * patch_spacing)) / ROWS
    
    def get_center(r, c):
        px = margin_x + c * (patch_w + patch_spacing)
        py = margin_y + r * (patch_h + patch_spacing)
        return [px + patch_w/2, py + patch_h/2]

    p_tl = get_center(0, 0)
    p_tr = get_center(0, COLS - 1)
    p_br = get_center(ROWS - 1, COLS - 1)
    p_bl = get_center(ROWS - 1, 0)
    
    return np.array([p_tl, p_tr, p_br, p_bl], dtype=np.float32), (int(W_render), int(H_render))


def compute_patch_grid(target_w, target_h, cols=11, rows=7):
    """Return all patch centers (rows x cols x 2) and patch size (w,h) in target coordinates."""
    W_render = float(target_w)
    H_render = float(target_h)

    margin_x = W_render * 0.052
    margin_y = H_render * 0.092
    patch_spacing = W_render * 0.0052

    grid_area_w = W_render - (2 * margin_x)
    grid_area_h = H_render - (2 * margin_y) - (H_render * 0.07)

    patch_w = (grid_area_w - ((cols - 1) * patch_spacing)) / cols
    patch_h = (grid_area_h - ((rows - 1) * patch_spacing)) / rows

    def get_center(r, c):
        px = margin_x + c * (patch_w + patch_spacing)
        py = margin_y + r * (patch_h + patch_spacing)
        return (px + patch_w/2.0, py + patch_h/2.0)

    centers = np.zeros((rows, cols, 2), dtype=float)
    for r in range(rows):
        for c in range(cols):
            centers[r, c] = get_center(r, c)

    return centers, (patch_w, patch_h)

def robust_grid_fitting(profile, num_patches, image_dim, axis_name="X"):
    # Mask edges (5%) to avoid artifacts
    margin_mask = int(image_dim * 0.05)
    profile[:margin_mask] = np.min(profile)
    profile[-margin_mask:] = np.min(profile)

    # Normalize profile
    p_min, p_max = np.min(profile), np.max(profile)
    norm_profile = (profile - p_min) / (p_max - p_min + 1e-6)
    
    # Prefer detecting valleys (drops of luminance)
    approx_pitch = image_dim / num_patches
    inv_profile = 1.0 - norm_profile
    
    # Use width constraint to filter noise (patches/gaps are wide)
    # Gaps are narrow, Patches are wide.
    # If we look for Gaps (valleys), width should be small or None.
    # If we look for Patches (peaks), width should be large.
    
    # First detect all valleys with relaxed constraints (no width constraint for gaps)
    all_valleys, valley_props = find_peaks(inv_profile, distance=max(3, approx_pitch * 0.3), prominence=0.01)
    
    # If not many valleys, try a slightly stricter criteria but still prefer valleys
    if len(all_valleys) < max(3, num_patches // 2):
        all_valleys, valley_props = find_peaks(inv_profile, distance=max(3, approx_pitch * 0.3), prominence=0.02)
    
    # Default we will use valleys (dips), fallback to peaks if no valleys found
    if len(all_valleys) >= 3:
        peaks = all_valleys
        properties = valley_props
        is_gaps = True
    else:
        # Try bright peaks (Patches) - use width constraint here
        min_width = approx_pitch * 0.2
        all_peaks, peak_props = find_peaks(norm_profile, distance=max(3, approx_pitch * 0.3), prominence=0.02, width=min_width)
        if len(all_peaks) < 3:
            # Insufficient features
            peaks = np.array([], dtype=int)
            properties = {}
            is_gaps = False
        else:
            peaks = all_peaks
            properties = peak_props
            is_gaps = False
        
    if len(peaks) < 3:
        print(f"Failed to find enough features for {axis_name}")
        return None, peaks, profile

    # Estimate Pitch based on detected features
    diffs = np.diff(peaks)
    # Relax pitch constraints: 0.3 to 2.0 times approx pitch
    valid_diffs = diffs[(diffs > 0.3 * approx_pitch) & (diffs < 2.0 * approx_pitch)]
    
    if len(valid_diffs) == 0:
        print(f"Could not estimate pitch for {axis_name}")
        return None, peaks, profile
        
    estimated_pitch = np.median(valid_diffs)
    print(f"Estimated Pitch {axis_name}: {estimated_pitch:.2f} ({'Gaps' if is_gaps else 'Patches'})")
    
    # Fit Grid
    center = image_dim / 2
    center_idx = np.argmin(np.abs(peaks - center))
    anchor_peak = peaks[center_idx]
    
    indices = np.round((peaks - anchor_peak) / estimated_pitch)
    # Relax outlier threshold: 0.35 * pitch
    valid_mask = np.abs(peaks - (anchor_peak + indices * estimated_pitch)) < (estimated_pitch * 0.35)
    
    valid_peaks = peaks[valid_mask]
    valid_indices = indices[valid_mask]
    
    if len(valid_peaks) < 3:
        print("Not enough valid grid points aligned.")
        return None, peaks, profile

    slope, intercept, r_value, _, _ = linregress(valid_indices, valid_peaks)
    print(f"Fit {axis_name}: R2={r_value**2:.4f}, Slope={slope:.2f}")
    
    # Reconstruct Grid Centers
    # We assume the grid is centered on the image.
    # Center index is (num_patches - 1) / 2.
    grid_center_idx = (num_patches - 1) / 2.0
    
    # Calculate shift to align grid center to image center
    # pos(k) = slope * k + intercept
    # We want pos(grid_center_idx - shift) ~ image_dim / 2
    
    # Calculate ideal shift (float)
    ideal_shift = grid_center_idx - (image_dim / 2 - intercept) / slope
    
    # Force shift to be the integer that puts the grid closest to center
    # But also check if the "anchor" (index 0) is actually the center patch.
    # If anchor is center patch, shift should be grid_center_idx.
    # If anchor is center patch, intercept ~ image_dim / 2.
    # ideal_shift = grid_center_idx - (0) = grid_center_idx.
    
    shift = int(round(ideal_shift))
    print(f"Shift {axis_name}: {shift} (Ideal: {ideal_shift:.2f})")
    
    # Sanity check: if shift is wildly off, force it based on center assumption?
    # If the detected grid is valid, the shift should be correct.
    # The only ambiguity is if we missed so many peaks that we misidentified the anchor.
    # But anchor is chosen as closest to center. So index 0 is always the "center-most detected peak".
    # So shift should always map index 0 to a patch index near grid_center_idx.
    
    patch_centers = []
    
    if is_gaps:
        # Gaps logic
        num_gaps = num_patches - 1
        gap_center_idx = (num_gaps - 1) / 2.0
        shift_gaps = int(round(gap_center_idx - (image_dim / 2 - intercept) / slope))
        
        # Gap 0 pos (relative to anchor 0)
        gap_0_pos = slope * (-shift_gaps) + intercept
        patch_0_pos = gap_0_pos - slope / 2.0
        
        for i in range(num_patches):
            patch_centers.append(patch_0_pos + i * slope)
            
    else:
        # Patches logic
        patch_0_pos = slope * (-shift) + intercept
        for i in range(num_patches):
            patch_centers.append(patch_0_pos + i * slope)

    display_profile = inv_profile if is_gaps else norm_profile
    return np.array(patch_centers, dtype=float), np.array(peaks, dtype=int), display_profile

def detect_grid_1d_robust(image_rgb, cols=11, rows=7, save_prefix=None):
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Projections 1D (Moyenne sur bande centrale 10%)
    h_band = int(h * 0.1)
    h_start = (h - h_band) // 2
    strip_h = gray[h_start:h_start+h_band, :]
    profile_x = np.mean(strip_h, axis=0)
    
    w_band = int(w * 0.1)
    w_start = (w - w_band) // 2
    strip_v = gray[:, w_start:w_start+w_band]
    profile_y = np.mean(strip_v, axis=1)
    
    # Lissage
    profile_x = cv2.GaussianBlur(profile_x.reshape(1, -1), (51, 1), 0).flatten()
    profile_y = cv2.GaussianBlur(profile_y.reshape(1, -1), (1, 51), 0).flatten()
    
    # 2. Robust Fitting
    patch_centers_x, peaks_x, inv_x = robust_grid_fitting(profile_x, cols, w, "X")
    patch_centers_y, peaks_y, inv_y = robust_grid_fitting(profile_y, rows, h, "Y")
    
    # If one axis failed, fail (we use 1D peak-based only)
    if patch_centers_x is None or patch_centers_y is None:
        print("1D detection failed on one axis; aborting (no intersections fallback).")
        return None, (w, h), image_rgb

    # 3. Construction des 4 coins (Centres des patchs extrêmes)
    # Build unclipped corners from the fitted patch centers (can be outside the image)
    src_pts_unclipped = np.array([
        [patch_centers_x[0], patch_centers_y[0]],   # TL
        [patch_centers_x[-1], patch_centers_y[0]],  # TR
        [patch_centers_x[-1], patch_centers_y[-1]], # BR
        [patch_centers_x[0], patch_centers_y[-1]]   # BL
    ], dtype=np.float32)
    # Keep unclipped info for debug/JSON, but clip coordinates to image bounds for homography and overlay
    src_pts = src_pts_unclipped.copy()
    src_pts[:, 0] = np.clip(src_pts[:, 0], 0, w - 1)
    src_pts[:, 1] = np.clip(src_pts[:, 1], 0, h - 1)

    print(f"Coins extrapolés (unclipped): \n{src_pts_unclipped}")
    print(f"Coins utilisés (clipped to image bounds): \n{src_pts}")

    # Plotting
    if save_prefix:
        plt.figure(figsize=(12, 10))
        
        # X Axis
        plt.subplot(2, 1, 1)
        plt.plot(inv_x, label='Profile X (Norm)')
        plt.plot(peaks_x, inv_x[peaks_x], "x", color='gray', label='Detected Peaks')
        for px in patch_centers_x:
            plt.axvline(x=px, color='green', linestyle='--', alpha=0.5, label='Patch Center' if px==patch_centers_x[0] else "")
        plt.title(f"X Axis: {cols} Patches")
        plt.legend()
        
        # Y Axis
        plt.subplot(2, 1, 2)
        plt.plot(inv_y, label='Profile Y (Norm)')
        plt.plot(peaks_y, inv_y[peaks_y], "x", color='gray', label='Detected Peaks')
        for py in patch_centers_y:
            plt.axvline(x=py, color='green', linestyle='--', alpha=0.5, label='Patch Center' if py==patch_centers_y[0] else "")
        plt.title(f"Y Axis: {rows} Patches")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_prefix + ".robust_profile.png")
        plt.close()
        print(f"Graphiques sauvegardés : {save_prefix}.robust_profile.png")

    # 4. Homographie
    dst_pts, (W_target, H_target) = get_theoretical_patch_centers(3840, 2160, cols=cols, rows=rows)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    
    # Debug Image
    debug_img = image_rgb.copy()
    for x in patch_centers_x:
        xi = int(np.clip(round(x), 0, w - 1))
        cv2.line(debug_img, (xi, 0), (xi, h - 1), (0, 255, 0), 2)
    for y in patch_centers_y:
        yi = int(np.clip(round(y), 0, h - 1))
        cv2.line(debug_img, (0, yi), (w - 1, yi), (0, 0, 255), 2)
    # If intersection debug image is present, overlay points
    try:
        if 'debug_overlay' in locals() and debug_overlay is not None:
            for (ix, iy) in zip(debug_overlay[0], debug_overlay[1]):
                cv2.circle(debug_img, (int(ix), int(iy)), 6, (255, 0, 255), 2)
    except Exception:
        pass
        
    return H, (W_target, H_target), debug_img, np.array(patch_centers_x, dtype=float), np.array(patch_centers_y, dtype=float), np.array(peaks_x, dtype=int), np.array(peaks_y, dtype=int), src_pts_unclipped


def detect_grid_intersections(image_rgb, cols=11, rows=7):
    """Detect grid intersections via morphological extraction of vertical/horizontal lines.
    Returns (unique_sorted_xs, unique_sorted_ys, (xs, ys)) or (None,None,None) if failed.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Blur and threshold
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = gray.shape
    # Kernel sizes based on approximate pitch
    approx_pitch_x = w / cols
    approx_pitch_y = h / rows
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(max(3, approx_pitch_x // 2)), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(max(3, approx_pitch_y // 2))))

    # Extract long horizontal and vertical lines
    horizontal = cv2.erode(bw, horiz_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=1)
    vertical = cv2.erode(bw, vert_kernel, iterations=1)
    vertical = cv2.dilate(vertical, vert_kernel, iterations=1)

    # Find intersections
    intersect = cv2.bitwise_and(horizontal, vertical)
    # Clean small noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    intersect = cv2.morphologyEx(intersect, cv2.MORPH_OPEN, kernel_small)

    # Find non-zero points (intersections)
    pts = cv2.findNonZero(intersect)
    if pts is None or len(pts) < (cols * rows * 0.5):
        # try inverted
        bw_inv = cv2.bitwise_not(bw)
        horizontal = cv2.erode(bw_inv, horiz_kernel, iterations=1)
        horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=1)
        vertical = cv2.erode(bw_inv, vert_kernel, iterations=1)
        vertical = cv2.dilate(vertical, vert_kernel, iterations=1)
        intersect = cv2.bitwise_and(horizontal, vertical)
        intersect = cv2.morphologyEx(intersect, cv2.MORPH_OPEN, kernel_small)
        pts = cv2.findNonZero(intersect)
        if pts is None or len(pts) < (cols * rows * 0.5):
            print("Intersection detection failed: not enough points")
            return None, None, None

    pts = pts.reshape(-1, 2)
    xs = pts[:, 0]
    ys = pts[:, 1]

    # Cluster xs and ys by rounding to get unique column/row coords
    ux = np.unique(np.round(xs).astype(int))
    uy = np.unique(np.round(ys).astype(int))

    # If we have more unique coords than expected, try clustering via binning
    def reduce_to_expected(u, expected):
        if len(u) == expected:
            return np.sort(u)
        if len(u) < expected:
            return None
        # Merge close bins using k-means-like approach: histogram-based
        # Compute bin width
        bins = expected
        hist, bin_edges = np.histogram(u, bins=bins)
        centers = ((bin_edges[:-1] + bin_edges[1:]) / 2.0).astype(int)
        return np.sort(centers)

    ux_r = reduce_to_expected(ux, cols)
    uy_r = reduce_to_expected(uy, rows)
    if ux_r is None or uy_r is None:
        print(f"Failed to find expected columns/rows from intersections: {len(ux)}x{len(uy)}")
        return None, None, None

    # Return sorted columns and rows and raw pts for overlay
    return ux_r.tolist(), uy_r.tolist(), (xs.tolist(), ys.tolist())

def process_raw(raw_path, output_root='output', kernel_scale=0.35, iterations=1, cols=11, rows=7):
    print(f"Traitement de {raw_path} (Méthode Robust 1D)...")
    # Prepare output directory
    base_name = os.path.basename(raw_path)
    base_no_ext = os.path.splitext(base_name)[0]
    output_dir = os.path.join(output_root, base_no_ext)
    os.makedirs(output_dir, exist_ok=True)
    # Copy the RAW file into output_dir (preserve original)
    dest_raw_path = os.path.join(output_dir, base_name)
    try:
        if os.path.abspath(raw_path) != os.path.abspath(dest_raw_path):
            if not os.path.exists(dest_raw_path):
                shutil.copy2(raw_path, dest_raw_path)
            raw_path = dest_raw_path
        elif not os.path.exists(raw_path) and os.path.exists(dest_raw_path):
            # The file was probably already moved/copied earlier; use the one in output_dir
            raw_path = dest_raw_path
            print(f"Using existing file {raw_path}")
    except Exception as e:
        print(f"Warning: could not copy {raw_path} into {output_dir}: {e}")

    with rawpy.imread(raw_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
        # Only 1D detection based on profile peaks; do not call other pipelines
        H = None
        dims = None
        debug_img = None
        result = detect_grid_1d_robust(rgb, cols=cols, rows=rows, save_prefix=os.path.join(output_dir, base_no_ext))
        if result is None or result[0] is None:
            print("Detection via 1D peaks failed.")
            return
        H, dims, debug_img, patch_centers_x, patch_centers_y, peaks_x, peaks_y, src_pts = result
        
        if H is None:
            print("Échec.")
            return
            
        w, h = dims
        # Preview
        rectified = cv2.warpPerspective(rgb, H, (w, h))
        preview_filename = base_no_ext + ".rectified_preview.jpg"
        preview_path = os.path.join(output_dir, preview_filename)
        cv2.imwrite(preview_path, cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR))

        debug_filename = base_no_ext + ".debug_grid.jpg"
        debug_path = os.path.join(output_dir, debug_filename)
        cv2.imwrite(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        print(f"Preview sauvegardée : {preview_path}")
        
        # RAW Rectification
        raw_image = raw.raw_image_visible
        ch1 = raw_image[0::2, 0::2]
        ch2 = raw_image[0::2, 1::2]
        ch3 = raw_image[1::2, 0::2]
        ch4 = raw_image[1::2, 1::2]
        stack = np.dstack((ch1, ch2, ch3, ch4))
        
        S = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]])
        H_stack = H @ S
        
        rectified_stack = cv2.warpPerspective(stack, H_stack, (w, h), flags=cv2.INTER_LINEAR)
        
        out_filename = base_no_ext + ".rectified.tiff"
        out_path = os.path.join(output_dir, out_filename)
        tifffile.imwrite(out_path, rectified_stack, photometric='rgb')
        print(f"Sauvegardé : {out_path}")
        # Save debug JSON with centers
        try:
            grid_json = {
                'cols': cols,
                'rows': rows,
                'centers_x': list(map(float, patch_centers_x)),
                'centers_y': list(map(float, patch_centers_y)),
                'homography_corners': src_pts.tolist()
            }
            json_path = os.path.join(output_dir, base_no_ext + '.grid.json')
            with open(json_path, 'w') as f:
                json.dump(grid_json, f, indent=2)
            print(f"Grid debug JSON saved: {json_path}")
        except Exception:
            pass
        # --- Create overlay images showing patch rectangles on rectified and source images ---
        try:
            centers_grid, (pw, ph) = compute_patch_grid(w, h, cols=cols, rows=rows)
            overlay_rectified = rectified.copy()
            overlay_debug = debug_img.copy()
            # Outer rectangle in green, inner 30% smaller in blue (70% scale)
            inner_scale = 0.7
            half_pw = pw / 2.0
            half_ph = ph / 2.0
            inner_half_pw = half_pw * inner_scale
            inner_half_ph = half_ph * inner_scale
            
            # Prepare extraction directory
            patches_dir = os.path.join(output_dir, "extracted patches")
            if os.path.exists(patches_dir):
                shutil.rmtree(patches_dir)
            os.makedirs(patches_dir, exist_ok=True)
            
            # Extract patches: Bottom-Right -> Left -> Up
            patch_count = 1
            for r in range(rows - 1, -1, -1):
                for c in range(cols - 1, -1, -1):
                    cx, cy = centers_grid[r, c]
                    
                    # Calculate inner rect coordinates
                    tl_x = int(round(cx - inner_half_pw))
                    tl_y = int(round(cy - inner_half_ph))
                    br_x = int(round(cx + inner_half_pw))
                    br_y = int(round(cy + inner_half_ph))
                    
                    # Clip to image bounds
                    tl_x = max(0, tl_x)
                    tl_y = max(0, tl_y)
                    br_x = min(w, br_x)
                    br_y = min(h, br_y)
                    
                    # Extract patch
                    if br_x > tl_x and br_y > tl_y:
                        patch_img = rectified[tl_y:br_y, tl_x:br_x]
                        patch_filename = f"patch_{patch_count}.png"
                        cv2.imwrite(os.path.join(patches_dir, patch_filename), cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR))
                    
                    patch_count += 1
            print(f"Extracted {patch_count-1} patches to {patches_dir}")

            # Draw on rectified image
            for r in range(rows):
                for c in range(cols):
                    cx, cy = centers_grid[r, c]
                    tl = (int(round(cx - half_pw)), int(round(cy - half_ph)))
                    br = (int(round(cx + half_pw)), int(round(cy + half_ph)))
                    # outer rectangle
                    cv2.rectangle(overlay_rectified, tl, br, (0, 255, 0), 2)
                    # inner rectangle
                    tl_in = (int(round(cx - inner_half_pw)), int(round(cy - inner_half_ph)))
                    br_in = (int(round(cx + inner_half_pw)), int(round(cy + inner_half_ph)))
                    cv2.rectangle(overlay_rectified, tl_in, br_in, (255, 0, 0), 2)
            # Map rectified rectangles back to source debug image using inverse homography
            H_inv = None
            try:
                H_inv = np.linalg.inv(H)
            except Exception:
                H_inv = None
            if H_inv is not None:
                pts_src = []
                # collect corners of all outer and inner rects
                for r in range(rows):
                    for c in range(cols):
                        cx, cy = centers_grid[r, c]
                        corners_outer = np.array([
                            [cx - half_pw, cy - half_ph],
                            [cx + half_pw, cy - half_ph],
                            [cx + half_pw, cy + half_ph],
                            [cx - half_pw, cy + half_ph]
                        ], dtype=np.float32)
                        corners_inner = np.array([
                            [cx - inner_half_pw, cy - inner_half_ph],
                            [cx + inner_half_pw, cy - inner_half_ph],
                            [cx + inner_half_pw, cy + inner_half_ph],
                            [cx - inner_half_pw, cy + inner_half_ph]
                        ], dtype=np.float32)
                        pts_src.append((corners_outer, True))
                        pts_src.append((corners_inner, False))
                # Transform and draw
                for corners, is_outer in pts_src:
                    pts = corners.reshape(-1, 1, 2).astype(np.float32)
                    warped = cv2.perspectiveTransform(pts, H_inv)
                    pts_int = np.int32(warped.reshape(-1, 2))
                    color = (0, 255, 0) if is_outer else (255, 0, 0)
                    cv2.polylines(overlay_debug, [pts_int], True, color, 2)
            # Save overlays
            overlay_rectified_path = os.path.join(output_dir, base_no_ext + '.rectified_overlay.jpg')
            overlay_debug_path = os.path.join(output_dir, base_no_ext + '.debug_overlay.jpg')
            cv2.imwrite(overlay_rectified_path, cv2.cvtColor(overlay_rectified, cv2.COLOR_RGB2BGR))
            cv2.imwrite(overlay_debug_path, cv2.cvtColor(overlay_debug, cv2.COLOR_RGB2BGR))
            print(f"Saved overlays: {overlay_rectified_path}, {overlay_debug_path}")
        except Exception as e:
            print("Could not produce overlays:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rectify RAW using 1D or intersection-based grid detection')
    parser.add_argument('input', help='Input RAW file (DNG)')
    parser.add_argument('--kernel-scale', type=float, default=0.35, help='Morphology kernel scale relative to pitch (default: 0.35)')
    parser.add_argument('--iter', type=int, default=1, help='Morphology iterations (default: 1)')
    parser.add_argument('--cols', type=int, default=11, help='Number of patch columns to detect (default: 11)')
    parser.add_argument('--rows', type=int, default=7, help='Number of patch rows to detect (default: 7)')
    parser.add_argument('--output', default='output', help='Root output directory (default: output)')
    args = parser.parse_args()
    process_raw(args.input, output_root=args.output, kernel_scale=args.kernel_scale, iterations=args.iter, cols=args.cols, rows=args.rows)
