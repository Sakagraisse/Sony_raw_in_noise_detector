import rawpy
import numpy as np
import cv2
import tifffile
import sys
import os
from sklearn.cluster import KMeans

def get_target_grid_coords():
    """
    Calculates the theoretical (x, y) coordinates of the 11x7 grid patch centers
    relative to the crop defined by the outer markers.
    
    Returns:
        dict: mapping (row, col) -> (x, y) in the target cropped image.
        tuple: (target_width, target_height)
    """
    # Base dimensions (Render Space)
    W_render = 3840.0
    H_render = 2160.0
    
    # Marker Margins (defines the crop)
    # From display_grid_dynamic.py:
    # corners = offset + render_w * 0.026 ...
    marker_margin_x = W_render * 0.026
    marker_margin_y = H_render * 0.046
    
    # Target Image Dimensions (Cropped to markers)
    W_target = W_render - 2 * marker_margin_x
    H_target = H_render - 2 * marker_margin_y
    
    # Grid Geometry (Render Space)
    grid_margin_x = W_render * 0.052
    grid_margin_y = H_render * 0.092
    
    patch_spacing = W_render * 0.0052
    
    # Grid Area
    # grid_area_w = render_w - (2 * margin_x)
    grid_area_w = W_render - 2 * grid_margin_x
    # grid_area_h = render_h - (2 * margin_y) - int(render_h * 0.07)
    grid_area_h = H_render - 2 * grid_margin_y - (H_render * 0.07)
    
    COLS = 11
    ROWS = 7
    
    patch_w = (grid_area_w - ((COLS - 1) * patch_spacing)) / COLS
    patch_h = (grid_area_h - ((ROWS - 1) * patch_spacing)) / ROWS
    
    coords_map = {}
    
    # Calculate centers
    for r in range(ROWS):
        for c in range(COLS):
            # Position in Render Space
            # grid_rect starts at margin_x, margin_y
            px_global = grid_margin_x + c * (patch_w + patch_spacing)
            py_global = grid_margin_y + r * (patch_h + patch_spacing)
            
            cx_global = px_global + patch_w / 2.0
            cy_global = py_global + patch_h / 2.0
            
            # Position in Target Space (Shifted by Marker Margin)
            cx_target = cx_global - marker_margin_x
            cy_target = cy_global - marker_margin_y
            
            coords_map[(r, c)] = (cx_target, cy_target)
            
    return coords_map, (int(W_target), int(H_target))

def detect_and_fit_grid(image_rgb):
    h, w = image_rgb.shape[:2]
    debug_img = image_rgb.copy()
    
    # 1. Detection (Adaptive Threshold)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Contrast stretching
    p_low, p_high = np.percentile(gray, (1, 99))
    gray_norm = np.clip((gray - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8) if p_high > p_low else gray
        
    thresh = cv2.adaptiveThreshold(gray_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 151, -5)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by Area
    img_area = h * w
    min_area = img_area / 20000
    max_area = img_area / 100
    
    valid_centers = []
    valid_contours = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                valid_centers.append([cX, cY])
                valid_contours.append(c)
    
    if len(valid_centers) < 20:
        print("Not enough patches found.")
        return None, debug_img

    points = np.array(valid_centers)
    
    # 2. Grid Fitting (Clustering)
    # We assume the image is roughly upright (landscape).
    # We expect 7 distinct Y clusters (Rows) and 11 distinct X clusters (Cols).
    
    # Cluster Y (Rows)
    kmeans_y = KMeans(n_clusters=7, n_init=10)
    y_labels = kmeans_y.fit_predict(points[:, 1].reshape(-1, 1))
    y_centers = kmeans_y.cluster_centers_.flatten()
    
    # Sort cluster indices by their center value (Top to Bottom)
    sorted_y_indices = np.argsort(y_centers)
    # Map label -> row_index (0..6)
    label_to_row = {label: i for i, label in enumerate(sorted_y_indices)}
    
    # Cluster X (Cols)
    kmeans_x = KMeans(n_clusters=11, n_init=10)
    x_labels = kmeans_x.fit_predict(points[:, 0].reshape(-1, 1))
    x_centers = kmeans_x.cluster_centers_.flatten()
    
    # Sort cluster indices by their center value (Left to Right)
    sorted_x_indices = np.argsort(x_centers)
    # Map label -> col_index (0..10)
    label_to_col = {label: i for i, label in enumerate(sorted_x_indices)}
    
    # Assign (row, col) to each point
    src_pts = []
    dst_pts = []
    
    target_map, (target_w, target_h) = get_target_grid_coords()
    
    found_grid_points = 0
    
    for i, pt in enumerate(points):
        row = label_to_row[y_labels[i]]
        col = label_to_col[x_labels[i]]
        
        # Check if this point is an outlier in its cluster?
        # For now, trust KMeans.
        
        if (row, col) in target_map:
            src_pts.append(pt)
            dst_pts.append(target_map[(row, col)])
            found_grid_points += 1
            
            # Debug: Draw (r,c) on image
            cv2.putText(debug_img, f"{row},{col}", (int(pt[0]), int(pt[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
    print(f"Matched {found_grid_points} patches to grid positions.")
    
    if found_grid_points < 20:
        print("Grid matching failed (too few points).")
        return None, debug_img
        
    # 3. Compute Homography
    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H, (target_w, target_h)

def process_raw(raw_path):
    print(f"Processing {raw_path}...")
    with rawpy.imread(raw_path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True)
        
        result = detect_and_fit_grid(rgb)
        
        if result[0] is None:
            print("Detection failed.")
            cv2.imwrite(raw_path + ".debug.jpg", cv2.cvtColor(result[1], cv2.COLOR_RGB2BGR))
            return

        H, (w, h) = result
        
        # Preview
        rectified = cv2.warpPerspective(rgb, H, (w, h))
        preview_path = raw_path + ".rectified_preview.jpg"
        cv2.imwrite(preview_path, cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR))
        print(f"Preview saved: {preview_path}")
        
        # RAW Rectification
        raw_image = raw.raw_image_visible
        # Split CFA
        ch1 = raw_image[0::2, 0::2] # R
        ch2 = raw_image[0::2, 1::2] # G1
        ch3 = raw_image[1::2, 0::2] # G2
        ch4 = raw_image[1::2, 1::2] # B
        
        stack = np.dstack((ch1, ch2, ch3, ch4))
        
        # Scale Homography for half-size RAW channels
        # Input coordinates (x,y) in RGB are 2x coordinates in Bayer planes
        # H maps (x_rgb, y_rgb) -> (u_target, v_target)
        # We want H' mapping (x_bayer, y_bayer) -> (u_target, v_target)
        # x_rgb = 2 * x_bayer
        # So H' = H * Scale(2)
        S = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]])
        H_stack = H @ S
        
        rectified_stack = cv2.warpPerspective(stack, H_stack, (w, h), flags=cv2.INTER_LINEAR)
        
        out_path = raw_path + ".rectified.tiff"
        tifffile.imwrite(out_path, rectified_stack, photometric='rgb')
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rectify_raw_robust.py <raw_file>")
        sys.exit(1)
        
    process_raw(sys.argv[1])
