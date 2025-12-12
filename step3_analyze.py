#!/usr/bin/env python3
"""measure_raw.py

Performs physics-based sensor analysis on RAW files.
Inputs:
  - Sorted RAW files (pairs of Dark + Chart)
  - Grid geometry (json) from rectification step

Outputs:
  - JSON file with ISO, Read Noise, Gain, PDR metrics.

Methodology aligned with PhotonsToPhotos:
  - Uses Green channel only.
  - Computes Pixel PDR (per pixel).
  - Computes Print PDR (normalized to 8MP).
"""

import os
import sys
import json
import argparse
import numpy as np
import rawpy
from scipy.stats import linregress

# Standard normalization target for "Print PDR" (approx 8MP, 8x10" @ 300ppi)
TARGET_RES_MEGAPIXELS = 8.0 

def get_green_pixels(raw, x, y, w, h):
    """
    Extracts only Green pixels from the raw Bayer array within the bounding box.
    """
    # Ensure coordinates are within bounds
    H, W = raw.raw_image_visible.shape
    x = max(0, min(int(x), W-1))
    y = max(0, min(int(y), H-1))
    w = min(int(w), W - x)
    h = min(int(h), H - y)
    
    if w <= 0 or h <= 0:
        return np.array([])

    # Extract the crop
    crop = raw.raw_image_visible[y:y+h, x:x+w].astype(np.float64)
    
    # Get the color pattern mask for this crop
    # raw.raw_colors is a matrix of 0,1,2,3 (R, G, B, G2)
    crop_colors = raw.raw_colors[y:y+h, x:x+w]
    
    # Filter for Green pixels (usually index 1 and 3 in rawpy)
    # Note: rawpy pattern: 0=Red, 1=Green, 2=Blue, 3=Green
    green_mask = (crop_colors == 1) | (crop_colors == 3)
    
    return crop[green_mask]

def robust_variance(data):
    """
    Computes variance robust to low-frequency texture (like screen pixels).
    Uses the difference between adjacent pixels to estimate noise.
    Var = E[ (x_i - x_{i+1})^2 ] / 2
    """
    if len(data) < 2:
        return 0.0
    # Since we extracted green pixels from a Bayer array, they are not strictly adjacent spatially 
    # in a simple line, but treating them as a stream for difference estimation is a good approximation 
    # for high-frequency noise vs low-frequency shading.
    diffs = np.diff(data)
    return np.mean(diffs**2) / 2.0

def analyze_iso(iso, dark_path, chart_path, grid_json_path):
    print(f"Analyzing ISO {iso}...")
    
    # 1. Analyze Dark Frame (Read Noise)
    with rawpy.imread(dark_path) as raw:
        # Use a central crop for Read Noise to avoid edge artifacts
        H, W = raw.raw_image_visible.shape
        cx, cy = W//2, H//2
        cw, ch = 512, 512 # 512x512 center crop
        
        # Extract Green pixels only
        dark_pixels = get_green_pixels(raw, cx-cw//2, cy-ch//2, cw, ch)
        
        # Read Noise in ADU (standard deviation of black)
        rn_adu = np.std(dark_pixels)
        
        # Black Level (Pedestal)
        black_level = np.mean(dark_pixels)
        
        # Check against metadata Black Level (Sanity Check)
        # raw.black_level_per_channel usually returns a list of 4 values [R, G, B, G]
        # We take the mean of the Green channels (indices 1 and 3 usually)
        black_level_warning = None
        try:
            meta_bl = raw.black_level_per_channel
            if meta_bl:
                # Average of Green channels if 4 values, else just take the value
                if len(meta_bl) >= 4:
                    meta_bl_green = (meta_bl[1] + meta_bl[3]) / 2.0
                else:
                    meta_bl_green = np.mean(meta_bl)
                
                # If difference is > 5% or > 25 ADU, warn user
                diff = abs(black_level - meta_bl_green)
                if diff > 25.0:
                    black_level_warning = f"Diff {diff:.2f} ADU (Meas {black_level:.1f} vs Meta {meta_bl_green:.1f})"
                    print(f"  [WARNING] Measured Black Level ({black_level:.2f}) differs from Metadata ({meta_bl_green:.2f}) by {diff:.2f} ADU.")
                    print("  -> Possible causes: Temperature drift, light leak in Dark frame, or metadata mismatch.")
        except Exception as e:
            pass # Metadata might not be available or standard

        # Saturation Level (White Level)
        white_level = raw.white_level
        
        # Total Resolution for Normalization
        total_pixels = W * H

    # 2. Analyze Chart Frame (Gain & Full Well)
    with open(grid_json_path, 'r') as f:
        grid_info = json.load(f)
    
    means = []
    vars = []
    
    with rawpy.imread(chart_path) as raw:
        # Use the inner rectangles defined in grid.json
        # They are stored as [x, y, w, h] in "inner_rects" if available, 
        # or we derive them from centers.
        
        # Fallback size if not in JSON (approx 50px)
        box_s = 50 
        
        centers_x = grid_info.get('centers_x', [])
        centers_y = grid_info.get('centers_y', [])
        
        for cy in centers_y:
            for cx in centers_x:
                # Extract Green pixels
                pixels = get_green_pixels(raw, cx - box_s//2, cy - box_s//2, box_s, box_s)
                
                if len(pixels) < 10:
                    continue
                    
                mu = np.mean(pixels) - black_level
                v = robust_variance(pixels)
                
                # Filter clipped data (saturation) and too dark data
                if 0 < mu < (white_level - black_level) * 0.95:
                    means.append(mu)
                    vars.append(v)

    # 3. Compute Gain (Inverse slope of Variance vs Signal)
    # Model: Var(ADU) = Gain(ADU/e-) * Signal(ADU) + ReadNoise(ADU)^2
    # Slope = Gain(ADU/e-)
    # We want Gain in e-/ADU, which is 1/Slope
    if len(means) > 5:
        slope, intercept, r_value, p_value, std_err = linregress(means, vars)
        if slope > 0.001:
            gain_e_adu = 1.0 / slope
        else:
            gain_e_adu = 0.0 # Invalid
            
        # Sanity check for gain
        if gain_e_adu < 0.001: 
            gain_e_adu = 0.001 # clamp
    else:
        gain_e_adu = 1.0 # Fallback
        print("Warning: Not enough data points for Gain estimation.")

    # 4. Compute Derived Metrics
    
    # Read Noise in Electrons
    rn_e = rn_adu * gain_e_adu
    
    # Full Well Capacity (e-)
    # Max signal in ADU * Gain
    full_well_adu = white_level - black_level
    full_well_e = full_well_adu * gain_e_adu
    
    # Engineering Dynamic Range (EDR) @ SNR=1
    # log2(FullWell / ReadNoise_e)
    if rn_e > 0:
        edr = np.log2(full_well_e / rn_e)
    else:
        edr = 0
        
    # Photographic Dynamic Range (PDR) @ SNR=20
    # This is the standard definition used by Bill Claff
    # We look for the signal level S where S / Noise = 20
    # Noise = sqrt(RN^2 + S*Gain) (Shot noise dominant usually)
    # S / sqrt(RN_e^2 + S) = 20  (working in electrons)
    # S^2 = 400 * (RN_e^2 + S)
    # S^2 - 400*S - 400*RN_e^2 = 0
    # Solve quadratic for S (Signal in electrons)
    
    # ax^2 + bx + c = 0
    a = 1
    b = -400
    c = -400 * (rn_e**2)
    
    delta = b**2 - 4*a*c
    if delta >= 0:
        s_target_e = (-b + np.sqrt(delta)) / (2*a)
        # PDR is ratio of Saturation to this target signal
        if s_target_e > 0:
            pdr_pixel = np.log2(full_well_e / s_target_e)
        else:
            pdr_pixel = 0
    else:
        pdr_pixel = 0

    # Normalization (Print PDR)
    # Normalize to 8 Megapixels
    norm_factor = np.log2(np.sqrt(total_pixels) / np.sqrt(TARGET_RES_MEGAPIXELS * 1e6))
    pdr_print = pdr_pixel + norm_factor

    return {
        "iso": int(iso),
        "rn_adu": round(rn_adu, 4),
        "gain": round(gain_e_adu, 4),
        "rn_e": round(rn_e, 4),
        "edr": round(edr, 2),
        "pdr_pixel": round(pdr_pixel, 2),
        "pdr_print": round(pdr_print, 2), # This is the "P2P" comparable value
        "black_level_warning": black_level_warning
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sorted', required=True, help='Folder containing sorted raw files')
    parser.add_argument('--output', required=True, help='Root output folder containing grid info')
    args = parser.parse_args()

    results = []
    
    # Find ISOs
    files = os.listdir(args.sorted)
    isos = set()
    for f in files:
        if "_chart" in f or "_dark" in f:
            parts = f.split('_iso_')
            if len(parts) > 1:
                iso_part = parts[1].split('_')[0]
                if iso_part.isdigit():
                    isos.add(int(iso_part))
    
    sorted_isos = sorted(list(isos))
    
    for iso in sorted_isos:
        # Find pair
        chart_f = None
        dark_f = None
        for f in files:
            if f"iso_{iso}_chart" in f:
                chart_f = os.path.join(args.sorted, f)
            if f"iso_{iso}_dark" in f:
                dark_f = os.path.join(args.sorted, f)
        
        if chart_f and dark_f:
            # Find grid json
            # Assuming output structure: output/filename_chart/filename_chart.grid.json
            # We need to reconstruct the folder name created by rectify_raw_1d
            chart_basename = os.path.basename(chart_f)
            chart_name_no_ext = os.path.splitext(chart_basename)[0]
            
            # rectify script creates folder based on filename without extension
            grid_folder = os.path.join(args.output, chart_name_no_ext)
            grid_json = os.path.join(grid_folder, chart_name_no_ext + ".grid.json")
            
            if os.path.exists(grid_json):
                res = analyze_iso(iso, dark_f, chart_f, grid_json)
                results.append(res)
                print(f"ISO {iso}: Gain={res['gain']}, RN={res['rn_e']}e-, PDR(Print)={res['pdr_print']} EV")
            else:
                print(f"Grid not found for ISO {iso}: {grid_json}")
        else:
            print(f"Incomplete pair for ISO {iso}")

    # Save results
    out_path = os.path.join(args.output, 'analysis_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved analysis to {out_path}")

if __name__ == "__main__":
    main()
