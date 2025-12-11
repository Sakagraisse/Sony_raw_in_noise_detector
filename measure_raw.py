import rawpy
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.stats import linregress
import argparse
import sys

def get_bayer_channel(raw, channel_index):
    """
    Extract a specific Bayer channel from the raw image.
    0=Red, 1=Green1, 2=Blue, 3=Green2 (usually, depends on pattern)
    Returns a 2D array of that channel's pixels.
    """
    # raw_image_visible is the full raw buffer.
    # We need to slice it.
    # Assuming standard RGGB or similar 2x2 pattern.
    h, w = raw.raw_image_visible.shape
    
    # Determine offsets based on pattern
    # raw.raw_pattern gives the pattern of the top-left 2x2 block.
    # e.g. [[0, 1], [3, 2]] for RGGB
    # We want to find coordinates (r, c) such that pattern[r%2, c%2] == channel_index
    
    pattern = raw.raw_pattern
    start_r, start_c = -1, -1
    
    for r in range(2):
        for c in range(2):
            if pattern[r, c] == channel_index:
                start_r, start_c = r, c
                break
        if start_r != -1: break
        
    if start_r == -1:
        raise ValueError(f"Channel {channel_index} not found in pattern {pattern}")
        
    return raw.raw_image_visible[start_r::2, start_c::2].astype(np.float64)

def compute_stats_in_patches(raw, centers_x, centers_y, patch_size=50):
    """
    Compute Mean and Variance for the Green channel in each patch.
    Using Green channel (usually index 1 or 3) as it has the most signal.
    """
    # Use Green1 (index 1)
    # Note: Coordinates in centers_x/y are for the full image.
    # The bayer channel image is half size.
    
    # Get Green channel
    green = get_bayer_channel(raw, 1)
    gh, gw = green.shape
    
    means = []
    vars = []
    
    half_size = patch_size // 2
    
    for cx, cy in zip(centers_x, centers_y):
        # Map full coords to half-size coords
        gx = cx / 2.0
        gy = cy / 2.0
        
        x0 = int(max(0, gx - half_size))
        y0 = int(max(0, gy - half_size))
        x1 = int(min(gw, gx + half_size))
        y1 = int(min(gh, gy + half_size))
        
        if x1 > x0 and y1 > y0:
            roi = green[y0:y1, x0:x1]
            means.append(np.mean(roi))
            
            # Robust variance estimation using difference of adjacent pixels
            # Var = 0.5 * mean((x_i - x_{i+1})^2)
            # This removes low-frequency gradients/texture
            diffs = roi[:, :-1] - roi[:, 1:]
            var_est = 0.5 * np.mean(diffs**2)
            vars.append(var_est)
        else:
            means.append(0)
            vars.append(0)
            
    return np.array(means), np.array(vars)

def analyze_iso_pair(chart_path, dark_path, json_path):
    print(f"Analyzing pair:\n  Chart: {os.path.basename(chart_path)}\n  Dark:  {os.path.basename(dark_path)}")
    
    # 1. Load Geometry
    with open(json_path, 'r') as f:
        geom = json.load(f)
    centers_x = geom['centers_x']
    centers_y = geom['centers_y']
    
    # 2. Analyze Dark Frame (Read Noise)
    rn_adu = 0
    black_level = 0
    with rawpy.imread(dark_path) as raw_dark:
        # Global Read Noise (std dev of the whole green channel)
        # Or center patch? Global is usually more stable if no light leak.
        green_dark = get_bayer_channel(raw_dark, 1)
        
        # Simple rejection of hot pixels?
        # For now, simple std dev.
        rn_adu = np.std(green_dark)
        black_level = np.mean(green_dark)
        white_level = raw_dark.white_level
        
    print(f"  Black Level: {black_level:.2f}, White Level: {white_level}")
    print(f"  Read Noise (ADU): {rn_adu:.4f}")
    
    # 3. Analyze Chart Frame (Signal & Variance)
    with rawpy.imread(chart_path) as raw_chart:
        means, variances = compute_stats_in_patches(raw_chart, centers_x, centers_y, patch_size=40)
        
    # 4. Photon Transfer Curve (PTC)
    # Model: Var = RN^2 + Gain^-1 * (Signal - Black)
    # We plot Var vs (Mean - Black)
    # Slope = 1/Gain
    # Intercept should be close to RN^2 (from dark)
    
    signals = means - black_level
    # Filter out saturated patches or too dark patches
    valid_mask = (signals > 0) & (means < (white_level * 0.95))
    
    if np.sum(valid_mask) < 5:
        print("  Not enough valid patches for PTC.")
        return None

    x_fit = signals[valid_mask]
    y_fit = variances[valid_mask]
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit)
    
    # Save PTC plot
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(x_fit, y_fit, alpha=0.5, label='Data')
        plt.plot(x_fit, slope * x_fit + intercept, 'r-', label=f'Fit: y={slope:.2f}x + {intercept:.2f}')
        plt.title(f"Photon Transfer Curve - ISO {os.path.basename(os.path.dirname(json_path)).split('_')[2]}")
        plt.xlabel("Signal (ADU)")
        plt.ylabel("Variance (ADU^2)")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(os.path.dirname(json_path), 'ptc_plot.png')
        plt.savefig(plot_path)
        plt.close()
    except Exception as e:
        print(f"Could not save plot: {e}")

    gain_e_adu = 1.0 / slope if slope > 0 else 0
    
    print(f"  PTC Fit: Slope={slope:.4f}, Intercept={intercept:.2f}, R2={r_value**2:.4f}")
    print(f"  Estimated Gain: {gain_e_adu:.4f} e-/ADU")
    
    # 5. Compute Metrics
    # RN in electrons
    rn_e = rn_adu * gain_e_adu
    
    # Full Well in electrons
    full_well_e = (white_level - black_level) * gain_e_adu
    
    # Engineering Dynamic Range (EDR) at SNR=1
    # EDR = log2( FullWell / RN_e )
    edr = np.log2(full_well_e / rn_e) if rn_e > 0 else 0
    
    # Photographic Dynamic Range (PDR) at SNR=20 (approximate)
    # Solve S^2 - K^2 S - K^2 RN^2 = 0 for S (in electrons)
    # S = (K^2 + sqrt(K^4 + 4 K^2 RN^2)) / 2
    K = 20.0
    if rn_e > 0:
        min_signal_e = (K**2 + np.sqrt(K**4 + 4 * (K**2) * (rn_e**2))) / 2.0
        pdr = np.log2(full_well_e / min_signal_e)
    else:
        pdr = 0
    
    return {
        "iso": None, # To be filled by caller
        "rn_adu": rn_adu,
        "gain": gain_e_adu,
        "rn_e": rn_e,
        "edr": edr,
        "pdr": pdr,
        "ptc_r2": r_value**2
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sorted', required=True, help='Path to sorted folder containing RAW pairs')
    parser.add_argument('--output', required=True, help='Path to output folder containing grid.json files')
    args = parser.parse_args()
    
    results = []
    
    # Iterate over output folders to find processed charts
    for root, dirs, files in os.walk(args.output):
        for f in files:
            if f.endswith('grid.json'):
                # Found a processed chart
                # Folder name is usually project_iso_XXX_chart
                folder_name = os.path.basename(root)
                json_path = os.path.join(root, f)
                
                # Parse ISO from folder name
                # Expected: {project}_iso_{ISO}_chart
                try:
                    parts = folder_name.split('_')
                    iso_idx = parts.index('iso')
                    iso = int(parts[iso_idx + 1])
                except:
                    print(f"Could not parse ISO from {folder_name}")
                    continue
                
                # Find corresponding RAW files in sorted folder
                # We need a file ending in _iso_{ISO}_chart.dng and _iso_{ISO}_dark.dng
                # We assume the project prefix matches or we just search by ISO tag
                
                chart_file = None
                dark_file = None
                
                for sf in os.listdir(args.sorted):
                    if f"_iso_{iso}_chart" in sf and sf.lower().endswith('.dng'):
                        chart_file = os.path.join(args.sorted, sf)
                    if f"_iso_{iso}_dark" in sf and sf.lower().endswith('.dng'):
                        dark_file = os.path.join(args.sorted, sf)
                        
                if chart_file and dark_file:
                    metrics = analyze_iso_pair(chart_file, dark_file, json_path)
                    if metrics:
                        metrics['iso'] = iso
                        results.append(metrics)
                else:
                    print(f"Missing RAW pair for ISO {iso} in {args.sorted}")

    # Sort by ISO
    results.sort(key=lambda x: x['iso'])
    
    # Save results
    out_json = os.path.join(args.output, 'analysis_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nAnalysis complete. Results saved to {out_json}")
    
    # Print summary table
    print(f"{'ISO':<8} {'RN(ADU)':<10} {'Gain':<10} {'RN(e-)':<10} {'EDR':<10} {'PDR':<10}")
    for r in results:
        print(f"{r['iso']:<8} {r['rn_adu']:<10.4f} {r['gain']:<10.4f} {r['rn_e']:<10.4f} {r['edr']:<10.2f} {r['pdr']:<10.2f}")

if __name__ == "__main__":
    main()
