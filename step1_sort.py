#!/usr/bin/env python3
"""prepare_pairs_and_rename.py

Scan a folder of RAW files, group them by ISO, identify the dark/chart in each pair and rename or copy them with new names:
    {project}_iso_{ISO}_chart.dng
    {project}_iso_{ISO}_dark.dng

Behavior:
- Uses rawpy to read RAW and extract ISO metadata when available.
- Falls back to PIL EXIF reading for standard formats if necessary.
- Groups files by ISO, expects pairs (2 files per ISO). If more/less, outputs warnings.
- Chooses the darkest file (lowest mean raw pixel value) as the dark frame and the other as chart.
- By default copies renamed files into an output folder; use --inplace to rename originals.
"""

import argparse
import os
import shutil
from collections import defaultdict
import rawpy
import numpy as np
from PIL import Image, ExifTags
import sys


def read_iso_rawpy(path):
    """Attempt to read ISO from rawpy metadata; return int or None."""
    try:
        with rawpy.imread(path) as raw:
            # Try several common attribute names
            md = getattr(raw, 'metadata', None)
            if md is not None:
                # metadata may be a SimpleNamespace-like or an object
                for name in ('iso', 'ISO', 'iso_speed', 'iso_speed_ratings', 'iso_value'):
                    val = getattr(md, name, None)
                    if val is not None:
                        try:
                            return int(val)
                        except Exception:
                            pass
                # Some rawpy versions provide 'shot_metadata' with ISO
                shot = getattr(raw, 'shot_metadata', None)
                if shot is not None:
                    for name in ('iso', 'iso_speed_ratings'):
                        val = getattr(shot, name, None)
                        if val is not None:
                            try:
                                return int(val)
                            except Exception:
                                pass
    except Exception:
        # Not a RAW supported by rawpy or read error
        return None
    return None


def read_iso_exif_pillow(path):
    """Fallback: read EXIF with Pillow; works for TIFF/JPEG that have EXIF.
    DNG sometimes returns EXIF via PIL, but not always.
    """
    try:
        img = Image.open(path)
        
        # Strategy 1: _getexif() (mostly for JPEGs)
        if hasattr(img, '_getexif'):
            try:
                info = img._getexif()
                if info and 34855 in info:
                    return int(info[34855])
            except: pass

        # Strategy 2: getexif() and ExifOffset (for TIFF/DNG)
        if hasattr(img, 'getexif'):
            try:
                exif = img.getexif()
                if exif:
                    # Check base IFD
                    if 34855 in exif:
                        return int(exif[34855])
                    # Check ExifOffset (34665)
                    if hasattr(exif, 'get_ifd'):
                        try:
                            exif_ifd = exif.get_ifd(34665)
                            if 34855 in exif_ifd:
                                return int(exif_ifd[34855])
                        except: pass
            except: pass

        # Strategy 3: tag_v2 (TIFF tags)
        if hasattr(img, 'tag_v2'):
            try:
                if 34855 in img.tag_v2:
                    return int(img.tag_v2[34855])
            except: pass

    except Exception:
        return None
    return None


def detect_iso(path):
    iso = read_iso_rawpy(path)
    if iso is not None:
        return iso
    iso = read_iso_exif_pillow(path)
    return iso


def mean_raw_brightness(path):
    try:
        with rawpy.imread(path) as raw:
            arr = raw.raw_image_visible.astype(np.float32)
            return float(np.mean(arr)), float(np.std(arr))
    except Exception:
        # Try to open as PIL as fallback and compute mean
        try:
            img = Image.open(path).convert('L')
            arr = np.array(img, dtype=np.float32)
            return float(np.mean(arr)), float(np.std(arr))
        except Exception:
            return None, None


def collect_raw_files(folder, extensions=None):
    if extensions is None:
        extensions = ('.dng', '.DNG', '.arw', '.ARW', '.nef', '.NEF', '.cr2', '.CR2', '.tiff', '.tif', '.jpg', '.jpeg', '.png')
    files = []
    for root, dirs, fnames in os.walk(folder):
        # Modify dirs in-place to skip output folders
        dirs[:] = [d for d in dirs if d not in ('output', 'prepared', '__pycache__')]
        
        for fn in fnames:
            if fn.lower().endswith(tuple([e.lower() for e in extensions])):
                files.append(os.path.join(root, fn))
    return sorted(files)


def prepare_pairs(folder, project, output_dir=None, inplace=False, dry_run=False):
    files = collect_raw_files(folder)
    if not files:
        print('No supported files found in', folder)
        return
    print(f'Found {len(files)} files')

    iso_groups = defaultdict(list)
    iso_map = {}
    for f in files:
        iso = detect_iso(f)
        if iso is None:
            # As fallback, attempt to infer ISO from filename digits e.g., "ISO_100" or "iso100"
            import re
            m = re.search(r'iso[_-]?(\d{2,4})', os.path.basename(f), re.IGNORECASE)
            if m:
                iso = int(m.group(1))
        if iso is None:
            print(f'Could not detect ISO for {f}; skipping')
            continue
        iso_groups[iso].append(f)
        iso_map[f] = iso

    # Prepare output location
    if not inplace:
        if output_dir is None:
            output_dir = os.path.join(folder, 'prepared')
        os.makedirs(output_dir, exist_ok=True)

    # Process groups
    renamed = []
    for iso, group in sorted(iso_groups.items()):
        if len(group) < 2:
            print(f'Skipping ISO {iso}: Only {len(group)} file found (need at least 1 Dark + 1 Chart). File: {os.path.basename(group[0])}')
            continue
            
        # Sort by brightness
        brightness = []
        for f in group:
            m, s = mean_raw_brightness(f)
            brightness.append((f, m if m is not None else 0.0, s if s is not None else 0.0))
        
        # Sort ascending: Darkest first, Brightest last
        brightness.sort(key=lambda t: (t[1], t[2]))
        
        # Strategy: Take the absolute darkest as Dark, and absolute brightest as Chart
        # Ignore anything in between (redundant shots)
        dark_file, dark_mean, _ = brightness[0]
        chart_file, chart_mean, _ = brightness[-1]
        
        # Safety check: ensure they are actually different files (should be covered by len < 2 check but good to be safe)
        if dark_file == chart_file:
             print(f'Skipping ISO {iso}: Dark and Chart are the same file (brightness analysis failed?).')
             continue

        # Create renamed names
        base = project
        ext_c = os.path.splitext(chart_file)[1]
        new_chart_name = f"{base}_iso_{iso}_chart{ext_c}"
        
        ext_d = os.path.splitext(dark_file)[1]
        new_dark_name = f"{base}_iso_{iso}_dark{ext_d}"

        # Perform action
        if dry_run:
            print(f'[DRY] ISO {iso}:')
            print(f'  Dark : {os.path.basename(dark_file)} -> {new_dark_name}')
            print(f'  Chart: {os.path.basename(chart_file)} -> {new_chart_name}')
            if len(group) > 2:
                print(f'  (Ignored {len(group)-2} other files for this ISO)')
        else:
            if inplace:
                # Rename in place
                try:
                    os.rename(dark_file, os.path.join(os.path.dirname(dark_file), new_dark_name))
                    renamed.append((dark_file, new_dark_name))
                    print(f'Renamed: {os.path.basename(dark_file)} -> {new_dark_name}')
                except Exception as e:
                    print('Could not rename', dark_file, e)
                
                try:
                    os.rename(chart_file, os.path.join(os.path.dirname(chart_file), new_chart_name))
                    renamed.append((chart_file, new_chart_name))
                    print(f'Renamed: {os.path.basename(chart_file)} -> {new_chart_name}')
                except Exception as e:
                    print('Could not rename', chart_file, e)
            else:
                # Copy to output_dir
                try:
                    dst1 = os.path.join(output_dir, new_dark_name)
                    shutil.copy2(dark_file, dst1)
                    renamed.append((dark_file, dst1))
                    print(f'Copied: {os.path.basename(dark_file)} -> {new_dark_name}')
                except Exception as e:
                    print('Could not copy', dark_file, e)
                
                try:
                    dst2 = os.path.join(output_dir, new_chart_name)
                    shutil.copy2(chart_file, dst2)
                    renamed.append((chart_file, dst2))
                    print(f'Copied: {os.path.basename(chart_file)} -> {new_chart_name}')
                except Exception as e:
                    print('Could not copy', chart_file, e)
            
            if len(group) > 2:
                print(f'  (Ignored {len(group)-2} redundant files for ISO {iso})')

    print('\nDone.')
    return renamed


def main():
    parser = argparse.ArgumentParser(description='Prepare RAW pairs by ISO and tag dark/chart')
    parser.add_argument('input_folder', help='Folder containing RAW files')
    parser.add_argument('--project', default='project', help='Project name prefix for renamed files')
    parser.add_argument('--output', default=None, help='Output folder where renamed files will be placed (if not inplace)')
    parser.add_argument('--inplace', action='store_true', help='Rename files in place (default: copy to output)')
    parser.add_argument('--dry-run', action='store_true', help='Show actions without renaming/copying')
    args = parser.parse_args()

    prepare_pairs(args.input_folder, args.project, output_dir=args.output, inplace=args.inplace, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
