#!/usr/bin/env python3
"""
detect_grid_ransac.py

Robust grid detection using Hough Lines + Periodicity Filtering (RANSAC-like approach).
Designed to ignore screen borders and background clutter by focusing on the repetitive nature of the grid.

Algorithm:
1. Preprocessing: CLAHE -> Blur -> Canny Edge Detection.
2. Line Detection: Probabilistic Hough Transform to find all candidate lines.
3. Clustering: Group lines by angle (Horizontal vs Vertical).
4. Periodicity Analysis:
   - Project lines onto a 1D axis (Y for horizontal lines, X for vertical lines).
   - Calculate distances between all pairs of lines.
   - Find the most frequent distance (the "pitch") using a histogram/KDE.
5. Grid Locking:
   - Select a "seed" line that has many neighbors at multiples of the pitch.
   - Grow the grid outwards from the seed: keep lines at pos = seed ± k * pitch.
6. Intersection: Compute intersections of the filtered horizontal and vertical lines.
7. Output: The 4 corners of the detected grid.

Usage:
    python3 detect_grid_ransac.py ip_test_chart.dng --out output_prefix
"""

import argparse
import os
import json
import math
import numpy as np
import cv2
import rawpy
from collections import Counter

def read_image_rgb(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.dng', '.nef', '.cr2', '.arw'):
        #!/usr/bin/env python3
        """
        Wrapper for detect_grid_ransac moved to old/ folder.
        """
        import os
        import sys
        import subprocess

        def main():
            old_script = os.path.join(os.path.dirname(__file__), 'old', 'detect_grid_ransac.py')
            # Ensure old script exists
            if not os.path.exists(old_script):
                print(f"ERROR: moved script not found: {old_script}")
                sys.exit(2)
            args = [sys.executable, old_script] + sys.argv[1:]
            rc = subprocess.call(args)
            sys.exit(rc)

        if __name__ == '__main__':
            main()
    
