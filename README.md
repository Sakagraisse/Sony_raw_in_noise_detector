# Sony Sensor Analysis Toolchain

A complete suite for analyzing Sony RAW sensor performance (Read Noise, Gain, Dynamic Range) using a methodology aligned with *PhotonsToPhotos*.

## Overview

This toolchain allows you to:
1.  **Sort** RAW files into pairs (Dark frame + Chart frame) by ISO.
2.  **Rectify** the chart images to detect the measurement grid.
3.  **Analyze** the sensor data to compute:
    *   **Read Noise** (in electrons and ADU).
    *   **Gain** (e-/ADU).
    *   **Photographic Dynamic Range (PDR)** (Normalized to 8MP Print).

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *(Requires: numpy, scipy, rawpy, opencv-python, matplotlib, PyQt6, pygame)*

## Usage

### 1. Display the Calibration Pattern
Use the included tool to display the measurement grid on your screen.
```bash
python3 tools_display_pattern.py
```
*   **Space**: Toggle between the Grid and the Flat Field (Pink).
*   **Esc**: Exit.

### 2. Run the Analysis GUI
Launch the main interface:
```bash
python3 SonySensorAnalysis.py
```

### 3. Workflow Steps (in GUI)

1.  **Load Files**: Select the folder containing your RAW files (`.ARW`, `.DNG`, etc.).
2.  **Choose Output**: Select a folder where results will be saved.
3.  **Step 1: Sort**:
    *   Organizes files into a `sorted/` folder.
    *   Pairs them by ISO (e.g., `iso_100_chart.dng` and `iso_100_dark.dng`).
4.  **Step 2: Rectify**:
    *   Detects the 11x7 grid in the chart images.
    *   Extracts measurement patches.
5.  **Step 3: Analyze**:
    *   Computes physics metrics (Green channel only).
    *   Generates graphs for PDR, Read Noise, and Gain.

---

## How to Take the Photos (Crucial!)

To get accurate results comparable to *PhotonsToPhotos*, you must follow this shooting protocol strictly.

### Equipment
*   **Camera**: Sony camera shooting in **Uncompressed RAW** (if available).
*   **Lens**: A sharp lens (50mm or 85mm recommended).
*   **Screen**: A high-quality monitor to display the grid.

### Shooting Protocol

For **EACH ISO** you want to measure (e.g., 100, 200, 400, 800, 1600, 3200, 6400, 12800), take **TWO** photos:

#### 1. The Dark Frame (`_dark`)
*   **Setup**: Put the lens cap ON. Cover the viewfinder if it's a DSLR.
*   **Settings**: Same ISO and Shutter Speed as the Chart frame.
*   **Goal**: Measure the electronic noise of the sensor in total darkness.

#### 2. The Chart Frame (`_chart`)
*   **Setup**: Display the grid using `tools_display_pattern.py`.
*   **Framing**: Fill the frame with the grid. The 4 corner markers must be visible.
*   **Focus**: **Slightly Defocus!** (This is important to blur the screen pixels/moiré without blurring the patch boundaries too much).
*   **Exposure**:
    *   Use the histogram.
    *   Expose to the right (ETTR) but **DO NOT CLIP**.
    *   The brightest patch (pink) should be near saturation but not white.
*   **Goal**: Measure the photon transfer curve (Variance vs Signal) to determine Gain.

### Tips for Accuracy
*   **Avoid Screen Texture**: The biggest source of error is the "texture" of the screen pixels acting as noise. Defocusing helps. Using a high-resolution screen (Retina/4K) helps.
*   **Uniformity**: Ensure the screen is uniformly lit (no reflections).
*   **Stability**: Use a tripod.

---

## File Structure

*   `SonySensorAnalysis.py`: Main GUI entry point.
*   `step1_sort.py`: Renames and pairs files based on Exif data.
*   `step2_rectify.py`: Computer vision script to find the grid.
*   `step3_analyze.py`: Physics engine (P2P logic).
*   `tools_display_pattern.py`: Pygame script to generate the target.
