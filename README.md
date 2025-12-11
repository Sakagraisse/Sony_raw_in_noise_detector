# Sony RAW Noise Detector

A Python tool to detect "baked-in" noise reduction (spatial filtering) in Sony RAW files (and potentially other camera brands).

Many cameras apply irreversible noise reduction to RAW data before writing the file, often disguised as "RAW". This tool analyzes the spatial correlation of noise in **Dark Frames** to reveal if such filtering has occurred.

## How it works

The tool performs a statistical analysis on the sensor noise:
1.  **Dark Frame Cleaning**: Removes fixed pattern noise (Black Level offsets) and banding (row/column noise) to isolate random read noise.
2.  **Lag-1 Correlation**: Calculates the correlation between adjacent pixels. In a true RAW file, pixel noise should be independent (correlation ≈ 0).
3.  **2D Autocorrelation & FFT**: Visualizes the noise structure. A sharp single peak indicates pure noise; a blurry blob indicates spatial smoothing (NR).

## Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

## Usage

1.  **Capture a Dark Frame**:
    *   Put the lens cap on your camera.
    *   Set your usual settings (ISO, Shutter Speed).
    *   Take a picture (RAW format, e.g., `.ARW`).
    *   *Note: The image must be completely black.*

2.  **Run the GUI**:
    ```bash
    python sony_dark_frame_gui.py
    ```

3.  **Analyze**:
    *   Click **Load RAW (.ARW)**.
    *   Select your dark frame.
    *   Wait for the analysis.

## Interpreting Results

| Metric | Value | Meaning |
| :--- | :--- | :--- |
| **Max Correlation** | `< 0.05` | **Clean RAW**. No significant spatial filtering detected. |
| **Max Correlation** | `> 0.10` | **Baked-in NR**. The camera applied spatial smoothing. |
| **Autocorrelation** | Sharp dot | White noise (Good). |
| **Autocorrelation** | Blurry blob | Smoothed noise (Bad). |

## Project Structure

- `sony_dark_frame_gui.py`: Main application with Graphical User Interface.
- `measure_v5.py`: Command-line script for batch analysis or debugging.
- `sample/`: Folder containing sample `.ARW` files for testing.

### Grid Detection & Rectification

- `detect_grid_corners.py`: Detects grid intersections (morphology/Hough), outputs overlay and JSON.
- `rectify_raw_1d.py`: Main rectifier, prefers intersection-based detection by default and falls back to 1D profiling.

Quick commands:

```bash
# detect intersections only
python3 detect_grid_corners.py ip_test_chart.dng --cols 11 --rows 7 --out ip_test_chart.grid

# rectify using intersection-based detection by default
python3 rectify_raw_1d.py ip_test_chart.dng

# Example: detect and rectify a 9x5 grid (valley peaks)
```bash
python3 rectify_raw_1d.py ip_test_chart.dng --cols 9 --rows 5 --no-intersections
```
```

Testing:

```bash
python3 -m pytest tests/test_detection.py
```
