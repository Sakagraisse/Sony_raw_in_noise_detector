import sys
import os

# Potential fix for numpy/fft freezes in QThreads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import rawpy
import matplotlib
from scipy.stats import kurtosis, skew, norm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QFileDialog, 
                             QLabel, QSpinBox, QSplitter, QProgressBar, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Matplotlib configuration for PyQt
matplotlib.use('QtAgg')

# ---------- BUSINESS LOGIC (MATHS) ----------

def center_crop(arr2d, size):
    h, w = arr2d.shape
    size = min(size, h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return arr2d[y0:y0+size, x0:x0+size]

def clean_dark_frame_pattern(img):
    """
    Cleans the fixed pattern of a Dark Frame (Bayer offsets + Banding).
    """
    img = img.astype(np.float64)
    
    # 1. Black Level Normalization per Bayer phase (2x2)
    for y in range(2):
        for x in range(2):
            sub = img[y::2, x::2]
            img[y::2, x::2] -= np.mean(sub)

    # 2. Destriping (Remove row/column mean)
    row_means = np.mean(img, axis=1, keepdims=True)
    img -= row_means
    col_means = np.mean(img, axis=0, keepdims=True)
    img -= col_means

    return img

def compute_lag1_correlations(noise):
    var = np.mean(noise**2)
    if var == 0:
        return 0.0, 0.0

    prod_x = noise[:, :-1] * noise[:, 1:]
    rho_x = np.mean(prod_x) / var

    prod_y = noise[:-1, :] * noise[1:, :]
    rho_y = np.mean(prod_y) / var

    return rho_x, rho_y

def compute_autocorr_2d(noise):
    F = np.fft.fft2(noise)
    S = np.abs(F)**2
    R = np.fft.ifft2(S)
    R = np.fft.fftshift(np.real(R))
    
    center = R[noise.shape[0] // 2, noise.shape[1] // 2]
    if center != 0:
        R /= center
    return R, S

def compute_radial_profile(spectrum_shifted):
    """
    Computes the radial profile of the 2D power spectrum.
    Used to detect frequency cutoffs (Low Pass Filters).
    """
    y, x = np.indices(spectrum_shifted.shape)
    center = np.array([x.shape[1] // 2, y.shape[0] // 2])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    # Average intensity per radius
    tbin = np.bincount(r.ravel(), spectrum_shifted.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1) # Avoid div by 0
    return radialprofile

def compute_histogram_stats(noise):
    """
    Computes Kurtosis and Skewness to check for Gaussianity.
    Normal Gaussian: Kurtosis ~ 0 (Fisher), Skewness ~ 0.
    """
    data = noise.ravel()
    k = kurtosis(data) # Fisher kurtosis (normal = 0)
    s = skew(data)
    return k, s

def guess_noise_reduction(rho_x, rho_y, threshold=0.05):
    max_corr = max(abs(rho_x), abs(rho_y))
    if max_corr > threshold:
        verdict = "NR PROBABLE (Corr > Threshold)"
    else:
        verdict = "NR UNLIKELY (White Noise)"
    return verdict, max_corr

# ---------- WORKER THREAD (TO AVOID BLOCKING UI) ----------

class AnalysisWorker(QThread):
    # Signals to communicate with the interface
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(object) # Sends a dictionary of results
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, filepath, crop_size):
        super().__init__()
        self.filepath = filepath
        self.crop_size = crop_size

    def run(self):
        try:
            self.log_signal.emit(f"Loading file: {os.path.basename(self.filepath)}...")
            
            # Read RAW
            raw = rawpy.imread(self.filepath)
            full_raw = raw.raw_image_visible
            h, w = full_raw.shape
            self.log_signal.emit(f"RAW Dimensions: {w}x{h}")

            # Crop
            self.log_signal.emit(f"Cropping center ({self.crop_size}x{self.crop_size})...")
            full_crop = center_crop(full_raw, self.crop_size)

            # Cleaning
            self.log_signal.emit("Cleaning Dark Frame (Black Level + Destriping)...")
            clean_noise = clean_dark_frame_pattern(full_crop)

            # Stats calculation
            self.log_signal.emit("Calculating correlation statistics...")
            var_clean = np.mean(clean_noise**2)
            rho_x, rho_y = compute_lag1_correlations(clean_noise)
            verdict, max_corr = guess_noise_reduction(rho_x, rho_y)
            
            # Gaussianity Stats
            self.log_signal.emit("Calculating Gaussianity stats (Kurtosis/Skewness)...")
            kurt, skw = compute_histogram_stats(clean_noise)

            self.log_signal.emit("-" * 40)
            self.log_signal.emit(f"Noise Variance: {var_clean:.3f}")
            self.log_signal.emit(f"Horiz. Corr. (rho_x): {rho_x:.4f}")
            self.log_signal.emit(f"Vert. Corr.   (rho_y): {rho_y:.4f}")
            self.log_signal.emit(f"Max Corr: {max_corr:.4f}")
            self.log_signal.emit(f"Kurtosis: {kurt:.4f} (Ideal: 0)")
            self.log_signal.emit(f"Skewness: {skw:.4f} (Ideal: 0)")
            self.log_signal.emit(f"VERDICT: {verdict}")
            self.log_signal.emit("-" * 40)

            # FFT
            self.log_signal.emit("Calculating FFT and 2D Autocorrelation...")
            R_cl, S_cl = compute_autocorr_2d(clean_noise)
            
            # Radial Profile
            self.log_signal.emit("Calculating Radial Spectral Power...")
            S_shift = np.fft.fftshift(S_cl)
            radial_prof = compute_radial_profile(S_shift)
            
            self.log_signal.emit("FFT done. Preparing results...")

            # Packaging results
            results = {
                "noise_img": clean_noise,
                "autocorr": R_cl,
                "spectrum": S_cl,
                "radial_psd": radial_prof,
                "var": var_clean,
                "kurtosis": kurt,
                "skewness": skw,
                "max_corr": max_corr,
                "verdict": verdict
            }
            self.log_signal.emit("Sending results to main thread...")
            self.result_signal.emit(results)
            self.log_signal.emit("Results sent.")

        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()

# ---------- GRAPHICAL USER INTERFACE (GUI) ----------

class MplCanvas(FigureCanvas):
    """Matplotlib Canvas integrated into Qt"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.subplots(2, 3)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sony Dark Frame Analyzer (Mode 2)")
        self.resize(1400, 900) # Increased size for more plots

        # Main Widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- Toolbar ---
        toolbar_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("Load RAW (.ARW)")
        self.btn_load.clicked.connect(self.load_file)
        self.btn_load.setStyleSheet("padding: 8px; font-weight: bold;")
        
        self.lbl_crop = QLabel("Crop Size:")
        self.spin_crop = QSpinBox()
        self.spin_crop.setRange(256, 4096)
        self.spin_crop.setValue(1024)
        self.spin_crop.setSingleStep(128)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0) # Indeterminate (loading animation)
        self.progress.hide()

        toolbar_layout.addWidget(self.btn_load)
        toolbar_layout.addSpacing(20)
        toolbar_layout.addWidget(self.lbl_crop)
        toolbar_layout.addWidget(self.spin_crop)
        toolbar_layout.addWidget(self.progress)
        toolbar_layout.addStretch()

        layout.addLayout(toolbar_layout)

        # --- Content Area (Vertical Splitter) ---
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 1. Graphs
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        # Initialize with clean empty axes
        self.clear_plots()
        splitter.addWidget(self.canvas)

        # 2. Log Console
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace;")
        splitter.addWidget(self.console)
        
        # Space distribution (70% graph, 30% logs)
        splitter.setSizes([700, 200])
        
        layout.addWidget(splitter)

        self.log("Welcome. Load a RAW file (Dark Frame / Lens Cap On) to start.")

    def log(self, message):
        self.console.append(message)
        # Auto-scroll to bottom
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_plots(self):
        for ax in self.canvas.axes.flat:
            ax.clear()
            ax.axis('off')
        
        self.canvas.axes[0, 0].set_title("Image (Waiting)")
        self.canvas.axes[0, 1].set_title("Histogram (Waiting)")
        self.canvas.axes[0, 2].set_title("Stats (Waiting)")
        self.canvas.axes[1, 0].set_title("Autocorr (Waiting)")
        self.canvas.axes[1, 1].set_title("Spectrum (Waiting)")
        self.canvas.axes[1, 2].set_title("Radial PSD (Waiting)")
        self.canvas.draw()

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open RAW File", "", "Sony RAW (*.ARW *.arw)")
        if fname:
            self.start_analysis(fname)

    def start_analysis(self, filepath):
        self.btn_load.setEnabled(False)
        self.progress.show()
        self.console.clear()
        self.clear_plots()
        
        crop_size = self.spin_crop.value()
        
        # Create and start Worker
        self.worker = AnalysisWorker(filepath, crop_size)
        self.worker.log_signal.connect(self.log)
        self.worker.error_signal.connect(self.handle_error)
        self.worker.result_signal.connect(self.update_plots)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.start()

    def handle_error(self, msg):
        QMessageBox.critical(self, "Error", f"An error occurred:\n{msg}")
        self.log(f"ERROR: {msg}")

    def analysis_finished(self):
        self.btn_load.setEnabled(True)
        self.progress.hide()

    def update_plots(self, res):
        noise = res['noise_img']
        R = res['autocorr']
        S = res['spectrum']
        var = res['var']
        rad = res['radial_psd']
        kurt = res['kurtosis']
        skw = res['skewness']
        max_corr = res['max_corr']

        axs = self.canvas.axes
        
        # 1. Noise Image (Top Left)
        sigma = np.sqrt(var)
        im0 = axs[0, 0].imshow(noise, cmap='gray', vmin=-3*sigma, vmax=3*sigma)
        axs[0, 0].set_title("Cleaned Noise")
        axs[0, 0].axis('off')

        # 2. Histogram (Top Middle)
        axs[0, 1].clear()
        axs[0, 1].hist(noise.ravel(), bins=100, density=True, alpha=0.6, color='g')
        # Gaussian fit
        mu, std = norm.fit(noise.ravel())
        xmin, xmax = axs[0, 1].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        axs[0, 1].plot(x, p, 'k', linewidth=2)
        axs[0, 1].set_title(f"Hist (K={kurt:.2f}, S={skw:.2f})")
        axs[0, 1].axis('on')

        # 3. Stats Text (Top Right)
        axs[0, 2].clear()
        axs[0, 2].axis('off')
        text_str = (f"Variance: {var:.2f}\n"
                    f"Sigma: {sigma:.2f}\n"
                    f"Kurtosis: {kurt:.4f}\n"
                    f"Skewness: {skw:.4f}\n"
                    f"Max Corr: {max_corr:.4f}")
        axs[0, 2].text(0.1, 0.5, text_str, fontsize=12, va='center')
        axs[0, 2].set_title("Statistics")

        # 4. Autocorrelation (Bottom Left)
        h, w = R.shape
        zoom = 32
        cx, cy = w//2, h//2
        R_zoom = R[cy-zoom:cy+zoom, cx-zoom:cx+zoom]
        im1 = axs[1, 0].imshow(R_zoom, cmap='viridis', extent=[-zoom, zoom, zoom, -zoom])
        axs[1, 0].set_title("Autocorrelation")
        
        # 5. Spectrum (Bottom Middle)
        S_shift = np.fft.fftshift(S)
        S_log = np.log10(S_shift + 1e-12)
        im2 = axs[1, 1].imshow(S_log, cmap='inferno')
        axs[1, 1].set_title("Power Spectrum")
        axs[1, 1].axis('off')

        # 6. Radial PSD (Bottom Right)
        axs[1, 2].clear()
        axs[1, 2].semilogy(rad)
        axs[1, 2].set_title("Radial PSD")
        axs[1, 2].grid(True, which="both", ls="-", alpha=0.5)
        axs[1, 2].axis('on')

        self.canvas.fig.suptitle(f"Result: {res['verdict']}", fontsize=12, color='red' if "PROBABLE" in res['verdict'] else 'green')
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
