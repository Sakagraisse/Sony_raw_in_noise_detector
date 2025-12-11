import sys
import os

# Fix potentiel pour les blocages numpy/fft dans les QThreads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import rawpy
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTextEdit, QFileDialog, 
                             QLabel, QSpinBox, QSplitter, QProgressBar, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Configuration de Matplotlib pour PyQt
matplotlib.use('QtAgg')

# ---------- LOGIQUE MÉTIER (MATHS) ----------

def center_crop(arr2d, size):
    h, w = arr2d.shape
    size = min(size, h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return arr2d[y0:y0+size, x0:x0+size]

def clean_dark_frame_pattern(img):
    """
    Nettoie le pattern fixe d'un Dark Frame (Bayer offsets + Banding).
    """
    img = img.astype(np.float64)
    
    # 1. Normalisation du Black Level par phase de Bayer (2x2)
    for y in range(2):
        for x in range(2):
            sub = img[y::2, x::2]
            img[y::2, x::2] -= np.mean(sub)

    # 2. Destriping (Retrait moyenne ligne/colonne)
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

def guess_noise_reduction(rho_x, rho_y, threshold=0.05):
    max_corr = max(abs(rho_x), abs(rho_y))
    if max_corr > threshold:
        verdict = "NR PROBABLE (Corr > Seuil)"
    else:
        verdict = "NR PEU PROBABLE (Bruit blanc)"
    return verdict, max_corr

# ---------- WORKER THREAD (POUR NE PAS BLOQUER L'UI) ----------

class AnalysisWorker(QThread):
    # Signaux pour communiquer avec l'interface
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(object) # Envoie un dictionnaire de résultats
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, filepath, crop_size):
        super().__init__()
        self.filepath = filepath
        self.crop_size = crop_size

    def run(self):
        try:
            self.log_signal.emit(f"Chargement du fichier : {os.path.basename(self.filepath)}...")
            
            # Lecture RAW
            raw = rawpy.imread(self.filepath)
            full_raw = raw.raw_image_visible
            h, w = full_raw.shape
            self.log_signal.emit(f"Dimensions RAW : {w}x{h}")

            # Crop
            self.log_signal.emit(f"Découpe du crop central ({self.crop_size}x{self.crop_size})...")
            full_crop = center_crop(full_raw, self.crop_size)

            # Nettoyage
            self.log_signal.emit("Nettoyage du Dark Frame (Black Level + Destriping)...")
            clean_noise = clean_dark_frame_pattern(full_crop)

            # Calculs stats
            self.log_signal.emit("Calcul des statistiques de corrélation...")
            var_clean = np.mean(clean_noise**2)
            rho_x, rho_y = compute_lag1_correlations(clean_noise)
            verdict, max_corr = guess_noise_reduction(rho_x, rho_y)

            self.log_signal.emit("-" * 40)
            self.log_signal.emit(f"Variance Bruit : {var_clean:.3f}")
            self.log_signal.emit(f"Corr. Horizontale (rho_x) : {rho_x:.4f}")
            self.log_signal.emit(f"Corr. Verticale   (rho_y) : {rho_y:.4f}")
            self.log_signal.emit(f"Max Corr : {max_corr:.4f}")
            self.log_signal.emit(f"VERDICT : {verdict}")
            self.log_signal.emit("-" * 40)

            # FFT
            self.log_signal.emit("Calcul FFT et Autocorrélation 2D...")
            R_cl, S_cl = compute_autocorr_2d(clean_noise)
            self.log_signal.emit("FFT terminée. Préparation des résultats...")

            # Packaging des résultats
            results = {
                "noise_img": clean_noise,
                "autocorr": R_cl,
                "spectrum": S_cl,
                "var": var_clean,
                "verdict": verdict
            }
            self.log_signal.emit("Envoi des résultats au thread principal...")
            self.result_signal.emit(results)
            self.log_signal.emit("Résultats envoyés.")

        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()

# ---------- INTERFACE GRAPHIQUE (GUI) ----------

class MplCanvas(FigureCanvas):
    """Canvas Matplotlib intégré à Qt"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.subplots(1, 3)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sony Dark Frame Analyzer (Mode 2)")
        self.resize(1200, 800)

        # Widget principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- Barre d'outils ---
        toolbar_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("Charger RAW (.ARW)")
        self.btn_load.clicked.connect(self.load_file)
        self.btn_load.setStyleSheet("padding: 8px; font-weight: bold;")
        
        self.lbl_crop = QLabel("Taille Crop :")
        self.spin_crop = QSpinBox()
        self.spin_crop.setRange(256, 4096)
        self.spin_crop.setValue(1024)
        self.spin_crop.setSingleStep(128)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0) # Indéterminé (animation de chargement)
        self.progress.hide()

        toolbar_layout.addWidget(self.btn_load)
        toolbar_layout.addSpacing(20)
        toolbar_layout.addWidget(self.lbl_crop)
        toolbar_layout.addWidget(self.spin_crop)
        toolbar_layout.addWidget(self.progress)
        toolbar_layout.addStretch()

        layout.addLayout(toolbar_layout)

        # --- Zone de contenu (Splitter Vertical) ---
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 1. Graphiques
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        # Initialiser avec des axes vides propres
        self.clear_plots()
        splitter.addWidget(self.canvas)

        # 2. Console de logs
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace;")
        splitter.addWidget(self.console)
        
        # Répartition de l'espace (70% graph, 30% logs)
        splitter.setSizes([600, 200])
        
        layout.addWidget(splitter)

        self.log("Bienvenue. Chargez un fichier RAW (Dark Frame / Capuchon mis) pour commencer.")

    def log(self, message):
        self.console.append(message)
        # Scroll automatique vers le bas
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_plots(self):
        for ax in self.canvas.axes:
            ax.clear()
            ax.axis('off')
        self.canvas.axes[0].set_title("Image (En attente)")
        self.canvas.axes[1].set_title("Autocorr (En attente)")
        self.canvas.axes[2].set_title("Spectre (En attente)")
        self.canvas.draw()

    def load_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Ouvrir un fichier RAW", "", "Sony RAW (*.ARW *.arw)")
        if fname:
            self.start_analysis(fname)

    def start_analysis(self, filepath):
        self.btn_load.setEnabled(False)
        self.progress.show()
        self.console.clear()
        self.clear_plots()
        
        crop_size = self.spin_crop.value()
        
        # Création et lancement du Worker
        self.worker = AnalysisWorker(filepath, crop_size)
        self.worker.log_signal.connect(self.log)
        self.worker.error_signal.connect(self.handle_error)
        self.worker.result_signal.connect(self.update_plots)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.start()

    def handle_error(self, msg):
        QMessageBox.critical(self, "Erreur", f"Une erreur est survenue :\n{msg}")
        self.log(f"ERREUR : {msg}")

    def analysis_finished(self):
        self.btn_load.setEnabled(True)
        self.progress.hide()

    def update_plots(self, res):
        noise = res['noise_img']
        R = res['autocorr']
        S = res['spectrum']
        var = res['var']

        axs = self.canvas.axes
        
        # 1. Image Bruit
        # On sature l'affichage à +/- 3 sigma pour bien voir le grain
        sigma = np.sqrt(var)
        im0 = axs[0].imshow(noise, cmap='gray', vmin=-3*sigma, vmax=3*sigma)
        axs[0].set_title("Bruit Nettoyé (Destriped)")
        axs[0].axis('off')

        # 2. Autocorrélation
        # On zoome un peu sur le centre pour voir la forme du pic
        h, w = R.shape
        zoom = 32 # Rayon autour du centre
        cx, cy = w//2, h//2
        R_zoom = R[cy-zoom:cy+zoom, cx-zoom:cx+zoom]
        
        im1 = axs[1].imshow(R_zoom, cmap='viridis', extent=[-zoom, zoom, zoom, -zoom])
        axs[1].set_title("Autocorrélation (Zoom Centre)")
        # axs[1].axis('off') # On garde les axes pour voir l'échelle en pixels

        # 3. Spectre
        S_shift = np.fft.fftshift(S)
        S_log = np.log10(S_shift + 1e-12)
        im2 = axs[2].imshow(S_log, cmap='inferno')
        axs[2].set_title("Spectre de Puissance (Log)")
        axs[2].axis('off')

        self.canvas.fig.suptitle(f"Résultat : {res['verdict']}", fontsize=12, color='red' if "PROBABLE" in res['verdict'] else 'green')
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
