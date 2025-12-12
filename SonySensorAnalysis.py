#!/usr/bin/env python3
"""rectify_gui.py

Simple PyQt6 GUI scaffold for grid rectification and analysis.
Tabs:
 - Load Files: a simple file selector and list
 - Temp: placeholder for future controls
 - View Graph: nested tabs Graph 1..4 to show plots/diagnostics

This file is purposely lightweight: it's an initial scaffold to build upon.
"""

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QFileDialog, QSizePolicy, QPlainTextEdit, QLineEdit,
    QSplitter, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QBrush
import sys
import os
import json
import numpy as np
import cv2

# Import geometry helpers from step2
try:
    import step2_rectify
except ImportError:
    step2_rectify = None

try:
    # Matplotlib optional; if missing, the GUI still runs with placeholders
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


class InteractiveImageWidget(QWidget):
    cornersChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None # QImage
        self.pixmap = None
        self.image_path = None
        self.corners = [] # List of QPointF (in image coordinates)
        self.dragging_idx = -1
        self.hover_idx = -1
        self.setMouseTracking(True)
        self.scale_factor = 1.0
        self.offset = QPointF(0, 0)
        
        # Pre-compute theoretical geometry for preview
        if step2_rectify:
            self.dst_markers, (W, H) = step2_rectify.get_theoretical_markers(3840, 2160)
            centers, _ = step2_rectify.compute_patch_grid(W, H, cols=11, rows=7)
            self.dst_grid_centers = centers.reshape(-1, 2) # (77, 2)
            self.grid_rows = 7
            self.grid_cols = 11
        else:
            self.dst_markers = None

    def set_image(self, image_path):
        self.image_path = image_path
        if not image_path or not os.path.exists(image_path):
            self.pixmap = None
            self.update()
            return

        # Load image using cv2 to apply auto-exposure for display
        img = cv2.imread(image_path)
        if img is None:
            self.pixmap = None
            self.update()
            return
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Auto-exposure (Simple normalization)
        # Map 1st percentile to 0 and 99th to 255 to stretch contrast
        try:
            vmin, vmax = np.percentile(img, (1, 99))
            if vmax > vmin:
                img = np.clip((img.astype(np.float32) - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        except Exception:
            pass # Fallback to original if percentile fails
            
        h, w, ch = img.shape
        bytes_per_line = ch * w
        self.image = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        if self.image.isNull():
            self.pixmap = None
        else:
            self.pixmap = QPixmap.fromImage(self.image)
        self.update()

    def set_corners(self, corners):
        # corners: list of [x, y] or QPointF
        self.corners = [QPointF(c[0], c[1]) for c in corners]
        self.update()

    def get_corners(self):
        return [[p.x(), p.y()] for p in self.corners]

    def _map_to_image(self, widget_pos):
        return (widget_pos - self.offset) / self.scale_factor

    def _map_from_image(self, image_pos):
        return image_pos * self.scale_factor + self.offset

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if not self.pixmap:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Image")
            return

        # Calculate scale and offset to fit image
        w_widget = self.width()
        h_widget = self.height()
        w_img = self.pixmap.width()
        h_img = self.pixmap.height()

        scale_w = w_widget / w_img
        scale_h = h_widget / h_img
        self.scale_factor = min(scale_w, scale_h) * 0.95 # 5% margin

        disp_w = w_img * self.scale_factor
        disp_h = h_img * self.scale_factor
        
        self.offset = QPointF((w_widget - disp_w) / 2, (h_widget - disp_h) / 2)

        # Draw Image
        target_rect = QRectF(self.offset.x(), self.offset.y(), disp_w, disp_h)
        painter.drawPixmap(target_rect.toRect(), self.pixmap)

        # Draw Corners and Grid
        if len(self.corners) == 4:
            # Draw Grid Preview (Dynamic)
            if self.dst_markers is not None:
                src_markers = np.array([[p.x(), p.y()] for p in self.corners], dtype=np.float32)
                try:
                    H, _ = cv2.findHomography(src_markers, self.dst_markers)
                    H_inv = np.linalg.inv(H)
                    
                    # Transform grid centers
                    dst_pts = self.dst_grid_centers.reshape(-1, 1, 2).astype(np.float32)
                    src_grid = cv2.perspectiveTransform(dst_pts, H_inv).reshape(self.grid_rows, self.grid_cols, 2)
                    
                    # Draw grid lines
                    painter.setPen(QPen(QColor(255, 255, 0, 128), 1)) # Yellow, semi-transparent
                    
                    # Draw rows
                    for r in range(self.grid_rows):
                        pts = []
                        for c in range(self.grid_cols):
                            pt = QPointF(src_grid[r, c, 0], src_grid[r, c, 1])
                            pts.append(self._map_from_image(pt))
                        for i in range(len(pts)-1):
                            painter.drawLine(pts[i], pts[i+1])
                            
                    # Draw cols
                    for c in range(self.grid_cols):
                        pts = []
                        for r in range(self.grid_rows):
                            pt = QPointF(src_grid[r, c, 0], src_grid[r, c, 1])
                            pts.append(self._map_from_image(pt))
                        for i in range(len(pts)-1):
                            painter.drawLine(pts[i], pts[i+1])
                            
                    # Draw centers as small dots
                    painter.setBrush(QBrush(QColor(255, 255, 0, 128)))
                    painter.setPen(Qt.PenStyle.NoPen)
                    for r in range(self.grid_rows):
                        for c in range(self.grid_cols):
                            pt = QPointF(src_grid[r, c, 0], src_grid[r, c, 1])
                            painter.drawEllipse(self._map_from_image(pt), 2, 2)

                except Exception:
                    pass

            # Map corners to widget coords
            pts = [self._map_from_image(c) for c in self.corners]
            
            # Draw perimeter
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
            path = [pts[0], pts[1], pts[2], pts[3], pts[0]]
            for i in range(4):
                painter.drawLine(path[i], path[i+1])

            # Draw handles
            for i, pt in enumerate(pts):
                color = QColor(255, 0, 0) if i == self.hover_idx or i == self.dragging_idx else QColor(0, 255, 0)
                painter.setBrush(QBrush(color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(pt, 8, 8)
                
                # Label corners (TL, TR, BR, BL)
                painter.setPen(Qt.GlobalColor.white)
                labels = ["TL", "TR", "BR", "BL"]
                painter.drawText(pt + QPointF(10, 10), labels[i])

    def mousePressEvent(self, event):
        if not self.corners:
            return
        pos = event.position()
        # Check if near a corner
        min_dist = 20 # pixels tolerance
        best_idx = -1
        
        for i, c in enumerate(self.corners):
            widget_pt = self._map_from_image(c)
            dist = (widget_pt - pos).manhattanLength()
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        
        if best_idx != -1:
            self.dragging_idx = best_idx
            self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        
        if self.dragging_idx != -1:
            # Update corner position
            img_pos = self._map_to_image(pos)
            # Clamp to image bounds? Maybe not strictly necessary but good
            if self.pixmap:
                x = max(0, min(img_pos.x(), self.pixmap.width()))
                y = max(0, min(img_pos.y(), self.pixmap.height()))
                self.corners[self.dragging_idx] = QPointF(x, y)
                self.cornersChanged.emit()
                self.update()
        else:
            # Hover effect
            min_dist = 20
            best_idx = -1
            for i, c in enumerate(self.corners):
                widget_pt = self._map_from_image(c)
                dist = (widget_pt - pos).manhattanLength()
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            
            if best_idx != self.hover_idx:
                self.hover_idx = best_idx
                self.update()

    def mouseReleaseEvent(self, event):
        self.dragging_idx = -1
        self.update()


class ManualFitTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        
        # Left: List of files
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self._on_file_selected)
        left_layout.addWidget(QLabel("Chart Files:"))
        left_layout.addWidget(self.file_list)
        
        self.btn_refresh = QPushButton("Refresh List")
        self.btn_refresh.clicked.connect(self.refresh_list)
        left_layout.addWidget(self.btn_refresh)
        
        self.btn_update = QPushButton("Update Photo & Recalculate")
        self.btn_update.clicked.connect(self._save_and_update)
        left_layout.addWidget(self.btn_update)
        
        # Right: Interactive Image
        self.image_widget = InteractiveImageWidget()
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.image_widget)
        splitter.setStretchFactor(1, 1)
        
        layout.addWidget(splitter)
        
        self.current_json_path = None
        self.current_chart_path = None
        self.output_dir = None

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
        self.refresh_list()

    def refresh_list(self):
        self.file_list.clear()
        if not self.output_dir or not os.path.exists(self.output_dir):
            return
            
        # Look for subfolders in output_dir that contain .grid.json
        try:
            items = sorted(os.listdir(self.output_dir))
            for item in items:
                full_path = os.path.join(self.output_dir, item)
                if os.path.isdir(full_path):
                    # Check if it looks like a chart folder
                    json_path = os.path.join(full_path, item + ".grid.json")
                    if os.path.exists(json_path):
                        self.file_list.addItem(item)
        except Exception as e:
            print(f"Error listing files: {e}")

    def _on_file_selected(self, item):
        folder_name = item.text()
        folder_path = os.path.join(self.output_dir, folder_name)
        
        # Try to find a preview image
        # Priority: source_preview.jpg (clean) -> debug_grid.jpg (old default) -> rectified_overlay.jpg
        preview_img = os.path.join(folder_path, folder_name + ".source_preview.jpg")
        if not os.path.exists(preview_img):
            preview_img = os.path.join(folder_path, folder_name + ".debug_grid.jpg")
        if not os.path.exists(preview_img):
            preview_img = os.path.join(folder_path, folder_name + ".rectified_overlay.jpg")
            
        if os.path.exists(preview_img):
            self.image_widget.set_image(preview_img)
            self.current_chart_path = preview_img
        else:
            print("No preview image found.")
            self.image_widget.set_image(None)
            
        # Load JSON
        self.current_json_path = os.path.join(folder_path, folder_name + ".grid.json")
        if os.path.exists(self.current_json_path):
            with open(self.current_json_path, 'r') as f:
                data = json.load(f)
                
            # Check if we have explicitly saved markers (from a previous manual fit)
            markers = data.get('markers')
            if markers:
                self.image_widget.set_corners(markers)
            else:
                # We only have 'homography_corners' which are likely the GRID corners from auto-detection
                grid_corners = data.get('homography_corners')
                if grid_corners and step2_rectify:
                    # Convert Grid Corners -> Markers
                    # 1. Compute H that maps src_grid_corners -> dst_grid_corners
                    src_grid = np.array(grid_corners, dtype=np.float32)
                    dst_grid, (W, H_dim) = step2_rectify.get_theoretical_patch_centers(3840, 2160)
                    
                    H_mat, _ = cv2.findHomography(src_grid, dst_grid)
                    
                    # 2. Get theoretical markers
                    dst_markers, _ = step2_rectify.get_theoretical_markers(3840, 2160)
                    
                    # 3. Map dst_markers -> src_markers using inv(H)
                    try:
                        H_inv = np.linalg.inv(H_mat)
                        dst_markers_reshaped = dst_markers.reshape(-1, 1, 2)
                        src_markers = cv2.perspectiveTransform(dst_markers_reshaped, H_inv)
                        src_markers = src_markers.reshape(-1, 2).tolist()
                        self.image_widget.set_corners(src_markers)
                    except Exception as e:
                        print(f"Error projecting markers: {e}")
                        self.image_widget.set_corners(grid_corners) # Fallback
                else:
                    self._set_default_corners()
        else:
            self._set_default_corners()

    def _set_default_corners(self):
        if self.image_widget.pixmap:
            w = self.image_widget.pixmap.width()
            h = self.image_widget.pixmap.height()
            # 10% margin
            tl = [w*0.1, h*0.1]
            tr = [w*0.9, h*0.1]
            br = [w*0.9, h*0.9]
            bl = [w*0.1, h*0.9]
            self.image_widget.set_corners([tl, tr, br, bl])

    def _save_grid(self):
        if not self.current_json_path or not step2_rectify:
            return False
            
        corners = self.image_widget.get_corners()
        src_markers = np.array(corners, dtype=np.float32)
        
        # 1. Compute Homography based on MARKERS
        # Get theoretical markers
        dst_markers, (W_target, H_target) = step2_rectify.get_theoretical_markers(3840, 2160)
        
        H, _ = cv2.findHomography(src_markers, dst_markers)
        
        # 2. Compute all patch centers
        # Get theoretical grid centers
        centers_grid, _ = step2_rectify.compute_patch_grid(W_target, H_target, cols=11, rows=7)
        # centers_grid is (rows, cols, 2)
        
        # Flatten to list of points
        flat_centers = centers_grid.reshape(-1, 2)
        
        # Inverse transform: We want to find where these theoretical centers are in the SOURCE image
        # So we need H_inv = inv(H)
        # src = H_inv * dst
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("Homography inversion failed")
            return False

        # Transform points
        # cv2.perspectiveTransform expects (N, 1, 2)
        dst_pts_reshaped = flat_centers.reshape(-1, 1, 2).astype(np.float32)
        src_centers = cv2.perspectiveTransform(dst_pts_reshaped, H_inv)
        src_centers = src_centers.reshape(-1, 2)
        
        # 3. Update JSON
        try:
            with open(self.current_json_path, 'r') as f:
                data = json.load(f)
        except:
            data = {}
            
        # Save markers so we can reload them correctly next time
        data['markers'] = corners
        # Save centers for analysis
        data['centers'] = src_centers.tolist() 
        
        with open(self.current_json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved updated grid to {self.current_json_path}")
        return True

    def _save_and_update(self):
        if self._save_grid():
            # Trigger regeneration
            if step2_rectify and hasattr(step2_rectify, 'regenerate_artifacts'):
                print("Regenerating artifacts...")
                # We need the original raw file path. It's usually in the JSON or we can infer it.
                # The JSON has 'raw_file' usually?
                try:
                    with open(self.current_json_path, 'r') as f:
                        data = json.load(f)
                    raw_file = data.get('raw_file')
                    if raw_file and os.path.exists(raw_file):
                        chart_output_dir = os.path.dirname(self.current_json_path)
                        step2_rectify.regenerate_artifacts(raw_file, self.current_json_path, chart_output_dir)
                        print("Regeneration complete.")
                        # Reload image to show new overlay if we were showing the overlay
                        # But we are showing the source preview now, so maybe just flash a message?
                        # Or reload the file list item to refresh status?
                        self._on_file_selected(self.file_list.currentItem())
                    else:
                        print(f"Could not find raw file: {raw_file}")
                except Exception as e:
                    print(f"Error during regeneration: {e}")
            else:
                print("Regeneration function not available in step2_rectify")


class GraphWidget(QWidget):
    def __init__(self, title="Graph", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)
        if MATPLOTLIB_AVAILABLE:
            self.fig = Figure(figsize=(4, 3))
            self.canvas = FigureCanvas(self.fig)
            layout.addWidget(self.canvas)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title(title)
            self.ax.plot([0], [0])
        else:
            self.placeholder = QLabel(f"{title} (Matplotlib not available)")
            self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.placeholder)

    def plot_data(self, x, y, xlabel, ylabel, title):
        if not MATPLOTLIB_AVAILABLE:
            return
        self.ax.clear()
        self.ax.plot(x, y, 'o-', linewidth=2)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        
        # Use Log scale for ISO (X-axis)
        if "ISO" in xlabel:
            self.ax.set_xscale('log')
            # Optional: Set ticks to match the actual ISO values present
            self.ax.set_xticks(x)
            self.ax.get_xaxis().set_major_formatter(lambda val, pos: str(int(val)))

        # Use Log2 scale for Read Noise ADU (Y-axis)
        if "Read Noise (ADU)" in title:
            self.ax.set_yscale('log', base=2)
            self.ax.get_yaxis().set_major_formatter(lambda val, pos: str(round(val, 2)))

        self.ax.grid(True, which="both", ls="-", alpha=0.5)
        self.canvas.draw()

    def plot_random(self):
        if not MATPLOTLIB_AVAILABLE:
            return
        self.ax.clear()
        x = np.linspace(0, 10, 200)
        y = np.sin(x * (1 + np.random.rand()))
        self.ax.plot(x, y)
        self.ax.set_title(self.ax.get_title())
        self.canvas.draw()


class RectifyGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rectify / Grid Tools - GUI")
        self.resize(1100, 700)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Load Files tab
        self.tab_load = QWidget()
        self.tabs.addTab(self.tab_load, "Load Files")
        self._build_load_tab()

        # Manual Fit tab
        self.tab_manual = ManualFitTab()
        self.tabs.addTab(self.tab_manual, "Manual Fit")

        # View Graph tab with nested tabs
        self.tab_view = QWidget()
        self.tabs.addTab(self.tab_view, "View Graph")
        self._build_view_tab()

    def _build_load_tab(self):
        layout = QVBoxLayout(self.tab_load)
        btn_box = QHBoxLayout()
        btn_choose = QPushButton("Choose Folder")
        btn_choose.clicked.connect(self._choose_folder)
        btn_out = QPushButton("Choose Output Folder")
        btn_out.clicked.connect(self._choose_output_folder)
        btn_process = QPushButton("1. Sort")
        btn_process.clicked.connect(self._process_files)
        btn_rectify = QPushButton("2. Rectify")
        btn_rectify.clicked.connect(self._run_rectify)
        btn_analyze = QPushButton("3. Analyze")
        btn_analyze.clicked.connect(self._run_analysis)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_files)
        btn_box.addWidget(btn_choose)
        btn_box.addWidget(btn_out)
        btn_box.addWidget(btn_process)
        btn_box.addWidget(btn_rectify)
        btn_box.addWidget(btn_analyze)
        btn_box.addWidget(btn_clear)
        layout.addLayout(btn_box)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.file_list.itemDoubleClicked.connect(self._on_file_double_clicked)
        layout.addWidget(self.file_list)
        # Project name input
        project_box = QHBoxLayout()
        project_box.addWidget(QLabel("Project Name:"))
        self.project_input = QLineEdit()
        self.project_input.setText('project')
        project_box.addWidget(self.project_input)
        layout.addLayout(project_box)
        # output folder label
        self.output_dir_label = QLabel("Output: (not set)")
        layout.addWidget(self.output_dir_label)

        # log area
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumBlockCount(2000)
        layout.addWidget(self.log_text)

    def _build_view_tab(self):
        layout = QVBoxLayout(self.tab_view)
        
        # Status Label for Warnings (Black Level, etc.)
        self.status_label = QLabel("Status: Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-weight: bold; color: gray; border: 1px solid gray; padding: 5px; margin-bottom: 5px;")
        layout.addWidget(self.status_label)
        
        self.view_tabs = QTabWidget()
        layout.addWidget(self.view_tabs)
        self.graph_widgets = []
        titles = ["PDR", "RN (e-)", "RN (ADU)", "Gain (e-/ADU)"]
        for title in titles:
            gw = GraphWidget(title=title)
            self.graph_widgets.append(gw)
            self.view_tabs.addTab(gw, title)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select files to load", os.getcwd(), "*.dng *.ARW *.jpg *.tiff *.png")
        for p in paths:
            if not any(self.file_list.item(i).text() == p for i in range(self.file_list.count())):
                self.file_list.addItem(p)

    def _clear_files(self):
        self.file_list.clear()

    def _on_file_double_clicked(self, item):
        path = item.text()
        # Right now, when a file is double-clicked, sample-plot into Graph 1
        # In future, it can trigger full processing + diagnostic plots
        if self.graph_widgets:
            self.graph_widgets[0].plot_random()

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder to load files", os.getcwd())
        if not folder:
            return
        # store selected folder
        self.source_folder = folder
        # find supported files
        supported = ('.dng', '.arw', '.ARW', '.nef', '.cr2', '.tiff', '.jpg', '.jpeg', '.png')
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(supported):
                    path = os.path.join(root, f)
                    if not any(self.file_list.item(i).text() == path for i in range(self.file_list.count())):
                        self.file_list.addItem(path)
        
        # Auto-load existing results if present
        res_path = os.path.join(folder, 'analysis_results.json')
        if os.path.exists(res_path):
            self._append_log(f"Found existing analysis_results.json in {folder}. Loading...")
            self._on_analysis_finished(res_path)
        self._append_log(f"Loaded folder: {folder} ({self.file_list.count()} files)")

    def _choose_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", os.getcwd())
        if not folder:
            return
        self.output_dir = folder
        self.output_dir_label.setText(f"Output: {folder}")
        
        # Update Manual Fit tab
        if hasattr(self, 'tab_manual'):
            self.tab_manual.set_output_dir(folder)
            
        # Auto-load existing results if present
        res_path = os.path.join(folder, 'analysis_results.json')
        if os.path.exists(res_path):
            self._append_log(f"Found existing analysis_results.json in {folder}. Loading...")
            self._on_analysis_finished(res_path)

    def _append_log(self, txt: str):
        self.log_text.appendPlainText(txt)

    def _process_files(self):
        # 1. Sort / Prepare
        if not hasattr(self, 'source_folder') or self.source_folder is None:
            self._append_log("No source folder selected. Use 'Choose Folder' first.")
            return
        output_dir = getattr(self, 'output_dir', None)
        if output_dir is None:
            self._append_log("Output folder not set. Please choose it first.")
            return
        
        # Prepare outputs to output_dir/sorted
        sorted_dir = os.path.join(output_dir, 'sorted')
        project_name = self.project_input.text().strip() or 'project'
        
        self.prepare_worker = PrepareWorker(self.source_folder, project_name, sorted_dir)
        self.prepare_worker.progress.connect(self._append_log)
        self.prepare_worker.finished.connect(lambda: self._append_log(f"Sorting finished. Files in {sorted_dir}"))
        self.prepare_worker.start()

    def _run_rectify(self):
        # 2. Rectify
        output_dir = getattr(self, 'output_dir', None)
        if output_dir is None:
            self._append_log("Output folder not set.")
            return
            
        sorted_dir = os.path.join(output_dir, 'sorted')
        if not os.path.exists(sorted_dir):
            self._append_log(f"Sorted folder not found: {sorted_dir}. Run 'Sort' first.")
            return

        # Rectify outputs to output_dir (it creates subfolders per file)
        self.rectify_worker = RectifyWorker(sorted_dir, output_dir)
        self.rectify_worker.progress.connect(self._append_log)
        self.rectify_worker.finished.connect(lambda: self._append_log("Rectification finished."))
        self.rectify_worker.start()

    def _run_analysis(self):
        # 3. Analyze
        output_dir = getattr(self, 'output_dir', None)
        if output_dir is None:
            self._append_log("Output folder not set.")
            return
            
        sorted_dir = os.path.join(output_dir, 'sorted')
        if not os.path.exists(sorted_dir):
            self._append_log(f"Sorted folder not found: {sorted_dir}. Run 'Sort' first.")
            return
            
        self.analysis_worker = AnalysisWorker(sorted_dir, output_dir)
        self.analysis_worker.progress.connect(self._append_log)
        self.analysis_worker.finished.connect(self._on_analysis_finished)
        self.analysis_worker.start()

    def _on_analysis_finished(self, results_path):
        self._append_log(f"Analysis finished. Results in {results_path}")
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    data = json.load(f)
                
                isos = [d['iso'] for d in data]
                pdr_print = [d.get('pdr_print', 0) for d in data]
                rn_e = [d['rn_e'] for d in data]
                rn_adu = [d['rn_adu'] for d in data]
                gain = [d['gain'] for d in data]
                
                # Sort by ISO just in case
                sorted_indices = np.argsort(isos)
                isos = np.array(isos)[sorted_indices]
                pdr_print = np.array(pdr_print)[sorted_indices]
                rn_e = np.array(rn_e)[sorted_indices]
                rn_adu = np.array(rn_adu)[sorted_indices]
                gain = np.array(gain)[sorted_indices]
                
                # Check for warnings
                warnings = []
                for d in data:
                    if d.get('black_level_warning'):
                        warnings.append(f"ISO {d['iso']}: {d['black_level_warning']}")
                
                if warnings:
                    warn_text = "WARNING: Black Level Mismatch detected!\n" + "\n".join(warnings)
                    self.status_label.setText(warn_text)
                    self.status_label.setStyleSheet("font-weight: bold; color: white; background-color: #d9534f; border: 2px solid darkred; padding: 10px;")
                else:
                    self.status_label.setText("Status: Analysis OK (Black Levels match Metadata)")
                    self.status_label.setStyleSheet("font-weight: bold; color: white; background-color: #5cb85c; border: 2px solid darkgreen; padding: 10px;")

                if self.graph_widgets:
                    # Plot Print PDR (Normalized to 8MP)
                    self.graph_widgets[0].plot_data(isos, pdr_print, "ISO", "PDR (EV)", "Photographic Dynamic Range (Print Normalized)")
                    self.graph_widgets[1].plot_data(isos, rn_e, "ISO", "Read Noise (e-)", "Read Noise (electrons)")
                    self.graph_widgets[2].plot_data(isos, rn_adu, "ISO", "Read Noise (ADU)", "Read Noise (ADU)")
                    self.graph_widgets[3].plot_data(isos, gain, "ISO", "Gain (e-/ADU)", "Gain")
                    
                self.tabs.setCurrentWidget(self.tab_view)
                
            except Exception as e:
                self._append_log(f"Error loading results: {e}")


class RectifyWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, sorted_folder, output_folder):
        super().__init__()
        self.sorted_folder = sorted_folder
        self.output_folder = output_folder

    def run(self):
        import subprocess, sys, glob
        # Find all chart files in sorted folder
        pattern = os.path.join(self.sorted_folder, '*_chart.dng')
        files = glob.glob(pattern)
        if not files:
            self.progress.emit(f"No chart files found in {self.sorted_folder}")
            self.finished.emit()
            return

        for i, f in enumerate(files):
            self.progress.emit(f"Rectifying {i+1}/{len(files)}: {os.path.basename(f)}")
            cmd = [sys.executable, os.path.join(os.getcwd(), 'step2_rectify.py'), f, '--output', self.output_folder]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.stdout:
                    for line in proc.stdout.splitlines():
                        self.progress.emit(line)
                if proc.stderr:
                    for line in proc.stderr.splitlines():
                        self.progress.emit('ERR: ' + line)
            except Exception as e:
                self.progress.emit(f"Error: {e}")
        self.finished.emit()


class PrepareWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, input_folder, project_name, output_dir):
        super().__init__()
        self.input_folder = input_folder
        self.project_name = project_name
        self.output_dir = output_dir

    def run(self):
        import subprocess, sys
        cmd = [sys.executable, os.path.join(os.getcwd(), 'step1_sort.py'), self.input_folder, '--project', self.project_name, '--output', self.output_dir]
        self.progress.emit('Running: ' + ' '.join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.stdout:
                for line in proc.stdout.splitlines():
                    self.progress.emit(line)
            if proc.stderr:
                for line in proc.stderr.splitlines():
                    self.progress.emit('ERR: ' + line)
        except Exception as e:
            self.progress.emit('Error running prepare script: ' + str(e))
        self.finished.emit()


class AnalysisWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str)

    def __init__(self, sorted_folder, output_folder):
        super().__init__()
        self.sorted_folder = sorted_folder
        self.output_folder = output_folder

    def run(self):
        import subprocess, sys
        cmd = [sys.executable, os.path.join(os.getcwd(), 'step3_analyze.py'), '--sorted', self.sorted_folder, '--output', self.output_folder]
        self.progress.emit('Running analysis: ' + ' '.join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.stdout:
                for line in proc.stdout.splitlines():
                    self.progress.emit(line)
            if proc.stderr:
                for line in proc.stderr.splitlines():
                    self.progress.emit('ERR: ' + line)
        except Exception as e:
            self.progress.emit('Error running analysis script: ' + str(e))
        
        results_path = os.path.join(self.output_folder, 'analysis_results.json')
        self.finished.emit(results_path)


def main():
    app = QApplication(sys.argv)
    window = RectifyGUI()
    window.show()
    try:
        sys.exit(app.exec())
    except SystemExit:
        pass

if __name__ == '__main__':
    main()
