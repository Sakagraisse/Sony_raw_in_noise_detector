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
    QPushButton, QLabel, QListWidget, QFileDialog, QSizePolicy, QPlainTextEdit, QLineEdit
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import sys
import os
import numpy as np

try:
    # Matplotlib optional; if missing, the GUI still runs with placeholders
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False


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

        # Temp tab (placeholder)
        self.tab_temp = QWidget()
        self.tabs.addTab(self.tab_temp, "Temp")
        self._build_temp_tab()

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
        btn_process = QPushButton("Process")
        btn_process.clicked.connect(self._process_files)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_files)
        btn_box.addWidget(btn_choose)
        btn_box.addWidget(btn_out)
        btn_box.addWidget(btn_process)
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

    def _build_temp_tab(self):
        layout = QVBoxLayout(self.tab_temp)
        layout.addWidget(QLabel("Temporary controls will be added here"))

    def _build_view_tab(self):
        layout = QVBoxLayout(self.tab_view)
        self.view_tabs = QTabWidget()
        layout.addWidget(self.view_tabs)
        self.graph_widgets = []
        for idx in range(4):
            gw = GraphWidget(title=f"Graph {idx+1}")
            self.graph_widgets.append(gw)
            self.view_tabs.addTab(gw, f"Graph {idx+1}")

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
        self._append_log(f"Loaded folder: {folder} ({self.file_list.count()} files)")

    def _choose_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder", os.getcwd())
        if not folder:
            return
        self.output_dir = folder
        self.output_dir_label.setText(f"Output: {folder}")

    def _append_log(self, txt: str):
        self.log_text.appendPlainText(txt)

    def _process_files(self):
        # For now, processing means preparing pairs + copying to output folder using our script
        # Use source_folder (selected via Choose Folder) as input
        if not hasattr(self, 'source_folder') or self.source_folder is None:
            self._append_log("No source folder selected. Use 'Choose Folder' first.")
            return
        files = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not files:
            self._append_log("No files to process.")
            return
        output_dir = getattr(self, 'output_dir', None)
        if output_dir is None:
            self._append_log("Output folder not set. Please choose it first.")
            return
        project_name = self.project_input.text().strip() or 'project'
        # Use PrepareWorker to run prepare_pairs_and_rename.py on source folder
        self.prepare_worker = PrepareWorker(self.source_folder, project_name, output_dir)
        self.prepare_worker.progress.connect(self._append_log)
        self.prepare_worker.finished.connect(lambda: self._append_log("Prepare finished."))
        self.prepare_worker.start()


class ProcessWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, files, output_dir, cols=11, rows=7, kernel_scale=0.35, iter=1):
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.cols = cols
        self.rows = rows
        self.kernel_scale = kernel_scale
        self.iter = iter

    def run(self):
        import subprocess, sys, shutil
        for i, f in enumerate(self.files):
            self.progress.emit(f"Processing {i+1}/{len(self.files)}: {f}")
            cmd = [sys.executable, os.path.join(os.getcwd(), 'rectify_raw_1d.py'), f, '--cols', str(self.cols), '--rows', str(self.rows), '--kernel-scale', str(self.kernel_scale), '--iter', str(self.iter)]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.stdout:
                    for line in proc.stdout.splitlines():
                        self.progress.emit(line)
                if proc.stderr:
                    for line in proc.stderr.splitlines():
                        self.progress.emit(line)
            except Exception as e:
                self.progress.emit(f"Error: {e}")
            # Copy outputs from script-generated output dir to chosen output_dir if present
            try:
                base_no_ext = os.path.splitext(os.path.basename(f))[0]
                script_out_dir = os.path.join('output', base_no_ext)
                if os.path.exists(script_out_dir):
                    dest = os.path.join(self.output_dir, base_no_ext)
                    if os.path.exists(dest):
                        shutil.rmtree(dest)
                    shutil.copytree(script_out_dir, dest)
                    self.progress.emit(f"Copied outputs to {dest}")
            except Exception as e:
                self.progress.emit(f"Copy outputs failed: {e}")
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
        cmd = [sys.executable, os.path.join(os.getcwd(), 'prepare_pairs_and_rename.py'), self.input_folder, '--project', self.project_name, '--output', self.output_dir]
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
