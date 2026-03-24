from __future__ import annotations

from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from data_manager_signal_loader import DataManager
from file_preview_dialog import FilePreviewDialog
from signal_analysis_utils import Analysis
from signal_data_class import SignalData
from plot_panel_widget import PlotPanel, PlotDataStore


class StftDebugWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("STFT Debug View")
        self.resize(950, 600)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.plot = pg.PlotWidget(title="Demodulated Signal STFT (Full)")
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Frequency", units="Hz")

        self.image_item = pg.ImageItem()
        self.image_item.setOpts(axisOrder="row-major")
        self._stft_lut = pg.colormap.get("inferno").getLookupTable(0.0, 1.0, 256)
        self.image_item.setLookupTable(self._stft_lut)
        self.plot.addItem(self.image_item)
        layout.addWidget(self.plot)

    def clear(self):
        self.image_item.clear()
        self.plot.setTitle("Demodulated Signal STFT (Full)")

    def set_stft(self, times: np.ndarray, freqs: np.ndarray, Z: np.ndarray, f0: float, time_unit: str, time_scale: float):
        if Z.size == 0 or times.size == 0 or freqs.size == 0:
            self.clear()
            return

        mag = np.asarray(Z, dtype=float)
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

        dt = float(np.median(np.diff(times))) if times.size > 1 else 1.0
        df = float(np.median(np.diff(freqs))) if freqs.size > 1 else 1.0

        x0 = float(times[0] / time_scale - dt / (2.0 * time_scale))
        width = float((times[-1] - times[0] + dt) / time_scale)
        y0 = float(freqs[0] - df / 2.0)
        height = float((freqs[-1] - freqs[0] + df))

        self.image_item.setImage(mag_db, autoLevels=True)
        self.image_item.setRect(QtCore.QRectF(x0, y0, width, height))

        self.plot.setLabel("bottom", "Time", units=time_unit)
        self.plot.setLabel("left", "Frequency", units="Hz")
        self.plot.setTitle(f"Demodulated Signal STFT (Full) @ {f0:.3f} Hz")
        self.plot.autoRange()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reusable Signal Visualization App")
        self.resize(1500, 900)

        self._selected_fft_frequency: float | None = None
        self.data: SignalData | None = None
        self.selected_data: SignalData | None = None
        self._stft_color_bar = None
        self._current_time_unit_scale = 1.0
        self._range_initialized = False
        self._debug_enabled = True
        self._stft_debug_window: StftDebugWindow | None = None

        self._build_ui()
        self._connect_signals()
        self.refresh_all_views()
        self.tabs.setCurrentWidget(self.time_plot)
        self._update_sidebar_visibility()

    def _debug(self, message: str):
        if self._debug_enabled:
            print(f"[DEBUG] {message}")

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        self.controls = self._build_controls()
        main_layout.addWidget(self.controls, 0)

        self.tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tabs, 1)

        self.time_plot = PlotPanel("Time Domain")
        self.freq_plot = PlotPanel("Frequency Domain")
        self.demo_plot = PlotPanel("Demodulation")

        self.tabs.addTab(self.time_plot, "Time")
        self.tabs.addTab(self.freq_plot, "Frequency")
        self.tabs.addTab(self.demo_plot, "Demodulation")

        self.statusBar().showMessage("Ready")
        self._build_menu()

        self._update_sidebar_visibility()

        self.setStyleSheet("""
            QPushButton {
                background-color: #d9d9d9;
                border: 1px solid #a6a6a6;
                border-radius: 6px;
                padding: 6px 10px;
            }
            QPushButton:hover {
                background-color: #e6e6e6;
            }
            QPushButton:pressed {
                background-color: #bfbfbf;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
            }
        """)

    def _build_controls(self):
        widget = QtWidgets.QWidget()
        widget.setMaximumWidth(380)
        layout = QtWidgets.QVBoxLayout(widget)

        group_data = QtWidgets.QGroupBox("Data")
        form_data = QtWidgets.QFormLayout(group_data)
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setReadOnly(True)
        self.btn_browse = QtWidgets.QPushButton("Select File...")
        self.btn_load = QtWidgets.QPushButton("Load Selected File")
        self.btn_demo = QtWidgets.QPushButton("Load Demo Signal")
        self.combo_time_column = QtWidgets.QComboBox()
        self.combo_amplitude_column = QtWidgets.QComboBox()
        form_data.addRow(self.path_edit)
        form_data.addRow(self.btn_browse)
        form_data.addRow(self.btn_load)
        form_data.addRow(self.btn_demo)
        form_data.addRow("Time Column", self.combo_time_column)
        form_data.addRow("Amplitude Column", self.combo_amplitude_column)

        group_range = QtWidgets.QGroupBox("Time Range")
        form_range = QtWidgets.QFormLayout(group_range)
        self.spin_start = QtWidgets.QDoubleSpinBox()
        self.spin_end = QtWidgets.QDoubleSpinBox()
        for s in (self.spin_start, self.spin_end):
            s.setDecimals(6)
            s.setRange(-1e12, 1e12)
            s.setSingleStep(0.1)

        self.combo_time_unit = QtWidgets.QComboBox()
        self.combo_time_unit.addItems(["s", "ms", "us"])
        self.combo_time_unit.setCurrentText("s")

        self.btn_apply_range = QtWidgets.QPushButton("Apply Range")
        form_range.addRow("Start", self.spin_start)
        form_range.addRow("End", self.spin_end)
        form_range.addRow("Unit", self.combo_time_unit)
        form_range.addRow(self.btn_apply_range)

        self.group_fft = QtWidgets.QGroupBox("FFT Settings")
        form_fft = QtWidgets.QFormLayout(self.group_fft)
        self.combo_fft_window = QtWidgets.QComboBox()
        self.combo_fft_window.addItems(["hann", "hamming", "blackman", "rectangular"])
        self.check_fft_remove_mean = QtWidgets.QCheckBox("Remove Mean")
        self.check_fft_remove_mean.setChecked(True)
        self.btn_update_fft = QtWidgets.QPushButton("Update FFT")
        form_fft.addRow("Window", self.combo_fft_window)
        form_fft.addRow(self.check_fft_remove_mean)
        form_fft.addRow(self.btn_update_fft)

        self.group_demod = QtWidgets.QGroupBox("Demodulation / STFT Settings")
        form_demod = QtWidgets.QFormLayout(self.group_demod)

        self.spin_demod_frequency = QtWidgets.QDoubleSpinBox()
        self.spin_demod_frequency.setRange(-1e12, 1e12)
        self.spin_demod_frequency.setDecimals(6)
        self.spin_demod_frequency.setSingleStep(1.0)
        self.spin_demod_frequency.setSuffix(" Hz")

        self.btn_use_fft_frequency = QtWidgets.QPushButton("Use Selected FFT Frequency")
        self.btn_update_demod = QtWidgets.QPushButton("Run Demodulation")
        self.lbl_selected_freq = QtWidgets.QLabel("No frequency selected")

        self.combo_stft_window = QtWidgets.QComboBox()
        self.combo_stft_window.addItems(["hann", "hamming", "blackman", "rectangular"])
        self.spin_nperseg = QtWidgets.QSpinBox()
        self.spin_nperseg.setRange(8, 100000)
        self.spin_nperseg.setValue(256)
        self.spin_noverlap = QtWidgets.QSpinBox()
        self.spin_noverlap.setRange(0, 100000)
        self.spin_noverlap.setValue(128)
        self.spin_nfft = QtWidgets.QSpinBox()
        self.spin_nfft.setRange(8, 100000)
        self.spin_nfft.setValue(256)
        self.check_stft_remove_mean = QtWidgets.QCheckBox("Remove Mean")
        self.check_stft_remove_mean.setChecked(True)

        form_demod.addRow("Frequency", self.spin_demod_frequency)
        form_demod.addRow(self.btn_use_fft_frequency)
        form_demod.addRow("STFT Window", self.combo_stft_window)
        form_demod.addRow("N Per Segment", self.spin_nperseg)
        form_demod.addRow("N Overlap", self.spin_noverlap)
        form_demod.addRow("N FFT", self.spin_nfft)
        form_demod.addRow(self.check_stft_remove_mean)
        form_demod.addRow(self.btn_update_demod)
        form_demod.addRow(self.lbl_selected_freq)

        self.group_info = QtWidgets.QGroupBox("Source / Metadata")
        info_layout = QtWidgets.QVBoxLayout(self.group_info)
        self.lbl_info = QtWidgets.QLabel("No data loaded.")
        self.lbl_info.setWordWrap(True)
        info_layout.addWidget(self.lbl_info)

        layout.addWidget(group_data)
        layout.addWidget(group_range)
        layout.addWidget(self.group_fft)
        layout.addWidget(self.group_demod)
        layout.addWidget(self.group_info)
        layout.addStretch(1)
        return widget

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("&File")

        action_open = QtGui.QAction("Open...", self)
        action_open.triggered.connect(self.browse_file)
        file_menu.addAction(action_open)

        action_export_data = QtGui.QAction("Export Data...", self)
        action_export_data.triggered.connect(self.export_data)
        file_menu.addAction(action_export_data)

        action_export_graph = QtGui.QAction("Export Graph...", self)
        action_export_graph.triggered.connect(self.export_graph)
        file_menu.addAction(action_export_graph)

        file_menu.addSeparator()

        action_exit = QtGui.QAction("Exit", self)
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

    def _connect_signals(self):
        self.btn_browse.clicked.connect(self.browse_file)
        self.btn_load.clicked.connect(self.load_selected_file)
        self.btn_demo.clicked.connect(self._load_demo_data)
        self.btn_apply_range.clicked.connect(self.refresh_all_views)

        self.btn_update_fft.clicked.connect(self._update_frequency_plot)
        self.btn_use_fft_frequency.clicked.connect(self._use_selected_fft_frequency)
        self.btn_update_demod.clicked.connect(self._update_demodulation_plot)

        self.combo_time_column.currentIndexChanged.connect(self.refresh_all_views)
        self.combo_amplitude_column.currentIndexChanged.connect(self.refresh_all_views)
        self.combo_time_unit.currentIndexChanged.connect(self._on_time_unit_changed)

        self.tabs.currentChanged.connect(lambda _: self._update_sidebar_visibility())

    def _update_sidebar_visibility(self):
        current = self.tabs.currentWidget()
        self.group_info.setVisible(current == self.time_plot)
        self.group_fft.setVisible(current == self.freq_plot)
        self.group_demod.setVisible(current == self.demo_plot)

    def browse_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select data file",
            "",
            "Data Files (*.csv *.txt *.npy *.npz);;All Files (*)",
        )
        if file_path:
            self.path_edit.setText(file_path)

    def load_selected_file(self):
        file_path = self.path_edit.text().strip()
        if not file_path:
            return
        try:
            suffix = Path(file_path).suffix.lower()
            if suffix in {".csv", ".txt"}:
                preview = DataManager.preview_csv_or_txt(file_path)
                dialog = FilePreviewDialog(preview, self)
                if dialog.exec() != QtWidgets.QDialog.Accepted:
                    return

                self.data = DataManager.load_file(
                    file_path,
                    time_column=dialog.selected_time_column,
                    amplitude_column=dialog.selected_amplitude_column,
                )
                self._sync_column_choices_from_data()
            else:
                self.data = DataManager.load_file(file_path)
                self._sync_column_choices_from_data()

            self._range_initialized = False
            self.refresh_all_views()
            self._debug(f"Loaded data: {self.data is not None}, samples={self.data.n_samples if self.data else 'n/a'}")
            self.tabs.setCurrentWidget(self.time_plot)
            self._update_sidebar_visibility()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Load Error", str(exc))

    def _sync_column_choices_from_data(self):
        self.combo_time_column.blockSignals(True)
        self.combo_amplitude_column.blockSignals(True)

        current_time = self.combo_time_column.currentText().strip()
        current_amp = self.combo_amplitude_column.currentText().strip()

        self.combo_time_column.clear()
        self.combo_amplitude_column.clear()

        if self.data is not None:
            available_columns = []

            if getattr(self.data, "column_names", None):
                tcol, acol = self.data.column_names
                for col in (tcol, acol):
                    if col and col not in available_columns:
                        available_columns.append(col)

            if not available_columns:
                available_columns = ["TIME", "AMPLITUDE"]

            self.combo_time_column.addItems(available_columns)
            self.combo_amplitude_column.addItems(available_columns)

            if current_time in available_columns:
                self.combo_time_column.setCurrentText(current_time)
            else:
                self.combo_time_column.setCurrentIndex(0)

            if current_amp in available_columns:
                self.combo_amplitude_column.setCurrentText(current_amp)
            else:
                self.combo_amplitude_column.setCurrentIndex(1 if len(available_columns) > 1 else 0)

        self.combo_time_column.blockSignals(False)
        self.combo_amplitude_column.blockSignals(False)

    def _time_unit_scale(self) -> float:
        unit = self.combo_time_unit.currentText().strip().lower()
        if unit == "ms":
            return 1e-3
        if unit == "us":
            return 1e-6
        return 1.0

    def _time_unit_label(self) -> str:
        unit = self.combo_time_unit.currentText().strip().lower()
        if unit == "ms":
            return "ms"
        if unit == "us":
            return "us"
        return "s"

    def _convert_time_range_to_unit(self, old_scale: float, new_scale: float):
        if old_scale <= 0 or new_scale <= 0:
            return

        self.spin_start.blockSignals(True)
        self.spin_end.blockSignals(True)
        self.spin_start.setValue(self.spin_start.value() * old_scale / new_scale)
        self.spin_end.setValue(self.spin_end.value() * old_scale / new_scale)
        self.spin_start.blockSignals(False)
        self.spin_end.blockSignals(False)

    def _on_time_unit_changed(self):
        old_scale = getattr(self, "_current_time_unit_scale", 1.0)
        new_scale = self._time_unit_scale()
        self._current_time_unit_scale = new_scale
        self._convert_time_range_to_unit(old_scale, new_scale)
        self.refresh_all_views()

    def _selected_data(self) -> SignalData | None:
        if self.data is None:
            return None

        scale = self._time_unit_scale()
        start_seconds = self.spin_start.value() * scale
        end_seconds = self.spin_end.value() * scale
        return Analysis.select_range(self.data, start_seconds, end_seconds)

    def _load_demo_data(self):
        fs = 1000.0
        t = np.arange(0, 5, 1 / fs)
        y = (
            0.9 * np.sin(2 * np.pi * 12 * t)
            + 0.4 * np.sin(2 * np.pi * 55 * t)
            + 0.2 * np.sin(2 * np.pi * 180 * t)
            + 0.05 * np.random.default_rng(42).normal(size=len(t))
        )
        self.data = SignalData(t, y, source_name="Demo Signal", sampling_rate=fs)
        self._sync_column_choices_from_data()
        self._range_initialized = False
        self.refresh_all_views()
        self.tabs.setCurrentWidget(self.time_plot)
        self.statusBar().showMessage("Demo data loaded.", 3000)

    def refresh_all_views(self):
        if self.data is None:
            self.time_plot.clear()
            self.freq_plot.clear()
            self.demo_plot.clear()
            return

        scale = self._time_unit_scale()
        self._current_time_unit_scale = scale

        if not self._range_initialized and self.data.n_samples > 0:
            self.spin_start.blockSignals(True)
            self.spin_end.blockSignals(True)
            self.spin_start.setValue(float(self.data.time[0]) / scale)
            self.spin_end.setValue(float(self.data.time[-1]) / scale)
            self.spin_start.blockSignals(False)
            self.spin_end.blockSignals(False)
            self._range_initialized = True

        self.selected_data = self._selected_data()

        meta_lines = ""
        if self.data.metadata:
            shown = list(self.data.metadata.items())[:8]
            meta_lines = "\n".join(f"{k}: {v}" for k, v in shown)

        self.lbl_info.setText(
            f"Source: {self.data.source_name}\n"
            f"Samples: {self.data.n_samples}\n"
            f"Sampling rate: {self.data.sampling_rate:.3f} Hz\n"
            f"Duration: {self.data.duration:.3f} s\n"
            + (f"\n\nMetadata:\n{meta_lines}" if meta_lines else "")
        )

        if self.selected_data is None:
            self.time_plot.clear()
            self.freq_plot.clear()
            self.demo_plot.clear()
            return

        self._update_time_plot()
        self._update_frequency_plot()
        self._install_fft_click_handler()
        self._update_demodulation_plot()
        self.statusBar().showMessage("Views updated", 2500)

    def _update_time_plot(self):
        if self.selected_data is None or self.selected_data.n_samples == 0:
            self.time_plot.plot.clear()
            return

        scale = self._time_unit_scale()
        x = self.selected_data.time / scale
        y = self.selected_data.amplitude

        self.time_plot.set_pen(pg.mkPen("b", width=1.2))
        self.time_plot.set_axis_labels(
            bottom="Time",
            left=self.data.column_names[1] if self.data else "Amplitude",
            bottom_units=self._time_unit_label(),
        )

        # Adaptive plot: initial draw uses full data, zoom redraws use visible range
        self.time_plot.set_data(x, y, auto_range=True)

    def _update_frequency_plot(self):
        self._debug("FFT update requested")
        self.freq_plot.clear()

        if self.data is None:
            self._debug("No data loaded")
            return

        self.selected_data = self._selected_data()
        if self.selected_data is None or self.selected_data.n_samples < 2:
            self._debug("selected_data is None or too short")
            return

        xf, mag = Analysis.fft_spectrum(
            self.selected_data,
            window_name=self.combo_fft_window.currentText(),
            remove_mean=self.check_fft_remove_mean.isChecked(),
        )

        self._debug(f"FFT result sizes: xf={xf.size}, mag={mag.size}")

        if xf.size == 0 or mag.size == 0:
            self._debug("FFT returned empty arrays")
            return

        # Adaptive plot: initial draw uses full data, zoom redraws use visible range
        self.freq_plot.set_pen(pg.mkPen("r", width=1.5))
        self.freq_plot.set_axis_labels(
            bottom="Frequency",
            left="Magnitude",
            bottom_units="Hz",
        )
        self.freq_plot.set_data(xf, mag, auto_range=True)

        self._debug("FFT plot updated successfully")

    def _install_fft_click_handler(self):
        # Call this once after the plots are created
        self.freq_plot.plot.scene().sigMouseClicked.connect(self._on_fft_plot_clicked)

    def _on_fft_plot_clicked(self, event):
        if self.tabs.currentWidget() != self.freq_plot:
            return

        vb = self.freq_plot.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return

        mouse_point = vb.mapSceneToView(event.scenePos())
        clicked_x = float(mouse_point.x())

        if self.freq_plot._full_data is None:
            self._debug("FFT plot has no stored data")
            return

        xf = self.freq_plot._full_data.x
        if xf.size == 0:
            self._debug("FFT stored data is empty")
            return

        idx = int(np.argmin(np.abs(xf - clicked_x)))
        selected_freq = float(xf[idx])

        self._selected_fft_frequency = selected_freq
        self.spin_demod_frequency.setValue(selected_freq)
        self.lbl_selected_freq.setText(f"Selected FFT frequency: {selected_freq:.6f} Hz")
        self.statusBar().showMessage(f"Selected frequency: {selected_freq:.6f} Hz", 3000)
        self._debug(f"Clicked FFT frequency selected: {selected_freq:.6f} Hz")

    def _use_selected_fft_frequency(self):
        if self._selected_fft_frequency is None:
            QtWidgets.QMessageBox.information(self, "Frequency", "No FFT frequency selected yet.")
            return

        self.spin_demod_frequency.setValue(self._selected_fft_frequency)
        self.lbl_selected_freq.setText(f"Selected FFT frequency: {self._selected_fft_frequency:.6f} Hz")
        self.tabs.setCurrentWidget(self.demo_plot)

    def _ensure_stft_debug_window(self) -> StftDebugWindow:
        if self._stft_debug_window is None:
            self._stft_debug_window = StftDebugWindow(self)
        return self._stft_debug_window

    def _update_demodulation_plot(self):
        self.demo_plot.clear()
        self._stft_color_bar = None

        if self.data is None:
            self.demo_plot.clear()
            if self._stft_debug_window is not None:
                self._stft_debug_window.clear()
            return

        # Ensure we use the full, non-downsampled data for the STFT calculation
        self.selected_data = self._selected_data()
        if self.selected_data is None or self.selected_data.n_samples < 2:
            self.demo_plot.clear()
            if self._stft_debug_window is not None:
                self._stft_debug_window.clear()
            return

        self._debug(f"Demodulation calculation on {self.selected_data.n_samples} full-resolution samples.")

        f0 = float(self.spin_demod_frequency.value())
        t = self.selected_data.time
        y = self.selected_data.amplitude

        # Apply frequency shift (demodulation) before STFT
        analytic = y * np.exp(-2j * np.pi * f0 * t)

        times, freqs, Z = Analysis.stft(
            SignalData(t, analytic, sampling_rate=self.selected_data.sampling_rate),
            window_name=self.combo_stft_window.currentText(),
            nperseg=self.spin_nperseg.value(),
            noverlap=self.spin_noverlap.value(),
            nfft=self.spin_nfft.value(),
            remove_mean=self.check_stft_remove_mean.isChecked(),
        )
        if Z.size == 0:
            if self._stft_debug_window is not None:
                self._stft_debug_window.clear()
            return

        # "plots the amplitude" -> amplitude of the DC bin after frequency shift
        # Z is (n_freqs, n_times)
        # freqs[0] should be 0 Hz (DC)
        amplitude = Z[0, :]

        scale = self._time_unit_scale()
        self.demo_plot.set_pen(pg.mkPen("m", width=1.5))
        self.demo_plot.set_axis_labels(
            bottom="Time",
            left="Amplitude",
            bottom_units=self._time_unit_label(),
        )
        self.demo_plot.set_data(times / scale, amplitude, auto_range=True)
        self.demo_plot.plot.setTitle(f"Demodulated Amplitude at {f0:.3f} Hz")

        debug_window = self._ensure_stft_debug_window()
        debug_window.set_stft(
            times=times,
            freqs=freqs,
            Z=Z,
            f0=f0,
            time_unit=self._time_unit_label(),
            time_scale=scale,
        )
        debug_window.show()
        debug_window.raise_()
        debug_window.activateWindow()

    def export_data(self):
        if self.selected_data is None or self.selected_data.n_samples == 0:
            QtWidgets.QMessageBox.information(self, "Export", "No selected data to export.")
            return

        file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export processed data",
            "",
            "CSV (*.csv);;JSON (*.json);;NumPy NPZ (*.npz)",
        )
        if not file_path:
            return

        try:
            ext = Path(file_path).suffix.lower()
            if not ext:
                if "csv" in selected_filter.lower():
                    file_path += ".csv"
                elif "json" in selected_filter.lower():
                    file_path += ".json"
                elif "npz" in selected_filter.lower():
                    file_path += ".npz"

            ext = Path(file_path).suffix.lower()
            if ext == ".csv":
                DataManager.save_csv(file_path, self.selected_data)
            elif ext == ".json":
                DataManager.save_json(file_path, self.selected_data)
            elif ext == ".npz":
                DataManager.save_npz(file_path, self.selected_data)
            else:
                raise ValueError("Unsupported export format.")

            self.statusBar().showMessage(f"Exported data to {file_path}", 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(exc))

    def export_graph(self):
        current = self.tabs.currentWidget()
        if not isinstance(current, PlotPanel):
            return

        file_path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export graph",
            "",
            "PNG Image (*.png);;SVG Image (*.svg)",
        )
        if not file_path:
            return

        try:
            ext = Path(file_path).suffix.lower()
            if not ext:
                if "png" in selected_filter.lower():
                    file_path += ".png"
                elif "svg" in selected_filter.lower():
                    file_path += ".svg"

            ext = Path(file_path).suffix.lower()
            if ext == ".png":
                exporter = pg.exporters.ImageExporter(current.plot.plotItem)
                exporter.export(file_path)
            elif ext == ".svg":
                exporter = pg.exporters.SVGExporter(current.plot.plotItem)
                exporter.export(file_path)
            else:
                raise ValueError("Unsupported graph export format.")

            self.statusBar().showMessage(f"Exported graph to {file_path}", 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(exc))

