from __future__ import annotations

from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from data_manager_signal_loader import DataManager
from file_preview_dialog import FilePreviewDialog
from signal_analysis_utils import Analysis
from signal_data_class import SignalData
from signal_data_import import OscilloscopeImporter, ScopeCaptureConfig
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


class ScopeAcquireWorker(QtCore.QObject):
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, config: ScopeCaptureConfig):
        super().__init__()
        self.config = config

    @QtCore.Slot()
    def run(self):
        try:
            data = OscilloscopeImporter.capture_channel(self.config)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(data)


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
        self._scope_thread: QtCore.QThread | None = None
        self._scope_worker: ScopeAcquireWorker | None = None
        self._last_scope_data: SignalData | None = None
        self._demod_cache_payload: dict | None = None

        self._build_ui()
        self._connect_signals()
        self._update_demod_scale_toggle_ui()
        self._update_demod_mode_ui()
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

        self.setStyleSheet("""
            QPushButton {
                background-color: #d9d9d9;
                border: 1px solid #a6a6a6;
                border-radius: 5px;
                padding: 3px 8px;
                min-height: 22px;
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

        def _button_grid(*buttons: QtWidgets.QPushButton) -> QtWidgets.QWidget:
            # Compact 2-column button container to reduce vertical space.
            container = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(container)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(6)
            grid.setVerticalSpacing(4)
            for idx, btn in enumerate(buttons):
                row = idx // 2
                col = idx % 2
                btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                grid.addWidget(btn, row, col)
            return container

        self.group_data = QtWidgets.QGroupBox("Data")
        form_data = QtWidgets.QFormLayout(self.group_data)
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setReadOnly(True)
        self.btn_browse = QtWidgets.QPushButton("Select File...")
        self.btn_load = QtWidgets.QPushButton("Load Selected File")
        self.btn_demo = QtWidgets.QPushButton("Load Demo Signal")
        self.combo_scope_resource = QtWidgets.QComboBox()
        self.btn_scope_refresh = QtWidgets.QPushButton("Refresh VISA Resources")
        self.combo_scope_channel = QtWidgets.QComboBox()
        self.combo_scope_channel.addItems(["CH1", "CH2", "CH3", "CH4"])
        self.spin_scope_points = QtWidgets.QSpinBox()
        self.spin_scope_points.setRange(100, 10000000)
        self.spin_scope_points.setValue(100000)
        self.spin_scope_timeout_ms = QtWidgets.QSpinBox()
        self.spin_scope_timeout_ms.setRange(1000, 120000)
        self.spin_scope_timeout_ms.setValue(5000)
        self.spin_scope_timeout_ms.setSuffix(" ms")
        self.btn_scope_acquire = QtWidgets.QPushButton("Acquire from Oscilloscope")
        self.btn_scope_save = QtWidgets.QPushButton("Save Last Scope Capture")
        self.btn_scope_save.setEnabled(False)
        self.combo_time_column = QtWidgets.QComboBox()
        self.combo_amplitude_column = QtWidgets.QComboBox()
        form_data.addRow(self.path_edit)
        form_data.addRow(_button_grid(self.btn_browse, self.btn_load, self.btn_demo))
        form_data.addRow(QtWidgets.QLabel("--- Oscilloscope Input ---"))
        form_data.addRow("Resource", self.combo_scope_resource)
        form_data.addRow("Channel", self.combo_scope_channel)
        form_data.addRow("Sample Points", self.spin_scope_points)
        form_data.addRow("Timeout", self.spin_scope_timeout_ms)
        form_data.addRow(_button_grid(self.btn_scope_refresh, self.btn_scope_acquire, self.btn_scope_save))
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

        self.spin_demod_target_max_amplitude = QtWidgets.QDoubleSpinBox()
        self.spin_demod_target_max_amplitude.setRange(1e-12, 1e12)
        self.spin_demod_target_max_amplitude.setDecimals(6)
        self.spin_demod_target_max_amplitude.setSingleStep(0.1)
        self.spin_demod_target_max_amplitude.setValue(1.0)
        self.spin_demod_target_max_amplitude.setToolTip("Target peak amplitude for demodulated magnitude")

        self.btn_demod_scale_toggle = QtWidgets.QPushButton("Scale Output")
        self.btn_demod_scale_toggle.setCheckable(True)
        self.btn_demod_scale_toggle.setChecked(True)
        self.btn_demod_scale_toggle.setToolTip("Toggle between scaled demodulation output and raw output")

        self.check_demod_mode_lockin = QtWidgets.QCheckBox("Use Lock-in Mode")
        self.check_demod_mode_lockin.setChecked(True)

        self.btn_use_fft_frequency = QtWidgets.QPushButton("Use Selected FFT Frequency")
        self.btn_update_demod = QtWidgets.QPushButton("Run Demodulation")
        self.lbl_selected_freq = QtWidgets.QLabel("No frequency selected")

        self.combo_stft_window = QtWidgets.QComboBox()
        self.combo_stft_window.addItems(["hann", "hamming", "blackman", "rectangular"])
        self.spin_nperseg = QtWidgets.QSpinBox()
        self.spin_nperseg.setRange(8, 10000000)
        self.spin_nperseg.setValue(256)
        self.spin_noverlap = QtWidgets.QSpinBox()
        self.spin_noverlap.setRange(0, 10000000)
        self.spin_noverlap.setValue(128)
        self.spin_nfft = QtWidgets.QSpinBox()
        self.spin_nfft.setRange(8, 10000000)
        self.spin_nfft.setValue(256)

        self.group_stft_settings = QtWidgets.QGroupBox("STFT Settings")
        form_stft = QtWidgets.QFormLayout(self.group_stft_settings)
        form_stft.addRow("Window", self.combo_stft_window)
        form_stft.addRow("N Per Segment", self.spin_nperseg)
        form_stft.addRow("N Overlap", self.spin_noverlap)
        form_stft.addRow("N FFT", self.spin_nfft)
        self.check_show_stft_debug = QtWidgets.QCheckBox("Show STFT Debug Plot")
        self.check_show_stft_debug.setChecked(True)
        form_stft.addRow(self.check_show_stft_debug)

        self.group_lockin = QtWidgets.QGroupBox("Lock-in Settings")
        form_lockin = QtWidgets.QFormLayout(self.group_lockin)
        self.spin_lockin_lowpass_cutoff = QtWidgets.QDoubleSpinBox()
        self.spin_lockin_lowpass_cutoff.setRange(1e-9, 1e12)
        self.spin_lockin_lowpass_cutoff.setDecimals(6)
        self.spin_lockin_lowpass_cutoff.setValue(1000.0)
        self.spin_lockin_lowpass_cutoff.setSuffix(" Hz")
        self.spin_lockin_lowpass_order = QtWidgets.QSpinBox()
        self.spin_lockin_lowpass_order.setRange(1, 10)
        self.spin_lockin_lowpass_order.setValue(2)
        self.check_lockin_use_iq = QtWidgets.QCheckBox("Use I/Q Magnitude")
        self.check_lockin_use_iq.setChecked(True)
        self.check_lockin_reconstruct_phase = QtWidgets.QCheckBox("Reconstruct Signal Using Phase")
        self.check_lockin_reconstruct_phase.setChecked(False)
        self.check_lockin_show_phase_separately = QtWidgets.QCheckBox("Show Magnitude & Phase Separately")
        self.check_lockin_show_phase_separately.setChecked(False)
        self.check_lockin_skip_transient = QtWidgets.QCheckBox("Skip Initial Transient")
        self.check_lockin_skip_transient.setChecked(True)
        form_lockin.addRow("Lowpass Cutoff", self.spin_lockin_lowpass_cutoff)
        form_lockin.addRow("Lowpass Order", self.spin_lockin_lowpass_order)
        form_lockin.addRow(self.check_lockin_use_iq)
        form_lockin.addRow(self.check_lockin_reconstruct_phase)
        form_lockin.addRow(self.check_lockin_show_phase_separately)
        form_lockin.addRow(self.check_lockin_skip_transient)

        form_demod.addRow("Frequency", self.spin_demod_frequency)
        form_demod.addRow("Max Amplitude", self.spin_demod_target_max_amplitude)
        form_demod.addRow("Amplitude Mode", self.btn_demod_scale_toggle)
        form_demod.addRow(self.check_demod_mode_lockin)
        form_demod.addRow(_button_grid(self.btn_use_fft_frequency, self.btn_update_demod))
        form_demod.addRow(self.group_stft_settings)
        form_demod.addRow(self.group_lockin)
        form_demod.addRow(self.lbl_selected_freq)

        self.group_stft_settings.setVisible(True)
        self.group_lockin.setVisible(False)

        self.group_info = QtWidgets.QGroupBox("Source / Metadata")
        info_layout = QtWidgets.QVBoxLayout(self.group_info)
        self.lbl_info = QtWidgets.QLabel("No data loaded.")
        self.lbl_info.setWordWrap(True)
        info_layout.addWidget(self.lbl_info)

        layout.addWidget(self.group_data)
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
        self.btn_scope_refresh.clicked.connect(self._refresh_scope_resources)
        self.btn_scope_acquire.clicked.connect(self._acquire_scope_data)
        self.btn_scope_save.clicked.connect(self._save_scope_capture)
        self.btn_apply_range.clicked.connect(self.refresh_all_views)

        self.btn_update_fft.clicked.connect(self._update_frequency_plot)
        self.btn_use_fft_frequency.clicked.connect(self._use_selected_fft_frequency)
        self.btn_update_demod.clicked.connect(self._update_demodulation_plot)
        self.check_demod_mode_lockin.stateChanged.connect(self._on_demod_mode_changed)
        self.btn_demod_scale_toggle.toggled.connect(self._on_demod_scale_toggled)
        self.spin_demod_target_max_amplitude.valueChanged.connect(self._on_demod_target_max_changed)
        self.check_lockin_reconstruct_phase.stateChanged.connect(self._on_lockin_reconstruct_mode_changed)
        self.check_lockin_show_phase_separately.stateChanged.connect(self._on_lockin_reconstruct_mode_changed)

        self.combo_time_column.currentIndexChanged.connect(self.refresh_all_views)
        self.combo_amplitude_column.currentIndexChanged.connect(self.refresh_all_views)
        self.combo_time_unit.currentIndexChanged.connect(self._on_time_unit_changed)

        self.tabs.currentChanged.connect(lambda _: self._update_sidebar_visibility())

    def _set_scope_controls_enabled(self, enabled: bool):
        controls = [
            self.combo_scope_resource,
            self.btn_scope_refresh,
            self.combo_scope_channel,
            self.spin_scope_points,
            self.spin_scope_timeout_ms,
            self.btn_scope_acquire,
        ]
        for control in controls:
            control.setEnabled(enabled)

    def _update_sidebar_visibility(self):
        current = self.tabs.currentWidget()
        self.group_data.setVisible(current == self.time_plot)
        self.group_info.setVisible(current == self.time_plot)
        self.group_fft.setVisible(current == self.freq_plot)
        self.group_demod.setVisible(current == self.demo_plot)

    def _on_demod_mode_changed(self):
        self._update_demod_mode_ui()
        if self.data is not None and self.selected_data is not None and self.selected_data.n_samples > 1:
            self._update_demodulation_plot()

    def _on_demod_target_max_changed(self):
        if not self.btn_demod_scale_toggle.isChecked():
            return
        if self.data is not None and self.selected_data is not None and self.selected_data.n_samples > 1:
            if not self._render_demodulation_from_cache():
                self._update_demodulation_plot()

    def _on_demod_scale_toggled(self, checked: bool):
        self._update_demod_scale_toggle_ui()
        if self.data is not None and self.selected_data is not None and self.selected_data.n_samples > 1:
            if not self._render_demodulation_from_cache():
                self._update_demodulation_plot()

    def _update_demod_scale_toggle_ui(self):
        scaling_enabled = self.btn_demod_scale_toggle.isChecked()
        self.btn_demod_scale_toggle.setText("Scale Output" if scaling_enabled else "Raw Output")
        self.spin_demod_target_max_amplitude.setEnabled(scaling_enabled)

    def _update_demod_mode_ui(self):
        is_lock_in = self.check_demod_mode_lockin.isChecked()
        self.group_stft_settings.setVisible(not is_lock_in)
        self.group_lockin.setVisible(is_lock_in)
        self._update_lockin_reconstruction_ui()

    def _on_lockin_reconstruct_mode_changed(self):
        self._update_lockin_reconstruction_ui()
        if self.data is not None and self.selected_data is not None and self.selected_data.n_samples > 1:
            self._update_demodulation_plot()

    def _update_lockin_reconstruction_ui(self):
        needs_iq_reconstruct = self.check_lockin_reconstruct_phase.isChecked()
        needs_iq_phase_sep = self.check_lockin_show_phase_separately.isChecked()
        
        # If either reconstruction option is active, I/Q must be enabled
        if needs_iq_reconstruct or needs_iq_phase_sep:
            self.check_lockin_use_iq.setChecked(True)
        
        # Disable if I/Q is not available
        self.check_lockin_use_iq.setEnabled(not (needs_iq_reconstruct or needs_iq_phase_sep))

    def browse_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select data file",
            "",
            "Data Files (*.csv *.txt *.npy *.npz);;All Files (*)",
        )
        if file_path:
            self.path_edit.setText(file_path)

    def _refresh_scope_resources(self):
        self.combo_scope_resource.clear()

        if not OscilloscopeImporter.pyvisa_available():
            self.combo_scope_resource.addItem("pyvisa not installed")
            self.combo_scope_resource.setEnabled(False)
            self.btn_scope_acquire.setEnabled(False)
            self.statusBar().showMessage("pyvisa fehlt - Oszilloskopfunktion deaktiviert", 5000)
            return

        try:
            resources = OscilloscopeImporter.list_resources()
        except Exception as exc:
            self.combo_scope_resource.addItem("No VISA resource")
            self.combo_scope_resource.setEnabled(False)
            self.btn_scope_acquire.setEnabled(False)
            QtWidgets.QMessageBox.warning(self, "VISA", f"VISA-Ressourcen konnten nicht gelesen werden:\n{exc}")
            return

        if not resources:
            self.combo_scope_resource.addItem("No VISA resource")
            self.combo_scope_resource.setEnabled(False)
            self.btn_scope_acquire.setEnabled(False)
            self.statusBar().showMessage("Keine VISA-Ressourcen gefunden", 5000)
            return

        self.combo_scope_resource.addItems(list(resources))
        self.combo_scope_resource.setEnabled(True)
        self.btn_scope_acquire.setEnabled(True)
        self.statusBar().showMessage(f"{len(resources)} VISA-Ressource(n) gefunden", 4000)

    def _acquire_scope_data(self):
        if self._scope_thread is not None:
            QtWidgets.QMessageBox.information(self, "Oszilloskop", "Akquise laeuft bereits.")
            return

        resource = self.combo_scope_resource.currentText().strip()
        if not resource or resource in {"No VISA resource", "pyvisa not installed"}:
            QtWidgets.QMessageBox.warning(self, "Oszilloskop", "Bitte zuerst eine gueltige VISA-Ressource waehlen.")
            return

        config = ScopeCaptureConfig(
            resource_name=resource,
            channel=self.combo_scope_channel.currentText().strip(),
            point_count=int(self.spin_scope_points.value()),
            timeout_ms=int(self.spin_scope_timeout_ms.value()),
        )

        self._scope_worker = ScopeAcquireWorker(config)
        self._scope_thread = QtCore.QThread(self)
        self._scope_worker.moveToThread(self._scope_thread)

        self._scope_thread.started.connect(self._scope_worker.run)
        self._scope_worker.finished.connect(self._on_scope_capture_finished)
        self._scope_worker.failed.connect(self._on_scope_capture_failed)
        self._scope_worker.finished.connect(self._scope_thread.quit)
        self._scope_worker.failed.connect(self._scope_thread.quit)
        self._scope_worker.finished.connect(self._scope_worker.deleteLater)
        self._scope_worker.failed.connect(self._scope_worker.deleteLater)
        self._scope_thread.finished.connect(self._scope_thread.deleteLater)
        self._scope_thread.finished.connect(self._on_scope_thread_finished)

        self._set_scope_controls_enabled(False)
        self.statusBar().showMessage("Oszilloskop-Akquise gestartet...", 3000)
        self._scope_thread.start()

    @QtCore.Slot(object)
    def _on_scope_capture_finished(self, data: SignalData):
        self._last_scope_data = data
        self.btn_scope_save.setEnabled(True)

        self.data = data
        self._sync_column_choices_from_data()
        self._range_initialized = False
        self.refresh_all_views()
        self.tabs.setCurrentWidget(self.time_plot)
        self._update_sidebar_visibility()
        self.statusBar().showMessage("Oszilloskop-Daten eingelesen", 5000)

    @QtCore.Slot(str)
    def _on_scope_capture_failed(self, message: str):
        QtWidgets.QMessageBox.critical(self, "Oszilloskop Fehler", message)
        self.statusBar().showMessage("Oszilloskop-Akquise fehlgeschlagen", 5000)

    @QtCore.Slot()
    def _on_scope_thread_finished(self):
        self._scope_thread = None
        self._scope_worker = None
        self._set_scope_controls_enabled(True)

    def _save_scope_capture(self):
        if self._last_scope_data is None or self._last_scope_data.n_samples == 0:
            QtWidgets.QMessageBox.information(self, "Oszilloskop", "Keine Oszilloskop-Daten zum Speichern vorhanden.")
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save oscilloscope capture",
            "",
            "CSV (*.csv)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"

        try:
            DataManager.save_scope_csv(file_path, self._last_scope_data)
            self.statusBar().showMessage(f"Oszilloskop-Daten gespeichert: {file_path}", 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Save Error", str(exc))

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

    def _demod_magnitude_axis_label(self, default_label: str) -> str:
        if not self.btn_demod_scale_toggle.isChecked():
            return default_label
        target_max = float(self.spin_demod_target_max_amplitude.value())
        return "Current (A)" if not np.isclose(target_max, 1.0, rtol=0.0, atol=1e-12) else default_label

    def _scale_to_target_max(self, signal: np.ndarray) -> np.ndarray:
        if not self.btn_demod_scale_toggle.isChecked():
            return np.asarray(signal, dtype=float)

        target_max = float(self.spin_demod_target_max_amplitude.value())
        arr = np.asarray(signal, dtype=float)
        finite = np.isfinite(arr)
        if not np.any(finite):
            return signal

        current_max = float(np.max(np.abs(arr[finite])))
        if current_max <= 0:
            return signal

        return arr * (target_max / current_max)

    @staticmethod
    def _wrap_phase_to_pi(phase_rad: np.ndarray) -> np.ndarray:
        phase = np.asarray(phase_rad, dtype=float)
        return (phase + np.pi) % (2.0 * np.pi) - np.pi

    def _render_demodulation_from_cache(self) -> bool:
        payload = self._demod_cache_payload
        if not payload:
            return False

        mode = payload.get("mode")
        f0 = float(payload.get("f0", 0.0))
        scale = self._time_unit_scale()

        if mode == "lockin":
            times = np.asarray(payload.get("times", []), dtype=float)
            amplitude_raw = np.asarray(payload.get("amplitude", []), dtype=float)
            phase_rad = self._wrap_phase_to_pi(np.asarray(payload.get("phase_rad", []), dtype=float))
            reconstructed_raw = np.asarray(payload.get("reconstructed", []), dtype=float)
            show_reconstructed = bool(payload.get("show_reconstructed", False))
            show_phase_separately = bool(payload.get("show_phase_separately", False))

            if amplitude_raw.size == 0:
                return False

            times_scaled = times / scale

            if show_phase_separately:
                amplitude = self._scale_to_target_max(amplitude_raw)
                self.demo_plot.set_pen(pg.mkPen("m", width=1.5))
                self.demo_plot.set_axis_labels(
                    bottom="Time",
                    left=self._demod_magnitude_axis_label("Magnitude"),
                    bottom_units=self._time_unit_label(),
                )
                self.demo_plot.set_dual_axis_data(
                    times_scaled,
                    amplitude,
                    phase_rad,
                    right_label="Phase",
                    right_units="rad",
                    right_pen=pg.mkPen("g", width=1.4),
                    auto_range=True,
                )
                self.demo_plot.plot.setTitle(f"Lock-in Magnitude & Phase at {f0:.3f} Hz")
            elif show_reconstructed:
                reconstructed = self._scale_to_target_max(reconstructed_raw)
                self.demo_plot.set_pen(pg.mkPen("m", width=1.5))
                self.demo_plot.set_axis_labels(
                    bottom="Time",
                    left="Reconstructed Signal",
                    bottom_units=self._time_unit_label(),
                )
                self.demo_plot.set_data(times_scaled, reconstructed, auto_range=True)
                self.demo_plot.plot.setTitle(f"Lock-in Reconstructed Signal at {f0:.3f} Hz")
            else:
                amplitude = self._scale_to_target_max(amplitude_raw)
                self.demo_plot.set_pen(pg.mkPen("m", width=1.5))
                self.demo_plot.set_axis_labels(
                    bottom="Time",
                    left=self._demod_magnitude_axis_label("Amplitude"),
                    bottom_units=self._time_unit_label(),
                )
                self.demo_plot.set_data(times_scaled, amplitude, auto_range=True)
                self.demo_plot.plot.setTitle(f"Lock-in Demodulated Amplitude at {f0:.3f} Hz")

            if self._stft_debug_window is not None:
                self._stft_debug_window.clear()
                self._stft_debug_window.hide()
            return True

        if mode == "stft":
            times = np.asarray(payload.get("times", []), dtype=float)
            amplitude_raw = np.asarray(payload.get("amplitude", []), dtype=float)
            freqs = np.asarray(payload.get("freqs", []), dtype=float)
            Z = np.asarray(payload.get("Z", []), dtype=float)

            if amplitude_raw.size == 0:
                return False

            amplitude = self._scale_to_target_max(amplitude_raw)

            self.demo_plot.set_pen(pg.mkPen("m", width=1.5))
            self.demo_plot.set_axis_labels(
                bottom="Time",
                left=self._demod_magnitude_axis_label("Amplitude"),
                bottom_units=self._time_unit_label(),
            )
            self.demo_plot.set_data(times / scale, amplitude, auto_range=True)
            self.demo_plot.plot.setTitle(f"Demodulated Amplitude at {f0:.3f} Hz")

            if self.check_show_stft_debug.isChecked():
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
            elif self._stft_debug_window is not None:
                self._stft_debug_window.clear()
                self._stft_debug_window.hide()
            return True

        return False

    def _update_demodulation_plot(self):
        self.demo_plot.clear()
        self._stft_color_bar = None

        if self.data is None:
            self._demod_cache_payload = None
            self.demo_plot.clear()
            if self._stft_debug_window is not None:
                self._stft_debug_window.clear()
            return

        # Ensure we use the full, non-downsampled data for the STFT calculation
        self.selected_data = self._selected_data()
        if self.selected_data is None or self.selected_data.n_samples < 2:
            self._demod_cache_payload = None
            self.demo_plot.clear()
            if self._stft_debug_window is not None:
                self._stft_debug_window.clear()
            return

        self._debug(f"Demodulation calculation on {self.selected_data.n_samples} full-resolution samples.")

        f0 = float(self.spin_demod_frequency.value())
        t = self.selected_data.time
        y = self.selected_data.amplitude

        is_lock_in = self.check_demod_mode_lockin.isChecked()
        if is_lock_in:
            times, amplitude, phase_rad, reconstructed = Analysis.lock_in_demod(
                SignalData(t, y, sampling_rate=self.selected_data.sampling_rate),
                reference_frequency=f0,
                lowpass_cutoff_hz=float(self.spin_lockin_lowpass_cutoff.value()),
                lowpass_order=int(self.spin_lockin_lowpass_order.value()),
                use_iq=self.check_lockin_use_iq.isChecked(),
            )
            phase_rad = self._wrap_phase_to_pi(phase_rad)
            if amplitude.size == 0:
                self._demod_cache_payload = None
                if self._stft_debug_window is not None:
                    self._stft_debug_window.clear()
                    self._stft_debug_window.hide()
                return

            # Skip transient if requested
            if self.check_lockin_skip_transient.isChecked() and amplitude.size > 1:
                fs = float(self.selected_data.sampling_rate) if self.selected_data.sampling_rate > 0 else 1.0
                cutoff = float(self.spin_lockin_lowpass_cutoff.value())
                order = int(self.spin_lockin_lowpass_order.value())
                tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-9))
                skip_samples = int(np.ceil(5.0 * order * tau * fs))
                skip_samples = min(skip_samples, amplitude.size - 1)
                times = times[skip_samples:]
                amplitude = amplitude[skip_samples:]
                phase_rad = phase_rad[skip_samples:]
                reconstructed = reconstructed[skip_samples:]

            self._demod_cache_payload = {
                "mode": "lockin",
                "f0": f0,
                "times": times,
                "amplitude": amplitude,
                "phase_rad": phase_rad,
                "reconstructed": reconstructed,
                "show_reconstructed": self.check_lockin_reconstruct_phase.isChecked(),
                "show_phase_separately": self.check_lockin_show_phase_separately.isChecked(),
            }
            self._render_demodulation_from_cache()
            return

        # Apply frequency shift (demodulation) before STFT
        analytic = y * np.exp(-2j * np.pi * f0 * t)

        times, freqs, Z = Analysis.stft(
            SignalData(t, analytic, sampling_rate=self.selected_data.sampling_rate),
            window_name=self.combo_stft_window.currentText(),
            nperseg=self.spin_nperseg.value(),
            noverlap=self.spin_noverlap.value(),
            nfft=self.spin_nfft.value(),
            remove_mean=False,
        )
        if Z.size == 0:
            self._demod_cache_payload = None
            if self._stft_debug_window is not None:
                self._stft_debug_window.clear()
            return

        # "plots the amplitude" -> amplitude of the DC bin after frequency shift
        # Z is (n_freqs, n_times)
        # freqs[0] should be 0 Hz (DC)
        self._demod_cache_payload = {
            "mode": "stft",
            "f0": f0,
            "times": times,
            "amplitude": Z[0, :],
            "freqs": freqs,
            "Z": Z,
        }
        self._render_demodulation_from_cache()

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

    def showEvent(self, event: QtGui.QShowEvent):
        super().showEvent(event)
        if self.combo_scope_resource.count() == 0:
            self._refresh_scope_resources()
