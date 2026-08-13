from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters

from frequency_sweep import (
    MeasurementPoint,
    MockClient,
    ScanConfig,
    SpectrumScanner,
    TektronixVisaClient,
    inspect_visa_resources,
    list_visa_resources,
)
from plot_panel_widget import PlotPanel


class FrequencyInput(QtWidgets.QWidget):
    UNIT_FACTORS = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}

    def __init__(
        self,
        value: float,
        unit: str = "Hz",
        parent=None,
        *,
        minimum: float = 0.0,
    ):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.value_spin = QtWidgets.QDoubleSpinBox()
        self.value_spin.setRange(minimum, 1e12)
        self.value_spin.setDecimals(9)
        self.value_spin.setValue(value)
        self.value_spin.setKeyboardTracking(False)
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(list(self.UNIT_FACTORS))
        self.unit_combo.setCurrentText(unit)
        layout.addWidget(self.value_spin, 1)
        layout.addWidget(self.unit_combo, 0)

    def value_hz(self) -> float:
        return float(self.value_spin.value()) * self.UNIT_FACTORS[self.unit_combo.currentText()]


class SweepWorker(QtCore.QObject):
    point = QtCore.Signal(object, int, int)
    completed = QtCore.Signal(object, bool)
    failed = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, config: ScanConfig):
        super().__init__()
        client = MockClient() if config.use_mock else TektronixVisaClient()
        self.scanner = SpectrumScanner(client)
        self.config = config

    @QtCore.Slot()
    def run(self) -> None:
        try:
            points = self.scanner.run(self.config, on_point=self.point.emit)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(points, self.scanner.stop_requested)
        finally:
            self.finished.emit()

    def stop(self) -> None:
        # This is intentionally a direct, thread-safe Event update. A queued Qt
        # slot would not run while the worker thread is blocked in VISA I/O.
        self.scanner.stop()


class FrequencySweepWidget(QtWidgets.QWidget):
    frequency_selected = QtCore.Signal(float)
    status_message = QtCore.Signal(str, int)
    hardware_busy_changed = QtCore.Signal(bool)

    DEFAULT_AWG_RESOURCE = "USB0::0x0699::0x0353::2017597::INSTR"
    DEFAULT_SCOPE_RESOURCE = "USB0::0x0699::0x0408::C032947::INSTR"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scan_thread: QtCore.QThread | None = None
        self._scan_worker: SweepWorker | None = None
        self._points: list[MeasurementPoint] = []
        self._active_mode = "frequency"
        self._expected_points = 0
        self._using_hardware = False
        self._external_hardware_busy = False
        self._build_ui()
        self._connect_signals()
        self._update_mode_controls()
        self._update_plot_axes()

    @property
    def is_running(self) -> bool:
        return self._scan_thread is not None

    @property
    def points(self) -> tuple[MeasurementPoint, ...]:
        return tuple(self._points)

    def _build_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)

        control_content = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_content)

        instrument_group = QtWidgets.QGroupBox("AWG and RF Oscilloscope")
        instrument_form = QtWidgets.QFormLayout(instrument_group)
        self.awg_resource = self._resource_combo(self.DEFAULT_AWG_RESOURCE)
        self.scope_resource = self._resource_combo(self.DEFAULT_SCOPE_RESOURCE)
        self.timeout_ms = QtWidgets.QSpinBox()
        self.timeout_ms.setRange(1000, 120000)
        self.timeout_ms.setValue(10000)
        self.timeout_ms.setSuffix(" ms")
        self.refresh_resources_button = QtWidgets.QPushButton("Refresh VISA")
        self.inspect_resources_button = QtWidgets.QPushButton("Inspect VISA")
        self.debug_scope_button = QtWidgets.QPushButton("Scope Debug")
        instrument_form.addRow("AWG Resource", self.awg_resource)
        instrument_form.addRow("Scope Resource", self.scope_resource)
        instrument_form.addRow("Timeout", self.timeout_ms)
        instrument_form.addRow(
            self._button_grid(
                self.refresh_resources_button,
                self.inspect_resources_button,
                self.debug_scope_button,
            )
        )

        scan_group = QtWidgets.QGroupBox("Sweep Settings")
        scan_form = QtWidgets.QFormLayout(scan_group)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("Frequency Sweep", "frequency")
        self.mode_combo.addItem("Amplitude Sweep", "amplitude")
        self.mode_combo.addItem("Offset Sweep", "offset")
        self.awg_vpp = self._double_spin(2.7, 0.0, 1000.0, 6, " Vpp")
        self.awg_offset = self._double_spin(0.0, -1000.0, 1000.0, 6, " V")
        self.subwindow_span = FrequencyInput(100.0, "kHz")
        self.rbw = FrequencyInput(1.0, "kHz")
        self.evaluation_offset = FrequencyInput(0.0, "Hz", minimum=-1e12)
        self.evaluation_offset.setToolTip(
            "Amplitude is evaluated at AWG frequency + this offset. "
            "For example, 50 Hz evaluates a 1 MHz sweep step at 1,000,050 Hz."
        )
        self.iterations = QtWidgets.QSpinBox()
        self.iterations.setRange(1, 10000)
        self.iterations.setValue(1)
        self.averages = QtWidgets.QSpinBox()
        self.averages.setRange(1, 10000)
        self.averages.setValue(1)
        self.dwell_s = self._double_spin(0.15, 0.0, 3600.0, 3, " s")

        scan_form.addRow("Mode", self.mode_combo)
        scan_form.addRow("AWG Base Amplitude", self.awg_vpp)
        scan_form.addRow("AWG Base Offset", self.awg_offset)
        scan_form.addRow("Scope Window Span", self.subwindow_span)
        scan_form.addRow("Resolution Bandwidth", self.rbw)
        scan_form.addRow("Evaluation Freq. Offset", self.evaluation_offset)
        scan_form.addRow("Measurements / Step", self.iterations)
        scan_form.addRow("Scope Averages", self.averages)
        scan_form.addRow("Dwell / Window", self.dwell_s)

        self.mode_stack = QtWidgets.QStackedWidget()
        self.frequency_page = QtWidgets.QWidget()
        frequency_form = QtWidgets.QFormLayout(self.frequency_page)
        frequency_form.setContentsMargins(0, 0, 0, 0)
        self.start_frequency = FrequencyInput(1.0, "MHz")
        self.stop_frequency = FrequencyInput(3.0, "MHz")
        self.step_frequency = FrequencyInput(100.0, "kHz")
        frequency_form.addRow("Start Frequency", self.start_frequency)
        frequency_form.addRow("Stop Frequency", self.stop_frequency)
        frequency_form.addRow("Frequency Step", self.step_frequency)
        self.mode_stack.addWidget(self.frequency_page)

        self.amplitude_page = QtWidgets.QWidget()
        amplitude_form = QtWidgets.QFormLayout(self.amplitude_page)
        amplitude_form.setContentsMargins(0, 0, 0, 0)
        self.amplitude_fixed_frequency = FrequencyInput(1.25, "MHz")
        self.amplitude_start = self._double_spin(0.5, 0.0, 1000.0, 6, " Vpp")
        self.amplitude_stop = self._double_spin(3.0, 0.0, 1000.0, 6, " Vpp")
        self.amplitude_step = self._double_spin(0.1, 0.0, 1000.0, 6, " Vpp")
        amplitude_form.addRow("Fixed Frequency", self.amplitude_fixed_frequency)
        amplitude_form.addRow("Start Amplitude", self.amplitude_start)
        amplitude_form.addRow("Stop Amplitude", self.amplitude_stop)
        amplitude_form.addRow("Amplitude Step", self.amplitude_step)
        self.mode_stack.addWidget(self.amplitude_page)

        self.offset_page = QtWidgets.QWidget()
        offset_form = QtWidgets.QFormLayout(self.offset_page)
        offset_form.setContentsMargins(0, 0, 0, 0)
        self.offset_fixed_frequency = FrequencyInput(1.25, "MHz")
        self.offset_start = self._double_spin(-0.5, -1000.0, 1000.0, 6, " V")
        self.offset_stop = self._double_spin(0.5, -1000.0, 1000.0, 6, " V")
        self.offset_step = self._double_spin(0.05, 0.0, 1000.0, 6, " V")
        offset_form.addRow("Fixed Frequency", self.offset_fixed_frequency)
        offset_form.addRow("Start Offset", self.offset_start)
        offset_form.addRow("Stop Offset", self.offset_stop)
        offset_form.addRow("Offset Step", self.offset_step)
        self.mode_stack.addWidget(self.offset_page)
        scan_form.addRow(self.mode_stack)

        output_group = QtWidgets.QGroupBox("Run and Output")
        output_form = QtWidgets.QFormLayout(output_group)
        self.mock_mode = QtWidgets.QCheckBox("Mock mode (no hardware)")
        self.auto_save_csv = QtWidgets.QCheckBox("Save CSV when scan finishes")
        self.auto_save_csv.setChecked(True)
        self.csv_path = QtWidgets.QLineEdit("scan_results.csv")
        self.browse_csv_button = QtWidgets.QPushButton("...")
        self.browse_csv_button.setMaximumWidth(36)
        path_widget = QtWidgets.QWidget()
        path_layout = QtWidgets.QHBoxLayout(path_widget)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.csv_path, 1)
        path_layout.addWidget(self.browse_csv_button)
        self.start_button = QtWidgets.QPushButton("Start Scan")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.export_button = QtWidgets.QPushButton("Save Results...")
        self.export_button.setEnabled(False)
        self.use_best_button = QtWidgets.QPushButton("Use Best Frequency for Demodulation")
        self.use_best_button.setEnabled(False)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.run_status = QtWidgets.QLabel("Ready")
        self.run_status.setWordWrap(True)
        output_form.addRow(self.mock_mode)
        output_form.addRow(self.auto_save_csv)
        output_form.addRow("CSV Path", path_widget)
        output_form.addRow(self._button_row(self.start_button, self.stop_button))
        output_form.addRow(self._button_row(self.clear_button, self.export_button))
        output_form.addRow(self.use_best_button)
        output_form.addRow(self.progress)
        output_form.addRow(self.run_status)

        control_layout.addWidget(instrument_group)
        control_layout.addWidget(scan_group)
        control_layout.addWidget(output_group)
        control_layout.addStretch(1)

        self._settings_groups = (instrument_group, scan_group)
        self.control_panel = QtWidgets.QScrollArea()
        self.control_panel.setWidgetResizable(True)
        self.control_panel.setWidget(control_content)
        self.control_panel.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.control_panel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.control_panel.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.control_panel.setMinimumWidth(400)
        self.control_panel.setMaximumWidth(400)
        self.control_panel.setMinimumHeight(0)
        self.control_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        result_widget = QtWidgets.QWidget()
        result_layout = QtWidgets.QVBoxLayout(result_widget)
        self.plot_panel = PlotPanel("Amplitude at the Current Sweep Step")
        self.plot_panel.set_pen(pg.mkPen("#1769aa", width=2.0))
        self.last_point_label = QtWidgets.QLabel("No measurements yet.")
        self.last_point_label.setWordWrap(True)
        self.best_point_label = QtWidgets.QLabel("Best: n/a")
        self.best_point_label.setWordWrap(True)
        result_layout.addWidget(self.plot_panel, 1)
        result_layout.addWidget(self.last_point_label)
        result_layout.addWidget(self.best_point_label)

        # The controls are hosted by MainWindow's fixed sidebar. Keeping them
        # outside this tab prevents the top navigation bar from shifting when
        # the user switches to or from Frequency Sweep.
        main_layout.addWidget(result_widget, 1)

    @staticmethod
    def _double_spin(
        value: float,
        minimum: float,
        maximum: float,
        decimals: int,
        suffix: str,
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setKeyboardTracking(False)
        return spin

    @staticmethod
    def _resource_combo(default: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.setEditable(True)
        combo.addItem(default)
        combo.setCurrentText(default)
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(24)
        return combo

    @staticmethod
    def _button_row(*buttons: QtWidgets.QPushButton) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for button in buttons:
            layout.addWidget(button)
        return widget

    @staticmethod
    def _button_grid(*buttons: QtWidgets.QPushButton) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)
        for index, button in enumerate(buttons):
            layout.addWidget(button, index // 2, index % 2)
        return widget

    def _connect_signals(self) -> None:
        self.mode_combo.currentIndexChanged.connect(self._update_mode_controls)
        self.mock_mode.toggled.connect(self._update_external_hardware_state)
        self.refresh_resources_button.clicked.connect(self.refresh_resources)
        self.inspect_resources_button.clicked.connect(self.inspect_resources)
        self.debug_scope_button.clicked.connect(self.debug_scope)
        self.browse_csv_button.clicked.connect(self.browse_csv_path)
        self.start_button.clicked.connect(self.start_scan)
        self.stop_button.clicked.connect(self.stop_scan)
        self.clear_button.clicked.connect(self.clear_results)
        self.export_button.clicked.connect(self.export_csv)
        self.use_best_button.clicked.connect(self.use_best_frequency)

    def _update_mode_controls(self) -> None:
        mode = str(self.mode_combo.currentData())
        self.mode_stack.setCurrentIndex({"frequency": 0, "amplitude": 1, "offset": 2}[mode])
        if not self._points:
            self._active_mode = mode
            self._update_plot_axes()
        self.use_best_button.setVisible(mode == "frequency")
        self._update_best_point()

    def _update_plot_axes(self) -> None:
        if self._active_mode == "frequency":
            bottom, units = "AWG Frequency", "Hz"
        elif self._active_mode == "amplitude":
            bottom, units = "AWG Amplitude", "Vpp"
        else:
            bottom, units = "AWG Offset", "V"
        self.plot_panel.set_axis_labels(bottom=bottom, left="Measured Amplitude (dBm)", bottom_units=units)

    def _collect_config(self) -> ScanConfig:
        mode = str(self.mode_combo.currentData())
        fixed_frequency_hz = 0.0
        if mode == "amplitude":
            fixed_frequency_hz = self.amplitude_fixed_frequency.value_hz()
        elif mode == "offset":
            fixed_frequency_hz = self.offset_fixed_frequency.value_hz()

        return ScanConfig(
            sweep_mode=mode,
            awg_resource=self.awg_resource.currentText().strip(),
            scope_resource=self.scope_resource.currentText().strip(),
            awg_vpp=float(self.awg_vpp.value()),
            awg_offset_v=float(self.awg_offset.value()),
            fixed_frequency_hz=fixed_frequency_hz,
            total_start_hz=self.start_frequency.value_hz(),
            total_stop_hz=self.stop_frequency.value_hz(),
            subwindow_span_hz=self.subwindow_span.value_hz(),
            step_size_hz=self.step_frequency.value_hz(),
            amp_start_vpp=float(self.amplitude_start.value()),
            amp_stop_vpp=float(self.amplitude_stop.value()),
            amp_step_vpp=float(self.amplitude_step.value()),
            offset_start_v=float(self.offset_start.value()),
            offset_stop_v=float(self.offset_stop.value()),
            offset_step_v=float(self.offset_step.value()),
            rbw_hz=self.rbw.value_hz(),
            iterations=int(self.iterations.value()),
            avg_count=int(self.averages.value()),
            dwell_s=float(self.dwell_s.value()),
            save_csv=self.auto_save_csv.isChecked(),
            csv_path=self.csv_path.text().strip(),
            use_mock=self.mock_mode.isChecked(),
            timeout_ms=int(self.timeout_ms.value()),
            evaluation_offset_hz=self.evaluation_offset.value_hz(),
        )

    @QtCore.Slot()
    def refresh_resources(self) -> None:
        try:
            resources = list_visa_resources()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "VISA Resources", str(exc))
            self._set_status("Could not read VISA resources.", 5000)
            return

        if not resources:
            self._set_status("No VISA resources found.", 5000)
            QtWidgets.QMessageBox.information(self, "VISA Resources", "No VISA resources found.")
            return
        self._replace_resource_items(self.awg_resource, resources)
        self._replace_resource_items(self.scope_resource, resources)
        self._set_status(f"Found {len(resources)} VISA resource(s).", 4000)

    @staticmethod
    def _replace_resource_items(combo: QtWidgets.QComboBox, resources: tuple[str, ...]) -> None:
        current = combo.currentText().strip()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(list(resources))
        if current and current not in resources:
            combo.addItem(current)
        if current:
            combo.setCurrentText(current)
        combo.blockSignals(False)

    @QtCore.Slot()
    def inspect_resources(self) -> None:
        try:
            lines = inspect_visa_resources(timeout_ms=min(5000, int(self.timeout_ms.value())))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "VISA Inspection", str(exc))
            return
        self._show_text_dialog("VISA Resources", "\n".join(lines))

    @QtCore.Slot()
    def debug_scope(self) -> None:
        if self.mock_mode.isChecked():
            QtWidgets.QMessageBox.information(
                self,
                "Scope Debug",
                "Scope diagnostics are only available with hardware mode enabled.",
            )
            return
        client = TektronixVisaClient()
        try:
            config = self._collect_config()
            client.connect(config)
            data = client.debug_read_scope_data()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Scope Debug", str(exc))
            return
        finally:
            client.disconnect()
        text = "\n".join(f"{key}: {value}" for key, value in data.items())
        self._show_text_dialog("Scope Debug", text)

    def _show_text_dialog(self, title: str, text: str) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(760, 520)
        layout = QtWidgets.QVBoxLayout(dialog)
        text_edit = QtWidgets.QPlainTextEdit(text)
        text_edit.setReadOnly(True)
        copy_button = QtWidgets.QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(text))
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(text_edit, 1)
        layout.addWidget(self._button_row(copy_button, close_button))
        dialog.exec()

    @QtCore.Slot()
    def browse_csv_path(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Choose sweep CSV output",
            self.csv_path.text().strip() or "scan_results.csv",
            "CSV Files (*.csv)",
        )
        if path:
            if not Path(path).suffix:
                path += ".csv"
            self.csv_path.setText(path)

    @QtCore.Slot()
    def start_scan(self) -> None:
        if self.is_running:
            return
        if self._external_hardware_busy and not self.mock_mode.isChecked():
            QtWidgets.QMessageBox.information(
                self,
                "Frequency Sweep",
                "The oscilloscope is currently being used for a waveform capture. "
                "Wait for it to finish or enable mock mode.",
            )
            return
        try:
            config = self._collect_config()
            self._expected_points = self._point_count(config)
            # Validate before creating a thread so input errors remain synchronous.
            SpectrumScanner._validate_config(config)
            self._sweep_values_for_validation(config)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Sweep Input", str(exc))
            return

        self.clear_results(force=True)
        self._active_mode = config.sweep_mode
        self._update_plot_axes()
        self.progress.setRange(0, max(1, self._expected_points))
        self.progress.setValue(0)

        self._scan_worker = SweepWorker(config)
        self._scan_thread = QtCore.QThread(self)
        self._scan_worker.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_worker.point.connect(self._on_point)
        self._scan_worker.completed.connect(self._on_completed)
        self._scan_worker.failed.connect(self._on_failed)
        self._scan_worker.finished.connect(self._scan_thread.quit)
        self._scan_worker.finished.connect(self._scan_worker.deleteLater)
        self._scan_thread.finished.connect(self._scan_thread.deleteLater)
        self._scan_thread.finished.connect(self._on_thread_finished)

        self._set_running_ui(True)
        self._using_hardware = not config.use_mock
        if self._using_hardware:
            self.hardware_busy_changed.emit(True)
        mode_label = self.mode_combo.currentText()
        self._set_status(f"{mode_label} started...", 0)
        self._scan_thread.start()

    @staticmethod
    def _sweep_values_for_validation(config: ScanConfig) -> list[float]:
        if config.sweep_mode == "frequency":
            return SpectrumScanner._build_sweep_values(
                config.total_start_hz, config.total_stop_hz, config.step_size_hz, "Frequency"
            )
        if config.sweep_mode == "amplitude":
            return SpectrumScanner._build_sweep_values(
                config.amp_start_vpp, config.amp_stop_vpp, config.amp_step_vpp, "Amplitude"
            )
        return SpectrumScanner._build_sweep_values(
            config.offset_start_v, config.offset_stop_v, config.offset_step_v, "Offset"
        )

    @classmethod
    def _point_count(cls, config: ScanConfig) -> int:
        return len(cls._sweep_values_for_validation(config))

    @QtCore.Slot()
    def stop_scan(self) -> None:
        if self._scan_worker is not None:
            try:
                self._scan_worker.stop()
            except RuntimeError:
                # The worker may already be queued for deletion while the
                # QThread's finished signal is still pending in the GUI loop.
                pass
            self.stop_button.setEnabled(False)
            self._set_status("Stop requested; waiting for the active instrument operation...", 0)

    @QtCore.Slot(object, int, int)
    def _on_point(self, point: MeasurementPoint, index: int, total: int) -> None:
        self._points.append(point)
        x = np.asarray([entry.sweep_value for entry in self._points], dtype=float)
        y = np.asarray([entry.amplitude_dbm for entry in self._points], dtype=float)
        self.plot_panel.set_data(x, y, auto_range=True)
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(index)

        if point.sweep_mode == "frequency":
            if math.isclose(point.target_freq_hz, point.sweep_value, rel_tol=0.0, abs_tol=1e-12):
                current = f"AWG f={point.sweep_value:.6g} Hz"
            else:
                current = (
                    f"AWG f={point.sweep_value:.6g} Hz, "
                    f"evaluation f={point.target_freq_hz:.6g} Hz"
                )
        elif point.sweep_mode == "amplitude":
            current = f"Vpp={point.sweep_value:.6g} V"
        else:
            current = f"offset={point.sweep_value:.6g} V"
        self.last_point_label.setText(
            f"Step {index}/{total}: {current}, measured amplitude={point.amplitude_dbm:.3f} dBm"
        )
        self._update_best_point()
        self._set_status(self.last_point_label.text(), 0)

    @QtCore.Slot(object, bool)
    def _on_completed(self, points: list[MeasurementPoint], cancelled: bool) -> None:
        self._points = list(points)
        self._update_best_point()
        if cancelled:
            message = f"Sweep stopped after {len(points)} of {self._expected_points} point(s)."
        else:
            message = f"Sweep completed with {len(points)} point(s)."
            self.progress.setValue(self.progress.maximum())
        self._set_status(message, 5000)

    @QtCore.Slot(str)
    def _on_failed(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Sweep Error", message)
        self._set_status(f"Sweep failed: {message}", 8000)

    @QtCore.Slot()
    def _on_thread_finished(self) -> None:
        self._scan_thread = None
        self._scan_worker = None
        used_hardware = self._using_hardware
        self._using_hardware = False
        self._set_running_ui(False)
        if used_hardware:
            self.hardware_busy_changed.emit(False)

    def _set_running_ui(self, running: bool) -> None:
        for group in self._settings_groups:
            group.setEnabled(not running)
        self.mock_mode.setEnabled(not running)
        self.auto_save_csv.setEnabled(not running)
        self.csv_path.setEnabled(not running)
        self.browse_csv_button.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.clear_button.setEnabled(not running)
        self.export_button.setEnabled(not running and bool(self._points))
        self.use_best_button.setEnabled(not running and self.best_frequency_hz() is not None)

    def _set_status(self, message: str, timeout_ms: int) -> None:
        self.run_status.setText(message)
        self.status_message.emit(message, timeout_ms)

    def _update_best_point(self) -> None:
        finite_points = [point for point in self._points if math.isfinite(point.amplitude_dbm)]
        if not finite_points:
            self.best_point_label.setText("Best: n/a")
            self.use_best_button.setEnabled(False)
            self.export_button.setEnabled(bool(self._points) and not self.is_running)
            return

        best = max(finite_points, key=lambda point: point.amplitude_dbm)
        if best.sweep_mode == "frequency":
            detail = f"AWG frequency={best.sweep_value:.9g} Hz"
            if not math.isclose(
                best.target_freq_hz,
                best.sweep_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                detail += f", evaluated at {best.target_freq_hz:.9g} Hz"
        elif best.sweep_mode == "amplitude":
            detail = f"AWG amplitude={best.sweep_value:.9g} Vpp at {best.target_freq_hz:.9g} Hz"
        else:
            detail = f"AWG offset={best.sweep_value:.9g} V at {best.target_freq_hz:.9g} Hz"
        self.best_point_label.setText(f"Best: {detail}, amplitude={best.amplitude_dbm:.3f} dBm")
        self.use_best_button.setEnabled(not self.is_running and best.sweep_mode == "frequency")
        self.export_button.setEnabled(not self.is_running)

    def best_frequency_hz(self) -> float | None:
        finite_points = [
            point
            for point in self._points
            if point.sweep_mode == "frequency" and math.isfinite(point.amplitude_dbm)
        ]
        if not finite_points:
            return None
        return float(max(finite_points, key=lambda point: point.amplitude_dbm).sweep_value)

    @QtCore.Slot()
    def use_best_frequency(self) -> None:
        frequency_hz = self.best_frequency_hz()
        if frequency_hz is None:
            QtWidgets.QMessageBox.information(self, "Frequency Sweep", "No frequency result is available.")
            return
        self.frequency_selected.emit(frequency_hz)

    @QtCore.Slot()
    def clear_results(self, force: bool = False) -> None:
        if self.is_running and not force:
            return
        self._points.clear()
        self.plot_panel.clear()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.last_point_label.setText("No measurements yet.")
        self.best_point_label.setText("Best: n/a")
        self.export_button.setEnabled(False)
        self.use_best_button.setEnabled(False)

    @QtCore.Slot()
    def export_csv(self) -> bool:
        if not self._points:
            QtWidgets.QMessageBox.information(self, "Sweep Export", "There are no sweep results to save.")
            return False
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save sweep results",
            self.csv_path.text().strip() or "scan_results.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return False
        if not Path(path).suffix:
            path += ".csv"
        try:
            SpectrumScanner.write_csv(path, self._points)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Sweep Export", str(exc))
            return False
        self.csv_path.setText(path)
        self._set_status(f"Sweep results saved to {path}", 5000)
        return True

    def export_graph(self) -> bool:
        if not self._points:
            QtWidgets.QMessageBox.information(self, "Sweep Export", "There is no sweep graph to export.")
            return False
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export sweep graph",
            "",
            "PNG Image (*.png);;SVG Image (*.svg)",
        )
        if not path:
            return False
        suffix = Path(path).suffix.lower()
        if not suffix:
            path += ".svg" if "svg" in selected_filter.lower() else ".png"
            suffix = Path(path).suffix.lower()
        try:
            if suffix == ".png":
                exporter = pg.exporters.ImageExporter(self.plot_panel.plot.plotItem)
            elif suffix == ".svg":
                exporter = pg.exporters.SVGExporter(self.plot_panel.plot.plotItem)
            else:
                raise ValueError("Use .png or .svg for graph export.")
            exporter.export(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Sweep Export", str(exc))
            return False
        self._set_status(f"Sweep graph exported to {path}", 5000)
        return True

    def set_scope_resource(self, resource: str) -> None:
        resource = resource.strip()
        if resource and not self.is_running:
            if self.scope_resource.findText(resource) < 0:
                self.scope_resource.addItem(resource)
            self.scope_resource.setCurrentText(resource)

    @QtCore.Slot(bool)
    def set_external_hardware_busy(self, busy: bool) -> None:
        self._external_hardware_busy = bool(busy)
        self._update_external_hardware_state()

    @QtCore.Slot()
    def _update_external_hardware_state(self) -> None:
        hardware_blocked = self._external_hardware_busy and not self.mock_mode.isChecked()
        if not self.is_running:
            self.start_button.setEnabled(not hardware_blocked)
        self.debug_scope_button.setEnabled(not self._external_hardware_busy and not self.is_running)

    def shutdown(self) -> bool:
        """Request a safe stop; return False while instrument work is still active."""
        if not self.is_running:
            return True
        self.stop_scan()
        return False
