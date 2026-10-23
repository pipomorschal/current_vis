"""USB TEC temperature sine generation, without waveform acquisition or recording."""
from __future__ import annotations

import math
from collections import deque
from datetime import datetime
from pathlib import Path
import threading
import time

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from tec_setpoint_process import set_tec_isolated, TecWriteCancelled
from tec_sine import TecSineSettings, TecSineSweep
from tec_temperature_sweep import TecTemperatureSweep
from tec_sine_recording import SineRunState, SineLogSettings, SineRecordingWorker
from signal_data_import import ScopeCaptureConfig
from keithley2000 import Keithley2000, KeithleySettings, MeterCancelled, FUNCTIONS, BAUD_RATES


class KeithleyReadWorker(QtCore.QThread):
    reading = QtCore.Signal(object)
    status = QtCore.Signal(str)
    outcome = QtCore.Signal(str)
    diagnostics = QtCore.Signal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.cancelled = threading.Event()
        self.recording_state = None

    def cancel(self):
        self.cancelled.set()

    def run(self):
        client = Keithley2000(self.settings, self.cancelled)
        message = "Meter stopped. Front-panel control restored."
        try:
            client.connect()
            self.status.emit(f"Connected: {client.identity}")
            started = time.monotonic()
            tick = 0
            while not self.cancelled.is_set():
                due = started + tick * self.settings.interval_seconds
                if self.cancelled.wait(max(0, due - time.monotonic())):
                    break
                value = client.read()
                if self.cancelled.is_set():
                    break
                now = time.monotonic()
                row = dict(value=value, unit=self.settings.unit, function=self.settings.function,
                           elapsed_seconds=now - started, monotonic=now,
                           port=self.settings.port, identity=client.identity,
                           timestamp=datetime.now().astimezone().isoformat(timespec="milliseconds"))
                if self.recording_state is not None:
                    self.recording_state.set_meter(row)
                self.reading.emit(row)
                tick = max(tick + 1, math.floor((now - started) / self.settings.interval_seconds) + 1)
        except MeterCancelled:
            pass
        except Exception as exc:
            message = f"Meter stopped on error: {exc}. TEC sine control is independent."
        finally:
            if self.recording_state is not None:
                self.recording_state.fail_meter(message)
            try:
                client.close()
            except Exception as exc:
                message += f" Serial close failed: {exc}"
            self.diagnostics.emit(str(client.diagnostics))
        self.outcome.emit(message)


class TecSineWorker(QtCore.QThread):
    applied = QtCore.Signal(object)
    outcome = QtCore.Signal(str)

    def __init__(self, settings, resource, timeout_ms=5000, parent=None, *, state=None, duration=None):
        super().__init__(parent)
        self.settings, self.resource, self.timeout_ms = settings, resource, timeout_ms
        self.cancelled = threading.Event()
        self.recording_state = state
        self.duration = duration if duration is not None else getattr(settings, "duration_seconds", None)

    def cancel(self):
        self.cancelled.set()

    def run(self):
        s = self.settings
        started = time.monotonic()
        due_elapsed = 0.0
        try:
            while not self.cancelled.is_set():
                due = started + due_elapsed
                if self.cancelled.wait(max(0.0, due - time.monotonic())):
                    break
                command_started = time.monotonic()
                elapsed = command_started - started
                if self.duration is not None and elapsed >= self.duration:
                    break
                target = s.target_at(elapsed)
                set_tec_isolated(self.resource, target, s.minimum, s.maximum,
                                 timeout_ms=self.timeout_ms, cancelled=self.cancelled)
                if self.cancelled.is_set():
                    break
                finished = time.monotonic()
                next_tick = math.floor((finished - started) / s.interval_seconds) + 1
                point = dict(elapsed_seconds=elapsed, target=target, io_seconds=finished - command_started,
                             skipped=max(0, next_tick - math.floor(elapsed / s.interval_seconds) - 1),
                             verified_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
                             verified_monotonic=finished)
                if self.recording_state is not None:
                    self.recording_state.set_target(point, started)
                self.applied.emit(point)
                # Skip missed ticks rather than issuing a burst of stale setpoints.
                due_elapsed = next_tick * s.interval_seconds
                if isinstance(s, TecSineSweep):
                    due_elapsed = min(due_elapsed, (s.offsets.step_at(finished - started) + 1) * s.offsets.step_seconds)
                if self.duration is not None:
                    due_elapsed = min(due_elapsed, self.duration)
            self.outcome.emit("Stopped. The TEC holds its last commanded setpoint." if self.cancelled.is_set()
                              else "Run complete. The TEC holds its final commanded setpoint.")
        except TecWriteCancelled:
            self.outcome.emit("Stopped. The TEC holds its last commanded setpoint.")
        except Exception as exc:
            self.outcome.emit(f"Stopped on error: {exc}. The TEC holds its last commanded setpoint.")


class TecSineWidget(QtWidgets.QWidget):
    hardware_busy_changed = QtCore.Signal(bool)
    use_time_input = QtCore.Signal()

    FREQUENCY_UNITS = {"Hz": 1.0, "mHz": 1e-3, "µHz": 1e-6}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.recording_worker = None
        self.run_state = None
        self._run_error = None
        self.meter_worker = None
        self.meter_readings = deque(maxlen=5000)
        self._meter_started_with_sine = False
        self.meter_serial_details = ""
        self.external_busy = False
        self._frequency_factor = 1e-3
        self._build_controls()
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("TEC temperature sine")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        description = QtWidgets.QLabel(
            "T(t) = center + amplitude × sin(2π × frequency × t). "
            "Amplitude is the peak deviation in °C. The sine starts at the center and rises. "
            "Temperature setpoints are sent to the ITC4005 over USB. Optional Keithley 2000 readings "
            "can be paired with optical scope waveforms in optional HDF5/CSV logging. "
            "Sweep the DC offset and set amplitude to 0 for a pure DC temperature sweep.")
        description.setWordWrap(True)
        layout.addWidget(description)
        self.status_label = QtWidgets.QLabel("Ready. Select the ITC4005 USB resource and press Start sine.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.recording_status = QtWidgets.QLabel("Logging is off.")
        self.recording_status.setWordWrap(True)
        layout.addWidget(self.recording_status)
        self.plot = pg.PlotWidget(title="Setpoint waveform — one period")
        self.plot.setBackground("w")
        self.plot.setTitle("Setpoint waveform — one period", color="#222222")
        self.plot.setLabel("left", "TEC setpoint", "°C")
        self.plot.setLabel("bottom", "Time within period", "s")
        for side in ("left", "bottom"):
            self.plot.getAxis(side).setPen(pg.mkPen("#333333"))
            self.plot.getAxis(side).setTextPen(pg.mkPen("#333333"))
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend(labelTextColor="#222222")
        self.preview_curve = self.plot.plot(pen=pg.mkPen("#2474bc", width=2), name="Requested sine")
        self.marker = self.plot.plot(pen=None, symbol="o", symbolSize=10,
                                     symbolBrush="#e6842a", symbolPen="#e6842a", name="Last verified setpoint")
        layout.addWidget(self.plot, 1)
        self.meter_value = QtWidgets.QLabel("Keithley 2000: no reading")
        self.meter_value.setStyleSheet("font-size: 17px; font-weight: bold;")
        self.meter_value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.meter_value)
        self.meter_status = QtWidgets.QLabel("Meter is off. Choose the RS232 COM port and function in the sidebar.")
        self.meter_status.setWordWrap(True)
        self.meter_status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.meter_status)
        self.meter_plot = pg.PlotWidget()
        self.meter_plot.setBackground("w")
        self.meter_plot.setTitle("Keithley 2000 readings", color="#222222")
        self.meter_plot.setLabel("bottom", "Elapsed since meter start", "s")
        self.meter_plot.setLabel("left", "DC voltage", "V")
        for side in ("left", "bottom"):
            self.meter_plot.getAxis(side).setPen(pg.mkPen("#333333"))
            self.meter_plot.getAxis(side).setTextPen(pg.mkPen("#333333"))
        self.meter_plot.showGrid(x=True, y=True, alpha=0.25)
        self.meter_curve = self.meter_plot.plot(pen=pg.mkPen("#279569", width=2))
        layout.addWidget(self.meter_plot, 1)
        self.meter_plot.hide()
        note = QtWidgets.QLabel(
            "The TEC must already be enabled in temperature-control mode. Its configured limits must contain "
            "the complete sine range. Actual temperature follows the TEC's thermal response. "
            "Stop leaves the last commanded setpoint in place.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self._update_preview()

    def _build_controls(self):
        self.control_panel = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(self.control_panel)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.settings_panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.settings_panel)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.resource = QtWidgets.QComboBox()
        self.resource.setEditable(True)
        self.resource.setMinimumWidth(0)
        self.resource.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        form.addRow("ITC4005 USB VISA", self.resource)
        buttons = QtWidgets.QHBoxLayout()
        refresh = QtWidgets.QPushButton("Find USB")
        refresh.clicked.connect(self._refresh_resources)
        copy = QtWidgets.QPushButton("Use Time tab inputs")
        copy.clicked.connect(self.use_time_input.emit)
        buttons.addWidget(refresh)
        buttons.addWidget(copy)
        form.addRow(buttons)
        self.center = QtWidgets.QDoubleSpinBox()
        self.center.setDecimals(3)
        self.center.setRange(-273.15, 1000)
        self.center.setValue(25)
        self.center.setSuffix(" °C")
        form.addRow("Center temperature", self.center)
        self.amplitude = QtWidgets.QDoubleSpinBox()
        self.amplitude.setDecimals(3)
        self.amplitude.setRange(0, 1000)
        self.amplitude.setValue(1)
        self.amplitude.setSuffix(" °C peak")
        form.addRow("Amplitude", self.amplitude)
        frequency_row = QtWidgets.QHBoxLayout()
        self.frequency = QtWidgets.QDoubleSpinBox()
        self.frequency.setDecimals(9)
        self.frequency.setRange(0.000001, 1000000)
        self.frequency.setValue(1)
        self.frequency.setMinimumWidth(0)
        self.frequency.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        self.frequency_unit = QtWidgets.QComboBox()
        self.frequency_unit.addItems(self.FREQUENCY_UNITS)
        self.frequency_unit.setCurrentText("mHz")
        self.frequency_unit.setMinimumWidth(60)
        frequency_row.addWidget(self.frequency, 1)
        frequency_row.addWidget(self.frequency_unit)
        form.addRow("Frequency", frequency_row)
        self.interval = QtWidgets.QDoubleSpinBox()
        self.interval.setDecimals(2)
        self.interval.setRange(1, 86400)
        self.interval.setValue(5)
        self.interval.setSuffix(" s")
        form.addRow("Setpoint update interval", self.interval)
        self.timeout = QtWidgets.QSpinBox()
        self.timeout.setRange(1000, 10000)
        self.timeout.setValue(5000)
        self.timeout.setSuffix(" ms")
        form.addRow("USB timeout", self.timeout)
        instructions = QtWidgets.QLabel(
            "1 mHz = 0.001 Hz = a 1000 s period. For nonzero AC amplitude, choose at least 20 setpoint updates per period. "
            "A DC sweep finishes after its final dwell; otherwise the sine runs until Stop or its logging time limit.")
        instructions.setWordWrap(True)
        form.addRow(instructions)
        self._build_sweep_controls(form)
        self._build_logging_controls(form)
        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.addWidget(self.settings_panel)
        self._build_meter_controls(body_layout)
        body_layout.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        self.start_button = QtWidgets.QPushButton("Start sine")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        outer.addWidget(self.start_button)
        outer.addWidget(self.stop_button)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        for spin in (self.center, self.amplitude, self.frequency, self.interval):
            spin.valueChanged.connect(self._update_preview)
        self.frequency_unit.currentTextChanged.connect(self._change_frequency_unit)

    def _build_sweep_controls(self, form):
        self.sweep_group = QtWidgets.QGroupBox("Sweep DC offset")
        self.sweep_group.setCheckable(True)
        self.sweep_group.setChecked(False)
        sweep = QtWidgets.QFormLayout(self.sweep_group)
        for name, label, default, low, high, suffix in (
                ("offset_min", "Minimum offset", 20, -273.15, 1000, " °C"),
                ("offset_max", "Maximum offset", 30, -273.15, 1000, " °C"),
                ("offset_increment", "Offset increment", 1, 0.001, 1000, " °C"),
                ("offset_dwell", "Dwell per offset", 10, 0.017, 10080, " min")):
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(low, high)
            spin.setValue(default)
            spin.setSuffix(suffix)
            spin.valueChanged.connect(self._update_preview)
            setattr(self, name, spin)
            sweep.addRow(label, spin)
        self.offset_repeats = QtWidgets.QSpinBox()
        self.offset_repeats.setRange(1, 100000)
        self.offset_repeats.setValue(1)
        self.offset_repeats.valueChanged.connect(self._update_preview)
        sweep.addRow("Repeat count", self.offset_repeats)
        note = QtWidgets.QLabel("The sine phase stays continuous across offset steps. Includes a full dwell at the maximum offset. Amplitude 0 gives DC steps only.")
        note.setWordWrap(True)
        sweep.addRow(note)
        self.sweep_group.toggled.connect(self._update_preview)
        form.addRow(self.sweep_group)

    def _build_logging_controls(self, form):
        self.logging_group = QtWidgets.QGroupBox("Log optical scope + DMM")
        self.logging_group.setCheckable(True)
        self.logging_group.setChecked(False)
        log = QtWidgets.QFormLayout(self.logging_group)
        log.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.scope_resource = QtWidgets.QComboBox()
        self.scope_resource.setEditable(True)
        self.scope_resource.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        log.addRow("Scope VISA", self.scope_resource)
        self.scope_channel = QtWidgets.QComboBox()
        self.scope_channel.addItems(["CH1", "CH2", "CH3", "CH4"])
        log.addRow("Optical channel", self.scope_channel)
        self.scope_points = QtWidgets.QSpinBox()
        self.scope_points.setRange(100, 10000000)
        self.scope_points.setValue(1000000)
        log.addRow("Transfer points", self.scope_points)
        self.scope_timeout = QtWidgets.QSpinBox()
        self.scope_timeout.setRange(1000, 120000)
        self.scope_timeout.setValue(10000)
        self.scope_timeout.setSuffix(" ms")
        log.addRow("Scope timeout", self.scope_timeout)
        self.scope_trigger_wait = QtWidgets.QDoubleSpinBox()
        self.scope_trigger_wait.setRange(1, 600)
        self.scope_trigger_wait.setValue(30)
        self.scope_trigger_wait.setSuffix(" s")
        log.addRow("Trigger wait", self.scope_trigger_wait)
        self.log_interval = QtWidgets.QDoubleSpinBox()
        self.log_interval.setRange(0.1, 86400)
        self.log_interval.setValue(10)
        self.log_interval.setSuffix(" s")
        log.addRow("Capture interval", self.log_interval)
        self.log_max_minutes = QtWidgets.QDoubleSpinBox()
        self.log_max_minutes.setRange(0, 100800)
        self.log_max_minutes.setValue(60)
        self.log_max_minutes.setSpecialValueText("Until Stop")
        self.log_max_minutes.setSuffix(" min")
        log.addRow("Maximum time (no sweep)", self.log_max_minutes)
        self.log_format = QtWidgets.QComboBox()
        self.log_format.addItem("HDF5", ".h5")
        self.log_format.addItem("CSV", ".csv")
        log.addRow("Format", self.log_format)
        self.log_folder = QtWidgets.QLineEdit()
        self.log_folder.setPlaceholderText("Existing recording directory")
        log.addRow("Folder", self.log_folder)
        browse = QtWidgets.QPushButton("Choose recording folder…")
        browse.clicked.connect(self._select_log_folder)
        log.addRow(browse)
        self.log_dmm = QtWidgets.QCheckBox("Include Keithley voltage")
        self.log_dmm.setChecked(True)
        log.addRow(self.log_dmm)
        note = QtWidgets.QLabel("Uses fresh triggered scope captures. Configure scope scale, timebase and trigger first (≥0.2 s of waveform recommended). Each file includes the verified TEC command and latest valid DMM voltage, with separate timestamps (not simultaneous). DMM is saved once per waveform. Failed scope/save/required-DMM readings stop the run.")
        note.setWordWrap(True)
        log.addRow(note)
        form.addRow(self.logging_group)

    def _select_log_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "TEC sine recordings", self.log_folder.text())
        if folder:
            self.log_folder.setText(folder)

    def program(self):
        sine = self.settings()
        if not self.sweep_group.isChecked():
            return sine
        sweep = TecTemperatureSweep(self.offset_min.value(), self.offset_max.value(), self.offset_increment.value(),
                                    self.offset_dwell.value() * 60, self.offset_repeats.value())
        if sweep.step_seconds < sine.interval_seconds:
            raise ValueError("Offset dwell must be at least the setpoint update interval")
        return TecSineSweep(sine, sweep)

    def _build_meter_controls(self, layout):
        group = QtWidgets.QGroupBox("Keithley 2000 · RS232")
        outer = QtWidgets.QVBoxLayout(group)
        self.meter_with_sine = QtWidgets.QCheckBox("Start/stop meter with sine")
        self.meter_with_sine.toggled.connect(self._show_meter_plot)
        outer.addWidget(self.meter_with_sine)
        self.meter_settings_panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.meter_settings_panel)
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.meter_port = QtWidgets.QComboBox()
        self.meter_port.setEditable(True)
        self.meter_port.setMinimumWidth(0)
        self.meter_port.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        form.addRow("COM port", self.meter_port)
        find = QtWidgets.QPushButton("Find serial ports")
        find.clicked.connect(self._refresh_meter_ports)
        form.addRow(find)
        self.meter_baud = QtWidgets.QComboBox()
        self.meter_baud.addItems([str(rate) for rate in BAUD_RATES])
        self.meter_baud.setCurrentText("4800")
        form.addRow("Baud rate (8N1)", self.meter_baud)
        self.meter_flow = QtWidgets.QComboBox()
        self.meter_flow.addItems(["None", "XON/XOFF"])
        form.addRow("Flow control", self.meter_flow)
        self.meter_terminator = QtWidgets.QComboBox()
        self.meter_terminator.addItems(["Auto", "CR", "CRLF", "LF"])
        self.meter_terminator.setToolTip("Auto tries CR, then retries identification with CRLF and uses the successful terminator for readings.")
        form.addRow("Command terminator", self.meter_terminator)
        self.meter_function = QtWidgets.QComboBox()
        self.meter_function.addItems(FUNCTIONS)
        self.meter_function.setMinimumWidth(0)
        self.meter_function.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        form.addRow("Function", self.meter_function)
        self.meter_interval = QtWidgets.QDoubleSpinBox()
        self.meter_interval.setRange(0.1, 86400)
        self.meter_interval.setValue(1)
        self.meter_interval.setSuffix(" s")
        form.addRow("Read interval", self.meter_interval)
        self.meter_timeout = QtWidgets.QDoubleSpinBox()
        self.meter_timeout.setRange(0.5, 60)
        self.meter_timeout.setValue(5)
        self.meter_timeout.setSuffix(" s")
        form.addRow("Read timeout", self.meter_timeout)
        note = QtWidgets.QLabel("Enable RS232 / SCPI on the meter and match baud/flow settings. "
                                "The selected function uses its default measurement settings. "
                                "AC readings are RMS. Live history holds the last 5000 points in memory.")
        note.setWordWrap(True)
        form.addRow(note)
        outer.addWidget(self.meter_settings_panel)
        buttons = QtWidgets.QHBoxLayout()
        self.meter_start_button = QtWidgets.QPushButton("Start meter")
        self.meter_stop_button = QtWidgets.QPushButton("Stop meter")
        self.meter_stop_button.setEnabled(False)
        self.meter_start_button.clicked.connect(self.start_meter)
        self.meter_stop_button.clicked.connect(self.stop_meter)
        buttons.addWidget(self.meter_start_button)
        buttons.addWidget(self.meter_stop_button)
        outer.addLayout(buttons)
        self.meter_copy_button = QtWidgets.QPushButton("Copy serial details")
        self.meter_copy_button.setEnabled(False)
        self.meter_copy_button.clicked.connect(self._copy_meter_details)
        outer.addWidget(self.meter_copy_button)
        layout.addWidget(group)

    def _copy_meter_details(self):
        QtWidgets.QApplication.clipboard().setText(self.meter_status.text() + "\n" + self.meter_serial_details)

    def _on_meter_diagnostics(self, details):
        self.meter_serial_details = details
        self.meter_copy_button.setEnabled(True)

    def _show_meter_plot(self, enabled):
        if hasattr(self, "meter_plot"):
            self.meter_plot.setVisible(enabled or self.meter_worker is not None or bool(self.meter_readings))

    def _refresh_meter_ports(self):
        try:
            from serial.tools import list_ports
            selected = self.meter_port.currentText()
            self.meter_port.clear()
            for port in list_ports.comports():
                self.meter_port.addItem(port.device)
                self.meter_port.setItemData(self.meter_port.count() - 1, port.description,
                                            QtCore.Qt.ItemDataRole.ToolTipRole)
            if selected:
                self.meter_port.setEditText(selected)
            if self.meter_port.count() == 0:
                self.meter_status.setText("No serial ports found. Enter the adapter's COM port manually.")
        except Exception as exc:
            self.meter_status.setText(f"Serial port discovery failed: {exc}")

    def meter_settings(self):
        return KeithleySettings(port=self.meter_port.currentText().strip(),
                                 baud_rate=int(self.meter_baud.currentText()),
                                 function=self.meter_function.currentText(),
                                 interval_seconds=self.meter_interval.value(),
                                 timeout_seconds=self.meter_timeout.value(),
                                 xonxoff=self.meter_flow.currentText() == "XON/XOFF",
                                 command_terminator=self.meter_terminator.currentText())

    def start_meter(self):
        if self.meter_worker is not None:
            return True
        try:
            settings = self.meter_settings()
        except ValueError as exc:
            self.meter_status.setText(str(exc))
            return False
        self.meter_readings.clear()
        self.meter_curve.clear()
        self.meter_value.setText("Keithley 2000: waiting for first reading…")
        self.meter_plot.setLabel("left", settings.function, settings.unit)
        self.meter_plot.show()
        self.meter_settings_panel.setEnabled(False)
        self.meter_start_button.setEnabled(False)
        self.meter_stop_button.setEnabled(True)
        self.meter_status.setText(f"Connecting to Keithley 2000 on {settings.port}…")
        self.meter_serial_details = ""
        self.meter_copy_button.setEnabled(False)
        self.meter_worker = KeithleyReadWorker(settings, self)
        self.meter_worker.recording_state = self.run_state
        self.meter_worker.reading.connect(self._on_meter_reading)
        self.meter_worker.status.connect(self.meter_status.setText)
        self.meter_worker.outcome.connect(self.meter_status.setText)
        self.meter_worker.diagnostics.connect(self._on_meter_diagnostics)
        self.meter_worker.finished.connect(self._meter_finished)
        self.meter_worker.start()
        return True

    def _on_meter_reading(self, row):
        self.meter_readings.append(row)
        self.meter_value.setText(f"{row['function']}: {row['value']:.9g} {row['unit']}   |   {row['timestamp']}")
        self.meter_curve.setData([point["elapsed_seconds"] for point in self.meter_readings],
                                 [point["value"] for point in self.meter_readings])

    def stop_meter(self):
        if self.meter_worker is not None:
            self.meter_worker.cancel()
            self.meter_stop_button.setEnabled(False)
            self.meter_status.setText("Stopping meter reads…")

    def _meter_finished(self):
        worker = self.meter_worker
        self.meter_worker = None
        self.meter_settings_panel.setEnabled(True)
        self.meter_start_button.setEnabled(True)
        self.meter_stop_button.setEnabled(False)
        if worker is not None:
            worker.deleteLater()

    def settings(self):
        return TecSineSettings(center=self.center.value(), amplitude=self.amplitude.value(),
                               frequency_hz=self.frequency.value() * self.FREQUENCY_UNITS[self.frequency_unit.currentText()],
                               interval_seconds=self.interval.value())

    def _change_frequency_unit(self, unit):
        hz = self.frequency.value() * self._frequency_factor
        self._frequency_factor = self.FREQUENCY_UNITS[unit]
        blocker = QtCore.QSignalBlocker(self.frequency)
        self.frequency.setMinimum(1e-9 / self._frequency_factor)
        self.frequency.setValue(hz / self._frequency_factor)
        del blocker
        self._update_preview()

    def _update_preview(self, *_args):
        # Controls are wired before the plot is created.
        if not hasattr(self, "preview_curve"):
            return
        try:
            s = self.settings()
            program = self.program()
            self.center.setEnabled(not self.sweep_group.isChecked())
            self.log_max_minutes.setEnabled(not self.sweep_group.isChecked())
            self.summary_label.setText(
                f"Period: {s.period_seconds:g} s ({s.period_seconds / 60:g} min)   |   "
                f"Range: {s.minimum:g}–{s.maximum:g} °C   |   "
                f"{s.period_seconds / s.interval_seconds:g} updates per period")
            if isinstance(program, TecSineSweep):
                self.summary_label.setText(
                    f"{program.offsets.steps_per_sweep} offsets × {program.offsets.repeat_count} sweep(s); "
                    f"{program.duration_seconds / 60:g} min total; envelope {program.minimum:g}–{program.maximum:g} °C; "
                    f"AC amplitude ±{s.amplitude:g} °C. Preview starts at the first offset.")
            x = np.linspace(0, s.period_seconds, 501)
            center = program.offsets.minimum if isinstance(program, TecSineSweep) else s.center
            self.preview_curve.setData(x, center + s.amplitude * np.sin(2 * np.pi * x / s.period_seconds))
            if getattr(self, "_preview_settings", None) != program:
                self.marker.clear()
            self._preview_settings = program
            self.start_button.setText("Start offset sweep" if self.sweep_group.isChecked() else "Start sine")
            self.start_button.setEnabled(self.worker is None and self.recording_worker is None and not self.external_busy)
        except ValueError as exc:
            self.summary_label.setText(str(exc))
            self.preview_curve.clear()
            self.marker.clear()
            self._preview_settings = None
            self.start_button.setEnabled(False)

    def _refresh_resources(self):
        try:
            import pyvisa
            manager = pyvisa.ResourceManager()
            try:
                resources = [r for r in manager.list_resources() if r.upper().startswith("USB")]
            finally:
                manager.close()
            selected = self.resource.currentText()
            self.resource.clear()
            self.resource.addItems(resources)
            selected_scope = self.scope_resource.currentText()
            self.scope_resource.clear()
            self.scope_resource.addItems(resources)
            if selected_scope:
                self.scope_resource.setEditText(selected_scope)
            if selected:
                self.resource.setEditText(selected)
            if not resources:
                self.status_label.setText("No USB VISA resources found. You can enter the ITC4005 resource manually.")
        except Exception as exc:
            self.status_label.setText(f"USB discovery failed: {exc}")

    def start(self):
        if self.worker is not None or self.recording_worker is not None or self.external_busy:
            return
        try:
            settings = self.settings()
            program = self.program()
            resource = self.resource.currentText().strip()
            if not resource.upper().startswith("USB"):
                raise ValueError("Select or enter the ITC4005 USB VISA resource")
            log_settings = None
            config = None
            duration = getattr(program, "duration_seconds", None)
            if self.logging_group.isChecked():
                scope = self.scope_resource.currentText().strip()
                if not scope or scope == resource or scope in {"No VISA resource", "pyvisa not installed"}:
                    raise ValueError("Select a separate oscilloscope VISA resource for logging")
                include_dmm = self.log_dmm.isChecked()
                meter = self.meter_worker.settings if self.meter_worker is not None else self.meter_settings() if include_dmm else None
                if include_dmm and meter.unit != "V":
                    raise ValueError("Select DC voltage or AC voltage for the recorded Keithley value")
                log_settings = SineLogSettings(Path(self.log_folder.text().strip()), self.log_interval.value(),
                                               self.log_format.currentData(), include_dmm, self.log_max_minutes.value() * 60,
                                               max(2 * meter.interval_seconds, meter.timeout_seconds) + 1 if meter else 10,
                                               max(15, 4 * meter.timeout_seconds) if meter else 30)
                if not self.log_folder.text().strip():
                    raise ValueError("Choose a recording folder")
                if duration is None and log_settings.maximum_seconds:
                    duration = log_settings.maximum_seconds
                config = ScopeCaptureConfig(scope, channel=self.scope_channel.currentText(),
                                            point_count=self.scope_points.value(), timeout_ms=self.scope_timeout.value(),
                                            fresh_trigger=True, trigger_wait_seconds=self.scope_trigger_wait.value())
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self._meter_started_with_sine = False
        self._run_error = None
        self.run_state = SineRunState() if log_settings is not None else None
        if (self.meter_with_sine.isChecked() or (log_settings and log_settings.include_dmm)) and self.meter_worker is None:
            if not self.start_meter():
                self.status_label.setText("Choose valid Keithley settings before starting the sine with meter readings.")
                return
            self._meter_started_with_sine = True
        if self.meter_worker is not None:
            self.meter_worker.recording_state = self.run_state
            if self.run_state is not None and self.meter_readings:
                self.run_state.set_meter(self.meter_readings[-1])
        self.meter_with_sine.setEnabled(False)
        self.marker.clear()
        self._run_settings = settings
        self._run_program = program
        self.settings_panel.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Starting sine; checking the full temperature range against controller limits…")
        self.worker = TecSineWorker(program, resource, self.timeout.value(), self, state=self.run_state, duration=duration)
        self.worker.applied.connect(self._on_applied)
        self.worker.outcome.connect(self.status_label.setText)
        self.worker.finished.connect(self._finished)
        if log_settings is not None:
            self.recording_worker = SineRecordingWorker(program, config, log_settings, self.run_state, duration, self)
            self.recording_worker.saved.connect(self._on_recording_saved)
            self.recording_worker.status.connect(self.recording_status.setText)
            self.recording_worker.failed.connect(self._on_recording_failed)
            self.recording_worker.finished.connect(self._recording_finished)
            self.recording_status.setText("Logging enabled; waiting for the first verified TEC command…")
            self.recording_worker.start()
        else:
            self.recording_status.setText("Logging is off.")
        self.hardware_busy_changed.emit(True)
        self.worker.start()

    def _on_recording_saved(self, info):
        self.recording_status.setText(f"Saved {info['count']} capture(s). Latest: {info['path']}")

    def _on_recording_failed(self, message):
        self._run_error = message
        self.recording_status.setText(f"Logging stopped on error: {message}")
        if self.worker is not None:
            self.worker.cancel()

    def _recording_finished(self):
        worker = self.recording_worker
        self.recording_worker = None
        if worker is not None:
            worker.deleteLater()
        self._release_run_controls()

    def _on_applied(self, point):
        s = self._run_settings
        phase_seconds = point["elapsed_seconds"] % s.period_seconds
        if isinstance(self._run_program, TecSineSweep):
            center = self._run_program.offsets.target_at(point["elapsed_seconds"])
            x = np.linspace(0, s.period_seconds, 501)
            self.preview_curve.setData(x, center + s.amplitude * np.sin(2 * np.pi * x / s.period_seconds))
        self.marker.setData([phase_seconds], [point["target"]])
        text = (f"Running — last verified setpoint: {point['target']:.3f} °C   |   "
                f"Elapsed: {point['elapsed_seconds'] / 60:.2f} min   |   USB write/readback: {point['io_seconds']:.2f} s")
        if point["skipped"]:
            text += f"   |   Skipped {point['skipped']} delayed update(s); consider a longer update interval."
        self.status_label.setText(text)

    def stop(self):
        if self.recording_worker is not None:
            self.recording_worker.cancel()
            self.recording_status.setText("Stopping scope logging; completing any file save already in progress…")
            self.stop_button.setEnabled(False)
        if self._meter_started_with_sine:
            self.stop_meter()
        if self.worker is not None:
            self.worker.cancel()
            self.stop_button.setEnabled(False)
            self.status_label.setText("Stopping USB updates; retaining the last commanded TEC setpoint…")

    def _finished(self):
        worker = self.worker
        self.worker = None
        if self.recording_worker is not None:
            self.recording_worker.finish_after_capture()
        if worker is not None:
            worker.deleteLater()
        self._release_run_controls()

    def _release_run_controls(self):
        # Keep the shared instrument lock until the final scope process and save
        # have finished, even when the TEC sweep has already completed.
        if self.worker is not None or self.recording_worker is not None:
            return
        self.settings_panel.setEnabled(True)
        self.meter_with_sine.setEnabled(True)
        if self._meter_started_with_sine:
            self.stop_meter()
            self._meter_started_with_sine = False
        self.stop_button.setEnabled(False)
        if self.meter_worker is not None:
            self.meter_worker.recording_state = None
        self.run_state = None
        self.hardware_busy_changed.emit(False)
        self._update_preview()
        if self._run_error:
            self.status_label.setText(f"Run stopped because logging failed: {self._run_error}")

    def set_external_hardware_busy(self, busy):
        self.external_busy = bool(busy)
        self._update_preview()

    def shutdown(self):
        if self.worker is not None or self.meter_worker is not None or self.recording_worker is not None:
            self.stop()
            self.stop_meter()
            return False
        return True
