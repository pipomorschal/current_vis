"""USB TEC temperature sine generation with optional scope and continuous DMM logging."""
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

from tec_setpoint_process import set_tec_isolated, read_tec_temperature_isolated, TecWriteCancelled
from tec_output_worker import TecShutdownWorker
from tec_sine import TecSineSettings, TecSineSweep
from tec_temperature_sweep import TecTemperatureSweep
from tec_sine_recording import SineRunState, SineLogSettings, SineRecordingWorker
from tec_sine_trace import TecSineTraceRecorder
from signal_data_import import ScopeCaptureConfig
from keithley2000 import (Keithley2000, KeithleySettings, MeterCancelled, FUNCTIONS, BAUD_RATES,
                          RECOVERABLE_METER_ERRORS)


class KeithleyReadWorker(QtCore.QThread):
    reading = QtCore.Signal(object)
    status = QtCore.Signal(str)
    outcome = QtCore.Signal(str)
    diagnostics = QtCore.Signal(str)
    logging_failed = QtCore.Signal(str)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.cancelled = threading.Event()
        self.recording_state = None

    def cancel(self):
        self.cancelled.set()

    def _event(self, client, message):
        details = str(client.diagnostics)
        self.diagnostics.emit(details)
        state = self.recording_state
        if state is not None:
            state.record_event("DMM", message, details)

    def _read_with_recovery(self, client):
        try:
            return client.read()
        except RECOVERABLE_METER_ERRORS as exc:
            error = exc
        budget = max(30, 12 * self.settings.timeout_seconds)
        deadline = time.monotonic() + budget
        state = self.recording_state
        if state is not None:
            state.recovering_meter(budget)
        for attempt, delay in enumerate((0.5, 1.0, 2.0), 1):
            message = f"Meter communication error: {error}. Recovery attempt {attempt}/3."
            self.status.emit(message)
            # Save the failed exchange before clearing/reopening overwrites it.
            self._event(client, message)
            if self.cancelled.wait(delay):
                raise MeterCancelled()
            if time.monotonic() >= deadline:
                break
            try:
                client.recover(reopen=attempt > 1 or
                               (isinstance(error, OSError) and not isinstance(error, TimeoutError)))
                if time.monotonic() >= deadline:
                    break
                value = client.read()
                self._event(client, f"Meter communication recovered on attempt {attempt}.")
                self.status.emit(f"Connected: {client.identity}. Communication recovered.")
                return value
            except RECOVERABLE_METER_ERRORS as exc:
                error = exc
        raise RuntimeError(f"Keithley serial recovery failed after bounded retries: {error}") from error

    def run(self):
        client = Keithley2000(self.settings, self.cancelled)
        message = "Meter stopped. Front-panel control restored."
        samples = 0
        try:
            client.connect()
            self.status.emit(f"Connected: {client.identity}")
            self._event(client, f"Connected: {client.identity}")
            started = time.monotonic()
            tick = 0
            while not self.cancelled.is_set():
                due = started + tick * self.settings.interval_seconds
                if self.cancelled.wait(max(0, due - time.monotonic())):
                    break
                value = self._read_with_recovery(client)
                now = time.monotonic()
                row = dict(value=value, unit=self.settings.unit, function=self.settings.function,
                           elapsed_seconds=now - started, monotonic=now,
                           port=self.settings.port, identity=client.identity,
                           timestamp=datetime.now().astimezone().isoformat(timespec="milliseconds"))
                recording_state = self.recording_state
                if recording_state is not None:
                    try:
                        recording_state.set_meter(row)
                    except Exception as exc:
                        self.logging_failed.emit(f"DMM sample could not be saved: {exc}")
                        raise
                self.reading.emit(row)
                samples += 1
                tick = max(tick + 1, math.floor((now - started) / self.settings.interval_seconds) + 1)
        except MeterCancelled:
            pass
        except Exception as exc:
            message = f"Meter stopped on error: {exc}. TEC sine control is independent."
        finally:
            self._event(client, f"{message} Completed readings: {samples}.")
            recording_state = self.recording_state
            if recording_state is not None:
                recording_state.fail_meter(message)
            try:
                client.close()
            except Exception as exc:
                message += f" Serial close failed: {exc}"
            self.diagnostics.emit(str(client.diagnostics))
        self.outcome.emit(message)


class TecTemperatureWorker(QtCore.QThread):
    reading = QtCore.Signal(object)

    def __init__(self, resource, timeout_ms=5000, parent=None, *, target=None):
        super().__init__(parent)
        self.resource, self.timeout_ms = resource, timeout_ms
        self.cancelled = threading.Event()
        self.target = target

    def cancel(self):
        self.cancelled.set()

    def run(self):
        try:
            if self.target is None:
                result = read_tec_temperature_isolated(self.resource, timeout_ms=self.timeout_ms, cancelled=self.cancelled)
            else:
                result = set_tec_isolated(self.resource, self.target, self.target, self.target,
                                          timeout_ms=self.timeout_ms, cancelled=self.cancelled,
                                          enable_output=True, read_temperature=True)
        except TecWriteCancelled:
            return
        except Exception as exc:
            result = dict(tec_temperature_status="error", tec_temperature_error=str(exc))
            if self.target is not None:
                result["tec_manual_setpoint_error"] = str(exc)
        if not self.cancelled.is_set():
            if self.target is not None:
                result = dict(result, tec_manual_setpoint=self.target)
            self.reading.emit(dict(result, tec_resource=self.resource))


class TecSineWorker(QtCore.QThread):
    applied = QtCore.Signal(object)
    outcome = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, settings, resource, timeout_ms=5000, parent=None, *, state=None, duration=None,
                 auto_output=False):
        super().__init__(parent)
        self.settings, self.resource, self.timeout_ms = settings, resource, timeout_ms
        self.cancelled = threading.Event()
        self.recording_state = state
        self.duration = duration if duration is not None else getattr(settings, "duration_seconds", None)
        self.auto_output = auto_output

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
                result = set_tec_isolated(self.resource, target, s.minimum, s.maximum,
                                         timeout_ms=self.timeout_ms, cancelled=self.cancelled,
                                         enable_output=self.auto_output, read_temperature=True)
                if self.cancelled.is_set():
                    break
                finished = time.monotonic()
                next_tick = math.floor((finished - started) / s.interval_seconds) + 1
                point = dict(elapsed_seconds=elapsed, target=target, io_seconds=finished - command_started,
                             skipped=max(0, next_tick - math.floor(elapsed / s.interval_seconds) - 1),
                             verified_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
                             verified_monotonic=finished, tec_resource=self.resource)
                if isinstance(result, dict):
                    point.update({key: value for key, value in result.items() if key.startswith("tec_temperature")})
                if self.recording_state is not None:
                    self.recording_state.set_target(point, started)
                self.applied.emit(point)
                # Skip missed ticks rather than issuing a burst of stale setpoints.
                due_elapsed = next_tick * s.interval_seconds
                if isinstance(s, TecSineSweep):
                    due_elapsed = min(due_elapsed, (s.offsets.step_at(finished - started) + 1) * s.offsets.step_seconds)
                if self.duration is not None:
                    due_elapsed = min(due_elapsed, self.duration)
            ending = " TEC output shutdown follows the final capture." if self.auto_output else " The TEC holds its last commanded setpoint."
            self.outcome.emit(("Stopped." if self.cancelled.is_set() else "Run complete.") + ending)
        except TecWriteCancelled:
            self.outcome.emit("Stopped. TEC output shutdown pending." if self.auto_output else
                              "Stopped. The TEC holds its last commanded setpoint.")
        except Exception as exc:
            self.failed.emit(str(exc))
            self.outcome.emit(f"Stopped on error: {exc}." + (" TEC output shutdown pending." if self.auto_output else
                              " The TEC holds its last commanded setpoint."))


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
        self._tec_shutdown = None
        self._shutdown_worker = None
        self._temperature_worker = None
        self._last_temperature = None
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
        temperature_row = QtWidgets.QHBoxLayout()
        self.temperature_value = QtWidgets.QLabel("TEC temperature: unavailable")
        self.temperature_value.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.temperature_value.setWordWrap(True)
        self.temperature_value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.temperature_refresh_button = QtWidgets.QPushButton("Refresh temperature")
        self.temperature_refresh_button.clicked.connect(self._refresh_temperature)
        temperature_row.addWidget(self.temperature_value, 1)
        temperature_row.addWidget(QtWidgets.QLabel("Setpoint:"))
        self.single_temperature = QtWidgets.QDoubleSpinBox()
        self.single_temperature.setRange(-273.15, 1000)
        self.single_temperature.setDecimals(3)
        self.single_temperature.setValue(25)
        self.single_temperature.setSuffix(" °C")
        self.single_temperature.setMaximumWidth(120)
        self.single_temperature.setKeyboardTracking(False)
        self.single_temperature.setToolTip("Apply one temperature setpoint. A later sine or sweep can overwrite it.")
        self.single_temperature_button = QtWidgets.QPushButton("Set")
        self.single_temperature_button.setToolTip("Apply this setpoint once and enable the TEC output; hold until another command changes it.")
        self.single_temperature_button.clicked.connect(self._set_single_temperature)
        temperature_row.addWidget(self.single_temperature)
        temperature_row.addWidget(self.single_temperature_button)
        temperature_row.addWidget(self.temperature_refresh_button)
        layout.addLayout(temperature_row)
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
            "The TEC must be in temperature-control mode. Its configured limits must contain "
            "the complete sine range. Actual temperature follows the TEC's thermal response. "
            "Automatic output control enables the TEC at Start and disables it after the final capture, Stop or error.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self._temperature_timer = QtCore.QTimer(self)
        self._temperature_timer.setInterval(5000)
        self._temperature_timer.timeout.connect(self._poll_temperature)
        self.resource.currentTextChanged.connect(self._temperature_resource_changed)
        self._update_preview()

    def _temperature_resource_changed(self):
        self._last_temperature = None
        self.temperature_value.setText("TEC temperature: unavailable")
        self.temperature_value.setToolTip("Select the ITC4005 USB resource to read its measured temperature.")
        self._update_temperature_controls()

    def _update_temperature_controls(self):
        available = (
            self.resource.currentText().strip().upper().startswith("USB") and not self.external_busy
            and self.worker is None and self.recording_worker is None
            and self._shutdown_worker is None and self._temperature_worker is None)
        self.temperature_refresh_button.setEnabled(available)
        self.single_temperature_button.setEnabled(available)

    def _poll_temperature(self):
        if self.isVisible():
            self._refresh_temperature()

    def _refresh_temperature(self):
        self._start_temperature_request()

    def _set_single_temperature(self):
        self.single_temperature.interpretText()
        self._start_temperature_request(target=self.single_temperature.value())

    def _start_temperature_request(self, *, target=None):
        self._update_temperature_controls()
        if not self.temperature_refresh_button.isEnabled():
            return
        self._temperature_worker = TecTemperatureWorker(self.resource.currentText().strip(), self.timeout.value(), self,
                                                       target=target)
        self._temperature_worker.reading.connect(self._on_temperature_reading)
        self._temperature_worker.finished.connect(self._temperature_finished)
        self.temperature_value.setToolTip("Reading the measured TEC temperature…")
        if target is not None:
            self.status_label.setText(f"Setting TEC temperature to {target:.3f} °C…")
        self.hardware_busy_changed.emit(True)
        self._update_preview()
        self._temperature_worker.start()

    def _on_temperature_reading(self, result):
        if result.get("tec_resource") != self.resource.currentText().strip():
            return
        if "tec_manual_setpoint" in result:
            if result.get("tec_sweep_setpoint_status") == "verified":
                self.status_label.setText(f"TEC setpoint: {result['tec_manual_setpoint']:.3f} °C. Output enabled.")
            elif "tec_manual_setpoint_error" in result:
                self.status_label.setText(f"Could not set TEC temperature: {result['tec_manual_setpoint_error']}")
        value = result.get("tec_temperature")
        if result.get("tec_temperature_status") == "ok" and isinstance(value, (int, float)) and math.isfinite(value):
            self._last_temperature = value
            timestamp = result.get("tec_temperature_read_at", "")
            try:
                timestamp = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
            except (ValueError, TypeError):
                timestamp = ""
            self.temperature_value.setText(f"TEC temperature: {value:.3f} °C" + (f" (read {timestamp})" if timestamp else ""))
            self.temperature_value.setToolTip("Measured TEC temperature from the ITC4005 sensor.")
        elif result.get("tec_temperature_status") == "error":
            previous = f" (last: {self._last_temperature:.3f} °C)" if self._last_temperature is not None else ""
            self.temperature_value.setText("TEC temperature: unavailable" + previous)
            self.temperature_value.setToolTip(result.get("tec_temperature_error", "Temperature reading failed"))

    def _temperature_finished(self):
        worker = self._temperature_worker
        self._temperature_worker = None
        if worker.target is not None and worker.cancelled.is_set():
            self.status_label.setText("Single temperature command stopped.")
        self.hardware_busy_changed.emit(False)
        self._update_preview()
        worker.deleteLater()

    def showEvent(self, event):
        super().showEvent(event)
        self._temperature_timer.start()
        QtCore.QTimer.singleShot(0, self._poll_temperature)

    def hideEvent(self, event):
        self._temperature_timer.stop()
        if self._temperature_worker is not None and self._temperature_worker.target is None:
            self._temperature_worker.cancel()
        super().hideEvent(event)

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
        self.auto_tec_output = QtWidgets.QCheckBox("TEC output ON at start / OFF after run")
        self.auto_tec_output.setChecked(True)
        form.addRow(self.auto_tec_output)
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
        note = QtWidgets.QLabel("Scope waveforms use the capture interval. A separate TEC/DMM trace saves every completed DMM reading and verified TEC update at its own timestamp. Select DMM voltage, TEC temperature or TEC commanded setpoint in Recordings to display the full traces.")
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
        self.meter_baud.setCurrentText("9600")
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
        self.meter_worker.logging_failed.connect(self._on_recording_failed)
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
            self.start_button.setEnabled(self.worker is None and self.recording_worker is None
                                         and self._shutdown_worker is None and self._temperature_worker is None
                                         and not self.external_busy)
        except ValueError as exc:
            self.summary_label.setText(str(exc))
            self.preview_curve.clear()
            self.marker.clear()
            self._preview_settings = None
            self.start_button.setEnabled(False)
        self._update_temperature_controls()

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
        if (self.worker is not None or self.recording_worker is not None or self._shutdown_worker is not None
                or self._temperature_worker is not None or self.external_busy):
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
        try:
            trace = (TecSineTraceRecorder(log_settings.folder, log_settings.extension, program,
                                          include_dmm=log_settings.include_dmm) if log_settings is not None else None)
        except Exception as exc:
            self.status_label.setText(f"Could not start TEC/DMM trace recording: {exc}")
            return
        self.run_state = SineRunState(trace) if log_settings is not None else None
        if (self.meter_with_sine.isChecked() or (log_settings and log_settings.include_dmm)) and self.meter_worker is None:
            if not self.start_meter():
                if self.run_state is not None:
                    self.run_state.close_trace()
                    self.run_state = None
                self.status_label.setText("Choose valid Keithley settings before starting the sine with meter readings.")
                return
            self._meter_started_with_sine = True
        if self.meter_worker is not None:
            self.meter_worker.recording_state = self.run_state
            if self.run_state is not None and self.meter_readings:
                # Seed waveform pairing with the previous live value, but do
                # not append a reading acquired before this recording began.
                with self.run_state.lock:
                    if self.run_state.meter is None:
                        self.run_state.meter = dict(self.meter_readings[-1])
        self.meter_with_sine.setEnabled(False)
        self.marker.clear()
        self._run_settings = settings
        self._run_program = program
        self.settings_panel.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Starting sine; checking the full temperature range against controller limits…")
        self._tec_shutdown = (resource, self.timeout.value()) if self.auto_tec_output.isChecked() else None
        self.worker = TecSineWorker(program, resource, self.timeout.value(), self, state=self.run_state, duration=duration,
                                    auto_output=self.auto_tec_output.isChecked())
        self.worker.applied.connect(self._on_applied)
        self.worker.outcome.connect(self.status_label.setText)
        self.worker.failed.connect(self._on_tec_failed)
        self.worker.finished.connect(self._finished)
        if log_settings is not None:
            self.recording_worker = SineRecordingWorker(program, config, log_settings, self.run_state, duration, self)
            self.recording_worker.saved.connect(self._on_recording_saved)
            self.recording_worker.status.connect(self.recording_status.setText)
            self.recording_worker.failed.connect(self._on_recording_failed)
            self.recording_worker.finished.connect(self._recording_finished)
            self.recording_status.setText(f"Trace: {trace.path.name}. Waiting for the first verified TEC command…")
            self.recording_worker.start()
        else:
            self.recording_status.setText("Logging is off.")
        self.hardware_busy_changed.emit(True)
        self.worker.start()

    def _on_recording_saved(self, info):
        self.recording_status.setText(f"Saved {info['count']} capture(s). Latest: {info['path']}")

    def _on_recording_failed(self, message):
        if self.run_state is not None:
            self.run_state.record_event("logging", message)
        self._run_error = self._run_error or message
        self.recording_status.setText(f"Logging stopped on error: {message}")
        if self.worker is not None:
            self.worker.cancel()

    def _on_tec_failed(self, message):
        self._run_error = self._run_error or f"TEC control failed: {message}"
        if self.run_state is not None:
            self.run_state.record_event("TEC", message)
        if self.recording_worker is not None:
            self.recording_worker.cancel()
        self.recording_status.setText(f"Run stopped: {self._run_error}")

    def _recording_finished(self):
        worker = self.recording_worker
        self.recording_worker = None
        if worker is not None:
            worker.deleteLater()
        self._release_run_controls()

    def _on_applied(self, point):
        self._on_temperature_reading(point)
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
            self.status_label.setText("Stopping USB updates; waiting for acquisition and TEC output cleanup…")

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
        if self._shutdown_worker is not None:
            return
        if self._tec_shutdown is not None:
            resource, timeout = self._tec_shutdown
            self._tec_shutdown = None
            self._shutdown_worker = TecShutdownWorker(resource, timeout, self)
            self._shutdown_worker.finished.connect(self._output_shutdown_finished)
            self.stop_button.setEnabled(False)
            self.status_label.setText(self.status_label.text() + " Switching TEC output OFF…")
            self._shutdown_worker.start()
            return
        self.settings_panel.setEnabled(True)
        self.meter_with_sine.setEnabled(True)
        if self._meter_started_with_sine:
            self.stop_meter()
            self._meter_started_with_sine = False
        self.stop_button.setEnabled(False)
        if self.meter_worker is not None:
            self.meter_worker.recording_state = None
        if self.run_state is not None:
            try:
                self.run_state.record_event("run", self._run_error or "Recording finished.")
                self.run_state.close_trace()
            except Exception as exc:
                self._run_error = f"TEC/DMM trace close failed: {exc}"
        self.run_state = None
        self.hardware_busy_changed.emit(False)
        self._update_preview()
        if self._run_error:
            self.status_label.setText(f"Run stopped: {self._run_error}")

    def _output_shutdown_finished(self):
        worker = self._shutdown_worker
        self._shutdown_worker = None
        self._release_run_controls()
        self.status_label.setText(self.status_label.text() + " " + worker.result_text())
        worker.deleteLater()

    def set_external_hardware_busy(self, busy):
        self.external_busy = bool(busy)
        self._update_preview()

    def shutdown(self):
        self._temperature_timer.stop()
        if self._temperature_worker is not None:
            self._temperature_worker.cancel()
        if (self.worker is not None or self.meter_worker is not None or self.recording_worker is not None
                or self._shutdown_worker is not None or self._temperature_worker is not None):
            self.stop()
            self.stop_meter()
            return False
        return True
