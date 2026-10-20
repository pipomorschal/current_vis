"""USB TEC temperature sine generation, without waveform acquisition or recording."""
from __future__ import annotations

import math
import threading
import time

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from tec_setpoint_process import set_tec_isolated, TecWriteCancelled
from tec_sine import TecSineSettings


class TecSineWorker(QtCore.QThread):
    applied = QtCore.Signal(object)
    outcome = QtCore.Signal(str)

    def __init__(self, settings, resource, timeout_ms=5000, parent=None):
        super().__init__(parent)
        self.settings, self.resource, self.timeout_ms = settings, resource, timeout_ms
        self.cancelled = threading.Event()

    def cancel(self):
        self.cancelled.set()

    def run(self):
        s = self.settings
        started = time.monotonic()
        tick = 0
        try:
            while not self.cancelled.is_set():
                due = started + tick * s.interval_seconds
                if self.cancelled.wait(max(0.0, due - time.monotonic())):
                    break
                command_started = time.monotonic()
                elapsed = command_started - started
                target = s.target_at(elapsed)
                set_tec_isolated(self.resource, target, s.minimum, s.maximum,
                                 timeout_ms=self.timeout_ms, cancelled=self.cancelled)
                if self.cancelled.is_set():
                    break
                finished = time.monotonic()
                next_tick = max(tick + 1, math.floor((finished - started) / s.interval_seconds) + 1)
                self.applied.emit(dict(elapsed_seconds=elapsed, target=target,
                                       io_seconds=finished - command_started, skipped=next_tick - tick - 1))
                # Skip missed ticks rather than issuing a burst of stale setpoints.
                tick = next_tick
            self.outcome.emit("Stopped. The TEC holds its last commanded setpoint.")
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
            "Only temperature setpoints are sent to the ITC4005 over USB; no recording is performed.")
        description.setWordWrap(True)
        layout.addWidget(description)
        self.status_label = QtWidgets.QLabel("Ready. Select the ITC4005 USB resource and press Start sine.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
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
        copy = QtWidgets.QPushButton("Use Time tab TEC")
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
            "1 mHz = 0.001 Hz = a 1000 s period. Choose at least 20 setpoint updates per period. "
            "Runs continuously until Stop. Timing follows elapsed time; delayed updates are skipped.")
        instructions.setWordWrap(True)
        form.addRow(instructions)
        scroll.setWidget(self.settings_panel)
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
            self.summary_label.setText(
                f"Period: {s.period_seconds:g} s ({s.period_seconds / 60:g} min)   |   "
                f"Range: {s.minimum:g}–{s.maximum:g} °C   |   "
                f"{s.period_seconds / s.interval_seconds:g} updates per period")
            x = np.linspace(0, s.period_seconds, 501)
            self.preview_curve.setData(x, s.center + s.amplitude * np.sin(2 * np.pi * x / s.period_seconds))
            if getattr(self, "_preview_settings", None) != s:
                self.marker.clear()
            self._preview_settings = s
            self.start_button.setEnabled(self.worker is None and not self.external_busy)
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
            if selected:
                self.resource.setEditText(selected)
            if not resources:
                self.status_label.setText("No USB VISA resources found. You can enter the ITC4005 resource manually.")
        except Exception as exc:
            self.status_label.setText(f"USB discovery failed: {exc}")

    def start(self):
        if self.worker is not None or self.external_busy:
            return
        try:
            settings = self.settings()
            resource = self.resource.currentText().strip()
            if not resource.upper().startswith("USB"):
                raise ValueError("Select or enter the ITC4005 USB VISA resource")
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self.marker.clear()
        self._run_settings = settings
        self.settings_panel.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Starting sine; checking the full temperature range against controller limits…")
        self.worker = TecSineWorker(settings, resource, self.timeout.value(), self)
        self.worker.applied.connect(self._on_applied)
        self.worker.outcome.connect(self.status_label.setText)
        self.worker.finished.connect(self._finished)
        self.hardware_busy_changed.emit(True)
        self.worker.start()

    def _on_applied(self, point):
        s = self._run_settings
        phase_seconds = point["elapsed_seconds"] % s.period_seconds
        self.marker.setData([phase_seconds], [point["target"]])
        text = (f"Running — last verified setpoint: {point['target']:.3f} °C   |   "
                f"Elapsed: {point['elapsed_seconds'] / 60:.2f} min   |   USB write/readback: {point['io_seconds']:.2f} s")
        if point["skipped"]:
            text += f"   |   Skipped {point['skipped']} delayed update(s); consider a longer update interval."
        self.status_label.setText(text)

    def stop(self):
        if self.worker is not None:
            self.worker.cancel()
            self.stop_button.setEnabled(False)
            self.status_label.setText("Stopping USB updates; retaining the last commanded TEC setpoint…")

    def _finished(self):
        worker = self.worker
        self.worker = None
        self.settings_panel.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.hardware_busy_changed.emit(False)
        self._update_preview()
        if worker is not None:
            worker.deleteLater()

    def set_external_hardware_busy(self, busy):
        self.external_busy = bool(busy)
        self._update_preview()

    def shutdown(self):
        if self.worker is not None:
            self.stop()
            return False
        return True
