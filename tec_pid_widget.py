"""Slow TEC working-point optimization with cancellable, isolated VISA acquisitions."""
from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime
import math
from pathlib import Path
import threading
import time

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from scope_capture_process import capture_isolated
from signal_data_class import SignalData
from signal_data_import import OscilloscopeImporter, ScopeCaptureConfig
from tec_pid_control import TecPidSettings, GradientPid, harmonic_amplitudes, calibration_recommendation
from tec_setpoint_process import disable_tec_isolated


class Cancelled(Exception):
    pass


LOG_COLUMNS = ("timestamp", "elapsed_s", "phase", "center_deg_c", "target_deg_c", "measured_deg_c",
               "a50_v_peak", "a100_v_peak", "a100_over_a50", "score", "gradient_per_deg_c")


class TecPidWorker(QtCore.QThread):
    sample = QtCore.Signal(object)
    status = QtCore.Signal(str)
    calibration = QtCore.Signal(object)
    outcome = QtCore.Signal(str)

    def __init__(self, settings, config, folder, simulation=True, calibrate=False, parent=None):
        super().__init__(parent)
        self.settings, self.config = settings, config
        self.folder = Path(folder)
        self.simulation, self.calibrate = simulation, calibrate
        self.cancelled = threading.Event()
        self.scale = 60.0 if simulation else 1.0
        self.normalization = None
        self.center = settings.initial
        self.gradient = None
        self._sim_temperature = settings.initial
        self._sim_time = 0.0
        self._started = 0.0
        self._tec_output_owned = False

    def cancel(self):
        self.cancelled.set()

    def _now(self):
        return (time.monotonic() - self._started) * self.scale

    def _wait(self, seconds):
        if self.cancelled.wait(max(0.0, seconds) / self.scale):
            raise Cancelled()

    def _measure(self, target, settle, phase, require_settled=True):
        if self.cancelled.is_set():
            raise Cancelled()
        s = self.settings
        if not math.isfinite(target) or not s.minimum <= target <= s.maximum:
            raise ValueError("Requested temperature is outside the allowed range")
        self.status.emit(f"{phase}: target {target:.3f} °C; settling {settle:g} s, then waiting for a fresh trigger…")
        if self.simulation:
            command_time = self._now()
            previous_target = getattr(self, "_sim_target", self.settings.initial)
            self._sim_temperature = previous_target + (self._sim_temperature - previous_target) * math.exp(
                -(command_time - self._sim_time) / 60)
            self._sim_target = target
            self._wait(settle + 1)
            now = self._now()
            self._sim_temperature = target + (self._sim_temperature - target) * math.exp(-(now - command_time) / 60)
            self._sim_time = now
            temperature = self._sim_temperature
            a50 = 1 + 0.8 * math.exp(-((temperature - 26) / 2) ** 2)
            a100 = 0.05 + 0.06 * (temperature - 26.2) ** 2
            t = np.arange(1000) / 2000
            data = SignalData(t, a50 * np.sin(2 * np.pi * 50 * t + 0.3)
                              + a100 * np.sin(2 * np.pi * 100 * t - 0.2), sampling_rate=2000)
        else:
            self._tec_output_owned = self.config.auto_tec_output
            config = replace(self.config, tec_setpoint_deg_c=target, tec_settle_seconds=settle,
                             fresh_trigger=True)
            data = capture_isolated(config, cancelled=self.cancelled)
            if self.cancelled.is_set():
                raise Cancelled()
            if data.metadata.get("tec_temperature_status") != "ok":
                raise RuntimeError("TEC temperature read failed; optimization stopped")
            temperature = float(data.metadata["tec_temperature"])
        if not math.isfinite(temperature) or not s.minimum <= temperature <= s.maximum:
            raise RuntimeError("Measured TEC temperature is outside the allowed range")
        if require_settled and abs(temperature - target) > s.temperature_tolerance:
            raise RuntimeError(f"TEC is {temperature:.3f} °C at target {target:.3f} °C; increase settling time or check the TEC")
        a50, a100 = harmonic_amplitudes(data)
        if not all(math.isfinite(value) for value in (a50, a100)):
            raise RuntimeError("Harmonic fit returned non-finite amplitudes")
        if a50 < s.minimum_a50:
            raise RuntimeError("50 Hz amplitude is below the configured signal floor; check scope input and waveform")
        if self.normalization is None:
            self.normalization = a50
        score = (a50 - s.harmonic_weight * a100) / self.normalization
        if not math.isfinite(score):
            raise RuntimeError("Normalized objective is non-finite; check the signal and harmonic weight")
        row = dict(zip(LOG_COLUMNS, (datetime.now().astimezone().isoformat(timespec="milliseconds"),
                                    self._now(), phase, self.center, target, temperature,
                                    a50, a100, a100 / a50, score, self.gradient)))
        self.writer.writerow(row)
        self.log_file.flush()
        self.sample.emit(row)
        return row

    def _probe_measure(self, target, phase):
        # Interval is a minimum start-to-start period; I/O and settling can extend it.
        self._wait(self._last_start + self.settings.interval_seconds - self._now())
        self._last_start = self._now()
        return self._measure(target, self.settings.settle_seconds, phase)

    def _run_control(self):
        s = self.settings
        controller = GradientPid(s)
        self._probe_measure(self.center, "Baseline")
        previous_update = self._now()
        for cycle in range(1, s.max_cycles + 1):
            low = self._probe_measure(self.center - s.probe, f"Cycle {cycle}: low probe")
            high = self._probe_measure(self.center + s.probe, f"Cycle {cycle}: high probe")
            self.gradient = (high["score"] - low["score"]) / (2 * s.probe)
            now = self._now()
            self.center = controller.update(self.gradient, max(0.001, now - previous_update))
            previous_update = now
            # Always return to the center, including the final cycle.
            self._probe_measure(self.center, f"Cycle {cycle}: PID center")

    def _run_calibration(self):
        s = self.settings
        baseline = self._probe_measure(self.center, "Calibration: baseline")
        step_started = self._now()
        response = [(0.0, baseline["score"])]
        # Capture the transient rather than waiting out the whole thermal response.
        high = self._measure(self.center + s.probe, s.interval_seconds,
                             "Calibration: positive step", require_settled=False)
        response.append((self._now() - step_started, high["score"]))
        while self._now() - step_started < s.settle_seconds:
            self._wait(min(s.interval_seconds, s.settle_seconds - (self._now() - step_started)))
            high = self._measure(self.center + s.probe, 0, "Calibration: step response", require_settled=False)
            response.append((self._now() - step_started, high["score"]))
        if abs(high["measured_deg_c"] - high["target_deg_c"]) > s.temperature_tolerance:
            raise RuntimeError("Calibration step did not settle; increase observation/settling time")
        low = self._probe_measure(self.center - s.probe, "Calibration: negative step")
        returned = self._probe_measure(self.center, "Calibration: return to start")
        result = calibration_recommendation(s, baseline["score"], low["score"], high["score"], response)
        if abs(returned["score"] - baseline["score"]) > 0.05:
            result["note"] += " Baseline drift exceeded 5% of initial A50; repeat calibration after stabilizing the setup."
        self.calibration.emit(result)
        return result

    def run(self):
        self._started = time.monotonic()
        self._last_start = -self.settings.interval_seconds
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            prefix = "simulation" if self.simulation else "hardware"
            path = self.folder / f"tec_pid_{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
            with path.open("x", newline="", encoding="utf-8") as self.log_file:
                self.writer = csv.DictWriter(self.log_file, fieldnames=LOG_COLUMNS)
                self.writer.writeheader()
                # Preserve settings with the scalar log for reproducibility.
                import json
                from dataclasses import asdict
                path.with_suffix(".json").write_text(json.dumps({
                    "settings": asdict(self.settings), "instruments": asdict(self.config),
                    "simulation": self.simulation, "calibration": self.calibrate,
                    "objective": "(A50 - weight*A100) / initial_A50",
                }, indent=2), encoding="utf-8")
                self.status.emit(f"Scalar log: {path}")
                if self.calibrate:
                    result = self._run_calibration()
                    path.with_name(path.stem + "_calibration.json").write_text(
                        json.dumps(result, indent=2), encoding="utf-8")
                else:
                    self._run_control()
            message = f"{'Calibration' if self.calibrate else 'Optimization'} complete. Log: {path}"
        except Cancelled:
            message = "Stopped."
        except Exception as exc:
            if self.cancelled.is_set():
                message = "Stopped."
            else:
                message = f"Stopped on error: {exc}."
        finally:
            if self._tec_output_owned and not self.simulation:
                self.status.emit("Switching TEC output OFF…")
                try:
                    disable_tec_isolated(self.config.itc4005_resource, timeout_ms=self.config.timeout_ms)
                    output_message = " TEC output OFF verified."
                except Exception as exc:
                    output_message = f" TEC output OFF could not be verified: {exc}. Check the controller."
            else:
                output_message = "" if self.simulation else " TEC retains its last commanded setpoint."
        self.outcome.emit(message + output_message)


class TecPidWidget(QtWidgets.QWidget):
    hardware_busy_changed = QtCore.Signal(bool)
    use_time_inputs = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.external_busy = False
        self.rows = []
        self.recommendation = None
        self._hardware_owned = False
        self._build_controls()
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("TEC working-point optimization")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        description = QtWidgets.QLabel(
            "Maximize 50 Hz and suppress 100 Hz: J = (A50 − weight × A100) / initial A50. "
            "Small temperature probes estimate dJ/dT; PID moves the center uphill toward a local optimum. "
            "The ITC4005 retains its own temperature-control loop.")
        description.setWordWrap(True)
        layout.addWidget(description)
        self.status_label = QtWidgets.QLabel("Ready — simulation runs 60× faster. Hardware starts only after Simulation is unchecked and Start is pressed.")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status_label)
        self.readings = QtWidgets.QLabel("A50: —    A100: —    TEC: —")
        self.readings.setWordWrap(True)
        layout.addWidget(self.readings)
        self.graphs = pg.GraphicsLayoutWidget()
        self.graphs.setBackground("w")
        self.amplitude_plot = self.graphs.addPlot(title="Harmonic peak amplitudes")
        self.amplitude_plot.setLabel("left", "Amplitude", "V")
        self.amplitude_plot.addLegend(labelTextColor="#222222")
        self.a50_curve = self.amplitude_plot.plot(pen=pg.mkPen("#2474bc", width=2), name="50 Hz")
        self.a100_curve = self.amplitude_plot.plot(pen=pg.mkPen("#e6842a", width=2), name="100 Hz")
        self.graphs.nextRow()
        self.temperature_plot = self.graphs.addPlot(title="TEC temperature and probe commands")
        self.temperature_plot.setLabel("left", "Temperature", "°C")
        self.temperature_plot.addLegend(labelTextColor="#222222")
        self.target_curve = self.temperature_plot.plot(pen=pg.mkPen("#8b52b5", width=1), name="Command")
        self.measured_curve = self.temperature_plot.plot(pen=pg.mkPen("#2c9b68", width=2), name="Measured")
        self.center_curve = self.temperature_plot.plot(pen=pg.mkPen("#666666", style=QtCore.Qt.PenStyle.DashLine), name="PID center")
        self.graphs.nextRow()
        self.score_plot = self.graphs.addPlot(title="Normalized objective (higher is better)")
        self.score_curve = self.score_plot.plot(pen=pg.mkPen("#2474bc", width=2))
        for plot in (self.amplitude_plot, self.temperature_plot, self.score_plot):
            plot.setTitle(plot.titleLabel.text, color="#222222")
            for side in ("left", "bottom"):
                plot.getAxis(side).setPen(pg.mkPen("#333333"))
                plot.getAxis(side).setTextPen(pg.mkPen("#333333"))
            plot.setLabel("bottom", "Elapsed", "min")
            plot.showGrid(x=True, y=True, alpha=0.2)
        layout.addWidget(self.graphs, 1)
        self.calibration_label = QtWidgets.QLabel("Calibration: measure the baseline, a positive step response and a negative probe, then return to start. Estimates are local; choose a sufficient observation time. This finds a local weighted optimum; a wide temperature sweep can help choose a starting point.")
        self.calibration_label.setWordWrap(True)
        layout.addWidget(self.calibration_label)

    def _build_controls(self):
        self.control_panel = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(self.control_panel)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.settings_panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.settings_panel)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.simulation = QtWidgets.QCheckBox("Simulation (no hardware, 60×)")
        self.simulation.setChecked(True)
        self.simulation.toggled.connect(lambda: self.set_external_hardware_busy(self.external_busy))
        form.addRow(self.simulation)
        self.scope_resource = QtWidgets.QComboBox()
        self.tec_resource = QtWidgets.QComboBox()
        for combo in (self.scope_resource, self.tec_resource):
            combo.setEditable(True)
            combo.setMinimumWidth(80)
            combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        form.addRow("Scope VISA", self.scope_resource)
        form.addRow("ITC4005 VISA", self.tec_resource)
        resource_buttons = QtWidgets.QHBoxLayout()
        refresh = QtWidgets.QPushButton("Find VISA")
        refresh.clicked.connect(self._refresh_resources)
        copy = QtWidgets.QPushButton("Use Time tab inputs")
        copy.setMinimumWidth(0)
        refresh.setMinimumWidth(0)
        copy.clicked.connect(self.use_time_inputs.emit)
        resource_buttons.addWidget(refresh)
        resource_buttons.addWidget(copy)
        form.addRow(resource_buttons)
        self.channel = QtWidgets.QComboBox()
        self.channel.addItems(["CH1", "CH2", "CH3", "CH4"])
        form.addRow("Scope channel", self.channel)
        self.points = QtWidgets.QSpinBox()
        self.points.setRange(1000, 10000000)
        self.points.setValue(100000)
        self.points.setSingleStep(1000)
        form.addRow("Transfer points", self.points)
        self.timeout = QtWidgets.QSpinBox()
        self.timeout.setRange(1000, 120000)
        self.timeout.setValue(10000)
        self.timeout.setSuffix(" ms")
        form.addRow("VISA timeout", self.timeout)
        self.trigger_wait = QtWidgets.QDoubleSpinBox()
        self.trigger_wait.setRange(1, 600)
        self.trigger_wait.setValue(30)
        self.trigger_wait.setSuffix(" s")
        form.addRow("Trigger wait limit", self.trigger_wait)
        self.auto_tec_output = QtWidgets.QCheckBox("TEC output ON at start / OFF after run")
        self.auto_tec_output.setChecked(True)
        form.addRow(self.auto_tec_output)
        instruction = QtWidgets.QLabel("Configure scope timebase (≥0.2 s transferred), scale, trigger source/level first. Every read arms a new normal-triggered single sequence. No forced trigger.")
        instruction.setWordWrap(True)
        form.addRow(instruction)
        self.fields = {}
        labels = {
            "minimum": ("Minimum TEC", " °C", -273.15, 1000, 3),
            "maximum": ("Maximum TEC", " °C", -273.15, 1000, 3),
            "initial": ("Starting TEC", " °C", -273.15, 1000, 3),
            "probe": ("Probe ±", " °C", 0.001, 10, 3),
            "max_move": ("Max center move / cycle", " °C", 0.001, 10, 3),
            "settle_seconds": ("Settle / step observation", " s", 1, 86400, 1),
            "interval_seconds": ("Minimum read interval", " s", 1, 86400, 1),
            "temperature_tolerance": ("Settled TEC tolerance", " °C", 0.001, 10, 3),
            "harmonic_weight": ("100 Hz penalty weight", "", 0, 1000, 3),
            "kp": ("Kp (°C²)", "", 0, 1000000, 6),
            "ki": ("Ki (°C²/s)", "", 0, 1000, 8),
            "kd": ("Kd (°C²·s)", "", 0, 1000000, 6),
            "gradient_deadband": ("Gradient deadband (1/°C)", "", 0, 100, 6),
            "minimum_a50": ("Minimum valid A50", " V peak", 1e-9, 1000, 9),
        }
        defaults = TecPidSettings()
        for key, (label, suffix, low, high, decimals) in labels.items():
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(low, high)
            spin.setValue(getattr(defaults, key))
            spin.setSuffix(suffix)
            self.fields[key] = spin
            form.addRow(label, spin)
        self.cycles = QtWidgets.QSpinBox()
        self.cycles.setRange(1, 100000)
        self.cycles.setValue(defaults.max_cycles)
        form.addRow("Optimization cycles", self.cycles)
        self.timing_label = QtWidgets.QLabel()
        self.timing_label.setWordWrap(True)
        for key in ("settle_seconds", "interval_seconds"):
            self.fields[key].valueChanged.connect(self._update_timing)
        self._update_timing()
        form.addRow(self.timing_label)
        gain_note = QtWidgets.QLabel("Initial guess: Kp=0.1, Ki=Kd=0. Add I/D only after measuring dynamics. Each cycle includes low probe, high probe and center; gains act on normalized dJ/dT.")
        gain_note.setWordWrap(True)
        form.addRow(gain_note)
        self.output_folder = QtWidgets.QLineEdit(str(Path.cwd() / "data" / "tec_pid"))
        form.addRow("CSV log folder", self.output_folder)
        browse = QtWidgets.QPushButton("Choose log folder…")
        browse.clicked.connect(self._choose_folder)
        form.addRow(browse)
        final_note = QtWidgets.QLabel("TEC must be in temperature mode. Automatic output control switches it off after completion, Stop or error. Scope settings are restored after a completed read; cancellation during VISA may leave a single sequence armed.")
        final_note.setWordWrap(True)
        form.addRow(final_note)
        scroll.setWidget(self.settings_panel)
        outer.addWidget(scroll, 1)
        self.start_button = QtWidgets.QPushButton("Start optimization")
        self.calibrate_button = QtWidgets.QPushButton("Measure calibration")
        self.apply_button = QtWidgets.QPushButton("Use calibration estimates")
        self.apply_button.setEnabled(False)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setEnabled(False)
        for button in (self.start_button, self.calibrate_button, self.apply_button, self.stop_button):
            outer.addWidget(button)
        self.start_button.clicked.connect(lambda: self.start(False))
        self.calibrate_button.clicked.connect(lambda: self.start(True))
        self.apply_button.clicked.connect(self._apply_calibration)
        self.stop_button.clicked.connect(self.stop)

    def _update_timing(self):
        seconds = 3 * max(self.fields["settle_seconds"].value(), self.fields["interval_seconds"].value())
        self.timing_label.setText(f"At least {seconds / 60:g} min per optimization cycle, plus trigger, transfer and analysis time. Settling begins after the verified TEC write.")

    def _refresh_resources(self):
        try:
            resources = OscilloscopeImporter.list_resources()
            for combo in (self.scope_resource, self.tec_resource):
                selected = combo.currentText()
                combo.clear()
                combo.addItems(resources)
                combo.setEditText(selected)
        except Exception as exc:
            self.status_label.setText(f"VISA discovery failed: {exc}")

    def _choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "TEC PID log folder", self.output_folder.text())
        if folder:
            self.output_folder.setText(folder)

    def settings(self):
        return TecPidSettings(**{key: spin.value() for key, spin in self.fields.items()}, max_cycles=self.cycles.value())

    def start(self, calibrate=False):
        if self.worker is not None or (self.external_busy and not self.simulation.isChecked()):
            return
        try:
            settings = self.settings()
            simulation = self.simulation.isChecked()
            scope, tec = self.scope_resource.currentText().strip(), self.tec_resource.currentText().strip()
            if not simulation and (not scope or not tec or scope == tec):
                raise ValueError("Enter separate oscilloscope and ITC4005 VISA resources")
            if not self.output_folder.text().strip():
                raise ValueError("Choose a CSV log folder")
            if calibrate and settings.settle_seconds < 4 * settings.interval_seconds:
                raise ValueError("Calibration observation time must span at least four read intervals")
            config = ScopeCaptureConfig(scope, channel=self.channel.currentText(), point_count=self.points.value(),
                                        timeout_ms=self.timeout.value(), itc4005_resource=tec,
                                        fresh_trigger=True, trigger_wait_seconds=self.trigger_wait.value(),
                                        auto_tec_output=self.auto_tec_output.isChecked())
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        self.rows.clear()
        self.recommendation = None
        for curve in (self.a50_curve, self.a100_curve, self.target_curve, self.measured_curve, self.center_curve, self.score_curve):
            curve.clear()
        self.calibration_label.setText("Calibration pending." if calibrate else "Optimization running; calibration is optional.")
        self._hardware_owned = not simulation
        self.settings_panel.setEnabled(False)
        self.start_button.setEnabled(False)
        self.calibrate_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker = TecPidWorker(settings, config, self.output_folder.text().strip(), simulation, calibrate, self)
        self.worker.sample.connect(self._on_sample)
        self.worker.status.connect(self.status_label.setText)
        self.worker.calibration.connect(self._on_calibration)
        self.worker.outcome.connect(self.status_label.setText)
        self.worker.finished.connect(self._finished)
        if self._hardware_owned:
            self.hardware_busy_changed.emit(True)
        self.worker.start()

    def _on_sample(self, row):
        self.rows.append(row)
        x = [r["elapsed_s"] / 60 for r in self.rows]
        for curve, key in ((self.a50_curve, "a50_v_peak"), (self.a100_curve, "a100_v_peak"),
                           (self.target_curve, "target_deg_c"), (self.measured_curve, "measured_deg_c"),
                           (self.center_curve, "center_deg_c"), (self.score_curve, "score")):
            curve.setData(x, [r[key] for r in self.rows])
        gradient = row["gradient_per_deg_c"]
        gradient_text = "—" if gradient is None else f"{gradient:.5g} /°C"
        self.readings.setText(f"A50: {row['a50_v_peak']:.6g} V peak    A100: {row['a100_v_peak']:.6g} V peak    "
                              f"100/50: {row['a100_over_a50']:.3%}\n"
                              f"TEC: {row['measured_deg_c']:.3f} °C    Center: {row['center_deg_c']:.3f} °C    "
                              f"Score: {row['score']:.5g}    Gradient: {gradient_text}")

    def _on_calibration(self, result):
        self.recommendation = (result, self.simulation.isChecked())
        tau = result["tau_seconds"]
        tau_text = "unresolved" if tau is None else f"{tau:.1f} s (includes acquisition delay)"
        self.calibration_label.setText(
            f"{'Simulation' if self.simulation.isChecked() else 'Hardware'} calibration: "
            f"Kp={result['kp']:.6g}, Ki=0, Kd=0; response time ≈ {tau_text}; "
            f"suggested dwell ≥ {result['settle_seconds']:.1f} s. {result['note']}")

    def _apply_calibration(self):
        if self.recommendation is None:
            return
        result, simulation = self.recommendation
        if simulation != self.simulation.isChecked():
            self.status_label.setText("Calibration belongs to a different mode. Measure a hardware calibration before applying hardware estimates.")
            return
        for key in ("kp", "ki", "kd", "settle_seconds"):
            self.fields[key].setValue(result[key])
        self.status_label.setText("Calibration estimates applied. Review settings before starting.")

    def stop(self):
        if self.worker is not None:
            self.worker.cancel()
            self.stop_button.setEnabled(False)
            self.status_label.setText("Stopping acquisition; waiting for TEC output cleanup…")

    def _finished(self):
        worker = self.worker
        self.worker = None
        self.settings_panel.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.apply_button.setEnabled(self.recommendation is not None)
        if self._hardware_owned:
            self._hardware_owned = False
            self.hardware_busy_changed.emit(False)
        self.set_external_hardware_busy(self.external_busy)
        if worker is not None:
            worker.deleteLater()

    def set_external_hardware_busy(self, busy):
        self.external_busy = bool(busy)
        available = self.worker is None and (not busy or self.simulation.isChecked())
        self.start_button.setEnabled(available)
        self.calibrate_button.setEnabled(available)

    def shutdown(self):
        if self.worker is not None:
            self.stop()
            return False
        return True
