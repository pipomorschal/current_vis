"""Single RF spectra and interval logging, independent of AWG/TEC controls."""
from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import threading
import time
from uuid import uuid4

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters

from frequency_sweep import list_visa_resources
from rf_capture import (RfCaptureConfig, RfCaptureCancelled, capture_rf_isolated,
                        load_rf_spectrum, save_rf_spectrum)


class RfCaptureWorker(QtCore.QThread):
    captured = QtCore.Signal(object, object)
    status = QtCore.Signal(str)
    outcome = QtCore.Signal(str)

    def __init__(self, config, *, folder=None, extension=".h5", interval=10, duration=3600, parent=None):
        super().__init__(parent)
        self.config, self.folder, self.extension = config, Path(folder) if folder is not None else None, extension
        if self.folder is not None and (not self.folder.is_dir() or extension not in {".h5", ".csv"}
                                       or not math.isfinite(interval) or interval <= 0
                                       or not math.isfinite(duration) or duration < 0):
            raise ValueError("Select a recording folder, HDF5/CSV format and valid logging times")
        self.interval, self.duration = interval, duration
        self.cancelled = threading.Event()

    def cancel(self):
        self.cancelled.set()

    def run(self):
        started, tick, count = time.monotonic(), 0, 0
        session = datetime.now().astimezone().isoformat(timespec="milliseconds")
        run_id = uuid4().hex[:8]
        log = None
        message = "RF capture stopped."
        try:
            if self.folder is not None:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                log = (self.folder / f"rf_capture_{stamp}_{run_id}.log").open("x", encoding="utf-8")
                log.write(f"Session {session}; resource={self.config.resource}; interval={self.interval:g} s\n")
                log.flush()
            while not self.cancelled.is_set():
                due = started + tick * self.interval
                if self.folder is not None and self.duration and due >= started + self.duration:
                    break
                if self.cancelled.wait(max(0, due - time.monotonic())):
                    break
                if self.folder is not None and self.duration and time.monotonic() >= started + self.duration:
                    break
                self.status.emit(f"Acquiring RF spectrum {count + 1}…")
                spectrum = capture_rf_isolated(self.config, self.cancelled)
                if self.cancelled.is_set():
                    break
                count += 1
                spectrum.metadata.update(logging=self.folder is not None, logging_session_started_at=session,
                                         logging_capture_index=count, logging_interval_seconds=self.interval,
                                         logging_maximum_seconds=self.duration)
                path = None
                if self.folder is not None:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    path = self.folder / f"rf_{stamp}_{run_id}_{count:06d}{self.extension}"
                    save_rf_spectrum(path, spectrum)
                    log.write(f"{spectrum.metadata['capture_completed_at']} Saved {path.name}\n")
                    log.flush()
                self.captured.emit(spectrum, path)
                if self.folder is None:
                    break
                # Preserve the original cadence, skipping slots missed during I/O.
                tick = max(tick + 1, math.floor((time.monotonic() - started) / self.interval) + 1)
                self.status.emit(f"Saved {count} RF spectrum(s). Waiting for next capture…")
            message = (f"RF logging {'stopped' if self.cancelled.is_set() else 'complete'}; {count} spectra saved."
                       if self.folder is not None else "RF capture complete.")
        except RfCaptureCancelled:
            message = f"RF acquisition cancelled; {count} spectra saved." if self.folder is not None else "RF capture cancelled."
        except Exception as exc:
            message = f"RF {'logging' if self.folder is not None else 'capture'} stopped on error: {exc}"
        finally:
            if log is not None:
                try:
                    log.write(f"{datetime.now().astimezone().isoformat()} {message}\n")
                    log.close()
                except OSError as exc:
                    message += f" Diagnostic log close failed: {exc}"
        self.outcome.emit(message)


class RfCaptureWidget(QtWidgets.QWidget):
    hardware_busy_changed = QtCore.Signal(bool)
    use_time_inputs = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker = None
        self.external_busy = False
        self._spectrum = None
        self.control_panel = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(self.control_panel)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        settings = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(settings)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.resource = QtWidgets.QComboBox()
        self.resource.setEditable(True)
        self.resource.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.resource.setMinimumContentsLength(12)
        form.addRow("RF scope VISA", self.resource)
        self.refresh_button = QtWidgets.QPushButton("Find VISA resources")
        self.refresh_button.clicked.connect(self.refresh_resources)
        form.addRow(self.refresh_button)
        self.copy_button = QtWidgets.QPushButton("Use Time tab scope")
        self.copy_button.clicked.connect(self.use_time_inputs.emit)
        form.addRow(self.copy_button)
        self.timeout = QtWidgets.QSpinBox()
        self.timeout.setRange(1000, 120000)
        self.timeout.setValue(10000)
        self.timeout.setSuffix(" ms")
        form.addRow("VISA timeout", self.timeout)
        self.acquisition_wait = QtWidgets.QDoubleSpinBox()
        self.acquisition_wait.setRange(1, 3600)
        self.acquisition_wait.setValue(30)
        self.acquisition_wait.setSuffix(" s")
        form.addRow("Acquisition wait", self.acquisition_wait)
        self.use_scope_settings = QtWidgets.QCheckBox("Use scope RF settings")
        self.use_scope_settings.setChecked(True)
        form.addRow(self.use_scope_settings)
        self.rf_settings = QtWidgets.QGroupBox("RF window")
        rf_form = QtWidgets.QFormLayout(self.rf_settings)
        rf_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.center = self._number(0.000001, 100000, 100, " MHz", 6)
        self.span = self._number(0.000001, 100000, 10, " MHz", 6)
        self.rbw = self._number(.001, 1e9, 100000, " Hz", 3)
        for label, control in (("Center", self.center), ("Span", self.span), ("RBW", self.rbw)):
            rf_form.addRow(label, control)
        form.addRow(self.rf_settings)
        self.use_scope_settings.toggled.connect(self._update_controls)
        self.logging_group = QtWidgets.QGroupBox("Logging mode")
        self.logging_group.setCheckable(True)
        self.logging_group.setChecked(False)
        logging_form = QtWidgets.QFormLayout(self.logging_group)
        logging_form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.interval = self._number(.1, 86400, 10, " s", 2)
        self.maximum_minutes = self._number(0, 1000000, 60, " min", 2)
        self.maximum_minutes.setSpecialValueText("Until Stop")
        self.folder = QtWidgets.QLineEdit()
        self.folder_button = QtWidgets.QPushButton("Browse…")
        self.folder_button.clicked.connect(self._choose_folder)
        self.format = QtWidgets.QComboBox()
        self.format.addItem("HDF5 (.h5)", ".h5")
        self.format.addItem("CSV (.csv)", ".csv")
        logging_form.addRow("Capture interval", self.interval)
        logging_form.addRow("Maximum time", self.maximum_minutes)
        logging_form.addRow("Output folder", self.folder)
        logging_form.addRow(self.folder_button)
        logging_form.addRow("Format", self.format)
        form.addRow(self.logging_group)
        self.logging_group.toggled.connect(self._update_controls)
        scroll.setWidget(settings)
        outer.addWidget(scroll, 1)
        self._settings_panel = settings
        self.start_button = QtWidgets.QPushButton("Capture RF spectrum")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.save_button = QtWidgets.QPushButton("Save spectrum…")
        self.open_button = QtWidgets.QPushButton("Open RF spectrum…")
        self.png_button = QtWidgets.QPushButton("Export plot PNG…")
        for button, slot in ((self.start_button, self.start), (self.stop_button, self.stop),
                             (self.save_button, self.save), (self.open_button, self.open),
                             (self.png_button, self.export_png)):
            outer.addWidget(button)
            button.clicked.connect(slot)
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("RF Capture")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        self.status = QtWidgets.QLabel("Ready. Configure the RF input and trigger on the oscilloscope.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.info)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "RF power", units="dBm")
        self.plot.showGrid(x=True, y=True, alpha=.25)
        for side in ("bottom", "left"):
            self.plot.getAxis(side).setPen(pg.mkPen("#333333"))
            self.plot.getAxis(side).setTextPen(pg.mkPen("#333333"))
        self.curve = self.plot.plot(pen=pg.mkPen("#2563eb", width=1.5), connect="finite")
        layout.addWidget(self.plot, 1)
        self._update_controls()

    @staticmethod
    def _number(low, high, value, suffix, decimals):
        control = QtWidgets.QDoubleSpinBox()
        control.setDecimals(decimals)
        control.setRange(low, high)
        control.setValue(value)
        control.setSuffix(suffix)
        control.setKeyboardTracking(False)
        control.setMaximumWidth(180)
        return control

    @property
    def is_busy(self):
        return self.worker is not None

    def set_external_hardware_busy(self, busy):
        self.external_busy = bool(busy)
        self._update_controls()

    def _update_controls(self, *_):
        active = self.is_busy
        self._settings_panel.setEnabled(not active and not self.external_busy)
        self.rf_settings.setEnabled(not self.use_scope_settings.isChecked())
        self.start_button.setEnabled(not active and not self.external_busy)
        self.start_button.setText("Start RF logging" if self.logging_group.isChecked() else "Capture RF spectrum")
        self.stop_button.setEnabled(active)
        self.open_button.setEnabled(not active)
        self.save_button.setEnabled(self._spectrum is not None and not active)
        self.png_button.setEnabled(self._spectrum is not None and not active)

    def refresh_resources(self):
        try:
            selected = self.resource.currentText()
            resources = list_visa_resources()
            self.resource.clear()
            self.resource.addItems(resources)
            if selected:
                self.resource.setEditText(selected)
            self.status.setText(f"Found {len(resources)} VISA resource(s).")
        except Exception as exc:
            self.status.setText(f"VISA discovery failed: {exc}")

    def _choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "RF logging folder", self.folder.text())
        if folder:
            self.folder.setText(folder)

    def start(self):
        if self.is_busy or self.external_busy:
            return
        try:
            config = RfCaptureConfig(self.resource.currentText().strip(), self.timeout.value(),
                                     self.acquisition_wait.value(), self.use_scope_settings.isChecked(),
                                     self.center.value() * 1e6, self.span.value() * 1e6, self.rbw.value())
            folder = None
            if self.logging_group.isChecked():
                if not self.folder.text().strip() or not Path(self.folder.text().strip()).is_dir():
                    raise ValueError("Select an existing RF logging folder")
                folder = Path(self.folder.text().strip())
            self.worker = RfCaptureWorker(config, folder=folder, extension=self.format.currentData(),
                                          interval=self.interval.value(), duration=self.maximum_minutes.value() * 60,
                                          parent=self)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.worker.captured.connect(self._captured)
        self.worker.status.connect(self.status.setText)
        self.worker.outcome.connect(self.status.setText)
        self.worker.finished.connect(self._finished)
        self._update_controls()
        self.hardware_busy_changed.emit(True)
        self.worker.start()

    def stop(self):
        if self.worker is not None:
            self.worker.cancel()
            self.stop_button.setEnabled(False)
            self.status.setText("Stopping RF acquisition; completing any file save already in progress…")

    def _finished(self):
        worker, self.worker = self.worker, None
        worker.deleteLater()
        self.hardware_busy_changed.emit(False)
        self._update_controls()

    def _captured(self, spectrum, path):
        self._spectrum = spectrum
        self.curve.setData(spectrum.frequency_hz, spectrum.power_dbm)
        self.plot.autoRange()
        m = spectrum.metadata
        finite = np.isfinite(spectrum.power_dbm)
        peak = ""
        if np.any(finite):
            index = int(np.argmax(np.where(finite, spectrum.power_dbm, -np.inf)))
            peak = f" | Peak {spectrum.power_dbm[index]:.3f} dBm @ {spectrum.frequency_hz[index]:.9g} Hz"
        self.info.setText(f"{m.get('capture_completed_at', '')} | {len(spectrum.frequency_hz):,} points"
                          f" | Center {float(m['center_hz']):.9g} Hz | Span {float(m['span_hz']):.9g} Hz"
                          f" | RBW {float(m['rbw_hz']):.9g} Hz{peak}")
        self.plot.setTitle(path.name if path else "RF spectrum", color="#333333")
        self._update_controls()

    def save(self):
        if self._spectrum is None:
            return
        path, selected = QtWidgets.QFileDialog.getSaveFileName(self, "Save RF spectrum", "rf_spectrum.h5",
                                                              "HDF5 (*.h5 *.hdf5);;CSV (*.csv)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".csv" if selected.startswith("CSV") else ".h5"
        try:
            save_rf_spectrum(path, self._spectrum)
            self.status.setText(f"Saved {path}")
        except Exception as exc:
            self.status.setText(f"RF save failed: {exc}")

    def open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open RF spectrum", self.folder.text(),
                                                       "RF spectra (*.h5 *.hdf5 *.csv)")
        if path:
            try:
                self._captured(load_rf_spectrum(path), Path(path))
                self.status.setText(f"Loaded {path}")
            except Exception as exc:
                self.status.setText(f"RF load failed: {exc}")

    def export_png(self):
        if self._spectrum is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export RF plot", "rf_spectrum.png", "PNG (*.png)")
        if path:
            try:
                if not Path(path).suffix:
                    path += ".png"
                pg.exporters.ImageExporter(self.plot.plotItem).export(path)
                self.status.setText(f"Exported {path}")
            except Exception as exc:
                self.status.setText(f"PNG export failed: {exc}")

    def shutdown(self):
        if self.is_busy:
            self.stop()
            return False
        return True
