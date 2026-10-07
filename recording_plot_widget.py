from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from data_manager_signal_loader import DataManager
from recording_analysis import recording_amplitude, recording_amplitudes


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(path))]


def recording_timestamp(path: Path, metadata: dict) -> float | None:
    """Use capture metadata, then the timestamp embedded in recording names."""
    try:
        timestamp = datetime.fromisoformat(metadata["logging_capture_started_at"]).timestamp()
        if np.isfinite(timestamp):
            return timestamp
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        pass
    # GUI names: scope_YYYYMMDD_HHMMSS_mmm_CH1_capture-index.h5.
    match = re.search(r"(?<!\d)(\d{8})_(\d{6})(?:_(\d{3,6})(?!\d))?(?!\d)", path.stem)
    if match:
        try:
            date = datetime.strptime(f"{match[1]}_{match[2]}", "%Y%m%d_%H%M%S")
            if match[3]:
                date = date.replace(microsecond=int(match[3].ljust(6, "0")))
            return date.timestamp()
        except (ValueError, OverflowError, OSError):
            pass
    return None


def optional_recording_values(data, highpass, lowpass):
    values = {"ambient": np.nan, "tec": np.nan, "reference": np.nan, "warnings": []}
    for prefix, key in (("ambient", "ambient"), ("tec", "tec")):
        try:
            if data.metadata.get(f"{prefix}_temperature_status", "ok") != "ok":
                continue
            value = float(data.metadata[f"{prefix}_temperature"])
            if np.isfinite(value):
                values[key] = value
        except (KeyError, ValueError, TypeError):
            pass
        try:
            values[f"{key}_timestamp"] = datetime.fromisoformat(
                data.metadata[f"{prefix}_temperature_read_at"]).timestamp()
        except (KeyError, ValueError, TypeError, OverflowError):
            pass
    if data.reference is not None:
        try:
            values["reference"] = recording_amplitude(data.reference, highpass, lowpass)
        except Exception as exc:
            values["warnings"].append(f"Reference amplitude: {exc}")
    return values


class RecordingTimeAxis(pg.AxisItem):
    """Display the local capture date and time stored in recording filenames."""

    def __init__(self):
        self.capture_time = False
        self.time_format = "HH:MM:SS"
        self._ready = False
        super().__init__(orientation="bottom")
        self._ready = True
        self.enableAutoSIPrefix(False)

    def set_capture_time(self, enabled):
        self.capture_time = enabled
        self.picture = None
        self._update_label()
        self.update()

    def _update_label(self):
        self.setLabel(f"Capture time ({self.time_format})"
                      if self.capture_time else "File index", units="")

    def setRange(self, minimum, maximum):
        span = abs(maximum - minimum)
        self.time_format = "HH:MM:SS" if span < 3600 else "HH:MM" if span < 86400 else "DD Mon HH:MM"
        super().setRange(minimum, maximum)
        if self._ready:
            self._update_label()

    def tickValues(self, minVal, maxVal, size):
        # Generate ticks in the displayed resolution to avoid repeated labels.
        unit = {"HH:MM:SS": 1, "HH:MM": 60, "DD Mon HH:MM": 3600}[self.time_format] if self.capture_time else 1
        levels = super().tickValues(minVal / unit, maxVal / unit, size)
        if self.capture_time and unit > 1:
            levels = [(spacing, values) for spacing, values in levels if spacing >= 1]
        return [(spacing * unit, [value * unit for value in values]) for spacing, values in levels]

    def tickStrings(self, values, scale, spacing):
        if not self.capture_time:
            return super().tickStrings(values, scale, spacing)
        labels = []
        date_format = "%d %b " if self.time_format == "DD Mon HH:MM" else ""
        time_format = "%H:%M:%S" if spacing < 60 else "%H:%M"
        for value in values:
            date = datetime.fromtimestamp(value)
            label = date.strftime(date_format + time_format)
            if 0 < spacing < 1:
                decimals = min(6, max(1, int(np.ceil(-np.log10(spacing)))))
                label += f".{date.microsecond:06d}"[:decimals + 1]
            labels.append(label)
        return labels


class RecordingWorker(QtCore.QObject):
    finished = QtCore.Signal(object, object)
    progress = QtCore.Signal(int, int)

    def __init__(self, paths, highpass, lowpass, frequencies=(50,)):
        super().__init__()
        self.paths, self.highpass, self.lowpass = paths, highpass, lowpass
        self.frequencies = tuple(frequencies)

    @QtCore.Slot()
    def run(self):
        rows, errors = [], []
        for index, path in enumerate(self.paths):
            if QtCore.QThread.currentThread().isInterruptionRequested():
                break
            try:
                data = DataManager.load_file(str(path))
                optional = optional_recording_values(data, self.highpass, self.lowpass)
                optional["optical_frequencies"] = self.frequencies
                try:
                    amplitudes = recording_amplitudes(data, self.frequencies, self.highpass, self.lowpass)
                except ValueError as exc:
                    if len(self.frequencies) == 1:
                        raise
                    amplitudes = [recording_amplitude(data, self.highpass, self.lowpass, self.frequencies[0]), np.nan]
                    optional["warnings"].append(f"Second optical frequency: {exc}")
                amplitude = amplitudes[0]
                if len(amplitudes) > 1:
                    optional["optical_secondary"] = amplitudes[1]
                timestamp = recording_timestamp(path, data.metadata)
                rows.append((path, amplitude, timestamp,
                             data.metadata.get("amplitude_unit", ""), index + 1,
                             optional))
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
            self.progress.emit(index + 1, len(self.paths))
        self.finished.emit(rows, errors)


class RecordingPlotWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._thread = None
        self._worker = None
        self._rows = []
        self._errors = []
        self._loaded_frequencies = (50,)
        self.control_panel = QtWidgets.QScrollArea()
        self.control_panel.setWidgetResizable(True)
        self.control_panel.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        control_content = QtWidgets.QWidget()
        self.control_panel.setWidget(control_content)
        controls = QtWidgets.QVBoxLayout(control_content)
        controls.addWidget(QtWidgets.QLabel("Recording directory"))
        self.directory = QtWidgets.QLineEdit()
        self.directory.setPlaceholderText("Select a recording folder")
        controls.addWidget(self.directory)
        self.browse = QtWidgets.QPushButton("Browse…")
        controls.addWidget(self.browse)
        self.recursive = QtWidgets.QCheckBox("Include nested subdirectories")
        controls.addWidget(self.recursive)
        frequency_form = QtWidgets.QFormLayout()
        self.optical_frequency = self._frequency(50)
        self.optical_frequency_2 = self._frequency(100)
        self.optical_frequency_2.setEnabled(False)
        self.evaluate_second_frequency = QtWidgets.QCheckBox("Evaluate second optical frequency")
        self.evaluate_second_frequency.toggled.connect(self.optical_frequency_2.setEnabled)
        self.evaluate_second_frequency.toggled.connect(self._refresh_normalization)
        frequency_form.addRow("Optical frequency 1", self.optical_frequency)
        frequency_form.addRow(self.evaluate_second_frequency)
        frequency_form.addRow("Optical frequency 2", self.optical_frequency_2)
        controls.addLayout(frequency_form)
        form = QtWidgets.QFormLayout()
        self.highpass = self._cutoff()
        self.lowpass = self._cutoff()
        form.addRow("High-pass (Hz)", self.highpass)
        form.addRow("Low-pass (Hz)", self.lowpass)
        controls.addLayout(form)
        note = QtWidgets.QLabel(
            "0 disables a filter. Filters apply to each waveform before the "
            "selected sine amplitudes are extracted (4th-order, zero-phase Butterworth). "
            "Click Plot / Refresh after changing evaluation frequencies. Reference current remains evaluated at 50 Hz.")
        note.setWordWrap(True)
        controls.addWidget(note)
        self.load_button = QtWidgets.QPushButton("Plot recordings / Refresh")
        controls.addWidget(self.load_button)
        self.normalize = QtWidgets.QCheckBox("Normalize displayed optical values")
        self.normalize.setToolTip(
            "Divide the displayed optical series by its maximum. When optical/reference "
            "division is enabled, normalize the resulting ratios. Recalculate when that option changes.")
        controls.addWidget(self.normalize)
        self.normalize.toggled.connect(self._refresh_normalization)
        self.divide_by_reference = QtWidgets.QCheckBox("Divide optical current by reference")
        self.divide_by_reference.setToolTip("Divide each optical peak current by the corresponding reference peak current, before optional normalization. Missing or zero references produce gaps.")
        controls.addWidget(self.divide_by_reference)
        self.divide_by_reference.toggled.connect(self._refresh_normalization)
        self.show_optical = QtWidgets.QCheckBox("Optical amplitudes")
        self.show_optical.setChecked(True)
        self.show_ambient = QtWidgets.QCheckBox("Ambient temperature")
        self.show_tec = QtWidgets.QCheckBox("TEC temperature")
        self.show_reference = QtWidgets.QCheckBox("Reference current (50 Hz peak)")
        for toggle in (self.show_optical, self.show_ambient, self.show_tec, self.show_reference):
            controls.addWidget(toggle)
            toggle.toggled.connect(self._refresh_normalization)
        current_form = QtWidgets.QFormLayout()
        self.reference_amps_per_volt = QtWidgets.QDoubleSpinBox()
        self.reference_amps_per_volt.setDecimals(6)
        self.reference_amps_per_volt.setRange(0.000001, 1e9)
        self.reference_amps_per_volt.setValue(1)
        self.reference_amps_per_volt.setToolTip("Current probe conversion: peak current = reference peak voltage × A/V. Set your probe's calibration.")
        self.reference_amps_per_volt.valueChanged.connect(self._refresh_normalization)
        current_form.addRow("Reference scale (A/V)", self.reference_amps_per_volt)
        self.optical_amps_per_volt = QtWidgets.QDoubleSpinBox()
        self.optical_amps_per_volt.setDecimals(6)
        self.optical_amps_per_volt.setRange(0.000001, 1e9)
        self.optical_amps_per_volt.setValue(1)
        self.optical_amps_per_volt.setToolTip("Optical current conversion used for the optical/reference ratio. Default 1 A/V; set your optical channel's calibration.")
        self.optical_amps_per_volt.valueChanged.connect(self._refresh_normalization)
        current_form.addRow("Optical ratio scale (A/V)", self.optical_amps_per_volt)
        controls.addLayout(current_form)
        self.status = QtWidgets.QLabel("Select a folder containing CSV, H5 or HDF5 recordings.")
        self.status.setWordWrap(True)
        controls.addWidget(self.status)
        controls.addStretch()
        layout = QtWidgets.QVBoxLayout(self)
        self.time_axis = RecordingTimeAxis()
        self.plot = pg.PlotWidget(title="Optical peak amplitudes per recording",
                                  axisItems={"bottom": self.time_axis})
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.legend = self.plot.addLegend()
        self.temperature_view = pg.ViewBox()
        self.current_view = pg.ViewBox()
        self.plot.scene().addItem(self.temperature_view)
        self.plot.scene().addItem(self.current_view)
        self.plot.showAxis("right")
        self.temperature_axis = self.plot.getAxis("right")
        self.temperature_axis.linkToView(self.temperature_view)
        self.current_axis = pg.AxisItem("right")
        self.plot.plotItem.layout.addItem(self.current_axis, 2, 3)
        self.current_axis.linkToView(self.current_view)
        self.temperature_view.setXLink(self.plot.getViewBox())
        self.current_view.setXLink(self.plot.getViewBox())
        self.plot.getViewBox().sigResized.connect(self._sync_overlay_views)
        self._curves = {}
        self._sync_overlay_views()
        self.temperature_axis.hide()
        self.current_axis.hide()
        layout.addWidget(self.plot, 3)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["File", "Capture time", "50 Hz peak amplitude", "Ambient (°C)", "TEC (°C)", "Reference peak (A)"])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        self.errors = QtWidgets.QPlainTextEdit()
        self.errors.setReadOnly(True)
        self.errors.setMaximumHeight(100)
        self.errors.hide()
        layout.addWidget(self.errors)
        self.browse.clicked.connect(self._browse)
        self.load_button.clicked.connect(self.load_recordings)

    @staticmethod
    def _frequency(value):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.001, 1e12)
        spin.setDecimals(3)
        spin.setSuffix(" Hz")
        spin.setValue(value)
        return spin

    @staticmethod
    def _cutoff():
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0, 1e12)
        spin.setDecimals(3)
        spin.setSpecialValueText("0 (disabled)")
        return spin

    def _sync_overlay_views(self):
        main = self.plot.getViewBox()
        for view in (self.temperature_view, self.current_view):
            view.setGeometry(main.sceneBoundingRect())
            view.linkedViewChanged(main, view.XAxis)

    def _clear_curves(self):
        self.plot.clear()
        self.temperature_view.clear()
        self.current_view.clear()
        self.legend.clear()
        self._curves = {}
        self.temperature_axis.hide()
        self.current_axis.hide()

    def _add_overlay(self, key, x, y, view, color, label):
        curve = pg.PlotDataItem(x, y, pen=pg.mkPen(color, width=1.5),
                               symbol="o", symbolSize=5, symbolBrush=color, symbolPen=color, connect="finite")
        view.addItem(curve)
        self.legend.addItem(curve, label)
        self._curves[key] = curve

    def _browse(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Recording directory", self.directory.text())
        if path:
            self.directory.setText(path)

    def load_recordings(self):
        if self._thread is not None:
            return
        folder = Path(self.directory.text().strip())
        if not self.directory.text().strip() or not folder.is_dir():
            self.status.setText("Please select an existing recording directory.")
            return
        hp, lp = self.highpass.value(), self.lowpass.value()
        frequencies = (self.optical_frequency.value(),)
        if self.evaluate_second_frequency.isChecked():
            frequencies += (self.optical_frequency_2.value(),)
            if frequencies[0] == frequencies[1]:
                self.status.setText("Choose two different optical frequencies.")
                return
        if hp and lp and hp >= lp:
            self.status.setText("High-pass cutoff must be below low-pass cutoff.")
            return
        try:
            paths = sorted((p for p in (folder.rglob("*") if self.recursive.isChecked() else folder.iterdir())
                            if p.is_file() and p.suffix.lower() in {".csv", ".h5", ".hdf5"}), key=natural_key)
        except OSError as exc:
            self.status.setText(str(exc))
            return
        self._clear_curves()
        self.table.setRowCount(0)
        self._rows = []
        self._errors = []
        self.errors.hide()
        if not paths:
            self.status.setText("No CSV, H5 or HDF5 files found.")
            return
        self._thread = QtCore.QThread(self)
        self._worker = RecordingWorker(paths, hp, lp, frequencies)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress)
        self._worker.finished.connect(self._display)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._cleanup)
        self._thread.finished.connect(self._thread.deleteLater)
        self.load_button.setEnabled(False)
        self.status.setText(f"Processing {len(paths)} files…")
        self._thread.start()

    @QtCore.Slot(int, int)
    def _progress(self, done, total):
        self.status.setText(f"Processing {done} / {total} files…")

    @QtCore.Slot(object, object)
    def _display(self, rows, errors):
        self._rows = list(rows)
        self._errors = list(errors)
        if self._rows and len(self._rows[0]) > 5:
            self._loaded_frequencies = self._rows[0][5].get("optical_frequencies", (50,))
        self._refresh_normalization()

    def _refresh_normalization(self):
        rows = list(self._rows)
        errors = list(self._errors)
        self._clear_curves()
        timed = bool(rows) and all(row[2] is not None for row in rows)
        if timed:
            rows.sort(key=lambda row: row[2])
        x = np.array([row[2] if timed else row[4] for row in rows])
        y = np.array([row[1] for row in rows])
        aux = [row[5] if len(row) > 5 else {} for row in rows]
        reference = np.array([v.get("reference", np.nan) for v in aux]) * self.reference_amps_per_volt.value()
        second_enabled = self.evaluate_second_frequency.isChecked() and len(self._loaded_frequencies) > 1
        secondary = np.array([v.get("optical_secondary", np.nan) for v in aux])
        ratio = self.divide_by_reference.isChecked()
        invalid_ratios = 0
        if ratio:
            optical_scale = np.array([1 if row[3] == "A" else self.optical_amps_per_volt.value() for row in rows])
            valid = np.isfinite(reference) & (reference != 0) & np.isfinite(y)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                y = np.divide(y * optical_scale, reference, out=np.full(y.shape, np.nan), where=valid)
            y[~np.isfinite(y)] = np.nan
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                secondary = np.divide(secondary * optical_scale, reference, out=np.full(secondary.shape, np.nan),
                                      where=np.isfinite(reference) & (reference != 0) & np.isfinite(secondary))
            secondary[~np.isfinite(secondary)] = np.nan
            invalid_ratios = int(np.count_nonzero(~np.isfinite(y)))
        normalized = self.normalize.isChecked()
        zero_maximum = False
        if normalized and np.any(np.isfinite(y)):
            maximum = float(np.nanmax(y))
            if maximum > 0:
                y = y / maximum
            else:
                zero_maximum = True
        if normalized and second_enabled and np.any(np.isfinite(secondary)):
            maximum = float(np.nanmax(secondary))
            if maximum > 0:
                secondary = secondary / maximum
        units = {row[3] for row in rows}
        unit = next(iter(units)) if len(units) == 1 else "mixed units"
        self.time_axis.set_capture_time(timed)
        first_frequency = self._loaded_frequencies[0]
        optical_label = "Optical / reference current" if ratio else "Optical peak amplitude"
        if normalized:
            optical_label = "Normalized " + optical_label
        self.plot.setLabel("left", optical_label, units="" if normalized or ratio else unit)
        self.plot.getAxis("left").setVisible(self.show_optical.isChecked())
        if self.show_optical.isChecked():
            self._curves["optical"] = self.plot.plot(x, y, pen=pg.mkPen("b", width=1.5), symbol="o", symbolSize=6,
                                                     name=f"Optical {first_frequency:g} Hz" + (" / reference" if ratio else ""), connect="finite")
            if second_enabled:
                second_frequency = self._loaded_frequencies[1]
                self._curves["optical_secondary"] = self.plot.plot(
                    x, secondary, pen=pg.mkPen("#dc2626", width=1.5), symbol="t", symbolSize=6,
                    symbolBrush="#dc2626", symbolPen="#dc2626", connect="finite",
                    name=f"Optical {second_frequency:g} Hz" + (" / reference" if ratio else ""))
        ambient = np.array([v.get("ambient", np.nan) for v in aux])
        tec = np.array([v.get("tec", np.nan) for v in aux])
        if normalized and np.any(np.isfinite(reference)):
            maximum = np.nanmax(reference)
            if maximum > 0:
                reference = reference / maximum
        missing = []
        for key, values, toggle, color, label in (
                ("ambient", ambient, self.show_ambient, "#d97706", "Ambient (°C)"),
                ("tec", tec, self.show_tec, "#15803d", "TEC (°C)"),
                ("reference", reference, self.show_reference, "#9333ea", "Reference current")):
            if toggle.isChecked():
                if np.any(np.isfinite(values)):
                    overlay_x = np.array([v.get(f"{key}_timestamp", row[2]) for row, v in zip(rows, aux)]) if timed else x
                    self._add_overlay(key, overlay_x, values,
                                      self.current_view if key == "reference" else self.temperature_view, color, label)
                else:
                    missing.append(key)
        self.temperature_axis.setLabel("Temperature", units="°C")
        self.current_axis.setLabel("Normalized reference peak" if normalized else "Reference 50 Hz peak current",
                                   units="" if normalized else "A")
        self.temperature_axis.setVisible("ambient" in self._curves or "tec" in self._curves)
        self.current_axis.setVisible("reference" in self._curves)
        self.plot.autoRange()
        visible_x = [curve.xData[np.isfinite(curve.xData)] for curve in self._curves.values()
                     if curve.xData is not None]
        if visible_x and any(values.size for values in visible_x):
            all_x = np.concatenate(visible_x)
            self.plot.setXRange(float(np.min(all_x)), float(np.max(all_x)), padding=0.02)
        self.temperature_view.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.current_view.enableAutoRange(axis=pg.ViewBox.YAxis)
        self._sync_overlay_views()
        self.table.setRowCount(len(rows))
        headers = ["File", "Capture time", f"{optical_label} ({first_frequency:g} Hz)",
                   "Ambient (°C)", "TEC (°C)", "Normalized reference" if normalized else "Reference peak (A)"]
        if second_enabled:
            headers.append(f"{optical_label} ({self._loaded_frequencies[1]:g} Hz)")
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        for i, row in enumerate(rows):
            path, amplitude, timestamp, unit, _ = row[:5]
            values = [path.name, datetime.fromtimestamp(timestamp).isoformat(timespec="milliseconds")
                      if timestamp is not None else "—", (f"{y[i]:.9g}" if np.isfinite(y[i]) else "—")
                      if normalized or ratio else f"{amplitude:.9g} {unit}".strip()]
            values.extend(f"{series[i]:.9g}" if np.isfinite(series[i]) else "—" for series in (ambient, tec, reference))
            if second_enabled:
                values.append((f"{secondary[i]:.9g}" + (f" {unit}" if not normalized and not ratio else ""))
                              if np.isfinite(secondary[i]) else "—")
            for j, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(str(path))
                self.table.setItem(i, j, item)
        warnings = [f"{row[0].name}: {warning}" for row, v in zip(rows, aux) for warning in v.get("warnings", [])]
        self.errors.setPlainText("\n".join(errors + warnings))
        self.errors.setVisible(bool(errors or warnings))
        self.status.setText(f"Plotted {len(rows)} recordings; skipped {len(errors)} files." +
                            (" Using filename order (capture timestamps unavailable)." if rows and not timed else "") +
                            (" Maximum amplitude is zero; values remain zero." if zero_maximum else "") +
                            (f" No recorded data for: {', '.join(missing)}." if missing else "") +
                            (f" Optical/reference ratio unavailable for {invalid_ratios} recording(s) (missing or zero reference)."
                             if invalid_ratios else ""))

    @QtCore.Slot()
    def _cleanup(self):
        self._thread = self._worker = None
        self.load_button.setEnabled(True)

    def shutdown(self):
        if self._thread is not None:
            self._thread.requestInterruption()
            return False
        return True
