from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from data_manager_signal_loader import DataManager
from recording_analysis import recording_amplitude, recording_amplitudes, recording_harmonic_analysis
from tec_sine_trace import is_trace_path, load_trace, trace_plot_series, DMM_SAMPLE, TEC_SAMPLE


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
    values["trace_session_id"] = data.metadata.get("tec_trace_session_id")
    if data.metadata.get("tec_dc_transition_during_capture") == "true":
        values["warnings"].append("DC offset changed during this capture; plotted offset is the start value")
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
    for key, metadata_key in (("dc_offset", "tec_dc_offset_deg_c"), ("tec_setpoint", "tec_sweep_setpoint_deg_c"),
                              ("dmm", "dmm_voltage")):
        try:
            if key == "dmm" and (data.metadata.get("dmm_voltage_status") != "ok"
                                  or data.metadata.get("dmm_voltage_unit", "V") != "V"):
                continue
            value = float(data.metadata[metadata_key])
            if np.isfinite(value):
                values[key] = value
        except (KeyError, ValueError, TypeError):
            pass
    try:
        values["dmm_timestamp"] = datetime.fromisoformat(data.metadata["dmm_voltage_read_at"]).timestamp()
    except (KeyError, ValueError, TypeError, OverflowError):
        pass
    return values


class RecordingTimeAxis(pg.AxisItem):
    """Display the local capture date and time stored in recording filenames."""

    def __init__(self):
        self.capture_time = False
        self.time_format = "HH:MM:SS"
        self.value_label = None
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
        label = (self.value_label if self.value_label else f"Capture time ({self.time_format})"
                 if self.capture_time else "File index")
        if self.labelText != label:
            self.setLabel(label, units="")

    def setRange(self, minimum, maximum):
        span = abs(maximum - minimum)
        self.time_format = "HH:MM:SS" if span < 3600 else "HH:MM" if span < 86400 else "DD Mon HH:MM"
        super().setRange(minimum, maximum)
        if self._ready:
            self._update_label()

    def tickValues(self, minVal, maxVal, size):
        if self.logMode:
            # Log tick levels may use spacing=None; these are already in the
            # coordinates expected by AxisItem and cannot be multiplied.
            return super().tickValues(minVal, maxVal, size)
        # Generate ticks in the displayed resolution to avoid repeated labels.
        unit = {"HH:MM:SS": 1, "HH:MM": 60, "DD Mon HH:MM": 3600}[self.time_format] if self.capture_time else 1
        levels = super().tickValues(minVal / unit, maxVal / unit, size)
        if self.capture_time and unit > 1:
            levels = [(spacing, values) for spacing, values in levels if spacing >= 1]
        return [(spacing * unit, [value * unit for value in values]) for spacing, values in levels]

    def tickStrings(self, values, scale, spacing):
        if not self.capture_time or self.logMode:
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

    def __init__(self, paths, highpass, lowpass, frequencies=(50,), thd_harmonics=None):
        super().__init__()
        self.paths, self.highpass, self.lowpass = paths, highpass, lowpass
        self.frequencies = tuple(frequencies)
        self.thd_harmonics = thd_harmonics

    @QtCore.Slot()
    def run(self):
        rows, errors = [], []
        for index, path in enumerate(self.paths):
            if QtCore.QThread.currentThread().isInterruptionRequested():
                break
            try:
                if is_trace_path(path):
                    trace = load_trace(path)
                    rows.append((path, np.nan, None, "", index + 1, {"trace": trace}))
                    errors.extend(f"{path.name}: {warning}" for warning in trace.warnings)
                    self.progress.emit(index + 1, len(self.paths))
                    continue
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
                if self.thd_harmonics is not None:
                    optional["thd_fundamental_hz"] = self.frequencies[0]
                    optional["thd_harmonics"] = self.thd_harmonics
                    try:
                        harmonics = recording_harmonic_analysis(data, self.frequencies[0], self.thd_harmonics,
                                                                 self.frequencies)
                        optional["thd"] = harmonics["thd_percent"]
                        optional["thd_status"] = "ok"
                        if not self.highpass and not self.lowpass:
                            amplitude = harmonics["amplitudes"][self.frequencies[0]]
                            if len(self.frequencies) > 1:
                                optional["optical_secondary"] = harmonics["amplitudes"][self.frequencies[1]]
                    except ValueError as exc:
                        optional["thd"] = np.nan
                        optional["thd_status"] = "unavailable"
                        optional["warnings"].append(f"THD: {exc}")
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
        self._traces = []
        self._preserve_trace_refresh = False
        self._errors = []
        self._loaded_frequencies = (50,)
        self._pending_thd_analysis = False
        self._shutting_down = False
        self._analysis_label = "Processing"
        self._thd_refresh_timer = QtCore.QTimer(self)
        self._thd_refresh_timer.setSingleShot(True)
        self._thd_refresh_timer.setInterval(250)
        self._thd_refresh_timer.timeout.connect(self._ensure_thd_analysis)
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
        self.show_thd = QtWidgets.QCheckBox("Evaluate / plot optical THD (%)")
        self.show_thd.toggled.connect(self._on_thd_toggled)
        controls.addWidget(self.show_thd)
        thd_form = QtWidgets.QFormLayout()
        self.thd_harmonics = QtWidgets.QSpinBox()
        self.thd_harmonics.setRange(2, 50)
        self.thd_harmonics.setValue(10)
        self.thd_harmonics.setEnabled(False)
        self.show_thd.toggled.connect(self.thd_harmonics.setEnabled)
        self.thd_harmonics.valueChanged.connect(self._schedule_thd_analysis)
        self.optical_frequency.valueChanged.connect(self._schedule_thd_analysis)
        thd_form.addRow("Highest THD harmonic", self.thd_harmonics)
        controls.addLayout(thd_form)
        self.x_axis = QtWidgets.QComboBox()
        self.x_axis.addItems(["Capture time / file index", "DC offset temperature"])
        self.x_axis.currentIndexChanged.connect(self._refresh_normalization)
        controls.addWidget(self.x_axis)
        form = QtWidgets.QFormLayout()
        self.highpass = self._cutoff()
        self.lowpass = self._cutoff()
        form.addRow("High-pass (Hz)", self.highpass)
        form.addRow("Low-pass (Hz)", self.lowpass)
        controls.addLayout(form)
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
        self.show_dmm = QtWidgets.QCheckBox("DMM voltage")
        self.show_offset = QtWidgets.QCheckBox("TEC DC offset")
        self.show_setpoint = QtWidgets.QCheckBox("TEC commanded setpoint")
        for toggle in (self.show_optical, self.show_ambient, self.show_tec, self.show_reference,
                       self.show_dmm, self.show_offset, self.show_setpoint):
            controls.addWidget(toggle)
            toggle.toggled.connect(self._refresh_visibility)
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
        self.legend = self.plot.addLegend(labelTextColor="#333333")
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
        self.voltage_view = pg.ViewBox()
        self.thd_view = pg.ViewBox()
        self.voltage_axis = pg.AxisItem("right")
        self.thd_axis = pg.AxisItem("right")
        for column, view, axis in ((4, self.voltage_view, self.voltage_axis), (5, self.thd_view, self.thd_axis)):
            self.plot.scene().addItem(view)
            self.plot.plotItem.layout.addItem(axis, 2, column)
            axis.linkToView(view)
            view.setXLink(self.plot.getViewBox())
        self.temperature_view.setXLink(self.plot.getViewBox())
        self.current_view.setXLink(self.plot.getViewBox())
        self.plot.getViewBox().sigResized.connect(self._sync_overlay_views)
        self._curves = {}
        self._sync_overlay_views()
        self.temperature_axis.hide()
        self.current_axis.hide()
        self.voltage_axis.hide()
        self.thd_axis.hide()
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
        for view in (self.temperature_view, self.current_view, self.voltage_view, self.thd_view):
            view.setGeometry(main.sceneBoundingRect())
            view.linkedViewChanged(main, view.XAxis)

    def _clear_curves(self):
        self.plot.clear()
        self.temperature_view.clear()
        self.current_view.clear()
        self.voltage_view.clear()
        self.thd_view.clear()
        self.legend.clear()
        self._curves = {}
        self.temperature_axis.hide()
        self.current_axis.hide()
        self.voltage_axis.hide()
        self.thd_axis.hide()

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
        self._shutting_down = False
        folder = Path(self.directory.text().strip())
        if not self.directory.text().strip() or not folder.is_dir():
            self.status.setText("Please select an existing recording directory.")
            return
        options = self._analysis_options()
        if options is None:
            return
        hp, lp, frequencies = options
        try:
            paths = sorted((p for p in (folder.rglob("*") if self.recursive.isChecked() else folder.iterdir())
                            if p.is_file() and p.suffix.lower() in {".csv", ".h5", ".hdf5"}), key=natural_key)
        except OSError as exc:
            self.status.setText(str(exc))
            return
        self._thd_refresh_timer.stop()
        self._pending_thd_analysis = False
        self._clear_curves()
        self.table.setRowCount(0)
        self._rows = []
        self._traces = []
        self._errors = []
        self.errors.hide()
        if not paths:
            self.status.setText("No CSV, H5 or HDF5 files found.")
            return
        self._start_analysis(paths, hp, lp, frequencies)

    def _analysis_options(self):
        hp, lp = self.highpass.value(), self.lowpass.value()
        frequencies = (self.optical_frequency.value(),)
        if self.evaluate_second_frequency.isChecked():
            frequencies += (self.optical_frequency_2.value(),)
            if frequencies[0] == frequencies[1]:
                self.status.setText("Choose two different optical frequencies.")
                return None
        if hp and lp and hp >= lp:
            self.status.setText("High-pass cutoff must be below low-pass cutoff.")
            return None
        return hp, lp, frequencies

    def _on_thd_toggled(self, enabled):
        self._refresh_normalization()
        if enabled:
            self._ensure_thd_analysis()
        else:
            self._thd_refresh_timer.stop()
            self._pending_thd_analysis = False

    def _schedule_thd_analysis(self):
        if self.show_thd.isChecked():
            self._thd_refresh_timer.start()

    def _ensure_thd_analysis(self):
        if self._shutting_down or not self.show_thd.isChecked():
            return
        if self._thread is not None:
            self._pending_thd_analysis = True
            return
        if not self._rows:
            self.status.setText("Load optical waveform recordings to calculate THD.")
            return
        fundamental = self.optical_frequency.value()
        highest = self.thd_harmonics.value()
        if all(len(row) > 5 and "thd" in row[5]
               and row[5].get("thd_fundamental_hz") == fundamental
               and row[5].get("thd_harmonics") == highest for row in self._rows):
            return
        options = self._analysis_options()
        if options is None:
            return
        # Reopen the files that were actually plotted, even if the directory
        # edit has since changed. Raw waveforms stay out of the GUI's memory.
        self._start_analysis([row[0] for row in self._rows], *options, thd_only_refresh=True)

    def _start_analysis(self, paths, hp, lp, frequencies, *, thd_only_refresh=False):
        self._preserve_trace_refresh = thd_only_refresh
        self._analysis_label = "Calculating optical THD" if thd_only_refresh else "Processing"
        self._thread = QtCore.QThread(self)
        self._worker = RecordingWorker(paths, hp, lp, frequencies,
                                        self.thd_harmonics.value() if self.show_thd.isChecked() else None)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._progress)
        self._worker.finished.connect(self._display)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._cleanup)
        self._thread.finished.connect(self._thread.deleteLater)
        self.load_button.setEnabled(False)
        self.status.setText(f"{self._analysis_label}: {len(paths)} files…")
        self._thread.start()

    @QtCore.Slot(int, int)
    def _progress(self, done, total):
        self.status.setText(f"{self._analysis_label}: {done} / {total} files…")

    @QtCore.Slot(object, object)
    def _display(self, rows, errors):
        traces = [row[5]["trace"] for row in rows if len(row) > 5 and "trace" in row[5]]
        if not self._preserve_trace_refresh:
            self._traces = traces
        self._preserve_trace_refresh = False
        self._rows = [row for row in rows if not (len(row) > 5 and "trace" in row[5])]
        self._errors = list(errors)
        if self._rows and len(self._rows[0]) > 5:
            self._loaded_frequencies = self._rows[0][5].get("optical_frequencies", (50,))
        self._refresh_normalization()

    def _refresh_visibility(self):
        # Visibility never changes scalar table values. In particular, do not
        # recreate thousands of cells just to show the DMM voltage curve.
        self._refresh_normalization(update_table=False)

    def _refresh_normalization(self, *, update_table=True):
        rows = list(self._rows)
        errors = list(self._errors)
        self._clear_curves()
        trace_series = trace_plot_series(self._traces)
        has_trace = bool(trace_series)
        timed = has_trace or (bool(rows) and all(row[2] is not None for row in rows))
        if timed:
            rows.sort(key=lambda row: row[2] if row[2] is not None else float("inf"))
        x = np.array([(row[2] if row[2] is not None else np.nan) if timed else row[4] for row in rows])
        y = np.array([row[1] for row in rows])
        aux = [row[5] if len(row) > 5 else {} for row in rows]
        offset_mode = self.x_axis.currentIndex() == 1
        offsets = np.array([v.get("dc_offset", np.nan) for v in aux])
        if offset_mode:
            x = offsets
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
        self.time_axis.value_label = "DC offset temperature (°C)" if offset_mode else None
        self.time_axis.set_capture_time(timed and not offset_mode)
        first_frequency = self._loaded_frequencies[0]
        optical_label = "Optical / reference current" if ratio else "Optical peak amplitude"
        if normalized:
            optical_label = "Normalized " + optical_label
        self.plot.setLabel("left", optical_label, units="" if normalized or ratio else unit)
        self.plot.getAxis("left").setVisible(self.show_optical.isChecked() and bool(rows))
        if self.show_optical.isChecked() and rows:
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
        setpoints = np.array([v.get("tec_setpoint", np.nan) for v in aux])
        dmm = np.array([v.get("dmm", np.nan) for v in aux])
        thd = np.array([v.get("thd", np.nan) for v in aux])
        if normalized and np.any(np.isfinite(reference)):
            maximum = np.nanmax(reference)
            if maximum > 0:
                reference = reference / maximum
        missing = []
        for key, values, toggle, color, label in (
                ("ambient", ambient, self.show_ambient, "#d97706", "Ambient (°C)"),
                ("tec", tec, self.show_tec, "#15803d", "TEC (°C)"),
                ("reference", reference, self.show_reference, "#9333ea", "Reference current"),
                ("dc_offset", offsets, self.show_offset, "#b45309", "DC offset (°C)"),
                ("tec_setpoint", setpoints, self.show_setpoint, "#0e7490", "TEC setpoint (°C)"),
                ("dmm", dmm, self.show_dmm, "#08916b", "DMM voltage (V)"),
                ("thd", thd, self.show_thd, "#7c3aed", "Optical THD (%)")):
            if toggle.isChecked():
                overlay_x, overlay_y = self._overlay_samples(key, values, x, rows, aux, trace_series, timed, offset_mode)
                if np.any(np.isfinite(overlay_y)):
                    view = {"reference": self.current_view, "dmm": self.voltage_view, "thd": self.thd_view}.get(key, self.temperature_view)
                    self._add_overlay(key, overlay_x, overlay_y,
                                      view, color, label)
                else:
                    if key != "thd":
                        missing.append(key)
        self.temperature_axis.setLabel("Temperature", units="°C")
        self.current_axis.setLabel("Normalized reference peak" if normalized else "Reference 50 Hz peak current",
                                   units="" if normalized else "A")
        self.temperature_axis.setVisible(any(key in self._curves for key in ("ambient", "tec", "dc_offset", "tec_setpoint")))
        self.current_axis.setVisible("reference" in self._curves)
        self.voltage_axis.setLabel("DMM voltage", units="V")
        self.thd_axis.setLabel("Optical THD", units="%")
        self.voltage_axis.setVisible("dmm" in self._curves)
        self.thd_axis.setVisible("thd" in self._curves)
        self.plot.setTitle("Recorded optical amplitudes, DMM voltage and distortion"
                           if self.show_thd.isChecked() or self.show_dmm.isChecked()
                           else "Optical peak amplitudes per recording", color="#333333")
        self.plot.autoRange()
        visible_x = [curve.xData[np.isfinite(curve.xData)] for curve in self._curves.values()
                     if curve.xData is not None]
        if visible_x and any(values.size for values in visible_x):
            all_x = np.concatenate(visible_x)
            self.plot.setXRange(float(np.min(all_x)), float(np.max(all_x)), padding=0.02)
        self.temperature_view.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.current_view.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.voltage_view.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.thd_view.enableAutoRange(axis=pg.ViewBox.YAxis)
        self._sync_overlay_views()
        headers = ["File", "Capture time", f"{optical_label} ({first_frequency:g} Hz)",
                   "Ambient (°C)", "TEC (°C)", "Normalized reference" if normalized else "Reference peak (A)"]
        if second_enabled:
            headers.append(f"{optical_label} ({self._loaded_frequencies[1]:g} Hz)")
        extra_columns = []
        for label, series, enabled in (("DMM voltage (V)", dmm, self.show_dmm.isChecked()),
                                       ("DC offset (°C)", offsets, False), ("TEC setpoint (°C)", setpoints, False),
                                       ("Optical THD (%)", thd, self.show_thd.isChecked())):
            if enabled or np.any(np.isfinite(series)):
                headers.append(label)
                extra_columns.append(series)
        table_values = []
        if update_table:
            for i, row in enumerate(rows):
                path, amplitude, timestamp, unit, _ = row[:5]
                values = [path.name, datetime.fromtimestamp(timestamp).isoformat(timespec="milliseconds")
                          if timestamp is not None else "—", (f"{y[i]:.9g}" if np.isfinite(y[i]) else "—")
                          if normalized or ratio else f"{amplitude:.9g} {unit}".strip()]
                values.extend(f"{series[i]:.9g}" if np.isfinite(series[i]) else "—" for series in (ambient, tec, reference))
                if second_enabled:
                    values.append((f"{secondary[i]:.9g}" + (f" {unit}" if not normalized and not ratio else ""))
                                  if np.isfinite(secondary[i]) else "—")
                values.extend(f"{series[i]:.9g}" if np.isfinite(series[i]) else "—" for series in extra_columns)
                table_values.append((str(path), values))
            self._populate_table(headers, table_values)
        warnings = [f"{row[0].name}: {warning}" for row, v in zip(rows, aux) for warning in v.get("warnings", [])]
        self.errors.setPlainText("\n".join(errors + warnings))
        self.errors.setVisible(bool(errors or warnings))
        self.status.setText(f"Plotted {len(rows)} recordings; skipped {len(errors)} files." +
                            (f" Full traces: {sum(np.count_nonzero(t.samples['kind'] == DMM_SAMPLE) for t in self._traces)} DMM values, "
                             f"{sum(np.count_nonzero(t.samples['kind'] == TEC_SAMPLE) for t in self._traces)} TEC updates."
                             if self._traces else "") +
                            (" Using filename order (capture timestamps unavailable)." if rows and not timed else "") +
                            (" Maximum amplitude is zero; values remain zero." if zero_maximum else "") +
                            (f" No recorded data for: {', '.join(missing)}." if missing else "") +
                            (f" DC offset missing for {int(np.count_nonzero(~np.isfinite(offsets)))} recording(s)."
                             if offset_mode and np.any(~np.isfinite(offsets)) else "") +
                            (f" Optical/reference ratio unavailable for {invalid_ratios} recording(s) (missing or zero reference)."
                             if invalid_ratios else ""))
        if self.show_thd.isChecked():
            if not rows:
                thd_message = " Load optical waveform recordings to calculate THD."
            elif not any("thd" in values for values in aux):
                thd_message = " THD has not been calculated from the optical waveforms yet."
            else:
                unavailable = int(np.count_nonzero(~np.isfinite(thd)))
                thd_message = (f" THD could not be calculated for {unavailable} waveform(s); see details below."
                               if unavailable else "")
            self.status.setText(self.status.text() + thd_message)

    @staticmethod
    def _overlay_samples(key, values, x, rows, aux, traces, timed, offset_mode):
        wave_time = np.array([v.get(f"{key}_timestamp", row[2]) if row[2] is not None else np.nan
                              for row, v in zip(rows, aux)], dtype=float)
        wave_x = wave_time if timed and not offset_mode else x
        trace = traces.get(key)
        if trace is None:
            return wave_x, values
        # Each scope file keeps a paired snapshot for older readers. Prefer the
        # complete stream when that same session is present, without duplicates.
        keep = np.array([v.get("trace_session_id") not in trace["sessions"] for v in aux], dtype=bool)
        full_x = np.concatenate([wave_x[keep], trace["offsets"] if offset_mode else trace["timestamp"]])
        full_y = np.concatenate([values[keep], trace["values"]])
        stamps = np.concatenate([wave_time[keep], trace["timestamp"]])
        order = np.argsort(stamps, kind="stable")
        return full_x[order], full_y[order]

    def _populate_table(self, headers, rows):
        table = self.table
        header = table.horizontalHeader()
        updates_enabled = table.updatesEnabled()
        sorting_enabled = table.isSortingEnabled()
        table.setUpdatesEnabled(False)
        table.setSortingEnabled(False)
        # ResizeToContents on a populated QTableWidget can scan the entire
        # column for every setItem. Populate with fixed widths, then size once.
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        try:
            table.setRowCount(len(rows))
            table.setColumnCount(len(headers))
            table.setHorizontalHeaderLabels(headers)
            for i, (path, values) in enumerate(rows):
                for j, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    item.setToolTip(path)
                    table.setItem(i, j, item)
            header.setMinimumSectionSize(90)
            header.setResizeContentsPrecision(1000)
            table.resizeColumnsToContents()
        finally:
            table.setSortingEnabled(sorting_enabled)
            table.setUpdatesEnabled(updates_enabled)

    @QtCore.Slot()
    def _cleanup(self):
        self._thread = self._worker = None
        self.load_button.setEnabled(True)
        if self._pending_thd_analysis:
            self._pending_thd_analysis = False
            QtCore.QTimer.singleShot(0, self._ensure_thd_analysis)

    def shutdown(self):
        self._shutting_down = True
        self._thd_refresh_timer.stop()
        self._pending_thd_analysis = False
        if self._thread is not None:
            self._thread.requestInterruption()
            return False
        return True
