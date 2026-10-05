from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from data_manager_signal_loader import DataManager
from recording_analysis import recording_amplitude


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

    def __init__(self, paths, highpass, lowpass):
        super().__init__()
        self.paths, self.highpass, self.lowpass = paths, highpass, lowpass

    @QtCore.Slot()
    def run(self):
        rows, errors = [], []
        for index, path in enumerate(self.paths):
            if QtCore.QThread.currentThread().isInterruptionRequested():
                break
            try:
                data = DataManager.load_file(str(path))
                amplitude = recording_amplitude(data, self.highpass, self.lowpass)
                timestamp = recording_timestamp(path, data.metadata)
                rows.append((path, amplitude, timestamp,
                             data.metadata.get("amplitude_unit", ""), index + 1))
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
        self.control_panel = QtWidgets.QWidget()
        controls = QtWidgets.QVBoxLayout(self.control_panel)
        controls.addWidget(QtWidgets.QLabel("Recording directory"))
        self.directory = QtWidgets.QLineEdit()
        self.directory.setPlaceholderText("Select a recording folder")
        controls.addWidget(self.directory)
        self.browse = QtWidgets.QPushButton("Browse…")
        controls.addWidget(self.browse)
        self.recursive = QtWidgets.QCheckBox("Include nested subdirectories")
        controls.addWidget(self.recursive)
        form = QtWidgets.QFormLayout()
        self.highpass = self._cutoff()
        self.lowpass = self._cutoff()
        form.addRow("High-pass (Hz)", self.highpass)
        form.addRow("Low-pass (Hz)", self.lowpass)
        controls.addLayout(form)
        note = QtWidgets.QLabel(
            "0 disables a filter. Filters apply to each waveform before the "
            "50 Hz peak amplitude is extracted (4th-order, zero-phase Butterworth). "
            "Cutoffs near 50 Hz attenuate the measured amplitude.")
        note.setWordWrap(True)
        controls.addWidget(note)
        self.load_button = QtWidgets.QPushButton("Plot recordings / Refresh")
        controls.addWidget(self.load_button)
        self.normalize = QtWidgets.QCheckBox("Normalize amplitude (divide by maximum)")
        controls.addWidget(self.normalize)
        self.normalize.toggled.connect(self._refresh_normalization)
        self.status = QtWidgets.QLabel("Select a folder containing CSV, H5 or HDF5 recordings.")
        self.status.setWordWrap(True)
        controls.addWidget(self.status)
        controls.addStretch()
        layout = QtWidgets.QVBoxLayout(self)
        self.time_axis = RecordingTimeAxis()
        self.plot = pg.PlotWidget(title="50 Hz peak amplitude per recording",
                                  axisItems={"bottom": self.time_axis})
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot, 3)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["File", "Capture time", "50 Hz peak amplitude"])
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
    def _cutoff():
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0, 1e12)
        spin.setDecimals(3)
        spin.setSpecialValueText("0 (disabled)")
        return spin

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
        if hp and lp and hp >= lp:
            self.status.setText("High-pass cutoff must be below low-pass cutoff.")
            return
        try:
            paths = sorted((p for p in (folder.rglob("*") if self.recursive.isChecked() else folder.iterdir())
                            if p.is_file() and p.suffix.lower() in {".csv", ".h5", ".hdf5"}), key=natural_key)
        except OSError as exc:
            self.status.setText(str(exc))
            return
        self.plot.clear()
        self.table.setRowCount(0)
        self._rows = []
        self._errors = []
        self.errors.hide()
        if not paths:
            self.status.setText("No CSV, H5 or HDF5 files found.")
            return
        self._thread = QtCore.QThread(self)
        self._worker = RecordingWorker(paths, hp, lp)
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
        self._refresh_normalization()

    @QtCore.Slot()
    def _refresh_normalization(self):
        rows = list(self._rows)
        errors = list(self._errors)
        self.plot.clear()
        timed = bool(rows) and all(row[2] is not None for row in rows)
        if timed:
            rows.sort(key=lambda row: row[2])
        x = np.array([row[2] if timed else row[4] for row in rows])
        y = np.array([row[1] for row in rows])
        normalized = self.normalize.isChecked()
        zero_maximum = False
        if normalized and y.size:
            maximum = float(np.max(y))
            if maximum > 0:
                y = y / maximum
            else:
                zero_maximum = True
        units = {row[3] for row in rows}
        unit = next(iter(units)) if len(units) == 1 else "mixed units"
        self.time_axis.set_capture_time(timed)
        self.plot.setLabel("left", "Normalized 50 Hz peak amplitude" if normalized else "50 Hz peak amplitude",
                           units="" if normalized else unit)
        self.plot.plot(x, y, pen=pg.mkPen("b", width=1.5), symbol="o", symbolSize=6)
        self.plot.autoRange()
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels([
            "File", "Capture time", "Normalized amplitude" if normalized else "50 Hz peak amplitude"])
        for i, (path, amplitude, timestamp, unit, _) in enumerate(rows):
            values = [path.name, datetime.fromtimestamp(timestamp).isoformat(timespec="milliseconds")
                      if timestamp is not None else "—", f"{y[i]:.9g}" if normalized else f"{amplitude:.9g} {unit}".strip()]
            for j, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(str(path))
                self.table.setItem(i, j, item)
        self.errors.setPlainText("\n".join(errors))
        self.errors.setVisible(bool(errors))
        self.status.setText(f"Plotted {len(rows)} recordings; skipped {len(errors)} files." +
                            (" Using filename order (capture timestamps unavailable)." if rows and not timed else "") +
                            (" Maximum amplitude is zero; values remain zero." if zero_maximum else ""))

    @QtCore.Slot()
    def _cleanup(self):
        self._thread = self._worker = None
        self.load_button.setEnabled(True)

    def shutdown(self):
        if self._thread is not None:
            self._thread.requestInterruption()
            return False
        return True
