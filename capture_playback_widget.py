"""Raw oscilloscope file playback, with one-frame background prefetch."""
from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import time

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg

from data_manager_signal_loader import DataManager
from recording_plot_widget import natural_key, recording_timestamp


CAPTURE_EXTENSIONS = {".h5", ".hdf5", ".csv", ".txt", ".wfm", ".npy", ".npz"}


def capture_metadata_lines(path, data, index, total):
    """Keep the visible legend short; the tooltip retains the complete metadata."""
    metadata = data.metadata
    name = path.name if len(path.name) <= 40 else path.name[:17] + "…" + path.name[-20:]
    lines = [f"{index + 1}/{total}  {name}"]
    timestamp = recording_timestamp(path, metadata)
    if timestamp is not None:
        lines.append(datetime.fromtimestamp(timestamp).isoformat(sep=" ", timespec="milliseconds"))

    def value(key, label, unit=""):
        unit = {"degC": "°C", "degF": "°F"}.get(unit, unit)
        raw = metadata.get(key)
        if raw is None:
            return None
        try:
            number = float(raw)
            if not np.isfinite(number):
                return None
            return f"{label}: {number:.6g}" + (f" {unit}" if unit else "")
        except (ValueError, TypeError):
            return None

    groups = [
        (("tec_temperature", "TEC", metadata.get("tec_temperature_unit", "°C")),
         ("ambient_temperature", "Ambient", metadata.get("ambient_temperature_unit", "°C"))),
        (("tec_sweep_setpoint_deg_c", "Target", "°C"), ("tec_dc_offset_deg_c", "DC offset", "°C")),
        (("dmm_voltage", "DMM", "V"),),
        (("tec_sine_amplitude_deg_c", "AC", "°C"), ("tec_sine_frequency_hz", "Sine", "Hz")),
        (("tec_sweep_step", "Step", ""), ("tec_sweep_repeat", "Repeat", "")),
    ]
    for group in groups:
        parts = [text for key, label, unit in group if (text := value(key, label, unit))]
        if parts:
            lines.append("  ·  ".join(parts))
    channel = metadata.get("channel", data.column_names[1])
    lines.append(f"{channel}  ·  {data.n_samples:,} samples  ·  {data.sampling_rate:.6g} Hz")
    return lines


class CaptureLoadWorker(QtCore.QThread):
    def __init__(self, generation, *, folder=None, recursive=False, index=None, path=None, parent=None):
        super().__init__(parent)
        self.generation, self.folder, self.recursive = generation, folder, recursive
        self.index, self.path = index, path
        self.result = None
        self.error = None

    def run(self):
        try:
            if self.folder is not None:
                paths = []
                entries = self.folder.rglob("*") if self.recursive else self.folder.iterdir()
                for path in entries:
                    if self.isInterruptionRequested():
                        return
                    if path.suffix.lower() in CAPTURE_EXTENSIONS and path.is_file():
                        paths.append(path)
                self.result = sorted(paths, key=natural_key)
            else:
                data = DataManager.load_file(str(self.path))
                if (data.time.ndim != 1 or data.amplitude.ndim != 1 or data.time.size != data.amplitude.size
                        or not data.time.size or np.iscomplexobj(data.amplitude)
                        or not np.any(np.isfinite(data.time) & np.isfinite(data.amplitude))):
                    raise ValueError("Capture has no matching, real waveform samples")
                self.result = data
        except Exception as exc:
            self.error = str(exc)


class CapturePlaybackWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths = []
        self._index = -1
        self._frame = None
        self._prefetched = None
        self._failed = set()
        self._worker = None
        self._generation = 0
        self._pending_scan = None
        self._desired_index = None
        self._playing = False
        self._waiting = False
        self._shutting_down = False
        self._last_frame_at = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._advance)

        self.control_panel = QtWidgets.QWidget()
        controls = QtWidgets.QVBoxLayout(self.control_panel)
        controls.addWidget(QtWidgets.QLabel("Capture directory"))
        self.directory = QtWidgets.QLineEdit()
        controls.addWidget(self.directory)
        self.browse_button = QtWidgets.QPushButton("Browse…")
        self.browse_button.clicked.connect(self._browse)
        controls.addWidget(self.browse_button)
        self.recursive = QtWidgets.QCheckBox("Include nested subdirectories")
        controls.addWidget(self.recursive)
        self.load_button = QtWidgets.QPushButton("Load directory")
        self.load_button.clicked.connect(self.load_directory)
        controls.addWidget(self.load_button)
        form = QtWidgets.QFormLayout()
        self.interval = QtWidgets.QDoubleSpinBox()
        self.interval.setDecimals(2)
        self.interval.setRange(0.02, 3600)
        self.interval.setValue(1)
        self.interval.setSuffix(" s/frame")
        self.interval.valueChanged.connect(self._arm_timer)
        form.addRow("Frame interval", self.interval)
        controls.addLayout(form)
        self.loop = QtWidgets.QCheckBox("Loop playback")
        self.loop.setChecked(True)
        self.loop.toggled.connect(self._dispatch)
        controls.addWidget(self.loop)
        self.auto_range = QtWidgets.QCheckBox("Auto-range each frame")
        self.auto_range.setChecked(True)
        self.auto_range.toggled.connect(lambda checked: self.fit_view() if checked else None)
        controls.addWidget(self.auto_range)
        self.show_metadata = QtWidgets.QCheckBox("Show metadata")
        self.show_metadata.setChecked(True)
        self.show_metadata.toggled.connect(self._metadata_visibility)
        controls.addWidget(self.show_metadata)
        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        controls.addWidget(self.play_button)
        navigation = QtWidgets.QHBoxLayout()
        self.previous_button = QtWidgets.QPushButton("Previous")
        self.next_button = QtWidgets.QPushButton("Next")
        self.previous_button.clicked.connect(lambda: self.step(-1))
        self.next_button.clicked.connect(lambda: self.step(1))
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.next_button)
        controls.addLayout(navigation)
        self._frame_number = QtWidgets.QSpinBox()
        self._frame_number.setRange(0, 0)
        self._frame_number.setPrefix("Frame ")
        self._frame_number.valueChanged.connect(lambda number: self.seek(number - 1))
        controls.addWidget(self._frame_number)
        self.fit_button = QtWidgets.QPushButton("Fit view")
        self.fit_button.clicked.connect(self.fit_view)
        controls.addWidget(self.fit_button)
        controls.addStretch()

        layout = QtWidgets.QVBoxLayout(self)
        self.plot = pg.PlotWidget(title="Raw oscilloscope capture")
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.setLabel("left", "Amplitude")
        for side in ("left", "bottom"):
            self.plot.getAxis(side).setPen(pg.mkPen("#333333"))
            self.plot.getAxis(side).setTextPen(pg.mkPen("#333333"))
        self.curve = self.plot.plot(pen=pg.mkPen("#2563eb", width=1), connect="finite")
        self.curve.setClipToView(True)
        self.curve.setDownsampling(auto=True, method="peak")
        self.metadata_item = pg.TextItem(anchor=(1, 0), fill=(255, 255, 255, 225), border=pg.mkPen("#cccccc"))
        self.metadata_item.setTextWidth(320)
        self.metadata_item.setZValue(100)
        self.plot.addItem(self.metadata_item, ignoreBounds=True)
        self.metadata_item.hide()
        view = self.plot.getViewBox()
        view.sigRangeChanged.connect(self._position_metadata)
        view.sigResized.connect(self._position_metadata)
        layout.addWidget(self.plot, 1)
        self.status = QtWidgets.QLabel("Select a folder of oscilloscope captures.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._update_controls()

    @property
    def is_busy(self):
        return self._worker is not None

    @property
    def is_playing(self):
        return self._playing

    def _browse(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Capture directory", self.directory.text())
        if folder:
            self.directory.setText(folder)
            self.load_directory()

    def load_directory(self):
        folder = Path(self.directory.text().strip())
        if not self.directory.text().strip() or not folder.is_dir():
            self.status.setText("Please select an existing capture directory.")
            return
        self.pause()
        self._shutting_down = False
        self._generation += 1
        self._paths, self._failed = [], set()
        self._index, self._frame, self._prefetched = -1, None, None
        self._desired_index = None
        self._pending_scan = (folder, self.recursive.isChecked())
        self.curve.setData([], [])
        self.metadata_item.hide()
        self.status.setText("Finding capture files…")
        if self._worker is not None:
            self._worker.requestInterruption()
        self._update_controls()
        self._dispatch()

    def _next_index(self, after=None, direction=1):
        after = self._index if after is None else after
        for distance in range(1, len(self._paths) + 1):
            index = after + direction * distance
            if self.loop.isChecked():
                index %= len(self._paths)
            elif not 0 <= index < len(self._paths):
                return None
            if index not in self._failed:
                return index
        return None

    def _dispatch(self):
        if self._shutting_down or self._worker is not None:
            return
        if self._pending_scan is not None:
            folder, recursive = self._pending_scan
            self._pending_scan = None
            self._start_worker(folder=folder, recursive=recursive)
            return
        if self._desired_index is not None:
            index = self._desired_index
            if self._prefetched is not None and self._prefetched[0] == index:
                self._show_frame(index, self._prefetched[1])
            elif index == self._index and self._frame is not None:
                self._show_frame(index, self._frame)
            else:
                self._start_worker(index=index, path=self._paths[index])
                return
        index = self._next_index()
        if index is None:
            if self._waiting or (self._paths and len(self._failed) == len(self._paths)):
                self.pause()
                if self._frame is None:
                    self.status.setText("No readable waveform captures found.")
            return
        if index == self._index:
            if self._playing and self._waiting:
                self._show_frame(index, self._frame)
            return
        if self._prefetched is not None and self._prefetched[0] == index:
            if self._playing and self._waiting:
                self._show_frame(index, self._prefetched[1])
                self._dispatch()
            return
        self._start_worker(index=index, path=self._paths[index])

    def _start_worker(self, **kwargs):
        self._worker = CaptureLoadWorker(self._generation, parent=self, **kwargs)
        self._worker.finished.connect(self._loaded)
        self._worker.start()

    def _loaded(self):
        worker = self._worker
        self._worker = None
        if not self._shutting_down and worker.generation == self._generation:
            if worker.folder is not None:
                if worker.error:
                    self.status.setText(f"Could not read directory: {worker.error}")
                else:
                    self._paths = worker.result or []
                    self._desired_index = 0 if self._paths else None
                    if not self._paths:
                        self.status.setText("No supported capture files found.")
            elif worker.error:
                self._failed.add(worker.index)
                self.status.setText(f"Skipped {worker.path.name}: {worker.error}")
                if self._desired_index == worker.index:
                    self._desired_index = self._next_index(worker.index)
            elif worker.index == self._desired_index:
                self._show_frame(worker.index, worker.result)
            elif self._desired_index is None:
                self._prefetched = (worker.index, worker.result)
        worker.deleteLater()
        self._update_controls()
        self._dispatch()

    def _show_frame(self, index, data):
        first_frame = self._frame is None
        self._desired_index, self._prefetched = None, None
        self._index, self._frame = index, data
        self._waiting = False
        self._last_frame_at = time.monotonic()
        self.curve.setData(data.time, data.amplitude)
        self.plot.setLabel("left", data.column_names[1], units=data.metadata.get("amplitude_unit", ""))
        lines = capture_metadata_lines(self._paths[index], data, index, len(self._paths))
        self.metadata_item.setHtml('<div style="font-size:9pt; color:#333333">' +
                                   "<br>".join(escape(line) for line in lines) + "</div>")
        self.metadata_item.setToolTip(f"File: {self._paths[index]}\n" +
                                     "\n".join(f"{key}: {value}" for key, value in sorted(data.metadata.items())))
        self._metadata_visibility()
        if first_frame or self.auto_range.isChecked():
            self.fit_view()
        self._position_metadata()
        self.status.setText(f"{'Playing' if self._playing else 'Paused'} · {index + 1}/{len(self._paths)} · {self._paths[index].name}"
                            + (f" · skipped {len(self._failed)} unreadable file(s)" if self._failed else ""))
        self._update_controls()
        self._arm_timer()

    def _position_metadata(self, *_args):
        view = self.plot.getViewBox()
        rectangle = view.sceneBoundingRect()
        self.metadata_item.setPos(view.mapSceneToView(QtCore.QPointF(rectangle.right() - 8, rectangle.top() + 8)))

    def _metadata_visibility(self):
        self.metadata_item.setVisible(self._frame is not None and self.show_metadata.isChecked())

    def fit_view(self):
        if self._frame is not None:
            self.plot.autoRange()

    def _update_controls(self):
        available = bool(self._paths) and len(self._failed) < len(self._paths) and not self._shutting_down
        for button in (self.play_button, self.previous_button, self.next_button):
            button.setEnabled(available)
        self.play_button.setText("Pause" if self._playing else "Play")
        self.fit_button.setEnabled(self._frame is not None)
        self._frame_number.setEnabled(available)
        blocker = QtCore.QSignalBlocker(self._frame_number)
        self._frame_number.setRange(1 if self._paths else 0, len(self._paths))
        self._frame_number.setValue(max(0, self._index + 1))
        self._frame_number.setSuffix(f" / {len(self._paths)}")
        del blocker

    def toggle_playback(self):
        if self._playing:
            self.pause()
            return
        if not self._paths or self._shutting_down or len(self._failed) == len(self._paths):
            return
        self._playing = True
        self._last_frame_at = time.monotonic()
        if self._frame is not None:
            self.status.setText(f"Playing · {self._index + 1}/{len(self._paths)} · {self._paths[self._index].name}")
        if self._frame is not None and self._next_index() is None:
            self._desired_index = 0
        self._update_controls()
        self._arm_timer()
        self._dispatch()

    def pause(self):
        self._playing, self._waiting = False, False
        self._timer.stop()
        if hasattr(self, "status") and self.status.text().startswith("Playing ·"):
            self.status.setText("Paused ·" + self.status.text()[len("Playing ·"):])
        self._update_controls()

    def _arm_timer(self):
        if self._playing and self._frame is not None and not self._waiting:
            remaining = self._last_frame_at + self.interval.value() - time.monotonic()
            self._timer.start(max(1, int(np.ceil(remaining * 1000))))

    def _advance(self):
        if not self._playing:
            return
        if self._next_index() is None:
            self.pause()
            self.status.setText("End of captures.")
            return
        self._waiting = True
        self._dispatch()

    def step(self, direction):
        index = self._next_index(direction=direction)
        if index is not None:
            self.seek(index)

    def seek(self, index):
        if not 0 <= index < len(self._paths) or self._shutting_down:
            return
        self.pause()
        self._failed.discard(index)
        if self._prefetched is not None and self._prefetched[0] != index:
            self._prefetched = None
        self._desired_index = index
        self._dispatch()

    def hideEvent(self, event):
        self.pause()
        super().hideEvent(event)

    def shutdown(self):
        self.pause()
        self._shutting_down = True
        self._pending_scan, self._desired_index = None, None
        if self._worker is not None:
            self._worker.requestInterruption()
            return False
        return True
