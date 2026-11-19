"""Two-channel Qt integration of the user's standalone TEC207L controller."""
from __future__ import annotations

from collections import deque
from datetime import datetime
import math
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
from serial.tools import list_ports

from plot_panel_widget import PlotPanel
from ui_controls import CollapsibleSection
from tec207l.device import OUTPUT_MODES, SENSOR_MODELS
from tec207l.autopid import AutopidSettings
from tec207l_control import Tec207LConnection, Tec207LJob, make_program, write_history


class HistoryExport(QtCore.QThread):
    done = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, path, records, parent=None):
        super().__init__(parent)
        self.path, self.records = path, records

    def run(self):
        try:
            write_history(self.path, self.records)
            self.done.emit(str(self.path))
        except Exception as exc:
            self.failed.emit(str(exc))


class Tec207LWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._connection = None
        self._connected = False
        self._disconnect_requested = False
        self._jobs = {}
        self._pending = set()
        self._all_off_pending = False
        self._history = {ch: deque(maxlen=10000) for ch in (1, 2)}
        self._started_at = None
        self._logging_path = ""
        self._export = None
        # Flat, stable names keep both channels' drafts in the shared config file.
        self.fields = {}
        self._channel_editors = {ch: [] for ch in (1, 2)}
        self._sections = []
        self._build_ui()
        self._refresh_ports()
        self._update_controls()

    def _field(self, channel, name, control):
        self.fields[f"CH{channel}.{name}"] = control
        self._channel_editors[channel].append(control)
        return control

    def _number(self, channel, name, value, minimum, maximum, decimals=3, suffix=""):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return self._field(channel, name, spin)

    def _integer(self, channel, name, value, minimum=0, maximum=9000000):
        spin = QtWidgets.QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return self._field(channel, name, spin)

    def _form(self, parent_layout, title):
        content = QtWidgets.QGroupBox()
        form = QtWidgets.QFormLayout(content)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        section = CollapsibleSection(title, content)
        parent_layout.addWidget(section)
        self._sections.append(section)
        return form

    def _action(self, form, channel, label, callback, beside=None):
        button = QtWidgets.QPushButton(label)
        button.clicked.connect(callback)
        self._channel_editors[channel].append(button)
        if beside is None:
            form.addRow(button)
        else:
            row = QtWidgets.QHBoxLayout()
            row.addWidget(beside, 1)
            row.addWidget(button)
            form.addRow(row)
        return button

    def _build_ui(self):
        sidebar = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(sidebar)
        connection = QtWidgets.QGroupBox("TEC207L connection")
        connect_form = QtWidgets.QFormLayout(connection)
        self.port = QtWidgets.QComboBox()
        self.port.setEditable(True)
        self.port.lineEdit().setPlaceholderText("COM port / USB-TTL adapter")
        self.baudrate = QtWidgets.QComboBox()
        self.baudrate.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.baudrate.setCurrentText("38400")
        self.poll_seconds = QtWidgets.QDoubleSpinBox()
        self.poll_seconds.setRange(.1, 60)
        self.poll_seconds.setValue(.3)
        self.poll_seconds.setSuffix(" s")
        self._refresh = QtWidgets.QPushButton("Refresh ports")
        self._refresh.clicked.connect(self._refresh_ports)
        self._connect_button = QtWidgets.QPushButton("Connect")
        self._connect_button.clicked.connect(self._toggle_connection)
        port_row = QtWidgets.QHBoxLayout()
        port_row.addWidget(self.port, 1)
        self._refresh.setText("Refresh")
        port_row.addWidget(self._refresh)
        connect_form.addRow("Port", port_row)
        connect_form.addRow("Baud", self.baudrate)
        connect_form.addRow(self._connect_button)
        side.addWidget(connection)
        advanced = self._form(side, "Connection timing")
        advanced.addRow("Poll interval", self.poll_seconds)
        self.channel = QtWidgets.QComboBox()
        self.channel.addItems(["CH1", "CH2"])
        side.addWidget(self.channel)
        self._channel_stack = QtWidgets.QStackedWidget()
        self.channel.currentIndexChanged.connect(self._select_channel)
        self.channel.currentIndexChanged.connect(self._redraw)
        for ch in (1, 2):
            self._channel_stack.addWidget(self._build_channel(ch))
        self._select_channel(0)
        side.addWidget(self._channel_stack)
        self._stop = QtWidgets.QPushButton("Stop selected channel program")
        self._stop.clicked.connect(self._stop_selected)
        side.addWidget(self._stop)
        self._all_off = QtWidgets.QPushButton("All outputs OFF")
        self._all_off.clicked.connect(self._outputs_off)
        side.addWidget(self._all_off)
        log = self._form(side, "Data logging / export")
        self.log_directory = QtWidgets.QLineEdit(str(Path.cwd() / "logs" / "tec207l"))
        folder_button = QtWidgets.QPushButton("Choose log folder")
        folder_button.clicked.connect(self._choose_folder)
        self.log_name = QtWidgets.QLineEdit()
        self.log_name.setPlaceholderText("Automatic timestamped filename")
        self._log_button = QtWidgets.QPushButton("Start logging both channels")
        self._log_button.clicked.connect(self._toggle_logging)
        log.addRow("Folder", self.log_directory)
        log.addRow(folder_button)
        log.addRow("CSV name", self.log_name)
        log.addRow(self._log_button)
        self._log_label = QtWidgets.QLabel()
        self._log_label.setWordWrap(True)
        log.addRow(self._log_label)
        side.addStretch(1)
        self.control_panel = QtWidgets.QScrollArea()
        self.control_panel.setWidgetResizable(True)
        self.control_panel.setWidget(sidebar)
        self.control_panel.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.control_panel.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        main = QtWidgets.QVBoxLayout(self)
        self._status = QtWidgets.QLabel("Disconnected; connecting reads both channels without applying GUI defaults.")
        self._status.setWordWrap(True)
        self._status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        main.addWidget(self._status)
        readouts = QtWidgets.QHBoxLayout()
        self._readouts = {}
        self._limit_readouts = {}
        for ch in (1, 2):
            group = QtWidgets.QGroupBox(f"CH{ch}")
            layout = QtWidgets.QVBoxLayout(group)
            label = QtWidgets.QLabel("Temperature: —\nTarget: —   Current: —\nResistance: —   Output: —")
            label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            layout.addWidget(label)
            self._readouts[ch] = label
            limit = self._integer(ch, "limit", 30, 0, 90)
            limit.setPrefix("Max output: ")
            limit.setSuffix(" % Vin")
            limit.setToolTip("Maximum output voltage relative to this channel's supply voltage (0–90%). "
                             "This is not a power limit in watts. Applied and verified before every program or PID tune.")
            limit_form = QtWidgets.QFormLayout()
            layout.addLayout(limit_form)
            self._action(limit_form, ch, "Apply limit", lambda _checked=False, channel=ch: self._apply(channel, "limit"), beside=limit)
            self._limit_readouts[ch] = QtWidgets.QLabel("Device limit: —")
            layout.addWidget(self._limit_readouts[ch])
            readouts.addWidget(group)
        main.addLayout(readouts)
        toolbar = QtWidgets.QHBoxLayout()
        self.combine_plot = QtWidgets.QCheckBox("Show both channels")
        self.combine_plot.setChecked(True)
        self.combine_plot.toggled.connect(self._redraw)
        self.follow_plot = QtWidgets.QCheckBox("Follow latest")
        self.follow_plot.setChecked(True)
        self.history_seconds = QtWidgets.QDoubleSpinBox()
        self.history_seconds.setRange(10, 86400)
        self.history_seconds.setValue(180)
        self.history_seconds.setSuffix(" s history")
        self.history_seconds.valueChanged.connect(self._redraw)
        self._export_button = QtWidgets.QPushButton("Export history CSV")
        self._export_button.clicked.connect(self._export_history)
        clear = QtWidgets.QPushButton("Clear plot")
        clear.clicked.connect(self._clear_plot)
        for control in (self.combine_plot, self.follow_plot, self.history_seconds, self._export_button, clear):
            toolbar.addWidget(control)
        main.addLayout(toolbar)
        self._views = QtWidgets.QTabWidget()
        self.plot = PlotPanel("TEC207L temperatures")
        self.plot.set_axis_labels(bottom="Elapsed time", bottom_units="s", left="Temperature (°C)")
        self._views.addTab(self.plot, "Live temperatures")
        self._results = QtWidgets.QTableWidget(0, 6)
        self._results.setHorizontalHeaderLabels(["Channel", "Target (°C)", "Settled", "Settle time (s)", "In band (s)", "Final (°C)"])
        self._results.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._results.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._views.addTab(self._results, "Step response results")
        self._communication = QtWidgets.QPlainTextEdit()
        self._communication.setReadOnly(True)
        self._communication.setMaximumBlockCount(1000)
        self._views.addTab(self._communication, "Communication")
        main.addWidget(self._views, 1)
        self._job_status = {}
        for ch in (1, 2):
            label = QtWidgets.QLabel(f"CH{ch}: idle")
            label.setWordWrap(True)
            label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            self._job_status[ch] = label
            main.addWidget(label)
        self.channel.currentIndexChanged.connect(self._update_controls)

    def _build_channel(self, ch):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        basic = QtWidgets.QGroupBox(f"CH{ch} setpoint / output")
        form = QtWidgets.QFormLayout(basic)
        target = self._number(ch, "target", 25, -40, 100, 5, " °C")
        target.setPrefix("Target: ")
        self._action(form, ch, "Set", lambda: self._apply(ch, "target"), beside=target)
        enabled = self._field(ch, "enable", QtWidgets.QCheckBox("Output enabled"))
        self._action(form, ch, "Apply output", lambda: self._apply(ch, "output"), beside=enabled)
        self._action(form, ch, "Read channel from device", lambda: self._submit(ch, "read"))
        layout.addWidget(basic)
        form = self._form(layout, "Output settings")
        mode = self._field(ch, "mode", QtWidgets.QComboBox())
        for key, label in OUTPUT_MODES.items():
            mode.addItem(label, key)
        form.addRow("Mode", mode)
        form = self._form(layout, "PID / auto-tuning")
        for name, value in (("p", 3000), ("i", 150), ("d", 0)):
            form.addRow(name.upper(), self._integer(ch, f"pid_{name}", value))
        self._action(form, ch, "Apply PID", lambda: self._apply(ch, "pid"))
        self._action(form, ch, "Read PID", lambda: self._submit(ch, "read_pid"))
        self._action(form, ch, "Start device PID auto-tune", lambda: self._start_job(ch, "hardware_tune"))
        self._action(form, ch, "Start Ziegler-Nichols auto-PID", lambda: self._start_job(ch, "zn_tune"))
        form = self._form(layout, "Ziegler-Nichols settings")
        form.addRow("Starting P", self._integer(ch, "zn_p_start", 2000, 1))
        form.addRow("Maximum P", self._integer(ch, "zn_p_max", 200000, 1))
        for name, label, value, low, high, decimals, suffix in (
                ("settle_s", "Settle per gain", 30, 0, 86400, 1, " s"),
                ("max_window_s", "Observe per gain", 600, 20, 86400, 1, " s"),
                ("sample_s", "Sample interval", .5, .1, 60, 2, " s"),
                ("hysteresis", "Reversal threshold", .005, .0001, 1, 4, " °C"),
                ("min_period_s", "Minimum period", 5, 1, 3600, 1, " s"),
                ("max_deviation", "Max error from target", 5, .01, 50, 3, " °C")):
            form.addRow(label, self._number(ch, "zn_" + name, value, low, high, decimals, suffix))
        note = QtWidgets.QLabel("Keeps this channel's Max output limit. Requires independent channels (CONTMODE=0). "
                               "Saves a tuning CSV and diagnostic JSON in the log folder.")
        note.setWordWrap(True)
        form.addRow(note)
        form = self._form(layout, "Sensor configuration")
        model = self._field(ch, "model", QtWidgets.QComboBox())
        for key, label in SENSOR_MODELS.items():
            model.addItem(label, key)
        form.addRow("Model", model)
        for name, label, value in (("ntc_b", "NTC B", "3950"), ("ntc_r0", "NTC R0 (Ω)", "10000"),
                                   ("ntc_ref", "NTC reference (Ω)", "10000"), ("pt_r0", "PT R0 (Ω)", "1000"),
                                   ("pt_a", "PT A", "3.9083e-3"), ("pt_b", "PT B", "-5.775e-7"),
                                   ("pt_c", "PT C", "-4.183e-12"), ("pt_ref", "PT reference (Ω)", "2000")):
            form.addRow(label, self._field(ch, name, QtWidgets.QLineEdit(value)))
        self._action(form, ch, "Apply sensor settings", lambda: self._apply(ch, "sensor"))
        self._action(form, ch, "Read sensor settings", lambda: self._submit(ch, "read_sensor"))
        form = self._form(layout, "Temperature range sweep")
        for name, label, value, low, high, suffix in (
                ("minimum", "Minimum", 20, -40, 100, " °C"), ("maximum", "Maximum", 30, -40, 100, " °C"),
                ("increment", "Increment", 1, .001, 140, " °C"), ("dwell_s", "Time per step", 60, .1, 604800, " s")):
            form.addRow(label, self._number(ch, name, value, low, high, 3, suffix))
        form.addRow("Repeats (N)", self._integer(ch, "repeats", 1, 1, 100000))
        self._action(form, ch, "Start temperature sweep", lambda: self._start_job(ch, "range"))
        form = self._form(layout, "Sine / DC offset sweep")
        form.addRow("Center", self._number(ch, "center", 25, -40, 100, 5, " °C"))
        form.addRow("AC amplitude", self._number(ch, "amplitude", 1, 0, 70, 5, " °C"))
        form.addRow("Frequency", self._number(ch, "frequency_hz", .001, 1e-9, .05, 9, " Hz"))
        form.addRow("Update interval", self._number(ch, "update_s", 5, 1, 86400, 3, " s"))
        form.addRow("Sine cycles", self._integer(ch, "cycles", 1, 1, 100000))
        sweep = self._field(ch, "offset_sweep", QtWidgets.QCheckBox("Sweep DC offset"))
        sweep.setToolTip("Uses minimum, maximum, increment, dwell and repeats from Temperature range sweep.")
        form.addRow(sweep)
        self._action(form, ch, "Start sine program", lambda: self._start_job(ch, "sine_sweep" if sweep.isChecked() else "sine"))
        form = self._form(layout, "Step response test")
        form.addRow("Targets (°C)", self._field(ch, "targets", QtWidgets.QLineEdit("20,22,24,26,28,30")))
        form.addRow("Accuracy ±", self._number(ch, "accuracy", .01, .00001, 10, 5, " °C"))
        form.addRow("Hold interval", self._number(ch, "hold_s", 20, .1, 3600, 2, " s"))
        form.addRow("Timeout per step", self._number(ch, "timeout_s", 1800, 1, 86400, 1, " s"))
        self._action(form, ch, "Start step response test", lambda: self._start_job(ch, "step"))
        auto = self._field(ch, "auto_output", QtWidgets.QCheckBox("Output ON at start / OFF after job"))
        auto.setChecked(True)
        layout.addWidget(auto)
        layout.addStretch(1)
        return panel

    def _select_channel(self, index):
        # An expanded hidden channel must not force the active page to grow.
        for i in range(self._channel_stack.count()):
            page = self._channel_stack.widget(i)
            page.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                               QtWidgets.QSizePolicy.Policy.Preferred if i == index else QtWidgets.QSizePolicy.Policy.Ignored)
        self._channel_stack.setCurrentIndex(index)
        self._channel_stack.updateGeometry()

    def _value(self, ch, name):
        control = self.fields[f"CH{ch}.{name}"]
        if isinstance(control, QtWidgets.QComboBox):
            return control.currentData()
        if isinstance(control, QtWidgets.QAbstractButton):
            return control.isChecked()
        if isinstance(control, QtWidgets.QLineEdit):
            return control.text()
        return control.value()

    def _set_value(self, ch, name, value):
        control = self.fields[f"CH{ch}.{name}"]
        blocker = QtCore.QSignalBlocker(control)
        if isinstance(control, QtWidgets.QComboBox):
            control.setCurrentIndex(control.findData(value))
        elif isinstance(control, QtWidgets.QAbstractButton):
            control.setChecked(bool(value))
        elif isinstance(control, QtWidgets.QLineEdit):
            control.setText(str(value))
        else:
            # Hardware readback must be represented exactly rather than silently clamped.
            control.setRange(min(control.minimum(), value), max(control.maximum(), value))
            control.setValue(value)
        del blocker

    def _fill_state(self, ch, state, category="all"):
        if category in {"all", "pid"}:
            for key, value in zip(("pid_p", "pid_i", "pid_d"), state["pid"]):
                self._set_value(ch, key, value)
        if category in {"all", "sensor"}:
            self._set_value(ch, "model", state["model"])
            for key, value in zip(("ntc_b", "ntc_r0", "ntc_ref"), state["ntc"]):
                self._set_value(ch, key, value)
            for key, value in zip(("pt_r0", "pt_a", "pt_b", "pt_c", "pt_ref"), state["pt"]):
                self._set_value(ch, key, value)
        if category == "all":
            for key, value in (("target", state["target"]), ("enable", state["enable"]),
                               ("mode", state["mode"]), ("limit", state["limit"])):
                self._set_value(ch, key, value)
            self._show_limit(ch, state["limit"])

    def _show_limit(self, ch, value):
        self._limit_readouts[ch].setText(f"Device limit: {value} % Vin")

    def _refresh_ports(self):
        current = self.port.currentText()
        self.port.clear()
        self.port.addItems([item.device for item in list_ports.comports()])
        if current:
            self.port.setEditText(current)

    def _toggle_connection(self):
        if self._connection is not None:
            self._request_disconnect()
            return
        if not self.port.currentText().strip():
            self._message("Select a TEC207L COM port first.")
            return
        self._disconnect_requested = False
        worker = Tec207LConnection(self.port.currentText().strip(), int(self.baudrate.currentText()), self.poll_seconds.value(), self)
        self._connection = worker
        worker.ready.connect(self._ready)
        worker.sample.connect(self._sample)
        worker.command_done.connect(self._command_done)
        worker.warning.connect(self._message)
        worker.failed.connect(self._connection_failed)
        worker.transaction.connect(self._communication.appendPlainText)
        worker.logging_changed.connect(self._logging_changed)
        worker.finished.connect(self._disconnected)
        self._message("Connecting and reading both channels…")
        worker.start()
        self._update_controls()

    def _ready(self, info):
        self._connected = True
        for ch, state in info["states"].items():
            self._fill_state(ch, state)
        self._message(f"{info['model']} firmware {info['firmware']} connected; channel configuration read from device.")
        self._update_controls()
        if self._disconnect_requested:
            self._request_disconnect()

    def _message(self, text):
        self._status.setText(str(text))
        self._communication.appendPlainText(str(text))

    def _connection_failed(self, text):
        self._message(f"TEC207L stopped: {text}")
        self._request_disconnect()

    def _request_disconnect(self):
        self._disconnect_requested = True
        for job in self._jobs.values():
            job.stop_event.set()
        if self._connection and not self._jobs:
            self._connection.stop_event.set()
        self._update_controls()

    def _disconnected(self):
        worker, self._connection = self._connection, None
        self._connected = False
        self._pending.clear()
        self._all_off_pending = False
        if not self._status.text().startswith("TEC207L stopped:"):
            self._message("Disconnected")
        if worker:
            worker.deleteLater()
        self._update_controls()

    def _submit(self, ch, action, args=()):
        try:
            if not self._connected or self._disconnect_requested:
                raise ValueError("Connect to the TEC207L first")
            if ch in self._pending:
                raise ValueError(f"CH{ch}: wait for the current command")
            self._connection.submit(ch, action, args)
            if ch:
                self._pending.add(ch)
            self._update_controls()
            return True
        except Exception as exc:
            self._message(str(exc))
            return False

    def _apply(self, ch, action):
        try:
            if action == "target":
                args = (self._value(ch, "target"),)
            elif action == "limit":
                args = (self._value(ch, "limit"),)
            elif action == "output":
                args = tuple(self._value(ch, key) for key in ("enable", "mode", "limit"))
            elif action == "pid":
                args = tuple(self._value(ch, key) for key in ("pid_p", "pid_i", "pid_d"))
            else:
                ntc = tuple(float(self._value(ch, key)) for key in ("ntc_b", "ntc_r0", "ntc_ref"))
                pt = tuple(float(self._value(ch, key)) for key in ("pt_r0", "pt_a", "pt_b", "pt_c", "pt_ref"))
                if not all(math.isfinite(value) for value in (*ntc, *pt)) or min(*ntc, pt[0], pt[4]) <= 0:
                    raise ValueError("Sensor coefficients must be finite; B and resistances must be positive")
                args = (self._value(ch, "model"), ntc, pt)
            self._submit(ch, action, args)
        except Exception as exc:
            self._message(str(exc))

    def _command_done(self, ch, action, result):
        self._pending.discard(ch)
        if ch == 0 and (action == "all_off" or (action == "error" and result["action"] == "all_off")):
            self._all_off_pending = False
        if action == "read":
            self._fill_state(ch, result)
        elif action in {"pid", "read_pid"}:
            self._fill_state(ch, {"pid": result}, "pid")
        elif action == "read_sensor":
            self._fill_state(ch, result, "sensor")
        elif action == "limit":
            self._show_limit(ch, result)
        elif action == "output":
            self._show_limit(ch, result["limit"])
        elif action == "all_off":
            for channel in (1, 2):
                self._set_value(channel, "enable", False)
        if action != "error":
            self._message(f"{action}: completed" + (f" on CH{ch}" if ch else ""))
        self._update_controls()

    def _job_settings(self, ch, kind):
        names = ("minimum", "maximum", "increment", "dwell_s", "repeats", "center", "amplitude", "frequency_hz",
                 "update_s", "cycles", "auto_output", "accuracy", "hold_s", "timeout_s", "limit")
        settings = {key: self._value(ch, key) for key in names}
        settings["mode"] = kind
        if kind in {"range", "sine", "sine_sweep"}:
            program, _, _ = make_program(settings)
            if program.minimum < -40 or program.maximum > 100:
                raise ValueError("Temperature program must stay between -40 and 100 °C")
        if kind == "step":
            targets = [float(value.strip()) for value in self._value(ch, "targets").split(",") if value.strip()]
            if not targets or len(targets) > 10000 or not all(math.isfinite(v) and -40 <= v <= 100 for v in targets):
                raise ValueError("Enter a comma-separated target list between -40 and 100 °C")
            settings.update(targets=targets, result_path=str(self._result_path(f"steptest_ch{ch}")))
        if kind == "zn_tune":
            params = {name: self._value(ch, "zn_" + name) for name in AutopidSettings.__dataclass_fields__}
            AutopidSettings(**params)
            settings.update(zn_settings=params, result_path=str(self._result_path(f"autopid_ch{ch}")))
        return settings

    def _start_job(self, ch, kind):
        reserved = False
        try:
            if not self._connected or self._disconnect_requested:
                raise ValueError("Connect to the TEC207L first")
            if self._all_off_pending:
                raise ValueError("Wait for the outputs to switch off")
            settings = self._job_settings(ch, kind)
            self._connection.reserve_channel(ch)
            reserved = True
            job = Tec207LJob(self._connection.device, ch, kind, settings, self)
            self._jobs[ch] = job
            job.progress.connect(self._job_progress)
            job.completed.connect(self._job_completed)
            job.failed.connect(lambda channel, text: self._job_status[channel].setText(f"CH{channel}: error: {text}"))
            job.finished.connect(lambda channel=ch: self._job_finished(channel))
            self._job_status[ch].setText(f"CH{ch}: starting {kind}")
            job.start()
            self._update_controls()
        except Exception as exc:
            if reserved and (ch not in self._jobs or not self._jobs[ch].isRunning()):
                job = self._jobs.pop(ch, None)
                if job:
                    job.deleteLater()
                self._connection.release_channel(ch)
            self._message(str(exc))

    def _job_progress(self, channel, text):
        self._job_status[channel].setText(f"CH{channel}: {text}")
        if self._jobs.get(channel) and self._jobs[channel].kind == "zn_tune":
            self._communication.appendPlainText(f"CH{channel}: {text}")

    def _job_completed(self, ch, result):
        self._job_status[ch].setText(f"CH{ch}: {'stopped' if result.get('aborted') else 'completed'}"
                                     + (f"; {result['result_path']}" if result.get("result_path") else ""))
        if result.get("pid") and "results" not in result:
            self._fill_state(ch, {"pid": result["pid"]}, "pid")
        elif "kp" in result:
            self._fill_state(ch, {"pid": (result["kp"], result["ki"], result["kd"])}, "pid")
        for row in result.get("results", []):
            index = self._results.rowCount()
            self._results.insertRow(index)
            for col, value in enumerate((ch, row["target"], row["settled"], row["settle_time_s"], row["in_band_s"], row["final_temp"])):
                self._results.setItem(index, col, QtWidgets.QTableWidgetItem("—" if value is None else str(value)))

    def _job_finished(self, ch):
        job = self._jobs.pop(ch)
        self._connection.release_channel(ch)
        job.deleteLater()
        if self._disconnect_requested:
            self._request_disconnect()
        self._update_controls()

    def _stop_selected(self):
        job = self._jobs.get(self.channel.currentIndex() + 1)
        if job:
            job.stop_event.set()

    def _outputs_off(self):
        if not self._connected or self._disconnect_requested or self._all_off_pending:
            return
        for job in self._jobs.values():
            job.force_off.set()
            job.stop_event.set()
        self._all_off_pending = True
        if not self._submit(0, "all_off"):
            self._all_off_pending = False
            self._update_controls()

    def _sample(self, ch, sample):
        if self._started_at is None:
            self._started_at = sample["monotonic"]
        self._history[ch].append(sample)
        if "limit" in sample:
            self._show_limit(ch, sample["limit"])
        self._readouts[ch].setText(f"Temperature: {sample['actual']:.5f} °C\nTarget: {sample['target']:.5f} °C   "
                                  f"Current: {sample['current']:.3f} A\nResistance: {sample['resistance']:.3f} Ω   "
                                  f"Output: {'ON' if sample['enable'] else 'OFF'}")
        self._redraw()

    def _redraw(self, *_):
        if not hasattr(self, "plot"):
            return
        traces = {}
        channels = (1, 2) if self.combine_plot.isChecked() else (self.channel.currentIndex() + 1,)
        for ch in channels:
            samples = self._history[ch]
            if not samples:
                continue
            selected = [s for s in samples if s["monotonic"] >= samples[-1]["monotonic"] - self.history_seconds.value()]
            t = np.array([s["monotonic"] - self._started_at for s in selected])
            for key, name in (("actual", "temperature"), ("target", "target")):
                traces[f"CH{ch} {name}"] = (t, np.array([s[key] for s in selected]))
        self.plot.set_traces(traces, auto_range=self.follow_plot.isChecked())
        curves = [self.plot._curve, *self.plot._additional_curves]
        for index, (label, curve) in enumerate(zip(traces, curves)):
            color = "#2563eb" if label.startswith("CH1") else "#dc2626"
            pen = pg.mkPen(color, width=1.4, style=QtCore.Qt.PenStyle.DashLine if label.endswith("target") else QtCore.Qt.PenStyle.SolidLine)
            if index == 0:
                self.plot.set_pen(pen)
            else:
                curve.setPen(pen)

    def _clear_plot(self):
        for history in self._history.values():
            history.clear()
        self._started_at = None
        self.plot.clear()

    def _result_path(self, prefix):
        folder = Path(self.log_directory.text().strip())
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"

    def _choose_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "TEC207L log folder", self.log_directory.text())
        if path:
            self.log_directory.setText(path)

    def _toggle_logging(self):
        try:
            if self._logging_path:
                self._submit(0, "log_stop")
            else:
                name = self.log_name.text().strip()
                if name and (Path(name).name != name or not name.lower().endswith(".csv")):
                    raise ValueError("Use a CSV filename without a directory")
                path = self._result_path("tec207l")
                if name:
                    path = path.with_name(name)
                self._submit(0, "log_start", (str(path),))
        except Exception as exc:
            self._message(str(exc))

    def _logging_changed(self, path):
        self._logging_path = path
        self._log_button.setText("Stop logging" if path else "Start logging both channels")
        self._log_label.setText(path)

    def _export_history(self):
        if self._export is not None:
            return
        records = [(ch, sample) for ch, samples in self._history.items() for sample in samples]
        if not records:
            self._message("No TEC207L history to export")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export TEC207L history", "tec207l_history.csv", "CSV (*.csv)")
        if path:
            self._export = HistoryExport(path, records, self)
            self._export.done.connect(lambda p: self._message(f"History saved: {p}"))
            self._export.failed.connect(self._message)
            self._export.finished.connect(self._export_finished)
            self._export.start()
            self._update_controls()

    def _export_finished(self):
        worker, self._export = self._export, None
        worker.deleteLater()
        self._update_controls()

    def _update_controls(self):
        active = self._connection is not None
        idle_connection = self._connected and not self._disconnect_requested
        for control in (self.port, self.baudrate, self.poll_seconds, self._refresh):
            control.setEnabled(not active)
        self._connect_button.setEnabled(not self._disconnect_requested or not active)
        self._connect_button.setText("Disconnect" if active else "Connect")
        for ch in (1, 2):
            for control in self._channel_editors[ch]:
                editable = ((idle_connection or not active) and not self._all_off_pending
                            and ch not in self._jobs and ch not in self._pending)
                control.setEnabled(editable and (idle_connection or not isinstance(control, QtWidgets.QPushButton)))
        self._stop.setEnabled(idle_connection and self.channel.currentIndex() + 1 in self._jobs)
        self._all_off.setEnabled(idle_connection and not self._all_off_pending)
        self._log_button.setEnabled(idle_connection)
        self._export_button.setEnabled(self._export is None)

    def refresh_configuration(self):
        self._select_channel(self.channel.currentIndex())
        self._update_controls()
        self._redraw()

    def configuration_busy(self):
        return self._connection is not None or self._export is not None

    def shutdown(self):
        if self._connection is not None:
            self._request_disconnect()
        return self._connection is None and self._export is None
