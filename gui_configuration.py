"""Versioned GUI settings only: loading never runs acquisition or instrument commands."""
from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import tempfile

from PySide6 import QtCore, QtGui, QtWidgets


def default_config_path():
    override = os.environ.get("SIGNAL_VISUALIZATION_CONFIG")
    if override:
        return Path(override)
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "SignalVisualizationApp" / "last_config.json"


def controls_for(window):
    """Use stable Python attribute names, never Qt's private plot/editor children."""
    controls = {}
    seen = set()

    def visit(key, value):
        if id(value) in seen:
            return
        supported = isinstance(value, (QtWidgets.QLineEdit, QtWidgets.QComboBox,
                                       QtWidgets.QAbstractSpinBox, QtWidgets.QAbstractButton,
                                       QtWidgets.QGroupBox, QtGui.QAction))
        if supported:
            if isinstance(value, (QtWidgets.QAbstractButton, QtWidgets.QGroupBox, QtGui.QAction)) and not value.isCheckable():
                return
            if isinstance(value, QtWidgets.QLineEdit) and value.isReadOnly():
                return
            seen.add(id(value))
            controls[key] = value
        elif isinstance(value, dict):
            for child_key, child in value.items():
                visit(f"{key}.{child_key}", child)
        elif hasattr(value, "UNIT_FACTORS") and hasattr(value, "value_spin") and hasattr(value, "unit_combo"):
            visit(key + ".value_spin", value.value_spin)
            visit(key + ".unit_combo", value.unit_combo)

    for section, owner in (("time", window), ("sweep", window.sweep_widget),
                           ("recordings", window.recording_widget), ("tec_sine", window.tec_sine_widget),
                           ("tec_pid", window.tec_pid_widget), ("playback", window.playback_widget)):
        for name, value in vars(owner).items():
            if not name.startswith("_"):
                visit(f"{section}.{name}", value)
    return controls


def collect_configuration(window):
    settings = {}
    for key, control in controls_for(window).items():
        if isinstance(control, QtWidgets.QComboBox):
            value = control.currentText()
        elif isinstance(control, QtWidgets.QLineEdit):
            value = control.text()
        elif isinstance(control, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            value = control.value()
        else:
            value = control.isChecked()
        settings[key] = value
    return {"application": "SignalVisualizationApp", "version": 1,
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"), "settings": settings}


def save_configuration(window, path):
    path = Path(path)
    payload = json.dumps(collect_configuration(window), indent=2, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after a complete write; retain the last good config on failure.
    descriptor, staging = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        os.replace(staging, path)
    finally:
        Path(staging).unlink(missing_ok=True)


def load_configuration(window, path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(payload, dict) or payload.get("application") != "SignalVisualizationApp"
            or type(payload.get("version")) is not int or payload["version"] != 1
            or not isinstance(payload.get("settings"), dict)):
        raise ValueError("Not a supported Signal Visualization configuration (version 1 required)")
    controls = controls_for(window)
    changes = []
    sine = window.tec_sine_widget
    frequency_unit = payload["settings"].get("tec_sine.frequency_unit", sine.frequency_unit.currentText())
    if frequency_unit not in sine.FREQUENCY_UNITS:
        raise ValueError("Invalid TEC sine frequency unit")
    # Validate the entire document before changing any controls.
    for key, value in payload["settings"].items():
        control = controls.get(key)
        if control is None:
            continue  # Settings from a newer GUI may be absent in this version.
        if isinstance(control, (QtWidgets.QComboBox, QtWidgets.QLineEdit)):
            valid = isinstance(value, str)
            if valid and isinstance(control, QtWidgets.QComboBox) and not control.isEditable():
                valid = (control.findText(value) >= 0 or "resource" in key
                         or key in {"time.combo_time_column", "time.combo_amplitude_column"}
                         or (not value and control.count() == 0))
        elif isinstance(control, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            minimum = 1e-9 / sine.FREQUENCY_UNITS[frequency_unit] if control is sine.frequency else control.minimum()
            valid = (type(value) in (int, float) and math.isfinite(value)
                     and minimum <= value <= control.maximum())
            if isinstance(control, QtWidgets.QSpinBox):
                valid = valid and int(value) == value
        else:
            valid = type(value) is bool
        if not valid:
            raise ValueError(f"Invalid configuration value for {key}: {value!r}")
        changes.append((control, value))
    blockers = [QtCore.QSignalBlocker(control) for control in controls.values()]
    try:
        sine.frequency.setMinimum(1e-9 / sine.FREQUENCY_UNITS[frequency_unit])
        for control, value in changes:
            if isinstance(control, QtWidgets.QComboBox):
                if control.isEditable():
                    control.setEditText(value)
                else:
                    if control.findText(value) < 0 and value:
                        control.addItem(value)
                    control.setCurrentText(value)
            elif isinstance(control, QtWidgets.QLineEdit):
                control.setText(value)
            elif isinstance(control, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                control.setValue(value)
            else:
                control.setChecked(value)
        # Prefix-conversion slots were suppressed; synchronize their caches
        # without converting the already restored display values a second time.
        for owner in (window, window.sweep_widget):
            for value in vars(owner).values():
                if hasattr(value, "_unit_factor") and hasattr(value, "UNIT_FACTORS"):
                    value._unit_factor = value.UNIT_FACTORS[value.unit_combo.currentText()]
        sine._frequency_factor = sine.FREQUENCY_UNITS[sine.frequency_unit.currentText()]
        sine.frequency.setMinimum(1e-9 / sine._frequency_factor)
    finally:
        blockers.clear()
    return len(changes)
