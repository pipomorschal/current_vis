"""Scope recordings paired with verified TEC commands and timestamped DMM voltage."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import math
from pathlib import Path
import threading
import time

from PySide6 import QtCore

from scope_capture_process import capture_isolated
from scope_recording_io import next_scope_logging_path, save_scope_recording
from tec_sine import sine_program_metadata


class SineRunState:
    """Small snapshots shared by independent TEC, DMM and scope workers."""
    def __init__(self):
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.setpoint = None
        self.meter = None
        self.meter_error = None
        self.started = None

    def set_target(self, point, started):
        with self.lock:
            self.setpoint = dict(point)
            self.started = started
        self.ready.set()

    def set_meter(self, row):
        with self.lock:
            self.meter = dict(row)
            self.meter_error = None

    def fail_meter(self, message):
        with self.lock:
            self.meter_error = message

    def snapshot(self):
        with self.lock:
            return self.started, dict(self.setpoint or {}), dict(self.meter or {}), self.meter_error


@dataclass(frozen=True)
class SineLogSettings:
    folder: Path
    interval_seconds: float = 10.0
    extension: str = ".h5"
    include_dmm: bool = True
    maximum_seconds: float = 3600.0
    meter_max_age: float = 10.0
    meter_wait_seconds: float = 30.0

    def __post_init__(self):
        if not Path(self.folder).is_dir():
            raise ValueError("Select an existing recording folder")
        if self.extension not in {".h5", ".csv"}:
            raise ValueError("Choose HDF5 or CSV recording format")
        if (not all(math.isfinite(v) for v in (self.interval_seconds, self.maximum_seconds,
                                              self.meter_max_age, self.meter_wait_seconds))
                or self.interval_seconds <= 0 or self.maximum_seconds < 0
                or self.meter_max_age <= 0 or self.meter_wait_seconds <= 0):
            raise ValueError("Logging interval and meter timing must be positive; maximum duration must be non-negative")


class SineRecordingWorker(QtCore.QThread):
    saved = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    status = QtCore.Signal(str)

    def __init__(self, program, config, settings, state, duration=None, parent=None):
        super().__init__(parent)
        self.program, self.settings, self.state = program, settings, state
        # Scope acquisition must never write or open the TEC while its sine worker
        # owns the controller. Only verified command metadata is attached here.
        self.config = replace(config, itc4005_resource=None, tec_setpoint_deg_c=None, tec_settle_seconds=0)
        self.duration = duration if duration is not None else getattr(program, "duration_seconds", None)
        if self.duration is None and settings.maximum_seconds:
            self.duration = settings.maximum_seconds
        self.cancelled = threading.Event()
        self.finishing = threading.Event()
        self.capture_count = 0

    def cancel(self):
        self.cancelled.set()
        self.finishing.set()

    def finish_after_capture(self):
        self.finishing.set()

    def _dmm_snapshot(self, *, allow_finishing=False):
        deadline = time.monotonic() + self.settings.meter_wait_seconds
        while not self.cancelled.is_set():
            if self.finishing.is_set() and not allow_finishing:
                return None
            _, _, meter, error = self.state.snapshot()
            if error:
                raise RuntimeError(f"Required DMM voltage unavailable: {error}")
            if meter:
                if meter.get("unit") != "V":
                    raise ValueError("DMM logging requires a voltage function")
                if not math.isfinite(meter["value"]):
                    raise ValueError("DMM voltage is not finite")
                age = time.monotonic() - meter["monotonic"]
                if not 0 <= age <= self.settings.meter_max_age:
                    raise RuntimeError(f"DMM voltage is stale ({age:.1f} s old)")
                return {"dmm_voltage": str(meter["value"]), "dmm_voltage_unit": "V",
                        "dmm_voltage_status": "ok", "dmm_voltage_read_at": meter["timestamp"],
                        "dmm_function": meter["function"], "dmm_sample_age_seconds": str(age),
                        "dmm_serial_port": meter.get("port", ""), "dmm_identity": meter.get("identity", "")}
            if time.monotonic() >= deadline:
                raise TimeoutError("No DMM voltage received before the logging startup timeout")
            self.cancelled.wait(0.05)
        return None

    def run(self):
        try:
            while not self.state.ready.wait(0.05):
                if self.finishing.is_set():
                    return
            started, _, _, _ = self.state.snapshot()
            session = datetime.now().astimezone().isoformat(timespec="milliseconds")
            tick = 0
            while not self.finishing.is_set():
                due = started + tick * self.settings.interval_seconds
                if self.finishing.wait(max(0.0, due - time.monotonic())):
                    break
                elapsed = time.monotonic() - started
                if self.duration is not None and elapsed >= self.duration:
                    break
                # Wait for a valid required meter sample before acquiring the first waveform.
                if self.settings.include_dmm and self._dmm_snapshot() is None:
                    break
                if self.finishing.is_set():
                    break
                elapsed = time.monotonic() - started
                if self.duration is not None and elapsed >= self.duration:
                    break
                captured_at = datetime.now().astimezone()
                _, target, _, _ = self.state.snapshot()
                metadata = sine_program_metadata(self.program, target["elapsed_seconds"])
                metadata.update(logging_mode="true", logging_source="tec_sine", logging_session_started_at=session,
                                logging_capture_started_at=captured_at.isoformat(timespec="milliseconds"),
                                logging_capture_index=str(self.capture_count + 1),
                                logging_interval_seconds=str(self.settings.interval_seconds),
                                tec_sweep_setpoint_deg_c=str(target["target"]), tec_sweep_setpoint_status="verified",
                                tec_sine_setpoint_verified_at=target["verified_at"],
                                tec_sine_setpoint_age_seconds=str(time.monotonic() - target["verified_monotonic"]))
                if self.duration is not None:
                    metadata["logging_maximum_minutes"] = str(self.duration / 60)
                self.status.emit(f"Acquiring recording {self.capture_count + 1}…")
                data = capture_isolated(self.config, cancelled=self.cancelled)
                if self.cancelled.is_set():
                    break
                _, target_end, _, _ = self.state.snapshot()
                end_metadata = sine_program_metadata(self.program, target_end["elapsed_seconds"])
                metadata.update(tec_sine_setpoint_end_deg_c=str(target_end["target"]),
                                tec_dc_offset_end_deg_c=end_metadata["tec_dc_offset_deg_c"],
                                tec_dc_transition_during_capture=str(
                                    end_metadata["tec_dc_offset_deg_c"] != metadata["tec_dc_offset_deg_c"]).lower(),
                                logging_capture_completed_at=datetime.now().astimezone().isoformat(timespec="milliseconds"))
                if self.settings.include_dmm:
                    dmm = self._dmm_snapshot(allow_finishing=True)
                    if dmm is None:
                        break
                    metadata.update(dmm)
                path = next_scope_logging_path(self.settings.folder, captured_at, self.config.channel,
                                                self.capture_count + 1, self.settings.extension)
                metadata["logging_output_file"] = path.name
                data.metadata.update(metadata)
                save_scope_recording(path, data)
                self.capture_count += 1
                self.saved.emit(dict(path=path, metadata=metadata, count=self.capture_count))
                del data
                tick = max(tick + 1, math.floor((time.monotonic() - started) / self.settings.interval_seconds) + 1)
        except Exception as exc:
            if not self.cancelled.is_set():
                self.failed.emit(str(exc))
