"""Append-only TEC/DMM samples at their own acquisition times, separate from scope waveforms."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from uuid import uuid4

import numpy as np

from tec_sine import sine_program_metadata


TRACE_FORMAT = "current_vis_tec_sine_trace_v1"
TRACE_PREFIX = "tec_sine_trace_"
FIELDS = ("timestamp", "elapsed_seconds", "dmm_voltage", "tec_setpoint", "tec_temperature",
          "tec_temperature_timestamp", "dc_offset", "sine_amplitude", "sine_frequency_hz")
SAMPLE_DTYPE = np.dtype([("kind", "u1"), *((name, "f8") for name in FIELDS)])
TEC_SAMPLE, DMM_SAMPLE = 1, 2


def is_trace_path(path):
    path = Path(path)
    return path.name.startswith(TRACE_PREFIX) and path.suffix.lower() in {".h5", ".hdf5", ".csv"}


def _timestamp(value):
    return datetime.fromisoformat(value).timestamp()


class TecSineTraceRecorder:
    """One writer shared by TEC/DMM threads; every completed sample is flushed."""
    def __init__(self, folder, extension, program, *, include_dmm=True):
        self.lock = threading.Lock()
        self.session_id = uuid4().hex
        self.started = time.monotonic()
        self.program, self.include_dmm = program, include_dmm
        self.closed = False
        self.count = 0
        self._handle = None
        self._events = None
        self._dataset = None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = Path(folder) / f"{TRACE_PREFIX}{stamp}_{self.session_id[:8]}{extension}"
        try:
            if extension == ".csv":
                self._handle = self.path.open("x", newline="", encoding="utf-8")
                self._handle.write(f"# {TRACE_FORMAT}\n# session_id={self.session_id}\n")
                self._writer = csv.writer(self._handle)
                self._writer.writerow(["kind", *FIELDS])
                self._handle.flush()
            else:
                import h5py
                self._handle = h5py.File(self.path, "x", libver="latest")
                self._handle.attrs.update(format=TRACE_FORMAT, session_id=self.session_id,
                                          program=json.dumps(sine_program_metadata(program, 0)))
                self._dataset = self._handle.create_dataset("samples", (0,), maxshape=(None,),
                                                            chunks=(256,), dtype=SAMPLE_DTYPE)
                self._handle.swmr_mode = True
                self._handle.flush()
            self._events = self.path.with_suffix(".log").open("x", encoding="utf-8")
            self.log_event("run", f"Recording started; session={self.session_id}; "
                           f"program={json.dumps(sine_program_metadata(program, 0))}")
        except Exception:
            if self._handle is not None:
                self._handle.close()
            if self._events is not None:
                self._events.close()
            raise

    def log_event(self, source, message, details=""):
        """Persist faults and recovery traffic, without logging every good sample."""
        with self.lock:
            if self.closed:
                return
            stamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
            self._events.write(f"{stamp} [{source}] {message}\n")
            if details:
                self._events.write(str(details) + "\n")
            self._events.flush()

    def append(self, kind, row, target=None):
        if kind == DMM_SAMPLE and not self.include_dmm:
            return
        target = target or {}
        values = dict.fromkeys(FIELDS, np.nan)
        at = row["timestamp"] if kind == DMM_SAMPLE else row["verified_at"]
        monotonic = row["monotonic"] if kind == DMM_SAMPLE else row["verified_monotonic"]
        values.update(timestamp=_timestamp(at), elapsed_seconds=monotonic - self.started)
        if target:
            metadata = sine_program_metadata(self.program, target["elapsed_seconds"])
            values.update(tec_setpoint=target["target"], dc_offset=float(metadata["tec_dc_offset_deg_c"]),
                          sine_amplitude=float(metadata["tec_sine_amplitude_deg_c"]),
                          sine_frequency_hz=float(metadata["tec_sine_frequency_hz"]))
        if kind == DMM_SAMPLE:
            if row.get("unit") != "V" or not np.isfinite(row["value"]):
                raise ValueError("Only valid DMM voltage readings can be recorded")
            values["dmm_voltage"] = row["value"]
        elif row.get("tec_temperature_status") == "ok":
            values["tec_temperature"] = float(row["tec_temperature"])
            try:
                values["tec_temperature_timestamp"] = _timestamp(row["tec_temperature_read_at"])
            except (KeyError, ValueError, TypeError):
                values["tec_temperature_timestamp"] = values["timestamp"]
        sample = (kind, *(values[name] for name in FIELDS))
        with self.lock:
            if self.closed:
                return
            if self._dataset is not None:
                self._dataset.resize((self.count + 1,))
                self._dataset[self.count] = sample
                self._dataset.flush()
            else:
                self._writer.writerow(sample)
                self._handle.flush()
            self.count += 1

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                try:
                    self._handle.close()
                finally:
                    self._events.close()


@dataclass
class TecSineTrace:
    path: Path
    session_id: str
    samples: np.ndarray
    warnings: list[str]


def load_trace(path):
    """Read even a currently open/flushed trace; never treat it as an optical waveform."""
    path = Path(path)
    warnings = []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            if handle.readline().strip() != f"# {TRACE_FORMAT}":
                raise ValueError("Unsupported TEC/DMM trace format")
            session = handle.readline().strip()
            if not session.startswith("# session_id="):
                raise ValueError("Missing trace session identifier")
            session = session.split("=", 1)[1]
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["kind", *FIELDS]:
                raise ValueError("Invalid TEC/DMM trace columns")
            records = []
            for number, row in enumerate(reader, 4):
                try:
                    records.append((int(row["kind"]), *(float(row[name]) for name in FIELDS)))
                except (TypeError, ValueError, KeyError):
                    warnings.append(f"Skipped incomplete trace row {number}")
            samples = np.array(records, dtype=SAMPLE_DTYPE)
    else:
        import h5py
        with h5py.File(path, "r", swmr=True) as handle:
            if handle.attrs.get("format") != TRACE_FORMAT:
                raise ValueError("Unsupported TEC/DMM trace format")
            session = str(handle.attrs["session_id"])
            samples = handle["samples"][:]
            if samples.dtype != SAMPLE_DTYPE:
                raise ValueError("Invalid TEC/DMM sample layout")
    valid = np.isin(samples["kind"], (TEC_SAMPLE, DMM_SAMPLE)) & np.isfinite(samples["timestamp"])
    if np.any(~valid):
        warnings.append("Ignored incomplete or invalid trace samples")
    return TecSineTrace(path, session, samples[valid], warnings)


def trace_plot_series(traces):
    """Combine each independently timed stream without expanding samples into table rows."""
    result = {}
    for key, field, kind in (("dmm", "dmm_voltage", DMM_SAMPLE), ("tec", "tec_temperature", TEC_SAMPLE),
                              ("tec_setpoint", "tec_setpoint", TEC_SAMPLE), ("dc_offset", "dc_offset", TEC_SAMPLE)):
        times, values, offsets, sessions = [], [], [], set()
        for trace in traces:
            data = trace.samples
            selected = data["kind"] == kind
            if not np.any(selected & np.isfinite(data[field])):
                continue
            part = data[selected]
            stamp = part["timestamp"]
            if key == "tec":
                stamp = np.where(np.isfinite(part["tec_temperature_timestamp"]), part["tec_temperature_timestamp"], stamp)
            times.append(stamp)
            values.append(part[field])
            offsets.append(part["dc_offset"])
            sessions.add(trace.session_id)
        if times:
            stamp = np.concatenate(times)
            order = np.argsort(stamp, kind="stable")
            result[key] = dict(timestamp=stamp[order], values=np.concatenate(values)[order],
                               offsets=np.concatenate(offsets)[order], sessions=sessions)
    return result
