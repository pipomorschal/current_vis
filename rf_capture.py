"""RF spectra and cancellable scope-only acquisitions using the sweep's RF client."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

import numpy as np

from frequency_sweep import TektronixVisaClient


RF_FORMAT = "current_vis_rf_spectrum_v1"


def is_rf_path(path):
    path = Path(path)
    return path.name.lower().startswith("rf_") and path.suffix.lower() in {".h5", ".hdf5", ".csv"}


class RfCaptureCancelled(Exception):
    pass


@dataclass(frozen=True)
class RfCaptureConfig:
    resource: str
    timeout_ms: int = 10000
    acquisition_wait_seconds: float = 30
    use_scope_settings: bool = True
    center_hz: float = 100e6
    span_hz: float = 10e6
    rbw_hz: float = 100e3

    def __post_init__(self):
        if not self.resource.strip():
            raise ValueError("Select an RF oscilloscope VISA resource")
        if not 1000 <= self.timeout_ms <= 120000:
            raise ValueError("VISA timeout must be between 1000 and 120000 ms")
        if not math.isfinite(self.acquisition_wait_seconds) or self.acquisition_wait_seconds <= 0:
            raise ValueError("Acquisition wait must be positive")
        if not self.use_scope_settings:
            if (not all(math.isfinite(v) and v > 0 for v in (self.center_hz, self.span_hz, self.rbw_hz))
                    or self.center_hz < self.span_hz / 2):
                raise ValueError("RF center/span/RBW must be positive; the span must not extend below 0 Hz")


@dataclass
class RfSpectrum:
    frequency_hz: np.ndarray
    power_dbm: np.ndarray
    raw_values: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.frequency_hz, self.power_dbm, self.raw_values = (
            np.asarray(a, dtype=float) for a in (self.frequency_hz, self.power_dbm, self.raw_values))
        if (any(a.ndim != 1 for a in (self.frequency_hz, self.power_dbm, self.raw_values))
                or not len(self.frequency_hz)
                or not self.frequency_hz.shape == self.power_dbm.shape == self.raw_values.shape
                or not np.all(np.isfinite(self.frequency_hz))
                or (len(self.frequency_hz) > 1 and not np.all(np.diff(self.frequency_hz) > 0))
                or not np.all(np.isfinite(self.raw_values))
                or np.any(np.isnan(self.power_dbm)) or np.any(np.isposinf(self.power_dbm))):
            raise ValueError("Invalid RF spectrum: expected matching frequency and power arrays")


def acquire_rf(config):
    """A fresh single sequence; use calibrated RF preamble coordinates in Hz."""
    client = TektronixVisaClient()
    try:
        client.connect_scope(config.resource, config.timeout_ms)
        scope = client.scope
        scope.write("ACQUIRE:STATE OFF")
        scope.write("ACQUIRE:STOPAFTER SEQUENCE")
        if not config.use_scope_settings:
            scope.write(f"RF:FREQuency {config.center_hz:.12g}")
            scope.write(f"RF:SPAN {config.span_hz:.12g}")
            scope.write(f"RF:RBW {config.rbw_hz:.12g}")
        started = datetime.now().astimezone().isoformat(timespec="milliseconds")
        scope.write("ACQUIRE:STATE RUN")
        deadline = time.monotonic() + config.acquisition_wait_seconds
        while True:
            state = scope.query("ACQUIRE:STATE?").strip().upper()
            if state in {"0", "OFF", "STOP"}:
                break
            if state not in {"1", "ON", "RUN"}:
                raise RuntimeError(f"Unsupported RF acquisition state: {state!r}")
            if time.monotonic() >= deadline:
                raise TimeoutError("RF acquisition did not finish; check the scope trigger and acquisition settings")
            time.sleep(.05)
        # Query these AFTER stopping, so settings and trace belong to the same acquisition.
        unit_x = scope.query("WFMOutpre:XUNit?").strip().strip('"').upper()
        if unit_x != "HZ":
            raise RuntimeError(f"Expected an RF frequency trace in Hz; received {unit_x!r}")
        count = int(scope.query("WFMOutpre:NR_Pt?").strip())
        if not 1 <= count <= 2_000_000:
            raise RuntimeError(f"Invalid RF trace length: {count}")
        xzero = float(scope.query("WFMOutpre:XZEro?"))
        xincr = float(scope.query("WFMOutpre:XINcr?"))
        if not math.isfinite(xzero) or not math.isfinite(xincr) or xincr <= 0:
            raise RuntimeError("Invalid RF frequency preamble")
        scope.write("DATa:STARt 1")
        scope.write(f"DATa:STOP {count}")
        raw = np.asarray(client._read_curve_values(), dtype=float)
        if len(raw) != count:
            raise RuntimeError(f"Incomplete RF spectrum: received {len(raw)} of {count} points")
        unit = client._detect_trace_unit()
        if unit not in {"W", "WATT", "WATTS", "MW", "DBM"}:
            raise RuntimeError(f"Unsupported RF power unit {unit!r}; select power units W, mW or dBm")
        client._trace_unit = unit
        if unit != "DBM" and np.any(raw < 0):
            raise RuntimeError("RF trace contains negative linear power values")
        power = np.array([client._value_to_dbm(float(v)) for v in raw])
        metadata = dict(format=RF_FORMAT, instrument=client.scope_idn, resource=config.resource,
                        source="RF_NORMal", raw_unit=unit, power_unit="dBm", frequency_unit="Hz",
                        capture_started_at=started,
                        capture_completed_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
                        xzero_hz=xzero, xincr_hz=xincr, point_count=count,
                        center_hz=float(scope.query("RF:FREQuency?")),
                        span_hz=float(scope.query("RF:SPAN?")), rbw_hz=float(scope.query("RF:RBW?")))
        return RfSpectrum(xzero + np.arange(count) * xincr, power, raw, metadata)
    finally:
        client.disconnect()


def save_rf_spectrum(path, spectrum):
    """Publish complete data only. RF axes are explicitly frequency/power, never time/voltage."""
    path = Path(path)
    staging = path.with_name(path.name + f".{uuid4().hex}.partial")
    try:
        if path.suffix.lower() in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(staging, "x") as handle:
                handle.attrs["format"] = RF_FORMAT
                for name, values in (("frequency_hz", spectrum.frequency_hz), ("power_dbm", spectrum.power_dbm),
                                     ("raw_values", spectrum.raw_values)):
                    handle.create_dataset(name, data=values)
                handle.attrs["metadata"] = json.dumps(spectrum.metadata, allow_nan=False)
        elif path.suffix.lower() == ".csv":
            with staging.open("x", encoding="utf-8", newline="") as handle:
                handle.write(f"# {RF_FORMAT}\n# metadata={json.dumps(spectrum.metadata, allow_nan=False)}\n")
                writer = csv.writer(handle)
                writer.writerow(["frequency_hz", "power_dbm", "raw_value"])
                writer.writerows(zip(spectrum.frequency_hz, spectrum.power_dbm, spectrum.raw_values))
        else:
            raise ValueError("Choose HDF5 or CSV for RF spectrum data")
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def load_rf_spectrum(path):
    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("format") != RF_FORMAT:
                raise ValueError("This is not an RF Capture spectrum")
            return RfSpectrum(handle["frequency_hz"][:], handle["power_dbm"][:], handle["raw_values"][:],
                              json.loads(handle.attrs["metadata"]))
    with path.open(encoding="utf-8", newline="") as handle:
        if handle.readline().strip() != f"# {RF_FORMAT}":
            raise ValueError("This is not an RF Capture spectrum")
        metadata = json.loads(handle.readline().removeprefix("# metadata="))
        reader = csv.reader(handle)
        if next(reader) != ["frequency_hz", "power_dbm", "raw_value"]:
            raise ValueError("Invalid RF spectrum columns")
        data = np.asarray([[float(v) for v in row] for row in reader], dtype=float)
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError("Invalid RF spectrum data")
        return RfSpectrum(data[:, 0], data[:, 1], data[:, 2], metadata)


def capture_rf_isolated(config, cancelled=None):
    if cancelled is not None and cancelled.is_set():
        raise RfCaptureCancelled()
    with TemporaryDirectory(prefix="rf_capture_") as folder:
        root = Path(folder)
        (root / "request.json").write_text(json.dumps(asdict(config)), encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        with (root / "process.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(root)],
                                       stdout=log, stderr=log, creationflags=flags)
            deadline = time.monotonic() + config.acquisition_wait_seconds + 12 * config.timeout_ms / 1000 + 10
            try:
                while process.poll() is None:
                    if cancelled is not None and cancelled.is_set():
                        raise RfCaptureCancelled()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("RF capture exceeded its deadline; isolated VISA process stopped")
                    try:
                        process.wait(timeout=.1)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
        if cancelled is not None and cancelled.is_set():
            raise RfCaptureCancelled()
        if process.returncode:
            error = root / "error.json"
            if error.exists():
                raise RuntimeError(json.loads(error.read_text(encoding="utf-8"))["error"])
            raise RuntimeError((root / "process.log").read_text(encoding="utf-8", errors="replace")[-2000:]
                               or f"RF capture process exited with code {process.returncode}")
        return load_rf_spectrum(root / "result.h5")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    try:
        result = acquire_rf(RfCaptureConfig(**json.loads((root / "request.json").read_text(encoding="utf-8"))))
        save_rf_spectrum(root / "result.h5", result)
    except Exception as exc:
        (root / "error.json").write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        raise SystemExit(1)
