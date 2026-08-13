from __future__ import annotations

import csv
import math
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass
class ScanConfig:
    sweep_mode: str
    awg_resource: str
    scope_resource: str
    awg_vpp: float
    awg_offset_v: float
    fixed_frequency_hz: float
    total_start_hz: float
    total_stop_hz: float
    subwindow_span_hz: float
    step_size_hz: float
    amp_start_vpp: float
    amp_stop_vpp: float
    amp_step_vpp: float
    offset_start_v: float
    offset_stop_v: float
    offset_step_v: float
    rbw_hz: float
    iterations: int
    avg_count: int
    dwell_s: float
    save_csv: bool
    csv_path: str
    use_mock: bool
    timeout_ms: int = 10000
    evaluation_offset_hz: float = 0.0


@dataclass(frozen=True)
class MeasurementPoint:
    sweep_value: float
    target_freq_hz: float
    amplitude_dbm: float
    sweep_mode: str


class InstrumentClient(Protocol):
    def connect(self, config: ScanConfig) -> None: ...

    def disconnect(self) -> None: ...

    def configure_awg_output(self, config: ScanConfig) -> None: ...

    def set_awg_frequency_hz(self, frequency_hz: float) -> None: ...

    def set_awg_amplitude_vpp(self, amplitude_vpp: float) -> None: ...

    def set_awg_offset_v(self, offset_v: float) -> None: ...

    def configure_scope_window(
        self,
        center_hz: float,
        span_hz: float,
        rbw_hz: float,
        avg_count: int,
    ) -> None: ...

    def acquire_window(self, dwell_s: float) -> None: ...

    def read_amplitude_at_hz(self, frequency_hz: float) -> float: ...


def _pyvisa_module():
    try:
        import pyvisa
    except ImportError as exc:  # pragma: no cover - depends on optional hardware stack
        raise RuntimeError(
            "pyvisa is not installed. Install the project requirements and a VISA backend."
        ) from exc
    return pyvisa


def list_visa_resources() -> tuple[str, ...]:
    """Return resource names without opening the attached instruments."""
    pyvisa = _pyvisa_module()
    manager = pyvisa.ResourceManager()
    try:
        return tuple(manager.list_resources())
    finally:
        try:
            manager.close()
        except Exception:
            pass


def inspect_visa_resources(timeout_ms: int = 2000) -> list[str]:
    """Return resource names and their identity strings for the diagnostics dialog."""
    pyvisa = _pyvisa_module()
    manager = pyvisa.ResourceManager()
    lines: list[str] = []
    try:
        resources = tuple(manager.list_resources())
        if not resources:
            return ["No VISA resources found."]

        for resource in resources:
            identity = "<no *IDN? response>"
            instrument = None
            try:
                instrument = manager.open_resource(resource)
                instrument.timeout = max(100, int(timeout_ms))
                identity = instrument.query("*IDN?").strip()
            except Exception as exc:
                identity = f"<error: {exc}>"
            finally:
                if instrument is not None:
                    try:
                        instrument.close()
                    except Exception:
                        pass
            lines.append(f"{resource} | {identity}")
        return lines
    finally:
        try:
            manager.close()
        except Exception:
            pass


class TektronixVisaClient:
    """AWG and MDO3000 RF-spectrum client adapted from modulation_freq_searcher."""

    def __init__(self) -> None:
        self.rm: Any = None
        self.awg: Any = None
        self.scope: Any = None
        self.scope_idn = ""
        self._last_center_hz = 0.0
        self._last_span_hz = 0.0
        self._trace_unit = ""

    def connect(self, config: ScanConfig) -> None:
        pyvisa = _pyvisa_module()
        self.disconnect()
        try:
            self.rm = pyvisa.ResourceManager()
            self.awg = self.rm.open_resource(config.awg_resource)
            self.scope = self.rm.open_resource(config.scope_resource)
            timeout_ms = max(1000, int(config.timeout_ms))
            self.awg.timeout = timeout_ms
            self.scope.timeout = timeout_ms
            try:
                self.scope_idn = self.scope.query("*IDN?").strip()
            except Exception:
                self.scope_idn = config.scope_resource

            # The MDO3024 exposes its RF trace through RF_NORMal/CURVE?.
            self.scope.write("HEADer 0")
            self.scope.write("VERBose 0")
            self.scope.write("SELect:RF_NORMal 1")
            self.scope.write("DATa:SOUrce RF_NORMal")
            self.scope.write("DATa:ENCdg ASCii")
            self.scope.write("DATa:WIDth 1")
            self.scope.write("DATa:STARt 1")
            self.scope.write("DATa:STOP 10000")
            self.scope.write("WFMOutpre:ENCdg ASCii")
            self.scope.write("WFMOutpre:BYT_Nr 4")
            self._trace_unit = self._detect_trace_unit()
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        for instrument in (self.awg, self.scope):
            if instrument is not None:
                try:
                    instrument.close()
                except Exception:
                    pass
        self.awg = None
        self.scope = None
        if self.rm is not None:
            try:
                self.rm.close()
            except Exception:
                pass
        self.rm = None

    def _require_awg(self):
        if self.awg is None:
            raise RuntimeError("AWG is not connected.")
        return self.awg

    def _require_scope(self):
        if self.scope is None:
            raise RuntimeError("Oscilloscope is not connected.")
        return self.scope

    def configure_awg_output(self, config: ScanConfig) -> None:
        awg = self._require_awg()
        awg.write("SOURce1:FUNCtion:SHAPe SINusoid")
        self.set_awg_offset_v(config.awg_offset_v)
        self.set_awg_amplitude_vpp(config.awg_vpp)
        awg.write("OUTPut1:STATe ON")

    def set_awg_frequency_hz(self, frequency_hz: float) -> None:
        self._require_awg().write(f"SOURce1:FREQuency:FIXed {frequency_hz}")

    def set_awg_amplitude_vpp(self, amplitude_vpp: float) -> None:
        self._require_awg().write(
            f"SOURce1:VOLTage:LEVel:IMMediate:AMPLitude {amplitude_vpp} VPP"
        )

    def set_awg_offset_v(self, offset_v: float) -> None:
        self._require_awg().write(f"SOURce1:VOLTage:LEVel:IMMediate:OFFSet {offset_v}")

    def configure_scope_window(
        self,
        center_hz: float,
        span_hz: float,
        rbw_hz: float,
        avg_count: int,
    ) -> None:
        scope = self._require_scope()
        self._last_center_hz = center_hz
        self._last_span_hz = span_hz
        scope.write("ACQUIRE:STATE OFF")
        scope.write("ACQUIRE:STOPAFTER SEQUENCE")
        scope.write(f"ACQuire:NUMAVG {max(1, int(avg_count))}")
        scope.write(f"RF:FREQuency {center_hz}")
        scope.write(f"RF:SPAN {span_hz}")
        scope.write(f"RF:RBW {rbw_hz}")
        scope.write("RF:MARKER1:STATE ON")

    def acquire_window(self, dwell_s: float) -> None:
        scope = self._require_scope()
        scope.write("ACQUIRE:STATE RUN")
        try:
            scope.query("*OPC?")
        except Exception:
            time.sleep(max(0.05, dwell_s))

    def read_amplitude_at_hz(self, frequency_hz: float) -> float:
        # Marker queries time out on the MDO3024 firmware used by the source project.
        return self._read_amplitude_from_trace(frequency_hz)

    def _read_amplitude_from_trace(self, frequency_hz: float) -> float:
        values = self._read_curve_values()
        if not values:
            identity = self.scope_idn or "scope"
            raise RuntimeError(
                f"[{identity}] CURVE? returned no numeric RF trace data. "
                "Check that RF_NORMal is active."
            )

        center_hz = self._last_center_hz or frequency_hz
        span_hz = self._last_span_hz or max(1.0, frequency_hz * 0.01)
        start_hz = center_hz - 0.5 * span_hz
        stop_hz = center_hz + 0.5 * span_hz
        if len(values) == 1:
            return self._value_to_dbm(values[0])

        if frequency_hz <= start_hz:
            value = values[0]
        elif frequency_hz >= stop_hz:
            value = values[-1]
        else:
            fraction = (frequency_hz - start_hz) / (stop_hz - start_hz)
            index = int(round(fraction * (len(values) - 1)))
            index = max(0, min(len(values) - 1, index))
            value = values[index]
        return self._value_to_dbm(value)

    def _detect_trace_unit(self) -> str:
        scope = self._require_scope()
        commands = (
            "WFMOutpre:YUNit?",
            "WFMOutpre:YUN?",
            "RF:UNIts?",
            "RF:VERTical:UNIts?",
        )
        for command in commands:
            try:
                unit = scope.query(command).strip()
                if unit:
                    return unit.strip('"').upper()
            except Exception:
                continue
        return ""

    def _value_to_dbm(self, value: float) -> float:
        unit = self._trace_unit.upper()
        if "DBM" in unit:
            return value
        if "MW" in unit:
            return 10.0 * math.log10(value) if value > 0 else float("-inf")
        if "W" in unit:
            return 10.0 * math.log10(value / 1e-3) if value > 0 else float("-inf")

        # The source MDO3024 returns small positive RF values as watts.
        if 0.0 < value < 1.0:
            return 10.0 * math.log10(value / 1e-3)
        return value

    def _read_curve_values(self) -> list[float]:
        scope = self._require_scope()
        errors: list[str] = []
        try:
            values = self._parse_curve_text(scope.query("CURVE?"))
            if values:
                return values
        except Exception as exc:
            errors.append(f"CURVE? direct: {exc}")

        try:
            data = scope.query_ascii_values("CURVE?", separator=",")
            values = [float(value) for value in data]
            if values:
                return values
        except Exception as exc:
            errors.append(f"query_ascii_values: {exc}")

        identity = self.scope_idn or "scope"
        raise RuntimeError(f"[{identity}] Could not read RF trace: {' | '.join(errors)}")

    @staticmethod
    def _parse_curve_text(raw: str) -> list[float]:
        text = str(raw).strip()
        if not text:
            return []

        if text.startswith("#") and len(text) > 2 and text[1].isdigit():
            digit_count = int(text[1])
            header_end = 2 + digit_count
            if len(text) >= header_end:
                try:
                    payload_length = int(text[2:header_end])
                except ValueError:
                    payload_length = 0
                text = text[header_end : header_end + payload_length]

        values: list[float] = []
        for token in re.split(r"[,\s]+", text):
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                continue
        return values

    def debug_read_scope_data(self) -> dict[str, str]:
        result: dict[str, str] = {}
        commands = (
            "*IDN?",
            "*OPC?",
            "SELect:RF_NORMal?",
            "DATa:SOUrce?",
            "DATa:ENCdg?",
            "DATa:WIDth?",
            "WFMOutpre:YUNit?",
            "RF:FREQuency?",
            "RF:SPAN?",
            "RF:RBW?",
            "RF:MARKER1:STATE?",
            "RF:MARKER1:X?",
            "RF:MARKER1:Y?",
            "RF:MARKER1:AMPLITUDE?",
            "RF:MARKER1:MAGNITUDE?",
            "RF:MARKER1:VALUE?",
        )
        for command in commands:
            result[command] = self._safe_query(command)
        curve = self._safe_query("CURVE?", max_len=500)
        result["CURVE? (first 500 chars)"] = curve
        result["CURVE? (captured length)"] = f"{len(curve)} chars"
        return result

    def _safe_query(self, command: str, max_len: int = 100) -> str:
        try:
            response = str(self._require_scope().query(command)).strip()
            if len(response) > max_len:
                return f"{response[:max_len]}... [truncated]"
            return repr(response)
        except Exception as exc:
            return f"<error: {exc}>"


class MockClient:
    """Synthetic resonance used for UI development and automated smoke tests."""

    def __init__(self) -> None:
        self.current_awg_hz = 0.0
        self.current_awg_vpp = 0.0
        self.current_awg_offset = 0.0
        self.current_center_hz = 0.0
        self.current_span_hz = 0.0
        self.current_rbw_hz = 0.0

    def connect(self, config: ScanConfig) -> None:
        self.current_awg_hz = 0.0

    def disconnect(self) -> None:
        return

    def configure_awg_output(self, config: ScanConfig) -> None:
        self.current_awg_vpp = config.awg_vpp
        self.current_awg_offset = config.awg_offset_v

    def set_awg_frequency_hz(self, frequency_hz: float) -> None:
        self.current_awg_hz = frequency_hz

    def set_awg_amplitude_vpp(self, amplitude_vpp: float) -> None:
        self.current_awg_vpp = amplitude_vpp

    def set_awg_offset_v(self, offset_v: float) -> None:
        self.current_awg_offset = offset_v

    def configure_scope_window(
        self,
        center_hz: float,
        span_hz: float,
        rbw_hz: float,
        avg_count: int,
    ) -> None:
        self.current_center_hz = center_hz
        self.current_span_hz = span_hz
        self.current_rbw_hz = rbw_hz

    def acquire_window(self, dwell_s: float) -> None:
        time.sleep(min(max(dwell_s, 0.0), 0.1))

    def read_amplitude_at_hz(self, frequency_hz: float) -> float:
        resonance_hz = 2.05e6
        sigma_hz = 80e3
        peak = -65.0 + 30.0 * math.exp(
            -0.5 * ((frequency_hz - resonance_hz) / sigma_hz) ** 2
        )
        coupling = 8.0 * math.exp(
            -0.5 * ((frequency_hz - self.current_awg_hz) / 20e3) ** 2
        )
        amplitude_effect = 5.0 * math.log10(max(self.current_awg_vpp, 1e-6))
        offset_effect = -1.5 * abs(self.current_awg_offset)
        rbw_penalty = -2.5 * math.log10(max(self.current_rbw_hz, 1.0) / 1e3)
        return peak + coupling + amplitude_effect + offset_effect + rbw_penalty + random.uniform(-0.7, 0.7)


class SpectrumScanner:
    def __init__(self, client: InstrumentClient) -> None:
        self.client = client
        self._stop_event = threading.Event()

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    @staticmethod
    def _build_sweep_values(start: float, stop: float, step: float, name: str) -> list[float]:
        if not all(math.isfinite(value) for value in (start, stop, step)):
            raise ValueError(f"{name} values must be finite.")
        if step <= 0:
            raise ValueError(f"{name} step must be greater than zero.")

        direction = 1.0 if stop >= start else -1.0
        signed_step = direction * step
        current = start
        values: list[float] = []
        while (current <= stop + 1e-12) if direction > 0 else (current >= stop - 1e-12):
            values.append(current)
            if len(values) > 1_000_000:
                raise ValueError(f"{name} sweep contains more than 1,000,000 points.")
            current += signed_step
        return values

    @staticmethod
    def _validate_config(config: ScanConfig) -> None:
        if not config.awg_resource.strip() and not config.use_mock:
            raise ValueError("Select an AWG VISA resource.")
        if not config.scope_resource.strip() and not config.use_mock:
            raise ValueError("Select an oscilloscope VISA resource.")
        if not math.isfinite(config.awg_vpp) or config.awg_vpp < 0:
            raise ValueError("AWG amplitude must be a finite, non-negative value.")
        if not math.isfinite(config.awg_offset_v):
            raise ValueError("AWG offset must be finite.")
        if not math.isfinite(config.evaluation_offset_hz):
            raise ValueError("Evaluation frequency offset must be finite.")
        if not math.isfinite(config.subwindow_span_hz) or config.subwindow_span_hz <= 0:
            raise ValueError("Subwindow span must be greater than zero.")
        if not math.isfinite(config.rbw_hz) or config.rbw_hz <= 0:
            raise ValueError("RBW must be greater than zero.")
        if config.iterations <= 0:
            raise ValueError("Repeats must be at least one.")
        if config.avg_count <= 0:
            raise ValueError("Scope averages must be at least one.")
        if not math.isfinite(config.dwell_s) or config.dwell_s < 0:
            raise ValueError("Dwell time cannot be negative.")
        if config.timeout_ms < 100:
            raise ValueError("VISA timeout must be at least 100 ms.")
        if config.save_csv and not config.csv_path.strip():
            raise ValueError("Choose a CSV output path or disable automatic CSV saving.")
        if config.sweep_mode == "frequency" and (
            config.total_start_hz <= 0 or config.total_stop_hz <= 0
        ):
            raise ValueError("Start and stop frequencies must be greater than zero.")
        if config.sweep_mode in {"amplitude", "offset"} and (
            not math.isfinite(config.fixed_frequency_hz) or config.fixed_frequency_hz <= 0
        ):
            raise ValueError("Fixed frequency must be greater than zero.")

    def run(
        self,
        config: ScanConfig,
        on_point: Callable[[MeasurementPoint, int, int], None] | None = None,
    ) -> list[MeasurementPoint]:
        self._validate_config(config)
        self._stop_event.clear()
        mode = config.sweep_mode
        if mode == "frequency":
            sweep_values = self._build_sweep_values(
                config.total_start_hz,
                config.total_stop_hz,
                config.step_size_hz,
                "Frequency",
            )
        elif mode == "amplitude":
            sweep_values = self._build_sweep_values(
                config.amp_start_vpp,
                config.amp_stop_vpp,
                config.amp_step_vpp,
                "Amplitude",
            )
        elif mode == "offset":
            sweep_values = self._build_sweep_values(
                config.offset_start_v,
                config.offset_stop_v,
                config.offset_step_v,
                "Offset",
            )
        else:
            raise ValueError(f"Unknown sweep mode: {mode}")
        if not sweep_values:
            raise ValueError("The selected range produces no sweep points.")

        lowest_awg_frequency_hz = (
            min(sweep_values[0], sweep_values[-1])
            if mode == "frequency"
            else config.fixed_frequency_hz
        )
        if lowest_awg_frequency_hz + config.evaluation_offset_hz <= 0:
            raise ValueError(
                "AWG frequency plus evaluation offset must remain greater than zero."
            )

        results: list[MeasurementPoint] = []
        try:
            self.client.connect(config)
            self.client.configure_awg_output(config)
            total = len(sweep_values)
            for index, sweep_value in enumerate(sweep_values, start=1):
                if self.stop_requested:
                    break

                if mode == "frequency":
                    awg_frequency_hz = sweep_value
                    self.client.set_awg_frequency_hz(awg_frequency_hz)
                    self.client.set_awg_amplitude_vpp(config.awg_vpp)
                    self.client.set_awg_offset_v(config.awg_offset_v)
                elif mode == "amplitude":
                    awg_frequency_hz = config.fixed_frequency_hz
                    self.client.set_awg_frequency_hz(awg_frequency_hz)
                    self.client.set_awg_offset_v(config.awg_offset_v)
                    self.client.set_awg_amplitude_vpp(sweep_value)
                else:
                    awg_frequency_hz = config.fixed_frequency_hz
                    self.client.set_awg_frequency_hz(awg_frequency_hz)
                    self.client.set_awg_amplitude_vpp(config.awg_vpp)
                    self.client.set_awg_offset_v(sweep_value)

                target_hz = awg_frequency_hz + config.evaluation_offset_hz
                self.client.configure_scope_window(
                    target_hz,
                    config.subwindow_span_hz,
                    config.rbw_hz,
                    config.avg_count,
                )
                samples: list[float] = []
                for _ in range(config.iterations):
                    if self.stop_requested:
                        break
                    self.client.acquire_window(config.dwell_s)
                    samples.append(self.client.read_amplitude_at_hz(target_hz))
                if self.stop_requested and not samples:
                    break

                finite_samples = [sample for sample in samples if math.isfinite(sample)]
                amplitude_dbm = (
                    sum(finite_samples) / len(finite_samples)
                    if finite_samples
                    else float("-inf")
                )
                point = MeasurementPoint(
                    sweep_value=sweep_value,
                    target_freq_hz=target_hz,
                    amplitude_dbm=amplitude_dbm,
                    sweep_mode=mode,
                )
                results.append(point)
                if on_point is not None:
                    on_point(point, index, total)
        finally:
            self.client.disconnect()

        if config.save_csv and results:
            self.write_csv(config.csv_path, results)
        return results

    @staticmethod
    def write_csv(path: str | Path, points: list[MeasurementPoint]) -> None:
        path_text = str(path).strip()
        if not path_text:
            raise ValueError("Choose a CSV output path.")
        output_path = Path(path_text)
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sweep_mode", "sweep_value", "target_freq_hz", "amplitude_dbm"])
            for point in points:
                writer.writerow(
                    [
                        point.sweep_mode,
                        point.sweep_value,
                        point.target_freq_hz,
                        point.amplitude_dbm,
                    ]
                )
