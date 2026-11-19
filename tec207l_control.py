"""Background serial monitoring and independent TEC207L channel programs."""
from __future__ import annotations

import csv
import json
from contextlib import nullcontext
from datetime import datetime
import math
from pathlib import Path
import queue
import threading
import time

from PySide6 import QtCore

from tec207l.device import Tec207L
from tec207l.autopid import run_autopid
from tec207l.hwautotune import run_hw_autotune
from tec207l.steptest import run_step_test
from tec_sine import TecSineSettings, TecSineSweep
from tec_temperature_sweep import TecTemperatureSweep


LOG_COLUMNS = ["Time", *[f"CH{ch}_{metric}" for ch in (1, 2)
                         for metric in ("temperature_C", "setpoint_C", "current_A", "resistance_ohm", "output_enabled")]]


def sample_row(channel, sample):
    values = [sample["timestamp"]] + [""] * 10
    offset = 1 + (channel - 1) * 5
    values[offset:offset + 5] = [sample[key] for key in ("actual", "target", "current", "resistance", "enable")]
    values[offset + 4] = int(sample["enable"])
    return values


def write_history(path, records):
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(LOG_COLUMNS)
        for channel, sample in sorted(records, key=lambda item: item[1]["monotonic"]):
            writer.writerow(sample_row(channel, sample))


def make_program(settings):
    mode = settings["mode"]
    if mode in {"range", "sine_sweep"}:
        offsets = TecTemperatureSweep(settings["minimum"], settings["maximum"], settings["increment"],
                                      settings["dwell_s"], settings["repeats"])
        if mode == "range":
            return offsets, offsets.duration_seconds, min(1., offsets.step_seconds)
    sine = TecSineSettings(settings["center"], settings["amplitude"], settings["frequency_hz"], settings["update_s"])
    if mode == "sine_sweep":
        program = TecSineSweep(sine, offsets)
        return program, program.duration_seconds, min(program.interval_seconds, offsets.step_seconds)
    if mode != "sine":
        raise ValueError("Unknown TEC207L temperature program")
    cycles = settings["cycles"]
    if type(cycles) is not int or cycles < 1:
        raise ValueError("Sine cycles must be a positive integer")
    return sine, sine.period_seconds * cycles, sine.interval_seconds


def apply_output_limit(device, channel, percent):
    """Use readback to confirm the limit before any output-enable operation."""
    requested = Tec207L.validate_max_output(percent)
    try:
        device.set_max_output(channel, requested)
        actual = device.get_max_output(channel)
        if actual != requested:
            raise RuntimeError(f"CH{channel} output limit was not confirmed: requested {requested}% Vin, read {actual!r}")
        return actual
    except Exception as exc:
        # A running TEC must not continue under an unconfirmed ceiling.
        # This is scoped to the selected channel; the other TEC keeps running.
        try:
            device.set_enable(channel, False)
            if device.get_enable(channel):
                raise RuntimeError("Output OFF was not confirmed")
        except Exception as off_error:
            raise RuntimeError(f"{exc}; Output OFF: {off_error}") from exc
        raise


class Tec207LConnection(QtCore.QThread):
    ready = QtCore.Signal(object)
    sample = QtCore.Signal(int, object)
    command_done = QtCore.Signal(int, str, object)
    warning = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    transaction = QtCore.Signal(str)
    logging_changed = QtCore.Signal(str)

    def __init__(self, port, baudrate=38400, poll_seconds=.3, parent=None, device_factory=None):
        super().__init__(parent)
        self.port, self.baudrate, self.poll_seconds = port, baudrate, poll_seconds
        self._device_factory = device_factory or Tec207L
        self.device = None
        self.stop_event = threading.Event()
        self._commands = queue.Queue()
        self._state_lock = threading.Lock()
        self._busy = set()
        self._pending = {1: 0, 2: 0}
        self._log_handle = None
        self._log_writer = None

    def reserve_channel(self, channel):
        with self._state_lock:
            if channel in self._busy or self._pending[channel]:
                raise ValueError(f"CH{channel} is busy; wait for the current operation")
            self._busy.add(channel)

    def release_channel(self, channel):
        with self._state_lock:
            self._busy.discard(channel)

    def submit(self, channel, action, args=()):
        if self.stop_event.is_set() or self.device is None:
            raise ValueError("TEC207L is not connected")
        with self._state_lock:
            if channel and channel in self._busy:
                raise ValueError(f"CH{channel} is running a program")
            if channel:
                self._pending[channel] += 1
        self._commands.put((channel, action, args))

    def _close_log(self):
        if self._log_handle is not None:
            self._log_handle.close()
        self._log_handle = self._log_writer = None
        self.logging_changed.emit("")

    def _command(self, channel, action, args):
        dev = self.device
        if action == "read":
            return dev.get_full_state(channel)
        if action == "read_pid":
            return dev.get_pid(channel)
        if action == "read_sensor":
            return {"model": dev.get_sensor_model(channel), "ntc": dev.get_ntc(channel), "pt": dev.get_pt(channel)}
        if action == "target":
            return dev.set_target_temp(channel, args[0])
        if action == "limit":
            return apply_output_limit(dev, channel, args[0])
        if action == "output":
            enabled, mode, limit = args
            Tec207L.validate_max_output(limit)
            # Configure limits before enabling a channel.
            if not enabled:
                dev.set_enable(channel, False)
            apply_output_limit(dev, channel, limit)
            dev.set_output_mode(channel, mode)
            if enabled:
                dev.set_enable(channel, True)
            actual = dev.get_enable(channel)
            if actual != enabled:
                raise RuntimeError(f"CH{channel} output {'ON' if enabled else 'OFF'} was not confirmed")
            return {"enable": actual, "limit": limit}
        if action == "pid":
            return dev.set_pid(channel, *args)
        if action == "sensor":
            model, ntc, pt = args
            dev.set_sensor_model(channel, model)
            dev.set_ntc(channel, *ntc)
            dev.set_pt(channel, *pt)
            return None
        if action == "all_off":
            errors = []
            for ch in (1, 2):
                try:
                    dev.set_enable(ch, False)
                    if dev.get_enable(ch):
                        raise RuntimeError(f"CH{ch} output is still enabled")
                except Exception as exc:
                    errors.append(f"CH{ch}: {exc}")
            if errors:
                raise RuntimeError("; ".join(errors))
            return None
        if action == "log_start":
            if self._log_handle:
                raise ValueError("Logging is already running")
            self._log_handle = Path(args[0]).open("x", newline="", encoding="utf-8")
            self._log_writer = csv.writer(self._log_handle)
            self._log_writer.writerow(LOG_COLUMNS)
            self._log_handle.flush()
            self.logging_changed.emit(str(args[0]))
            return None
        if action == "log_stop":
            self._close_log()
            return None
        if action == "export":
            write_history(*args)
            return str(args[0])
        raise ValueError(f"Unknown TEC207L operation: {action}")

    def run(self):
        try:
            self.device = self._device_factory(self.port, self.baudrate)
            self.device.on_transaction = lambda direction, data: self.transaction.emit(f"{direction} {data!r}")
            self.device.connect()
            model, firmware = self.device.get_model(), self.device.get_firmware()
            if model != "TEC207L":
                raise ValueError(f"Expected TEC207L, received {model}")
            # Connecting only reads the hardware; GUI/config defaults are never written.
            states = {ch: self.device.get_full_state(ch) for ch in (1, 2)}
            self.ready.emit({"model": model, "firmware": firmware, "states": states})
            deadline, failures = 0., 0
            while not self.stop_event.is_set():
                try:
                    channel, action, args = self._commands.get(timeout=max(0., min(.05, deadline - time.monotonic())))
                except queue.Empty:
                    pass
                else:
                    try:
                        result = self._command(channel, action, args)
                        self.command_done.emit(channel or 0, action, result)
                    except Exception as exc:
                        self.warning.emit(f"{action} CH{channel or '1/2'}: {exc}")
                        self.command_done.emit(channel or 0, "error", {"action": action, "error": str(exc)})
                    finally:
                        if channel:
                            with self._state_lock:
                                self._pending[channel] -= 1
                if time.monotonic() < deadline:
                    continue
                try:
                    for ch in (1, 2):
                        sample = {"actual": self.device.get_actual_temp(ch), "target": self.device.get_target_temp(ch),
                                  "current": self.device.get_output_current(ch), "resistance": self.device.get_sensor_resistance(ch),
                                  "enable": self.device.get_enable(ch), "limit": self.device.get_max_output(ch),
                                  "monotonic": time.monotonic(),
                                  "timestamp": datetime.now().astimezone().isoformat(timespec="microseconds")}
                        if not all(math.isfinite(sample[key]) for key in ("actual", "target", "current", "resistance")):
                            raise ValueError(f"CH{ch} returned non-finite data")
                        if self._log_writer:
                            self._log_writer.writerow(sample_row(ch, sample))
                        self.sample.emit(ch, sample)
                    if self._log_handle:
                        self._log_handle.flush()
                    failures = 0
                except Exception as exc:
                    failures += 1
                    self.warning.emit(f"TEC207L read failed ({failures}/3): {exc}")
                    if failures >= 3:
                        raise
                deadline = time.monotonic() + self.poll_seconds
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.stop_event.set()
            # The GUI cancels jobs when failed is emitted. Keep the port open
            # until their finally blocks restore the setpoint / turn output off.
            while True:
                with self._state_lock:
                    busy = bool(self._busy)
                if not busy:
                    break
                time.sleep(.02)
            try:
                self._close_log()
            except Exception as exc:
                self.failed.emit(f"Closing log: {exc}")
            finally:
                if self.device is not None:
                    try:
                        self.device.disconnect()
                    except Exception as exc:
                        self.failed.emit(f"Disconnecting: {exc}")


class Tec207LJob(QtCore.QThread):
    progress = QtCore.Signal(int, str)
    completed = QtCore.Signal(int, object)
    failed = QtCore.Signal(int, str)

    def __init__(self, device, channel, kind, settings, parent=None):
        super().__init__(parent)
        self.device, self.channel, self.kind, self.settings = device, channel, kind, dict(settings)
        self.stop_event = threading.Event()
        self.force_off = threading.Event()

    def _zn_tune(self, callbacks, limit):
        path = Path(self.settings["result_path"]) if self.settings.get("result_path") else None
        report = dict(channel=self.channel, settings=self.settings, trials=[], outcome="running", events=[])
        keys = ["elapsed_s", "gain_p", "phase", "target_C", "temperature_C", "limit_percent", "current_A", "output_percent"]
        try:
            with path.open("x", newline="", encoding="utf-8") if path else nullcontext(None) as handle:
                writer = csv.writer(handle) if handle else None
                if writer:
                    writer.writerow(["Time", "channel", *keys])
                    handle.flush()
                def sample(row):
                    if writer:
                        writer.writerow([datetime.now().astimezone().isoformat(timespec="microseconds"),
                                         self.channel, *[row[key] for key in keys]])
                        handle.flush()
                def progress(message):
                    report["events"].append(dict(time=datetime.now().astimezone().isoformat(), message=message))
                    callbacks["on_step"](message)
                result = run_autopid(self.device, self.channel, on_step=progress, should_stop=callbacks["should_stop"],
                                     output_limit=limit, on_sample=sample, on_trial=report["trials"].append,
                                     **self.settings.get("zn_settings", {}))
                report.update(outcome="cancelled" if result.get("aborted") else "completed", result=result)
                if path:
                    result["result_path"] = str(path)
                return result
        except Exception as exc:
            report.update(outcome="error", error=str(exc))
            raise RuntimeError(f"{exc}" + (f"; tuning trace: {path}" if path else "")) from exc
        finally:
            if path:
                try:
                    with path.with_suffix(".json").open("x", encoding="utf-8") as handle:
                        json.dump(report, handle, indent=2, allow_nan=False, default=str)
                except Exception as exc:
                    self.progress.emit(self.channel, f"Could not save tuning summary: {exc}")

    def _program(self):
        program, duration, interval = make_program(self.settings)
        target = self.device.get_target_temp(self.channel)
        started, previous = time.monotonic(), None
        try:
            if isinstance(program, TecTemperatureSweep):
                # Complete every step and repetition, even when serial I/O or
                # Windows timer resolution takes longer than the nominal tick.
                for step in range(program.step_count):
                    if self.stop_event.is_set():
                        break
                    value = program.target_at((step + .5) * program.step_seconds)
                    self.device.set_target_temp(self.channel, value)
                    self.progress.emit(self.channel, f"range: {value:.5f} °C; step {step + 1}/{program.step_count}")
                    self.stop_event.wait(program.step_seconds)
                return {"ok": not self.stop_event.is_set(), "aborted": self.stop_event.is_set()}
            while not self.stop_event.is_set():
                elapsed = time.monotonic() - started
                if elapsed >= duration:
                    break
                value = program.target_at(elapsed)
                if previous is None or abs(value - previous) >= .000005:
                    self.device.set_target_temp(self.channel, value)
                    previous = value
                    self.progress.emit(self.channel, f"{self.kind}: {value:.5f} °C; {elapsed:.1f}/{duration:.1f} s")
                elapsed = time.monotonic() - started
                next_tick = (math.floor(elapsed / interval) + 1) * interval
                if isinstance(program, TecSineSweep):
                    dwell = program.offsets.step_seconds
                    next_tick = min(next_tick, (math.floor(elapsed / dwell) + 1) * dwell)
                self.stop_event.wait(max(0., min(next_tick, duration) - elapsed))
            return {"ok": not self.stop_event.is_set(), "aborted": self.stop_event.is_set()}
        finally:
            self.device.set_target_temp(self.channel, target)

    def run(self):
        result, error = None, None
        try:
            if self.stop_event.is_set():
                return
            # Per-channel drafts also apply to sweeps and both tuning methods,
            # even when automatic output control is disabled. Never reuse a
            # stale device limit or copy the other TEC element's setting.
            limit = self.settings["limit"] if "limit" in self.settings else self.device.get_max_output(self.channel)
            apply_output_limit(self.device, self.channel, limit)
            if self.stop_event.is_set():
                return
            if self.settings.get("auto_output", True) and self.kind != "zn_tune":
                self.device.set_enable(self.channel, True)
            callbacks = dict(on_step=lambda text: self.progress.emit(self.channel, text), should_stop=self.stop_event.is_set)
            if self.kind in {"range", "sine", "sine_sweep"}:
                result = self._program()
            elif self.kind == "step":
                result = run_step_test(self.device, self.channel, self.settings["targets"],
                                       accuracy=self.settings["accuracy"], interval=self.settings["hold_s"],
                                       max_step_s=self.settings["timeout_s"], **callbacks)
                path = Path(self.settings["result_path"])
                keys = ["target", "settled", "settle_time_s", "elapsed_s", "in_band_s", "final_temp", "max_abs_dev"]
                with path.open("x", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["channel", *keys, "accuracy_C", "hold_interval_s", "pid_p", "pid_i", "pid_d"])
                    for row in result["results"]:
                        writer.writerow([self.channel, *[row[key] for key in keys], result["accuracy"], result["interval"], *result["pid"]])
                result["result_path"] = str(path)
            elif self.kind == "hardware_tune":
                result = run_hw_autotune(self.device, self.channel, **callbacks)
            elif self.kind == "zn_tune":
                result = self._zn_tune(callbacks, limit)
            else:
                raise ValueError("Unknown TEC207L job")
        except Exception as exc:
            error = str(exc)
        finally:
            if self.settings.get("auto_output", True) or self.force_off.is_set():
                try:
                    self.device.set_enable(self.channel, False)
                    if self.device.get_enable(self.channel):
                        raise RuntimeError("Output OFF was not confirmed")
                except Exception as exc:
                    error = f"{error + '; ' if error else ''}Output OFF: {exc}"
            if error:
                self.failed.emit(self.channel, error)
            elif result is not None:
                self.completed.emit(self.channel, result)
