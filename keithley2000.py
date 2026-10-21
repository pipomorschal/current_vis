"""Keithley 2000 SCPI readings through a USB-to-RS232 adapter.

Protocol reference: https://download.tek.com/manual/2000-900_J-Aug2010_User.pdf
RS232 uses 8N1, CR commands, and no hardware handshaking.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import csv
import math
import re
import threading
import time

import serial


# Label -> SCPI function, display unit, optional ASCII reading suffix.
FUNCTIONS = {
    "DC voltage": ("VOLT:DC", "V", "VDC"),
    "AC voltage (RMS)": ("VOLT:AC", "V", "VAC"),
    "DC current": ("CURR:DC", "A", "ADC"),
    "AC current (RMS)": ("CURR:AC", "A", "AAC"),
    "Resistance (2-wire)": ("RES", "Ω", "OHM"),
    "Resistance (4-wire)": ("FRES", "Ω", "OHM4W"),
    "Frequency": ("FREQ", "Hz", "HZ"),
}
BAUD_RATES = (300, 600, 1200, 2400, 4800, 9600, 19200)


class MeterCancelled(Exception):
    pass


@dataclass(frozen=True)
class KeithleySettings:
    port: str
    baud_rate: int = 4800
    function: str = "DC voltage"
    interval_seconds: float = 1.0
    timeout_seconds: float = 5.0
    xonxoff: bool = False

    def __post_init__(self):
        if not self.port.strip():
            raise ValueError("Select the Keithley USB-to-RS232 adapter's COM port")
        if self.baud_rate not in BAUD_RATES or self.function not in FUNCTIONS:
            raise ValueError("Unsupported Keithley baud rate or measurement function")
        if (not all(math.isfinite(v) for v in (self.interval_seconds, self.timeout_seconds))
                or self.interval_seconds < 0.1 or self.timeout_seconds <= 0):
            raise ValueError("Read interval must be at least 0.1 s and timeout must be positive")

    @property
    def unit(self):
        return FUNCTIONS[self.function][1]


def parse_reading(response, function):
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*([A-Za-z0-9]*)\s*", response)
    if match is None:
        raise ValueError(f"Invalid Keithley reading: {response[:100]!r}")
    suffix = match[2].upper()
    if suffix and suffix != FUNCTIONS[function][2]:
        raise ValueError(f"Unexpected Keithley reading units: {suffix}")
    value = float(match[1])
    if not math.isfinite(value) or abs(value) >= 9e36:
        raise ValueError("Keithley reading is overrange or invalid; check the selected function and input")
    return value


class Keithley2000:
    def __init__(self, settings: KeithleySettings, cancelled=None):
        self.settings = settings
        self.cancelled = cancelled if cancelled is not None else threading.Event()
        self.port = None
        self.identity = None
        self.traffic = deque(maxlen=24)
        self._commands = deque(maxlen=16)

    @property
    def diagnostics(self):
        s = self.settings
        return (f"Keithley 2000 serial session: {s.port}, {s.baud_rate} baud, 8N1, "
                f"flow={'XON/XOFF' if s.xonxoff else 'None'}, timeout={s.timeout_seconds:g} s\n"
                + "\n".join(self.traffic))

    def _check_cancelled(self):
        if self.cancelled.is_set():
            raise MeterCancelled()

    def _write(self, command):
        self._check_cancelled()
        data = (command + "\r").encode("ascii")
        self.traffic.append(f"TX {data!r}")
        self._commands.append(command.upper().lstrip(":"))
        if self.port.write(data) != len(data):
            raise TimeoutError("Incomplete Keithley serial command")

    def query(self, command, *, accept=None):
        self._write(command)
        deadline = time.monotonic() + self.settings.timeout_seconds
        response = bytearray()
        while time.monotonic() < deadline:
            self._check_cancelled()
            byte = self.port.read(1)
            if byte in (b"\r", b"\n"):
                if response:
                    raw = bytes(response)
                    self.traffic.append(f"RX {raw!r} (hex: {raw.hex(' ')})")
                    response.clear()
                    try:
                        line = raw.decode("ascii").strip()
                    except UnicodeDecodeError as exc:
                        raise ValueError(
                            f"Non-ASCII serial response {raw[:80]!r}; check baud rate, 8N1 and the selected COM port") from exc
                    # Some serial paths echo commands before the actual response.
                    if line.upper().lstrip(":") in self._commands:
                        continue
                    if line and (accept is None or accept(line)):
                        return line
            elif byte:
                if byte in (b"\x11", b"\x13"):
                    continue
                response.extend(byte)
                if len(response) > 1024:
                    raise ValueError("Keithley response exceeded 1024 bytes")
        if response:
            self.traffic.append(f"RX incomplete {bytes(response)!r} (hex: {response.hex(' ')})")
        raise TimeoutError(f"Keithley timed out on {command}; check COM port, baud rate, flow control and RS232/SCPI mode")

    @staticmethod
    def _accept_identity(identity):
        parts = next(csv.reader([identity], skipinitialspace=True))
        if len(parts) < 2 or "KEITHLEY" not in parts[0].upper():
            return False
        model = parts[1].strip().strip('"').upper()
        if re.fullmatch(r"(?:MODEL\s*)?2000", model):
            return True
        # A different recognizable model should not be configured as a 2000.
        raise ValueError(f"Expected a Keithley 2000; received {identity!r}")

    def _identify(self):
        for attempt in range(2):
            try:
                return self.query("*IDN?", accept=self._accept_identity)
            except TimeoutError:
                if attempt == 1:
                    received = [line for line in self.traffic if line.startswith("RX")]
                    details = "; ".join(received[-3:]) if received else "no response"
                    raise TimeoutError(
                        f"Could not identify a Keithley 2000 on {self.settings.port} at "
                        f"{self.settings.baud_rate} baud; received {details}. "
                        "Check adapter COM port, meter baud/flow settings, RS232/SCPI mode and cable. "
                        "Use Copy serial details for the full exchange.") from None
                self.traffic.append("Retrying identification after clearing stale input")
                if self.cancelled.wait(0.2):
                    raise MeterCancelled()
                self.port.reset_input_buffer()

    def connect(self):
        self._check_cancelled()
        self.identity = None
        s = self.settings
        try:
            self.port = serial.Serial(port=s.port, baudrate=s.baud_rate, bytesize=serial.EIGHTBITS,
                                      parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                                      timeout=0.1, write_timeout=1, xonxoff=s.xonxoff,
                                      rtscts=False, dsrdtr=False)
            # ^C clears any unfinished query/output from an earlier session.
            self.port.write(b"\x03")
            self.port.reset_input_buffer()
            self.identity = self._identify()
            # CONF establishes a one-shot immediate trigger and disables math/scanning.
            self._write(":SYST:REM")
            self._write("*CLS")
            self._write(f":CONF:{FUNCTIONS[s.function][0]}")
            self._write(":FORM:DATA ASC")
            self._write(":FORM:ELEM READ")
            error = self.query(":SYST:ERR?")
            if int(error.split(",", 1)[0]) != 0:
                raise RuntimeError(f"Keithley configuration error: {error}")
            return self
        except Exception:
            self.close()
            raise

    def read(self):
        return parse_reading(self.query(":READ?"), self.settings.function)

    def close(self):
        if self.port is not None:
            try:
                if self.identity is not None:
                    # Abort any pending read before returning front-panel control.
                    self.port.write(b"\x03:SYST:LOC\r")
            except Exception:
                pass
            finally:
                self.port.close()
                self.port = None
