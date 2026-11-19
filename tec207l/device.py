"""Serial driver for the SenseFuture TEC207L dual-channel TEC controller.

Uses the ASCII protocol (see protocol.py) over a COM port, typically a
USB-TTL adapter wired to the controller's TTL interface (default 38400 baud,
8N1). All multi-byte values are scaled integers per the protocol spec:

* temperatures (TG, TCADJTEMP) : raw = degC * 1e5        (2500000 = 25.00000 C)
* sensor resistance (RESISTOR) : raw = ohm * 1e6         (1e10 = 10.000000 kOhm)
* NTC B value (BX)             : raw = B * 100           (395000 = 3950.00)
* NTC R0 (RP)                  : raw = ohm               (10000 = 10 kOhm)
* NTC internal ref (NTCRP)     : raw = ohm * 1e6
* PT R0 (PT1000RP)             : raw = ohm * 1e3         (1000000 = 1.000 kOhm)
* PT A (PTA)                   : raw = A * 1e9          (3908300 = 3.9083e-3)
* PT B (PTB)                   : raw = B * 1e12         (-577500 = -5.775e-7)
* PT C (PTC)                   : raw = C * 1e16         (-41830 = -4.183e-12)
* PT internal ref (PTRP)       : raw = ohm * 1e6
* output current (CURRENT)     : raw = amp * 1e3         (3256 = 3.256 A)
"""

from __future__ import annotations

import threading
import time
import math

import serial

from . import protocol
from .protocol import ProtocolError

DEFAULT_BAUDRATE = 38400  # TTL interface default (RS485 default is 9600)

MODEL_NAMES = {
    1: "TEC103", 2: "TEC207L", 3: "TEC207", 4: "TEC215L", 5: "TEC215",
    6: "TEC215Pro", 7: "TEC107L", 8: "TEC107", 12: "TEC100L", 13: "TEC100",
    15: "TEC403L", 18: "TEC415L", 20: "TEC603L", 22: "TEC615L",
    25: "TEC803L", 27: "TEC815L", 30: "TEC203L",
}

SENSOR_MODELS = {0: "NTC (B-value)", 1: "PT1000 (Pt RTD)", 2: "Steinhart-Hart", 3: "MF501"}
OUTPUT_MODES = {0: "Cool & Heat", 1: "Cool only", 2: "Heat only", 3: "PWM via comms"}


class TecError(Exception):
    """Raised for connection problems or failed device transactions."""


class Tec207L:
    """Thread-safe serial connection to a TEC207L controller."""

    def __init__(self, port: str, baudrate: int = DEFAULT_BAUDRATE, timeout: float = 0.4):
        self.port = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self._ser: serial.Serial | None = None
        self._lock = threading.Lock()
        self.on_transaction = lambda direction, data: None

    # ------------------------------------------------------------------ #
    # connection handling
    # ------------------------------------------------------------------ #
    @property
    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def connect(self) -> None:
        if self.is_connected:
            return
        try:
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout,
                write_timeout=self.timeout,
            )
        except serial.SerialException as exc:
            raise TecError(f"cannot open {self.port}: {exc}") from exc

    def disconnect(self) -> None:
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                finally:
                    self._ser = None

    # ------------------------------------------------------------------ #
    # low-level transaction
    # ------------------------------------------------------------------ #
    def _transact(self, frame: bytes, expected: str) -> int | str:
        with self._lock:
            if not self.is_connected:
                raise TecError("not connected")
            assert self._ser is not None
            try:
                self._ser.reset_input_buffer()
                self.on_transaction("TX", frame)
                self._ser.write(frame)
                self._ser.flush()
                # read only the bytes that are already there: read(n) would
                # block until n bytes or the full timeout, which makes every
                # query take ~timeout even though the device replies in ~2 ms
                buf = bytearray()
                deadline = time.monotonic() + self.timeout + 0.25
                while time.monotonic() < deadline:
                    waiting = self._ser.in_waiting
                    if waiting:
                        buf += self._ser.read(waiting)
                        if b"\n" in buf:
                            break
                    else:
                        time.sleep(0.002)
                self.on_transaction("RX", bytes(buf))
            except serial.SerialException as exc:
                raise TecError(f"serial I/O error on {self.port}: {exc}") from exc
        return protocol.parse_reply(bytes(buf), expected)

    def query(self, channel: int, command: str) -> int | str:
        """Query a channel register, e.g. query(1, 'TG')."""
        return self._transact(protocol.build_query(channel, command),
                              protocol.expected_key(channel, command))

    def set(self, channel: int, command: str, value: int) -> int | str:
        """Write a channel register; returns the value echoed by the device."""
        return self._transact(protocol.build_set(channel, command, value),
                              protocol.expected_key(channel, command))

    def general_query(self, command: str) -> int | str:
        """Query a general (device-wide) register, e.g. general_query('FPV')."""
        return self._transact(protocol.build_general_query(command),
                              protocol.expected_key(None, command))

    def general_set(self, command: str, value: int) -> int | str:
        return self._transact(protocol.build_general_set(command, value),
                              protocol.expected_key(None, command))

    # ------------------------------------------------------------------ #
    # device info
    # ------------------------------------------------------------------ #
    def get_model(self) -> str:
        """Model name; the real firmware answers the TEC register with e.g. '207L'."""
        raw = self.general_query("TEC")
        if isinstance(raw, str):
            return raw.upper() if raw.upper().startswith("TEC") else f"TEC{raw}"
        return MODEL_NAMES.get(raw, f"unknown ({raw})")

    def get_firmware(self) -> str:
        raw = self.general_query("FPV")  # spec: 100 -> "1.0.0"; real FW returns e.g. 5007
        if isinstance(raw, str):
            return raw
        if raw >= 1000:  # e.g. 5007 -> "5.0.7"
            return f"{raw // 1000}.{(raw // 100) % 10}.{raw % 100}"
        return f"{raw // 100}.{(raw // 10) % 10}.{raw % 10}"

    # ------------------------------------------------------------------ #
    # temperatures
    # ------------------------------------------------------------------ #
    @staticmethod
    def _temp_to_raw(celsius: float) -> int:
        return int(round(celsius * 1e5))

    @staticmethod
    def _raw_to_temp(raw: int) -> float:
        return raw / 1e5

    def get_target_temp(self, channel: int) -> float:
        """Target/setpoint temperature in degC."""
        return self._raw_to_temp(self.query(channel, "TG"))

    def set_target_temp(self, channel: int, celsius: float) -> float:
        return self._raw_to_temp(self.set(channel, "TG", self._temp_to_raw(celsius)))

    def get_actual_temp(self, channel: int) -> float:
        """Measured temperature in degC (this is the live value to plot)."""
        return self._raw_to_temp(self.query(channel, "TCADJTEMP"))

    def get_sensor_resistance(self, channel: int) -> float:
        """Raw sensor resistance in ohm."""
        return self.query(channel, "RESISTOR") / 1e6

    # ------------------------------------------------------------------ #
    # sensor configuration (NTC and PT)
    # ------------------------------------------------------------------ #
    def get_sensor_model(self, channel: int) -> int:
        """0 = NTC B-value, 1 = PT1000, 2 = Steinhart-Hart, 3 = MF501."""
        return self.query(channel, "POLYOMIAL")

    def set_sensor_model(self, channel: int, model: int) -> int:
        return self.set(channel, "POLYOMIAL", model)

    def get_ntc(self, channel: int) -> tuple[float, float, float]:
        """Returns (B value, R0 in ohm, internal reference resistor in ohm)."""
        b = self.query(channel, "BX") / 100.0
        r0 = float(self.query(channel, "RP"))
        ref = self.query(channel, "NTCRP") / 1e6
        return b, r0, ref

    def set_ntc(self, channel: int, b_value: float, r0_ohm: float, ref_ohm: float) -> None:
        self.set(channel, "BX", int(round(b_value * 100)))
        self.set(channel, "RP", int(round(r0_ohm)))
        self.set(channel, "NTCRP", int(round(ref_ohm * 1e6)))

    def get_pt(self, channel: int) -> tuple[float, float, float, float, float]:
        """Returns (R0 ohm, CvD A, CvD B, CvD C, internal reference resistor ohm)."""
        r0 = self.query(channel, "PT1000RP") / 1e3
        a = self.query(channel, "PTA") / 1e9
        b = self.query(channel, "PTB") / 1e12
        c = self.query(channel, "PTC") / 1e16
        ref = self.query(channel, "PTRP") / 1e6
        return r0, a, b, c, ref

    def set_pt(self, channel: int, r0_ohm: float, a: float, b: float, c: float,
               ref_ohm: float) -> None:
        self.set(channel, "PT1000RP", int(round(r0_ohm * 1e3)))
        self.set(channel, "PTA", int(round(a * 1e9)))
        self.set(channel, "PTB", int(round(b * 1e12)))
        self.set(channel, "PTC", int(round(c * 1e16)))
        self.set(channel, "PTRP", int(round(ref_ohm * 1e6)))

    # ------------------------------------------------------------------ #
    # PID
    # ------------------------------------------------------------------ #
    def get_pid(self, channel: int) -> tuple[int, int, int]:
        """Returns (P, I, D) as raw integer coefficients."""
        return (self.query(channel, "KP"),
                self.query(channel, "KI"),
                self.query(channel, "KD"))

    def set_pid(self, channel: int, p: int, i: int, d: int) -> tuple[int, int, int]:
        self.set(channel, "KP", int(p))
        self.set(channel, "KI", int(i))
        self.set(channel, "KD", int(d))
        return self.get_pid(channel)

    def start_autotune(self, channel: int) -> int | str:
        """Start the on-device PID self-tune (AUTOPID=1); returns the echoed value."""

        return self.set(channel, "AUTOPID", 1)

    def stop_autotune(self, channel: int) -> int | str:
        """Abort the on-device PID self-tune (AUTOPID=0)."""

        return self.set(channel, "AUTOPID", 0)

    def get_autotune_state(self, channel: int) -> int:
        """AUTOPID register read-back: 0 = idle, 1 = PID self-tuning running,
        2 = PID real-time automatic optimization."""

        return int(self.query(channel, "AUTOPID"))

    # ------------------------------------------------------------------ #
    # output control
    # ------------------------------------------------------------------ #
    def get_enable(self, channel: int) -> bool:
        return self.query(channel, "ENABLE") == 1

    def set_enable(self, channel: int, enabled: bool) -> None:
        self.set(channel, "ENABLE", 1 if enabled else 0)

    def get_output_mode(self, channel: int) -> int:
        """0 = cool&heat, 1 = cool only, 2 = heat only, 3 = PWM via comms."""
        return self.query(channel, "MODE")

    def set_output_mode(self, channel: int, mode: int) -> None:
        self.set(channel, "MODE", mode)

    def get_max_output(self, channel: int) -> int:
        """Max output as percent of input voltage (0..90)."""
        return self.query(channel, "LIMITED")

    def set_max_output(self, channel: int, percent: int) -> None:
        self.validate_max_output(percent)
        self.set(channel, "LIMITED", int(percent))

    @staticmethod
    def validate_max_output(percent):
        """LIMITED is an integer voltage percentage, not a limit in watts."""
        if (isinstance(percent, bool) or not isinstance(percent, (int, float))
                or not math.isfinite(percent) or not 0 <= percent <= 90 or int(percent) != percent):
            raise ValueError("TEC207L output voltage limit must be a whole percentage from 0 to 90")
        return int(percent)

    def get_output_current(self, channel: int) -> float:
        """Actual output current in ampere."""
        return self.query(channel, "CURRENT") / 1e3

    def get_output_percent(self, channel: int) -> float:
        """PWMDUTY diagnostic: raw 200000 corresponds to 10% of Vin."""
        return self.query(channel, "PWMDUTY") / 20000.

    def get_control_mode(self) -> int:
        """CONTMODE=0 is independent; modes 2/3 couple CH2 to CH1."""
        return int(self.general_query("CONTMODE"))

    def get_full_state(self, channel: int) -> dict:
        """Read the complete current configuration of one channel.

        Used at startup/connect so the GUI shows the values actually stored
        in the TEC controller instead of placeholder defaults.
        """
        return {
            "target": self.get_target_temp(channel),
            "actual": self.get_actual_temp(channel),
            "resistance": self.get_sensor_resistance(channel),
            "pid": self.get_pid(channel),
            "model": self.get_sensor_model(channel),
            "ntc": self.get_ntc(channel),
            "pt": self.get_pt(channel),
            "enable": self.get_enable(channel),
            "mode": self.get_output_mode(channel),
            "limit": self.get_max_output(channel),
            "current": self.get_output_current(channel),
        }
