"""Optional temperature readings associated with each waveform capture."""
from __future__ import annotations

from datetime import datetime
import logging
import math
import struct
import time


def read_itc4005(resource: str, timeout_ms: int = 5000) -> tuple[float, str]:
    import pyvisa
    manager = pyvisa.ResourceManager()
    instrument = None
    try:
        instrument = manager.open_resource(resource)
        instrument.timeout = timeout_ms
        # Query the existing communication unit; leave controller settings alone.
        unit = instrument.query("UNIT:TEMP?").strip().split()[-1].upper().strip('"')
        value = float(instrument.query("MEAS:TEMP?").strip().split()[-1])
        if not math.isfinite(value) or abs(value) > 1e6:
            raise ValueError("Invalid ITC4005 temperature reading")
        if unit in {"CEL", "C", "CELSIUS"}:
            return value, "degC"
        if unit in {"K", "KEL", "KELVIN"}:
            return value - 273.15, "degC"
        if unit in {"FAR", "F", "FAHRENHEIT"}:
            return (value - 32) * 5 / 9, "degC"
        raise ValueError(f"Unknown ITC4005 temperature unit: {unit}")
    finally:
        try:
            if instrument is not None:
                instrument.close()
        finally:
            manager.close()


class T4200:
    """Read a prefix and four-byte float using the observed T4200 framing."""

    def __init__(self, comport: str, float_offset: int = 1):
        import serial
        if not 0 <= float_offset <= 5:
            raise ValueError("Float offset must be between 0 and 5")
        self.float_offset = float_offset
        self.o = serial.Serial(comport, 9600, timeout=0.05, write_timeout=1)
        try:
            self.o.setRTS(True)
            self.o.setDTR(False)
        except Exception:
            self.o.close()
            raise
        self.last_response = b""

    def close_serial_port(self):
        self.o.close()

    def read_current_value(self, channel: int):
        if channel not in (1, 2):
            raise ValueError("T4200 channel must be 1 (A) or 2 (B)")
        self.o.reset_input_buffer()
        self.o.write(b"h" + bytes([channel - 1]))
        time.sleep(0.8)
        frame = bytearray()
        expected_bytes = self.float_offset + struct.calcsize("<f")
        deadline = time.monotonic() + 1.5
        while len(frame) < expected_bytes and time.monotonic() < deadline:
            frame.extend(self.o.read(expected_bytes - len(frame)))
        self.last_response = bytes(frame)
        if len(frame) < expected_bytes:
            raise TimeoutError(f"T4200 response has {len(frame)} bytes, expected at least {expected_bytes}: {frame.hex()}")
        value = struct.unpack_from("<f", frame, self.float_offset)[0]
        if not math.isfinite(value):
            raise ValueError(f"Non-finite T4200 reading: {frame.hex()}")
        return value


def read_optional_temperatures(config) -> dict[str, str]:
    metadata = {}
    logger = logging.getLogger("scope_recording")
    if config.itc4005_resource:
        metadata["tec_resource"] = config.itc4005_resource
        metadata["tec_temperature_read_at"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        try:
            value, unit = read_itc4005(config.itc4005_resource, min(config.timeout_ms, 5000))
            metadata.update(tec_temperature=str(value), tec_temperature_unit=unit, tec_temperature_status="ok")
            logger.info("TEC temperature: %.6g %s", value, unit)
        except Exception as exc:
            metadata.update(tec_temperature_status="error", tec_temperature_error=str(exc))
            logger.exception("Optional ITC4005 reading failed; saving waveform without TEC temperature")
    if config.t4200_port:
        metadata.update(ambient_port=config.t4200_port, ambient_channel=str(config.t4200_channel),
                        ambient_float_offset=str(config.t4200_float_offset))
        metadata["ambient_temperature_read_at"] = datetime.now().astimezone().isoformat(timespec="milliseconds")
        probe = None
        try:
            probe = T4200(config.t4200_port, config.t4200_float_offset)
            value = probe.read_current_value(config.t4200_channel)
            metadata.update(ambient_temperature=str(value), ambient_temperature_unit="degC",
                            ambient_temperature_status="ok", ambient_response_hex=probe.last_response.hex())
            logger.info("Ambient temperature: %.6g degC; raw response=%s", value, probe.last_response.hex())
        except Exception as exc:
            metadata.update(ambient_temperature_status="error", ambient_temperature_error=str(exc))
            if probe is not None:
                metadata["ambient_response_hex"] = probe.last_response.hex()
            logger.exception("Optional T4200 reading failed; saving waveform without ambient temperature")
        finally:
            if probe is not None:
                try:
                    probe.close_serial_port()
                except Exception:
                    logger.exception("Failed to close T4200 serial port")
    return metadata
