from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging

import numpy as np

from signal_data_class import SignalData

try:
	import pyvisa
except Exception:  # pragma: no cover - optional dependency
	pyvisa = None


@dataclass
class ScopeCaptureConfig:
	resource_name: str
	channel: str = "CH1"
	point_count: int = 10000000
	timeout_ms: int = 5000
	reference_channel: str | None = None
	itc4005_resource: str | None = None
	t4200_port: str | None = None
	t4200_channel: int = 1
	t4200_float_offset: int = 1
	tec_setpoint_deg_c: float | None = None


class ScopeCommunicationError(RuntimeError):
	"""An acquisition failed before a valid waveform could be obtained."""


class OscilloscopeImporter:
	@staticmethod
	def pyvisa_available() -> bool:
		return pyvisa is not None

	@staticmethod
	def _ensure_pyvisa():
		if pyvisa is None:
			raise RuntimeError("pyvisa ist nicht installiert. Bitte 'pyvisa' (und ein VISA Backend) installieren.")

	@staticmethod
	def _parse_query_response(response: str) -> str:
		response = str(response).strip()
		if " " in response:
			return response.split()[-1]
		return response

	@classmethod
	def _query_float(cls, instrument: Any, command: str, fallback: float = 0.0) -> float:
		try:
			val = cls._parse_query_response(instrument.query(command))
			value = float(val)
			if not np.isfinite(value):
				raise ValueError("Non-finite calibration value")
			return value
		except Exception as exc:
			raise ScopeCommunicationError(f"Calibration query {command} failed: {exc}") from exc

	@classmethod
	def list_resources(cls) -> tuple[str, ...]:
		cls._ensure_pyvisa()
		rm = pyvisa.ResourceManager()
		try:
			return tuple(rm.list_resources())
		finally:
			rm.close()

	@classmethod
	def capture_channel(cls, config: ScopeCaptureConfig) -> SignalData:
		cls._ensure_pyvisa()

		rm = None
		inst = None
		stage = "opening VISA connection"
		try:
			logging.getLogger("scope_recording").info("%s: %s", stage, config.resource_name)
			rm = pyvisa.ResourceManager()
			inst = rm.open_resource(config.resource_name)
			inst.timeout = max(1000, int(config.timeout_ms))
			stage = "configuring waveform transfer"

			if config.reference_channel:
				if config.reference_channel.upper() == config.channel.upper():
					raise ValueError("Optical and reference channels must be different")
				stage = "freezing the paired waveform record"
				was_running = bool(cls._query_float(inst, "ACQUIRE:STATE?"))
				try:
					inst.write("ACQUIRE:STATE STOP")
					data = cls._read_channel(inst, config, config.channel)
					data.reference = cls._read_channel(inst, config, config.reference_channel)
					data.metadata.update({"optical_channel": config.channel,
					                      "reference_channel": config.reference_channel,
					                      "paired_capture": "same stopped oscilloscope record"})
				finally:
					if was_running:
						inst.write("ACQUIRE:STATE RUN")
				return data
			stage = "reading waveform"
			return cls._read_channel(inst, config, config.channel)

		except MemoryError:
			raise
		except Exception as exc:
			logging.getLogger("scope_recording").exception("Acquisition failed during %s", stage)
			raise ScopeCommunicationError(f"{stage}: {exc}") from exc
		finally:
			if inst is not None:
				try:
					inst.close()
				except Exception:
					logging.getLogger("scope_recording").exception("Failed to close oscilloscope session")
			if rm is not None:
				try:
					rm.close()
				except Exception:
					logging.getLogger("scope_recording").exception("Failed to close VISA resource manager")

	@classmethod
	def _read_channel(cls, inst, config, channel):
		channel = channel.strip().upper()
		points = int(max(100, config.point_count))

		inst.write(f"DATA:SOURCE {channel}")
		inst.write("DATA:WIDTH 2")
		inst.write("DATA:START 1")
		inst.write(f"DATA:STOP {points}")
		inst.write("DATA:ENC RPB")

		stage = "reading waveform calibration"
		logging.getLogger("scope_recording").info(stage)
		xincr = cls._query_float(inst, "WFMPRE:XINCR?", fallback=1.0)
		ymult = cls._query_float(inst, "WFMPRE:YMULT?", fallback=1.0)
		yzero = cls._query_float(inst, "WFMPRE:YZERO?", fallback=0.0)
		yoff = cls._query_float(inst, "WFMPRE:YOFF?", fallback=0.0)
		xzero = cls._query_float(inst, "WFMPRE:XZERO?", fallback=0.0)
		if xincr <= 0 or ymult == 0:
			raise ValueError("Invalid waveform sample interval or voltage scale")

		stage = "reading CURVe binary waveform"
		logging.getLogger("scope_recording").info(stage)
		raw = inst.query_binary_values(
			"CURVe?",
			datatype="H",
			is_big_endian=True,
			container=np.array,
		)
		raw = np.asarray(raw, dtype=float)
		if raw.size == 0 or not np.all(np.isfinite(raw)):
			raise ValueError("Empty or non-finite waveform received")
		volts = (raw - yoff) * ymult + yzero
		time = xzero + np.arange(volts.size, dtype=float) * xincr

		metadata = {
			"source": "oscilloscope",
			"amplitude_unit": "V",
			"resource": config.resource_name,
			"channel": channel,
			"point_count": str(points),
			"Sample Interval": str(xincr),
			"YMULT": str(ymult),
			"YZERO": str(yzero),
			"YOFF": str(yoff),
		}

		fs = 1.0 / xincr if xincr > 0 else 1.0
		return SignalData(
			time=time,
			amplitude=volts,
			source_name=f"{channel} @ {config.resource_name}",
			sampling_rate=fs,
			metadata=metadata,
			column_names=("TIME", channel),
		)
