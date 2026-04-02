from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
	point_count: int = 100000
	timeout_ms: int = 5000


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
			return float(val)
		except Exception:
			return float(fallback)

	@classmethod
	def list_resources(cls) -> tuple[str, ...]:
		cls._ensure_pyvisa()
		rm = pyvisa.ResourceManager()
		resources = tuple(rm.list_resources())
		try:
			rm.close()
		except Exception:
			pass
		return resources

	@classmethod
	def capture_channel(cls, config: ScopeCaptureConfig) -> SignalData:
		cls._ensure_pyvisa()

		rm = pyvisa.ResourceManager()
		inst = None
		try:
			inst = rm.open_resource(config.resource_name)
			inst.timeout = max(1000, int(config.timeout_ms))

			channel = (config.channel or "CH1").strip().upper()
			points = int(max(100, config.point_count))

			inst.write(f"DATA:SOURCE {channel}")
			inst.write("DATA:WIDTH 2")
			inst.write("DATA:START 1")
			inst.write(f"DATA:STOP {points}")
			inst.write("DATA:ENC RPB")

			xincr = cls._query_float(inst, "WFMPRE:XINCR?", fallback=1.0)
			ymult = cls._query_float(inst, "WFMPRE:YMULT?", fallback=1.0)
			yzero = cls._query_float(inst, "WFMPRE:YZERO?", fallback=0.0)
			yoff = cls._query_float(inst, "WFMPRE:YOFF?", fallback=0.0)
			xzero = cls._query_float(inst, "WFMPRE:XZERO?", fallback=0.0)

			raw = inst.query_binary_values(
				"CURVe?",
				datatype="H",
				is_big_endian=True,
				container=np.array,
			)
			raw = np.asarray(raw, dtype=float)
			volts = (raw - yoff) * ymult + yzero
			time = xzero + np.arange(volts.size, dtype=float) * xincr

			metadata = {
				"source": "oscilloscope",
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
		finally:
			if inst is not None:
				try:
					inst.close()
				except Exception:
					pass
			try:
				rm.close()
			except Exception:
				pass
