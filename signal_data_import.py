from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import logging
import time

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
	tec_settle_seconds: float = 0.0
	fresh_trigger: bool = False
	trigger_wait_seconds: float = 30.0
	auto_tec_output: bool = False
	additional_channels: tuple[str, ...] = ()

	def capture_channels(self):
		primary = self.channel.strip().upper()
		if self.reference_channel and self.reference_channel.strip().upper() == primary:
			raise ValueError("Optical and reference channels must be different")
		channels = list(dict.fromkeys([primary, *(c.strip().upper() for c in self.additional_channels),
		                              *([self.reference_channel.strip().upper()] if self.reference_channel else [])]))
		if any(channel not in {"CH1", "CH2", "CH3", "CH4"} for channel in channels):
			raise ValueError("Select analog scope channels CH1 through CH4")
		return channels


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
	def _parse_mode_response(cls, response: str, keywords: tuple[str, ...]) -> str | None:
		"""Expand SCPI short/long keywords without changing the scope's VERBOSE setting.

		Uppercase letters in each keyword specify its minimum accepted spelling.
		"""
		value = cls._parse_query_response(response).upper()
		for keyword in keywords:
			minimum = sum(letter.isupper() for letter in keyword)
			if minimum <= len(value) <= len(keyword) and keyword.upper().startswith(value):
				return keyword.upper()
		return None

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
		config.capture_channels()

		rm = None
		inst = None
		stage = "opening VISA connection"
		try:
			logging.getLogger("scope_recording").info("%s: %s", stage, config.resource_name)
			rm = pyvisa.ResourceManager()
			inst = rm.open_resource(config.resource_name)
			inst.timeout = max(1000, int(config.timeout_ms))
			stage = "configuring waveform transfer"
			if config.fresh_trigger:
				stage = "waiting for a fresh triggered waveform"
				return cls._capture_triggered(inst, config)

			# Calibration and CURVe must describe the same stopped record, even
			# for a single channel while the scope is continuously acquiring.
			stage = "freezing the waveform record"
			was_running = bool(cls._query_float(inst, "ACQUIRE:STATE?"))
			try:
				inst.write("ACQUIRE:STATE STOP")
				return cls._read_channels(inst, config)
			finally:
				if was_running:
					inst.write("ACQUIRE:STATE RUN")

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
	def _capture_triggered(cls, inst, config):
		"""Acquire a new single sequence using the scope's configured trigger source/level."""
		if not np.isfinite(config.trigger_wait_seconds) or config.trigger_wait_seconds <= 0:
			raise ValueError("Trigger wait must be finite and positive")
		was_running = bool(cls._query_float(inst, "ACQUIRE:STATE?"))
		stop_after_response = inst.query("ACQUIRE:STOPAFTER?")
		trigger_mode_response = inst.query("TRIGGER:A:MODE?")
		stop_after = cls._parse_mode_response(stop_after_response, ("RUNStop", "SEQuence"))
		trigger_mode = cls._parse_mode_response(trigger_mode_response, ("AUTO", "NORMal"))
		if stop_after is None or trigger_mode is None:
			raise ValueError(
				"Unsupported oscilloscope acquisition/trigger mode readback: "
				f"ACQUIRE:STOPAFTER? returned {stop_after_response!r}; "
				f"TRIGGER:A:MODE? returned {trigger_mode_response!r}")
		try:
			inst.write("ACQUIRE:STATE STOP")
			inst.write("TRIGGER:A:MODE NORMAL")
			inst.write("ACQUIRE:STOPAFTER SEQUENCE")
			inst.write("ACQUIRE:STATE RUN")
			deadline = time.monotonic() + config.trigger_wait_seconds
			while cls._query_float(inst, "ACQUIRE:STATE?") != 0:
				if time.monotonic() >= deadline:
					raise TimeoutError("No completed triggered acquisition before trigger timeout")
				time.sleep(0.1)
			data = cls._read_channels(inst, config)
			data.metadata["fresh_trigger"] = "true"
			return data
		finally:
			inst.write("ACQUIRE:STATE STOP")
			inst.write(f"ACQUIRE:STOPAFTER {stop_after}")
			inst.write(f"TRIGGER:A:MODE {trigger_mode}")
			if was_running:
				inst.write("ACQUIRE:STATE RUN")

	@classmethod
	def _read_channels(cls, inst, config):
		channels = config.capture_channels()
		traces = {channel: cls._read_channel(inst, config, channel) for channel in channels}
		data = traces[channels[0]]
		data.channels = {channel: traces[channel] for channel in channels[1:]
		                 if channel in {c.strip().upper() for c in config.additional_channels}}
		if config.reference_channel:
			data.reference = traces[config.reference_channel.strip().upper()]
			data.metadata.update(optical_channel=channels[0], reference_channel=config.reference_channel.strip().upper())
		if len(channels) > 1:
			data.metadata.update(captured_channels=",".join(channels),
			                     paired_capture="same stopped oscilloscope record")
		return data

	@classmethod
	def _read_channel(cls, inst, config, channel):
		channel = channel.strip().upper()
		points = int(max(100, config.point_count))

		inst.write(f"DATA:SOURCE {channel}")
		inst.write("DATA:ENC RPB")
		inst.write("DATA:WIDTH 2")
		inst.write("DATA:START 1")
		inst.write(f"DATA:STOP {points}")

		stage = "reading waveform calibration"
		logging.getLogger("scope_recording").info(stage)
		# On MDO scopes WFMOutpre describes CURVe?; WFMInpre describes
		# uploaded reference data. Never use unrelated/stale reference scaling.
		xunit = cls._parse_query_response(inst.query("WFMOutpre:XUNIT?")).strip('"').lower()
		point_format = cls._parse_query_response(inst.query("WFMOutpre:PT_FMT?")).upper()
		if xunit != "s":
			raise ValueError(f"Expected a time-domain waveform in seconds; received {xunit!r}")
		if point_format != "Y":
			raise ValueError(f"Unsupported waveform point format {point_format!r}; select Sample, HiRes or Average, not Envelope")
		count = cls._query_float(inst, "WFMOutpre:NR_PT?")
		if count < 1 or count != int(count) or count > points:
			raise ValueError(f"Invalid waveform transfer length: {count}; requested at most {points}")
		xincr = cls._query_float(inst, "WFMOutpre:XINCR?")
		ymult = cls._query_float(inst, "WFMOutpre:YMULT?")
		yzero = cls._query_float(inst, "WFMOutpre:YZERO?")
		yoff = cls._query_float(inst, "WFMOutpre:YOFF?")
		xzero = cls._query_float(inst, "WFMOutpre:XZERO?")
		pt_off = cls._query_float(inst, "WFMOutpre:PT_OFF?")
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
		if raw.ndim != 1 or raw.size != int(count):
			raise ValueError(f"Incomplete waveform: received {raw.size} samples, expected {int(count)}")
		volts = (raw - yoff) * ymult + yzero
		time = xzero + (np.arange(volts.size, dtype=float) - pt_off) * xincr

		metadata = {
			"source": "oscilloscope",
			"amplitude_unit": "V",
			"resource": config.resource_name,
			"channel": channel,
			"point_count": str(raw.size),
			"requested_point_count": str(points),
			"waveform_preamble": "WFMOutpre",
			"Sample Interval": str(xincr),
			"XZERO": str(xzero),
			"PT_OFF": str(pt_off),
			"YMULT": str(ymult),
			"YZERO": str(yzero),
			"YOFF": str(yoff),
		}

		fs = 1.0 / xincr if xincr > 0 else 1.0
		logging.getLogger("scope_recording").info(
			"Captured %s: %d samples (limit %d), interval %.12g s, rate %.12g Hz, time %.12g to %.12g s",
			channel, raw.size, points, xincr, fs, time[0], time[-1])
		return SignalData(
			time=time,
			amplitude=volts,
			source_name=f"{channel} @ {config.resource_name}",
			sampling_rate=fs,
			metadata=metadata,
			column_names=("TIME", channel),
		)
