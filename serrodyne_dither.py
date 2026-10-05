from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rectangular_ramp import (
    AFG1062_DAC_MAX_CODE,
    AFG1062_MAX_ARB_POINTS,
    AFG1062_MAX_ARB_REPETITION_HZ,
    AFG1062_MAX_SAMPLE_RATE_SPS,
    AFG1062_MIN_ARB_REPETITION_HZ,
)


TWO_PI = 2.0 * math.pi
# A sampled ramp is linear between resets, but its reset edge still needs a
# reasonably dense grid. Sixteen samples at the highest instantaneous
# frequency is the minimum accepted by this generator.
SERRODYNE_MIN_SAMPLES_PER_CYCLE = 16.0


@dataclass(frozen=True)
class SerrodyneDitherSettings:
    """Requested physical parameters for one repeating dither ARB record."""

    center_frequency_hz: float = 250_000.0
    frequency_deviation_hz: float = 8_000.0
    dither_frequency_hz: float = 7.3
    v_pi: float = 1.2
    dither_periods: int = 1
    optical_delay_s: float | None = None


@dataclass(frozen=True)
class SerrodyneDitherWaveform:
    """Phase-accumulated serrodyne samples and their AFG1062 settings."""

    settings: SerrodyneDitherSettings
    time_s: np.ndarray
    voltage_v: np.ndarray
    instantaneous_frequency_hz: np.ndarray
    dac_codes: np.ndarray
    arb_repetition_hz: float
    effective_sample_rate_sps: float
    record_duration_s: float
    total_waveform_vpp: float
    afg_offset_v: float
    minimum_frequency_hz: float
    maximum_frequency_hz: float
    samples_per_max_frequency_cycle: float
    phase_dither_amplitude_rad: float | None
    accumulated_cycles: float
    boundary_phase_error_rad: float
    record_end_v: float
    record_start_v: float
    record_wrap_jump_v: float

    @property
    def point_count(self) -> int:
        return int(self.voltage_v.size)


def _validate_settings(settings: SerrodyneDitherSettings) -> None:
    center_hz = float(settings.center_frequency_hz)
    deviation_hz = float(settings.frequency_deviation_hz)
    dither_hz = float(settings.dither_frequency_hz)
    v_pi = float(settings.v_pi)

    if not math.isfinite(center_hz) or center_hz <= 0.0:
        raise ValueError("Center sawtooth frequency f_Q must be greater than zero.")
    if not math.isfinite(deviation_hz) or deviation_hz < 0.0:
        raise ValueError("Frequency deviation must be finite and non-negative.")
    if deviation_hz >= center_hz:
        raise ValueError(
            "Frequency deviation must be smaller than f_Q so the instantaneous "
            "sawtooth frequency remains positive."
        )
    if not math.isfinite(dither_hz) or dither_hz <= 0.0:
        raise ValueError("Dither frequency f_d must be greater than zero.")
    if not math.isfinite(v_pi) or v_pi == 0.0:
        raise ValueError(
            "Signed V_pi must be finite and non-zero. Use a positive value for "
            "a rising sawtooth or a negative value for a falling sawtooth."
        )
    if (
        isinstance(settings.dither_periods, bool)
        or int(settings.dither_periods) != settings.dither_periods
        or int(settings.dither_periods) < 1
    ):
        raise ValueError("The number of dither periods must be a positive integer.")
    if settings.optical_delay_s is not None:
        optical_delay_s = float(settings.optical_delay_s)
        if not math.isfinite(optical_delay_s) or optical_delay_s < 0.0:
            raise ValueError("Optical delay tau must be finite and non-negative.")

    arb_repetition_hz = dither_hz / int(settings.dither_periods)
    if arb_repetition_hz < AFG1062_MIN_ARB_REPETITION_HZ:
        raise ValueError(
            "The required ARB repetition frequency is below the AFG1062 1 microhertz limit."
        )
    if arb_repetition_hz > AFG1062_MAX_ARB_REPETITION_HZ:
        raise ValueError(
            "The required ARB repetition frequency exceeds the AFG1062 30 MHz limit."
        )


def _choose_point_count(settings: SerrodyneDitherSettings) -> int:
    """Return the smallest four-point-aligned record with adequate phase density."""

    periods = int(settings.dither_periods)
    dither_hz = float(settings.dither_frequency_hz)
    maximum_frequency_hz = (
        float(settings.center_frequency_hz) + float(settings.frequency_deviation_hz)
    )
    arb_repetition_hz = dither_hz / periods
    minimum_sample_rate_sps = (
        SERRODYNE_MIN_SAMPLES_PER_CYCLE * maximum_frequency_hz
    )
    if minimum_sample_rate_sps > AFG1062_MAX_SAMPLE_RATE_SPS:
        raise ValueError(
            "The highest instantaneous sawtooth frequency requires more than the "
            "AFG1062 300 MS/s sample-rate limit at 16 samples per cycle."
        )

    point_count = max(4, math.ceil(minimum_sample_rate_sps / arb_repetition_hz))
    point_count += (-point_count) % 4
    if point_count > AFG1062_MAX_ARB_POINTS:
        available_density = (
            AFG1062_MAX_ARB_POINTS * arb_repetition_hz / maximum_frequency_hz
        )
        raise ValueError(
            "The requested dither record needs "
            f"{point_count:,} points for 16 samples/cycle, exceeding the AFG1062 "
            f"{AFG1062_MAX_ARB_POINTS:,}-point memory (only "
            f"{available_density:.3g} samples/cycle would be available). Reduce the "
            "number of dither periods or the sawtooth frequency."
        )
    if point_count * arb_repetition_hz > AFG1062_MAX_SAMPLE_RATE_SPS + 1e-6:
        raise ValueError("The generated waveform exceeds the AFG1062 300 MS/s limit.")
    return point_count


def generate_serrodyne_dither(
    settings: SerrodyneDitherSettings,
) -> SerrodyneDitherWaveform:
    """Generate a frequency-dithered serrodyne waveform with a phase accumulator.

    The sinusoidal dither changes only the per-sample phase increment. The
    output voltage is derived solely from the wrapped accumulated phase and
    therefore contains no additively mixed low-frequency voltage sine.
    """

    _validate_settings(settings)
    center_hz = float(settings.center_frequency_hz)
    deviation_hz = float(settings.frequency_deviation_hz)
    dither_hz = float(settings.dither_frequency_hz)
    v_pi = float(settings.v_pi)
    v_pi_magnitude = abs(v_pi)
    periods = int(settings.dither_periods)

    point_count = _choose_point_count(settings)
    arb_repetition_hz = dither_hz / periods
    record_duration_s = periods / dither_hz
    effective_sample_rate_sps = point_count * arb_repetition_hz
    sample_index = np.arange(point_count, dtype=np.float64)
    time_s = sample_index / effective_sample_rate_sps
    instantaneous_frequency_hz = center_hz + deviation_hz * np.sin(
        TWO_PI * dither_hz * time_s
    )

    # theta[n] is the accumulated phase at the start of sample interval n.
    # Summing the instantaneous phase increments implements
    # theta(t) = 2*pi*integral(f_saw(t) dt) directly.
    phase_increment_rad = (
        TWO_PI * instantaneous_frequency_hz / effective_sample_rate_sps
    )
    theta_rad = np.empty(point_count, dtype=np.float64)
    theta_rad[0] = 0.0
    theta_rad[1:] = np.cumsum(phase_increment_rad[:-1], dtype=np.float64)
    wrapped_phase_rad = np.remainder(theta_rad, TWO_PI) - math.pi
    voltage_v = (v_pi / math.pi) * wrapped_phase_rad

    # Normalize the requested voltage rather than the unsigned wrapped phase.
    # This reverses the DAC codes when signed V_pi is negative, while the AFG
    # amplitude command remains the required positive peak-to-peak magnitude.
    normalized = (voltage_v + v_pi_magnitude) / (2.0 * v_pi_magnitude)
    dac_codes = np.rint(normalized * AFG1062_DAC_MAX_CODE)
    dac_codes = np.clip(dac_codes, 0, AFG1062_DAC_MAX_CODE).astype(np.uint16)

    phase_at_record_end_rad = float(np.sum(phase_increment_rad, dtype=np.float64))
    accumulated_cycles = phase_at_record_end_rad / TWO_PI
    boundary_phase_error_rad = (
        (phase_at_record_end_rad + math.pi) % TWO_PI - math.pi
    )
    record_end_wrapped_phase_rad = (
        phase_at_record_end_rad % TWO_PI - math.pi
    )
    record_end_v = (v_pi / math.pi) * record_end_wrapped_phase_rad
    record_start_v = -v_pi
    record_wrap_jump_v = record_start_v - record_end_v

    phase_dither_amplitude_rad = None
    if settings.optical_delay_s is not None:
        phase_dither_amplitude_rad = (
            TWO_PI * float(settings.optical_delay_s) * deviation_hz
        )

    for array in (time_s, voltage_v, instantaneous_frequency_hz, dac_codes):
        array.setflags(write=False)

    maximum_frequency_hz = center_hz + deviation_hz
    return SerrodyneDitherWaveform(
        settings=settings,
        time_s=time_s,
        voltage_v=voltage_v,
        instantaneous_frequency_hz=instantaneous_frequency_hz,
        dac_codes=dac_codes,
        arb_repetition_hz=arb_repetition_hz,
        effective_sample_rate_sps=effective_sample_rate_sps,
        record_duration_s=record_duration_s,
        total_waveform_vpp=2.0 * v_pi_magnitude,
        afg_offset_v=0.0,
        minimum_frequency_hz=center_hz - deviation_hz,
        maximum_frequency_hz=maximum_frequency_hz,
        samples_per_max_frequency_cycle=(
            effective_sample_rate_sps / maximum_frequency_hz
        ),
        phase_dither_amplitude_rad=phase_dither_amplitude_rad,
        accumulated_cycles=accumulated_cycles,
        boundary_phase_error_rad=boundary_phase_error_rad,
        record_end_v=record_end_v,
        record_start_v=record_start_v,
        record_wrap_jump_v=record_wrap_jump_v,
    )
