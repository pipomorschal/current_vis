from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


AFG1062_MAX_SAMPLE_RATE_SPS = 300_000_000.0
# The AFG1062 technical reference and programmer manual specify a 1 Mpoint
# arbitrary-waveform/edit-memory record. An older series overview table still
# says 8 kpoints, but that conflicts with both detailed model-specific manuals.
AFG1062_MAX_ARB_POINTS = 1_048_576
AFG1062_MIN_ARB_REPETITION_HZ = 1e-6
AFG1062_MAX_ARB_REPETITION_HZ = 30_000_000.0
AFG1062_DAC_MAX_CODE = 16_382


@dataclass(frozen=True)
class RectangularRampSettings:
    """Requested physical parameters for one repeating ARB record."""

    modulation_frequency_hz: float = 395_000.0
    rectangular_vpp: float = 2.4
    ramp_slope_mv_per_period: float = 0.0
    periods: int = 10


@dataclass(frozen=True)
class RectangularRampWaveform:
    """Generated samples plus the AFG settings needed to reproduce them."""

    settings: RectangularRampSettings
    samples_per_period: int
    time_s: np.ndarray
    voltage_v: np.ndarray
    dac_codes: np.ndarray
    arb_repetition_hz: float
    effective_sample_rate_sps: float
    record_duration_s: float
    total_waveform_vpp: float
    afg_offset_v: float
    baseline_change_v: float
    reset_from_v: float
    reset_to_v: float
    baseline_reset_jump_v: float
    record_wrap_jump_v: float

    @property
    def point_count(self) -> int:
        return int(self.voltage_v.size)


def _validate_settings(settings: RectangularRampSettings) -> None:
    frequency_hz = float(settings.modulation_frequency_hz)
    rectangular_vpp = float(settings.rectangular_vpp)
    slope_mv = float(settings.ramp_slope_mv_per_period)

    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("Rectangular modulation frequency must be greater than zero.")
    if not math.isfinite(rectangular_vpp) or rectangular_vpp <= 0.0:
        raise ValueError("Rectangular amplitude must be a finite value greater than zero.")
    if not math.isfinite(slope_mv):
        raise ValueError("Ramp slope must be finite.")
    if isinstance(settings.periods, bool) or int(settings.periods) != settings.periods:
        raise ValueError("The ARB record period count must be an integer.")
    if not 1 <= int(settings.periods) <= AFG1062_MAX_ARB_POINTS // 4:
        raise ValueError(
            "The ARB record must contain between 1 and "
            f"{AFG1062_MAX_ARB_POINTS // 4} rectangular periods."
        )

    arb_repetition_hz = frequency_hz / int(settings.periods)
    if arb_repetition_hz < AFG1062_MIN_ARB_REPETITION_HZ:
        raise ValueError(
            "The required ARB repetition frequency is below the AFG1062 1 µHz limit."
        )
    if arb_repetition_hz > AFG1062_MAX_ARB_REPETITION_HZ:
        raise ValueError(
            "The required ARB repetition frequency exceeds the AFG1062 30 MHz limit."
        )


def _samples_per_period(settings: RectangularRampSettings) -> int:
    """Choose the densest exact-duty grid within both AFG1062 limits."""

    periods = int(settings.periods)
    frequency_hz = float(settings.modulation_frequency_hz)
    memory_limit = AFG1062_MAX_ARB_POINTS // periods
    sample_rate_limit = math.floor(AFG1062_MAX_SAMPLE_RATE_SPS / frequency_hz)
    available = min(memory_limit, sample_rate_limit)

    # Four-point alignment places both square edges on sample locations while
    # retaining precisely the same number of high and low samples per period.
    samples_per_period = available - (available % 4)
    if samples_per_period < 4:
        raise ValueError(
            "The requested modulation frequency/period count cannot be represented "
            "with a 50% duty cycle within the AFG1062 300 MS/s and "
            "1,048,576-point limits."
        )
    return samples_per_period


def generate_rectangular_ramp(
    settings: RectangularRampSettings,
) -> RectangularRampWaveform:
    """Create an AFG1062-ready rectangular waveform on a linear baseline ramp.

    The rectangular period starts high and ends low, so its normal low-to-high
    edge is aligned with the ARB record boundary. The preview can therefore
    distinguish that edge from the simultaneous intentional baseline reset.
    """

    _validate_settings(settings)
    frequency_hz = float(settings.modulation_frequency_hz)
    rectangular_vpp = float(settings.rectangular_vpp)
    periods = int(settings.periods)
    slope_v_per_period = float(settings.ramp_slope_mv_per_period) * 1e-3
    samples_per_period = _samples_per_period(settings)
    point_count = periods * samples_per_period
    effective_sample_rate_sps = samples_per_period * frequency_hz
    arb_repetition_hz = frequency_hz / periods
    record_duration_s = periods / frequency_hz
    baseline_change_v = periods * slope_v_per_period
    if not math.isfinite(baseline_change_v):
        raise ValueError("The requested ramp excursion is too large to represent.")

    sample_index = np.arange(point_count, dtype=np.float64)
    phase_index = np.remainder(sample_index, samples_per_period)
    high = phase_index < samples_per_period // 2
    half_vpp = 0.5 * rectangular_vpp
    rectangular = np.where(high, half_vpp, -half_vpp)
    baseline = slope_v_per_period * sample_index / samples_per_period
    voltage_v = rectangular + baseline

    minimum_v = float(np.min(voltage_v))
    maximum_v = float(np.max(voltage_v))
    total_waveform_vpp = maximum_v - minimum_v
    afg_offset_v = 0.5 * (maximum_v + minimum_v)
    # Use the full 14-bit DAC range for the record's actual extrema. Applying
    # that same span and center as AFG amplitude/offset reconstructs the
    # requested voltages, so the rectangular contribution remains exactly the
    # requested Vpp (apart from unavoidable DAC quantization) as the baseline moves.
    normalized = (voltage_v - minimum_v) / total_waveform_vpp
    dac_codes = np.rint(normalized * AFG1062_DAC_MAX_CODE)
    dac_codes = np.clip(dac_codes, 0, AFG1062_DAC_MAX_CODE).astype(np.uint16)

    reset_from_v = -half_vpp + baseline_change_v
    reset_to_v = half_vpp
    baseline_reset_jump_v = -baseline_change_v
    record_wrap_jump_v = reset_to_v - reset_from_v

    time_s = sample_index / effective_sample_rate_sps
    for array in (time_s, voltage_v, dac_codes):
        array.setflags(write=False)

    return RectangularRampWaveform(
        settings=settings,
        samples_per_period=samples_per_period,
        time_s=time_s,
        voltage_v=voltage_v,
        dac_codes=dac_codes,
        arb_repetition_hz=arb_repetition_hz,
        effective_sample_rate_sps=effective_sample_rate_sps,
        record_duration_s=record_duration_s,
        total_waveform_vpp=total_waveform_vpp,
        afg_offset_v=afg_offset_v,
        baseline_change_v=baseline_change_v,
        reset_from_v=reset_from_v,
        reset_to_v=reset_to_v,
        baseline_reset_jump_v=baseline_reset_jump_v,
        record_wrap_jump_v=record_wrap_jump_v,
    )
