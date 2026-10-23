"""Wall-clock temperature sine setpoints for a slow TEC."""
from dataclasses import dataclass
import math
from tec_temperature_sweep import TecTemperatureSweep


@dataclass(frozen=True)
class TecSineSettings:
    center: float = 25.0
    amplitude: float = 1.0
    frequency_hz: float = 0.001
    interval_seconds: float = 5.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.center, self.amplitude, self.frequency_hz, self.interval_seconds)):
            raise ValueError("Sine settings must be finite")
        if self.amplitude < 0 or self.frequency_hz <= 0 or self.interval_seconds < 1:
            raise ValueError("Amplitude must be non-negative; frequency must be positive; update interval must be at least 1 s")
        if self.minimum < -273.15:
            raise ValueError("The temperature sine must stay above absolute zero")
        if self.amplitude and self.frequency_hz * self.interval_seconds > 1 / 20 + 1e-12:
            raise ValueError("Choose a lower frequency or shorter update interval for at least 20 setpoints per period")

    @property
    def minimum(self):
        return self.center - self.amplitude

    @property
    def maximum(self):
        return self.center + self.amplitude

    @property
    def period_seconds(self):
        return 1 / self.frequency_hz

    def target_at(self, elapsed_seconds):
        if not math.isfinite(elapsed_seconds):
            raise ValueError("Elapsed time must be finite")
        phase = (elapsed_seconds % self.period_seconds) / self.period_seconds
        return self.center + self.amplitude * math.sin(math.tau * phase)


@dataclass(frozen=True)
class TecSineSweep:
    sine: TecSineSettings
    offsets: TecTemperatureSweep

    def __post_init__(self):
        if self.minimum < -273.15:
            raise ValueError("The offset sweep minus AC amplitude must stay above absolute zero")

    @property
    def minimum(self):
        return self.offsets.minimum - self.sine.amplitude

    @property
    def maximum(self):
        return self.offsets.maximum + self.sine.amplitude

    @property
    def interval_seconds(self):
        return self.sine.interval_seconds

    @property
    def duration_seconds(self):
        return self.offsets.duration_seconds

    def target_at(self, elapsed_seconds):
        return self.offsets.target_at(elapsed_seconds) + self.sine.target_at(elapsed_seconds) - self.sine.center


def sine_program_metadata(program, elapsed):
    sine = program.sine if isinstance(program, TecSineSweep) else program
    offset = program.offsets.target_at(elapsed) if isinstance(program, TecSineSweep) else sine.center
    result = {"tec_dc_offset_deg_c": str(offset), "tec_sine_amplitude_deg_c": str(sine.amplitude),
              "tec_sine_frequency_hz": str(sine.frequency_hz), "tec_sine_elapsed_seconds": str(elapsed)}
    if isinstance(program, TecSineSweep):
        sweep = program.offsets
        result.update(tec_sweep_step=str(sweep.step_at(elapsed) + 1), tec_sweep_steps=str(sweep.step_count),
                      tec_sweep_repeat=str(sweep.repeat_at(elapsed) + 1), tec_sweep_repeats=str(sweep.repeat_count),
                      tec_dc_min_deg_c=str(sweep.minimum), tec_dc_max_deg_c=str(sweep.maximum),
                      tec_dc_increment_deg_c=str(sweep.increment), tec_dc_dwell_seconds=str(sweep.step_seconds))
    return result
