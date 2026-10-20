"""Wall-clock temperature sine setpoints for a slow TEC."""
from dataclasses import dataclass
import math


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
        if self.frequency_hz * self.interval_seconds > 1 / 20 + 1e-12:
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
