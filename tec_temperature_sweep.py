"""A finite ascending temperature sweep, including a dwell at its maximum."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TecTemperatureSweep:
    minimum: float
    maximum: float
    increment: float
    step_seconds: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.minimum, self.maximum, self.increment, self.step_seconds)):
            raise ValueError("Sweep settings must be finite")
        if self.maximum < self.minimum or self.increment <= 0 or self.step_seconds <= 0:
            raise ValueError("Maximum must be at least minimum; increment and time per step must be positive")
        if self.step_count > 100000:
            raise ValueError("Temperature sweep exceeds 100000 steps")

    @property
    def step_count(self):
        return int(math.ceil(round((self.maximum - self.minimum) / self.increment, 10))) + 1

    @property
    def duration_seconds(self):
        return self.step_count * self.step_seconds

    def step_at(self, elapsed_seconds):
        return min(self.step_count - 1, max(0, int(elapsed_seconds // self.step_seconds)))

    def target_at(self, elapsed_seconds):
        return min(self.maximum, self.minimum + self.step_at(elapsed_seconds) * self.increment)
