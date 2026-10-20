"""A finite, repeatable ascending temperature sweep with a dwell at each maximum."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TecTemperatureSweep:
    minimum: float
    maximum: float
    increment: float
    step_seconds: float
    repeat_count: int = 1

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.minimum, self.maximum, self.increment, self.step_seconds)):
            raise ValueError("Sweep settings must be finite")
        if self.maximum < self.minimum or self.increment <= 0 or self.step_seconds <= 0:
            raise ValueError("Maximum must be at least minimum; increment and time per step must be positive")
        if isinstance(self.repeat_count, bool) or not isinstance(self.repeat_count, int) or self.repeat_count < 1:
            raise ValueError("Repeat count must be a positive integer")
        if self.step_count > 100000:
            raise ValueError("Temperature sweep exceeds 100000 steps")

    @property
    def steps_per_sweep(self):
        return int(math.ceil(round((self.maximum - self.minimum) / self.increment, 10))) + 1

    @property
    def step_count(self):
        return self.steps_per_sweep * self.repeat_count

    @property
    def duration_seconds(self):
        return self.step_count * self.step_seconds

    def step_at(self, elapsed_seconds):
        return min(self.step_count - 1, max(0, int(elapsed_seconds // self.step_seconds)))

    def target_at(self, elapsed_seconds):
        step_in_sweep = self.step_at(elapsed_seconds) % self.steps_per_sweep
        return min(self.maximum, self.minimum + step_in_sweep * self.increment)

    def repeat_at(self, elapsed_seconds):
        return self.step_at(elapsed_seconds) // self.steps_per_sweep
