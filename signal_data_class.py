from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SignalData:
    time: np.ndarray
    amplitude: np.ndarray
    source_name: str = "Untitled"
    sampling_rate: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)
    column_names: tuple[str, str] = ("TIME", "AMPLITUDE")

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        self.amplitude = np.asarray(self.amplitude)

    @property
    def n_samples(self) -> int:
        return int(self.amplitude.size)

    @property
    def duration(self) -> float:
        if self.time.size < 2:
            return 0.0
        return float(self.time[-1] - self.time[0])
