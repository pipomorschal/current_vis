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
    reference: SignalData | None = None
    channels: dict[str, SignalData] = field(default_factory=dict)

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        self.amplitude = np.asarray(self.amplitude)

    @property
    def n_samples(self) -> int:
        return int(self.amplitude.size)

    def channel_traces(self) -> dict[str, SignalData]:
        """Primary, additional captured channels and the legacy reference, once each."""
        label = self.metadata.get("channel") or self.column_names[1] or "Primary"
        traces = {label: self, **self.channels}
        if self.reference is not None and all(self.reference is not trace for trace in traces.values()):
            ref_label = self.reference.metadata.get("channel") or self.reference.column_names[1] or "Reference"
            if ref_label in traces:
                ref_label += " (reference)"
            traces[ref_label] = self.reference
        return traces

    @property
    def duration(self) -> float:
        if self.time.size < 2:
            return 0.0
        return float(self.time[-1] - self.time[0])
