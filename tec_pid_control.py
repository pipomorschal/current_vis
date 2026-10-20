"""Bounded extremum seeking: estimate a harmonic-score gradient and drive it to zero."""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np
from scipy.signal import resample_poly

from recording_analysis import recording_amplitudes
from signal_data_class import SignalData


@dataclass(frozen=True)
class TecPidSettings:
    minimum: float = 20.0
    maximum: float = 30.0
    initial: float = 25.0
    probe: float = 0.2
    max_move: float = 0.1
    settle_seconds: float = 300.0
    interval_seconds: float = 60.0
    temperature_tolerance: float = 0.1
    harmonic_weight: float = 1.0
    kp: float = 0.1
    ki: float = 0.0
    kd: float = 0.0
    gradient_deadband: float = 0.001
    minimum_a50: float = 1e-6
    max_cycles: int = 100

    def __post_init__(self):
        if not all(math.isfinite(getattr(self, f.name)) for f in fields(self)):
            raise ValueError("All controller settings must be finite")
        if self.probe <= 0 or self.maximum - self.minimum <= 2 * self.probe:
            raise ValueError("Temperature range must be wider than two positive probe steps")
        if not self.minimum + self.probe <= self.initial <= self.maximum - self.probe:
            raise ValueError("Starting temperature must leave room for a probe in both directions")
        if min(self.max_move, self.settle_seconds, self.interval_seconds,
               self.temperature_tolerance, self.minimum_a50) <= 0:
            raise ValueError("Step limit, settling time, interval, tolerance and minimum A50 must be positive")
        if min(self.harmonic_weight, self.kp, self.ki, self.kd, self.gradient_deadband) < 0:
            raise ValueError("Harmonic weight, PID gains and deadband must be non-negative")
        if isinstance(self.max_cycles, bool) or not isinstance(self.max_cycles, int) or self.max_cycles < 1:
            raise ValueError("Cycle count must be a positive integer")


def harmonic_amplitudes(data: SignalData) -> tuple[float, float]:
    """Joint least-squares peak amplitudes; anti-alias before reducing large scope records."""
    t, y, fs = data.time, data.amplitude, float(data.sampling_rate)
    if t.ndim != 1 or y.ndim != 1 or t.size != y.size or t.size < 4:
        raise ValueError("Invalid waveform dimensions")
    if not math.isfinite(fs) or fs <= 200 or data.duration < 0.2:
        raise ValueError("Use at least 0.2 s of waveform and a sample rate above 200 Hz")
    # Two seconds cover 100 fundamental cycles without fitting millions of points.
    count = min(t.size, int(math.ceil(2 * fs)) + 1)
    t, y = t[:count], y[:count]
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("Non-finite waveform")
    if not np.allclose(np.diff(t), 1 / fs, rtol=0.01, atol=1e-12):
        raise ValueError("Harmonic analysis requires uniformly sampled waveform data")
    factor = max(1, int(fs // 2000))
    if factor > 1:
        y = resample_poly(y, 1, factor)
        fs /= factor
        # Discard resampling edges before the sine fit.
        trim = min(20, y.size // 10)
        y = y[trim:-trim] if trim else y
        t = np.arange(y.size) / fs
    fitted = SignalData(t, y, sampling_rate=fs)
    a50, a100 = recording_amplitudes(fitted, [50, 100])
    return a50, a100


class GradientPid:
    """PID correction of the center temperature using dJ/dT, with conditional integration."""

    def __init__(self, settings: TecPidSettings):
        self.settings = settings
        self.center = settings.initial
        self.integral = 0.0
        self.previous_gradient = None

    def update(self, gradient: float, dt: float) -> float:
        if not math.isfinite(gradient) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Gradient and elapsed update time must be finite; time must be positive")
        s = self.settings
        if abs(gradient) <= s.gradient_deadband:
            self.integral = 0.0
            self.previous_gradient = gradient
            return self.center
        if self.previous_gradient is not None and gradient * self.previous_gradient < 0:
            self.integral = 0.0
        derivative = 0.0 if self.previous_gradient is None else (gradient - self.previous_gradient) / dt
        candidate_integral = self.integral + gradient * dt if s.ki else 0.0
        correction = s.kp * gradient + s.ki * candidate_integral + s.kd * derivative
        limited = float(np.clip(correction, -s.max_move, s.max_move))
        target = float(np.clip(self.center + limited, s.minimum + s.probe, s.maximum - s.probe))
        if math.isclose(correction, limited, abs_tol=1e-12) and math.isclose(target, self.center + limited, abs_tol=1e-12):
            self.integral = candidate_integral
        self.center = target
        self.previous_gradient = gradient
        return target


def calibration_recommendation(settings, center_score, low_score, high_score, response):
    """Local curvature suggests P; a measured 63% crossing suggests a minimum dwell."""
    curvature = (high_score - 2 * center_score + low_score) / settings.probe ** 2
    result = {"ki": 0.0, "kd": 0.0, "kp": settings.kp, "tau_seconds": None,
              "settle_seconds": settings.settle_seconds, "curvature": curvature}
    if curvature < -1e-6:
        result["kp"] = min(1.0, 0.2 / abs(curvature))
        result["note"] = "Local score curve is concave; suggested P uses a conservative fraction of inverse curvature."
    else:
        result["note"] = "No local maximum resolved; P is unchanged. Consider a wider temperature survey."
    times, values = np.asarray(response, dtype=float).T
    change = values[-1] - values[0]
    # No dynamics estimate when the response is flat, noisy or still drifting.
    threshold = max(0.002, 0.02 * max(abs(values[0]), abs(values[-1])))
    tail_change = abs(values[-1] - values[-2])
    if abs(change) > threshold and tail_change < 0.1 * abs(change):
        progress = (values - values[0]) / change
        crossings = np.flatnonzero(progress >= 1 - math.exp(-1))
        if crossings.size and crossings[0] > 0:
            i = int(crossings[0])
            tau = float(np.interp(1 - math.exp(-1), progress[i - 1:i + 1], times[i - 1:i + 1]))
            result["tau_seconds"] = tau
            result["settle_seconds"] = max(settings.settle_seconds, 5 * tau)
    if result["tau_seconds"] is None:
        result["note"] += " Response time was not resolved; increase the observation duration or reduce the sampling interval."
    result["note"] += " I and D start at zero; these are local initial estimates, not a stability guarantee."
    return result
