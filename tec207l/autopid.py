"""Ziegler-Nichols auto-PID tuning for the SenseFuture TEC207L.

Classic ultimate-gain method: run the loop with P-only control (I = D = 0),
ramp the proportional gain until the measured temperature oscillates around
the setpoint with a stable period, then compute PID parameters from the
ultimate gain Ku and the oscillation period Tu:

    Kp = 0.6 * Ku
    Ki = 1.2 * Ku / Tu        (Ti = Tu / 2)
    Kd = 0.075 * Ku * Tu      (Td = Tu / 8)

The channel's current setpoint (read from the device) is used as the test
point - set the setpoint beforehand if you want the oscillation at a
different temperature. During the oscillation search the loop runs P-only
(I = D = 0): integral action would wind up and destroy the limit cycle the
method needs to measure Ku and Tu.
"""

from __future__ import annotations

import time

P_START = 2000           # first P gain tried (raw device units)
P_MAX = 200_000          # stop ramping here
P_FACTOR = 1.6           # multiply P by this when no oscillation yet
HYSTERESIS_C = 0.05      # dead band around the setpoint for zero-crossings
MIN_PERIODS = 3          # periods needed for a stable Tu
PERIOD_TOLERANCE = 0.25  # allowed spread of the last MIN_PERIODS periods
SAMPLE_S = 0.1           # temperature sampling interval
SETTLE_S = 5.0           # wait after a P change before detecting oscillations
MAX_WINDOW_S = 90.0      # max observation time per P gain


class AutopidError(Exception):
    """Raised when auto-PID cannot find a sustained oscillation."""


class Aborted(Exception):
    """Raised internally when the user stops auto-PID."""


def ziegler_nichols(ku: float, tu: float) -> tuple[int, int, int]:
    """Classic Ziegler-Nichols PID settings, clamped to the device range 0..9e6."""
    kp = 0.6 * ku
    ki = 1.2 * ku / tu
    kd = 0.075 * ku * tu
    clamp = lambda v: max(0, min(9_000_000, round(v)))  # noqa: E731
    return clamp(kp), clamp(ki), clamp(kd)


class OscillationDetector:
    """Detects upward zero-crossings of (temp - setpoint) with hysteresis."""

    def __init__(self, setpoint: float, hysteresis: float = HYSTERESIS_C):
        self.setpoint = setpoint
        self.hysteresis = hysteresis
        self._sign: int | None = None
        self._last_up: float | None = None

    def feed(self, temp: float, t: float) -> float | None:
        """Feed one sample; returns the period [s] when a full period completed."""
        e = temp - self.setpoint
        if e > self.hysteresis:
            sign = 1
        elif e < -self.hysteresis:
            sign = -1
        else:
            sign = self._sign if self._sign is not None else 0
        period = None
        if sign > 0 and self._sign is not None and self._sign <= 0:
            if self._last_up is not None:
                period = t - self._last_up
            self._last_up = t
        if sign != 0:
            self._sign = sign
        return period


def _check_stop(should_stop) -> None:
    if should_stop():
        raise Aborted()


def _measure(dev, channel: int, setpoint: float, should_stop, on_step,
             settle_s: float, max_window_s: float) -> float | None:
    """Watch the temperature for the current P gain; return mean period Tu or None."""
    det = OscillationDetector(setpoint)
    periods: list[float] = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < settle_s:  # let the loop settle after the P change
        _check_stop(should_stop)
        time.sleep(SAMPLE_S)
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_window_s:
        _check_stop(should_stop)
        temp = dev.get_actual_temp(channel)
        period = det.feed(temp, time.monotonic())
        if period is not None:
            periods.append(period)
            on_step(f"step 3/5: oscillation period {period:.1f} s ({len(periods)} measured)")
            if len(periods) >= MIN_PERIODS:
                recent = periods[-MIN_PERIODS:]
                mean = sum(recent) / len(recent)
                if max(recent) - min(recent) <= PERIOD_TOLERANCE * mean:
                    return mean
        time.sleep(SAMPLE_S)
    return None


def _restore(dev, channel: int, saved: dict) -> None:
    """Best-effort restore of the configuration saved at the start."""
    try:
        p, i, d = saved["pid"]
        dev.set_pid(channel, p, i, d)
        dev.set_target_temp(channel, saved["target"])
        dev.set_output_mode(channel, saved["mode"])
        dev.set_max_output(channel, saved["limit"])
        dev.set_enable(channel, saved["enable"])
    except Exception:
        pass  # device may be gone; nothing more we can do


def run_autopid(dev, channel: int, *,
                on_step=lambda msg: None, should_stop=lambda: False,
                p_start: int = P_START, p_max: int = P_MAX,
                settle_s: float = SETTLE_S, max_window_s: float = MAX_WINDOW_S) -> dict:
    """Run Ziegler-Nichols auto-PID on one channel. Blocking - call in a thread.

    dev: a Tec207L instance (connected). on_step(msg): progress callback.
    should_stop(): returns True when the user aborts.

    The channel's current setpoint is read from the device and used as the
    test point (the setpoint is NOT changed). Returns {"ok": True, "ku", "tu",
    "kp", "ki", "kd", "saved"} on success, {"ok": False, "aborted": True,
    "saved"} when aborted (previous configuration restored), or raises
    AutopidError (previous configuration restored as well).
    """
    saved = dev.get_full_state(channel)
    setpoint = saved["target"]
    on_step("step 1/5: saved current configuration")
    try:
        _check_stop(should_stop)
        dev.set_output_mode(channel, 0)  # cool & heat
        dev.set_enable(channel, True)
        on_step(f"step 2/5: output enabled (cool & heat), tuning around the "
                f"current setpoint {setpoint:.3f} C")

        ku = tu = None
        p = p_start
        while ku is None:
            _check_stop(should_stop)
            dev.set_pid(channel, p, 0, 0)
            on_step(f"step 3/5: oscillation detection at P={p} (I=0, D=0)")
            tu = _measure(dev, channel, setpoint, should_stop, on_step,
                          settle_s, max_window_s)
            if tu is not None:
                ku = p
                break
            if p >= p_max:
                raise AutopidError(f"no sustained oscillation found up to P={p_max}")
            p = min(p_max, int(p * P_FACTOR))

        kp, ki, kd = ziegler_nichols(ku, tu)
        on_step(f"step 4/5: Ku={ku}, Tu={tu:.1f} s -> Z-N PID: P={kp} I={ki} D={kd}")
        dev.set_pid(channel, kp, ki, kd)
        on_step("step 5/5: PID applied (setpoint left unchanged)")
        return {"ok": True, "aborted": False, "ku": ku, "tu": tu,
                "kp": kp, "ki": ki, "kd": kd, "saved": saved}
    except Aborted:
        _restore(dev, channel, saved)
        return {"ok": False, "aborted": True, "saved": saved}
    except Exception:
        _restore(dev, channel, saved)
        raise
