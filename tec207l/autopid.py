"""Bounded Ziegler-Nichols search for slow thermal plants.

P-only control can have a steady offset. Detect cycles between local extrema,
not crossings of the target. Require stable amplitudes as well as periods.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
from .device import Tec207L
from .protocol import ProtocolError

P_START, P_MAX, P_FACTOR = 2000, 200_000, 1.6
HYSTERESIS_C, MIN_PERIODS, PERIOD_TOLERANCE = .005, 3, .25
SAMPLE_S, SETTLE_S, MAX_WINDOW_S = .5, 30., 600.


class AutopidError(Exception):
    """No usable linear ultimate gain, or a tuning precondition failed."""


class Aborted(Exception):
    pass


@dataclass(frozen=True)
class AutopidSettings:
    p_start: int = P_START
    p_max: int = P_MAX
    settle_s: float = SETTLE_S
    max_window_s: float = MAX_WINDOW_S
    hysteresis: float = HYSTERESIS_C
    sample_s: float = SAMPLE_S
    min_period_s: float = 5.
    max_deviation: float = 5.

    def __post_init__(self):
        if (type(self.p_start) is not int or type(self.p_max) is not int
                or not 1 <= self.p_start <= self.p_max <= 9_000_000):
            raise ValueError("ZN requires 1 <= starting P <= maximum P <= 9000000")
        values = (self.settle_s, self.max_window_s, self.hysteresis,
                  self.sample_s, self.min_period_s, self.max_deviation)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("ZN settings must be finite")
        if self.settle_s < 0 or min(values[1:]) <= 0 or self.max_deviation <= 2 * self.hysteresis:
            raise ValueError("ZN timing, reversal threshold and temperature excursion must be positive")
        if self.max_window_s < (MIN_PERIODS + 1) * max(self.min_period_s, 8 * self.sample_s):
            raise ValueError("ZN observation window must allow four minimum-length cycles (8 samples per cycle)")


def ziegler_nichols(ku, tu):
    if not math.isfinite(ku) or not math.isfinite(tu) or ku <= 0 or tu <= 0:
        raise ValueError("Ultimate gain and period must be finite and positive")
    clamp = lambda value: max(0, min(9_000_000, round(value)))
    return clamp(.6 * ku), clamp(1.2 * ku / tu), clamp(.075 * ku * tu)


class OscillationDetector:
    """Hysteretic extrema tracking, independent of the P-only steady offset."""
    def __init__(self, setpoint, hysteresis=HYSTERESIS_C):
        self.setpoint, self.hysteresis = setpoint, hysteresis
        self.direction = 0
        self.high = self.low = self.last_peak = self.trough = None
        self.cycles, self.times = [], []

    def feed(self, temp, t):
        if not math.isfinite(temp) or not math.isfinite(t):
            raise AutopidError("Non-finite tuning temperature or timestamp")
        if self.times and t <= self.times[-1]:
            raise AutopidError("Tuning sample timestamps must increase")
        point = (temp, t, len(self.times))
        self.times.append(t)
        if self.high is None:
            self.high = self.low = point
            return None
        if self.direction == 0:
            self.high = max(self.high, point, key=lambda item: item[0])
            self.low = min(self.low, point, key=lambda item: item[0])
            if temp - self.low[0] >= 2 * self.hysteresis:
                self.direction, self.high = 1, point
            elif self.high[0] - temp >= 2 * self.hysteresis:
                # A peak observed after the first sample is already complete,
                # even if its initial rise was smaller than the reversal band.
                if self.high[2] > 0:
                    self.last_peak = self.high
                self.direction, self.low = -1, point
            return None
        if self.direction > 0:
            if temp > self.high[0]:
                self.high = point
            elif self.high[0] - temp >= self.hysteresis:
                peak, self.direction, self.low = self.high, -1, point
                period = None
                if self.last_peak is not None and self.trough is not None:
                    period = peak[1] - self.last_peak[1]
                    ts = self.times[self.last_peak[2]:peak[2] + 1]
                    self.cycles.append(dict(period_s=period,
                        peak_to_peak_C=min(peak[0], self.last_peak[0]) - self.trough[0],
                        samples=peak[2] - self.last_peak[2],
                        max_gap_s=max(b - a for a, b in zip(ts, ts[1:]))))
                self.last_peak = peak
                return period
        else:
            if temp < self.low[0]:
                self.low = point
            elif temp - self.low[0] >= self.hysteresis:
                self.trough, self.direction, self.high = self.low, 1, point
        return None

    def assessment(self, min_period_s=5.):
        recent = self.cycles[-MIN_PERIODS:]
        if len(recent) < MIN_PERIODS:
            return None, f"only {len(recent)}/{MIN_PERIODS} full periods detected; increase observation time for slow cycles"
        periods, amplitudes = ([c[key] for c in recent] for key in ("period_s", "peak_to_peak_C"))
        mean_period, mean_amplitude = sum(periods) / len(recent), sum(amplitudes) / len(recent)
        if any(c["samples"] < 8 or c["max_gap_s"] > c["period_s"] / 4 for c in recent):
            return None, "cycles are undersampled; shorten sample interval or reduce serial contention"
        if min(periods) < min_period_s:
            return None, "cycles are too fast; likely sensor noise rather than thermal oscillation"
        if min(amplitudes) < 2 * self.hysteresis:
            return None, "oscillation amplitude is below the detection threshold"
        if max(periods) - min(periods) > PERIOD_TOLERANCE * mean_period:
            return None, "oscillation periods are not stable"
        if max(amplitudes) - min(amplitudes) > PERIOD_TOLERANCE * mean_amplitude:
            return None, "oscillation amplitude is decaying or growing, not sustained"
        return mean_period, "stable periods and amplitudes"


def _check_stop(should_stop):
    if should_stop():
        raise Aborted()


def _wait(seconds, should_stop):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        _check_stop(should_stop)
        time.sleep(min(.1, max(0., deadline - time.monotonic())))


class _Envelope:
    """Read-only trial monitoring; never raises a channel's ceiling."""
    def __init__(self, dev, channel, target, ceiling, max_deviation, on_step):
        self.dev, self.channel, self.target = dev, channel, target
        self.ceiling, self.max_deviation, self.on_step = ceiling, max_deviation, on_step
        self.checked_at = -math.inf
        self.current = self.output_percent = None
        self.output_supported = hasattr(dev, "get_output_percent")

    def check(self, temp, now):
        if not math.isfinite(temp) or abs(temp - self.target) > self.max_deviation:
            raise AutopidError(f"CH{self.channel} temperature {temp!r} exceeds the allowed +/-{self.max_deviation:g} C excursion")
        if now - self.checked_at < 5.:
            return
        actual = Tec207L.validate_max_output(self.dev.get_max_output(self.channel))
        previous = self.ceiling
        self.ceiling = min(self.ceiling, actual)
        if actual != previous:
            raise AutopidError(f"CH{self.channel} output limit changed from {previous}% to {actual}% Vin; tuning stopped")
        if self.dev.get_control_mode() != 0:
            raise AutopidError("Controller left independent-channel mode (CONTMODE=0)")
        if not self.dev.get_enable(self.channel):
            raise AutopidError(f"CH{self.channel} output is disabled; temperature response cannot identify a gain")
        if abs(self.dev.get_target_temp(self.channel) - self.target) > .00001:
            raise AutopidError(f"CH{self.channel} setpoint changed during tuning")
        self.current = self.dev.get_output_current(self.channel)
        if not math.isfinite(self.current):
            raise AutopidError("Invalid output-current reading")
        if self.output_supported:
            try:
                self.output_percent = self.dev.get_output_percent(self.channel)
                if not math.isfinite(self.output_percent):
                    raise AutopidError("Invalid PWMDUTY reading")
            except ProtocolError as exc:
                self.output_supported = False
                self.on_step(f"PWMDUTY diagnostic unavailable ({exc}); hardware LIMITED ceiling remains monitored")
        self.checked_at = now


def _measure(dev, channel, setpoint, should_stop, on_step, settle_s, max_window_s, *,
             hysteresis=HYSTERESIS_C, sample_s=SAMPLE_S, min_period_s=5., guard=None,
             gain=0, on_sample=lambda row: None, on_trial=lambda row: None, run_started=None):
    detector = OscillationDetector(setpoint, hysteresis)
    start = time.monotonic()
    run_started = start if run_started is None else run_started
    temperatures, clipped = [], []
    reason, status, tu = "no repeated extrema", "incomplete", None
    last_report = -math.inf
    try:
        while time.monotonic() - start < settle_s + max_window_s:
            _check_stop(should_stop)
            temp = dev.get_actual_temp(channel)
            now = time.monotonic()  # after serial read, not a nominal timer tick
            if guard:
                guard.check(temp, now)
            elapsed, phase = now - start, "settling" if now - start < settle_s else "observing"
            on_sample(dict(elapsed_s=now - run_started, gain_p=gain, phase=phase, target_C=setpoint,
                           temperature_C=temp, limit_percent=guard.ceiling if guard else None,
                           current_A=guard.current if guard else None,
                           output_percent=guard.output_percent if guard else None))
            if phase == "observing":
                temperatures.append(temp)
                if guard and guard.output_percent is not None:
                    clipped.append(abs(guard.output_percent) >= .98 * guard.ceiling)
                period = detector.feed(temp, now)
                tu, reason = detector.assessment(min_period_s)
                if period is not None:
                    on_step(f"CH{channel} P={gain}: period {period:.1f} s; {reason}")
                if tu is not None:
                    if clipped and sum(clipped) / len(clipped) >= .2:
                        raise AutopidError("Oscillations hit the configured output ceiling; cannot identify an unsaturated ultimate gain. "
                                           "The output limit will not be increased")
                    status = "stable"
                    break
            if elapsed - last_report >= 10.:
                on_step(f"CH{channel} P={gain}: {phase}, {elapsed:.0f}/{settle_s + max_window_s:.0f} s; "
                        f"T={temp:.5f} C; {reason}")
                last_report = elapsed
            _wait(sample_s, should_stop)
        if tu is None:
            status = "not_stable"
            if temperatures and max(temperatures) - min(temperatures) < 2 * hysteresis:
                reason = f"temperature variation below {2 * hysteresis:g} C; no measurable excitation at this gain"
            if clipped and sum(clipped) / len(clipped) >= .5:
                raise AutopidError("Output is predominantly at the configured ceiling; the working point may be unreachable "
                                   "with the allowed output. Tuning stopped without increasing the limit")
        return tu
    except Aborted:
        status, reason = "cancelled", "user stopped tuning"
        raise
    except Exception as exc:
        status, reason = "error", str(exc)
        raise
    finally:
        summary = dict(gain_p=gain, status=status, reason=reason, samples=len(temperatures),
                       peak_to_peak_C=max(temperatures) - min(temperatures) if temperatures else None,
                       mean_error_C=sum(temperatures) / len(temperatures) - setpoint if temperatures else None,
                       cycles=detector.cycles[-MIN_PERIODS:],
                       clipped_fraction=sum(clipped) / len(clipped) if clipped else None)
        on_trial(summary)
        on_step(f"CH{channel} P={gain}: {reason}")


def _restore(dev, channel, saved, ceiling):
    """Restore configuration with output OFF, without raising an observed limit."""
    errors = []
    def attempt(name, action):
        try:
            action()
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    def off():
        dev.set_enable(channel, False)
        if dev.get_enable(channel):
            raise AutopidError("Output OFF was not confirmed")
    attempt("output OFF", off)
    try:
        ceiling = min(ceiling, Tec207L.validate_max_output(dev.get_max_output(channel)))
    except Exception as exc:
        errors.append(f"limit readback: {exc}")
    limit = min(saved["limit"], ceiling)
    def restore_limit():
        dev.set_max_output(channel, limit)
        if dev.get_max_output(channel) != limit:
            raise AutopidError("Output limit restore was not confirmed")
    attempt("limit", restore_limit)
    def restore_pid():
        if tuple(dev.set_pid(channel, *saved["pid"])) != tuple(saved["pid"]):
            raise AutopidError("PID restore was not confirmed")
    attempt("PID", restore_pid)
    attempt("setpoint", lambda: dev.set_target_temp(channel, saved["target"]))
    attempt("mode", lambda: dev.set_output_mode(channel, saved["mode"]))
    if errors:
        raise AutopidError("; ".join(errors))


def run_autopid(dev, channel, *, on_step=lambda msg: None, should_stop=lambda: False,
                p_start=P_START, p_max=P_MAX, settle_s=SETTLE_S, max_window_s=MAX_WINDOW_S,
                hysteresis=HYSTERESIS_C, sample_s=SAMPLE_S, min_period_s=5., max_deviation=5.,
                output_limit=None, on_sample=lambda row: None, on_trial=lambda row: None):
    settings = AutopidSettings(p_start, p_max, settle_s, max_window_s, hysteresis, sample_s, min_period_s, max_deviation)
    if should_stop():
        return dict(ok=False, aborted=True, trials=[])
    saved = dev.get_full_state(channel)
    ceiling = min(Tec207L.validate_max_output(saved["limit"]),
                  Tec207L.validate_max_output(saved["limit"] if output_limit is None else output_limit))
    if ceiling == 0:
        raise AutopidError(f"CH{channel} output limit is 0%; tuning needs excitation and will not raise the limit")
    mode = dev.get_control_mode()
    if mode != 0:
        raise AutopidError(f"CONTMODE={mode}: independent-channel mode (0) is required; CH2 can follow CH1 in mode 2/3")
    if dev.get_autotune_state(channel) != 0:
        raise AutopidError(f"CH{channel} device AUTOPID is active; stop device tuning before a Ziegler-Nichols search")
    if (not all(math.isfinite(saved[key]) for key in ("actual", "target"))
            or abs(saved["actual"] - saved["target"]) > settings.max_deviation):
        raise AutopidError("Start closer to the setpoint, or configure the permitted temperature excursion")
    guard = _Envelope(dev, channel, saved["target"], ceiling, settings.max_deviation, on_step)
    trials, started = [], time.monotonic()
    def trial(row):
        trials.append(row)
        on_trial(row)
    try:
        dev.set_max_output(channel, ceiling)
        if dev.get_max_output(channel) != ceiling:
            raise AutopidError("Output limit was not confirmed before tuning")
        dev.set_output_mode(channel, 0)
        if tuple(dev.set_pid(channel, p_start, 0, 0)) != (p_start, 0, 0):
            raise AutopidError("Initial P-only PID write was not confirmed")
        dev.set_enable(channel, True)
        on_step(f"CH{channel}: bounded ZN search; ceiling {ceiling}% Vin, P {p_start}–{p_max}; "
                f"settle {settle_s:g} s, observe {max_window_s:g} s per gain")
        p = p_start
        while True:
            _check_stop(should_stop)
            guard.checked_at = -math.inf
            guard.check(dev.get_actual_temp(channel), time.monotonic())
            if tuple(dev.set_pid(channel, p, 0, 0)) != (p, 0, 0):
                raise AutopidError(f"CH{channel} P-only PID write was not confirmed")
            tu = _measure(dev, channel, saved["target"], should_stop, on_step, settle_s, max_window_s,
                          hysteresis=hysteresis, sample_s=sample_s, min_period_s=min_period_s,
                          guard=guard, gain=p, on_sample=on_sample, on_trial=trial, run_started=started)
            if tu is not None:
                break
            if p >= p_max:
                last = trials[-1]["reason"] if trials else "insufficient oscillation data"
                raise AutopidError(f"CH{channel}: no stable oscillation up to P={p_max}; {last}. "
                                   "A limited-output thermal plant may not have a reachable ZN ultimate gain; "
                                   "use device auto-tune or a step response. Output ceiling unchanged")
            p = min(p_max, max(p + 1, int(p * P_FACTOR)))
        guard.checked_at = -math.inf
        guard.check(dev.get_actual_temp(channel), time.monotonic())
        kp, ki, kd = ziegler_nichols(p, tu)
        if tuple(dev.set_pid(channel, kp, ki, kd)) != (kp, ki, kd):
            raise AutopidError("Calculated PID write was not confirmed")
        on_step(f"CH{channel}: Ku={p}, Tu={tu:.2f} s; applied PID {kp}/{ki}/{kd}; ceiling {guard.ceiling}% Vin")
        return dict(ok=True, aborted=False, ku=p, tu=tu, kp=kp, ki=ki, kd=kd, saved=saved, trials=trials)
    except Exception as exc:
        try:
            _restore(dev, channel, saved, guard.ceiling)
        except Exception as cleanup:
            raise AutopidError(f"{exc or 'Tuning cancelled'}; cleanup failed: {cleanup}") from exc
        if isinstance(exc, Aborted):
            return dict(ok=False, aborted=True, saved=saved, trials=trials)
        raise
