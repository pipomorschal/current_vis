"""Step-response / settling-time test for the SenseFuture TEC207L.

For each target temperature the setpoint is changed and the actual
temperature is sampled until it stays within +/- accuracy of the target
for the whole interval (no sample may leave the band inside that window).
The time from the setpoint change until the band has been held for
`interval` seconds is the settling time.

Results are returned as a list of dicts and are meant to be logged/saved
by the caller (the GUI writes them as CSV into the logs folder).
"""

from __future__ import annotations

import time

SAMPLE_S = 0.1   # temperature sampling interval
MAX_STEP_S = 1800.0  # per-target timeout: report "not settled" after this


def run_step_test(dev, channel: int, targets: list[float], *,
                  accuracy: float = 0.01, interval: float = 20.0,
                  on_step=lambda msg: None, should_stop=lambda: False,
                  sample_s: float = SAMPLE_S,
                  max_step_s: float = MAX_STEP_S) -> dict:
    """Run the step test on one channel. Blocking - call in a thread.

    dev: connected Tec207L. targets: list of setpoints in degC.
    accuracy: allowed deviation in degC (e.g. 0.01). interval: seconds the
    temperature must stay continuously within the band (e.g. 20 s).

    Returns {"ok", "aborted", "channel", "pid": (P, I, D), "accuracy",
    "interval", "results": [{target, settled, settle_time_s, elapsed_s,
    in_band_s, final_temp, max_abs_dev}, ...]}. in_band_s is the total time
    the actual temperature spent within +/- accuracy of the target during
    that step. accuracy (allowed variation in degC) and interval (hold time
    in s) are returned so the caller can log them with the results, as is
    the PID active during the test (read once at the start). The previous
    setpoint is restored at the end (also on abort).
    """
    saved_target = dev.get_target_temp(channel)
    saved_pid = dev.get_pid(channel)
    on_step(f"PID during test: P={saved_pid[0]} I={saved_pid[1]} D={saved_pid[2]}")
    results: list[dict] = []
    try:
        for i, tgt in enumerate(targets, 1):
            if should_stop():
                on_step(f"aborted before target {tgt:.2f} C")
                break
            on_step(f"step {i}/{len(targets)}: setpoint -> {tgt:.2f} C "
                    f"(band +/-{accuracy:.3f} C held for {interval:.0f} s)")
            dev.set_target_temp(channel, tgt)
            t0 = time.monotonic()
            band_since: float | None = None
            in_band_s = 0.0  # total time within +/- accuracy of the target
            last_t = t0
            last_temp = None
            max_dev = 0.0
            settled = False
            settle_time = None
            while not should_stop():
                now = time.monotonic()
                if now - t0 > max_step_s:
                    on_step(f"step {i}/{len(targets)}: target {tgt:.2f} C NOT settled "
                            f"after {max_step_s:.0f} s (timeout)")
                    break
                temp = dev.get_actual_temp(channel)
                last_temp = temp
                dev_abs = abs(temp - tgt)
                if dev_abs > max_dev:
                    max_dev = dev_abs
                if dev_abs <= accuracy:
                    in_band_s += now - last_t  # count this sample interval as in-band
                    if band_since is None:
                        band_since = now
                    if now - band_since >= interval:  # no sample left the band
                        settled = True
                        settle_time = now - t0
                        break
                else:
                    band_since = None  # a value escaped -> restart the window
                last_t = now
                time.sleep(sample_s)
            if should_stop():
                break
            elapsed = time.monotonic() - t0
            results.append({
                "target": tgt,
                "settled": settled,
                "settle_time_s": settle_time,
                "elapsed_s": elapsed,
                "in_band_s": in_band_s,
                "final_temp": last_temp,
                "max_abs_dev": max_dev,
            })
            on_step(f"step {i}/{len(targets)}: target {tgt:.2f} C -> "
                    + (f"settled in {settle_time:.1f} s" if settled
                       else f"NOT settled after {elapsed:.1f} s")
                    + f", in band {in_band_s:.1f} s / {elapsed:.1f} s")
        return {"ok": not should_stop(), "aborted": should_stop(),
                "channel": channel, "pid": tuple(saved_pid),
                "accuracy": accuracy, "interval": interval, "results": results}
    finally:
        try:
            dev.set_target_temp(channel, saved_target)
        except Exception:
            pass
