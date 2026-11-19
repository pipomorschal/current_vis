"""Progress watcher for the TEC207L on-device PID auto-tune (AUTOPID register).

Writing 1 to the AUTOPID register (0x1107) makes the firmware run its own PID
self-tuning on that channel. While the tuning runs, reading AUTOPID back
returns 1; when the firmware finishes it resets the register to 0 and stores
the new coefficients in KP/KI/KD. Writing 0 aborts the tuning.

run_hw_autotune() starts the tuning and polls AUTOPID until the firmware
clears it, reporting progress through on_step callbacks.
"""

from __future__ import annotations

import time

POLL_S = 1.0           # AUTOPID poll interval while the tuning runs
REPORT_EVERY_S = 10.0  # how often to emit a "still running" progress message
TIMEOUT_S = 600.0      # give up if the firmware never clears AUTOPID


class HwAutotuneError(Exception):
    """Raised when the device does not start or finish the self-tuning."""


class Aborted(Exception):
    """Raised internally when the user stops the hardware auto-tune."""


def _check_stop(should_stop) -> None:
    if should_stop():
        raise Aborted()


def _abort(dev, channel: int) -> None:
    """Best-effort abort of the on-device tuning (AUTOPID=0)."""

    try:
        dev.stop_autotune(channel)
    except Exception:
        pass  # device may be gone; nothing more we can do


def run_hw_autotune(dev, channel: int, *,
                    on_step=lambda msg: None, should_stop=lambda: False,
                    poll_s: float = POLL_S, timeout_s: float = TIMEOUT_S) -> dict:
    """Start the on-device PID auto-tune and watch it until it finishes.

    Blocking - call in a worker thread. on_step(msg) receives progress
    messages, should_stop() returns True when the user aborts.

    Returns {"ok": True, "pid": (P, I, D), "elapsed_s": ...} on success,
    {"ok": False, "aborted": True} when aborted (AUTOPID=0 sent to the
    device), or raises HwAutotuneError (AUTOPID=0 sent as well).
    """

    try:
        _check_stop(should_stop)
        echo = dev.start_autotune(channel)
        on_step(f"step 1/3: AUTOPID=1 sent to CH{channel}, "
                f"PID self-tuning started on device")

        state = dev.get_autotune_state(channel)
        if state != 1:
            raise HwAutotuneError(
                f"device did not accept self-tuning (AUTOPID={state}, echo={echo!r})")
        on_step("step 2/3: tuning running on device (AUTOPID=1), "
                "waiting for completion...")

        t0 = time.monotonic()
        last_report = 0.0
        while True:
            _check_stop(should_stop)
            state = dev.get_autotune_state(channel)
            if state == 0:
                break  # firmware finished and applied the new PID
            if state == 2:
                raise HwAutotuneError(
                    "device switched to real-time optimization (AUTOPID=2)")
            elapsed = time.monotonic() - t0
            if elapsed > timeout_s:
                raise HwAutotuneError(
                    f"auto-tune still running after {timeout_s:.0f} s")
            if elapsed - last_report >= REPORT_EVERY_S:
                last_report = elapsed
                on_step(f"step 2/3: still tuning on device... "
                        f"({elapsed:.0f} s elapsed, AUTOPID=1)")
            time.sleep(poll_s)

        elapsed = time.monotonic() - t0
        pid = dev.get_pid(channel)
        on_step(f"step 3/3: tuning finished after {elapsed:.0f} s - "
                f"new PID on device: P={pid[0]} I={pid[1]} D={pid[2]}")
        return {"ok": True, "aborted": False, "channel": channel,
                "pid": pid, "elapsed_s": elapsed}
    except Aborted:
        _abort(dev, channel)
        return {"ok": False, "aborted": True, "channel": channel}
    except Exception:
        _abort(dev, channel)
        raise
