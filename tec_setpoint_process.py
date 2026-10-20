"""TEC-only setpoint writes in a cancellable process; no acquisition or recording."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from recording_temperatures import apply_tec_sweep_setpoint


class TecWriteCancelled(Exception):
    pass


def apply_request(request):
    config = SimpleNamespace(itc4005_resource=request["resource"],
                             tec_setpoint_deg_c=request["target"], timeout_ms=request["timeout_ms"])
    return apply_tec_sweep_setpoint(config, required_range=(request["minimum"], request["maximum"]))


def set_tec_isolated(resource, target, minimum, maximum, *, timeout_ms=5000,
                     cancelled=None, deadline_seconds=None):
    if cancelled is not None and cancelled.is_set():
        raise TecWriteCancelled()
    if not resource or not all(math.isfinite(v) for v in (target, minimum, maximum)) or not minimum <= target <= maximum:
        raise ValueError("A TEC resource and a target inside the waveform range are required")
    request = dict(resource=resource, target=target, minimum=minimum, maximum=maximum, timeout_ms=timeout_ms)
    deadline_seconds = max(15, timeout_ms / 1000 * 12) if deadline_seconds is None else deadline_seconds
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), json.dumps(request)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               encoding="utf-8", creationflags=flags)
    deadline = time.monotonic() + deadline_seconds
    try:
        while True:
            if cancelled is not None and cancelled.is_set():
                raise TecWriteCancelled()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"TEC setpoint write exceeded {deadline_seconds:g} s")
            try:
                output, errors = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
        if cancelled is not None and cancelled.is_set():
            raise TecWriteCancelled()
        if process.returncode:
            try:
                message = json.loads(output)["error"]
            except (ValueError, KeyError):
                message = errors[-2000:] or "TEC process exited unexpectedly"
            raise RuntimeError(message)
        result = json.loads(output)
        if result.get("tec_sweep_setpoint_status") != "verified":
            raise RuntimeError("TEC setpoint was not verified")
        return result
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()


if __name__ == "__main__":
    try:
        print(json.dumps(apply_request(json.loads(sys.argv[1]))))
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        raise SystemExit(1)
