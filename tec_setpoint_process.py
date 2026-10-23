"""TEC-only setpoint writes in a cancellable process; no acquisition or recording."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from recording_temperatures import apply_tec_sweep_setpoint, disable_tec_output


class TecWriteCancelled(Exception):
    pass


def apply_request(request):
    if request.get("operation") == "disable_output":
        return disable_tec_output(request["resource"], request["timeout_ms"])
    config = SimpleNamespace(itc4005_resource=request["resource"],
                             tec_setpoint_deg_c=request["target"], timeout_ms=request["timeout_ms"])
    return apply_tec_sweep_setpoint(config, required_range=(request["minimum"], request["maximum"]),
                                    enable_output=request.get("enable_output", False))


def set_tec_isolated(resource, target, minimum, maximum, *, timeout_ms=5000,
                     cancelled=None, deadline_seconds=None, enable_output=False):
    if cancelled is not None and cancelled.is_set():
        raise TecWriteCancelled()
    if not resource or not all(math.isfinite(v) for v in (target, minimum, maximum)) or not minimum <= target <= maximum:
        raise ValueError("A TEC resource and a target inside the waveform range are required")
    request = dict(resource=resource, target=target, minimum=minimum, maximum=maximum,
                   timeout_ms=timeout_ms, enable_output=enable_output)
    return _run_request(request, cancelled=cancelled, deadline_seconds=deadline_seconds)


def disable_tec_isolated(resource, *, timeout_ms=5000, deadline_seconds=None):
    # Shutdown intentionally has no acquisition cancellation token: Stop must
    # still send and verify OFF, even after a cancelled/stuck setpoint process.
    if not resource:
        raise ValueError("A TEC resource is required for output shutdown")
    return _run_request(dict(operation="disable_output", resource=resource, timeout_ms=timeout_ms),
                        deadline_seconds=deadline_seconds)


def _run_request(request, *, cancelled=None, deadline_seconds=None):
    timeout_ms = request["timeout_ms"]
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
                raise TimeoutError(f"TEC command exceeded {deadline_seconds:g} s")
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
        key, expected = (("tec_output_status", "off_verified") if request.get("operation") == "disable_output"
                         else ("tec_sweep_setpoint_status", "verified"))
        if result.get(key) != expected:
            raise RuntimeError("TEC command was not verified")
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
