"""Run VISA in a disposable process so a stuck driver cannot freeze Qt."""
from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np

from data_manager_signal_loader import DataManager
from signal_data_class import SignalData
from signal_data_import import OscilloscopeImporter, ScopeCaptureConfig, ScopeCommunicationError
from recording_temperatures import read_optional_temperatures


def capture_isolated(config, output_path=None, extra_metadata=None, cancelled=None,
                     deadline_seconds=None):
    channels = 2 if config.reference_channel else 1
    deadline_seconds = deadline_seconds if deadline_seconds is not None else max(
        120, config.timeout_ms / 1000 * 12 * channels + 10 * bool(config.itc4005_resource) + 3 * bool(config.t4200_port))
    with TemporaryDirectory(prefix="scope_capture_") as folder:
        root = Path(folder)
        request = {"config": asdict(config), "output_path": str(output_path) if output_path else None,
                   "metadata": dict(extra_metadata or {})}
        (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        with open(root / "diagnostics.log", "w", encoding="utf-8") as diagnostics:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), str(root)],
                                       stdout=diagnostics, stderr=diagnostics, creationflags=flags)
            deadline = time.monotonic() + deadline_seconds
            reason = None
            try:
                while process.poll() is None:
                    if cancelled is not None and cancelled.is_set():
                        reason = "Acquisition cancelled; isolated VISA process stopped."
                        break
                    if time.monotonic() >= deadline:
                        reason = f"Acquisition exceeded {deadline_seconds:g} s; isolated VISA process stopped."
                        break
                    try:
                        process.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
        log = (root / "diagnostics.log").read_text(encoding="utf-8", errors="replace")
        if log:
            logging.getLogger("scope_recording").info("VISA process diagnostics:\n%s", log[-16000:])
        if reason:
            if output_path:
                staging = Path(output_path).with_name(Path(output_path).name + ".partial")
                try:
                    staging.unlink(missing_ok=True)
                except OSError:
                    logging.getLogger("scope_recording").exception("Could not remove incomplete capture %s", staging)
            raise ScopeCommunicationError(reason)
        if process.returncode:
            error_file = root / "error.json"
            error = json.loads(error_file.read_text(encoding="utf-8")) if error_file.exists() else {
                "communication": True, "message": f"VISA process exited unexpectedly ({process.returncode})."}
            if error["communication"]:
                raise ScopeCommunicationError(error["message"])
            raise RuntimeError(error["message"])
        info = json.loads((root / "result.json").read_text(encoding="utf-8"))
        data = SignalData(time=np.load(root / "time.npy"), amplitude=np.load(root / "amplitude.npy"),
                          source_name=info["source_name"], sampling_rate=info["sampling_rate"],
                          metadata=info["metadata"], column_names=tuple(info["column_names"]))
        if "reference" in info:
            ref = info["reference"]
            data.reference = SignalData(time=np.load(root / "reference_time.npy"),
                                       amplitude=np.load(root / "reference_amplitude.npy"),
                                       source_name=ref["source_name"], sampling_rate=ref["sampling_rate"],
                                       metadata=ref["metadata"], column_names=tuple(ref["column_names"]))
        return data


def run_capture(root):
    import faulthandler
    faulthandler.enable(all_threads=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        request = json.loads((root / "request.json").read_text(encoding="utf-8"))
        config = ScopeCaptureConfig(**request["config"])
        data = OscilloscopeImporter.capture_channel(config)
        data.metadata.update(request["metadata"])
        data.metadata.update(read_optional_temperatures(config))
        if request["output_path"]:
            target = Path(request["output_path"])
            # Publish only complete recordings, even if this process is killed.
            staging = target.with_name(target.name + ".partial")
            try:
                if target.suffix.lower() in {".h5", ".hdf5"}:
                    DataManager.save_scope_hdf5(str(staging), data)
                else:
                    DataManager.save_scope_csv(str(staging), data)
                staging.replace(target)
            finally:
                staging.unlink(missing_ok=True)
        np.save(root / "time.npy", data.time)
        np.save(root / "amplitude.npy", data.amplitude)
        info = {"source_name": data.source_name, "sampling_rate": data.sampling_rate,
                "metadata": data.metadata, "column_names": data.column_names}
        if data.reference is not None:
            ref = data.reference
            np.save(root / "reference_time.npy", ref.time)
            np.save(root / "reference_amplitude.npy", ref.amplitude)
            info["reference"] = {"source_name": ref.source_name, "sampling_rate": ref.sampling_rate,
                                 "metadata": ref.metadata, "column_names": ref.column_names}
        (root / "result.json").write_text(json.dumps(info), encoding="utf-8")
        return 0
    except Exception as exc:
        logging.exception("Isolated acquisition failed")
        (root / "error.json").write_text(json.dumps({
            "communication": isinstance(exc, ScopeCommunicationError), "message": str(exc)}), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(run_capture(Path(sys.argv[1])))
