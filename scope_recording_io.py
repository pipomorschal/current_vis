"""Common scope logging filenames and atomic publication of complete captures."""
from pathlib import Path
from data_manager_signal_loader import DataManager


def next_scope_logging_path(output_dir, captured_at, channel, capture_index, extension):
    safe_channel = "".join(c if c.isalnum() else "_" for c in channel).strip("_") or "channel"
    extension = extension if extension.startswith(".") else f".{extension}"
    timestamp = captured_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    stem = f"scope_{timestamp}_{safe_channel}_{capture_index:06d}"
    candidate = Path(output_dir) / f"{stem}{extension}"
    collision = 1
    while candidate.exists() or candidate.with_name(candidate.name + ".partial").exists():
        candidate = Path(output_dir) / f"{stem}_{collision:02d}{extension}"
        collision += 1
    return candidate


def save_scope_recording(path, data):
    path = Path(path)
    staging = path.with_name(path.name + ".partial")
    if path.exists() or staging.exists():
        raise FileExistsError(f"Recording or staging file already exists: {path}")
    try:
        if path.suffix.lower() in {".h5", ".hdf5"}:
            DataManager.save_scope_hdf5(str(staging), data)
        else:
            DataManager.save_scope_csv(str(staging), data)
        # On Windows rename refuses overwriting an existing destination.
        if path.exists():
            raise FileExistsError(f"Recording already exists: {path}")
        staging.rename(path)
    finally:
        staging.unlink(missing_ok=True)
