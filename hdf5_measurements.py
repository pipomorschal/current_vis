"""Describe/select HDF5 measurements without confusing their individual time axes."""
from __future__ import annotations

import json
from pathlib import PurePosixPath

import numpy as np

from signal_data_class import SignalData
from text_measurements import TextImportCancelled


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _time_name(name):
    leaf = PurePosixPath(name).name.lower()
    return leaf in {"time", "timestamp", "timestamps", "time_s", "time_seconds"} or leaf.endswith("_time")


def describe_hdf5_measurements(handle, datasets, metadata):
    """Only inspect shapes/attributes; preview must not read large sample arrays."""
    vectors = {name: handle[name] for name in datasets
               if handle[name].ndim == 1 and handle[name].dtype.kind in "biuf"}
    default_time = _text(handle.attrs.get("time_dataset", "time")).lstrip("/")
    default_amplitude = _text(handle.attrs.get("amplitude_dataset", "amplitude")).lstrip("/")
    source = _text(handle.attrs.get("source_name", "Untitled"))
    specs = {}
    native_extras = set()
    for name, dataset in vectors.items():
        if _time_name(name) or name == default_time:
            continue
        parent = dataset.parent
        native = name.startswith("captured_channels/") and PurePosixPath(name).name == "amplitude"
        if native:
            label = _text(parent.attrs["label"])
            time_path = parent.name.strip("/") + "/time"
            own_metadata = json.loads(_text(parent.attrs.get("metadata", "{}")))
            column_names = tuple(json.loads(_text(parent.attrs.get("column_names", '["TIME", "AMPLITUDE"]'))))
            rate = float(parent.attrs.get("sampling_rate", 1))
            own_source = _text(parent.attrs.get("source_name", source))
            native_extras.add(name)
        elif name == "reference_amplitude" and "reference_time" in vectors:
            label = _text(handle.attrs.get("reference_channel", metadata.get("reference_channel", "Reference")))
            time_path = "reference_time"
            own_metadata = {str(k): _text(v) for k, v in handle["reference_metadata"].attrs.items()} if "reference_metadata" in handle else {}
            column_names = ("REFERENCE_TIME", label)
            rate = float(handle.attrs.get("reference_sampling_rate", 1))
            own_source = source
            native_extras.add(name)
        elif name == default_amplitude or (name == "signal" and default_amplitude not in vectors):
            label = _text(handle.attrs.get("amplitude_column", metadata.get("channel", name)))
            time_path = default_time if default_time in vectors else None
            own_metadata = metadata.copy()
            column_names = (_text(handle.attrs.get("time_column", "TIME")), label)
            rate = float(handle.attrs["sampling_rate"]) if "sampling_rate" in handle.attrs else None
            own_source = source
        else:
            label = name
            # A vector in its own group can carry a separate time axis.
            local_time = parent.name.strip("/") + "/time" if parent.name != "/" else "time"
            declared_time = dataset.attrs.get("time_dataset")
            if declared_time is not None:
                declared = _text(declared_time)
                relative = parent.name.strip("/") + "/" + declared
                time_path = relative if not declared.startswith("/") and relative in vectors else declared.lstrip("/")
            else:
                time_path = local_time if local_time in vectors else None
            own_metadata = {str(k): _text(v) for k, v in dataset.attrs.items() if k != "time_dataset"}
            column_names = ("TIME", label)
            rate = float(dataset.attrs["sampling_rate"]) if "sampling_rate" in dataset.attrs else None
            own_source = source
        specs[name] = dict(label=label, time_dataset=time_path, count=int(dataset.shape[0]),
                           metadata=own_metadata, column_names=column_names, sampling_rate=rate,
                           source_name=own_source)
    # New captures may hard-link the reference into captured_channels. Show it once.
    reference = specs.get("reference_amplitude")
    if reference and any(specs[key]["label"] == reference["label"] for key in native_extras if key != "reference_amplitude"):
        del specs["reference_amplitude"]
        native_extras.discard("reference_amplitude")
    if default_amplitude not in specs:
        default_amplitude = "signal" if "signal" in specs else next(iter(specs), None)
    time_paths = list(dict.fromkeys([*(name for name in vectors if _time_name(name)),
                                    *(s["time_dataset"] for s in specs.values() if s["time_dataset"] in vectors)]))
    if default_time not in vectors:
        default_time = time_paths[0] if time_paths else default_amplitude
    return dict(measurements=specs, time_datasets=time_paths or list(vectors),
                default_time_dataset=default_time, default_amplitude_dataset=default_amplitude,
                default_additional_datasets=[name for name in specs if name in native_extras])


def load_hdf5_selection(handle, path, time_dataset, amplitude_dataset, additional_datasets, description,
                         root_metadata, cancelled=None):
    specs = description["measurements"]
    primary_path = (amplitude_dataset or description["default_amplitude_dataset"] or "").lstrip("/")
    time_path = (time_dataset.lstrip("/") if time_dataset else
                 specs.get(primary_path, {}).get("time_dataset") or description["default_time_dataset"])
    names = list(dict.fromkeys([primary_path, *(name.lstrip("/") for name in additional_datasets)]))
    if any(name not in specs for name in names):
        raise ValueError("Select numeric one-dimensional HDF5 measurement datasets")
    traces, labels, cache = {}, {}, {}

    def check():
        if cancelled is not None and cancelled():
            raise TextImportCancelled()

    def read(name):
        check()
        if name not in cache:
            if name not in handle or handle[name].ndim != 1 or handle[name].dtype.kind not in "biuf":
                raise ValueError(f"HDF5 dataset {name!r} must be a numeric one-dimensional vector")
            cache[name] = np.asarray(handle[name], dtype=float)
        check()
        return cache[name]

    for index, name in enumerate(names):
        spec = specs[name]
        amplitude = read(name)
        axis = time_path if index == 0 else spec["time_dataset"] or time_path
        if axis == name or axis is None:
            rate = spec["sampling_rate"] or 1
            times = np.arange(amplitude.size) / rate
        else:
            times = read(axis)
        if times.shape != amplitude.shape:
            raise ValueError(f"HDF5 measurement {name!r} has {amplitude.size} values but time dataset {axis!r} has {times.size}")
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
            raise ValueError(f"HDF5 time dataset {axis!r} must be finite and chronological")
        spacing = np.diff(times)
        spacing = spacing[spacing > 0]
        rate = spec["sampling_rate"] or (1 / float(np.median(spacing)) if spacing.size else 1)
        label = spec["label"]
        if label in labels:
            label = name
        labels[label] = name
        metadata = {**root_metadata, **spec["metadata"]}
        metadata["channel"] = label
        metadata["hdf5_dataset"] = name
        traces[name] = SignalData(times, amplitude, source_name=spec["source_name"] if spec["source_name"] != "Untitled" else str(path),
                                  sampling_rate=rate, metadata=metadata,
                                  column_names=(spec["column_names"][0], label))
    primary = traces[primary_path]
    primary.channels = {label: traces[name] for label, name in labels.items() if name != primary_path}
    reference_label = root_metadata.get("reference_channel")
    if reference_label in primary.channels:
        primary.reference = primary.channels[reference_label]
    return primary
