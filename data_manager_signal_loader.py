from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

try:
    import h5py
except Exception:  # pragma: no cover - optional dependency guarded at runtime
    h5py = None

from signal_data_class import SignalData


class DataManager:
    @staticmethod
    def _require_h5py():
        if h5py is None:
            raise ImportError("h5py is required for HDF5 import/export. Install it via requirements.txt.")
        return h5py

    @staticmethod
    def _to_text(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, np.ndarray):
            try:
                return json.dumps(value.tolist())
            except Exception:
                return str(value)
        return str(value)

    @classmethod
    def _collect_hdf5_datasets(cls, group, prefix: str = "") -> list[str]:
        datasets: list[str] = []
        for key, item in group.items():
            name = f"{prefix}{key}"
            if h5py is not None and isinstance(item, h5py.Dataset):
                datasets.append(name)
            elif h5py is not None and isinstance(item, h5py.Group) and key != "metadata":
                datasets.extend(cls._collect_hdf5_datasets(item, prefix=f"{name}/"))
        return datasets

    @classmethod
    def _read_hdf5_metadata(cls, h5file) -> dict[str, str]:
        metadata: dict[str, str] = {}

        for key, value in h5file.attrs.items():
            if key in {"source_name", "sampling_rate", "time_dataset", "amplitude_dataset", "column_names"}:
                continue
            metadata[str(key)] = cls._to_text(value)

        if "metadata" in h5file:
            meta_group = h5file["metadata"]
            for key, value in meta_group.attrs.items():
                metadata[str(key)] = cls._to_text(value)

        return metadata

    @staticmethod
    def estimate_sampling_rate(time: np.ndarray) -> float:
        if len(time) < 2:
            return 1.0
        dt = np.diff(time)
        dt = dt[np.isfinite(dt)]
        if len(dt) == 0:
            return 1.0
        median_dt = float(np.median(dt))
        return 1.0 / median_dt if median_dt > 0 else 1.0

    @staticmethod
    def _try_float(text: str) -> float | None:
        try:
            return float(text)
        except Exception:
            return None

    @classmethod
    def preview_csv_or_txt(cls, file_path: str) -> dict:
        path = Path(file_path)
        metadata, header, rows = cls._parse_metadata_and_table(path)

        return {
            "format": "text",
            "file_path": str(path),
            "metadata": metadata,
            "columns": header,
            "row_count": len(rows),
        }

    @classmethod
    def preview_hdf5(cls, file_path: str) -> dict:
        cls._require_h5py()
        path = Path(file_path)

        with h5py.File(path, "r") as h5:
            datasets = cls._collect_hdf5_datasets(h5)
            metadata = cls._read_hdf5_metadata(h5)

            row_count = 0
            for candidate in ("amplitude", "signal", "time"):
                if candidate in h5 and isinstance(h5[candidate], h5py.Dataset):
                    try:
                        row_count = int(h5[candidate].shape[0])
                    except Exception:
                        row_count = 0
                    break

            if row_count == 0 and datasets:
                try:
                    row_count = int(h5[datasets[0]].shape[0])
                except Exception:
                    row_count = 0

            return {
                "format": "hdf5",
                "file_path": str(path),
                "metadata": metadata,
                "datasets": datasets,
                "columns": datasets,
                "row_count": row_count,
                "source_name": cls._to_text(h5.attrs.get("source_name", path.name)),
            }

    @classmethod
    def load_file(
        cls,
        file_path: str,
        time_column: str | None = None,
        amplitude_column: str | None = None,
        time_dataset: str | None = None,
        amplitude_dataset: str | None = None,
    ) -> SignalData:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in {".csv", ".txt"}:
            return cls._load_csv_or_txt(path, time_column=time_column, amplitude_column=amplitude_column)
        if suffix in {".h5", ".hdf5"}:
            return cls._load_hdf5(path, time_dataset=time_dataset, amplitude_dataset=amplitude_dataset)
        if suffix == ".npy":
            return cls._load_npy(path)
        if suffix == ".npz":
            return cls._load_npz(path)

        raise ValueError(f"Unsupported file type: {suffix}")

    @classmethod
    def _parse_metadata_and_table(cls, path: Path) -> tuple[dict[str, str], list[str], list[list[str]]]:
        metadata: dict[str, str] = {}
        header: list[str] | None = None
        rows: list[list[str]] = []

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                parts = [p.strip() for p in line.split(",")]

                if header is None:
                    if len(parts) >= 2:
                        left, right = parts[0], parts[1]
                        left_float = cls._try_float(left)
                        right_float = cls._try_float(right)

                        if left.upper() == "TIME" or any(p.upper() == "TIME" for p in parts):
                            header = parts
                            continue

                        if left_float is None or right_float is None:
                            metadata[left] = ",".join(parts[1:]).strip()
                            continue

                    continue

                if len(parts) >= len(header):
                    rows.append(parts[:len(header)])

        if header is None:
            raise ValueError("Could not find table header line (for example: TIME,CH3).")

        return metadata, header, rows

    @classmethod
    def _load_csv_or_txt(
        cls,
        path: Path,
        time_column: str | None = None,
        amplitude_column: str | None = None,
    ) -> SignalData:
        metadata, header, rows = cls._parse_metadata_and_table(path)

        if not rows:
            raise ValueError("No numeric data rows found after the table header.")

        header_upper = [h.strip().upper() for h in header]

        if time_column is None:
            time_column = header[header_upper.index("TIME")] if "TIME" in header_upper else header[0]

        if amplitude_column is None:
            candidates = [h for h in header if h.upper() != time_column.upper()]
            if not candidates:
                raise ValueError("No amplitude column found.")
            amplitude_column = candidates[0]

        if time_column.upper() not in header_upper:
            raise ValueError(f"Time column '{time_column}' not found in header: {header}")
        if amplitude_column.upper() not in header_upper:
            raise ValueError(f"Amplitude column '{amplitude_column}' not found in header: {header}")

        time_idx = header_upper.index(time_column.upper())
        amp_idx = header_upper.index(amplitude_column.upper())

        time_vals = []
        amp_vals = []

        for row in rows:
            try:
                t = float(row[time_idx])
                y = float(row[amp_idx])
            except Exception:
                continue
            time_vals.append(t)
            amp_vals.append(y)

        if not time_vals:
            raise ValueError("No valid numeric samples could be parsed from the selected columns.")

        time = np.asarray(time_vals, dtype=float)
        amplitude = np.asarray(amp_vals, dtype=float)

        fs = cls.estimate_sampling_rate(time)

        if "Sample Interval" in metadata:
            try:
                sample_interval = float(metadata["Sample Interval"])
                if sample_interval > 0:
                    fs = 1.0 / sample_interval
            except Exception:
                pass

        return SignalData(
            time=time,
            amplitude=amplitude,
            source_name=path.name,
            sampling_rate=fs,
            metadata=metadata,
            column_names=(time_column, amplitude_column),
        )

    @classmethod
    def _load_hdf5(
        cls,
        path: Path,
        time_dataset: str | None = None,
        amplitude_dataset: str | None = None,
    ) -> SignalData:
        cls._require_h5py()

        with h5py.File(path, "r") as h5:
            metadata = cls._read_hdf5_metadata(h5)
            source_name = cls._to_text(h5.attrs.get("source_name", path.name))

            time_name = time_dataset or cls._to_text(h5.attrs.get("time_dataset", "")) or "time"
            amp_name = amplitude_dataset or cls._to_text(h5.attrs.get("amplitude_dataset", "")) or "amplitude"
            display_time_name = cls._to_text(h5.attrs.get("time_column", time_name)) or time_name
            display_amp_name = cls._to_text(h5.attrs.get("amplitude_column", amp_name)) or amp_name

            if time_name == amp_name and time_name in h5 and isinstance(h5[time_name], h5py.Dataset):
                ds = np.asarray(h5[time_name])
                if ds.ndim == 1:
                    amplitude = ds.astype(float).reshape(-1)
                    time = np.arange(len(amplitude), dtype=float)
                    fs = float(h5.attrs.get("sampling_rate", 1.0))
                    return SignalData(
                        time=time,
                        amplitude=amplitude,
                        source_name=source_name,
                        sampling_rate=fs,
                        metadata=metadata,
                        column_names=("TIME", display_amp_name or time_name),
                    )
                if ds.ndim == 2 and ds.shape[1] >= 2:
                    time = ds[:, 0].astype(float)
                    amplitude = ds[:, 1].astype(float)
                    fs = float(h5.attrs.get("sampling_rate", cls.estimate_sampling_rate(time)))
                    return SignalData(
                        time=time,
                        amplitude=amplitude,
                        source_name=source_name,
                        sampling_rate=fs,
                        metadata=metadata,
                        column_names=(f"{display_time_name}[0]", f"{display_amp_name}[1]"),
                    )

            if time_name in h5 and amp_name in h5:
                time = np.asarray(h5[time_name], dtype=float)
                amplitude = np.asarray(h5[amp_name], dtype=float)
                if time.shape != amplitude.shape:
                    if time.size != amplitude.size:
                        raise ValueError("HDF5 time and amplitude datasets must have the same length.")
                    time = time.reshape(-1)
                    amplitude = amplitude.reshape(-1)
                fs = float(h5.attrs.get("sampling_rate", cls.estimate_sampling_rate(time)))
                return SignalData(
                    time=time,
                    amplitude=amplitude,
                    source_name=source_name,
                    sampling_rate=fs,
                    metadata=metadata,
                    column_names=(display_time_name, display_amp_name),
                )

            if "signal" in h5 and isinstance(h5["signal"], h5py.Dataset):
                amplitude = np.asarray(h5["signal"], dtype=float).reshape(-1)
                time = np.arange(len(amplitude), dtype=float)
                fs = float(h5.attrs.get("sampling_rate", 1.0))
                return SignalData(
                    time=time,
                    amplitude=amplitude,
                    source_name=source_name,
                    sampling_rate=fs,
                    metadata=metadata,
                    column_names=("TIME", "signal"),
                )

            dataset_names = cls._collect_hdf5_datasets(h5)
            if len(dataset_names) == 1:
                ds = np.asarray(h5[dataset_names[0]])
                if ds.ndim == 2 and ds.shape[1] >= 2:
                    time = ds[:, 0].astype(float)
                    amplitude = ds[:, 1].astype(float)
                    fs = float(h5.attrs.get("sampling_rate", cls.estimate_sampling_rate(time)))
                    return SignalData(
                        time=time,
                        amplitude=amplitude,
                        source_name=source_name,
                        sampling_rate=fs,
                        metadata=metadata,
                        column_names=(f"{dataset_names[0]}[0]", f"{dataset_names[0]}[1]"),
                    )

        raise ValueError("Could not find HDF5 datasets for time/amplitude. Expected 'time' and 'amplitude' datasets or a saved current_vis file.")

    @classmethod
    def _load_npy(cls, path: Path) -> SignalData:
        arr = np.load(path, allow_pickle=True)

        if arr.ndim == 1:
            amplitude = arr.astype(float)
            time = np.arange(len(amplitude), dtype=float)
            return SignalData(time=time, amplitude=amplitude, source_name=path.name, sampling_rate=1.0)

        if arr.ndim == 2 and arr.shape[1] >= 2:
            time = arr[:, 0].astype(float)
            amplitude = arr[:, 1].astype(float)
            fs = cls.estimate_sampling_rate(time)
            return SignalData(time=time, amplitude=amplitude, source_name=path.name, sampling_rate=fs)

        raise ValueError("Unsupported .npy format. Use 1D amplitude array or 2D [time, amplitude] array.")

    @classmethod
    def _load_npz(cls, path: Path) -> SignalData:
        data = np.load(path, allow_pickle=True)

        if "time" in data and "amplitude" in data:
            time = np.asarray(data["time"], dtype=float)
            amplitude = np.asarray(data["amplitude"], dtype=float)
            fs = float(data["sampling_rate"]) if "sampling_rate" in data else cls.estimate_sampling_rate(time)
            return SignalData(time=time, amplitude=amplitude, source_name=path.name, sampling_rate=fs)

        if "signal" in data:
            amplitude = np.asarray(data["signal"], dtype=float)
            time = np.arange(len(amplitude), dtype=float)
            fs = float(data["sampling_rate"]) if "sampling_rate" in data else 1.0
            return SignalData(time=time, amplitude=amplitude, source_name=path.name, sampling_rate=fs)

        raise ValueError("NPZ must contain either ('time', 'amplitude') or 'signal'.")

    @staticmethod
    def save_csv(file_path: str, data: SignalData):
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "amplitude"])
            writer.writerows(zip(data.time.tolist(), data.amplitude.tolist()))

    @classmethod
    def save_hdf5(cls, file_path: str, data: SignalData):
        cls._require_h5py()
        with h5py.File(file_path, "w") as h5:
            h5.attrs["source_name"] = data.source_name
            h5.attrs["sampling_rate"] = float(data.sampling_rate)
            h5.attrs["time_dataset"] = "time"
            h5.attrs["amplitude_dataset"] = "amplitude"
            h5.attrs["time_column"] = str(data.column_names[0] or "TIME")
            h5.attrs["amplitude_column"] = str(data.column_names[1] or "AMPLITUDE")
            h5.attrs["column_names"] = np.array([
                str(data.column_names[0] or "TIME"),
                str(data.column_names[1] or "AMPLITUDE"),
            ], dtype=h5py.string_dtype(encoding="utf-8"))

            h5.create_dataset("time", data=np.asarray(data.time, dtype=float), compression="gzip", compression_opts=4)
            h5.create_dataset("amplitude", data=np.asarray(data.amplitude, dtype=float), compression="gzip", compression_opts=4)

            if data.metadata:
                meta_group = h5.create_group("metadata")
                for key, value in data.metadata.items():
                    meta_group.attrs[str(key)] = cls._to_text(value)

    @staticmethod
    def save_scope_csv(file_path: str, data: SignalData):
        time_col, amp_col = data.column_names
        amp_col = amp_col or "CH1"

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Metadaten zuerst, damit der bestehende Loader sie mit einliest.
            if data.metadata:
                for key, value in data.metadata.items():
                    writer.writerow([str(key), str(value)])
            if data.sampling_rate > 0:
                writer.writerow(["Sample Interval", f"{1.0 / data.sampling_rate:.16g}"])

            writer.writerow([time_col or "TIME", amp_col])
            writer.writerows(zip(data.time.tolist(), data.amplitude.tolist()))

    @classmethod
    def save_scope_hdf5(cls, file_path: str, data: SignalData):
        cls.save_hdf5(file_path, data)

    @staticmethod
    def save_json(file_path: str, data: SignalData):
        payload = {
            "source_name": data.source_name,
            "sampling_rate": data.sampling_rate,
            "time": data.time.tolist(),
            "amplitude": data.amplitude.tolist(),
            "metadata": data.metadata,
            "column_names": list(data.column_names),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def save_npz(file_path: str, data: SignalData):
        np.savez(file_path, time=data.time, amplitude=data.amplitude, sampling_rate=data.sampling_rate)
