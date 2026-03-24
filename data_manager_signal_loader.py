from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from signal_data_class import SignalData


class DataManager:
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
            "file_path": str(path),
            "metadata": metadata,
            "columns": header,
            "row_count": len(rows),
        }

    @classmethod
    def load_file(
        cls,
        file_path: str,
        time_column: str | None = None,
        amplitude_column: str | None = None,
    ) -> SignalData:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in {".csv", ".txt"}:
            return cls._load_csv_or_txt(path, time_column=time_column, amplitude_column=amplitude_column)
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
