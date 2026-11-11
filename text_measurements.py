"""Streaming delimited measurements with numeric or ISO-8601 time columns."""
from __future__ import annotations

from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

import numpy as np

from signal_data_class import SignalData


class TextImportCancelled(Exception):
    pass


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise TextImportCancelled()


def _number(value, delimiter):
    value = value.strip()
    if delimiter != ",":
        value = value.replace(",", ".")
    return float(value)


def _delimiter(sample):
    lines = [line for line in sample.splitlines()[:30] if line.strip()]
    for line in lines:
        for delimiter in (",", ";", "\t"):
            fields = next(csv.reader([line], delimiter=delimiter))
            if len(fields) > 1 and fields[0].strip().upper() in {"TIME", "TIMESTAMP", "DATETIME"}:
                return delimiter
    if not lines:
        return ","
    return max((",", ";", "\t"), key=lambda sep: len(next(csv.reader([lines[0]], delimiter=sep))))


@contextmanager
def open_text_table(path, cancelled=None):
    """Keep only a single CSV row in memory, including for very wide exports."""
    with Path(path).open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        delimiter = _delimiter(handle.read(65536))
        handle.seek(0)
        reader = csv.reader(handle, delimiter=delimiter)
        metadata, header = {}, None
        first = []
        for row in reader:
            _check_cancelled(cancelled)
            row = [value.strip() for value in row]
            if not any(row):
                continue
            if len(row) < 2:
                continue
            if row[0].upper() in {"TIME", "TIMESTAMP", "DATETIME"} or "TIME" in [v.upper() for v in row]:
                header = row
                break
            try:
                _number(row[0], delimiter)
                _number(row[1], delimiter)
                header = ["TIME", "VOLTAGE", *(f"COLUMN_{i}" for i in range(3, len(row) + 1))]
                first = [row]
                break
            except ValueError:
                metadata[row[0]] = delimiter.join(row[1:])
        if header is None:
            raise ValueError("Could not find a table header or a row with at least two numeric columns.")

        def rows():
            for index, row in enumerate(chain(first, reader)):
                if index % 256 == 0:
                    _check_cancelled(cancelled)
                if len(row) >= len(header) and any(value.strip() for value in row):
                    yield row
        yield metadata, header, rows(), delimiter


class TimeDecoder:
    def __init__(self, delimiter):
        self.delimiter, self.kind, self.origin = delimiter, None, None

    def decode(self, text):
        if self.kind != "ISO8601":
            try:
                value = _number(text, self.delimiter)
                if not np.isfinite(value):
                    raise ValueError("Non-finite time")
                self.kind = "numeric"
                return value
            except ValueError:
                if self.kind == "numeric":
                    raise
        stamp = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        stamp = stamp.astimezone(timezone.utc)
        self.kind = "ISO8601"
        if self.origin is None:
            self.origin = stamp
        return (stamp - self.origin).total_seconds()


def preview_text(path, cancelled=None):
    with open_text_table(path, cancelled) as (metadata, header, rows, delimiter):
        count, decoder = 0, TimeDecoder(delimiter)
        populated = np.zeros(len(header), dtype=np.int64)
        for row in rows:
            count += 1
            populated += np.fromiter((bool(v.strip()) for v in row[:len(header)]), dtype=np.int8)
            if decoder.kind is None:
                try:
                    decoder.decode(row[0])
                except ValueError:
                    pass
        return dict(format="text", file_path=str(path), metadata=metadata, columns=header, row_count=count,
                    delimiter=delimiter, timestamp_format=decoder.kind,
                    time_origin_utc=decoder.origin.isoformat() if decoder.origin is not None else None,
                    populated_counts=dict(zip(header, populated.tolist())))


def load_text_measurements(path, time_column=None, amplitude_column=None, additional_columns=(), cancelled=None):
    with open_text_table(path, cancelled) as (metadata, header, rows, delimiter):
        upper = [name.upper() for name in header]
        if time_column is None:
            time_column = header[upper.index("TIME")] if "TIME" in upper else header[0]
        amplitude_column = amplitude_column or next((name for name in header if name != time_column), None)
        names = list(dict.fromkeys([amplitude_column, *additional_columns]))
        if any(name not in header for name in [time_column, *names]) or time_column in names:
            raise ValueError("Choose a time column and different measurement columns from the file")
        time_index = header.index(time_column)
        columns = [(name, header.index(name), [], []) for name in names]
        decoder = TimeDecoder(delimiter)
        invalid_times, invalid_values = 0, 0
        for row in rows:
            try:
                elapsed = decoder.decode(row[time_index])
            except ValueError:
                invalid_times += 1
                continue
            for name, index, times, values in columns:
                text = row[index].strip()
                if not text:
                    continue  # Empty cells are absent measurements, never zero or held values.
                try:
                    value = _number(text, delimiter)
                    if not np.isfinite(value):
                        raise ValueError("Non-finite measurement")
                except ValueError:
                    invalid_values += 1
                    continue
                times.append(elapsed)
                values.append(value)
        if invalid_times:
            metadata["skipped_invalid_time_rows"] = str(invalid_times)
        if invalid_values:
            metadata["skipped_invalid_measurements"] = str(invalid_values)
        metadata["csv_delimiter"] = "tab" if delimiter == "\t" else delimiter
        if decoder.kind == "ISO8601":
            metadata.update(source_time_format="ISO8601", time_origin_utc=decoder.origin.isoformat(),
                            time_representation="elapsed_seconds")
        traces = {}
        for name, index, times, values in columns:
            if not values:
                raise ValueError(f"No numeric measurements found in column {name!r}. Choose another column.")
            times, values = np.asarray(times), np.asarray(values)
            if np.any(np.diff(times) < 0):
                order = np.argsort(times, kind="stable")
                times, values = times[order], values[order]
            spacing = np.diff(times)
            spacing = spacing[spacing > 0]
            sampling_rate = 1 / float(np.median(spacing)) if spacing.size else 1
            traces[name] = SignalData(times, values, source_name=Path(path).name, sampling_rate=sampling_rate,
                                      metadata={**metadata, "measurement_column": name},
                                      column_names=(time_column, name))
        primary = traces.pop(amplitude_column)
        primary.channels = traces
        return primary


def needs_measurement_import(path):
    """Recognize ISO timestamp tables/alternate delimiters without reading the whole file."""
    with open_text_table(path) as (metadata, header, rows, delimiter):
        if "captured_channel_data" in metadata or "reference_channel" in metadata:
            return False
        if delimiter != ",":
            return True
        for index, row in enumerate(rows):
            if index >= 100:
                break
            try:
                decoder = TimeDecoder(delimiter)
                decoder.decode(row[0])
                return decoder.kind == "ISO8601"
            except ValueError:
                continue
    return False
