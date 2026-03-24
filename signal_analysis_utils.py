from __future__ import annotations

import numpy as np

from signal_data_class import SignalData


class Analysis:
    @staticmethod
    def select_range(data: SignalData, start_time: float, end_time: float) -> SignalData:
        if data.n_samples == 0:
            return data

        lo = min(start_time, end_time)
        hi = max(start_time, end_time)
        mask = (data.time >= lo) & (data.time <= hi)

        if not np.any(mask):
            return SignalData(
                time=np.array([], dtype=float),
                amplitude=np.array([], dtype=float),
                source_name=data.source_name,
                sampling_rate=data.sampling_rate,
            )

        return SignalData(
            time=data.time[mask],
            amplitude=data.amplitude[mask],
            source_name=data.source_name,
            sampling_rate=data.sampling_rate,
        )

    @staticmethod
    def window(name: str, n: int) -> np.ndarray:
        name = (name or "hann").lower()
        if name == "hann":
            return np.hanning(n)
        if name == "hamming":
            return np.hamming(n)
        if name == "blackman":
            return np.blackman(n)
        if name == "rectangular":
            return np.ones(n)
        return np.hanning(n)

    @staticmethod
    def fft_spectrum(data: SignalData, window_name: str = "hann", remove_mean: bool = True):
        if data.n_samples < 2:
            return np.array([]), np.array([])

        y = np.asarray(data.amplitude, dtype=float)
        if remove_mean:
            y = y - np.mean(y)

        y = y * Analysis.window(window_name, len(y))
        n = len(y)

        yf = np.fft.rfft(y)
        xf = np.fft.rfftfreq(n, d=1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0)
        mag = np.abs(yf) / max(1, n)
        return xf, mag

    @staticmethod
    def stft(
        data: SignalData,
        window_name: str = "hann",
        nperseg: int = 256,
        noverlap: int = 128,
        nfft: int | None = None,
        remove_mean: bool = True,
    ):
        y = np.asarray(data.amplitude)
        if y.size < 2:
            return np.array([]), np.array([]), np.empty((0, 0))

        nperseg = int(max(8, min(nperseg, len(y))))
        noverlap = int(max(0, min(noverlap, nperseg - 1)))
        step = nperseg - noverlap
        nfft = int(max(nperseg, nfft or nperseg))

        window = Analysis.window(window_name, nperseg)

        frames = []
        times = []

        for start in range(0, len(y) - nperseg + 1, step):
            segment = y[start:start + nperseg].copy()
            if remove_mean:
                segment -= np.mean(segment)
            segment *= window
            spec = np.fft.fft(segment, n=nfft)
            frames.append(np.abs(spec[:nfft // 2 + 1]))
            times.append(data.time[start + nperseg // 2])

        if not frames:
            return np.array([]), np.array([]), np.empty((0, 0))

        Z = np.array(frames).T
        freqs = np.fft.rfftfreq(nfft, d=1.0 / data.sampling_rate if data.sampling_rate > 0 else 1.0)
        return np.asarray(times, dtype=float), freqs, Z
