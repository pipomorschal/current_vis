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

    @staticmethod
    def lock_in_demod(
        data: SignalData,
        reference_frequency: float,
        lowpass_cutoff_hz: float,
        lowpass_order: int = 1,
        use_iq: bool = True,
    ):
        y = np.asarray(data.amplitude, dtype=float)
        t = np.asarray(data.time, dtype=float)
        if y.size < 2 or t.size < 2:
            return np.array([]), np.array([]), np.array([]), np.array([])

        fs = float(data.sampling_rate) if data.sampling_rate > 0 else 1.0
        f0 = float(abs(reference_frequency))
        nyquist = 0.5 * fs
        cutoff = float(max(1e-9, min(lowpass_cutoff_hz, 0.95 * nyquist)))
        order = int(max(1, lowpass_order))

        # Relative Zeitachse vermeidet grosse Argumente in sin/cos bei MHz-Frequenzen.
        t_rel = t - float(t[0])
        phase = 2.0 * np.pi * f0 * t_rel
        lo = np.exp(-1j * phase)

        # Komplexe Mischung auf Basisband; Faktor 2 kompensiert den 0.5-Term der Mischung.
        baseband = 2.0 * y * lo

        # Numerisch robuster als dt/(dt+tau), besonders bei sehr kleinen dt.
        alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff / fs)

        def _lpf_one_pole(x: np.ndarray) -> np.ndarray:
            out = np.empty_like(x)
            warmup = min(16, x.size)
            out[0] = np.mean(x[:warmup])
            for idx in range(1, x.size):
                out[idx] = out[idx - 1] + alpha * (x[idx] - out[idx - 1])
            return out

        def _apply_lowpass_real(x: np.ndarray) -> np.ndarray:
            y_f = x
            for _ in range(order):
                y_f = _lpf_one_pole(y_f)
            # Vor-/Rueckwaertslauf reduziert Starttransienten und Ripple-Artefakte.
            y_b = y_f[::-1]
            for _ in range(order):
                y_b = _lpf_one_pole(y_b)
            return y_b[::-1]

        i_f = _apply_lowpass_real(np.real(baseband))
        if use_iq:
            q_f = _apply_lowpass_real(np.imag(baseband))
            baseband_f = i_f + 1j * q_f
            amplitude = np.hypot(i_f, q_f)
            phase_rad = np.unwrap(np.angle(baseband_f))
        else:
            q_f = np.zeros_like(i_f)
            baseband_f = i_f + 0j
            amplitude = np.abs(i_f)
            phase_rad = np.where(i_f >= 0.0, 0.0, np.pi)

        reconstructed = amplitude * np.cos(phase_rad)
        return t, amplitude, phase_rad, reconstructed

