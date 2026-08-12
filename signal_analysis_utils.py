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
                metadata=data.metadata.copy(),
                column_names=data.column_names,
            )

        return SignalData(
            time=data.time[mask],
            amplitude=data.amplitude[mask],
            source_name=data.source_name,
            sampling_rate=data.sampling_rate,
            metadata=data.metadata.copy(),
            column_names=data.column_names,
        )

    @staticmethod
    def lowpass_filter(data: SignalData, cutoff_hz: float, order: int = 2) -> np.ndarray:
        """Return a zero-phase low-pass-filtered copy of the signal amplitude.

        A non-positive cutoff disables the filter. Frequencies at or above
        Nyquist are also returned unchanged because there is no representable
        frequency content above that limit to remove.
        """
        values = np.asarray(data.amplitude, dtype=float)
        if values.size < 2:
            return values.copy()

        fs = float(data.sampling_rate)
        cutoff = float(cutoff_hz)
        if not np.isfinite(fs) or fs <= 0 or not np.isfinite(cutoff) or cutoff <= 0:
            return values.copy()

        nyquist = 0.5 * fs
        if cutoff >= nyquist:
            return values.copy()

        stages = int(max(1, order))
        alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff / fs)

        def _one_pole(samples: np.ndarray) -> np.ndarray:
            filtered = np.empty_like(samples)
            filtered[0] = samples[0]
            for idx in range(1, samples.size):
                filtered[idx] = filtered[idx - 1] + alpha * (samples[idx] - filtered[idx - 1])
            return filtered

        def _filter_finite_segment(samples: np.ndarray) -> np.ndarray:
            filtered = samples.copy()
            for _ in range(stages):
                filtered = _one_pole(filtered)
            filtered = filtered[::-1].copy()
            for _ in range(stages):
                filtered = _one_pole(filtered)
            return filtered[::-1]

        if np.all(np.isfinite(values)):
            return _filter_finite_segment(values)

        # Keep gaps as gaps and filter each finite run independently so one NaN
        # does not contaminate the remainder of a long acquisition.
        result = values.copy()
        start = 0
        while start < values.size:
            while start < values.size and not np.isfinite(values[start]):
                start += 1
            end = start
            while end < values.size and np.isfinite(values[end]):
                end += 1
            if end > start:
                result[start:end] = _filter_finite_segment(values[start:end])
            start = end
        return result

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
        else:
            q_f = np.zeros_like(i_f)
            baseband_f = i_f + 0j
            amplitude = np.abs(i_f)

        phase_raw = np.unwrap(np.angle(baseband_f))
        if t.size > 1:
            phase_derivative = np.gradient(phase_raw, t)
        else:
            phase_derivative = np.zeros_like(phase_raw)

        reconstructed = amplitude * np.cos(phase_raw)
        return t, amplitude, phase_derivative, reconstructed

