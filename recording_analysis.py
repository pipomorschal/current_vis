"""Extract peak sine amplitudes at selected frequencies."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from signal_data_class import SignalData


def recording_amplitude(data: SignalData, highpass_hz: float = 0,
                        lowpass_hz: float = 0, frequency_hz: float = 50) -> float:
    return recording_amplitudes(data, [frequency_hz], highpass_hz, lowpass_hz)[0]


def recording_amplitudes(data: SignalData, frequencies, highpass_hz=0, lowpass_hz=0):
    t = np.asarray(data.time, dtype=float).reshape(-1)
    y = np.asarray(data.amplitude, dtype=float).reshape(-1)
    fs = float(data.sampling_rate)
    if t.size != y.size or t.size < 4:
        raise ValueError("At least four matching time and amplitude samples are required.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("Waveform contains non-finite samples.")
    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
    if (not frequencies.size or not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0)
            or len(set(frequencies)) != frequencies.size):
        raise ValueError("Evaluation frequencies must be distinct, finite and positive.")
    if not np.isfinite(fs) or np.any(frequencies >= fs / 2):
        raise ValueError("Evaluation frequencies must be below the waveform Nyquist frequency.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time samples must be strictly increasing.")
    if any(not np.isfinite(v) or v < 0 for v in (highpass_hz, lowpass_hz)):
        raise ValueError("Filter cutoffs must be finite and non-negative.")
    if highpass_hz >= fs / 2 or lowpass_hz >= fs / 2:
        raise ValueError("Filter cutoffs must be below the waveform Nyquist frequency.")
    if highpass_hz and lowpass_hz and highpass_hz >= lowpass_hz:
        raise ValueError("High-pass cutoff must be below low-pass cutoff.")
    if highpass_hz or lowpass_hz:
        dt = np.diff(t)
        if not np.allclose(dt, 1 / fs, rtol=0.01, atol=1e-12):
            raise ValueError("Filtering requires uniformly sampled waveform data.")
        if highpass_hz and lowpass_hz:
            cutoff, kind = [highpass_hz, lowpass_hz], "bandpass"
        else:
            cutoff = highpass_hz or lowpass_hz
            kind = "highpass" if highpass_hz else "lowpass"
        sos = butter(4, cutoff, btype=kind, fs=fs, output="sos")
        y = sosfiltfilt(sos, y)
    width = 2 * frequencies.size + 1
    block_size = max(1024, 2_000_000 // width)
    reduced = None
    for start in range(0, t.size, block_size):
        times = t[start:start + block_size] - t[0]
        columns = []
        for frequency in frequencies:
            phase = 2 * np.pi * frequency * times
            columns.extend((np.sin(phase), np.cos(phase)))
        design = np.column_stack([*columns, np.ones(times.size)])
        if t.size <= block_size:
            coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
            break
        # Incremental QR retains the least-squares problem without a multi-GB
        # full-resolution harmonic design matrix or ill-conditioned normal equations.
        augmented = np.column_stack([design, y[start:start + block_size]])
        if reduced is not None:
            augmented = np.vstack([reduced, augmented])
        reduced = np.linalg.qr(augmented, mode="r")
    else:
        coefficients, _, rank, _ = np.linalg.lstsq(reduced[:, :-1], reduced[:, -1], rcond=None)
    if rank < width:
        raise ValueError("Insufficient time coverage to fit the selected sine frequencies.")
    return [float(np.hypot(*coefficients[2 * i:2 * i + 2])) for i in range(len(frequencies))]


def recording_harmonic_analysis(data: SignalData, fundamental_hz=50.0, highest_harmonic=10,
                                extra_frequencies=()):
    """Joint raw-waveform harmonic amplitudes and THD-F, excluding DC/noise."""
    if (not np.isfinite(fundamental_hz) or fundamental_hz <= 0 or isinstance(highest_harmonic, bool)
            or not isinstance(highest_harmonic, int) or not 2 <= highest_harmonic <= 50):
        raise ValueError("THD requires a positive fundamental and highest harmonic between 2 and 50")
    if data.duration * fundamental_hz < 5:
        raise ValueError("THD requires at least five fundamental cycles")
    harmonics = list(fundamental_hz * np.arange(1, highest_harmonic + 1))
    frequencies = list(dict.fromkeys(harmonics + list(extra_frequencies)))
    fitted = recording_amplitudes(data, frequencies)
    amplitudes = np.asarray(fitted[:highest_harmonic])
    floor = max(1e-12, float(np.max(np.abs(data.amplitude))) * 1e-10)
    if not np.isfinite(amplitudes[0]) or amplitudes[0] <= floor:
        raise ValueError("THD is undefined because the fundamental amplitude is zero or too small")
    return {"thd_percent": float(100 * np.linalg.norm(amplitudes[1:]) / amplitudes[0]),
            "amplitudes": dict(zip(frequencies, fitted))}


def recording_thd(data: SignalData, fundamental_hz=50.0, highest_harmonic=10) -> float:
    """Unfiltered THD-F (%): 100 * sqrt(A2² + ... + AH²) / A1."""
    return recording_harmonic_analysis(data, fundamental_hz, highest_harmonic)["thd_percent"]
