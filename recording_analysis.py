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
    columns = []
    for frequency in frequencies:
        phase = 2 * np.pi * frequency * (t - t[0])
        columns.extend((np.sin(phase), np.cos(phase)))
    design = np.column_stack([*columns, np.ones(t.size)])
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("Insufficient time coverage to fit the selected sine frequencies.")
    return [float(np.hypot(*coefficients[2 * i:2 * i + 2])) for i in range(len(frequencies))]
