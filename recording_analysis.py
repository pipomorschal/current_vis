"""Extract one peak 50 Hz amplitude from each recorded waveform."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from signal_data_class import SignalData


def recording_amplitude(data: SignalData, highpass_hz: float = 0,
                        lowpass_hz: float = 0) -> float:
    t = np.asarray(data.time, dtype=float).reshape(-1)
    y = np.asarray(data.amplitude, dtype=float).reshape(-1)
    fs = float(data.sampling_rate)
    if t.size != y.size or t.size < 4:
        raise ValueError("At least four matching time and amplitude samples are required.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("Waveform contains non-finite samples.")
    if not np.isfinite(fs) or fs <= 100:
        raise ValueError("Sampling rate must exceed 100 Hz to resolve 50 Hz.")
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
    phase = 2 * np.pi * 50 * (t - t[0])
    design = np.column_stack((np.sin(phase), np.cos(phase), np.ones(t.size)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 3:
        raise ValueError("Insufficient time coverage to fit a 50 Hz sine.")
    return float(np.hypot(*coefficients[:2]))
