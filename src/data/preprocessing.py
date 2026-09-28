"""
Signal processing, filtering, windowing, and spectral feature extraction for bridge sensors.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy import signal


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 35.0,
    fs: float = 100.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter along the last axis.
    Removes low-frequency baseline drift (< 0.5 Hz) and unmodeled high-frequency noise (> 35 Hz).
    """
    nyquist = 0.5 * fs
    low = max(1e-4, lowcut / nyquist)
    high = min(0.999, highcut / nyquist)
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    
    # Apply along last axis (time axis)
    filtered = signal.sosfiltfilt(sos, data, axis=-1)
    return filtered


def create_sliding_windows(
    data: np.ndarray,
    window_size: int = 256,
    step_size: int = 64,
    labels: Optional[np.ndarray] = None,
    health_index: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Partition continuous multi-channel time-series into overlapping sliding windows.
    Args:
        data: Shape [num_sensors, num_timesteps] or [num_timesteps, num_sensors]
        window_size: Number of timesteps per window (e.g. 256 timesteps @ 100Hz = 2.56s)
        step_size: Stride between successive windows (overlap = window_size - step_size)
        labels: Optional binary anomaly labels [num_timesteps]
        health_index: Optional continuous structural health index [num_timesteps]
    Returns:
        windows: Shape [num_windows, num_sensors, window_size]
        window_labels: Shape [num_windows] (1 if any sample in window is anomalous, else 0)
        window_shi: Shape [num_windows] (average health index in window)
    """
    # Ensure shape is [num_sensors, num_timesteps]
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.shape[0] > data.shape[1]:
        data = data.T

    num_sensors, num_timesteps = data.shape
    if num_timesteps < window_size:
        raise ValueError(f"Time-series length ({num_timesteps}) is shorter than window size ({window_size})")

    num_windows = (num_timesteps - window_size) // step_size + 1
    windows = np.zeros((num_windows, num_sensors, window_size), dtype=np.float32)

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        windows[i] = data[:, start:end]

    window_labels = None
    if labels is not None:
        window_labels = np.zeros(num_windows, dtype=np.int64)
        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            # Window is labeled anomalous if at least 20% of timesteps are anomalous
            window_labels[i] = int(np.mean(labels[start:end]) >= 0.20)

    window_shi = None
    if health_index is not None:
        window_shi = np.zeros(num_windows, dtype=np.float32)
        for i in range(num_windows):
            start = i * step_size
            end = start + window_size
            window_shi[i] = float(np.mean(health_index[start:end]))

    return windows, window_labels, window_shi


class BridgeFeatureScaler:
    """
    Robust sensor-wise feature scaler fitted on baseline healthy vibration data.
    Normalizes multi-channel window RMS energy so that traffic volume variations (cars vs trucks)
    do not distort modal feature extraction, enabling clean structural damage detection.
    """

    def __init__(self, method: str = "energy"):
        self.method = method
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, windows: np.ndarray) -> "BridgeFeatureScaler":
        """
        Fit scaler on windows of shape [num_windows, num_sensors, window_size]
        Computes per-sensor DC offset across baseline data.
        """
        self.mean = np.mean(windows, axis=(0, 2), keepdims=True)
        self.std = np.std(windows, axis=(0, 2), keepdims=True)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        """Standardize windows with zero-mean and window RMS energy normalization."""
        if self.mean is None:
            raise ValueError("Scaler is not fitted yet. Call fit() first.")
        centered = windows - self.mean
        # Compute RMS energy per window across all sensors and timesteps: shape [N, 1, 1]
        win_rms = np.sqrt(np.mean(centered ** 2, axis=(1, 2), keepdims=True)) + 1e-4
        return (centered / win_rms).astype(np.float32)

    def inverse_transform(self, windows: np.ndarray) -> np.ndarray:
        """Invert normalization."""
        if self.mean is None or self.std is None:
            raise ValueError("Scaler is not fitted yet.")
        return (windows * self.std + self.mean).astype(np.float32)


def compute_fft_spectrum(
    signal_1d: np.ndarray,
    fs: float = 100.0,
    top_k_peaks: int = 4,
) -> Dict[str, np.ndarray]:
    """
    Compute single-sided Fast Fourier Transform (FFT) spectrum and identify peak frequencies.
    """
    n = len(signal_1d)
    # Apply Hanning window to mitigate spectral leakage
    w = np.hanning(n)
    sig_win = (signal_1d - np.mean(signal_1d)) * w

    fft_vals = np.fft.rfft(sig_win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    magnitude = (2.0 / n) * np.abs(fft_vals)

    # Detect prominent spectral peaks (modal frequencies)
    peak_indices, properties = signal.find_peaks(
        magnitude,
        distance=int(0.8 * (n / fs)), # at least 0.8 Hz apart
        prominence=0.05 * np.max(magnitude) if np.max(magnitude) > 0 else None,
    )
    
    if len(peak_indices) > 0:
        sorted_by_prom = sorted(
            peak_indices,
            key=lambda idx: magnitude[idx],
            reverse=True,
        )[:top_k_peaks]
        peak_freqs = freqs[sorted_by_prom]
        peak_mags = magnitude[sorted_by_prom]
    else:
        peak_freqs = np.array([])
        peak_mags = np.array([])

    return {
        "frequencies": freqs,
        "magnitude": magnitude,
        "peak_frequencies": peak_freqs,
        "peak_magnitudes": peak_mags,
    }


def compute_spectrogram(
    signal_1d: np.ndarray,
    fs: float = 100.0,
    nperseg: int = 128,
    noverlap: int = 96,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Short-Time Fourier Transform (STFT) spectrogram."""
    f, t, Sxx = signal.spectrogram(
        signal_1d - np.mean(signal_1d),
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
    )
    return f, t, Sxx
