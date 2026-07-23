"""Turn raw vibration traces into fixed-length windows and features.

Two representations are supported so the autoencoder result can be compared
against a like-for-like alternative rather than reported in isolation:

* ``raw``      -- amplitude-normalised time windows (input to the 1-D conv AE)
* ``logspec``  -- log magnitude spectrum of each window (input to the dense AE)

Per-window amplitude normalisation is deliberate: without it the model can
separate healthy from faulty on overall vibration energy alone, which is a
trivial detector and tells you nothing about whether the AE learned structure.
"""

from __future__ import annotations

import numpy as np


def window(signal: np.ndarray, size: int, hop: int) -> np.ndarray:
    """Split a 1-D signal into overlapping windows -> (n_windows, size)."""
    if signal.size < size:
        return np.empty((0, size))
    n = 1 + (signal.size - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return signal[idx]


def normalise_windows(windows: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean, unit-RMS each window independently."""
    centred = windows - windows.mean(axis=1, keepdims=True)
    rms = np.sqrt((centred**2).mean(axis=1, keepdims=True))
    return centred / (rms + eps)


def log_spectrum(windows: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """One-sided log magnitude spectrum, DC bin dropped -> (n, size//2)."""
    tapered = windows * np.hanning(windows.shape[1])[None, :]
    magnitude = np.abs(np.fft.rfft(tapered, axis=1))[:, 1:]
    return np.log(magnitude + eps)


def featurise(signal: np.ndarray, cfg: dict) -> np.ndarray:
    """Apply the configured representation to one raw trace."""
    windows = normalise_windows(window(signal, cfg["window"], cfg["hop"]))
    if cfg["representation"] == "raw":
        return windows.astype(np.float32)
    if cfg["representation"] == "logspec":
        return log_spectrum(windows).astype(np.float32)
    raise ValueError(f"unknown representation: {cfg['representation']!r}")


class StandardScaler:
    """Feature-wise standardisation, fitted on healthy training data only."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("scaler not fitted")
        return ((x - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def rms_baseline(windows_raw: np.ndarray) -> np.ndarray:
    """Kurtosis of each unnormalised window -- the classic cheap detector.

    Included so the autoencoder has something honest to beat.
    """
    centred = windows_raw - windows_raw.mean(axis=1, keepdims=True)
    variance = (centred**2).mean(axis=1)
    fourth = (centred**4).mean(axis=1)
    return fourth / (variance**2 + 1e-12)
