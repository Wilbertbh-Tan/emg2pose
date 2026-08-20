"""Windowed feature extraction for emg2pose EDA.

Turns raw 2 kHz, 16-channel sEMG into a windowed feature matrix suitable for
PCA / UMAP / PHATE, keeping the paired joint angles and session metadata aligned
with each window.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy import signal

SAMPLE_RATE: float = 2000.0
N_CHANNELS: int = 16
N_JOINTS: int = 20

# sEMG power is concentrated well below 500 Hz; these bands follow standard
# practice for surface EMG feature extraction.
BANDS: tuple[tuple[float, float], ...] = (
    (20.0, 50.0),
    (50.0, 100.0),
    (100.0, 200.0),
    (200.0, 450.0),
)


@dataclass
class SessionMeta:
    """Metadata attached to a single emg2pose recording."""

    filename: str
    user: str
    stage: str
    side: str
    split: str
    generalization: str
    held_out_user: bool
    held_out_stage: bool
    sample_rate: float
    n_samples: int


def read_session(path: str | Path) -> tuple[np.ndarray, np.ndarray, SessionMeta]:
    """Read one HDF5 session.

    Returns (emg, joint_angles, meta) where emg is (n_samples, 16) and
    joint_angles is (n_samples, 20).
    """
    with h5py.File(path, "r") as f:
        group = f["emg2pose"]
        attrs = dict(group.attrs)
        series = group["timeseries"]
        emg = np.asarray(series["emg"], dtype=np.float32)
        joint_angles = np.asarray(series["joint_angles"], dtype=np.float32)

    meta = SessionMeta(
        filename=str(attrs["filename"]),
        user=str(attrs["user"]),
        stage=str(attrs["stage"]),
        side=str(attrs["side"]),
        split=str(attrs["split"]),
        generalization=str(attrs["generalization"]),
        held_out_user=bool(attrs["held_out_user"]),
        held_out_stage=bool(attrs["held_out_stage"]),
        sample_rate=float(attrs["sample_rate"]),
        n_samples=int(emg.shape[0]),
    )
    return emg, joint_angles, meta


def window_features(
    emg: np.ndarray,
    joint_angles: np.ndarray,
    window_s: float = 0.25,
    hop_s: float = 0.125,
    sample_rate: float = SAMPLE_RATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-window features from raw sEMG.

    Features per window are, for each of the 16 channels: log RMS amplitude,
    then log power in each of the four bands in ``BANDS``. That gives
    16 * 5 = 80 features per window.

    Returns (features, joint_angles_per_window, window_start_times).
    """
    win = int(round(window_s * sample_rate))
    hop = int(round(hop_s * sample_rate))
    n_windows = 1 + (emg.shape[0] - win) // hop
    if n_windows < 1:
        raise ValueError(f"session too short for a {window_s}s window")

    n_features = N_CHANNELS * (1 + len(BANDS))
    features = np.empty((n_windows, n_features), dtype=np.float32)
    angles = np.empty((n_windows, joint_angles.shape[1]), dtype=np.float32)
    times = np.empty(n_windows, dtype=np.float64)

    # Frequency bins are the same for every window, so compute the band masks once.
    freqs = np.fft.rfftfreq(win, d=1.0 / sample_rate)
    band_masks = [(freqs >= lo) & (freqs < hi) for lo, hi in BANDS]
    taper = signal.windows.hann(win).astype(np.float32)

    eps = 1e-10
    for i in range(n_windows):
        start = i * hop
        block = emg[start : start + win]  # (win, 16)

        rms = np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=0))

        spectrum = np.fft.rfft(block * taper[:, None], axis=0)
        power = np.abs(spectrum) ** 2
        band_power = np.stack(
            [power[mask].sum(axis=0) for mask in band_masks], axis=0
        )  # (n_bands, 16)

        features[i, :N_CHANNELS] = np.log10(rms + eps)
        features[i, N_CHANNELS:] = np.log10(band_power + eps).ravel()

        # Represent the window's pose by its midpoint sample.
        angles[i] = joint_angles[start + win // 2]
        times[i] = start / sample_rate

    return features, angles, times


def build_matrix(
    data_dir: str | Path,
    window_s: float = 0.25,
    hop_s: float = 0.125,
    max_files: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[SessionMeta], np.ndarray]:
    """Build a feature matrix across every session in ``data_dir``.

    Returns (features, joint_angles, metas, session_index) where session_index[i]
    gives the index into ``metas`` for row i of the feature matrix.
    """
    paths = sorted(glob.glob(str(Path(data_dir) / "*.hdf5")))
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"no .hdf5 files under {data_dir}")

    all_features: list[np.ndarray] = []
    all_angles: list[np.ndarray] = []
    metas: list[SessionMeta] = []
    session_index: list[np.ndarray] = []

    for idx, path in enumerate(paths):
        emg, joint_angles, meta = read_session(path)
        feats, angles, _ = window_features(
            emg, joint_angles, window_s=window_s, hop_s=hop_s,
            sample_rate=meta.sample_rate,
        )
        all_features.append(feats)
        all_angles.append(angles)
        metas.append(meta)
        session_index.append(np.full(feats.shape[0], idx, dtype=np.int32))

    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_angles, axis=0),
        metas,
        np.concatenate(session_index, axis=0),
    )
