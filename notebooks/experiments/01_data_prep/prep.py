"""Reference data prep for the THANOS notebooks.

Builds ``data/processed_mini.npz`` from the 30 raw emg2pose mini recordings, so
every teammate works on identical data. Notebooks 02-09 run this script when
the file is missing or was built by an older version (``DATA_VERSION``).

Notebook 01 walks through the same steps as fill-in-the-blank exercises and
compares your version against this one. Try the notebook before reading this.

Run from anywhere inside the repo:
    python notebooks/experiments/01_data_prep/prep.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import h5py
import numpy as np
from scipy import signal

FS_OUT: int = 50  # samples per second after downsampling
TRAIN_FRACTION: float = 0.8
GAP: int = FS_OUT  # 1 s gap between train and test, so smoothing can't leak across
POSE_CUTOFF_HZ: float = 6.0  # low-pass for the pose (standard for human movement; keeps ~96% of its power)
POSE_PAD: int = 2  # frames (40 ms) marked invalid on each side of a tracking dropout
POSE_MIN_LEN: int = 25  # valid stretches shorter than 0.5 s are marked invalid
DATA_VERSION: int = 2  # bump when the pipeline changes; notebooks rebuild an older processed_mini.npz
MINI_URL: str = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar"


def repo_root() -> Path:
    """Return the repo root (the folder holding setup.py), starting from this file."""
    root = Path(__file__).resolve().parent
    while not (root / "setup.py").exists() and root != root.parent:
        root = root.parent
    return root


def find_data_dir(root: Path, download: bool = True) -> Path:
    """Return the folder of mini-dataset .hdf5 files, downloading it (~617 MB) if absent."""
    for candidate in (root / "data" / "emg2pose_dataset_mini", root / "emg2pose_dataset_mini"):
        if candidate.exists():
            return candidate
    if not download:
        raise FileNotFoundError("emg2pose_dataset_mini not found in data/ or the repo root")
    target = root / "data"
    target.mkdir(exist_ok=True)
    tar = target / "emg2pose_dataset_mini.tar"
    subprocess.run(["curl", "-s", MINI_URL, "-o", str(tar)], check=True)
    subprocess.run(["tar", "-xf", str(tar), "-C", str(target)], check=True)
    tar.unlink()
    return target / "emg2pose_dataset_mini"


def load_recording(path: str | Path) -> tuple[np.ndarray, np.ndarray, float, str, str]:
    """Load one recording: raw EMG (T, 16), pose in degrees (T, 20), sample rate, stage, side."""
    with h5py.File(path, "r") as f:
        g = f["emg2pose"]
        ts = g["timeseries"]
        emg_raw = ts["emg"][:].astype(np.float32)
        pose_deg = np.rad2deg(ts["joint_angles"][:].astype(np.float32))
        fs = float(1.0 / np.median(np.diff(ts["time"][:1000])))
        return emg_raw, pose_deg, fs, str(g.attrs["stage"]), str(g.attrs["side"])


def to_envelope(emg_raw: np.ndarray, fs: float, fs_out: int = FS_OUT) -> np.ndarray:
    """Raw EMG (T, 16) -> scaled, non-negative envelope (T // (fs/fs_out), 16)."""
    nyq = fs / 2
    sos_bp = signal.butter(4, [20 / nyq, min(450, 0.95 * nyq) / nyq], btype="bandpass", output="sos")
    b_notch, a_notch = signal.iirnotch(60.0, Q=30.0, fs=fs)
    sos_lp = signal.butter(4, 10 / nyq, btype="lowpass", output="sos")

    x = signal.sosfiltfilt(sos_bp, emg_raw, axis=0)
    x = signal.filtfilt(b_notch, a_notch, x, axis=0)
    env = signal.sosfiltfilt(sos_lp, np.abs(x), axis=0)
    env = np.clip(env, 1e-8, None)  # strictly non-negative
    env = env[:: int(round(fs / fs_out))]  # downsample
    return (env / (np.percentile(env, 95) + 1e-8)).astype(np.float32)


def valid_pose_mask(pose: np.ndarray) -> np.ndarray:
    """(T, 20) pose -> (T,) bool, False where tracking failed (all angles exactly 0, or any NaN)."""
    return ~np.all(np.isclose(pose, 0.0), axis=1) & ~np.isnan(pose).any(axis=1)


def smooth_pose(pose: np.ndarray, valid: np.ndarray, fs_out: int = FS_OUT, cutoff: float = POSE_CUTOFF_HZ,
                pad: int = POSE_PAD, min_len: int = POSE_MIN_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Clean the 50 Hz pose of one recording.

    1. Shrink the valid mask: mark ``pad`` frames on each side of every dropout invalid (tracking is wobbly next
       to a failure), then drop valid stretches shorter than ``min_len`` frames.
    2. Low-pass each remaining valid stretch separately (4th-order Butterworth, zero phase), never across a
       dropout, so the zeros of a failed frame can't smear into good frames.

    pose (T, 20), valid (T,) bool -> smoothed pose (T, 20) float32 (invalid rows unchanged), new valid (T,) bool.
    """
    bad = ~valid
    grown = bad.copy()
    for k in range(1, pad + 1):
        grown[k:] |= bad[:-k]
        grown[:-k] |= bad[k:]
    ok = ~grown
    edges = np.diff(np.r_[0, ok.astype(int), 0])
    starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
    out = pose.astype(np.float32).copy()
    sos = signal.butter(4, cutoff, fs=fs_out, output="sos")
    for a, b in zip(starts, stops):
        if b - a < min_len:
            ok[a:b] = False
        else:
            out[a:b] = signal.sosfiltfilt(sos, pose[a:b], axis=0)
    return out, ok


def split_labels(n: int, train_fraction: float = TRAIN_FRACTION, gap: int = GAP) -> np.ndarray:
    """(n,) int8 labels: 0 = train (first part), -1 = gap, 1 = test (the rest)."""
    lab = np.zeros(n, dtype=np.int8)
    cut = int(train_fraction * n)
    lab[cut: cut + gap] = -1
    lab[cut + gap:] = 1
    return lab


def build_dataset(files: list[str]) -> dict[str, np.ndarray]:
    """Run every recording through the pipeline and stack them into one dict of arrays."""
    cols: dict[str, list[np.ndarray]] = {k: [] for k in ["emg", "pose", "pose_valid", "rec", "split"]}
    stages: list[str] = []
    sides: list[str] = []
    names: list[str] = []

    for r, path in enumerate(files):
        emg_raw, pose_deg, fs, stage, side = load_recording(path)
        env = to_envelope(emg_raw, fs)
        p = pose_deg[:: int(round(fs / FS_OUT))][: len(env)]
        env = env[: len(p)]

        p, p_valid = smooth_pose(p, valid_pose_mask(p))

        values = [env, p.astype(np.float32), p_valid,
                  np.full(len(env), r, np.int16), split_labels(len(env))]
        for k, v in zip(cols, values):
            cols[k].append(v)
        stages.append(stage)
        sides.append(side)
        names.append(Path(path).stem)

    data = {k: np.concatenate(v) for k, v in cols.items()}
    data.update(stage=np.array(stages), side=np.array(sides),
                filename=np.array(names), fs=np.array(FS_OUT), version=np.array(DATA_VERSION))
    return data


def is_current(path: Path) -> bool:
    """True if ``path`` exists and was built by this version of the pipeline."""
    if not path.exists():
        return False
    with np.load(path) as d:
        return "version" in d.files and int(d["version"]) == DATA_VERSION


def main() -> None:
    root = repo_root()
    files = sorted(str(p) for p in find_data_dir(root).glob("*.hdf5"))
    data = build_dataset(files)

    out = root / "data" / "processed_mini.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, **data)

    assert data["emg"].shape[1] == 16 and data["pose"].shape[1] == 20
    assert (data["emg"] >= 0).all(), "envelope must be non-negative for NMF"
    print(f"Saved {out.relative_to(root)}: {len(data['emg'])} frames from {len(files)} recordings | "
          f"train {np.mean(data['split'] == 0):.0%} · test {np.mean(data['split'] == 1):.0%} · "
          f"pose valid {data['pose_valid'].mean():.1%}")


if __name__ == "__main__":
    main()
