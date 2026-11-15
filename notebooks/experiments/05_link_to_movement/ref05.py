"""Reference answers for notebook 05 (stage 4: what does the EMG encode?).

The notebook asks you to write each of these functions yourself; its check cells compare your
output with these. Everything is plain NumPy: no scikit-learn. Notebook 08 (bottlenecks) imports the
metrics, ridge and lag helpers from here, so a blank in notebook 05 never breaks it.

Shapes: T = frames, d = input features (16 channels, k latents, ...), J = output columns
(20 joint angles, or K_POSE scores).
"""

from __future__ import annotations

import numpy as np


# ── Metrics ──────────────────────────────────────────────────────────────────
def angular_error(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    """Mean absolute joint-angle error in degrees, over all joints and frames."""
    return float(np.mean(np.abs(Y - Y_hat)))


def r2(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    """R² per column, then averaged over columns (sklearn's r2_score default, 'uniform_average').

    Per column: 1 - sum((y - y_hat)^2) / sum((y - mean(y))^2). 1 = perfect, 0 = no better than the mean.
    """
    ss_res = np.sum((Y - Y_hat) ** 2, axis=0)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2, axis=0)
    return float(np.mean(1.0 - ss_res / ss_tot))


def score(Y: np.ndarray, Y_hat: np.ndarray) -> dict[str, float]:
    """Both metrics at once."""
    return {"angle_deg": angular_error(Y, Y_hat), "r2": r2(Y, Y_hat)}


# ── Ridge regression in closed form ──────────────────────────────────────────
def ridge_fit(F: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Fit Y ≈ F @ w + b, minimising sum((Y - F w - b)^2) + alpha * sum(w^2).

    The intercept b is NOT penalised. Centring F and Y removes it from the problem; then
        w = (Fcᵀ Fc + alpha I)⁻¹ Fcᵀ Yc      (solved with np.linalg.solve, never an explicit inverse)
        b = mean(Y) - mean(F) @ w
    Returns w (d, J) and b (J,).
    """
    f_mean, y_mean = F.mean(axis=0), Y.mean(axis=0)
    Fc, Yc = F - f_mean, Y - y_mean
    w = np.linalg.solve(Fc.T @ Fc + alpha * np.eye(F.shape[1]), Fc.T @ Yc)
    return w, y_mean - f_mean @ w


def ridge_predict(F: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(T, d) features -> (T, J) predictions."""
    return F @ w + b


def fit_and_score_ridge(F_train: np.ndarray, Y_train: np.ndarray, F_test: np.ndarray,
                        Y_test: np.ndarray, alpha: float = 1.0) -> dict[str, float]:
    """Linear decoder features -> pose: fit on train, score on test."""
    w, b = ridge_fit(F_train, Y_train, alpha)
    return score(Y_test, ridge_predict(F_test, w, b))


# ── Choosing alpha without touching the test set ─────────────────────────────
def inner_split(rec_tr: np.ndarray, frac: float = 0.8) -> np.ndarray:
    """(T,) recording ids of the training rows -> (T,) bool, True = inner-train, False = validation.

    Within each recording the first `frac` of its training rows are inner-train and the rest are
    validation, so validation is always LATER in time than the data it is judged against
    (the same idea as the train/test split itself).
    """
    keep = np.zeros(len(rec_tr), dtype=bool)
    for r in np.unique(rec_tr):
        idx = np.flatnonzero(rec_tr == r)
        keep[idx[: int(frac * len(idx))]] = True
    return keep


def choose_alpha(F: np.ndarray, Y: np.ndarray, rec_tr: np.ndarray,
                 alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)) -> float:
    """The alpha with the lowest validation angular error, using training rows only."""
    inner = inner_split(rec_tr)
    errs = [angular_error(Y[~inner], ridge_predict(F[~inner], *ridge_fit(F[inner], Y[inner], a)))
            for a in alphas]
    return float(alphas[int(np.argmin(errs))])


# ── EMG history as features ───────────────────────────────────────────
def lag_features(X: np.ndarray, rec: np.ndarray, lags: tuple[int, ...] = (5, 10, 20)) -> np.ndarray:
    """(N, d) features -> (N, d * (1 + len(lags))): [X(t), X(t - lag_1), X(t - lag_2), ...].

    Works on the full, time-ordered arrays (all N frames), so call it BEFORE masking with tr / te.
    A lag never reaches into the previous recording: it stops at the recording's first frame.
    """
    t = np.arange(len(X))
    is_start = np.r_[True, rec[1:] != rec[:-1]]
    first = np.maximum.accumulate(np.where(is_start, t, 0))      # first frame of each frame's recording
    return np.hstack([X] + [X[np.maximum(t - lag, first)] for lag in lags])


# ── Stage 4: honest baselines, per-block scores, velocity ────────────────────
def per_recording_median(Y: np.ndarray, rec: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """(N, J) prediction: each recording's own median over its fit_mask frames (global median if it has none)."""
    out = np.tile(np.median(Y[fit_mask], axis=0), (len(Y), 1))
    for r in np.unique(rec):
        m = (rec == r) & fit_mask
        if m.any():
            out[rec == r] = np.median(Y[m], axis=0)
    return out


def per_block_error(Y: np.ndarray, Y_hat: np.ndarray, mask: np.ndarray, block: np.ndarray,
                    n_blocks: int = 15) -> np.ndarray:
    """(n_blocks,) mean angular error over the mask frames of each block (NaN if a block has none)."""
    out = np.full(n_blocks, np.nan)
    for b in range(n_blocks):
        m = mask & (block == b)
        if m.any():
            out[b] = angular_error(Y[m], Y_hat[m])
    return out


def joint_velocity(pose: np.ndarray, rec: np.ndarray, valid: np.ndarray, fs: int = 50
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference joint velocity in °/s, never across a gap or a recording boundary.

    v(t) = (pose(t+1) - pose(t-1)) / 2 * fs. Returns V (N, J) and ok (N,) bool: True only where frames
    t-1, t and t+1 are all valid and in the same recording (V is 0 elsewhere).
    """
    V = np.zeros_like(pose)
    ok = np.zeros(len(pose), dtype=bool)
    same = (rec[2:] == rec[1:-1]) & (rec[1:-1] == rec[:-2])
    ok[1:-1] = valid[2:] & valid[1:-1] & valid[:-2] & same
    V[1:-1] = (pose[2:] - pose[:-2]) / 2 * fs
    V[~ok] = 0.0
    return V, ok
