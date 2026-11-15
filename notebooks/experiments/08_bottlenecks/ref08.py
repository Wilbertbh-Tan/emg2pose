"""Reference answers for notebook 08 (stage 6: do the bottlenecks keep the pose information?).

Metrics, ridge, alpha selection and lag features come from notebook 05's reference (ref05.py); the
reduced-rank regression ("shared subspace") comes from notebook 06's reference (ref06.py); NMF and PCA
from notebook 02 (ref02.py). Everything is plain NumPy.

Shapes: N frames, J = 20 joint angles, k = size of the sEMG bottleneck, K_POSE = size of the pose bottleneck.
"""

from __future__ import annotations

import numpy as np
import ref06
from ref05 import angular_error, choose_alpha, lag_features, ridge_fit, ridge_predict, score  # noqa: F401


# ── Pose bottleneck ──────────────────────────────────────────────────────────
def fit_and_score_pose_bottleneck(F_train: np.ndarray, Y_train: np.ndarray, F_test: np.ndarray,
                                  Y_test: np.ndarray, pose_mean: np.ndarray, pose_comps: np.ndarray,
                                  alpha: float = 1.0) -> dict[str, float]:
    """Decoder with a pose bottleneck: features -> K_POSE scores -> 20 angles.

    pose_mean (20,) and pose_comps (K_POSE, 20, orthonormal rows) define the bottleneck: eigenpostures from
    pca_fit, the pose-side directions of a reduced-rank regression, or random directions.
    """
    targets = (Y_train - pose_mean) @ pose_comps.T
    w, b = ridge_fit(F_train, targets, alpha)
    Y_hat = ridge_predict(F_test, w, b) @ pose_comps + pose_mean
    return score(Y_test, Y_hat)


# ── Pose-informed bottlenecks from the shared subspace (notebook 06) ─────────
def rrr_channel_basis(X: np.ndarray, Y: np.ndarray, k: int, alpha: float = 1.0
                      ) -> tuple[np.ndarray, np.ndarray]:
    """EMG-side bottleneck chosen for the POSE: the k channel directions a rank-k regression uses.

    X (T, 16) lagged log-EMG on the fit frames, Y (T, 20) pose on the same frames.
    Returns Q (16, k), orthonormal columns, and x_mean (16,). Latents of any EMG: (log_emg - x_mean) @ Q.
    """
    B_r, x_mean, _, _ = ref06.rrr_fit(X, Y, k, alpha)
    return ref06.predictive_subspace(B_r, k), x_mean


def rrr_pose_basis(X: np.ndarray, Y: np.ndarray, K: int, alpha: float = 1.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Pose-side bottleneck chosen for the EMG: the K pose directions the EMG can predict best.

    Returns (y_mean (20,), comps (K, 20)), with the same layout as pca_fit, so it plugs into the pose bottleneck.
    """
    _, _, y_mean, V = ref06.rrr_fit(X, Y, K, alpha)
    return y_mean, V.T


def projection_ceiling(Y: np.ndarray, pose_mean: np.ndarray, pose_comps: np.ndarray) -> float:
    """Angular error of the TRUE poses squeezed through the pose bottleneck (the best any decoder could do)."""
    return angular_error(Y, (Y - pose_mean) @ pose_comps.T @ pose_comps + pose_mean)


# ── Calibration ──────────────────────────────────────────────────────────────
def offset_recalibrate(pred: np.ndarray, target: np.ndarray, cal: np.ndarray) -> np.ndarray:
    """Shift every prediction by the mean error on the calibration frames (refit only the offset).

    pred, target : (N, J); cal : (N,) bool. With no calibration frames, pred is returned unchanged.
    """
    if not cal.any():
        return pred
    return pred + (target[cal] - pred[cal]).mean(axis=0)
