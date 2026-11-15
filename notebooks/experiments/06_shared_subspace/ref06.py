"""Reference answers for notebook 06 (stage 4b: what is shared between EMG and pose?).

Core functions below are also imported by notebooks 07 (dynamics in the shared subspace) and 08 (pose-informed
bottleneck), so their signatures are fixed. Notebook 06 may add more functions after the "Notebook-06 extras"
marker. Plain NumPy only.

Shapes: N frames, D EMG features (16 channels, or 16 x number of lags), J pose columns (20 joints), r = rank.
"""

from __future__ import annotations

import numpy as np


# ── Features ─────────────────────────────────────────────────────────────────
def lag_features(X: np.ndarray, rec: np.ndarray, lags: list[int]) -> np.ndarray:
    """Stack past copies of X: column block i holds X from ``lags[i]`` frames earlier, within each recording.

    The first ``lag`` frames of a recording reuse its first frame (no data leaks in from the previous recording).
    (N, D) -> (N, D * len(lags)). lags are non-negative (0 = now); at 50 Hz, 10 frames = 200 ms.
    """
    blocks = []
    for lag in lags:
        out = np.empty_like(X)
        for r in np.unique(rec):
            idx = np.where(rec == r)[0]
            x = X[idx]
            if lag == 0:
                out[idx] = x
            else:
                out[idx[lag:]] = x[:-lag]
                out[idx[:lag]] = x[0]
        blocks.append(out)
    return np.concatenate(blocks, axis=1)


def log_emg(emg: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """log(envelope + eps): EMG amplitude noise grows with the signal, and the log evens that out."""
    return np.log(emg + eps)


# ── Reduced-rank regression (the "communication subspace") ───────────────────
def ridge_coef(Xc: np.ndarray, Yc: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge coefficients for centred data: solve (Xc'Xc + alpha I) B = Xc'Yc. (D, J)."""
    return np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ Yc)


def rrr_fit(X: np.ndarray, Y: np.ndarray, rank: int, alpha: float = 1.0
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reduced-rank (ridge) regression: the best map X -> Y that passes through only ``rank`` dimensions.

    Steps: centre X and Y; full ridge fit B; the fitted values Yhat = Xc B; their top-``rank`` right singular
    vectors V (J, rank) span the part of pose that EMG can predict; project B onto them: B_r = B V V'.

    Returns B_r (D, J), x_mean (D,), y_mean (J,), V (J, rank).
    """
    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - x_mean, Y - y_mean
    B = ridge_coef(Xc, Yc, alpha)
    _, _, Vt = np.linalg.svd(Xc @ B, full_matrices=False)
    V = Vt[:rank].T
    return B @ V @ V.T, x_mean, y_mean, V


def rrr_predict(X: np.ndarray, B_r: np.ndarray, x_mean: np.ndarray, y_mean: np.ndarray) -> np.ndarray:
    """(N, D) -> (N, J) predictions of a fitted reduced-rank map."""
    return (X - x_mean) @ B_r + y_mean


def predictive_subspace(B_r: np.ndarray, rank: int) -> np.ndarray:
    """Orthonormal basis (D, rank) of the EMG-side directions the reduced-rank map actually uses
    (top left singular vectors of B_r). Columns are the "shared latents" seen from the EMG side."""
    U, _, _ = np.linalg.svd(B_r, full_matrices=False)
    return U[:, :rank]


def shared_latents(X: np.ndarray, B_r: np.ndarray, x_mean: np.ndarray, rank: int) -> np.ndarray:
    """(N, D) EMG -> (N, rank) shared latent time courses: centred EMG projected on the predictive subspace."""
    return (X - x_mean) @ predictive_subspace(B_r, rank)


# ── Comparing subspaces ──────────────────────────────────────────────────────
def orthonormal(A: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the column space of A (D, r) via QR."""
    Q, _ = np.linalg.qr(A)
    return Q


def principal_angles(Qa: np.ndarray, Qb: np.ndarray) -> np.ndarray:
    """Principal angles in degrees between the column spaces of two orthonormal bases (smallest first)."""
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def random_subspace(D: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """Orthonormal basis of a uniformly random r-dimensional subspace of R^D (the null for angles)."""
    return orthonormal(rng.normal(size=(D, r)))


# ── Notebook-06 extras ───────────────────────────────────────────────────────
def heldout_rrr_errors(X: np.ndarray, Y: np.ndarray, stage_of_rec: np.ndarray, rec: np.ndarray,
                       filename: np.ndarray, valid: np.ndarray, ranks: list[int], alpha: float = 1.0
                       ) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Leave-one-stage-out (14 folds, simultaneous blocks never split) mean absolute pose error.

    Returns fold names, errors (n_folds, len(ranks)) of the rank-r maps, and the median-pose baseline (n_folds,).
    """
    import protocol
    names, errs, base = [], [], []
    for name, tr, te in protocol.leave_one_stage_out(stage_of_rec, rec, valid, filename):
        names.append(name)
        base.append(float(np.mean(np.abs(Y[te] - np.median(Y[tr], axis=0)))))
        row = []
        for r in ranks:
            B, mx, my, _ = rrr_fit(X[tr], Y[tr], r, alpha)
            row.append(float(np.mean(np.abs(Y[te] - rrr_predict(X[te], B, mx, my)))))
        errs.append(row)
    return names, np.array(errs), np.array(base)


def choose_rank(errs: np.ndarray, ranks: list[int], n_boot: int = 2000, seed: int = 0) -> int:
    """Smallest rank whose paired difference from the best rank has a bootstrap CI (over folds) that includes 0
    or lies within 0.1°, i.e. statistically indistinguishable from the best."""
    import protocol
    best = int(np.argmin(errs.mean(axis=0)))
    for i, r in enumerate(ranks):
        _, lo, hi = protocol.session_bootstrap_ci(errs[:, i] - errs[:, best], n_boot=n_boot, seed=seed)
        if lo <= 0.1:
            return int(r)
    return int(ranks[best])


def latent_lag_profile(X0: np.ndarray, Y: np.ndarray, rec: np.ndarray, valid: np.ndarray, Q: np.ndarray,
                       V: np.ndarray, x_mean: np.ndarray, y_mean: np.ndarray, lags: list[int]) -> np.ndarray:
    """Correlation between each shared latent's EMG side, taken ``lag`` frames earlier, and its pose side.

    X0 (N, D) un-lagged EMG features; Q (D, r) EMG-side basis; V (J, r) pose-side basis. EMG side z = (X0 - x_mean) Q,
    pose side y = (Y - y_mean) V. Returns (len(lags), r) correlations over valid frames (sign fixed so the
    maximum is positive). The best lag of latent i is lags[argmax(column i)].
    """
    z = (X0 - x_mean) @ Q
    y = (Y - y_mean) @ V
    out = np.zeros((len(lags), Q.shape[1]))
    for a, lag in enumerate(lags):
        zl = lag_features(z, rec, [lag])
        for i in range(Q.shape[1]):
            out[a, i] = np.corrcoef(zl[valid, i], y[valid, i])[0, 1]
    signs = np.sign(out[np.abs(out).argmax(axis=0), np.arange(Q.shape[1])])
    return out * signs


def angle_null(D: int, r: int, Q: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, min(r, Q.shape[1])) principal angles between Q and n random r-dim subspaces of R^D."""
    return np.array([principal_angles(Q, random_subspace(D, r, rng)) for _ in range(n)])


FINGERS: dict[str, list[int]] = {"thumb": [0, 1, 2, 3], "index": [4, 5, 6, 7], "middle": [8, 9, 10, 11],
                                 "ring": [12, 13, 14, 15], "pinky": [16, 17, 18, 19]}


def finger_directions(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """(5, D) unit EMG direction that best predicts each finger's 4 angles (rank-1 RRR), fingers in FINGERS order,
    sign fixed so the largest-magnitude entry is positive."""
    dirs = []
    for cols in FINGERS.values():
        B, _, _, _ = rrr_fit(X, Y[:, cols], 1, alpha)
        u = predictive_subspace(B, 1)[:, 0]
        dirs.append(u * np.sign(u[np.abs(u).argmax()]))
    return np.array(dirs)


def abs_cos_matrix(dirs: np.ndarray) -> np.ndarray:
    """|cosine| between every pair of unit rows."""
    n = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    return np.abs(n @ n.T)


def variance_in_subspace(X: np.ndarray, Q: np.ndarray) -> float:
    """Share (0-1) of the centred variance of X that lies in the column space of the orthonormal basis Q."""
    Xc = X - X.mean(axis=0)
    return float(np.sum((Xc @ Q) ** 2) / np.sum(Xc ** 2))


def supervised_nmf(X: np.ndarray, Y: np.ndarray, k: int, lam: float = 1.0, n_iter: int = 300, seed: int = 0
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Pose-informed synergies: minimise ||X - C W||^2 + lam * ||Ys - C M||^2 with C, W >= 0.

    X (T, D) non-negative EMG; Y (T, J) pose. Ys = (Y - mean) / overall std of Y, rescaled so the two terms start
    on a comparable footing: Ys is multiplied by sqrt(||X||^2 / ||Ys||^2). Each round:
        W <- W * (C'X) / (C'C W)                     (multiplicative, keeps W >= 0)
        M <- least squares of Ys on C                 (unconstrained)
        C <- max(0, C - step * grad),  grad = (C W - X) W' + lam (C M - Ys) M',  step = 1 / L,
             L = largest eigenvalue of (W W' + lam M M')   (projected gradient, keeps C >= 0)
    Rows of W are scaled to length 1 at the end (C and M absorb the scale).
    Returns C (T, k), W (k, D), M (k, J) mapping C to the SCALED pose, and the loss per round.
    """
    rng = np.random.default_rng(seed)
    Yc = Y - Y.mean(axis=0)
    Ys = Yc / Yc.std() * np.sqrt(np.sum(X ** 2) / np.sum((Yc / Yc.std()) ** 2))
    scale = np.sqrt(X.mean() / k)
    C = scale * rng.random((X.shape[0], k))
    W = scale * rng.random((k, X.shape[1]))
    losses: list[float] = []
    eps = 1e-10
    for _ in range(n_iter):
        W *= (C.T @ X) / (C.T @ C @ W + eps)
        M = np.linalg.solve(C.T @ C + eps * np.eye(k), C.T @ Ys)
        L = np.linalg.eigvalsh(W @ W.T + lam * M @ M.T).max()
        grad = (C @ W - X) @ W.T + lam * (C @ M - Ys) @ M.T
        C = np.maximum(0.0, C - grad / L)
        losses.append(float(np.sum((X - C @ W) ** 2) + lam * np.sum((Ys - C @ M) ** 2)))
    n = np.linalg.norm(W, axis=1) + eps
    return C * n, W / n[:, None], M / n[:, None], losses
