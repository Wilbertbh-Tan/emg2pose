"""Reference answers for notebook 04 (stage 3: do stages share structure?).

Plain NumPy. Subspaces are fitted on TRAIN frames and scored on TEST frames, so a stage's own subspace gets no
in-sample advantage over another stage's subspace.

Shapes: T = frames, D = features (16 EMG channels or 20 joints), d = subspace dimension, S = 15 stages.
"""

from __future__ import annotations

import numpy as np

IDLE_STAGES: tuple[str, ...] = ("unconstrained",)   # the idle hand recorded while the other hand did free style


# ── Stages ───────────────────────────────────────────────────────────────────
def is_free_style(stage: str) -> bool:
    """True for deliberate free-style stages (name contains 'Free'). 'unconstrained' is the IDLE hand, not free style."""
    return "Free" in stage


def is_idle(stage: str) -> bool:
    return stage in IDLE_STAGES


def simultaneous_pairs(stages: list[str], stage_of_rec: np.ndarray, block_of_rec: np.ndarray) -> np.ndarray:
    """(S, S) bool: True where two DIFFERENT stages were recorded at the same time (they share a block).

    Such pairs are excluded from cross-stage comparisons: their similarity could come from the shared session.
    """
    S = len(stages)
    out = np.zeros((S, S), bool)
    for i, a in enumerate(stages):
        for j, b in enumerate(stages):
            if i != j:
                out[i, j] = bool(np.intersect1d(block_of_rec[stage_of_rec == a], block_of_rec[stage_of_rec == b]).size)
    return out


def stage_masks(stage_of_rec: np.ndarray, rec: np.ndarray, within: np.ndarray) -> dict[str, np.ndarray]:
    """{stage: (N,) bool}: frames of that stage (BOTH hands pooled: one session) AND-ed with ``within``."""
    stage_per_frame = stage_of_rec[rec]
    return {str(s): (stage_per_frame == s) & within for s in np.unique(stage_of_rec)}


# ── Subspaces ────────────────────────────────────────────────────────────────
def dims_for_var(X: np.ndarray, var_target: float = 0.9) -> int:
    """Smallest number of principal components explaining >= var_target of X's centred variance."""
    S = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    frac = np.cumsum(S**2) / np.sum(S**2)
    return int(np.searchsorted(frac, var_target) + 1)


def stage_basis(X: np.ndarray, d: int) -> np.ndarray:
    """(D, d) orthonormal COLUMNS spanning X's top-d principal directions (most variance first)."""
    _, _, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    return Vt[:d].T


def captured(X: np.ndarray, Q: np.ndarray) -> float:
    """Variance of X (centred on its own mean) that lies inside the subspace Q: ||Xc Q||^2."""
    Xc = X - X.mean(axis=0)
    return float(np.sum((Xc @ Q) ** 2))


def ratio(X_a: np.ndarray, Q_a: np.ndarray, Q_b: np.ndarray) -> float:
    """Variance-explained ratio: how much of A's variance B's subspace captures, relative to A's own subspace."""
    return captured(X_a, Q_b) / captured(X_a, Q_a)


def random_basis(D: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """(D, d) random orthonormal columns (QR of a Gaussian matrix)."""
    Q, _ = np.linalg.qr(rng.normal(size=(D, d)))
    return Q


def ratio_matrix(X_test: dict[str, np.ndarray], Q: dict[str, np.ndarray]) -> np.ndarray:
    """(S, S) matrix M[a, b] = ratio(X_test[a], Q[a], Q[b]), stages in sorted order. Diagonal = 1."""
    names = sorted(X_test)
    return np.array([[ratio(X_test[a], Q[a], Q[b]) for b in names] for a in names])


def null_ratio_matrix(X_test: dict[str, np.ndarray], Q: dict[str, np.ndarray], n_rand: int = 20,
                      seed: int = 0) -> np.ndarray:
    """(S, S) expected ratio if B's subspace were a RANDOM subspace of the same size as B's."""
    rng = np.random.default_rng(seed)
    names = sorted(X_test)
    D = next(iter(Q.values())).shape[0]
    out = np.zeros((len(names), len(names)))
    for j, b in enumerate(names):
        rand = [random_basis(D, Q[b].shape[1], rng) for _ in range(n_rand)]
        for i, a in enumerate(names):
            out[i, j] = np.mean([ratio(X_test[a], Q[a], R) for R in rand])
    return out


def sharing_index(M: np.ndarray, M_null: np.ndarray) -> np.ndarray:
    """(M - null) / (1 - null): 0 = no better than a random subspace, 1 = as good as the stage's own."""
    return (M - M_null) / (1.0 - M_null)


def off_diagonal_row_means(M: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray:
    """(S,) mean of each row excluding the diagonal (and any True entry of ``exclude``, e.g. simultaneous pairs):
    how well OTHER stages' subspaces capture stage a."""
    S = len(M)
    mask = ~np.eye(S, dtype=bool)
    if exclude is not None:
        mask &= ~exclude
    return np.array([M[i, mask[i]].mean() for i in range(S)])


def principal_angles(Q_a: np.ndarray, Q_b: np.ndarray) -> np.ndarray:
    """Principal angles (degrees, smallest first) between two subspaces with orthonormal columns."""
    s = np.linalg.svd(Q_a.T @ Q_b, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def lag_within(X: np.ndarray, rec: np.ndarray, lag: int) -> np.ndarray:
    """Row t of the output = X[t - lag] from the SAME recording (the first rows repeat the first frame).

    Use it to pair pose at time t with EMG from ``lag`` frames earlier (EMG leads movement).
    """
    out = X.copy()
    if lag == 0:
        return out
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        x = X[idx]
        out[idx] = np.concatenate([np.repeat(x[:1], lag, axis=0), x[: len(idx) - lag]])
    return out


# ── CCA ──────────────────────────────────────────────────────────────────────
def _inv_sqrt(C: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(C)
    return V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T


def cca_fit(X: np.ndarray, Y: np.ndarray, reg: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canonical correlation analysis from scratch.

    Whiten X and Y (C_xx^-1/2, C_yy^-1/2, with a small ridge ``reg`` times the mean variance), then the SVD of the
    whitened cross-covariance gives the canonical correlations (singular values) and directions.
    Returns mean_x (Dx,), mean_y (Dy,), Wx (Dx, m), Wy (Dy, m), rho (m,), m = min(Dx, Dy).
    """
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mx, Y - my
    n = len(X) - 1
    Cxx, Cyy, Cxy = Xc.T @ Xc / n, Yc.T @ Yc / n, Xc.T @ Yc / n
    Cxx = Cxx + reg * np.trace(Cxx) / len(Cxx) * np.eye(len(Cxx))
    Cyy = Cyy + reg * np.trace(Cyy) / len(Cyy) * np.eye(len(Cyy))
    Kx, Ky = _inv_sqrt(Cxx), _inv_sqrt(Cyy)
    U, rho, Vt = np.linalg.svd(Kx @ Cxy @ Ky, full_matrices=False)
    return mx, my, Kx @ U, Ky @ Vt.T, rho


def cca_test_corr(X: np.ndarray, Y: np.ndarray, fit: tuple[np.ndarray, ...]) -> np.ndarray:
    """Correlation of each canonical pair on NEW data, using directions fitted elsewhere. (m,)"""
    mx, my, Wx, Wy, _ = fit
    U, V = (X - mx) @ Wx, (Y - my) @ Wy
    U, V = U - U.mean(axis=0), V - V.mean(axis=0)
    return np.sum(U * V, axis=0) / np.sqrt(np.sum(U**2, axis=0) * np.sum(V**2, axis=0))


# ── Reconstruction scores ────────────────────────────────────────────────────
def vaf_centred(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Centred VAF in %: 100 * (1 - ||X - X_hat||^2 / ||X - mean(X)||^2). 0 = no better than the mean."""
    return float(100.0 * (1.0 - np.sum((X - X_hat) ** 2) / np.sum((X - X.mean(axis=0)) ** 2)))


def random_W(k: int, D: int, seed: int) -> np.ndarray:
    """(k, D) random non-negative synergies, rows of length 1."""
    W = np.random.default_rng(seed).random((k, D))
    return W / np.linalg.norm(W, axis=1, keepdims=True)
