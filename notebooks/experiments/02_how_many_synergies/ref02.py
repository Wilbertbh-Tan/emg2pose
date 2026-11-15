"""Reference answers for notebook 02 (how many synergies?).

The notebook asks you to write each of these functions yourself; its check cells compare your
output with these. Notebooks 03 and 04 import NMF and PCA from here, so a blank in your copy of
notebook 02 never breaks theirs. Everything is plain NumPy: no scikit-learn.

Shapes: T = frames, D = features (16 EMG channels or 20 joints), k = components.
"""

from __future__ import annotations

import numpy as np

EPS: float = 1e-10  # keeps multiplicative updates away from dividing by zero


# ── VAF ──────────────────────────────────────────────────────────────────────
def vaf(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Uncentered variance accounted for, in %: 100 * (1 - ||X - X_hat||^2 / ||X||^2)."""
    return float(100.0 * (1.0 - np.sum((X - X_hat) ** 2) / np.sum(X**2)))


# ── PCA via the SVD ──────────────────────────────────────────────────────────
def pca_fit(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA with k components.

    Returns mean (D,) and components (k, D): orthonormal rows, ordered by variance explained.
    Sign convention (SVD signs are arbitrary): each row's largest-magnitude entry is positive.
    """
    mean = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mean, full_matrices=False)
    comps = Vt[:k]
    signs = np.sign(comps[np.arange(k), np.abs(comps).argmax(axis=1)])
    return mean, comps * signs[:, None]


def pca_transform(X: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """(T, D) data -> (T, k) scores: centre, then project onto each component."""
    return (X - mean) @ comps.T


def pca_inverse(scores: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """(T, k) scores -> (T, D) rebuilt data: weighted sum of components, plus the mean back."""
    return scores @ comps + mean


# ── NMF via Lee & Seung multiplicative updates ───────────────────────────────
def nmf_init(X: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random non-negative starting C (T, k) and W (k, D), scaled so C @ W is about the size of X."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(X.mean() / k)
    return scale * rng.random((X.shape[0], k)), scale * rng.random((k, X.shape[1]))


def nmf_fit(X: np.ndarray, k: int, n_iter: int = 300, seed: int = 0
            ) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Factorise non-negative X (T, D) ~= C (T, k) @ W (k, D), both non-negative.

    Multiplicative updates (Lee & Seung 2001) for the squared error ||X - C W||^2:
        W <- W * (C.T X) / (C.T C W)
        C <- C * (X W.T) / (C W W.T)
    Each update can only lower the error, and multiplying non-negative numbers keeps them
    non-negative. Rows of W are then scaled to length 1 (C absorbs the scale, so C W is unchanged).

    Returns C, W and losses: the squared error after each iteration (length n_iter).
    """
    C, W = nmf_init(X, k, seed)
    losses: list[float] = []
    for _ in range(n_iter):
        W *= (C.T @ X) / (C.T @ C @ W + EPS)
        C *= (X @ W.T) / (C @ W @ W.T + EPS)
        losses.append(float(np.sum((X - C @ W) ** 2)))
    norms = np.linalg.norm(W, axis=1) + EPS
    return C * norms, W / norms[:, None], losses


def nmf_transform(X: np.ndarray, W: np.ndarray, n_iter: int = 300, seed: int = 0) -> np.ndarray:
    """Activations C (T, k) for new data X (T, D) with the synergies W (k, D) held fixed.

    The same C update as nmf_fit, but W is never changed.
    """
    C, _ = nmf_init(X, W.shape[0], seed)
    for _ in range(n_iter):
        C *= (X @ W.T) / (C @ W @ W.T + EPS)
    return C


# ── Pose error ───────────────────────────────────────────────────────────────
def pose_error_deg(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    """Mean absolute joint-angle error in degrees, over all joints and frames."""
    return float(np.mean(np.abs(Y - Y_hat)))


# ── Fairer references (stage 1 of the investigation ladder) ──────────────────
def vaf_centred(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Centred VAF in %: the error is compared with X minus its column means, so the mean earns no credit."""
    return float(100.0 * (1.0 - np.sum((X - X_hat) ** 2) / np.sum((X - X.mean(axis=0)) ** 2)))


def random_W(k: int, D: int, seed: int) -> np.ndarray:
    """k random non-negative 'synergies' (k, D), rows of length 1: the null that real synergies must beat."""
    W = np.random.default_rng(seed).random((k, D))
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def heldout_vaf_table(X: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]], ks: list[int],
                      step: int = 5, n_iter: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Held-out VAF for every fold and k, for NMF and for a random W of the same size.

    folds : list of (train_mask, test_mask) over the rows of X (e.g. protocol.leave_one_stage_out).
    Fits use every `step`-th frame and `n_iter` update rounds (to keep 15 folds x len(ks) fits fast).
    Returns real, rand: each (n_folds, len(ks)) in %.
    """
    real = np.zeros((len(folds), len(ks)))
    rand = np.zeros((len(folds), len(ks)))
    for f, (tr, te) in enumerate(folds):
        X_tr, X_te = X[tr][::step], X[te][::step]
        for j, k in enumerate(ks):
            _, W, _ = nmf_fit(X_tr, k, n_iter=n_iter, seed=seed)
            real[f, j] = vaf(X_te, nmf_transform(X_te, W, n_iter=n_iter, seed=seed) @ W)
            Wr = random_W(k, X.shape[1], seed)
            rand[f, j] = vaf(X_te, nmf_transform(X_te, Wr, n_iter=n_iter, seed=seed) @ Wr)
    return real, rand


def choose_k(real: np.ndarray, ks: list[int], min_gain: float = 1.0) -> int:
    """Smallest k after which one more synergy adds less than `min_gain` VAF points (mean over folds)."""
    gains = np.diff(real.mean(axis=0))
    for j, g in enumerate(gains):
        if g < min_gain:
            return ks[j]
    return ks[-1]
