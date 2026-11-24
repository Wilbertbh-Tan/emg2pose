"""Reference answers for notebook 02 (how many synergies?).

The notebook asks you to write each of these functions yourself; its check cells compare your
output with these. Notebooks 03 and 04 import NMF and PCA from here, so a blank in your copy of
notebook 02 never breaks theirs. Everything is plain NumPy: no scikit-learn.

Shapes: T = frames, D = features (16 EMG channels or 20 joints), k = components.
"""

from __future__ import annotations

from collections.abc import Callable

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


# ── NMF from its mathematical foundation (Task 1, steps 1a–1e) ──────────────
def nmf_loss(X: np.ndarray, C: np.ndarray, W: np.ndarray) -> float:
    """The NMF objective L(C, W) = 1/2 * sum of (X - C W)^2 over every entry.

    The 1/2 only cancels the 2 that differentiation brings down; it does not move the minimum.
    """
    return float(0.5 * np.sum((X - C @ W) ** 2))


def nmf_grads(X: np.ndarray, C: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of nmf_loss. With the residual R = C W - X (T, D):

        dL/dC = R W.T   (T, k)        dL/dW = C.T R   (k, D)
    """
    R = C @ W - X
    return R @ W.T, C.T @ R


def pgd_step(X: np.ndarray, C: np.ndarray, W: np.ndarray, lr: float) -> tuple[np.ndarray, np.ndarray]:
    """One round of projected gradient descent with ONE learning rate for both matrices.

    W <- max(0, W - lr * dL/dW), then (with the new W) C <- max(0, C - lr * dL/dC). Returns new (C, W).
    """
    _, gW = nmf_grads(X, C, W)
    W = np.maximum(0.0, W - lr * gW)
    gC, _ = nmf_grads(X, C, W)
    C = np.maximum(0.0, C - lr * gC)
    return C, W


def numerical_grad(f: Callable[[np.ndarray], float], A: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Slope of the scalar function f at A, one entry at a time, by central differences:
    g[idx] = (f(A with A[idx] + h) - f(A with A[idx] - h)) / 2h. A is not changed."""
    g = np.zeros(A.shape, dtype=float)
    for idx in np.ndindex(A.shape):
        Ap, Am = A.astype(float), A.astype(float)
        Ap[idx] += h
        Am[idx] -= h
        g[idx] = (f(Ap) - f(Am)) / (2 * h)
    return g


def safe_steps(C: np.ndarray, W: np.ndarray) -> tuple[float, float]:
    """Largest safe gradient step for each matrix: (1 / λmax(Cᵀ C) for W, 1 / λmax(W Wᵀ) for C)."""
    return float(1 / np.linalg.eigvalsh(C.T @ C).max()), float(1 / np.linalg.eigvalsh(W @ W.T).max())


def run_pgd(X: np.ndarray, C: np.ndarray, W: np.ndarray, lr: float, n_rounds: int) -> list[float]:
    """n_rounds of pgd_step from (C, W); returns nmf_loss after each round. The inputs are not changed."""
    losses: list[float] = []
    for _ in range(n_rounds):
        C, W = pgd_step(X, C, W, lr)
        losses.append(nmf_loss(X, C, W))
    return losses


def update_W(X: np.ndarray, C: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Lee & Seung multiplicative update of W: W * (C.T X) / (C.T C W).

    This is gradient descent W - eta * dL/dW with a step for every entry, eta = W / (C.T C W).
    """
    return W * (C.T @ X) / (C.T @ C @ W + EPS)


def update_C(X: np.ndarray, C: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Multiplicative update of C: C * (X W.T) / (C W W.T). Same as update_W on X.T = W.T C.T, transposed back."""
    return C * (X @ W.T) / (C @ W @ W.T + EPS)


def nmf_init(X: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random non-negative starting C (T, k) and W (k, D), scaled so C @ W is about the size of X."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(X.mean() / k)
    return scale * rng.random((X.shape[0], k)), scale * rng.random((k, X.shape[1]))


def nmf_fit(X: np.ndarray, k: int, n_iter: int = 300, seed: int = 0
            ) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Factorise non-negative X (T, D) ~= C (T, k) @ W (k, D), both non-negative.

    Each round: W <- update_W, then C <- update_C (with the new W), then record nmf_loss.
    Each update can only lower the loss, and multiplying non-negative numbers keeps them
    non-negative. Rows of W are then scaled to length 1 (C absorbs the scale, so C W is unchanged).

    Returns C, W and losses: nmf_loss after each round (length n_iter).
    """
    C, W = nmf_init(X, k, seed)
    losses: list[float] = []
    for _ in range(n_iter):
        W = update_W(X, C, W)
        C = update_C(X, C, W)
        losses.append(nmf_loss(X, C, W))
    norms = np.linalg.norm(W, axis=1) + EPS
    return C * norms, W / norms[:, None], losses


def nmf_transform(X: np.ndarray, W: np.ndarray, n_iter: int = 300, seed: int = 0) -> np.ndarray:
    """Activations C (T, k) for new data X (T, D) with the synergies W (k, D) held fixed.

    The same C update as nmf_fit, but W is never changed.
    """
    C, _ = nmf_init(X, W.shape[0], seed)
    for _ in range(n_iter):
        C = update_C(X, C, W)
    return C


# ── KL-divergence NMF (step 1f: a Poisson noise model instead of a Gaussian one) ──
def kl_loss(X: np.ndarray, C: np.ndarray, W: np.ndarray) -> float:
    """Generalised KL divergence D(X || C W) = sum of X log(X / CW) - X + CW over every entry.

    EPS inside the log keeps 0 log 0 = 0 and avoids log(0). 0 means C @ W rebuilds X exactly.
    """
    R = C @ W
    return float(np.sum(X * np.log((X + EPS) / (R + EPS)) - X + R))


def kl_grads(X: np.ndarray, C: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of kl_loss. With the ratio Q = X / (C W + EPS) (T, D) and ones = 1 (T, D):

        dL/dC = (ones - Q) W.T   (T, k)        dL/dW = C.T (ones - Q)   (k, D)
    """
    G = 1.0 - X / (C @ W + EPS)
    return G @ W.T, C.T @ G


def kl_update_W(X: np.ndarray, C: np.ndarray, W: np.ndarray) -> np.ndarray:
    """W ⊙ (Cᵀ Q) / (Cᵀ 1): 'asks' = Cᵀ Q, 'gives' = column sums of C (one per synergy, same for every electrode)."""
    Q = X / (C @ W + EPS)
    return W * (C.T @ Q) / (C.sum(axis=0)[:, None] + EPS)


def kl_update_C(X: np.ndarray, C: np.ndarray, W: np.ndarray) -> np.ndarray:
    """C ⊙ (Q Wᵀ) / (1 Wᵀ): 'gives' = row sums of W (one per synergy, same for every frame)."""
    Q = X / (C @ W + EPS)
    return C * (Q @ W.T) / (W.sum(axis=1)[None, :] + EPS)


def nmf_fit_kl(X: np.ndarray, k: int, n_iter: int = 300, seed: int = 0
               ) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """nmf_fit with the KL updates: same start (nmf_init), same order (W then C), kl_loss recorded each round,
    rows of W scaled to length 1 at the end."""
    C, W = nmf_init(X, k, seed)
    losses: list[float] = []
    for _ in range(n_iter):
        W = kl_update_W(X, C, W)
        C = kl_update_C(X, C, W)
        losses.append(kl_loss(X, C, W))
    norms = np.linalg.norm(W, axis=1) + EPS
    return C * norms, W / norms[:, None], losses


def planted_synergies(T: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic EMG with known answers: 3 bumps of activity around a 16-electrode ring (centred on electrodes
    2, 7 and 12, rows of length 1), sparse non-negative activations, and a little positive noise.

    Returns X (T, 16), C_true (T, 3), W_true (3, 16).
    """
    rng = np.random.default_rng(seed)
    ch = np.arange(16)
    W_true = np.array([np.exp(3 * np.cos(2 * np.pi * (ch - c) / 16)) for c in (2, 7, 12)])
    W_true /= np.linalg.norm(W_true, axis=1, keepdims=True)
    C_true = rng.gamma(0.5, 1.0, (T, 3)) * (rng.random((T, 3)) < 0.5)
    X = C_true @ W_true + 0.01 * rng.random((T, 16))
    return X, C_true, W_true


def match_synergies(W: np.ndarray, W_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair each true synergy with its best-matching fitted one (NMF returns them in any order).

    Tries every ordering (fine for k <= 6). Returns order (k,) so W[order[i]] matches W_true[i], and the cosine
    similarity of each pair (k,).
    """
    from itertools import permutations
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    Tn = W_true / np.linalg.norm(W_true, axis=1, keepdims=True)
    S = Tn @ Wn.T
    k = len(W_true)
    best = max(permutations(range(len(W)), k), key=lambda p: sum(S[i, p[i]] for i in range(k)))
    order = np.array(best)
    return order, S[np.arange(k), order]


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
