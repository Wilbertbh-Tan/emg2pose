"""Reference answers for notebook 03 (are the synergies stable and real?).

The notebook asks you to write each of these functions yourself; its check cells compare your
output with these. NMF and VAF come from notebook 02's reference (ref02.py), so this file only
holds the new ideas: comparing, shuffling and re-aligning synergies. Plain NumPy, plus SciPy's
Hungarian algorithm for the one step that pairs synergies up.

Shapes: T = frames, k = synergies, 16 = electrodes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_how_many_synergies"))
import ref02  # noqa: E402  (NMF and VAF, written in notebook 02)

N_CH: int = 16


def fit_synergies(X: np.ndarray, k: int, seed: int, n_iter: int = 200) -> np.ndarray:
    """NMF synergies W (k, 16) of non-negative EMG X (T, 16), each row of length 1."""
    _, W, _ = ref02.nmf_fit(X, k, n_iter=n_iter, seed=seed)
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def cosine_similarity_matrix(W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """(k1, 16) and (k2, 16) -> (k1, k2): entry [i, j] = cosine of the angle between W1[i] and W2[j]."""
    A = W1 / np.linalg.norm(W1, axis=1, keepdims=True)
    B = W2 / np.linalg.norm(W2, axis=1, keepdims=True)
    return A @ B.T


def match_synergies(W1: np.ndarray, W2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair each synergy in W1 with one in W2 so the total similarity is as large as possible.

    Returns sims (k,), the similarity of each pair, and order (k,), with W2[order[i]] the partner of W1[i].
    The pairing is the Hungarian algorithm (linear_sum_assignment), which minimises a cost, so we give it
    minus the similarity.
    """
    S = cosine_similarity_matrix(W1, W2)
    rows, cols = linear_sum_assignment(-S)
    order = cols[np.argsort(rows)]
    return S[np.arange(len(order)), order], order


def shuffle_channels(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Roll each column of X (T, D) in time by its own random shift.

    Draw all shifts at once, shifts = rng.integers(0, T, size=D), then column j is np.roll-ed by shifts[j].
    """
    T, D = X.shape
    shifts = rng.integers(0, T, size=D)
    return np.column_stack([np.roll(X[:, j], shifts[j]) for j in range(D)])


def neighbour_share(w: np.ndarray) -> float:
    """Share of a synergy's total weight on its strongest electrode and its two ring neighbours."""
    i = int(np.argmax(w))
    n = len(w)
    return float((w[(i - 1) % n] + w[i] + w[(i + 1) % n]) / w.sum())


def channel_variants(W: np.ndarray) -> list[np.ndarray]:
    """The 32 ways the band could be re-aligned: 16 rotations of the columns, normal then reversed."""
    out: list[np.ndarray] = []
    for Wv in (W, W[:, ::-1]):
        for s in range(W.shape[1]):
            out.append(np.roll(Wv, s, axis=1))
    return out


def best_match(W_left: np.ndarray, W_right: np.ndarray) -> float:
    """Mean matched similarity between W_left and the best-aligned of the 32 variants of W_right."""
    return max(float(match_synergies(W_left, V)[0].mean()) for V in channel_variants(W_right))


def random_synergies(k: int, rng: np.random.Generator, n_ch: int = N_CH) -> np.ndarray:
    """k random non-negative synergies (k, n_ch), rows of length 1: the null for the two-hands test."""
    W = rng.random((k, n_ch))
    return W / np.linalg.norm(W, axis=1, keepdims=True)


# ── Spatial nulls for the crosstalk check (stage 2) ──────────────────────────
def permuted_shares(w: np.ndarray, rng: np.random.Generator, n_perm: int = 1000) -> np.ndarray:
    """Neighbour shares of n_perm random re-orderings of w's electrodes: the same weights, no spatial layout."""
    return np.array([neighbour_share(rng.permutation(w)) for _ in range(n_perm)])


def ring_profile(C: np.ndarray) -> np.ndarray:
    """(16, 16) channel correlation matrix -> (9,) mean correlation at ring distance 0, 1, …, 8."""
    n = C.shape[0]
    return np.array([np.mean([C[i, (i + d) % n] for i in range(n)]) for d in range(n // 2 + 1)])


def single_source_pattern(C: np.ndarray, far_from: int = 5) -> np.ndarray:
    """What ONE muscle source would look like across the ring if its spread followed the channel correlations.

    Take the ring profile, subtract the far-away plateau (mean over distances >= far_from: shared activity, not
    spread), divide by (1 - plateau), clip at 0, and lay it out centred on electrode 0.
    The correlations also contain real co-activation, so this spread is GENEROUS (wider than true crosstalk).
    """
    prof = ring_profile(C)
    plateau = prof[far_from:].mean()
    bump = np.clip((prof - plateau) / (1 - plateau), 0, None)
    n = C.shape[0]
    dist = np.minimum(np.arange(n), n - np.arange(n))
    return bump[dist]
