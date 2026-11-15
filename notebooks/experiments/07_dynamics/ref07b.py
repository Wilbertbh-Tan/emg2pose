"""Reference answers for notebook 07b (stage 5, part B: compositional dynamics).

Two models of how the hand state evolves, both asking Lazzari & Saxena's (2025) question on real data:
do different tasks REUSE a shared set of dynamical building blocks?

1. Switching linear motifs (a simplified recurrent switching linear dynamical system, fitted by hard EM): K small
   linear models z(t+1) = A_k z(t) + B_k u(t) + b_k; every 10-frame segment uses one of them.
2. A small multitask RNN (PyTorch) with a task (stage) input, trained with the paper's regularisers, and
   "compositional transfer": a new task learned by fitting only its task-input vector.

NumPy for the motifs; PyTorch for the RNN. Shapes: N frames, d state dims, m input dims, S segments, L = segment
length, K motifs, H hidden units, G task groups.
"""

from __future__ import annotations

import numpy as np

# ═════════════════════════════════════════════════════════════════════════════
# Part 1 · Switching linear motifs (NumPy)
# ═════════════════════════════════════════════════════════════════════════════
SEG_LEN: int = 10  # frames per segment (200 ms)


def make_segments(rec: np.ndarray, mask: np.ndarray, seg_len: int = SEG_LEN) -> np.ndarray:
    """Start indices of non-overlapping segments of ``seg_len`` steps (seg_len + 1 frames) inside one recording,
    with every frame in ``mask``. Segments are cut back to back along each run of usable frames, so consecutive
    starts that differ by exactly seg_len are neighbours in time. (S,) int."""
    starts: list[int] = []
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        ok = mask[idx]
        e = np.diff(np.r_[0, ok.astype(int), 0])
        for a, b in zip(np.where(e == 1)[0], np.where(e == -1)[0]):
            for s in range(a, b - seg_len, seg_len):          # needs frames s … s + seg_len
                starts.append(int(idx[s]))
    return np.array(starts, dtype=np.int64)


def segment_triples(Z: np.ndarray, U: np.ndarray, starts: np.ndarray, seg_len: int = SEG_LEN
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(z(t), u(t), z(t+1)) for every step of every segment: shapes (S, L, d), (S, L, m), (S, L, d)."""
    t = starts[:, None] + np.arange(seg_len)[None, :]
    return Z[t], U[t], Z[t + 1]


def fit_motif(Zt: np.ndarray, Ut: np.ndarray, Zn: np.ndarray, alpha: float = 1e-3
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ridge least squares for z(t+1) = A z(t) + B u(t) + b, from triples of any leading shape.

    Normal equations on F = [z, u, 1]: (F'F + alpha I) W = F'Z_next. Returns A (d, d), B (d, m), b (d,).
    """
    d, m = Zt.shape[-1], Ut.shape[-1]
    F = np.concatenate([Zt.reshape(-1, d), Ut.reshape(-1, m), np.ones((Zt[..., 0].size, 1))], axis=1)
    Y = Zn.reshape(-1, d)
    W = np.linalg.solve(F.T @ F + alpha * np.eye(F.shape[1]), F.T @ Y)
    return W[:d].T, W[d:d + m].T, W[-1]


def step(z: np.ndarray, u: np.ndarray, motif: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """One step of a motif: works on (..., d) z and (..., m) u."""
    A, B, b = motif
    return z @ A.T + u @ B.T + b


def motif_errors(Zt: np.ndarray, Ut: np.ndarray, Zn: np.ndarray,
                 motifs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """(S, K): summed squared one-step error of every segment under every motif."""
    return np.stack([((step(Zt, Ut, mo) - Zn) ** 2).sum(axis=(1, 2)) for mo in motifs], axis=1)


def fit_switching(Zt: np.ndarray, Ut: np.ndarray, Zn: np.ndarray, K: int, n_iter: int = 20, seed: int = 0,
                  alpha: float = 1e-3) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], np.ndarray, list[float]]:
    """Hard-EM switching linear dynamics.

    Start: every segment gets a random motif label. Repeat: (M) refit each motif on its segments; (E) give each
    segment the motif with the smallest one-step error. An empty motif is restarted on the worst-fit segments.
    Returns motifs (list of K (A, B, b)), labels (S,), total error after each iteration.
    """
    rng = np.random.default_rng(seed)
    S = len(Zt)
    labels = rng.integers(0, K, size=S)
    history: list[float] = []
    motifs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    fit_err = rng.random(S)                                    # before the first E-step: a random order
    for _ in range(n_iter):
        motifs = []
        for k in range(K):
            sel = labels == k
            if sel.sum() < 5:                                   # restart an empty motif on the worst-fit segments
                worst = np.argsort(fit_err)[-max(5, S // (4 * K)):]
                sel = np.zeros(S, bool)
                sel[worst] = True
            motifs.append(fit_motif(Zt[sel], Ut[sel], Zn[sel], alpha))
        errs = motif_errors(Zt, Ut, Zn, motifs)
        labels = errs.argmin(axis=1)
        fit_err = errs.min(axis=1)
        history.append(float(fit_err.sum()))
    return motifs, labels, history


def causal_motif_choice(errs_prev: np.ndarray, has_prev: np.ndarray, default: int) -> np.ndarray:
    """For each segment, the motif that fit the PREVIOUS segment best (causal: uses only the past).
    errs_prev (S, K) are the errors of each segment's predecessor; segments without one get ``default``."""
    out = errs_prev.argmin(axis=1)
    out[~has_prev] = default
    return out


def predecessor(starts: np.ndarray, seg_len: int = SEG_LEN) -> tuple[np.ndarray, np.ndarray]:
    """Index of the segment that ends where each segment starts, and whether it exists. (S,) int, (S,) bool."""
    pos = {int(s): i for i, s in enumerate(starts)}
    prev = np.array([pos.get(int(s) - seg_len, -1) for s in starts])
    return np.maximum(prev, 0), prev >= 0


def heldout_errors(Zt: np.ndarray, Ut: np.ndarray, Zn: np.ndarray, starts: np.ndarray,
                   motifs: list[tuple[np.ndarray, np.ndarray, np.ndarray]], default: int = 0
                   ) -> tuple[float, float, np.ndarray]:
    """Score motifs on new segments, choosing each segment's motif from its predecessor (causal).

    Returns mean one-step error per step (sum over state dims), mean error of a free rollout over the whole segment
    (feeding the model its own predictions and the real input), and the chosen motif per segment.
    Only segments that have a predecessor are scored.
    """
    errs = motif_errors(Zt, Ut, Zn, motifs)
    prev, has = predecessor(starts)
    choice = causal_motif_choice(errs[prev], has, default)
    sel = has
    one = float(np.mean([errs[i, choice[i]] for i in np.where(sel)[0]]) / Zt.shape[1])
    roll = []
    for i in np.where(sel)[0]:
        z = Zt[i, 0]
        for t in range(Zt.shape[1]):
            z = step(z, Ut[i, t], motifs[choice[i]])
        roll.append(float(((z - Zn[i, -1]) ** 2).sum()))
    return one, float(np.mean(roll)), choice


def infer_and_score(Zt: np.ndarray, Ut: np.ndarray, Zn: np.ndarray,
                    motifs: list[tuple[np.ndarray, np.ndarray, np.ndarray]], h: int | None = None
                    ) -> tuple[float, float, np.ndarray]:
    """Score a motif library on new segments without peeking: pick each segment's motif from its first ``h`` steps
    (default: half), then score the remaining steps.

    Returns mean one-step squared error per step, mean squared error of a free rollout from step h to the segment's
    end (feeding the model its own predictions and the real input), and the chosen motif per segment.
    """
    L = Zt.shape[1]
    h = L // 2 if h is None else h
    first = np.stack([((step(Zt[:, :h], Ut[:, :h], mo) - Zn[:, :h]) ** 2).sum(axis=(1, 2)) for mo in motifs], axis=1)
    ch = first.argmin(axis=1)
    one, roll = [], []
    for i in range(len(Zt)):
        mo = motifs[ch[i]]
        one.append(((step(Zt[i, h:], Ut[i, h:], mo) - Zn[i, h:]) ** 2).sum() / (L - h))
        z = Zt[i, h]
        for t in range(h, L):
            z = step(z, Ut[i, t], mo)
        roll.append(((z - Zn[i, -1]) ** 2).sum())
    return float(np.mean(one)), float(np.mean(roll)), ch


def describe_error(Zt: np.ndarray, Ut: np.ndarray, Zn: np.ndarray,
                   motifs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    """How well a library DESCRIBES segments when each segment may pick its best motif (uses the scored data, so
    it is optimistic; good for comparing two libraries of the same size). Mean squared one-step error per step."""
    return float(motif_errors(Zt, Ut, Zn, motifs).min(axis=1).sum() / (Zt.shape[0] * Zt.shape[1]))


def usage_matrix(labels: np.ndarray, group_of_seg: np.ndarray, n_groups: int, K: int) -> np.ndarray:
    """(G, K): share of each group's segments that use each motif (rows sum to 1)."""
    M = np.zeros((n_groups, K))
    for g in range(n_groups):
        sel = group_of_seg == g
        if sel.any():
            M[g] = np.bincount(labels[sel], minlength=K) / sel.sum()
    return M


def mutual_information(labels: np.ndarray, groups: np.ndarray) -> float:
    """Mutual information (bits) between motif labels and group labels: 0 = every group uses the same mix."""
    joint = np.zeros((groups.max() + 1, labels.max() + 1))
    np.add.at(joint, (groups, labels), 1)
    p = joint / joint.sum()
    pg, pl = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (pg @ pl)[nz])).sum())


def groups_per_motif(M: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """(K,): how many groups use each motif for at least ``threshold`` of their segments (reuse count)."""
    return (M >= threshold).sum(axis=0)


# ═════════════════════════════════════════════════════════════════════════════
# Part 2 · Multitask RNN (PyTorch)
# ═════════════════════════════════════════════════════════════════════════════
def rnn_params(n_in: int, n_task: int, n_hidden: int, n_out: int, seed: int = 0) -> dict:
    """Initial weights (torch tensors with requires_grad): W_h (H, H), W_u (H, n_in), W_task (H, G), b (H,),
    W_o (n_out, H), b_o (n_out,). W_h is scaled so the network starts close to stable (spectral radius ~0.9)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    p = {"W_h": 0.9 * torch.randn(n_hidden, n_hidden, generator=g) / np.sqrt(n_hidden),
         "W_u": torch.randn(n_hidden, n_in, generator=g) / np.sqrt(n_in),
         "W_task": 0.1 * torch.randn(n_hidden, n_task, generator=g),
         "b": torch.zeros(n_hidden),
         "W_o": torch.randn(n_out, n_hidden, generator=g) / np.sqrt(n_hidden),
         "b_o": torch.zeros(n_out)}
    for v in p.values():
        v.requires_grad_(True)
    return p


def rnn_step(h, u, task, p: dict, alpha: float):
    """One RNN update, Lazzari & Saxena's form:  h' = h + alpha * (-h + W_h tanh(h) + W_u u + W_task task + b).
    h (B, H), u (B, n_in), task (B, G) -> h' (B, H)."""
    import torch
    return h + alpha * (-h + torch.tanh(h) @ p["W_h"].T + u @ p["W_u"].T + task @ p["W_task"].T + p["b"])


def rnn_run(U, T, p: dict, alpha: float, h0=None):
    """Run the RNN over a batch of sequences. U (B, L, n_in), T (B, L, G) -> outputs (B, L, n_out), rates (B, L, H).
    Rates r = tanh(h); outputs y = W_o r + b_o."""
    import torch
    Bsz, L, _ = U.shape
    h = torch.zeros(Bsz, p["W_h"].shape[0]) if h0 is None else h0
    rates = []
    for t in range(L):
        h = rnn_step(h, U[:, t], T[:, t], p, alpha)
        rates.append(torch.tanh(h))
    R = torch.stack(rates, dim=1)
    return R @ p["W_o"].T + p["b_o"], R


def rnn_loss(Y_hat, Y, M, R, p: dict, lam_rate: float, lam_weight: float, lam_dyn: float):
    """Regularised loss (Lazzari & Saxena's terms):
      masked MSE on the outputs (M (B, L) bool: valid pose and past the burn-in)
      + lam_rate * mean |r|                       (L1 on rates: few active units)
      + lam_weight * mean(|W_h| + |W_u| + |W_o|)  (L1 on weights)
      + lam_dyn * mean ||W_h diag(1 - r^2)||^2    ("simple dynamics": small Jacobian of the recurrence)
    Returns (total, mse) as torch scalars."""
    import torch
    w = M.unsqueeze(-1).float()
    mse = ((Y_hat - Y) ** 2 * w).sum() / (w.sum() * Y.shape[-1]).clamp(min=1)
    l_rate = R.abs().mean()
    l_w = p["W_h"].abs().mean() + p["W_u"].abs().mean() + p["W_o"].abs().mean()
    l_dyn = ((p["W_h"] ** 2).sum(0) * (1 - R ** 2) ** 2).sum(-1).mean() / p["W_h"].shape[0]
    return mse + lam_rate * l_rate + lam_weight * l_w + lam_dyn * l_dyn, mse


def chunk_starts(rec: np.ndarray, use: np.ndarray, length: int, stride: int) -> np.ndarray:
    """Start indices of training chunks of ``length`` frames inside one recording whose frames are all in ``use``."""
    out: list[int] = []
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        ok = use[idx]
        for s in range(0, len(idx) - length + 1, stride):
            if ok[s:s + length].all():
                out.append(int(idx[s]))
    return np.array(out, dtype=np.int64)


def train_rnn(p: dict, U: np.ndarray, T: np.ndarray, Y: np.ndarray, M: np.ndarray, starts: np.ndarray,
              length: int, alpha: float, epochs: int, batch: int, lr: float, regs: dict, seed: int = 0,
              params_to_train: list[str] | None = None, burn_in: int = 10) -> list[float]:
    """Adam on random batches of chunks. U (N, n_in), T (N, G), Y (N, n_out), M (N,) bool (frames that count in
    the loss). Only ``params_to_train`` (default: all) are updated. Returns the mean MSE of each epoch."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    names = params_to_train or list(p)
    opt = torch.optim.Adam([p[k] for k in names], lr=lr)
    t_idx = starts[:, None] + np.arange(length)[None, :]
    Ut, Tt, Yt = (torch.tensor(a[t_idx], dtype=torch.float32) for a in (U, T, Y))
    Mt = torch.tensor(M[t_idx])
    Mt[:, :burn_in] = False
    hist: list[float] = []
    for _ in range(epochs):
        order = rng.permutation(len(starts))
        tot = []
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            Yh, R = rnn_run(Ut[b], Tt[b], p, alpha)
            loss, mse = rnn_loss(Yh, Yt[b], Mt[b], R, p, **regs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot.append(float(mse))
        hist.append(float(np.mean(tot)))
    return hist


def rnn_predict(p: dict, U: np.ndarray, T: np.ndarray, rec: np.ndarray, alpha: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """Run each recording from its first frame (h = 0) and return outputs (N, n_out) and rates (N, H)."""
    import torch
    n_out, H = p["W_o"].shape
    out, rates = np.zeros((len(U), n_out)), np.zeros((len(U), H))
    with torch.no_grad():
        for r in np.unique(rec):
            idx = np.where(rec == r)[0]
            Yh, R = rnn_run(torch.tensor(U[idx][None], dtype=torch.float32),
                            torch.tensor(T[idx][None], dtype=torch.float32), p, alpha)
            out[idx], rates[idx] = Yh[0].numpy(), R[0].numpy()
    return out, rates


def clone_params(p: dict) -> dict:
    """A detached copy of the weights (so transfer experiments never change the pretrained network)."""
    return {k: v.detach().clone().requires_grad_(True) for k, v in p.items()}


# ═════════════════════════════════════════════════════════════════════════════
# Part 3 · Shared subspaces of the network's activity
# ═════════════════════════════════════════════════════════════════════════════
def top_subspace(R: np.ndarray, d: int) -> np.ndarray:
    """(H, d) orthonormal basis of the top-d principal components of activity R (N, H)."""
    _, _, Vt = np.linalg.svd(R - R.mean(axis=0), full_matrices=False)
    return Vt[:d].T


def captured_variance(R: np.ndarray, Q: np.ndarray) -> float:
    """Variance of centred R captured by the subspace Q (sum over its directions)."""
    Rc = R - R.mean(axis=0)
    return float(((Rc @ Q) ** 2).sum() / len(R))


def cross_ratio(R_by_group: list[np.ndarray], d: int) -> np.ndarray:
    """(G, G): variance of group a captured by group b's top-d subspace, divided by that captured by a's own."""
    Qs = [top_subspace(R, d) for R in R_by_group]
    G = len(R_by_group)
    out = np.zeros((G, G))
    for a in range(G):
        own = captured_variance(R_by_group[a], Qs[a])
        for b in range(G):
            out[a, b] = captured_variance(R_by_group[a], Qs[b]) / own
    return out


def sharing_index(ratio: np.ndarray, rand: float, exclude: np.ndarray | None = None) -> float:
    """Mean off-diagonal cross ratio rescaled so 0 = random subspace and 1 = the group's own subspace.
    ``exclude`` (G, G) bool marks pairs to leave out (e.g. simultaneous recordings)."""
    G = len(ratio)
    off = ~np.eye(G, dtype=bool)
    if exclude is not None:
        off &= ~exclude
    return float((ratio[off].mean() - rand) / (1 - rand))


def calibrate(p: dict, method: str, new_task: int, U: np.ndarray, T: np.ndarray, Y: np.ndarray, M: np.ndarray,
              starts: np.ndarray, length: int, alpha: float, epochs: int, lr: float, regs: dict, seed: int = 0) -> dict:
    """Adapt a COPY of a pretrained network to a new task from a few calibration chunks.

    method "vector": zero the new task's input column W_task[:, new_task] and train only W_task (only that column
                     receives gradient, because only it is active for the new task): compositional transfer.
    method "all":    start from the pretrained weights and train every parameter on the same chunks.
    Returns the adapted copy; ``p`` is unchanged.
    """
    import torch
    q = clone_params(p)
    if method == "vector":
        with torch.no_grad():
            q["W_task"][:, new_task] = 0.0
        names = ["W_task"]
    elif method == "all":
        names = None
    else:
        raise ValueError(method)
    train_rnn(q, U, T, Y, M, starts, length, alpha, epochs, batch=64, lr=lr, regs=regs, seed=seed,
              params_to_train=names)
    return q


def random_ratio(R_by_group: list[np.ndarray], d: int, rng: np.random.Generator, n: int = 20) -> float:
    """Mean ratio when the 'other' subspace is a random d-dim subspace (the null)."""
    H = R_by_group[0].shape[1]
    vals = []
    for R in R_by_group:
        own = captured_variance(R, top_subspace(R, d))
        for _ in range(n):
            Q, _ = np.linalg.qr(rng.normal(size=(H, d)))
            vals.append(captured_variance(R, Q) / own)
    return float(np.mean(vals))
