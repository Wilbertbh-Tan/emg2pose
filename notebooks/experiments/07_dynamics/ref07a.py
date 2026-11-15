"""Reference answers for notebook 07a (stage 5, part A: dynamics).

Merges the reference code of the former notebooks 06 (linear dynamics, least squares, Kalman filter) and 07
(activation dynamics and the physical, grey-box spring-damper model), and adds Part D (shared dynamics with a
per-task resting posture) and Part E (dynamics in the shared EMG-pose subspace, using notebook 06's ref06).

The notebook asks you to write each of these functions yourself; its check cells compare your output with these.
Plain NumPy (SciPy only for the ready-made smoothing helper).

Conventions: each frame is a ROW. Pose z has 20 columns (degrees); EMG u / y has 16 columns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "06_shared_subspace"))
import ref06  # noqa: E402  (notebook 06: shared EMG-pose subspace; fixed API)

# ════ Parts A–B · linear dynamics and the Kalman filter (former notebook 06) ════


# ── Data: consecutive frames ─────────────────────────────────────────────────
def make_triples(pose: np.ndarray, emg: np.ndarray, rec: np.ndarray,
                 mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every step t → t+1 where both frames are allowed and in the same recording.

    Returns Z (M, 20) = z(t), U (M, 16) = u(t), Z_next (M, 20) = z(t+1).
    """
    ok = mask[:-1] & mask[1:] & (rec[:-1] == rec[1:])
    t = np.flatnonzero(ok)
    return pose[t], emg[t], pose[t + 1]


def rollout_starts(mask: np.ndarray, rec: np.ndarray, horizon: int) -> np.ndarray:
    """Frames t where t, t+1, …, t+horizon are all allowed and in the same recording. Returns (S,) int."""
    ok = mask[:-1] & mask[1:] & (rec[:-1] == rec[1:])        # step t → t+1 is usable
    ok = np.r_[ok, False].astype(int)
    # number of usable steps in a row starting at t, capped at horizon (a running window sum)
    window = np.convolve(ok, np.ones(horizon, dtype=int), mode="full")[horizon - 1:]
    return np.flatnonzero(window[: len(mask)] == horizon)


# ── Least squares from scratch ───────────────────────────────────────────────
def add_bias(X: np.ndarray) -> np.ndarray:
    """(M, p) → (M, p + 1): a column of ones on the right, so the fit can learn a constant offset."""
    return np.hstack([X, np.ones((len(X), 1))])


def least_squares(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Θ (p, q) minimising ||X Θ − Y||² via the normal equations (XᵀX) Θ = XᵀY."""
    return np.linalg.solve(X.T @ X, X.T @ Y)


# ── Linear dynamics with and without EMG ─────────────────────────────────────
def fit_AB(Z: np.ndarray, U: np.ndarray, Z_next: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z(t+1) = A z(t) + B u(t) + b. Returns A (20, 20), B (20, 16), b (20,)."""
    theta = least_squares(add_bias(np.hstack([Z, U])), Z_next)   # (20 + 16 + 1, 20)
    nz = Z.shape[1]
    return theta[:nz].T, theta[nz:-1].T, theta[-1]


def fit_A(Z: np.ndarray, Z_next: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z(t+1) = A z(t) + b (no EMG). Returns A (20, 20), b (20,)."""
    theta = least_squares(add_bias(Z), Z_next)                   # (20 + 1, 20)
    return theta[:-1].T, theta[-1]


def rollout(z0: np.ndarray, U_future: np.ndarray, A: np.ndarray, b: np.ndarray,
            B: np.ndarray | None = None) -> np.ndarray:
    """Run the model forward on its own predictions. Returns (H, 20): the pose after 1, 2, …, H steps."""
    z = z0.copy()
    out = np.empty((len(U_future), len(z0)))
    for h in range(len(U_future)):
        z = A @ z + b + (B @ U_future[h] if B is not None else 0.0)
        out[h] = z
    return out


# ── Kalman filter decoder ────────────────────────────────────────────────────
def fit_kalman(Z: np.ndarray, Z_next: np.ndarray, Z_obs: np.ndarray, Y_obs: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the Kalman model on CENTRED data (no offsets needed).

    State:       z(t+1) = A z(t) + w,   w ~ N(0, Q)   from consecutive pairs (Z, Z_next)
    Observation: y(t)   = H z(t) + v,   v ~ N(0, R)   from same-frame pairs (Z_obs, Y_obs)
    Returns A (20, 20), Q (20, 20), H (16, 20), R (16, 16). A is passed through stabilise (any eigenvalue
    above 0.995 in size is shrunk to 0.995); Q is computed from the raw fit's residuals.
    """
    A = least_squares(Z, Z_next).T
    res_w = Z_next - Z @ A.T
    Q = res_w.T @ res_w / len(Z)
    H = least_squares(Z_obs, Y_obs).T
    res_v = Y_obs - Z_obs @ H.T
    R = res_v.T @ res_v / len(Z_obs)
    return stabilise(A), Q, H, R


def kalman_filter(Y: np.ndarray, A: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray,
                  z0: np.ndarray, P0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode states from observations Y (T, 16), one frame at a time, using only the past.

    Returns Z_hat (T, 20) and P_trace (T,): the total uncertainty (trace of P) after each update.
    """
    n = len(z0)
    z, P = z0.copy(), P0.copy()
    Z_hat = np.empty((len(Y), n))
    P_trace = np.empty(len(Y))
    I = np.eye(n)
    for t in range(len(Y)):
        # predict: where should the hand be, from the dynamics alone?
        z = A @ z
        P = A @ P @ A.T + Q
        # update: correct the guess with this frame's EMG
        S = H @ P @ H.T + R
        K = np.linalg.solve(S, H @ P).T          # = P Hᵀ S⁻¹ (S and P are symmetric)
        z = z + K @ (Y[t] - H @ z)
        P = (I - K @ H) @ P
        Z_hat[t], P_trace[t] = z, np.trace(P)
    return Z_hat, P_trace


def stabilise(A: np.ndarray, max_abs: float = 0.995) -> np.ndarray:
    """Shrink any eigenvalue of A with |λ| > max_abs back to |λ| = max_abs (same direction), keeping the rest.

    A least-squares fit can return a slightly unstable A (|λ| > 1) when the state is very predictable from one
    frame to the next, e.g. a smooth pose and its velocity. A Kalman filter run with such an A blows up over a
    long block. Complex eigenvalues come in conjugate pairs and are shrunk together, so the result stays real.
    """
    lam, V = np.linalg.eig(A)
    mag = np.abs(lam)
    if mag.max() <= max_abs:
        return A
    lam = np.where(mag > max_abs, lam / mag * max_abs, lam)
    return np.real(V @ np.diag(lam) @ np.linalg.inv(V))


def angular_error(Y: np.ndarray, Y_hat: np.ndarray) -> float:
    """Mean absolute joint-angle error in degrees."""
    return float(np.mean(np.abs(Y - Y_hat)))


# ════ Part C · the physical (grey-box) model (former notebook 07) ════
FS: int = 50


# ── Ready-made helper ────────────────────────────────────────────────────────
def smooth_per_recording(X: np.ndarray, rec: np.ndarray, fc: float = 5.0, fs: int = FS) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth low-pass at fc Hz, applied within each recording (never across)."""
    from scipy import signal

    sos = signal.butter(2, fc, fs=fs, output="sos")
    out = X.copy()
    for r in np.unique(rec):
        idx = np.flatnonzero(rec == r)
        if len(idx) > 30:
            out[idx] = signal.sosfiltfilt(sos, X[idx], axis=0)
    return out


# ── Task 1 · activation dynamics ─────────────────────────────────────────────
def activation_filter(U: np.ndarray, rec: np.ndarray, tau_s: float, delay: int, fs: int = FS) -> np.ndarray:
    """Causal first-order low-pass with a pure delay, applied within each recording.

    a(t) = a(t-1) + alpha * (u(t - delay) - a(t-1)),  alpha = 1 - exp(-1 / (fs * tau_s)).
    Before the first sample, the delayed input repeats the recording's first value and a starts at it.
    """
    alpha = 1.0 - np.exp(-1.0 / (fs * tau_s))
    out = np.empty_like(U)
    for r in np.unique(rec):
        idx = np.flatnonzero(rec == r)
        u = U[idx]
        if delay > 0:
            u = np.vstack([np.repeat(u[:1], delay, axis=0), u[:-delay]])
        a = u[0].copy()
        for t in range(len(u)):
            a = a + alpha * (u[t] - a)
            out[idx[t]] = a
    return out


def inner_split(train: np.ndarray, rec: np.ndarray, frac: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    """Split each recording's training block in time: first ``frac`` to fit, the rest to validate."""
    fit_in = np.zeros(len(rec), bool)
    val_in = np.zeros(len(rec), bool)
    for r in np.unique(rec):
        idx = np.flatnonzero(train & (rec == r))
        c = int(frac * len(idx))
        fit_in[idx[:c]] = True
        val_in[idx[c:]] = True
    return fit_in, val_in


def static_error(F: np.ndarray, pose: np.ndarray, rec: np.ndarray, fit_m: np.ndarray, val_m: np.ndarray) -> float:
    """Mean over recordings of the error of a per-recording least-squares map [F, 1] -> pose (skip < 50 frames)."""
    errs = []
    for r in np.unique(rec):
        a, b = fit_m & (rec == r), val_m & (rec == r)
        if b.sum() < 50 or a.sum() < 50:
            continue
        th = least_squares(add_bias(F[a]), pose[a])
        errs.append(angular_error(pose[b], add_bias(F[b]) @ th))
    return float(np.mean(errs))


def activation_grid(U: np.ndarray, pose: np.ndarray, rec: np.ndarray, fit_m: np.ndarray, val_m: np.ndarray,
                    taus: list[float], delays: list[int]) -> dict[tuple[float, int], float]:
    """Validation error of static_error for every (tau, delay) pair."""
    return {(tau, dl): static_error(activation_filter(U, rec, tau, dl), pose, rec, fit_m, val_m)
            for tau in taus for dl in delays}


# ── Task 2 · the spring-damper forward model ─────────────────────────────────
def transition_matrix(K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """(2n, 2n) one-frame transition of the state [q, v] with zero input.

    v(t+1) = v - K q - D v ;  q(t+1) = q + v(t+1)
    """
    n = len(K)
    F = np.zeros((2 * n, 2 * n))
    F[:n, :n] = np.diag(1 - K)
    F[:n, n:] = np.diag(1 - D)
    F[n:, :n] = np.diag(-K)
    F[n:, n:] = np.diag(1 - D)
    return F


def simulate(q0: np.ndarray, v0: np.ndarray, A_in: np.ndarray, K: np.ndarray, D: np.ndarray,
             M: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Run the model forward H = len(A_in) frames. Returns (H, n): q after 1, 2, …, H frames.

    acceleration = -K q - D v + M a + b ;  v <- v + acceleration ;  q <- q + v
    """
    q, v = q0.astype(float).copy(), v0.astype(float).copy()
    out = np.empty((len(A_in), len(q0)))
    for h in range(len(A_in)):
        v = v - K * q - D * v + M @ A_in[h] + b
        q = q + v
        out[h] = q
    return out


def max_abs_eig(K: np.ndarray, D: np.ndarray) -> float:
    """Largest |eigenvalue| of transition_matrix(K, D): < 1 means every free motion dies out."""
    return float(np.abs(np.linalg.eigvals(transition_matrix(K, D))).max())


# ── Task 3 · fitting it ──────────────────────────────────────────────────────
def second_order_steps(mask: np.ndarray, rec: np.ndarray) -> np.ndarray:
    """Frames t where t-1, t and t+1 are all in mask and in the same recording (so v and the
    acceleration are real)."""
    ok = mask[:-2] & mask[1:-1] & mask[2:] & (rec[:-2] == rec[2:])
    return np.flatnonzero(ok) + 1


def fit_spring_damper(Q: np.ndarray, A: np.ndarray, t: np.ndarray, ridge: float = 1e-6
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares fit of  acc_i = -K_i q_i - D_i v_i + M_i · a + b_i  for each pose dimension i,
    with K_i >= 0 and D_i >= 0.

    Q : (N, n) pose scores; A : (N, k) activations; t : frames from second_order_steps.
    Constraint by an active set: fit freely; any of K_i, D_i that comes out negative is fixed at 0 and the
    rest is refitted. Returns K (n,), D (n,), M (n, k), b (n,).
    """
    qp, q, qn, a = Q[t - 1], Q[t], Q[t + 1], A[t]
    v, acc = q - qp, qn - 2 * q + qp
    n, k = Q.shape[1], A.shape[1]
    K, D, M, b = np.zeros(n), np.zeros(n), np.zeros((n, k)), np.zeros(n)
    for i in range(n):
        free = [True, True]
        own = [q[:, i], v[:, i]]
        for _ in range(3):
            X = np.column_stack([c for c, f in zip(own, free) if f] + [a, np.ones(len(t))])
            th = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ acc[:, i])
            kd, j = [0.0, 0.0], 0
            for m in range(2):
                if free[m]:
                    kd[m] = -th[j]
                    j += 1
            neg = [m for m in range(2) if free[m] and kd[m] < 0]
            if not neg:
                break
            for m in neg:
                free[m] = False
        K[i], D[i] = kd
        M[i], b[i] = th[j:j + k], th[j + k]
    return K, D, M, b


# ── Task 5 · the physical Kalman decoder ────────────────────────────────────
def physical_matrices(K: np.ndarray, D: np.ndarray, M: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """(2n + k, 2n + k) transition of the state [q, v, a]: spring-damper driven by a, and a(t+1) = rho * a(t)."""
    n, k = len(K), len(rho)
    F = np.zeros((2 * n + k, 2 * n + k))
    F[:2 * n, :2 * n] = transition_matrix(K, D)
    F[:n, 2 * n:] = M          # q(t+1) = q + v(t+1), and v(t+1) contains + M a
    F[n:2 * n, 2 * n:] = M
    F[2 * n:, 2 * n:] = np.diag(rho)
    return F


def physical_state(Qc: np.ndarray, Ac: np.ndarray) -> np.ndarray:
    """(N, 2n + k) state [q, v, a] with v(t) = q(t) - q(t-1) (first row 0). Inputs already centred."""
    v = np.vstack([np.zeros((1, Qc.shape[1])), np.diff(Qc, axis=0)])
    return np.hstack([Qc, v, Ac])


def fit_physical(Qs: np.ndarray, A: np.ndarray, Y: np.ndarray, fit_mask: np.ndarray, rec: np.ndarray
                  ) -> dict[str, np.ndarray]:
    """Fit every part of the physical state-space model on fit_mask frames.

    Qs : (N, n) pose scores, A : (N, k) filtered activations, Y : (N, 16) observation (log EMG).
    Centre with fit_mask means; spring-damper by fit_spring_damper; rho by per-synergy AR(1) least squares;
    Q = covariance of the one-step state residuals; H, R by least squares Y_c ~ state (as in notebook 06).
    """
    t = second_order_steps(fit_mask, rec)
    mu_q, mu_a, mu_y = Qs[fit_mask].mean(0), A[fit_mask].mean(0), Y[fit_mask].mean(0)
    Qc, Ac, Yc = Qs - mu_q, A - mu_a, Y - mu_y
    K, D, M, _ = fit_spring_damper(Qc, Ac, t)
    a0, a1 = Ac[t], Ac[t + 1]
    rho = (a0 * a1).sum(0) / (a0 * a0).sum(0)
    F = physical_matrices(K, D, M, rho)
    Z = physical_state(Qc, Ac)
    res = Z[t + 1] - Z[t] @ F.T
    Qn = res.T @ res / len(t)
    H = least_squares(Z[t], Yc[t]).T
    rv = Yc[t] - Z[t] @ H.T
    R = rv.T @ rv / len(t)
    P0 = np.cov(Z[t].T, bias=True)
    return {"F": F, "Q": Qn, "H": H, "R": R, "P0": P0, "mu_q": mu_q, "mu_y": mu_y,
            "K": K, "D": D, "M": M, "rho": rho}


def physical_decode(model: dict[str, np.ndarray], Y_block: np.ndarray) -> np.ndarray:
    """Kalman-decode one contiguous block of observations. Returns (T, n) pose scores (mean added back)."""
    n = len(model["mu_q"])
    Zh, _ = kalman_filter(Y_block - model["mu_y"], model["F"], model["Q"], model["H"], model["R"],
                                np.zeros(len(model["F"])), model["P0"])
    return Zh[:, :n] + model["mu_q"]


# ── Ready-made comparison decoder (Part B's best) ───────────────────────
def fit_kalman_velocity(pose: np.ndarray, Y: np.ndarray, fit_mask: np.ndarray, rec: np.ndarray
                        ) -> dict[str, np.ndarray]:
    """Notebook 06's Kalman decoder with state [pose, velocity] (40-D), fitted on fit_mask frames."""
    mu_z, mu_y = pose[fit_mask].mean(0), Y[fit_mask].mean(0)
    pc, yc = pose - mu_z, Y - mu_y
    st = np.hstack([pc, np.vstack([np.zeros((1, pose.shape[1])), np.diff(pc, axis=0)])])
    fv = np.r_[False, fit_mask[1:] & fit_mask[:-1] & (rec[1:] == rec[:-1])]
    Zr, _, Zn = make_triples(st, yc, rec, fv)
    A, Q, H, R = fit_kalman(Zr, Zn, st[fv], yc[fv])
    return {"A": A, "Q": Q, "H": H, "R": R, "P0": np.cov(st[fv].T, bias=True), "mu_z": mu_z, "mu_y": mu_y}


def kalman_velocity_decode(model: dict[str, np.ndarray], Y_block: np.ndarray) -> np.ndarray:
    """(T, 20) decoded pose for one contiguous block."""
    n = len(model["mu_z"])
    Zh, _ = kalman_filter(Y_block - model["mu_y"], model["A"], model["Q"], model["H"], model["R"],
                                np.zeros(2 * n), model["P0"])
    return Zh[:, :n] + model["mu_z"]


# ════ Part D · shared dynamics with a per-task resting posture ════
def recording_means(X: np.ndarray, rec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(R, D) mean of X over the ``mask`` frames of each recording r = 0 … R-1 (NaN rows where none)."""
    R = int(rec.max()) + 1
    out = np.full((R, X.shape[1]), np.nan)
    for r in range(R):
        m = mask & (rec == r)
        if m.any():
            out[r] = X[m].mean(axis=0)
    return out


def velocity_state(Zc: np.ndarray, rec: np.ndarray) -> np.ndarray:
    """(N, 2n) state [z, v] with v(t) = z(t) - z(t-1) inside each recording (0 at a recording's first frame)."""
    v = np.vstack([np.zeros((1, Zc.shape[1])), np.diff(Zc, axis=0)])
    first = np.r_[True, rec[1:] != rec[:-1]]
    v[first] = 0.0
    return np.hstack([Zc, v])


def fit_kv(Z: np.ndarray, Y: np.ndarray, fit_mask: np.ndarray, rec: np.ndarray, per_recording: bool
           ) -> dict[str, np.ndarray]:
    """Kalman + velocity model (state [z, v], observation Y), shared across all fit_mask recordings.

    per_recording=False: centre Z and Y with their pooled fit_mask means (one resting posture for all tasks).
    per_recording=True : centre each recording with ITS OWN fit_mask means, so the shared dynamics describe motion
    around each task's own resting posture (the offset is supplied again at decoding time).
    Returns {"A","Q","H","R","P0","mu_z","mu_y"}; mu_* are the pooled means (used by per_recording=False).
    """
    mu_z, mu_y = Z[fit_mask].mean(axis=0), Y[fit_mask].mean(axis=0)
    if per_recording:
        mz, my = recording_means(Z, rec, fit_mask), recording_means(Y, rec, fit_mask)
        Zc, Yc = Z - np.nan_to_num(mz[rec]), Y - np.nan_to_num(my[rec])
    else:
        Zc, Yc = Z - mu_z, Y - mu_y
    st = velocity_state(Zc, rec)
    fv = np.r_[False, fit_mask[1:] & fit_mask[:-1] & (rec[1:] == rec[:-1])]   # frames whose velocity is real
    Zr, _, Zn = make_triples(st, Yc, rec, fv)
    A, Q, H, R = fit_kalman(Zr, Zn, st[fv], Yc[fv])
    return {"A": A, "Q": Q, "H": H, "R": R, "P0": np.cov(st[fv].T, bias=True), "mu_z": mu_z, "mu_y": mu_y}


def kv_decode(model: dict[str, np.ndarray], Y_block: np.ndarray, z_offset: np.ndarray, y_offset: np.ndarray
              ) -> np.ndarray:
    """Kalman-decode one contiguous block: filter (Y_block - y_offset) from a zero state, return z + z_offset."""
    n = model["A"].shape[0] // 2
    Zh, _ = kalman_filter(Y_block - y_offset, model["A"], model["Q"], model["H"], model["R"],
                          np.zeros(2 * n), model["P0"])
    return Zh[:, :n] + z_offset


def offset_from_emg(emg_means: np.ndarray, pose_means: np.ndarray, new_emg_mean: np.ndarray,
                    alpha: float = 10.0) -> np.ndarray:
    """Guess a new task's resting posture from its mean EMG: ridge regression across training recordings.

    emg_means (R, m), pose_means (R, n): one row per training recording; new_emg_mean (m,). Returns (n,).
    """
    ex, pz = emg_means.mean(axis=0), pose_means.mean(axis=0)
    Xc, Yc = emg_means - ex, pose_means - pz
    B = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(Xc.shape[1]), Xc.T @ Yc)
    return (new_emg_mean - ex) @ B + pz


def calibration_masks(te: np.ndarray, rec: np.ndarray, cal_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Split held-out frames: the first ``cal_frames`` of each held-out recording (calibration) and the rest
    (scored). Returns (cal_mask, after_mask)."""
    cal = np.zeros(len(rec), bool)
    after = np.zeros(len(rec), bool)
    for r in np.unique(rec[te]):
        idx = np.flatnonzero(te & (rec == r))
        cal[idx[:cal_frames]] = True
        after[idx[cal_frames:]] = True
    return cal, after


# ════ Part E · dynamics in the shared EMG–pose subspace ════
def latent_spaces(X: np.ndarray, pose: np.ndarray, fit_mask: np.ndarray, rank: int, alpha: float = 1.0
                  ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Fit notebook 06's reduced-rank map X -> pose on fit_mask frames and return both sides of the shared subspace.

    Returns S (N, rank): EMG-side shared latents  (ref06.shared_latents);
            Zq (N, rank): pose-side latents, (pose - y_mean) @ V;
            sub: {"B_r", "x_mean", "y_mean", "V"} from ref06.rrr_fit.
    """
    B_r, x_mean, y_mean, V = ref06.rrr_fit(X[fit_mask], pose[fit_mask], rank, alpha)
    S = ref06.shared_latents(X, B_r, x_mean, rank)
    Zq = (pose - y_mean) @ V
    return S, Zq, {"B_r": B_r, "x_mean": x_mean, "y_mean": y_mean, "V": V}


def latent_to_pose(z_lat: np.ndarray, sub: dict[str, np.ndarray], pose_offset: np.ndarray) -> np.ndarray:
    """(T, rank) latents centred on a task's resting posture -> (T, 20) angles: pose_offset + z_lat @ V'."""
    return pose_offset + z_lat @ sub["V"].T
