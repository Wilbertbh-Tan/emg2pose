"""Reference answers for notebook 10 (stage 8: goal-driven simulation).

A differentiable hand (PyTorch) with lumped muscles, shared tendons and a forearm electrode ring, plus a
Lazzari & Saxena-style recurrent controller trained to IMITATE the recorded hand movements (pose only).

Rules that keep this honest with one person's data:
- Every body and electrode parameter below is fixed IN ADVANCE from rough literature values (see PARAMETER NOTES);
  none is fitted to this person's EMG. Only the rest posture and joint limits come from the person's POSE.
- The controller learns from pose only; the EMG is never used for training. Simulated EMG is then compared with
  the real EMG as an independent test.

Units: angles enter and leave in DEGREES; inside the body they are RADIANS. 1 frame = 20 ms = SUBSTEPS substeps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

# ── Joints (exactly the dataset's column order) ──────────────────────────────
JOINT_NAMES: list[str] = [
    "THUMB_CMC_FE", "THUMB_CMC_AA", "THUMB_MCP_FE", "THUMB_IP_FE",
    "INDEX_MCP_AA", "INDEX_MCP_FE", "INDEX_PIP_FE", "INDEX_DIP_FE",
    "MIDDLE_MCP_AA", "MIDDLE_MCP_FE", "MIDDLE_PIP_FE", "MIDDLE_DIP_FE",
    "RING_MCP_AA", "RING_MCP_FE", "RING_PIP_FE", "RING_DIP_FE",
    "PINKY_MCP_AA", "PINKY_MCP_FE", "PINKY_PIP_FE", "PINKY_DIP_FE",
]
J = 20
FINGERS = ["INDEX", "MIDDLE", "RING", "PINKY"]
FINGER_JOINTS = {f: {"AA": 4 + 4 * i, "MCP": 5 + 4 * i, "PIP": 6 + 4 * i, "DIP": 7 + 4 * i} for i, f in enumerate(FINGERS)}
DT_FRAME = 0.02
SUBSTEPS = 4

# ── Muscles ──────────────────────────────────────────────────────────────────
# PARAMETER NOTES (all approximate, chosen before looking at any EMG):
# - moment-arm RATIOS between joints: An et al. 1983 (J Biomech) and Brand & Hollister 1999 (Clinical Mechanics of
#   the Hand): FDP ~ MCP 11 : PIP 7.5 : DIP 5 mm; FDS ~ MCP 13 : PIP 8 mm; EDC ~ MCP 8 : PIP 3 : DIP 2 mm (extension).
# - relative strengths (proportional to physiological cross-sectional area): Lieber et al. 1992 (J Hand Surg),
#   Jacobson et al. 1992; the global torque scale is set so a fully active flexor can sweep a finger through its range
#   in about 0.1 s.
# - activation / deactivation time constants 10 / 40 ms: Zajac 1989; Thelen 2003.
# - tendon coupling: FDP slips share a muscle belly, EDC slips are linked by juncturae tendinum, FDS is the most
#   independent (Leijnse 1997; Schieber & Santello 2004): coupling c = 0.3 for FDP and EDC, 0.1 for FDS.
# - forearm positions of the muscles around the wrist ring and their depth: standard cross-sectional anatomy near the
#   distal forearm (flexors volar, extensors dorsal; FDP and FPL deep).
MUSCLES: list[str] = (
    [f"FDP_{f}" for f in FINGERS] + [f"FDS_{f}" for f in FINGERS] + [f"EDC_{f}" for f in FINGERS]
    + ["EIP", "EDM", "FPL", "EPL", "EPB", "APL"]
    + ["THENAR", "ADP"] + [f"INT_{f}_POS" for f in FINGERS] + [f"INT_{f}_NEG" for f in FINGERS]
    + [f"LUM_{f}" for f in FINGERS]
)
M = len(MUSCLES)                     # 32
EXTRINSIC = [m for m in MUSCLES if not (m.startswith(("THENAR", "ADP", "INT_", "LUM_")))]   # forearm muscles (18)
SLIP_GROUPS = {"FDP": 0.3, "FDS": 0.1, "EDC": 0.3}   # tendon-coupling strength per slip group (ablation scales all)

STRENGTH: dict[str, float] = {"FDP": 1.0, "FDS": 0.9, "EDC": 0.5, "EIP": 0.3, "EDM": 0.3, "FPL": 0.8, "EPL": 0.35,
                              "EPB": 0.25, "APL": 0.6, "THENAR": 0.6, "ADP": 0.8, "INT": 0.4, "LUM": 0.15}

# angle (degrees) around the forearm: 0 = volar midline, 90 = ulnar border, 180 = dorsal midline, 270 = radial border
POSITION_DEG: dict[str, float] = {"FDS": 0.0, "FDP": 40.0, "FPL": 320.0, "EDC": 180.0, "EDM": 140.0, "EIP": 160.0,
                                  "EPL": 200.0, "EPB": 235.0, "APL": 250.0}
SLIP_OFFSET_DEG: dict[str, float] = {"INDEX": -9.0, "MIDDLE": -3.0, "RING": 3.0, "PINKY": 9.0}
DEEP = {"FDP", "FPL", "EIP", "EPL"}
GAIN_SUPERFICIAL, GAIN_DEEP, GAIN_INTRINSIC = 1.0, 0.5, 0.05
KAPPA_SUPERFICIAL, KAPPA_DEEP = 3.0, 1.5          # von Mises concentration: larger = narrower pick-up (less crosstalk)


def muscle_family(name: str) -> str:
    """'FDP_INDEX' -> 'FDP'; 'INT_RING_POS' -> 'INT'."""
    return name.split("_")[0]


def coupling_matrix(c: float) -> np.ndarray:
    """(4, 4) share of slip i's force that reaches finger j: 1 − c stays on its own finger, c is split between its
    neighbours (index and pinky have one neighbour each). Rows sum to 1."""
    C = np.zeros((4, 4))
    for i in range(4):
        nb = [j for j in (i - 1, i + 1) if 0 <= j < 4]
        C[i, i] = 1.0 - c
        for j in nb:
            C[i, j] = c / len(nb)
    return C


def base_moment_arms() -> np.ndarray:
    """(M, J) signed moment arms (relative units; + = flexion / + AA). Finger slips act on their own finger only."""
    R = np.zeros((M, J))
    idx = {m: i for i, m in enumerate(MUSCLES)}
    for f in FINGERS:
        jt = FINGER_JOINTS[f]
        R[idx[f"FDP_{f}"], [jt["MCP"], jt["PIP"], jt["DIP"]]] = [1.0, 0.68, 0.45]
        R[idx[f"FDS_{f}"], [jt["MCP"], jt["PIP"]]] = [1.0, 0.62]
        R[idx[f"EDC_{f}"], [jt["MCP"], jt["PIP"], jt["DIP"]]] = [-1.0, -0.38, -0.25]
        R[idx[f"INT_{f}_POS"], [jt["AA"], jt["MCP"]]] = [1.0, 0.3]
        R[idx[f"INT_{f}_NEG"], [jt["AA"], jt["MCP"]]] = [-1.0, 0.3]
        R[idx[f"LUM_{f}"], [jt["MCP"], jt["PIP"], jt["DIP"]]] = [0.6, -0.4, -0.3]
    R[idx["EIP"], [5, 6]] = [-0.8, -0.3]
    R[idx["EDM"], [17, 18]] = [-0.8, -0.3]
    R[idx["FPL"], [3, 2, 0]] = [1.0, 0.6, 0.3]
    R[idx["EPL"], [3, 2, 0, 1]] = [-1.0, -0.6, -0.3, -0.3]
    R[idx["EPB"], [2, 0]] = [-1.0, -0.4]
    R[idx["APL"], [1, 0]] = [1.0, -0.3]
    R[idx["THENAR"], [0, 1, 2]] = [0.6, 0.5, 0.5]
    R[idx["ADP"], [1, 0]] = [-1.0, 0.2]
    return R


def coupled_moment_arms(R: np.ndarray, c_scale: float = 1.0) -> np.ndarray:
    """(M, J) moment arms with shared tendons: the row of slip i (of FDP, FDS or EDC) is redistributed over the four
    fingers with coupling_matrix(c_scale * c_group). With c_scale = 0 the fingers are mechanically independent."""
    R_eff = R.copy()
    idx = {m: i for i, m in enumerate(MUSCLES)}
    for g, c in SLIP_GROUPS.items():
        C = coupling_matrix(c_scale * c)
        for i, fi in enumerate(FINGERS):
            row = idx[f"{g}_{fi}"]
            own = R[row, [FINGER_JOINTS[fi][k] for k in ("MCP", "PIP", "DIP")]]
            R_eff[row] = 0.0
            for j, fj in enumerate(FINGERS):
                R_eff[row, [FINGER_JOINTS[fj][k] for k in ("MCP", "PIP", "DIP")]] += C[i, j] * own
    return R_eff


def strengths() -> np.ndarray:
    """(M,) relative maximum force of each muscle."""
    return np.array([STRENGTH[muscle_family(m)] for m in MUSCLES])


# ── Electrodes ───────────────────────────────────────────────────────────────
def muscle_positions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each muscle: angle around the forearm (radians), gain, von Mises concentration. Intrinsic: angle 0 (unused)."""
    ang, gain, kap = np.zeros(M), np.zeros(M), np.zeros(M)
    for i, m in enumerate(MUSCLES):
        fam = muscle_family(m)
        if m not in EXTRINSIC:
            gain[i], kap[i] = GAIN_INTRINSIC, 0.0
            continue
        deg = POSITION_DEG[fam] + (SLIP_OFFSET_DEG[m.split("_")[1]] if "_" in m else 0.0)
        ang[i] = math.radians(deg)
        deep = fam in DEEP
        gain[i] = GAIN_DEEP if deep else GAIN_SUPERFICIAL
        kap[i] = KAPPA_DEEP if deep else KAPPA_SUPERFICIAL
    return ang, gain, kap


def electrode_gains(n_electrodes: int = 16, kappa_scale: float = 1.0, rotation: int = 0, flip: bool = False
                    ) -> np.ndarray:
    """(n_electrodes, M) mixing matrix: electrode e picks up muscle m with gain * exp(kappa (cos(theta_e - theta_m) - 1)).

    kappa_scale > 1 narrows each muscle's pick-up (less crosstalk). Intrinsic hand muscles reach every electrode with
    a small constant gain (they are far from the wrist ring). rotation/flip move the band (virtual people).
    """
    ang, gain, kap = muscle_positions()
    theta = 2 * np.pi * np.arange(n_electrodes) / n_electrodes
    G = np.zeros((n_electrodes, M))
    for m in range(M):
        if kap[m] == 0.0:
            G[:, m] = gain[m]
        else:
            G[:, m] = gain[m] * np.exp(kappa_scale * kap[m] * (np.cos(theta - ang[m]) - 1.0))
    if flip:
        G = G[::-1]
    return np.roll(G, rotation, axis=0)


# ── Body ─────────────────────────────────────────────────────────────────────
TAU_ACT, TAU_DEACT = 0.010, 0.040
PASSIVE_HZ, DAMPING_RATIO = 1.5, 0.7
TORQUE_SCALE = 600.0          # rad/s^2 per unit of (strength x moment arm) at full activation
LIMIT_STIFFNESS = 2000.0


def inertia() -> np.ndarray:
    """(J,) relative inertia: proximal joints move more mass than distal ones."""
    I = np.ones(J)
    for f in FINGERS:
        jt = FINGER_JOINTS[f]
        I[jt["AA"]], I[jt["MCP"]], I[jt["PIP"]], I[jt["DIP"]] = 1.0, 1.0, 0.6, 0.35
    I[[0, 1, 2, 3]] = [1.2, 1.2, 0.7, 0.4]
    return I


def activation_step(a: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
    """First-order activation dynamics: a moves towards the excitation u with time constant TAU_ACT when rising and
    TAU_DEACT when falling. Exact exponential update, so it is stable for any dt."""
    tau = torch.where(u > a, torch.full_like(a, TAU_ACT), torch.full_like(a, TAU_DEACT))
    return a + (1.0 - torch.exp(-dt / tau)) * (u - a)


def muscle_torques(a: torch.Tensor, R_eff: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """(B, M) activations -> (B, J) joint torques: force of each muscle (F * a) times its moment arms, summed."""
    return TORQUE_SCALE * (a * F) @ R_eff


def body_step(q: torch.Tensor, qd: torch.Tensor, tau: torch.Tensor, p: "BodyParams", dt: float
              ) -> tuple[torch.Tensor, torch.Tensor]:
    """One semi-implicit Euler step. Acceleration = (muscle torque + passive spring to rest + damping + soft limits) / I.
    Velocity is updated first, then position uses the NEW velocity (more stable than plain Euler)."""
    over = torch.relu(q - p.q_hi) - torch.relu(p.q_lo - q)
    acc = (tau - p.k * (q - p.q_rest) - p.b * qd - LIMIT_STIFFNESS * over) / p.I
    qd = qd + dt * acc
    q = q + dt * qd
    return q, qd


@dataclass
class BodyParams:
    q_rest: torch.Tensor          # (J,) radians
    q_lo: torch.Tensor
    q_hi: torch.Tensor
    I: torch.Tensor
    k: torch.Tensor
    b: torch.Tensor
    R_eff: torch.Tensor           # (M, J)
    F: torch.Tensor               # (M,)
    c_scale: float = 1.0


def make_body(pose_train_deg: np.ndarray, c_scale: float = 1.0, device: str = "cpu") -> BodyParams:
    """Rest posture = median TRAIN pose; joint limits = 1st/99th percentile of TRAIN pose (both from pose, not EMG)."""
    I = inertia()
    k = I * (2 * np.pi * PASSIVE_HZ) ** 2
    b = 2 * DAMPING_RATIO * np.sqrt(k * I)
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
    lo, hi = np.percentile(pose_train_deg, [1, 99], axis=0)
    return BodyParams(q_rest=t(np.radians(np.median(pose_train_deg, axis=0))), q_lo=t(np.radians(lo)),
                      q_hi=t(np.radians(hi)), I=t(I), k=t(k), b=t(b),
                      R_eff=t(coupled_moment_arms(base_moment_arms(), c_scale)), F=t(strengths()), c_scale=c_scale)


def frame_step(q: torch.Tensor, qd: torch.Tensor, a: torch.Tensor, u: torch.Tensor, p: BodyParams
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the body by one 20 ms frame (SUBSTEPS substeps) with excitation u held constant."""
    dt = DT_FRAME / SUBSTEPS
    for _ in range(SUBSTEPS):
        a = activation_step(a, u, dt)
        q, qd = body_step(q, qd, muscle_torques(a, p.R_eff, p.F), p, dt)
    return q, qd, a


# ── Controller (Lazzari & Saxena-style RNN) ─────────────────────────────────
def rnn_step(h: torch.Tensor, x: torch.Tensor, W_h: torch.Tensor, W_u: torch.Tensor, b: torch.Tensor, alpha: float
             ) -> torch.Tensor:
    """h' = h + alpha * (−h + tanh(h) W_h' + x W_u' + b): a leaky recurrent network (rows of h are batch items)."""
    return h + alpha * (-h + torch.tanh(h) @ W_h.T + x @ W_u.T + b)


class Controller(torch.nn.Module):
    """Inputs per frame: task one-hot (S), reference (low-dimensional future target, K_REF), proprioception (q, qd of
    DELAY frames ago, 2J). Output: muscle excitations in (0, 1)."""

    def __init__(self, n_tasks: int, n_ref: int, hidden: int = 128, alpha: float = 0.5, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        n_in = n_tasks + n_ref + 2 * J
        self.alpha = alpha
        self.W_h = torch.nn.Parameter(0.9 * torch.linalg.qr(torch.randn(hidden, hidden, generator=g))[0])
        self.W_u = torch.nn.Parameter(torch.randn(hidden, n_in, generator=g) / math.sqrt(n_in))
        self.b = torch.nn.Parameter(torch.zeros(hidden))
        self.W_o = torch.nn.Parameter(torch.randn(M, hidden, generator=g) * 0.1 / math.sqrt(hidden))
        self.b_o = torch.nn.Parameter(torch.full((M,), -2.0))
        self.n_tasks, self.n_ref, self.hidden = n_tasks, n_ref, hidden

    def step(self, h: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = rnn_step(h, x, self.W_h, self.W_u, self.b, self.alpha)
        return h, torch.sigmoid(torch.tanh(h) @ self.W_o.T + self.b_o)


@dataclass
class Normaliser:
    """Fixed input scalings so every input is O(1)."""
    ref_mean: np.ndarray
    ref_comps: np.ndarray        # (K_REF, J) eigenposture directions (from TRAIN pose)
    ref_std: np.ndarray          # (K_REF,)
    q_scale: float = math.radians(30.0)
    qd_scale: float = math.radians(300.0)


DELAY = 2         # proprioceptive delay, frames (40 ms)
LOOKAHEAD = 5     # the reference shows the target this many frames ahead (100 ms)


def simulate(ctrl: Controller, body: BodyParams, norm: Normaliser, task: torch.Tensor, target_deg: torch.Tensor,
             q0_deg: torch.Tensor, qd0_deg: torch.Tensor, task_vec_override: torch.Tensor | None = None,
             in_scale: tuple[float, float, float] = (1.0, 1.0, 1.0), unit_mask: torch.Tensor | None = None
             ) -> dict[str, torch.Tensor]:
    """Closed-loop run. task (B, S) one-hot; target_deg (B, T, J) target pose; q0/qd0 (B, J) start state.

    The reference at frame t is the target at t + LOOKAHEAD projected on the K_REF eigenpostures.
    task_vec_override (B, H): replaces the task input's contribution task @ W_u[:, :S]' (to fit a new task's vector).
    in_scale: multipliers for the (task, reference, proprioception) inputs: 0 removes an input (lesion).
    unit_mask (H,): 0 silences hidden units (their rate is set to 0 everywhere it is used).
    Returns q_deg (B, T, J), u, a (B, T, M), rates (B, T, H).
    """
    B, T, _ = target_deg.shape
    dev = target_deg.device
    rc = torch.as_tensor(norm.ref_comps, dtype=torch.float32, device=dev)
    rm = torch.as_tensor(norm.ref_mean, dtype=torch.float32, device=dev)
    rs = torch.as_tensor(norm.ref_std, dtype=torch.float32, device=dev)
    fut = torch.cat([target_deg[:, LOOKAHEAD:], target_deg[:, -1:].expand(B, LOOKAHEAD, J)], dim=1)
    ref = ((fut - rm) @ rc.T) / rs                                        # (B, T, K_REF)
    q, qd = torch.deg2rad(q0_deg), torch.deg2rad(qd0_deg)
    a = torch.zeros(B, M, device=dev)
    h = torch.zeros(B, ctrl.hidden, device=dev)
    hist = [(q, qd)] * (DELAY + 1)
    S = ctrl.n_tasks
    mask = torch.ones(ctrl.hidden, device=dev) if unit_mask is None else unit_mask.to(dev)
    st, sr, sp = in_scale
    outs_q, outs_u, outs_a, outs_r = [], [], [], []
    for t in range(T):
        qp, qdp = hist[-(DELAY + 1)]
        prop = torch.cat([(qp - body.q_rest) / norm.q_scale, qdp / norm.qd_scale], dim=1)
        task_in = torch.zeros_like(task) if task_vec_override is not None else task
        x = torch.cat([st * task_in, sr * ref[:, t], sp * prop], dim=1)
        drive = x @ ctrl.W_u.T + ctrl.b
        if task_vec_override is not None:
            drive = drive + task_vec_override
        r_ = torch.tanh(h) * mask
        h = h + ctrl.alpha * (-h + r_ @ ctrl.W_h.T + drive)
        r_ = torch.tanh(h) * mask
        u = torch.sigmoid(r_ @ ctrl.W_o.T + ctrl.b_o)
        q, qd, a = frame_step(q, qd, a, u, body)
        hist.append((q, qd))
        hist = hist[-(DELAY + 1):]
        outs_q.append(q); outs_u.append(u); outs_a.append(a); outs_r.append(r_)
    return {"q_deg": torch.rad2deg(torch.stack(outs_q, 1)), "u": torch.stack(outs_u, 1),
            "a": torch.stack(outs_a, 1), "rates": torch.stack(outs_r, 1)}


def imitation_loss(q_deg: torch.Tensor, target_deg: torch.Tensor, valid: torch.Tensor, u: torch.Tensor,
                   rates: torch.Tensor, ctrl: Controller, lam_effort: float = 1e-2, lam_rate: float = 1e-3,
                   lam_weight: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """Pose error on valid frames (in units of 30 degrees, squared) + effort (mean u^2) + L1 on rates + L1 on weights.
    Returns (total loss, pose MSE term)."""
    err = ((q_deg - target_deg) / 30.0) ** 2
    mse = (err.mean(-1) * valid).sum() / valid.sum().clamp(min=1)
    l1w = sum(p.abs().mean() for p in (ctrl.W_h, ctrl.W_u, ctrl.W_o))
    total = mse + lam_effort * (u ** 2).mean() + lam_rate * rates.abs().mean() + lam_weight * l1w
    return total, mse


# ── Simulated EMG ────────────────────────────────────────────────────────────
def simulate_emg(a: np.ndarray, G: np.ndarray, rec: np.ndarray, rng: np.random.Generator,
                 noise_rel: float = 0.15, noise_floor: float = 0.02) -> np.ndarray:
    """(N, M) muscle activations -> (N, 16) envelope-like signal, scaled like prep.py.

    1. Mix: x = a G' (each electrode sums nearby muscles: crosstalk).
    2. Noise: signal-dependent (proportional to x, Harris & Wolpert 1998) plus a small floor, both smoothed at
       10 Hz like the real envelope.
    3. Scale each recording by its own 95th percentile (one number for all channels), as prep.py does.
    """
    from scipy import signal as sps
    x = a @ G.T
    sos = sps.butter(4, 10.0, fs=50.0, output="sos")
    eta1 = sps.sosfiltfilt(sos, rng.normal(size=x.shape), axis=0)
    eta2 = np.abs(sps.sosfiltfilt(sos, rng.normal(size=x.shape), axis=0))
    eta1 /= eta1.std() + 1e-12
    eta2 /= eta2.std() + 1e-12
    x = np.clip(x * (1.0 + noise_rel * eta1) + noise_floor * eta2 * x.mean(), 1e-8, None)
    out = np.empty_like(x)
    for r in np.unique(rec):
        m = rec == r
        out[m] = x[m] / (np.percentile(x[m], 95) + 1e-8)
    return out


# ── Ring alignment (unknown band orientation and mirroring) ─────────────────
def ring_variants(X: np.ndarray) -> list[tuple[int, bool, np.ndarray]]:
    """All 32 ways the 16 columns could line up: 16 rotations x (normal, reversed)."""
    out = []
    for flip in (False, True):
        Y = X[:, ::-1] if flip else X
        for r in range(X.shape[1]):
            out.append((r, flip, np.roll(Y, r, axis=1)))
    return out


def channel_corr(A: np.ndarray, B: np.ndarray) -> float:
    """Mean over channels of the Pearson correlation between column j of A and column j of B."""
    A = A - A.mean(0)
    B = B - B.mean(0)
    num = (A * B).sum(0)
    den = np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0)) + 1e-12
    return float(np.mean(num / den))


def align_ring(sim: np.ndarray, real: np.ndarray) -> tuple[int, bool, float]:
    """Best (rotation, flip, mean channel correlation) of sim's columns against real."""
    best = (0, False, -np.inf)
    for r, flip, Y in ring_variants(sim):
        c = channel_corr(Y, real)
        if c > best[2]:
            best = (r, flip, c)
    return best


def emg_match_loss(sim: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Condition 2 (data-constrained): mismatch between simulated and real envelopes, (B, T, 16) each.

    The electrode model is fixed; the only free factor is ONE gain per window (all channels together, like prep.py's
    one-number-per-recording scaling), solved in closed form: s = <sim, real> / <sim, sim>. No per-channel gains.
    Returns the mean squared error of s * sim against real.
    """
    s = (sim * real).sum(dim=(1, 2), keepdim=True) / ((sim * sim).sum(dim=(1, 2), keepdim=True) + 1e-8)
    return ((s * sim - real) ** 2).mean()


# ═════════════════════════════════════════════════════════════════════════════
# Data handling and closed-loop runs over whole recordings
# ═════════════════════════════════════════════════════════════════════════════
def ffill_targets(pose: np.ndarray, valid: np.ndarray, rec: np.ndarray) -> np.ndarray:
    """(N, J) targets: inside each recording, invalid rows take the last valid pose (the first rows the first one)."""
    out = pose.copy()
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        v = valid[idx]
        if not v.any():
            continue
        k = np.where(v, np.arange(len(idx)), 0)
        np.maximum.accumulate(k, out=k)
        first = int(np.argmax(v))
        k[:first] = first
        out[idx] = pose[idx][k]
    return out


def fold_body(pose: np.ndarray, valid: np.ndarray, rec: np.ndarray, split: np.ndarray, train_recs: np.ndarray,
              c_scale: float = 1.0, device: str = "cpu") -> BodyParams:
    """The body of one fold: rest posture and joint limits from the TRAINING split of that fold's TRAINING
    recordings only, so a fold's held-out tasks never shape the body its controller is trained and tested on."""
    return make_body(pose[(split == 0) & valid & np.isin(rec, train_recs)], c_scale=c_scale, device=device)


def emg_by_fold(a: np.ndarray, rec: np.ndarray, side_of_rec: np.ndarray, recs_by_fold: dict[int, np.ndarray],
                rot_by_fold: dict[int, tuple[int, int, int, int]], rng: np.random.Generator,
                kappa_scale: float = 1.0) -> np.ndarray:
    """Simulated envelopes where the recordings in recs_by_fold[f] were simulated by fold f's controller and are
    therefore recorded with fold f's band orientation rot_by_fold[f] (chosen on fold f's training data)."""
    out = np.full((len(rec), 16), np.nan)
    for f, recs in recs_by_fold.items():
        m = np.isin(rec, recs)
        part = emg_from_activations(np.where(m[:, None], a, np.nan), rec, side_of_rec, rot_by_fold[f], rng,
                                    kappa_scale=kappa_scale)
        out[m] = part[m]
    return out


def fold_of_groups(groups: list[tuple[str, np.ndarray]], fold: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """The training split used for every checkpoint: fold f holds out the task groups with index % 2 == f.
    Returns (held-out group names, held-out recordings, training recordings)."""
    held = [g for i, g in enumerate(groups) if i % 2 == fold]
    held_recs = np.concatenate([recs for _, recs in held])
    n_rec = int(max(max(recs) for _, recs in groups)) + 1
    return [n for n, _ in held], held_recs, np.setdiff1d(np.arange(n_rec), held_recs)


def load_controller(path: str, device: str = "cpu") -> tuple[Controller, Normaliser, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    ctrl = Controller(cfg["n_tasks"], 3, seed=cfg["seed"]).to(device)
    ctrl.load_state_dict(ck["state"])
    ctrl.eval()
    n = ck["norm"]
    return ctrl, Normaliser(n["ref_mean"], n["ref_comps"], n["ref_std"]), cfg


def run_segments(ctrl: Controller, body: BodyParams, norm: Normaliser, target: np.ndarray, rec: np.ndarray,
                 seg_mask: np.ndarray, task_of_rec: np.ndarray | None, n_tasks: int, recs: np.ndarray,
                 task_vec: np.ndarray | None = None, in_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 unit_mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Run the controller closed-loop over the frames seg_mask of each recording in recs (one contiguous stretch per
    recording: from its first to its last seg_mask frame), all recordings in one padded batch.

    task_of_rec: (R,) task index per recording, or None for no task input. task_vec (H,): a fitted task vector used
    instead of any one-hot. Returns full-length (N, ...) arrays q_deg, a, rates (NaN outside the simulated frames).
    """
    N = len(rec)
    out = {"q_deg": np.full((N, J), np.nan), "a": np.full((N, M), np.nan), "rates": np.full((N, ctrl.hidden), np.nan)}
    spans = []
    for r in recs:
        idx = np.where((rec == r) & seg_mask)[0]
        if len(idx):
            spans.append((r, idx[0], idx[-1] + 1))
    if not spans:
        return out
    T = max(e - s for _, s, e in spans)
    B = len(spans)
    tgt = np.zeros((B, T, J), np.float32)
    for i, (_, s, e) in enumerate(spans):
        tgt[i, : e - s] = target[s:e]
        tgt[i, e - s:] = target[e - 1]
    tgt_t = torch.tensor(tgt)
    task = torch.zeros(B, n_tasks)
    if task_of_rec is not None and task_vec is None:
        for i, (r, _, _) in enumerate(spans):
            task[i, task_of_rec[r]] = 1.0
    override = None if task_vec is None else torch.tensor(task_vec, dtype=torch.float32)[None].expand(B, -1)
    qd0 = (tgt_t[:, 1] - tgt_t[:, 0]) / DT_FRAME
    um = None if unit_mask is None else torch.tensor(unit_mask, dtype=torch.float32)
    with torch.no_grad():
        o = simulate(ctrl, body, norm, task, tgt_t, tgt_t[:, 0], qd0, task_vec_override=override, in_scale=in_scale,
                     unit_mask=um)
    for i, (_, s, e) in enumerate(spans):
        for k in out:
            out[k][s:e] = o[k][i, : e - s].numpy()
    return out


def block_errors(Y: np.ndarray, Y_hat: np.ndarray, mask: np.ndarray, block: np.ndarray) -> np.ndarray:
    """(n_blocks,) mean absolute joint-angle error per block over mask frames (NaN for blocks without frames)."""
    out = []
    for b in np.unique(block):
        m = mask & (block == b) & ~np.isnan(Y_hat).any(1)
        out.append(np.mean(np.abs(Y[m] - Y_hat[m])) if m.any() else np.nan)
    return np.array(out)


def co_contraction(a: np.ndarray) -> np.ndarray:
    """(N,) mean over the four fingers of min(flexor activation, extensor activation): flexor = mean of the FDP and
    FDS slips, extensor = the EDC slip. High values mean antagonists are active together (stiffening, not moving)."""
    idx = {m: i for i, m in enumerate(MUSCLES)}
    vals = []
    for f in FINGERS:
        flex = 0.5 * (a[:, idx[f"FDP_{f}"]] + a[:, idx[f"FDS_{f}"]])
        vals.append(np.minimum(flex, a[:, idx[f"EDC_{f}"]]))
    return np.mean(vals, axis=0)


def emg_from_activations(a: np.ndarray, rec: np.ndarray, side_of_rec: np.ndarray, rot: tuple[int, int, int, int],
                         rng: np.random.Generator, kappa_scale: float = 1.0) -> np.ndarray:
    """Simulated envelopes for every frame with activations (NaN rows stay NaN): left and right recordings use their
    own (rotation, flip) of the ring, chosen on training data."""
    out = np.full((len(rec), 16), np.nan)
    ok = ~np.isnan(a).any(1)
    for side, (r_, f_) in (("left", rot[:2]), ("right", rot[2:])):
        m = ok & (side_of_rec[rec] == side)
        if m.any():
            G = electrode_gains(kappa_scale=kappa_scale, rotation=r_, flip=bool(f_))
            out[m] = simulate_emg(a[m], G, rec[m], rng)
    return out


def emg_prediction_r(sim: np.ndarray, real: np.ndarray, mask: np.ndarray, block: np.ndarray) -> np.ndarray:
    """(n_blocks,) mean per-channel correlation between simulated and real envelopes on mask frames, per block."""
    out = []
    for b in np.unique(block):
        m = mask & (block == b) & ~np.isnan(sim).any(1)
        out.append(channel_corr(sim[m], real[m]) if m.sum() > 50 else np.nan)
    return np.array(out)


# ═════════════════════════════════════════════════════════════════════════════
# Reverse-engineering the controller
# ═════════════════════════════════════════════════════════════════════════════
def rnn_map(ctrl: Controller, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """F(h) for a held input x: one step of the controller's recurrence (no body)."""
    return rnn_step(h, x, ctrl.W_h, ctrl.W_u, ctrl.b, ctrl.alpha)


def speed(ctrl: Controller, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """q(h) = 1/2 ||F(h) − h||^2 per row: zero at a fixed point, small at a slow point (Sussillo & Barak 2013)."""
    return 0.5 * ((rnn_map(ctrl, h, x) - h) ** 2).sum(-1)


def jacobian(ctrl: Controller, h: np.ndarray) -> np.ndarray:
    """(H, H) dF/dh at h: I + alpha (−I + W_h diag(1 − tanh(h)^2)). Its eigenvalues describe the local dynamics."""
    W = ctrl.W_h.detach().numpy()
    H = len(h)
    return np.eye(H) + ctrl.alpha * (-np.eye(H) + W * (1.0 - np.tanh(h) ** 2)[None, :])


def find_fixed_points(ctrl: Controller, x: np.ndarray, h_starts: np.ndarray, n_steps: int = 300, lr: float = 0.05,
                      tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Minimise q(h) from every start with Adam (input x held fixed). Returns (h*, q(h*)) for all starts."""
    h = torch.tensor(h_starts, dtype=torch.float32, requires_grad=True)
    xt = torch.tensor(x, dtype=torch.float32)[None].expand(len(h_starts), -1)
    opt = torch.optim.Adam([h], lr=lr)
    for _ in range(n_steps):
        opt.zero_grad()
        qv = speed(ctrl, h, xt)
        qv.sum().backward()
        opt.step()
    with torch.no_grad():
        qv = speed(ctrl, h, xt)
    return h.detach().numpy(), qv.numpy()


def unique_points(H: np.ndarray, q: np.ndarray, q_max: float = 1e-4, min_dist: float = 0.1) -> np.ndarray:
    """Keep slow points with q < q_max and drop near-duplicates (Euclidean distance < min_dist)."""
    keep = []
    for i in np.argsort(q):
        if q[i] > q_max:
            break
        if all(np.linalg.norm(H[i] - H[j]) > min_dist for j in keep):
            keep.append(i)
    return H[keep]


def classify_eigs(eigs: np.ndarray, tol: float = 0.02) -> str:
    """'unstable' if any |λ| > 1 + tol; 'integrator-like' if the largest |λ| is within tol of 1 and real;
    'rotational' if the slowest mode is complex; otherwise 'stable'."""
    lead = eigs[np.argmax(np.abs(eigs))]
    if np.abs(lead) > 1 + tol:
        return "unstable"
    if abs(np.abs(lead) - 1) <= tol and abs(lead.imag) < 1e-3:
        return "integrator-like"
    if abs(lead.imag) > 1e-3:
        return "rotational"
    return "stable"


def fit_lds(Z: np.ndarray, X: np.ndarray, starts_ok: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least squares z(t+1) = A z(t) + B x(t) + c over steps t with starts_ok[t] (t and t+1 in the same run).
    Returns A (d, d), B (d, m), c (d,)."""
    t = np.where(starts_ok)[0]
    D = np.hstack([Z[t], X[t], np.ones((len(t), 1))])
    W, *_ = np.linalg.lstsq(D, Z[t + 1], rcond=None)
    d, m = Z.shape[1], X.shape[1]
    return W[:d].T, W[d:d + m].T, W[-1]


def cosine_matrix(V: np.ndarray) -> np.ndarray:
    """(n, n) cosine similarity between the ROWS of V."""
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return Vn @ Vn.T


def offdiag_corr(A: np.ndarray, B: np.ndarray) -> float:
    """Pearson correlation between the off-diagonal entries of two square matrices (a Mantel-style comparison)."""
    m = ~np.eye(len(A), dtype=bool)
    return float(np.corrcoef(A[m], B[m])[0, 1])


def task_variance(rates: np.ndarray, task: np.ndarray, tasks: list[int]) -> np.ndarray:
    """(H, n_tasks) variance of each unit's rate over time within each task (Yang et al. 2019)."""
    return np.stack([rates[task == k].var(axis=0) for k in tasks], axis=1)


def kmeans(X: np.ndarray, k: int, seed: int = 0, n_iter: int = 100) -> np.ndarray:
    """Plain k-means (Lloyd's algorithm) with k-means++ starts. Returns labels (n,)."""
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min([((X - c) ** 2).sum(1) for c in C], axis=0)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(n_iter):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab


def silhouette(X: np.ndarray, lab: np.ndarray) -> float:
    """Mean silhouette: (b − a) / max(a, b), a = mean distance to own cluster, b = to the nearest other cluster."""
    Dm = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    s = []
    for i in range(len(X)):
        own = lab == lab[i]
        if own.sum() <= 1:
            s.append(0.0)
            continue
        a = Dm[i, own].sum() / (own.sum() - 1)
        b = min(Dm[i, lab == c].mean() for c in np.unique(lab) if c != lab[i])
        s.append((b - a) / max(a, b))
    return float(np.mean(s))


def cca_corrs(X: np.ndarray, Y: np.ndarray, k: int = 10, reg: float = 1e-3) -> np.ndarray:
    """Top-k canonical correlations between two sets of variables (rows = the same time points)."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    def isqrt(C: np.ndarray) -> np.ndarray:
        w, V = np.linalg.eigh(C + reg * np.eye(len(C)))
        return V @ np.diag(w ** -0.5) @ V.T
    Cxx, Cyy, Cxy = X.T @ X / len(X), Y.T @ Y / len(Y), X.T @ Y / len(X)
    s = np.linalg.svd(isqrt(Cxx) @ Cxy @ isqrt(Cyy), compute_uv=False)
    return s[:k]


def procrustes_disparity(X: np.ndarray, Y: np.ndarray) -> float:
    """Shape difference after the best rotation + scaling (0 = identical up to rotation, 1 = unrelated)."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    X = X / np.linalg.norm(X)
    Y = Y / np.linalg.norm(Y)
    s = np.linalg.svd(X.T @ Y, compute_uv=False)
    return float(1.0 - s.sum() ** 2)


# ═════════════════════════════════════════════════════════════════════════════
# Training (given code; the notebook loads checkpoints by default)
# ═════════════════════════════════════════════════════════════════════════════
N_REF = 3          # the reference is the future target projected on the top-3 eigenpostures


def make_normaliser(pose: np.ndarray, fit_mask: np.ndarray) -> Normaliser:
    """Eigenposture reference fitted on the training frames only."""
    Pc = pose[fit_mask] - pose[fit_mask].mean(0)
    _, _, Vt = np.linalg.svd(Pc, full_matrices=False)
    comps = Vt[:N_REF]
    return Normaliser(pose[fit_mask].mean(0), comps, (Pc @ comps.T).std(0))


def train_windows(rec: np.ndarray, split: np.ndarray, valid: np.ndarray, recs: np.ndarray, T: int,
                  extra: np.ndarray | None = None, stride: int = 10) -> np.ndarray:
    """(n, 2) array of (recording, first frame) for every T-frame window inside the TRAIN split of recordings recs
    (and inside ``extra`` if given) that starts on a valid frame."""
    out = []
    for r_ in recs:
        idx = np.where(rec == r_)[0]
        ok = split[idx] == 0
        if extra is not None:
            ok &= extra[idx]
        for s0 in range(0, len(idx) - T, stride):
            if ok[s0:s0 + T].all() and valid[idx[s0]]:
                out.append((r_, idx[s0]))
    return np.array(out, dtype=np.int64)


def train_controller(pose: np.ndarray, valid: np.ndarray, rec: np.ndarray, split: np.ndarray, emg: np.ndarray,
                     side_of_rec: np.ndarray, task_of_rec: np.ndarray, n_tasks: int, train_recs: np.ndarray,
                     cond: str = "goal", seed: int = 0, c_scale: float = 1.0, iters: int = 2000,
                     lam_effort: float = 0.1, lam_emg: float = 1.0, rot: tuple[int, int, int, int] = (0, 0, 0, 0),
                     inner: np.ndarray | None = None, T: int = 100, batch: int = 128, device: str = "cpu",
                     log_every: int = 100) -> tuple[Controller, Normaliser, list[tuple[int, float, float]]]:
    """Backprop through the body. cond 'goal': pose loss only. cond 'emg': + lam_emg * emg_match_loss against the
    REAL envelopes, through the fixed electrode model with the ring orientation ``rot`` (chosen on training data)."""
    torch.manual_seed(seed)
    fit = (split == 0) & valid & np.isin(rec, train_recs)
    body = fold_body(pose, valid, rec, split, train_recs, c_scale=c_scale, device=device)
    norm = make_normaliser(pose, fit)
    target = ffill_targets(pose, valid, rec)
    W = train_windows(rec, split, valid, train_recs, T, inner)
    side = side_of_rec[rec]
    G_left = torch.tensor(electrode_gains(rotation=rot[0], flip=bool(rot[1])), dtype=torch.float32, device=device)
    G_right = torch.tensor(electrode_gains(rotation=rot[2], flip=bool(rot[3])), dtype=torch.float32, device=device)
    ctrl = Controller(n_tasks, N_REF, seed=seed).to(device)
    opt = torch.optim.Adam(ctrl.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    rng = np.random.default_rng(seed)
    log = []
    for it in range(iters):
        pick = W[rng.integers(0, len(W), batch)]
        ix = pick[:, 1:2] + np.arange(T)[None]
        tgt = torch.tensor(target[ix], dtype=torch.float32, device=device)
        val = torch.tensor(valid[ix], dtype=torch.float32, device=device)
        task = torch.zeros(batch, n_tasks, device=device)
        task[torch.arange(batch), torch.tensor(task_of_rec[pick[:, 0]], device=device)] = 1.0
        out = simulate(ctrl, body, norm, task, tgt, tgt[:, 0], (tgt[:, 1] - tgt[:, 0]) / DT_FRAME)
        loss, mse = imitation_loss(out["q_deg"], tgt, val, out["u"], out["rates"], ctrl, lam_effort=lam_effort)
        e_term = torch.zeros(())
        if cond == "emg":
            left = torch.tensor(side[ix[:, 0]] == "left", device=device)[:, None, None]
            sim = torch.where(left, out["a"] @ G_left.T, out["a"] @ G_right.T)
            e_term = emg_match_loss(sim, torch.tensor(emg[ix], dtype=torch.float32, device=device))
            loss = loss + lam_emg * e_term
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % log_every == 0 or it == iters - 1:
            log.append((it, float(mse.detach()), float(e_term.detach())))
    return ctrl, norm, log


# ═════════════════════════════════════════════════════════════════════════════
# Compositional transfer: learn a NEW task by fitting only its task-input vector
# ═════════════════════════════════════════════════════════════════════════════
def calibration_windows(cal_mask: np.ndarray, rec: np.ndarray, T: int, stride: int = 10) -> np.ndarray:
    """First frames of every T-frame window lying entirely inside cal_mask and inside one recording."""
    starts = []
    for r_ in np.unique(rec[cal_mask]):
        idx = np.where((rec == r_) & cal_mask)[0]
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for run in runs:
            for s0 in range(0, len(run) - T + 1, stride):
                starts.append(run[s0])
    return np.array(starts, dtype=np.int64)


def fit_task_vector(ctrl: Controller, body: BodyParams, norm: Normaliser, target: np.ndarray, valid: np.ndarray,
                    starts: np.ndarray, T: int = 50, steps: int = 40, mode: str = "vector", lr: float | None = None,
                    lam_effort: float = 0.3, seed: int = 0) -> tuple[Controller, np.ndarray]:
    """Calibrate on the windows starting at ``starts`` (pose loss only).

    mode 'vector': learn only a new task-input vector v (H,), everything else frozen (Lazzari & Saxena's test).
    mode 'all': copy the controller and fine-tune every weight (no task input), the usual "recalibrate everything".
    Returns (controller, v): v is zeros for mode 'all'.
    """
    import copy
    torch.manual_seed(seed)
    H = ctrl.hidden
    if len(starts) == 0:
        return ctrl, np.zeros(H)
    ix = starts[:, None] + np.arange(T)[None]
    tgt = torch.tensor(target[ix], dtype=torch.float32)
    val = torch.tensor(valid[ix], dtype=torch.float32)
    task = torch.zeros(len(starts), ctrl.n_tasks)
    qd0 = (tgt[:, 1] - tgt[:, 0]) / DT_FRAME
    if mode == "vector":
        v = torch.zeros(H, requires_grad=True)
        opt = torch.optim.Adam([v], lr=lr or 0.05)
        net = ctrl
    else:
        net = copy.deepcopy(ctrl)
        for p in net.parameters():
            p.requires_grad_(True)
        v = torch.zeros(H)
        opt = torch.optim.Adam(net.parameters(), lr=lr or 1e-3)
    for _ in range(steps):
        out = simulate(net, body, norm, task, tgt, tgt[:, 0], qd0,
                       task_vec_override=v[None].expand(len(starts), -1) if mode == "vector" else None)
        loss, _ = imitation_loss(out["q_deg"], tgt, val, out["u"], out["rates"], net, lam_effort=lam_effort)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net, v.detach().numpy()
