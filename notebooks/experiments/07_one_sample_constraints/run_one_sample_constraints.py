#!/usr/bin/env python3
"""One-sample sEMG→pose constraint experiments.

Runs the do-next list from docs/sensorimotor-improvements-one-sample.md
on a single emg2pose mini recording (true n=1 file), with:
  1) honest gap-split protocol + baselines
  2) kinematic soft constraints on NMF→PCA linear decode
  3) Box 5.4 regularizers on a tiny leaky RNN
  4) synergy-space RNN vs raw-EMG RNN
  5) within-train augmentation
  6) leakage stress test (fit bases on all time vs train-only)

Outputs JSON + a short markdown summary under data/one_sample_constraints/.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from sklearn.decomposition import NMF, PCA
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "emg2pose_dataset_mini"
OUT_DIR = ROOT / "data" / "one_sample_constraints"

FS_IN = 2000
FS_OUT = 50
DECIM = FS_IN // FS_OUT
TRAIN_FRAC = 0.8
GAP = FS_OUT  # 1 s
SEQ_LEN = 100  # 2 s
STRIDE = 50
SEED = 0
K_EMG = 4
K_POSE = 7
HIDDEN = 32
EPOCHS_LINEAR = 150
EPOCHS_RNN = 40
BATCH = 32
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def envelope_emg(emg: np.ndarray) -> np.ndarray:
    x = np.abs(emg.astype(np.float64))
    b, a = signal.butter(2, 5.0 / (FS_IN / 2.0), btype="low")
    x = signal.filtfilt(b, a, x, axis=0)
    n = (len(x) // DECIM) * DECIM
    x = x[:n].reshape(-1, DECIM, x.shape[1]).mean(axis=1)
    return x.astype(np.float32)


def load_recording(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        ts = f["emg2pose/timeseries"]
        emg = np.asarray(ts["emg"])
        pose = np.asarray(ts["joint_angles"])
    env = envelope_emg(emg)
    n = (len(pose) // DECIM) * DECIM
    pose = pose[:n].reshape(-1, DECIM, pose.shape[1]).mean(axis=1).astype(np.float32)
    t = min(len(env), len(pose))
    env, pose = env[:t], pose[:t]
    valid = np.isfinite(pose).all(axis=1)
    m = re.search(r"recording-(\d+)_(left|right)", path.name)
    stage = m.group(1) if m else "?"
    side = 0.0 if (m and m.group(2) == "left") else 1.0
    return {
        "emg": env,
        "pose": pose,
        "valid": valid,
        "stage": stage,
        "side": side,
        "path": str(path),
        "name": path.name,
    }


def split_indices(t: int) -> tuple[int, int]:
    n_train = int(t * TRAIN_FRAC)
    train_end = max(SEQ_LEN, n_train - GAP // 2)
    test_start = min(t - SEQ_LEN, n_train + GAP // 2)
    return train_end, test_start


def mae_deg(yhat: np.ndarray, y: np.ndarray) -> float:
    # joint angles in radians → degrees
    return float(np.mean(np.abs(yhat - y)) * 180.0 / math.pi)


def window_xy(
    x: np.ndarray,
    y: np.ndarray,
    valid: np.ndarray,
    split: str,
    augment: str | None = None,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    t = len(x)
    train_end, test_start = split_indices(t)
    if split == "train":
        starts = range(0, train_end - SEQ_LEN + 1, STRIDE)
    else:
        starts = range(test_start, t - SEQ_LEN + 1, STRIDE)
    out = []
    for s in starts:
        sl = slice(s, s + SEQ_LEN)
        if valid[sl].mean() < 0.9:
            continue
        xi = x[sl].astype(np.float32).copy()
        yi = y[sl].astype(np.float32).copy()
        if split == "train" and augment and rng is not None:
            xi, yi = apply_augment(xi, yi, augment, rng)
        out.append((xi, yi))
    return out


def apply_augment(
    x: np.ndarray, y: np.ndarray, kind: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    x = x.copy()
    y = y.copy()
    if kind in ("noise", "all"):
        x = x + rng.normal(0, 0.05, size=x.shape).astype(np.float32) * (x.std() + 1e-6)
    if kind in ("dropout", "all"):
        mask = rng.random(x.shape[1]) > 0.15
        if not mask.any():
            mask[0] = True
        x = x * mask.astype(np.float32)
    if kind in ("gain", "all"):
        g = rng.uniform(0.85, 1.15, size=(1, x.shape[1])).astype(np.float32)
        x = x * g
    return x, y


class SeqDataset(Dataset):
    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray]]):
        self.x = [torch.from_numpy(a) for a, _ in pairs]
        self.y = [torch.from_numpy(b) for _, b in pairs]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


class LinearBottleneck(nn.Module):
    """x (synergies) → pose angles via optional eigenposture bottleneck."""

    def __init__(self, n_in: int, n_out: int, k_pose: int | None = None):
        super().__init__()
        self.k_pose = k_pose
        if k_pose:
            self.net = nn.Sequential(
                nn.Linear(n_in, k_pose),
                nn.Linear(k_pose, n_out),
            )
        else:
            self.net = nn.Linear(n_in, n_out)

    def forward(self, x):  # (B,T,F)
        return self.net(x)


class LeakyRNN(nn.Module):
    def __init__(self, n_in: int, n_hidden: int, n_out: int, tau: float = 5.0, dt: float = 1.0):
        super().__init__()
        self.alpha = dt / tau
        self.W_h = nn.Parameter(0.01 * torch.randn(n_hidden, n_hidden))
        self.W_in = nn.Linear(n_in, n_hidden, bias=True)
        self.readout = nn.Linear(n_hidden, n_out)
        self.act = nn.Softplus()
        self.n_hidden = n_hidden

    def forward(self, u, return_h: bool = False):
        # u: (B,T,F)
        b, t, _ = u.shape
        h = u.new_zeros(b, self.n_hidden)
        x = u.new_zeros(b, self.n_hidden)
        hs = []
        ys = []
        for ti in range(t):
            x = (1 - self.alpha) * x + self.alpha * (h @ self.W_h.T + self.W_in(u[:, ti]))
            h = self.act(x)
            ys.append(self.readout(h))
            if return_h:
                hs.append(h)
        y = torch.stack(ys, dim=1)
        if return_h:
            return y, torch.stack(hs, dim=1)
        return y


def kinematic_penalties(yhat: torch.Tensor, y_max: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # yhat: (B,T,J)
    vel = yhat[:, 1:] - yhat[:, :-1]
    smooth = (vel**2).mean()
    if yhat.size(1) >= 3:
        acc = yhat[:, 2:] - 2 * yhat[:, 1:-1] + yhat[:, :-2]
        jerk = (acc**2).mean()
    else:
        jerk = yhat.new_zeros(())
    lim = torch.relu(yhat.abs() - y_max).pow(2).mean()
    return smooth, jerk, lim


def simple_dynamics_penalty(model: LeakyRNN, h: torch.Tensor, x_pre: torch.Tensor | None = None) -> torch.Tensor:
    # Approximate ||∂(W_h h)/∂x||_F^2 via softplus' * W_h on batch of hidden states.
    # h = softplus(x) ⇒ ∂h/∂x = sigmoid(x). Use h itself: softplus'(x)=sigmoid(x)=1-exp(-h) approx unstable;
    # use torch autodiff on a probe instead for stability.
    # Practical surrogate used in pilots: ||W_h||_F^2 * mean(h^2) — weaker than true Jacobian.
    # Better: finite-diff Jacobian Frobenius on a subsample of states.
    wh = model.W_h
    # For softplus, J_h = diag(σ(x)) W_h; we don't have x, but σ(x)=1-e^{-softplus(x)} no —
    # softplus'(x) = sigmoid(x). Recover x ≈ log(e^h - 1) for h>0.
    h_flat = h.reshape(-1, h.size(-1)).detach()
    if h_flat.size(0) > 128:
        idx = torch.randperm(h_flat.size(0), device=h.device)[:128]
        h_flat = h_flat[idx]
    # softplus inverse
    x = torch.log(torch.expm1(h_flat.clamp(min=1e-4)))
    s = torch.sigmoid(x)  # (N,H)
    # ∂(W_h h)/∂x = W_h * diag(s)  → rows scaled; Frobenius^2 = sum_ij (W_hij * s_j)^2 averaged over samples
    # For each sample: ||W_h @ diag(s)||_F^2 = sum_i sum_j W_hij^2 * s_j^2
    w2 = wh.pow(2)  # (H,H) row i col j
    pen = (w2.unsqueeze(0) * s.unsqueeze(1).pow(2)).sum(dim=(1, 2)).mean()
    return pen


@dataclass
class RunResult:
    name: str
    test_mae_deg: float
    train_mae_deg: float | None = None
    notes: str = ""


def fit_bases(
    emg_train: np.ndarray, pose_train: np.ndarray, leak: bool = False, emg_all=None, pose_all=None
):
    if leak:
        assert emg_all is not None and pose_all is not None
        emg_fit, pose_fit = emg_all, pose_all
    else:
        emg_fit, pose_fit = emg_train, pose_train
    emg_fit = np.clip(emg_fit, 0, None)
    nmf = NMF(n_components=K_EMG, init="nndsvda", max_iter=800, random_state=SEED)
    nmf.fit(emg_fit)
    pca = PCA(n_components=K_POSE, random_state=SEED)
    pca.fit(pose_fit)
    emg_mean = emg_fit.mean(axis=0)
    emg_std = emg_fit.std(axis=0) + 1e-6
    return nmf, pca, emg_mean, emg_std


def transform_emg(emg, nmf, mean, std):
    z = (emg - mean) / std
    z = np.clip(z, 0, None)  # NMF needs nonneg; after z-score use abs or shift
    # Better: NMF on raw nonnegative envelopes — transform with same scale as fit
    # Refit transform on raw clipped envelopes consistently:
    return nmf.transform(np.clip(emg, 0, None)).astype(np.float32)


def _init_toward_mean_pose(model: nn.Module, pose_mean: np.ndarray) -> None:
    """Start near mean-pose baseline so scarce-data runs don't starve."""
    last = None
    if isinstance(model, LinearBottleneck):
        last = model.net[-1] if isinstance(model.net, nn.Sequential) else model.net
    elif isinstance(model, LeakyRNN):
        last = model.readout
    if last is None:
        return
    with torch.no_grad():
        last.weight.zero_()
        last.bias.copy_(torch.tensor(pose_mean, dtype=last.bias.dtype))


def train_linear(
    train_pairs,
    test_pairs,
    n_in,
    n_out,
    alphas: dict,
    y_max: torch.Tensor,
    pose_mean: np.ndarray,
    epochs=EPOCHS_LINEAR,
):
    model = LinearBottleneck(n_in, n_out, k_pose=None).to(DEVICE)
    _init_toward_mean_pose(model, pose_mean)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl = DataLoader(SeqDataset(train_pairs), batch_size=BATCH, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            yhat = model(xb)
            loss = ((yhat - yb) ** 2).mean()
            if alphas.get("smooth", 0) or alphas.get("jerk", 0) or alphas.get("lim", 0):
                s, j, lim = kinematic_penalties(yhat, y_max)
                loss = (
                    loss
                    + alphas.get("smooth", 0) * s
                    + alphas.get("jerk", 0) * j
                    + alphas.get("lim", 0) * lim
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_mae = eval_model(model, train_pairs)
    test_mae = eval_model(model, test_pairs)
    return model, train_mae, test_mae


@torch.no_grad()
def eval_model(model, pairs) -> float:
    if not pairs:
        return float("nan")
    model.eval()
    ys, yh = [], []
    dl = DataLoader(SeqDataset(pairs), batch_size=BATCH)
    for xb, yb in dl:
        xb = xb.to(DEVICE)
        pred = model(xb) if not isinstance(model, LeakyRNN) else model(xb)
        ys.append(yb.numpy())
        yh.append(pred.cpu().numpy())
    return mae_deg(np.concatenate(yh), np.concatenate(ys))


def train_rnn(
    train_pairs,
    test_pairs,
    n_in,
    n_out,
    alphas: dict,
    y_max: torch.Tensor,
    pose_mean: np.ndarray,
    hidden=HIDDEN,
    epochs=EPOCHS_RNN,
) -> tuple[nn.Module, float, float]:
    model = LeakyRNN(n_in, hidden, n_out).to(DEVICE)
    _init_toward_mean_pose(model, pose_mean)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    dl = DataLoader(SeqDataset(train_pairs), batch_size=min(BATCH, max(1, len(train_pairs))), shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            yhat, h = model(xb, return_h=True)
            loss = ((yhat - yb) ** 2).mean()
            if alphas.get("smooth", 0) or alphas.get("jerk", 0) or alphas.get("lim", 0):
                s, j, lim = kinematic_penalties(yhat, y_max)
                loss = (
                    loss
                    + alphas.get("smooth", 0) * s
                    + alphas.get("jerk", 0) * j
                    + alphas.get("lim", 0) * lim
                )
            if alphas.get("rate", 0):
                loss = loss + alphas["rate"] * h.abs().mean()
            if alphas.get("w", 0):
                loss = loss + alphas["w"] * (
                    model.W_in.weight.pow(2).mean()
                    + model.readout.weight.pow(2).mean()
                    + model.W_h.abs().mean()
                )
            if alphas.get("dyn", 0):
                loss = loss + alphas["dyn"] * simple_dynamics_penalty(model, h)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model, eval_model(model, train_pairs), eval_model(model, test_pairs)


def mean_pose_baseline(pose_train: np.ndarray, pose_test: np.ndarray) -> float:
    mu = pose_train.mean(axis=0, keepdims=True)
    return mae_deg(np.repeat(mu, len(pose_test), axis=0), pose_test)


def ridge_decode(x_tr, y_tr, x_te, y_te, lam=1e-1) -> float:
    # Flatten time is wrong for sequences; do per-timestep pooled ridge on frames
    xt = x_tr.reshape(-1, x_tr.shape[-1])
    yt = y_tr.reshape(-1, y_tr.shape[-1])
    xe = x_te.reshape(-1, x_te.shape[-1])
    ye = y_te.reshape(-1, y_te.shape[-1])
    # center
    xmu, ymu = xt.mean(0), yt.mean(0)
    xtc, ytc = xt - xmu, yt - ymu
    xtc = np.concatenate([xtc, np.ones((len(xtc), 1))], axis=1)
    xe_c = np.concatenate([xe - xmu, np.ones((len(xe), 1))], axis=1)
    a = xtc.T @ xtc + lam * np.eye(xtc.shape[1])
    w = np.linalg.solve(a, xtc.T @ ytc)
    pred = xe_c @ w + ymu
    return mae_deg(pred, ye)


def stack_pairs(pairs):
    if not pairs:
        return np.zeros((0, SEQ_LEN, 1), np.float32), np.zeros((0, SEQ_LEN, 20), np.float32)
    x = np.stack([a for a, _ in pairs], axis=0)
    y = np.stack([b for _, b in pairs], axis=0)
    return x, y


def run_all(recording_path: Path) -> dict:
    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = load_recording(recording_path)
    emg, pose, valid = rec["emg"], rec["pose"], rec["valid"]
    train_end, test_start = split_indices(len(emg))
    emg_tr, pose_tr = emg[:train_end], pose[:train_end]
    emg_te, pose_te = emg[test_start:], pose[test_start:]
    y_max_np = np.percentile(np.abs(pose_tr), 99, axis=0).astype(np.float32)
    y_max = torch.tensor(y_max_np, device=DEVICE).view(1, 1, -1)
    pose_mean = pose_tr.mean(axis=0).astype(np.float32)

    results: list[dict] = []

    # --- Protocol / baselines ---
    mean_mae = mean_pose_baseline(pose_tr, pose_te)
    results.append({"name": "mean_pose_baseline", "test_mae_deg": mean_mae, "notes": "predict train mean pose"})

    # Fit bases train-only (honest)
    nmf, pca, emg_mean, emg_std = fit_bases(emg_tr, pose_tr, leak=False)
    syn_all = transform_emg(emg, nmf, emg_mean, emg_std)
    train_syn = window_xy(syn_all, pose, valid, "train")
    test_syn = window_xy(syn_all, pose, valid, "test")
    train_raw = window_xy(emg, pose, valid, "train")
    test_raw = window_xy(emg, pose, valid, "test")

    xtr, ytr = stack_pairs(train_syn)
    xte, yte = stack_pairs(test_syn)
    results.append(
        {
            "name": "ridge_nmf_synergies",
            "test_mae_deg": ridge_decode(xtr, ytr, xte, yte),
            "notes": f"K_EMG={K_EMG}, train-only NMF, gap split",
            "n_train_windows": len(train_syn),
            "n_test_windows": len(test_syn),
        }
    )
    xtr_r, ytr_r = stack_pairs(train_raw)
    xte_r, yte_r = stack_pairs(test_raw)
    results.append(
        {
            "name": "ridge_raw_emg",
            "test_mae_deg": ridge_decode(xtr_r, ytr_r, xte_r, yte_r),
            "notes": "raw 16-ch envelopes → pose ridge",
        }
    )

    # --- Linear + kinematic constraints ---
    constraint_grid = [
        ("linear_nmf_mse", {"smooth": 0.0, "jerk": 0.0, "lim": 0.0}),
        ("linear_nmf_smooth", {"smooth": 1e-2, "jerk": 0.0, "lim": 0.0}),
        ("linear_nmf_smooth_jerk", {"smooth": 1e-2, "jerk": 1e-3, "lim": 0.0}),
        ("linear_nmf_kin_full", {"smooth": 1e-2, "jerk": 1e-3, "lim": 1e-2}),
    ]
    for name, alphas in constraint_grid:
        _, tr, te = train_linear(train_syn, test_syn, K_EMG, 20, alphas, y_max, pose_mean)
        results.append({"name": name, "train_mae_deg": tr, "test_mae_deg": te, "alphas": alphas})

    # --- Box 5.4 RNN ablations on synergies ---
    rnn_grid = [
        ("rnn_syn_mse", {"rate": 0.0, "w": 0.0, "dyn": 0.0, "smooth": 0.0}),
        ("rnn_syn_box54", {"rate": 1e-3, "w": 1e-3, "dyn": 1e-4, "smooth": 0.0}),
        (
            "rnn_syn_box54_kin",
            {"rate": 1e-3, "w": 1e-3, "dyn": 1e-4, "smooth": 1e-2, "jerk": 1e-3, "lim": 1e-2},
        ),
    ]
    for name, alphas in rnn_grid:
        _, tr, te = train_rnn(train_syn, test_syn, K_EMG, 20, alphas, y_max, pose_mean)
        results.append(
            {"name": name, "train_mae_deg": tr, "test_mae_deg": te, "alphas": alphas, "hidden": HIDDEN}
        )

    # --- Raw EMG RNN vs synergy RNN (matched capacity) ---
    _, tr, te = train_rnn(
        train_raw, test_raw, 16, 20, {"rate": 1e-3, "w": 1e-3, "dyn": 1e-4}, y_max, pose_mean
    )
    results.append(
        {
            "name": "rnn_raw_box54",
            "train_mae_deg": tr,
            "test_mae_deg": te,
            "notes": "matched H=32, Box 5.4 regs, raw EMG input",
        }
    )

    # --- Augmentation (within train only) ---
    rng = np.random.default_rng(SEED)
    for aug in (None, "noise", "dropout", "gain", "all"):
        tr_pairs = window_xy(syn_all, pose, valid, "train", augment=aug, rng=rng)
        if aug is not None:
            tr_pairs = train_syn + tr_pairs
        _, tr, te = train_linear(
            tr_pairs,
            test_syn,
            K_EMG,
            20,
            {"smooth": 1e-2, "jerk": 1e-3, "lim": 1e-2},
            y_max,
            pose_mean,
            epochs=100,
        )
        results.append(
            {
                "name": f"linear_aug_{aug or 'none'}",
                "train_mae_deg": tr,
                "test_mae_deg": te,
                "notes": "train-only aug; kin losses on",
            }
        )

    # --- Leakage stress test: NMF on all time + shuffled windows (no temporal gap) ---
    nmf_leak, _, mean_l, std_l = fit_bases(emg_tr, pose_tr, leak=True, emg_all=emg, pose_all=pose)
    syn_leak = transform_emg(emg, nmf_leak, mean_l, std_l)

    def windows_no_gap(x, y, v):
        out = []
        for s in range(0, len(x) - SEQ_LEN + 1, STRIDE):
            sl = slice(s, s + SEQ_LEN)
            if v[sl].mean() < 0.9:
                continue
            out.append((x[sl].astype(np.float32), y[sl].astype(np.float32)))
        return out

    all_w = windows_no_gap(syn_leak, pose, valid)
    rng2 = np.random.default_rng(SEED)
    order = rng2.permutation(len(all_w))
    all_w = [all_w[i] for i in order]
    n_tr = int(0.8 * len(all_w))
    leak_tr, leak_te = all_w[:n_tr], all_w[n_tr:]
    _, tr, te = train_linear(leak_tr, leak_te, K_EMG, 20, {}, y_max, pose_mean, epochs=100)
    results.append(
        {
            "name": "leaky_protocol_nmf_all_plus_shuffle",
            "train_mae_deg": tr,
            "test_mae_deg": te,
            "notes": "DISHONEST: NMF on all time + shuffled overlapping windows (no gap)",
        }
    )
    summary = {
        "recording": rec["name"],
        "stage": rec["stage"],
        "side": "left" if rec["side"] == 0 else "right",
        "T": int(len(emg)),
        "train_end": int(train_end),
        "test_start": int(test_start),
        "device": str(DEVICE),
        "K_EMG": K_EMG,
        "K_POSE": K_POSE,
        "hidden": HIDDEN,
        "results": results,
    }
    return summary


def write_markdown(summary: dict, path: Path) -> None:
    lines = [
        "# One-sample constraint experiment results",
        "",
        f"**Recording:** `{summary['recording']}` (stage {summary['stage']}, {summary['side']})",
        f"**Length:** T={summary['T']} frames @ 50 Hz; train_end={summary['train_end']}, test_start={summary['test_start']} (1 s gap).",
        "",
        "| Run | Test MAE (°) | Train MAE (°) | Notes |",
        "|-----|-------------:|--------------:|-------|",
    ]
    for r in summary["results"]:
        lines.append(
            f"| `{r['name']}` | {r['test_mae_deg']:.2f} | {r.get('train_mae_deg', float('nan')) if r.get('train_mae_deg') is not None else float('nan'):.2f} | {r.get('notes', '')} |"
        )
    lines += [
        "",
        "## Takeaways (auto)",
        "",
        "- Compare `linear_nmf_*` / `rnn_syn_*` against `mean_pose_baseline` and `ridge_*`.",
        "- `leaky_protocol_*` should look artificially better than honest `linear_nmf_mse` — that gap is leakage, not method.",
        "- Prefer synergy RNN over raw only if `rnn_syn_box54` beats `rnn_raw_box54` and ridge.",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recording",
        type=str,
        default="",
        help="HDF5 filename or path; default = first right-hand recording-1",
    )
    args = parser.parse_args()
    if args.recording:
        path = Path(args.recording)
        if not path.exists():
            path = DATA_DIR / args.recording
    else:
        cands = sorted(DATA_DIR.glob("*recording-1_right.hdf5"))
        path = cands[0] if cands else sorted(DATA_DIR.glob("*.hdf5"))[0]
    print("Using", path, "on", DEVICE)
    summary = run_all(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "results.json"
    md_path = OUT_DIR / "results.md"
    json_path.write_text(json.dumps(summary, indent=2))
    write_markdown(summary, md_path)
    print(json.dumps(summary, indent=2))
    print("Wrote", json_path)
    print("Wrote", md_path)


if __name__ == "__main__":
    main()
