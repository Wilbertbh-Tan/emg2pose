"""Shared evaluation protocol (stage 0 of the investigation ladder).

Every number in notebooks 02-08 should be computed with these helpers, so that all notebooks agree on
what counts as independent data, what the baselines are and how uncertainty is reported.

Key facts about the mini dataset that this module encodes:
- It is ONE participant. 30 recordings = 15 stages x 2 hands.
- Recordings come in 15 simultaneous pairs (left + right wrist recorded AT THE SAME TIME; time stamps overlap
  100%). The pair is identified by the recording number in the file name ("...recording-13_left/right"). These
  pairs are not independent, so the unit of replication is the **block** = one recording period, both hands.
- Usually both hands of a block have the same stage, but not always: in blocks 14 and 15 one hand did
  "OneHandedFreeStyle" while the other (idle) hand's recording is labelled "unconstrained". So pairing hands by
  stage name is WRONG; always pass ``filename`` so blocks come from the recording number.
- Within a recording, frames 20 ms apart are strongly correlated, so frames are never the unit of replication.

Run ``python protocol.py`` to execute the self-tests.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterator

import numpy as np

FS: int = 50  # frames per second in processed_mini.npz


# ── Blocks (independent units) ───────────────────────────────────────────────
def block_of_rec(filename: np.ndarray) -> np.ndarray:
    """(R,) file names -> (R,) int block id (0 … 14): the two simultaneous recordings share an id."""
    nums = np.array([int(re.search(r"recording-(\d+)_", str(f)).group(1)) for f in filename])
    _, bid = np.unique(nums, return_inverse=True)
    return bid.astype(np.int64)


def session_of_rec(stage_of_rec: np.ndarray, filename: np.ndarray | None = None) -> np.ndarray:
    """(R,) independent-unit id per recording. With ``filename`` (recommended): the simultaneous block.
    Without it: one id per stage name (wrong for the OneHandedFreeStyle / unconstrained blocks; kept only so
    old code runs)."""
    if filename is not None:
        return block_of_rec(filename)
    warnings.warn("session_of_rec without filename pairs hands by stage name, which is wrong for 2 blocks; "
                  "pass filename=d['filename']", stacklevel=2)
    _, sid = np.unique(stage_of_rec, return_inverse=True)
    return sid.astype(np.int64)


def session_ids(stage_of_rec: np.ndarray, rec: np.ndarray, filename: np.ndarray | None = None) -> np.ndarray:
    """(N,) independent-unit id of every frame (block id when ``filename`` is given)."""
    return session_of_rec(stage_of_rec, filename)[rec]


# ── Folds ────────────────────────────────────────────────────────────────────
def leave_one_block_out(filename: np.ndarray, rec: np.ndarray, within: np.ndarray | None = None
                        ) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (block_name, train_mask, test_mask): one fold per simultaneous block (15 folds)."""
    blk = block_of_rec(filename)[rec]
    ok = np.ones(len(rec), bool) if within is None else within
    for b in np.unique(blk):
        yield f"block {b}", (blk != b) & ok, (blk == b) & ok


def stage_groups(stage_of_rec: np.ndarray, filename: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Group recordings so that holding out a stage also holds out everything recorded at the same time.

    Returns [(name, recording_ids)]: stages linked through a shared block are merged (the two free-style blocks
    give one group "OneHandedFreeStyle + unconstrained"), so there are 14 groups, not 15.
    """
    blk = block_of_rec(filename)
    stages = list(np.unique(stage_of_rec))
    parent = {s: s for s in stages}

    def find(x: str) -> str:
        while parent[x] != x:
            x = parent[x]
        return x

    for b in np.unique(blk):
        a, *rest = [str(s) for s in stage_of_rec[blk == b]]
        for o in rest:
            parent[find(o)] = find(a)
    groups: dict[str, list[str]] = {}
    for s in stages:
        groups.setdefault(find(str(s)), []).append(str(s))
    out = []
    for members in groups.values():
        recs = np.where(np.isin(stage_of_rec, members))[0]
        recs = np.where(np.isin(blk, blk[recs]))[0]
        out.append((" + ".join(sorted(members)), recs))
    return sorted(out)


def leave_one_stage_out(stage_of_rec: np.ndarray, rec: np.ndarray, within: np.ndarray | None = None,
                        filename: np.ndarray | None = None) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (held_out_stage, train_mask, test_mask) over frames.

    With ``filename`` (recommended): one fold per stage group from ``stage_groups`` (14 folds); a fold's test set
    is every recording of that stage plus everything recorded at the same time, so no simultaneous data leaks
    into train. Without ``filename``: one fold per stage name (15 folds, both hands by name; leaks for 2 blocks).
    ``within`` (N,) bool, e.g. ``pose_valid``, is AND-ed into both masks.
    """
    ok = np.ones(len(rec), bool) if within is None else within
    if filename is None:
        warnings.warn("leave_one_stage_out without filename can leak simultaneous recordings; "
                      "pass filename=d['filename']", stacklevel=2)
        for name in np.unique(stage_of_rec):
            te = np.isin(rec, np.where(stage_of_rec == name)[0])
            yield str(name), (~te) & ok, te & ok
        return
    for name, recs in stage_groups(stage_of_rec, filename):
        te = np.isin(rec, recs)
        yield name, (~te) & ok, te & ok


# ── Baselines ────────────────────────────────────────────────────────────────
def global_median_baseline(Y: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """(N, D) prediction: the median of Y over fit_mask frames, repeated for every frame."""
    return np.tile(np.median(Y[fit_mask], axis=0), (len(Y), 1))


def per_recording_median_baseline(Y: np.ndarray, rec: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """(N, D) prediction: each recording's own median over its fit_mask frames.

    Only meaningful for the within-recording time split (train = start of each recording). In a
    leave-one-stage-out fold the held-out recordings have no training frames; their rows fall back to the
    global median of fit_mask.
    """
    out = global_median_baseline(Y, fit_mask)
    for r in np.unique(rec):
        m = (rec == r) & fit_mask
        if m.any():
            out[rec == r] = np.median(Y[m], axis=0)
    return out


# ── Nulls ────────────────────────────────────────────────────────────────────
def circular_shift_null(X: np.ndarray, rec: np.ndarray, rng: np.random.Generator,
                        min_shift_s: float = 5.0, fs: int = FS) -> np.ndarray:
    """Shift X in time, circularly WITHIN each recording, by a random amount of at least min_shift_s.

    Each signal keeps its own structure (spectrum, autocorrelation), but its alignment with the other signal
    (e.g. EMG vs pose) is broken. A decoder trained and tested on shifted EMG should do no better than baseline.
    The whole recording is shifted by one amount, so co-activation ACROSS channels is preserved.
    """
    out = X.copy()
    lo = int(min_shift_s * fs)
    for r in np.unique(rec):
        idx = np.where(rec == r)[0]
        n = len(idx)
        if n <= 2 * lo:
            shift = n // 2
        else:
            shift = int(rng.integers(lo, n - lo))
        out[idx] = np.roll(X[idx], shift, axis=0)
    return out


# ── Uncertainty ──────────────────────────────────────────────────────────────
def session_bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
                         ) -> tuple[float, float, float]:
    """Mean and (1 - alpha) percentile bootstrap CI of per-session values (resampling sessions).

    values : (S,) one number per session (e.g. 15 fold errors). Returns (mean, low, high).
    For a paired comparison of two methods, pass the per-session DIFFERENCE.
    """
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


# ── Calibration curves (within-person proxy for "less calibration on a new user") ─
def calibration_splits(test_mask: np.ndarray, rec: np.ndarray, seconds: list[float], fs: int = FS
                       ) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Split a held-out stage into a calibration pool (first half of each recording) and an evaluation set
    (second half). Return, for each budget in ``seconds``, (seconds, calibration_mask, eval_mask), where the
    calibration mask takes the first ``seconds`` of the pool of each recording. eval_mask is the same for all
    budgets, so the curve compares like with like.
    """
    pool = np.zeros_like(test_mask)
    eval_mask = np.zeros_like(test_mask)
    for r in np.unique(rec[test_mask]):
        idx = np.where((rec == r) & test_mask)[0]
        half = len(idx) // 2
        pool[idx[:half]] = True
        eval_mask[idx[half:]] = True
    out = []
    for s in seconds:
        cal = np.zeros_like(test_mask)
        for r in np.unique(rec[pool]):
            idx = np.where((rec == r) & pool)[0]
            cal[idx[: int(s * fs)]] = True
        out.append((float(s), cal, eval_mask))
    return out


def cross_validate(fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray], Y: np.ndarray,
                   stage_of_rec: np.ndarray, rec: np.ndarray, within: np.ndarray | None,
                   score: Callable[[np.ndarray, np.ndarray], float], filename: np.ndarray | None = None
                   ) -> dict[str, float]:
    """Run leave-one-stage-out: fit_predict(train_mask, test_mask) -> predictions for the test frames.

    Returns {held_out_stage: score(Y[test], prediction)}; feed the values to session_bootstrap_ci.
    """
    out: dict[str, float] = {}
    for name, tr, te in leave_one_stage_out(stage_of_rec, rec, within, filename):
        if te.sum() == 0:
            out[name] = float("nan")
            continue
        out[name] = float(score(Y[te], fit_predict(tr, te)))
    return out


# ── Self-tests ───────────────────────────────────────────────────────────────
def _self_test() -> None:
    rng = np.random.default_rng(0)
    stage_of_rec = np.repeat([f"s{i}" for i in range(15)], 2)          # 30 recordings, hands paired
    stage_of_rec[[26, 29]] = "free"; stage_of_rec[[27, 28]] = "idle"     # blocks 13/14 mix two stages, like the data
    filename = np.array([f"x-recording-{i // 2 + 1}_{'left' if i % 2 == 0 else 'right'}" for i in range(30)])
    rec = np.repeat(np.arange(30), 200)                               # 200 frames each
    # 1. folds never split a block, and cover every frame exactly once as test
    seen = np.zeros(len(rec), int)
    blk = block_of_rec(filename)[rec]
    for _, tr, te in leave_one_stage_out(stage_of_rec, rec, filename=filename):
        assert not (tr & te).any()
        assert not np.isin(blk[tr], blk[te]).any(), "a block is split between train and test"
        seen += te
    assert (seen == 1).all()
    assert len(stage_groups(stage_of_rec, filename)) == 14
    assert sum(1 for _ in leave_one_block_out(filename, rec)) == 15
    # 2. a shifted null destroys a perfect EMG -> pose relation
    x = rng.normal(size=(len(rec), 1)).cumsum(0) * 0.1
    y = 3 * x
    xs = circular_shift_null(x, rec, rng, min_shift_s=1.0)
    r_true = np.corrcoef(x[:, 0], y[:, 0])[0, 1]
    r_null = np.corrcoef(xs[:, 0], y[:, 0])[0, 1]
    assert r_true > 0.99 and abs(r_null) < r_true
    # 3. the bootstrap CI covers the true mean of a known distribution most of the time
    cover = 0
    for s in range(200):
        v = np.random.default_rng(s).normal(1.0, 1.0, size=15)
        _, lo, hi = session_bootstrap_ci(v, n_boot=500, seed=s)
        cover += lo <= 1.0 <= hi
    assert cover / 200 > 0.85, f"coverage {cover / 200:.2f}"
    # 4. calibration budgets are nested and never touch the evaluation half
    te = rec < 2
    sp = calibration_splits(te, rec, [0, 1, 2], fs=50)
    assert sp[0][1].sum() == 0 and sp[1][1].sum() == 100 and sp[2][1].sum() == 200
    assert not (sp[2][1] & sp[2][2]).any()
    # 5. per-recording baseline uses each recording's own median
    Y = np.repeat(np.arange(30.0), 200)[:, None]
    b = per_recording_median_baseline(Y, rec, np.ones(len(rec), bool))
    assert np.allclose(b, Y)
    print("protocol self-tests passed")


if __name__ == "__main__":
    _self_test()
