"""Generate the EDA figures for the emg2pose manifold-learning proposal."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import BANDS, SAMPLE_RATE, build_matrix, read_session  # noqa: E402

DATA_DIR = Path.home() / "emg2pose_dataset_mini"
FIG_DIR = Path(__file__).parent / "figures"
CACHE = Path(__file__).parent / ".cache"
N_EMBED_SAMPLE = 6000
RANDOM_SEED = 0

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def fig_raw_signal() -> None:
    """Raw sEMG and the paired joint angles on a shared time axis."""
    path = sorted(DATA_DIR.glob("*.hdf5"))[0]
    emg, angles, meta = read_session(path)

    t0, t1 = int(10 * SAMPLE_RATE), int(14 * SAMPLE_RATE)
    t = np.arange(t0, t1) / SAMPLE_RATE

    fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)

    offset = 0.0
    step = float(np.abs(emg[t0:t1]).max()) * 2.2
    for ch in range(4):
        axes[0].plot(t, emg[t0:t1, ch] + offset, lw=0.4)
        axes[0].text(t[0] - 0.12, offset, f"ch{ch}", va="center", ha="right", fontsize=7)
        offset += step
    axes[0].set_ylabel("sEMG (a.u.)")
    axes[0].set_title(
        f"Raw sEMG and paired hand pose — user {meta.user[:6]}…, "
        f"stage {meta.stage}, {meta.side} hand, {meta.sample_rate:.0f} Hz",
        fontsize=10,
    )
    axes[0].set_yticks([])

    win = int(0.05 * SAMPLE_RATE)
    kernel = np.ones(win) / win
    for ch in range(4):
        env = np.sqrt(np.convolve(emg[t0:t1, ch].astype(np.float64) ** 2, kernel, "same"))
        axes[1].plot(t, env, lw=0.9, label=f"ch{ch}")
    axes[1].set_ylabel("RMS envelope")
    axes[1].legend(ncol=4, fontsize=7, loc="upper right")

    for j in range(0, 20, 4):
        axes[2].plot(t, angles[t0:t1, j], lw=0.9, label=f"joint {j}")
    axes[2].set_ylabel("Joint angle (rad)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(ncol=5, fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_raw_signal.png", bbox_inches="tight")
    plt.close(fig)


def fig_spectra() -> None:
    """Per-channel power spectra and per-channel amplitude spread across sessions."""
    paths = sorted(DATA_DIR.glob("*.hdf5"))
    emg, _, meta = read_session(paths[0])

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    for ch in range(16):
        freqs, psd = signal.welch(emg[:, ch], fs=SAMPLE_RATE, nperseg=2048)
        axes[0].semilogy(freqs, psd, lw=0.7, alpha=0.75)
    for lo, hi in BANDS:
        axes[0].axvspan(lo, hi, alpha=0.06, color="k")
    axes[0].set_xlim(0, 500)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD (a.u.²/Hz)")
    axes[0].set_title("Power spectrum, all 16 channels\n(shaded = feature bands)", fontsize=9)

    rms = np.zeros((len(paths), 16))
    for i, p in enumerate(paths):
        e, _, _ = read_session(p)
        rms[i] = np.sqrt(np.mean(e.astype(np.float64) ** 2, axis=0))

    axes[1].boxplot([rms[:, c] for c in range(16)], widths=0.6,
                    flierprops={"markersize": 2})
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Channel")
    axes[1].set_ylabel("RMS amplitude (a.u.)")
    axes[1].set_title(
        f"Per-channel RMS across {len(paths)} sessions\n(one user, 15 stages, both hands)",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_spectra_channels.png", bbox_inches="tight")
    plt.close(fig)


def fig_pca(Xs: np.ndarray) -> PCA:
    """Scree plot and cumulative variance — the dimensionality result."""
    pca = PCA().fit(Xs)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    lam = pca.explained_variance_
    pr = float(lam.sum() ** 2 / (lam**2).sum())

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    axes[0].bar(np.arange(1, 21), ev[:20] * 100, color="#3b6ea5")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Variance explained (%)")
    axes[0].set_title(f"Scree — PC1 alone is {ev[0]*100:.0f}%", fontsize=9)

    axes[1].plot(np.arange(1, len(cum) + 1), cum * 100, "o-", ms=3, color="#3b6ea5")
    for thresh, colour in [(0.80, "#c44"), (0.90, "#e88"), (0.95, "#aaa")]:
        n = int(np.searchsorted(cum, thresh) + 1)
        axes[1].axhline(thresh * 100, ls="--", lw=0.8, color=colour)
        axes[1].annotate(f"{int(thresh*100)}% → {n} PCs", (n, thresh * 100),
                         textcoords="offset points", xytext=(8, -11), fontsize=7.5,
                         color=colour)
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title(
        f"80 features → participation ratio {pr:.1f}", fontsize=9
    )
    axes[1].set_xlim(0, 40)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_pca_dimensionality.png", bbox_inches="tight")
    plt.close(fig)
    return pca


def fig_embeddings(pcs: np.ndarray, U: np.ndarray, P: np.ndarray,
                   stage: np.ndarray, side: np.ndarray, accs: dict[str, float],
                   side_accs: dict[str, float]) -> None:
    """PCA / UMAP / PHATE side by side, coloured by stage and by hand side."""
    stages = sorted(set(stage))
    cmap = plt.get_cmap("tab20")
    colours = {s: cmap(i % 20) for i, s in enumerate(stages)}

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.4))
    panels = [("PCA", pcs), ("UMAP", U), ("PHATE", P)]

    for col, (name, emb) in enumerate(panels):
        ax = axes[0, col]
        for s in stages:
            m = stage == s
            ax.scatter(emb[m, 0], emb[m, 1], s=1.6, color=colours[s], alpha=0.55,
                       linewidths=0, rasterized=True)
        ax.set_title(f"{name} — coloured by stage\nstage kNN accuracy "
                     f"{accs[name]*100:.0f}% (chance {100/len(stages):.0f}%)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[1, col]
        for s, c in [("left", "#2b7bba"), ("right", "#d1622b")]:
            m = side == s
            ax.scatter(emb[m, 0], emb[m, 1], s=1.6, color=c, alpha=0.5,
                       linewidths=0, label=s, rasterized=True)
        ax.set_title(f"{name} — coloured by hand\nhand kNN accuracy "
                     f"{side_accs[name]*100:.0f}% (chance 50%)", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.legend(markerscale=6, fontsize=8, loc="best")

    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=colours[s], label=s)
               for s in stages]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        "Windowed sEMG features in 2D — task structure dominates, but hand side leaks in too",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG_DIR / "04_embeddings.png", bbox_inches="tight")
    plt.close(fig)


def compute_embeddings(Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (or load cached) UMAP and PHATE embeddings on a fixed subsample."""
    CACHE.mkdir(exist_ok=True)
    paths = {k: CACHE / f"{k}.npy" for k in ("idx", "umap", "phate")}
    if all(p.exists() for p in paths.values()):
        return tuple(np.load(p) for p in paths.values())  # type: ignore[return-value]

    rng = np.random.default_rng(RANDOM_SEED)
    n = min(N_EMBED_SAMPLE, Xs.shape[0])
    idx = np.sort(rng.choice(Xs.shape[0], n, replace=False))
    subset = Xs[idx]

    import umap  # imported lazily; heavy and only needed here
    import phate

    print(f"  computing UMAP on {n} windows…")
    U = umap.UMAP(n_neighbors=30, min_dist=0.1,
                  random_state=RANDOM_SEED).fit_transform(subset)
    print(f"  computing PHATE on {n} windows…")
    P = phate.PHATE(n_jobs=-1, random_state=RANDOM_SEED,
                    verbose=0).fit_transform(subset)

    np.save(paths["idx"], idx)
    np.save(paths["umap"], U)
    np.save(paths["phate"], P)
    return idx, U, P


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)

    print("building features…")
    X, _, metas, sess = build_matrix(DATA_DIR)
    Xs = StandardScaler().fit_transform(X)
    stage_all = np.array([metas[i].stage for i in sess])
    side_all = np.array([metas[i].side for i in sess])

    print("figure 1: raw signal")
    fig_raw_signal()
    print("figure 2: spectra")
    fig_spectra()
    print("figure 3: PCA")
    pca = fig_pca(Xs)

    idx, U, P = compute_embeddings(Xs)
    pcs = pca.transform(Xs)[idx][:, :2]

    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    accs, side_accs = {}, {}
    for name, emb in [("PCA", pcs), ("UMAP", U), ("PHATE", P)]:
        accs[name] = float(
            cross_val_score(KNeighborsClassifier(15), emb, stage_all[idx], cv=5).mean()
        )
        side_accs[name] = float(
            cross_val_score(KNeighborsClassifier(15), emb, side_all[idx], cv=5).mean()
        )

    print("figure 4: embeddings")
    fig_embeddings(pcs, U, P, stage_all[idx], side_all[idx], accs, side_accs)
    print("done →", FIG_DIR)


if __name__ == "__main__":
    main()
