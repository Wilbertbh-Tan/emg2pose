# One-sample constraint experiment results (snapshot)

Tracked copy of a run on 2026-11-12. Re-run with:

```bash
python notebooks/experiments/07_one_sample_constraints/run_one_sample_constraints.py
```

**Recording:** `2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-1_right.hdf5` (stage 1, right)  
**Protocol:** 80/20 temporal split + 1 s gap; train-only NMF (`K=4`); windows 2 s @ 50 Hz.

| Run | Test MAE (°) | Train MAE (°) | Notes |
|-----|-------------:|--------------:|-------|
| `mean_pose_baseline` | 26.92 | — | predict train mean pose |
| `ridge_nmf_synergies` | 26.43 | — | honest train-only NMF |
| `ridge_raw_emg` | 26.46 | — | raw 16-ch → pose |
| `linear_nmf_mse` | 26.45 | 20.13 | |
| `linear_nmf_smooth` | 26.46 | 20.14 | |
| `linear_nmf_smooth_jerk` | 26.45 | 20.14 | |
| `linear_nmf_kin_full` | 26.46 | 20.14 | |
| `rnn_syn_mse` | 26.61 | 20.34 | H=32 |
| `rnn_syn_box54` | 26.54 | 20.28 | rate+ℓ₂+simple-dyn |
| `rnn_syn_box54_kin` | 26.62 | 20.57 | Box 5.4 + kin |
| `rnn_raw_box54` | 26.63 | 17.01 | raw EMG, matched H |
| `linear_aug_all` | 26.29 | 20.10 | best honest aug combo |
| `leaky_protocol_nmf_all_plus_shuffle` | **23.50** | 22.02 | **dishonest** — looks better |

## Takeaways

1. On this single recording, **nothing honestly beats ridge/mean by >0.7°**.
2. **RNN does not help** vs NMF ridge under n=1; raw-EMG RNN overfits train (17°) without test gains.
3. **Leakage invents a ~3° “win”** (23.5° vs ~26.4°) — keep gap splits and train-only bases.
4. Contrast with Lazzari notebook `05`, which used **all 30 mini files** and reported ~15.3° — different data budget.
