# 05 Lazzari compositional RNN — log

## Expected
- Mini EMG envelopes + stage one-hot train a leaky RNN to beat mean-pose baseline on held-out time.
- Principal angles across stages smaller than random bases (shared manifold signal).

## Ran (cloud, 2026-11-12)
- Loaded all 30 mini HDF5 recordings → 1632 train / 364 test windows (50 Hz, seq=100).
- Softplus leaky RNN (64 hidden): test MAE **15.29°** vs mean-pose **17.36°** after 8 epochs (CPU).
- Mean principal angle across stage pairs **0.312 rad** vs random **1.154 rad**.
- Transfer sketch (freeze \(W_h\), adapt input/readout on held-out stage 9): **15.70° → 12.91°**.
- Stubbed: MotorNet closed loop, composite rules, sequencing, fixed points, DSA.

## Notes
- `Wilbert_EDA/` copy is gitignored; tracked path is this folder.
- Methodology: Project store `docs/paper-methodology-emg-adaptation.md`.
