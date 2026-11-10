# Experiment 01: Geometry ("the skatepark")

**Status:** planning · **Notebook:** `notebooks/experiments/advanced_plan/exp01_geometry/geometry.ipynb` (to be written by the user)
**Builds on:** `notebooks/eda/02_eda_features_embeddings.ipynb`, Experiment 1 in `notebooks/eda/03_THANOS_DimRed.ipynb`

## Question
Can muscle synergies be identified in this data (H0), and do they pass the tests a physiological
synergy should pass? Tests **H1** in `../research_plan.md`; its results (W and c(t)) feed H2–H5.

## Hypothesis and what would falsify it
- **H1:** sEMG envelopes are low-dimensional (participation ratio ≈ 2–4 of 16) and captured by a
  linear non-negative model (NMF). The synergies generalise across tasks, beat a time-shuffled null,
  span non-adjacent electrodes, and mirror between hands. Nonlinear methods add little at the same k.
- **Falsified if:** the synergies fail the step 1.4 tests, or nonlinear methods clearly beat NMF at every k.

## Methods
PCA (baseline) · NMF (the synergy model) · synergy tests 1.4a–d · linear vs nonlinear AE ·
UMAP (pictures only) · CEBRA-time (no labels; CEBRA with pose labels only as an upper bound) ·
participation ratio vs TwoNN. Metrics and why: "Metrics" section of `../research_plan.md`.

## Decisions to make first
- [ ] Input: 80 windowed features (notebook 02) or the 16-channel envelope (notebook 03)? NMF needs non-negative input.
- [ ] Split: which recordings are held out, and what does that test (cross-hand? cross-stage?)
- [ ] Values of k to sweep (suggestion: 1, 2, 4, 8)

## What to show
0. Synergy tests 1.4a–d: cross-task VAF ratio, VAF gap vs shuffle null, neighbour concentration, left–right similarity
1. Variance or reconstruction curve vs k, per method
2. Stage-decoding accuracy vs k, per method, recordings held out (`GroupKFold`)
3. Pose-decoding R² vs k, per method (continuous, closer to the real task)
4. Hand-side accuracy (nuisance check)
5. 2-D embeddings for illustration only

## Done when
All three curves exist for all four methods with 3 seeds, and the brief's question has a one-paragraph answer.

## Concepts to learn
Linear vs nonlinear manifolds · participation ratio · NMF non-negativity and why it suits muscle synergies ·
contrastive learning (CEBRA) · temporal leakage and grouped cross-validation
