# Project outline: T.H.A.N.O.S. on the mini dataset

## The research question (from the proposal)

> **Does constraining both the input (muscle synergies) and the output (eigenpostures) of a decoder with
> low-dimensional structure improve generalisation to held-out users?**

**H1:** if these low-dimensional representations capture structure shared across people, a decoder constrained
on both sides will need **less calibration on a new user** to reach a given joint-angle error.

**What is a muscle synergy?** A group of muscles that work together. We write the EMG as a mix of a few synergies:

```
EMG(t)  ≈  W  ×  c(t)
           │      └─ how strongly each synergy is used at time t
           └─ the synergies: which muscles (EMG channels) work together
```

We find W and c(t) with **NMF** (non-negative matrix factorisation), since muscle activity can't be negative.
**Eigenpostures** are the same idea for the hand: PCA on the 20 joint angles finds a few patterns of joints moving together.

### What the notebooks cover

The mini dataset has **one person**, so it can't test calibration on new users. Here we do **Aim 1** and a
**pilot of Aim 2**. Anything marked ➕ goes beyond the proposal and is optional.

| | Question | Proposal | We are wrong if… |
|---|---|---|---|
| **Aim 1a** | How many sEMG synergies (k)? | ✅ Aim 1 | We need nearly as many synergies as channels (16) |
| **Aim 1b** | How many eigenpostures (K_POSE)? | ✅ Aim 1 | Few components can't rebuild the pose to within ~2° |
| **Aim 1c** | Are the synergies stable across fits? | ✅ Aim 1 | Different random starts give different synergies |
| **Pilot** | Do the bottlenecks keep the pose information (mini 2 × 2)? | 🧪 Aim 2 pilot | Synergies or eigenpostures predict pose much worse than the full signals |
| ➕ | Are the synergies real (shuffle, new movements, neighbours, two hands)? | beyond | They fail these checks |
| ➕ | Does EMG drive the pose over time? | beyond | A model with EMG does no better than one without |

**Not started yet (needs the full dataset or a deep decoder):** reproducing the vEMG2Pose baseline, the full
2 × 2 with random controls (Naveen), calibration curves and REACT (Andres), electrode and sampling-rate
ablations (Youssef). Santi's [master pipeline](notebooks/eda/03_THANOS_master_pipeline.ipynb) sketches all of these.

**What we will *not* claim:** surface EMG alone can't prove synergies come from the brain. We can
show they are consistent, real (not artifacts), and useful. That is our conclusion's wording.

## The data

emg2pose "mini" dataset: **1 participant, 30 recordings** (15 movement tasks × 2 hands), each about
1 minute. Each recording has **16 EMG channels** (a wristband, 2,000 samples per second) and
**20 hand joint angles** (motion capture).

## Notebooks: the investigation ladder

The notebooks follow one ladder (details, gate results and the evaluation of every notebook:
[`notebooks/experiments/EVALUATION.md`](notebooks/experiments/EVALUATION.md)). Each stage asks one question, has a
baseline and a null, and ends in a **gate** that decides whether the next stage is worth running. Every notebook has
the same format: from-scratch `# TODO` tasks with three hints, ✅ checks against a reference module, ✍️ questions,
a methodology section and a gate. Shared evaluation rules: [`common/protocol.py`](notebooks/experiments/common/protocol.py).

**The data has 15 independent blocks**, not 30 recordings: each block is a left and a right recording made at the
same time (in two blocks the idle hand is labelled `unconstrained`).

| # | Stage | Question | Owner |
|---|---|---|---|
| [01](notebooks/experiments/01_data_prep/01_data_prep.ipynb) | 0 Protocol | Build the data; what counts as independent? | Person A |
| [02](notebooks/experiments/02_how_many_synergies/02_how_many_synergies.ipynb) | 1 Existence | How many synergies, beyond chance? How many eigenpostures? | Person B |
| [03](notebooks/experiments/03_are_synergies_real/03_are_synergies_real.ipynb) | 1–2 Artefacts | Stable? Not crosstalk? | Person C |
| [04](notebooks/experiments/04_compositional_structure/04_compositional_structure.ipynb) | 3 Sharing | Do tasks reuse shared structure (Lazzari & Saxena)? | |
| [05](notebooks/experiments/05_link_to_movement/05_link_to_movement.ipynb) | 4 Movement | What does the EMG encode, at what lag? | Person D |
| [06](notebooks/experiments/06_shared_subspace/06_shared_subspace.ipynb) | 4b Shared subspace | Which latents do EMG and pose share? | |
| [07a](notebooks/experiments/07_dynamics/07a_dynamics.ipynb), [07b](notebooks/experiments/07_dynamics/07b_compositional_dynamics.ipynb) | 5 Dynamics | Input-driven? Physical model, resting posture, shared-subspace dynamics; switching motifs and a multitask RNN | Person A |
| [08](notebooks/experiments/08_bottlenecks/08_bottlenecks.ipynb) | 6 Bottlenecks | Do variance-based vs pose-informed bottlenecks keep information and cut calibration? | |
| [10](notebooks/experiments/10_goal_driven_simulation/10_goal_driven_simulation.ipynb) | 8 Simulation | Does a simulated hand + controller reproduce the real EMG structure? Which ingredient causes it? | |
| [09](notebooks/experiments/09_results_summary/09_results_summary.ipynb) | — | Results, gate by gate | Everyone |

Each notebook lives in its own folder under `notebooks/experiments/`, next to a `log.md` and its reference module
`refNN.py`. Notebooks 02–09 build `data/processed_mini.npz` with `prep.py` when it's missing or out of date, so no
notebook waits on another person's TODOs.

## Checklist for each notebook
- [ ] Your prediction written before running
- [ ] Every `# TODO` done and every ✅ check passing
- [ ] Every ✍️ question answered
- [ ] The Result (gate) filled in: number with a 95% CI over blocks, baseline and null, pass or fail
- [ ] The gate copied into notebook 09

## Phases

| Phase | Who | What |
|---|---|---|
| **1. Setup and learning** | Everyone | Get Colab working; run `notebooks/experiments/01_data_prep` together; read 0922 (Cunningham & Yu). then everyone opens their own notebook and reads its first cell |
| **2. Main work** | B, C, D in parallel; A on 06 or helping | Notebooks 02, 03, 04 |
| **3. Combine** | Everyone | Notebook 05, presentation |

## Rules for everyone

1. **Test on data the model didn't see.** Always score on the test part of the split.
2. **Always compare with a simple baseline.** A number alone means nothing.
3. **Use the shared data from notebook 01.** No private preprocessing, or results won't match.
4. **One person per notebook.** Restart and "Run all" before committing; pull before you push.
5. **Write down what you expect before running.** Then compare. Surprises are the interesting part.

## The only five metrics we need

| Metric | Measures | Used in |
|---|---|---|
| **VAF** (variance accounted for) | How much of the EMG the synergies explain (0–100%) | 02, 03 |
| **Cosine similarity** | How alike two synergies are (1 = identical) | 03 |
| **Angular error** (degrees) | How wrong the pose prediction is; same unit as published papers | 02 (pose), 04 |
| **R²** | Fraction of pose variation explained | 04 |
| **Balanced accuracy** | How often the gesture class is right, averaged over classes (chance = 1/number of classes) | 02 (extra) |

---
More detail for later (advanced methods, full metrics, extra hypotheses):
[`notebooks/experiments/advanced_plan/research_plan.md`](notebooks/experiments/advanced_plan/research_plan.md).
