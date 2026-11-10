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

## Notebooks: 5 core + 1 optional

| # | Notebook | Answers | Proposal | Owner | Needs first |
|---|---|---|---|---|---|
| 01 | [`01_data_prep`](notebooks/experiments/01_data_prep/01_data_prep.ipynb) | Makes clean data for everyone | ✅ | Person A | none |
| 02 | [`02_how_many_synergies`](notebooks/experiments/02_how_many_synergies/02_how_many_synergies.ipynb) | Aim 1a + 1b: k and K_POSE | ✅ | Person B | 01 |
| 03 | [`03_are_synergies_real`](notebooks/experiments/03_are_synergies_real/03_are_synergies_real.ipynb) | Aim 1c stability ✅, realness checks ➕ | ✅ + ➕ | Person C | 01, and k from 02 |
| 04 | [`04_synergies_predict_pose`](notebooks/experiments/04_synergies_predict_pose/04_synergies_predict_pose.ipynb) | Pilot: mini 2 × 2 | 🧪 | Person D | 01, k and K_POSE from 02 |
| 05 | [`05_results_summary`](notebooks/experiments/05_results_summary/05_results_summary.ipynb) | Final figures and answers | ✅ | Everyone | 02–04 |
| 06 | [`06_synergy_dynamics`](notebooks/experiments/06_synergy_dynamics/06_synergy_dynamics.ipynb) *(optional)* | Does EMG drive pose over time? | ➕ | Person A | 01, 04 |

Each notebook lives in its own folder under `notebooks/experiments/`, next to a `log.md` for notes. Notebook 01 is
already built; 02–06 are starter notebooks with `# TODO` gaps. 03 and 04 can start before 02 is
finished by using k = 4 as a placeholder. Exploratory notebooks are in `notebooks/eda/`.

## Tasks for each notebook

### 01 · Data prep (Person A)
*Already built and tested on synthetic files. Person A runs it on Colab and checks the output.*
- [ ] Run it on Colab; confirm it saves `data/processed_mini.npz` (30 recordings, 15 stages, 2 hands)
- [ ] Read each step and explain to the team what the envelope is (filter → rectify → smooth)
- [ ] Check the printed numbers: % of frames with valid pose, train/test sizes
- **Done when:** a teammate's notebook loads the file and prints the same numbers as yours.

### 02 · How many dimensions? Synergies and eigenpostures (Person B) ✅
- [ ] Fit PCA and NMF with k = 1, 2, …, 10
- [ ] For each k, compute **VAF** on the test data (how much EMG the synergies explain)
- [ ] Plot VAF vs k for both methods; choose k where the curve flattens
- [ ] Plot the chosen synergies (bar chart of the 16 channel weights for each)
- [ ] **Gesture classes:** group the training poses into 6–10 hand shapes (k-means on the 20 joint angles) and name each one
- [ ] Heatmap of gestures × average synergy activation: are gestures mixtures of shared synergies, or does each have its own?
- [ ] Predict the gesture from k synergies, k PCA components and all 16 channels (balanced accuracy vs k, with chance = 1/number of classes). Does accuracy level off at the same k as VAF?
- [ ] **Eigenpostures:** PCA on valid training poses; test-pose error (°) for k = 1 … 10; choose K_POSE (start with a 2° cut-off)
- **Done when:** a VAF-vs-k figure and chosen k, a figure of the synergies, and a pose-error-vs-k figure with chosen K_POSE. The gesture tasks are extras.
- **Labels check, never build:** NMF never sees the gesture labels. A method trained with the labels would separate them by design, which proves nothing.
- **Read first:** Lee & Seung 1999 (NMF), Santello et al. 1998 (eigenpostures), Cunningham & Yu 2014 (dimensionality reduction)

### 03 · Are the synergies stable and real? (Person C) ✅ stability, ➕ the rest
Four checks (plus a quick stability check first), each giving one number and one figure:
- [ ] **Stability check:** fit NMF 10 times with different random starts; the synergies should come out the same
- [ ] ➕ **Shuffle test:** shuffle each channel's timing separately (this breaks co-activation) and refit. Real synergies explain much more than shuffled ones
- [ ] ➕ **New-movement test:** fit synergies on 12 tasks, test on the other 3. Real synergies still explain the new tasks
- [ ] ➕ **Neighbour test:** does each synergy spread across the band, or sit on 2–3 neighbouring electrodes (which suggests signal leaking between electrodes)?
- [ ] ➕ **Two-hands test:** fit left and right hands separately and compare synergies (cosine similarity)
- **Done when:** a table of the four checks with pass/fail and the number behind each.
- **Read first:** Tresch, Cheung & d'Avella 2006 (checking extracted synergies)

### 04 · Aim 2 pilot: mini 2 × 2 (Person D) 🧪
- [ ] **Baseline:** always predict the average pose. This is the number to beat
- [ ] Predict pose from **all 16 channels** with ridge regression (the best a simple model can do)
- [ ] Predict pose from the **k synergy activations** c(t) with the same method
- [ ] Compare using angular error (degrees) and R²
- [ ] **Mini 2 × 2:** add the pose bottleneck (predict K_POSE eigenposture scores, rebuild 20 angles) and both bottlenecks
- [ ] *(Extra)* random non-negative basis of size k as a control
- **Done when:** one table and bar chart of baseline + the four 2 × 2 conditions.
- **Read first:** [Hadidi et al. 2026](https://arxiv.org/abs/2603.08212), Salter et al. 2024 (emg2pose)

### 05 · Results summary (everyone)
- [ ] Collect the key figure from each notebook
- [ ] One paragraph for Aim 1 and one for the Aim 2 pilot, including where we were wrong; then what it means for H1

### 06 · Synergy dynamics ➕ *(beyond the proposal, optional, Person A)*
- [ ] Fit a simple linear model where EMG drives the pose, and one without EMG; compare their predictions

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
