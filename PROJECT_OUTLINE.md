# Project outline: muscle synergies in wrist sEMG

## The research question

> **Can we find muscle synergies in wrist sEMG during hand movements, and are they real and useful?**

**What is a muscle synergy?** A group of muscles that work together. Instead of controlling
every muscle separately, the nervous system may control the hand with a few of these groups.

**The model.** We write the EMG as a mix of a few synergies:

```
EMG(t)  ≈  W  ×  c(t)
           │      └─ how strongly each synergy is used at time t
           └─ the synergies: which muscles (EMG channels) work together
```

We find W and c(t) with **NMF** (non-negative matrix factorisation). Muscle activity can't be
negative, so NMF fits this model naturally.

### Three sub-questions (+ one optional)

| | Question | What we expect | We are wrong if… |
|---|---|---|---|
| **Q1** | **How many** synergies are there? | 3–5 synergies explain at least 90% of the EMG, and gesture accuracy levels off at about the same k | We need nearly as many synergies as channels (16) |
| **Q2** | Are they **real**? | They beat shuffled data, work on movements they weren't fitted on, aren't just neighbouring electrodes, and match between the two hands | They fail these four checks |
| **Q3** | Are they **useful**? | Hand pose predicted from the synergies is nearly as good as from all 16 channels, and clearly better than "always guess the average pose" | Predicting from the synergies is much worse than from all 16 channels |
| Q4 *(optional)* | Are they **driven by commands**? | A model where EMG drives the hand beats a model without EMG | Both models do equally well |

**What we will *not* claim:** surface EMG alone can't prove synergies come from the brain. We can
show they are consistent, real (not artifacts), and useful. That is our conclusion's wording.

## The data

emg2pose "mini" dataset: **1 participant, 30 recordings** (15 movement tasks × 2 hands), each about
1 minute. Each recording has **16 EMG channels** (a wristband, 2,000 samples per second) and
**20 hand joint angles** (motion capture).

## Notebooks: 5 core + 1 optional

| # | Notebook | Answers | Owner | Needs first |
|---|---|---|---|---|
| 01 | [`01_data_prep`](notebooks/01_data_prep/01_data_prep.ipynb) | Makes clean data for everyone | Person A | none |
| 02 | [`02_how_many_synergies`](notebooks/02_how_many_synergies/02_how_many_synergies.ipynb) | Q1 | Person B | 01 |
| 03 | [`03_are_synergies_real`](notebooks/03_are_synergies_real/03_are_synergies_real.ipynb) | Q2 | Person C | 01, and k from 02 |
| 04 | [`04_synergies_predict_pose`](notebooks/04_synergies_predict_pose/04_synergies_predict_pose.ipynb) | Q3 | Person D | 01, and k from 02 |
| 05 | [`05_results_summary`](notebooks/05_results_summary/05_results_summary.ipynb) | Final figures and answers | Everyone | 02–04 |
| 06 | [`06_synergy_dynamics`](notebooks/06_synergy_dynamics/06_synergy_dynamics.ipynb) *(optional)* | Q4 | Person A | 01, 04 |

Each notebook lives in its own folder under `notebooks/`, next to a `log.md` for notes. Notebook 01 is
already built; 02–06 are starter notebooks with `# TODO` gaps. 03 and 04 can start before 02 is
finished by using k = 4 as a placeholder. Older exploratory notebooks are in `notebooks/background/`.

## Tasks for each notebook

### 01 · Data prep (Person A)
*Already built and tested on synthetic files. Person A runs it on Colab and checks the output.*
- [ ] Run it on Colab; confirm it saves `data/processed_mini.npz` (30 recordings, 15 stages, 2 hands)
- [ ] Read each step and explain to the team what the envelope is (filter → rectify → smooth)
- [ ] Check the printed numbers: % of frames with valid pose, train/test sizes
- **Done when:** a teammate's notebook loads the file and prints the same numbers as yours.
- **Read first:** Lecture 2, p. 17 (filtering)

### 02 · How many synergies? (Person B)
- [ ] Fit PCA and NMF with k = 1, 2, …, 10
- [ ] For each k, compute **VAF** on the test data (how much EMG the synergies explain)
- [ ] Plot VAF vs k for both methods; choose k where the curve flattens
- [ ] Plot the chosen synergies (bar chart of the 16 channel weights for each)
- [ ] **Gesture classes:** group the training poses into 6–10 hand shapes (k-means on the 20 joint angles) and name each one
- [ ] Heatmap of gestures × average synergy activation: are gestures mixtures of shared synergies, or does each have its own?
- [ ] Predict the gesture from k synergies, k PCA components and all 16 channels (balanced accuracy vs k, with chance = 1/number of classes). Does accuracy level off at the same k as VAF?
- **Done when:** one VAF-vs-k figure, a chosen k, a figure of the synergies, and a gesture-accuracy-vs-k figure.
- **Labels check, never build:** NMF never sees the gesture labels. A method trained with the labels would separate them by design, which proves nothing.
- **Read first:** Lecture 6 (PCA), Lecture 3 p. 23–27 (NMF), reading 0922 (Cunningham & Yu)

### 03 · Are the synergies real? (Person C)
Four checks (plus a quick stability check first), each giving one number and one figure:
- [ ] **Stability check:** fit NMF 10 times with different random starts; the synergies should come out the same
- [ ] **Shuffle test:** shuffle each channel's timing separately (this breaks co-activation) and refit. Real synergies explain much more than shuffled ones
- [ ] **New-movement test:** fit synergies on 12 tasks, test on the other 3. Real synergies still explain the new tasks
- [ ] **Neighbour test:** does each synergy spread across the band, or sit on 2–3 neighbouring electrodes (which suggests signal leaking between electrodes)?
- [ ] **Two-hands test:** fit left and right hands separately and compare synergies (cosine similarity)
- **Done when:** a table of the four checks with pass/fail and the number behind each.
- **Read first:** reading 0924 (Perich et al.), Lecture 7 (comparing subspaces)

### 04 · Do the synergies predict hand pose? (Person D)
- [ ] **Baseline:** always predict the average pose. This is the number to beat
- [ ] Predict pose from **all 16 channels** with ridge regression (the best a simple model can do)
- [ ] Predict pose from the **k synergy activations** c(t) with the same method
- [ ] Compare using angular error (degrees) and R²
- **Done when:** one bar chart with baseline vs 16 channels vs synergies.
- **Read first:** Lecture 16 p. 13–15 (linear decoders), reading 1027 (Paninski), [Hadidi et al. 2026](https://arxiv.org/abs/2603.08212)

### 05 · Results summary (everyone)
- [ ] Collect the key figure from each notebook
- [ ] Answer Q1, Q2 and Q3 in one paragraph each, including where we were wrong

### 06 · Synergy dynamics *(optional, Person A)*
- [ ] Fit a simple linear model where EMG drives the pose, and one without EMG; compare their predictions
- **Read first:** Lecture 9 (linear dynamical systems)

## Phases

| Phase | Who | What |
|---|---|---|
| **1. Setup and learning** | Everyone | Get Colab working; run `notebooks/01_data_prep` together; read 0922 (Cunningham & Yu). then everyone opens their own notebook and reads its first cell |
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
| **VAF** (variance accounted for) | How much of the EMG the synergies explain (0–100%) | Q1, Q2 |
| **Cosine similarity** | How alike two synergies are (1 = identical) | Q2 |
| **Angular error** (degrees) | How wrong the pose prediction is; same unit as published papers | Q3 |
| **R²** | Fraction of pose variation explained | Q3 |
| **Balanced accuracy** | How often the gesture class is right, averaged over classes (chance = 1/number of classes) | Q1 |

---
More detail for later (advanced methods, full metrics, extra hypotheses):
[`notebooks/background/advanced_plan/research_plan.md`](notebooks/background/advanced_plan/research_plan.md).
