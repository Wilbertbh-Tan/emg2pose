# Research plan

**Overall question (H0):** can physiologically plausible **muscle synergies** be identified from
wrist sEMG during hand movement? A synergy is a fixed pattern of co-activated muscles; the classic
model is EMG(t) ≈ W·c(t), with synergies W and recruitment c(t), both non-negative (fit by NMF).

The manifold and dynamics questions are how we test H0: *where* the state sits (geometry of the
synergy space), *how* it moves, and whether that motion is driven by input (descending commands)
or self-sustaining.

**Limit to keep in mind:** surface EMG alone cannot prove synergies are neural (Kutch &
Valero-Cuevas 2012; the hand is their key example). This project can establish the *necessary
conditions*: synergies that are consistent, generalise across tasks, beat a null model, aren't
electrode crosstalk, and matter for movement. Conclusions are worded accordingly.

**Data:** emg2pose mini split: 1 participant, 30 recordings, 15 stages, both hands, 16-channel sEMG at
2 kHz with 20 joint angles. Metadata for the full dataset (193 users) is in `emg2pose_metadata.csv`.
Cross-user claims need part of the full dataset (stage 5).

**Rule for every step:** start with the simplest model, compare it with a baseline, and move to a more
complex model only when the simple one is understood. Each step follows the same loop:
predict → code → check → review → run → compare → log.

## Hypotheses

Each is written so a result can prove it wrong. H1–H5 each test one property that physiological
synergies should have. The overall expectation, by analogy with the olfactory work that started
this project: **input-driven trajectories on a low-dimensional synergy manifold**.

| | If synergies are physiological, then… | Tested by |
|---|---|---|
| H1 | EMG lives in a few dimensions, captured by a *linear non-negative* model, and the synergies pass the four synergy tests (step 1.4) | NMF vs PCA vs AE; cross-task, null, crosstalk, left/right tests |
| H2 | Recruitment c(t) explains movement, and EMG synergies are not just copies of postural (kinematic) synergies | Decoding pose from c(t); EMG vs kinematic synergy similarity |
| H3 | Tasks recruit *shared* synergies in different combinations, rather than private muscles per task | Where tasks sit in synergy space; repeat stereotypy |
| H4 | c(t) is driven by descending commands (input), not by its own momentum | EMG-driven vs autonomous models in free rollouts |
| H5 | Similar synergies appear in different people | Synergy similarity across users |

| | Hypothesis | Falsified if | Grounding |
|---|---|---|---|
| **H1** Geometry | sEMG envelopes use only a few dimensions (participation ratio ≈ 2–4 of 16), and tasks are separable at k ≤ 4. The synergies generalise across tasks, beat a time-shuffled null, span non-adjacent electrodes, and mirror between hands. Nonlinear methods add little over NMF at the same k. | Synergies fail the tests in step 1.4; or nonlinear methods clearly beat NMF at every k (the linear synergy model is then incomplete) | EDA participation ratio ≈ 2.0 (80 features); UMAP 40% vs PCA 24% stage accuracy; Perich 2025 (linear methods overestimate dimension on curved manifolds); Tresch 2006; Kutch & Valero-Cuevas 2012 |
| **H2** Multimodal | Instantaneous EMG predicts joint *velocity* better than joint *angle*. EMG history helps angle more than velocity. The best lag is ~30–100 ms. Pose → EMG (encoding) is worse than EMG → pose, because one pose can be held with different co-contraction. | Angle is decoded as well as velocity without history; or encoding matches decoding | EMG reflects activation/force; electromechanical delay literature; Osu 2004 (co-contraction). *Caveat:* Hadidi 2026 found deep models decoding position beat velocity integration, a different question (accumulated error) |
| **H3** Trajectories | Task identity is carried more by *where* the state is than by *how it moves*. Repeated movements within a task follow similar paths. | Stage decoding from latent velocity ≥ from latent position | Different tasks recruit different muscle combinations; Vyas 2020 (rotational dynamics are not seen in muscles) |
| **H4** Dynamics | Hand movement is input-driven: a model with EMG input beats an autonomous model by a wide margin in free rollouts. Its time constants are short (~50–200 ms). With the input switched off, the state decays to rest instead of persisting. | The autonomous model matches the driven one, or zero-input rollouts persist | Khona & Fiete 2022 (criteria for self-sustained dynamics); Durstewitz 2023 (judge free rollouts, not one-step error) |
| **H5** Generalisation | Recordings cluster more by participant than by task, so cross-user decoding needs alignment. | Stage structure dominates participant structure on multi-user data | EDA hand-side leak (UMAP 67% vs 50% chance) as a within-person proxy |

## How the hypotheses link

```
                       H0  Physiological muscle synergies can be identified
                                            │
                                            ▼
                 H1  Synergies exist: few dimensions, linear non-negative fit,
                     pass the four tests (cross-task, null, crosstalk, left/right)
                     ── produces W (the synergies) and c(t) (their recruitment) ──
                 ┌──────────────┬───────────┴───────────┬──────────────┐
                 ▼              ▼                       ▼              ▼
   H2  c(t) explains     H3  tasks reuse         H4  c(t) is driven    H5  same synergies
       movement, and         shared synergies        by input, not         in other people
       isn't just            in different            self-sustaining
       posture               combinations
                 │              │                       │
                 └──────────────┴───────► H4 uses ◄─────┘
                                 H2's pose model (pose = state, EMG = input)
                                 H3's description of how c(t) moves
```

**H1 is the gate.** H2–H5 all use the W and c(t) that H1 produces, so H1 is tested first. The
others then each add one independent line of evidence:

| Hypothesis | What it adds to H0 | Depends on |
|---|---|---|
| H1 | Synergies *exist* in this data and aren't artifacts | none |
| H2 | They are *functional*: they explain movement, beyond what posture alone gives | H1 (uses c(t)) |
| H3 | They are *shared modules*: tasks combine them rather than using private muscles | H1 (uses the synergy space) |
| H4 | They are *commanded*: recruitment follows descending input | H1 (c(t)); H2 (the EMG→pose relation and lag); H3 (how c(t) moves) |
| H5 | They are *general*: not specific to one person | H1, repeated per user |

**What a failure means for the rest:**

| If this fails… | …then | And the project continues as |
|---|---|---|
| H1, null test (1.4b) | Low-dimensional structure is no better than chance co-activation | Stop calling them synergies. H2–H4 are still valid questions about EMG, just not about synergies |
| H1, crosstalk (1.4c) | "Synergies" are mostly electrode overlap | Same as above; also try removing crosstalk (e.g. differential channel pairs) before refitting |
| H1, nonlinear wins clearly | The *linear* synergy model is incomplete | Synergies may still exist but combine nonlinearly. Use the AE latents as c(t) in H2–H4 |
| H2, EMG synergies ≈ postural synergies | They may reflect biomechanics rather than neural modules (Kutch & Valero-Cuevas) | Report this. It is a real finding, not a dead end |
| H3, each task has private patterns | Synergies are task-specific, not shared modules | Weakens H0. H4 can still test whether recruitment is input-driven |
| H4, autonomous model matches | Recruitment carries its own dynamics (e.g. rhythmic tasks) | Look for which stages are self-sustaining; rSLDS (step 4.4) tests this per region |

**Order of evidence:** H0 is supported only to the degree that H1 passes *and* H2–H5 add support.
The strongest claim this data can make is "consistent, functional, shared, input-driven synergies
in one person" (H1–H4). H5 needs more users.

## Metrics: how each comparison is made, and why

**Rules for every number:**
1. **Held-out data.** Scored on recordings or time blocks the model never saw. Say which.
2. **A reference.** Next to a baseline, a null or a ceiling; a number alone means nothing.
3. **Spread.** Mean ± SD over folds or seeds (≥ 3). Compare methods *paired* on the same folds.
4. **Matched size.** Methods are compared at the same k (number of synergies or latent dimensions).
5. **Honest n.** One participant: the unit of replication is the recording or stage, and every claim
   is within-participant until stage 5.

### Reconstruction and dimensionality (H1)
| Metric | Definition | Why this one |
|---|---|---|
| **VAF** (variance accounted for) | 1 − ‖X − X̂‖² / ‖X‖² (uncentered), on held-out data | The synergy literature's standard; NMF has no mean, so uncentered fits it |
| **Centered R²** | 1 − ‖X − X̂‖² / ‖X − X̄‖² | Needed to compare NMF with PCA fairly. PCA's "variance explained" is centered, NMF's VAF is not, so **compute both for every method** |
| **Per-channel VAF** | VAF for each of the 16 channels; report the minimum | Stops one loud channel from hiding a badly fit one |
| **Number of synergies** | Smallest k where held-out VAF plateaus *and* clearly beats the shuffle null | Replaces the arbitrary "90% VAF" threshold with a data-driven one |
| **Participation ratio** | (Σλ)² / Σλ² of PCA eigenvalues | Single-number linear dimensionality |
| **TwoNN** | Intrinsic dimension from nearest-neighbour distance ratios | Nonlinear estimate. If it is much lower than the participation ratio, the manifold is curved. Downsample first: 2 kHz neighbours are near-duplicates |

### Synergy tests (H1, step 1.4)
| Metric | Definition | Why this one |
|---|---|---|
| **Matched cosine similarity** | Pair synergies between two sets with the Hungarian algorithm, then cos(wᵢ, wⱼ) | Synergies are directions with arbitrary order and scale, so they must be matched before comparing |
| **Principal angles** | Angles between the two synergy subspaces (Lecture07) | Compares the whole set without one-to-one matching |
| **Similarity null** | The same similarity for random non-negative vectors, or for synergies from shuffled data | A cosine of 0.8 means little unless random vectors give 0.5 |
| **Cross-task VAF ratio** | VAF on a held-out stage with W fixed (only c(t) refit, by non-negative least squares), ÷ VAF of synergies fit on that stage | Near 1 means the synergies are shared across tasks (1.4a) |
| **VAF gap vs null** | VAF(real) − VAF(each channel time-shuffled), for each k | Shuffling destroys co-activation but keeps each channel's own statistics (1.4b) |
| **Neighbour concentration** | Share of a synergy's weight on its peak electrode and the two adjacent ones | High means "a few neighbouring electrodes", which is suspicious for crosstalk (1.4c). Assumes channels are ordered around the band, which the benchmark's rotation augmentation implies. Verify |
| **Left–right similarity** | Matched cosine between hands, after mirroring the channel order | Physiological synergies should mirror between hands (1.4d) |

### Separability (H1, H3)
| Metric | Definition | Why this one |
|---|---|---|
| **Balanced accuracy** (kNN and LDA) | Stage decoding with grouped cross-validation; chance = 1/15 | Balanced because stages differ in length. Grouped because overlapping windows leak |
| **Silhouette** | Cluster separation on held-out data | Secondary; independent of any classifier |

### Decoding, encoding and shared space (H2)
| Metric | Definition | Why this one |
|---|---|---|
| **R²** per joint, averaged | On valid pose frames only | Fraction of pose variance explained |
| **Mean angular error (°)** | Mean absolute joint-angle error | The emg2pose and Hadidi 2026 unit, so results compare with published numbers |
| **Pearson r** | Correlation per joint, averaged | Trajectory shape, ignoring offset |
| **Static baseline** | The same metrics for "always the median pose" | Hadidi 2026: this gets within ~4° of deep models, so it is the number to beat |
| **Lag curve** | Held-out R² vs EMG–pose shift (0–150 ms) | Its peak estimates the electromechanical delay |
| **Velocity scored two ways** | R² on velocity, and angle error after integrating | Integration lets velocity and angle decoders compete on the same footing |
| **EMG vs postural synergies** | Similarity between EMG synergy recruitment and kinematic PCA components | High overlap would suggest synergies reflect biomechanics, not neural modules |
| **Canonical correlations** | CCA on held-out blocks, against a large time-shift null | How many shared EMG–pose dimensions exist. Circular shifts can still give false positives (Harris 2020) |

### Dynamics (H4)
| Metric | Definition | Why this one |
|---|---|---|
| **k-step rollout error** | MSE after 1, 5 and 25 free-running steps, ÷ persistence ("no change") MSE | One-step error looks good on any smooth signal; free rollouts don't (Durstewitz 2023) |
| **Time constants** | τ = −Δt / ln\|λ\| from the eigenvalues of A | Converts the fitted model into milliseconds you can compare with muscle physiology |
| **Input ablation** | Rollout error with B = 0 vs with B | The direct input-driven vs self-sustaining test |
| **Decay after input off** | Time for the state to return near rest | Self-sustained dynamics would persist (Khona & Fiete 2022) |

### Generalisation (H5)
| Metric | Definition | Why this one |
|---|---|---|
| **Cross-user synergy similarity** | Matched cosine between users' synergies, vs within-user | Physiological synergies should be shared across people |
| **Cross-user decoding drop** | Within-user minus cross-user error | How much alignment is needed |

## Progression at a glance

```
 Stage 0   Foundations        notebooks 00–03 (done; 03 revised)
    │
 Stage 1   01 Geometry        DO synergies exist?          PCA → NMF → synergy tests → AE → UMAP/CEBRA
    │
 Stage 2   02 Multimodal      HOW do EMG and pose relate?  baselines → ridge → lags → CCA → shared AE
    │
 Stage 3   03 Trajectories    HOW does the state move?     plots → speed → position vs velocity
    │
 Stage 4   04 Dynamics        WHAT drives the motion?      LDS → free rollouts → rSLDS → LFADS-style
    │
 Stage 5   Generalise         Does it hold across people?  repeat key results on a full-dataset subset
```

Each stage uses the outputs of the one before: 01 gives the latent space, 02 gives the reference
numbers and the EMG↔pose relationship, 03 describes motion in the latent space, 04 models that motion.

## Stage 0: Foundations
| Step | What | Status |
|---|---|---|
| 0.1 | `00_getting_started`: Meta's tutorial (data loading, pre-trained model, hand mesh) | done |
| 0.2 | `01_the_semg_dream`: first look at raw signals | done |
| 0.3 | `02_eda_features_embeddings`: 80 windowed features, PCA dimensionality, UMAP/PHATE | done |
| 0.4 | Exercise: fix the kNN cross-validation leak in notebook 02, section 6 (`GroupKFold`) | **to do** |
| 0.5 | `03_THANOS_DimRed` (revised): run it on Colab, record the numbers in `exp01_geometry/log.md` | **to do** |

## Stage 1: Geometry ([brief](exp01_geometry/brief.md))
| Step | What | Shows |
|---|---|---|
| 1.1 | PCA on the 16-channel envelope: variance curve, participation ratio | How many dimensions the data uses (linear estimate) |
| 1.2 | Stage decoding from PCA at k = 1, 2, 4, 8, with recordings or time blocks held out | The baseline separability curve |
| 1.3 | NMF at the same k: VAF curve, same decoding | Do non-negative synergies separate tasks better than PCA? |
| **1.4** | **Synergy tests** (for the k chosen in 1.3): | **Whether the synergies are plausibly physiological** |
| 1.4a | Cross-task generalisation: synergies fit on some stages reconstruct held-out stages | Shared modules vs task-specific patterns |
| 1.4b | Null model: VAF vs the same data with each channel's timing shuffled | Real co-activation vs chance structure |
| 1.4c | Crosstalk check: how concentrated each synergy is on neighbouring electrodes | Neural pattern vs electrode overlap |
| 1.4d | Left vs right hand: similarity of the two hands' synergies | Consistency within one person |
| 1.5 | Linear AE vs nonlinear AE at the same k | The curvature test (only nonlinearity differs) |
| 1.6 | UMAP (pictures only) and CEBRA-time; CEBRA with pose labels as an upper bound | Nonlinear structure, without label circularity |
| 1.7 | Intrinsic dimension: participation ratio vs TwoNN | Linear vs nonlinear dimensionality estimates |
| ✓ | **Done when:** curves for all methods, the four synergy tests, 3 seeds, and a one-paragraph answer in the brief | |

## Stage 2: Multimodal ([brief](exp02_multimodal/brief.md))
| Step | What | Shows |
|---|---|---|
| 2.1 | Static baselines (median pose; "no change") and ridge envelope → angle and → velocity | Reference numbers; tests your angle-vs-velocity prediction |
| 2.2 | Lag sweep: shift EMG 0–150 ms against pose | Electromechanical delay (literature: ~30–100 ms) |
| 2.3 | Add EMG history (previous 5–10 frames) | Whether pose needs memory of past activation |
| 2.4 | Encoding: pose (+ velocity) → EMG | What pose can't explain (co-contraction, force) |
| 2.5 | CCA / PLS with a time-shift null | How many shared EMG–pose dimensions there are |
| 2.6 | Shared vs private autoencoder | Open niche in the literature for hand sEMG |
| ✓ | **Done when:** decoding/encoding R² table vs baselines, CCA spectrum vs null | |

## Stage 3: Trajectories ([brief](exp03_trajectories/brief.md))
| Step | What | Shows |
|---|---|---|
| 3.1 | Plot latent trajectories per stage (PCA or CEBRA space from stage 1) | What motion looks like in the manifold |
| 3.2 | Speed and dwell time along trajectories | Moving between regions vs staying in one |
| 3.3 | Stage decoding from position vs from velocity in latent space | Is task identity in *where* or *how*? |
| 3.4 | Segment movements (joint-velocity thresholds); compare repeats | Are repeated movements the same path? |

## Stage 4: Latent dynamics ([brief](exp04_latent_dynamics/brief.md))
| Step | What | Shows |
|---|---|---|
| 4.1 | Linear model z(t+1) = A z(t) + B u(t), pose as state, EMG as input, least squares | The simplest dynamics model; persistence baseline |
| 4.2 | Eigenvalues of A as time constants (ms) | Decay, oscillation or persistence |
| 4.3 | Free rollouts: with EMG, without EMG (B = 0), input switched off | **The input-driven vs self-sustaining test** |
| 4.4 | rSLDS (switching linear regimes) | Whether dynamics differ by region of the manifold |
| 4.5 | LFADS-style model on pose only; compare inferred inputs with measured EMG | Can the model recover the drive it never saw? |

## Stage 5: Generalisation (later)
Download a subset of the full dataset (several users) and repeat the key result of each stage.
The EDA predicts that participants will cluster more strongly than stages, which motivates alignment.

## Reading: which lecture to re-read before each stage
Course slides are shared separately. Page numbers are PDF positions, not printed slide numbers.

| Stage | Re-read first |
|---|---|
| 0 | Lecture2 p17 (zero-phase filtering) · Lecture1 p9 (the course's own version of the 01 and 04 questions) |
| 1 Geometry | Lecture06 (PCA cost and SVD, FA, LDA) · Lecture07 (participation ratio, cross-validated PCA, Isomap/t-SNE caveats, CEBRA, principal angles between tasks) · Lecture3 p23–27 (NMF update rules) |
| 2 Multimodal | Lecture16 p13–15 (optimal linear estimator / Wiener filter), p14 (Kalman decoder), p32–33 (nested cross-validation), p44–47 (shared-latent decoding) · Lecture06 (reduced-rank regression; output-null/potent spaces) |
| 3 Trajectories | Lecture07 (trajectories in PC space) · Lecture2 p28–35 (K-means/GMM, for segmenting) |
| 4 Dynamics | Lecture9 (LDS, Kalman equations p31–37, M-step for A p44, inputs p53, eigenvalues p51/p54) · Lecture11_LG (switching LDS with force as input, the closest analogue to EMG driving pose) · Lecture10 (PLRNN, LFADS inferred inputs p32, judge by free generation p34) · Lecture12 (fixed points; p31: inputs and autonomous dynamics can explain the same data) |

**Warm-up:** Problem Set 2 (PCA/FA, EM for FA, NMF vs PCA) is close to stage 1. Only the solutions
to 1(b) and 4(a) are in the folder.
**Notation:** follow Lecture 9: x = observation, z = latent, A, C, Q, R = parameters, B and u = input.
**Not taught in the course:** CCA/PLS (reduced-rank regression is the closest), UMAP, sEMG preprocessing.

- Course pre-readings are shared separately (file names like `0922_PreReading.pdf`).
- Kao lab (Hadidi et al. 2026, arXiv:2603.08212): position vs velocity decoding on emg2pose. Read
  before step 2.1. Key lesson: a "Static" baseline comes close to every model.

## Updating this plan
Change a brief when a decision is made or a result changes the question. Record *why* in that
experiment's `log.md`.
