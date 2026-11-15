# Notebook evaluation and the investigation ladder

*Updated 2026-11-15.* This document evaluates the project's notebooks, lists the problems that cut across them, and
sets out the **investigation ladder**: the order in which the project's questions are answered, each with a gate
that decides whether the next step is worth running. Every number below comes from a notebook that runs end to end
with all its ✅ checks passing, on data version 2 (see stage 0).

**Two facts change how every number should be read.**
1. **The 30 recordings are 15 simultaneous left/right pairs ("blocks").** Their time stamps overlap 100%, and the
   pair shares the recording number in its file name. In two blocks one wrist did `OneHandedFreeStyle` while the
   idle wrist is labelled `unconstrained`, so pairing the hands by stage name is wrong. The honest number of
   independent units is **15 blocks**, not 30 recordings or 100,000 frames. The shared protocol
   ([`common/protocol.py`](common/protocol.py)) encodes this; notebook 01, Task 6, teaches it.
2. **The data is one person.** Every claim is within-person. Claims about new users need the full dataset (stage 7).

---

## 1 · The ladder and its gates

Each stage asks **one question**, with the **simplest method first**, a **baseline and a null**, and a **gate**.
CIs are 95% bootstrap intervals over blocks (or over the 14 stage folds). ✅ pass · ◐ partial · ✗ fail.

| Stage | Question | Notebook | Key result | Gate |
|---|---|---|---|---|
| **0 Protocol** | Can every later number be trusted? | [01](01_data_prep/01_data_prep.ipynb) | 15 blocks; folds never split a block; data v2: pose low-passed at 6 Hz, dropouts padded (valid pose 84% → 77%) | ✅ |
| **1 Existence** | Is the EMG low-dimensional beyond chance? | [02](02_how_many_synergies/02_how_many_synergies.ipynb), [03](03_are_synergies_real/03_are_synergies_real.ipynb) check 0 | k = 5 by leave-one-stage-out, +18.5 points [16.6, 20.4] over random synergies; k = 4: 92.1% uncentred, 81.3% centred VAF (random W 73%, mean only 57%); stability ≥ 0.99 | ✅ |
| **2 Artefacts** | Muscle structure, not crosstalk? | [03](03_are_synergies_real/03_are_synergies_real.ipynb) | All 4 synergies more clustered than a permutation null (p ≈ 0.01); **2 of 4 wider than one muscle + crosstalk**, 2 indistinguishable from it. Left vs right 0.85 (null 0.70), but the hands are simultaneous | ◐ |
| **3 Sharing** | Do tasks reuse shared structure? | [04](04_compositional_structure/04_compositional_structure.ipynb) | EMG subspaces shared far above random (index 0.90; first angle 5° vs 42° random). Pose much less shared (0.25; the CI of the difference, +0.25 to +1.43, shows the pose index is unstable). Instructed synergies explain free-style tasks: 0.97, 0.91, 0.87 of their own fit. EMG–pose alignment only with EMG leading (CCA 0.02 at 0 ms, 0.39 at 200 ms) | ✅ / ◐ — the **opposite** of Lazzari & Saxena's "muscles less shared"; a shared co-activation direction may drive it |
| **4 Movement** | What does EMG encode, at what lag? | [05](05_link_to_movement/05_link_to_movement.ipynb) | Lagged log-EMG ridge: on new tasks −2.38° [−3.86, −1.05] vs global median and −2.35° [−3.56, −1.21] vs shifted EMG. But it **ties with each recording's own median posture** (+0.72° [−2.88, +4.15]). Best single lag 200 ms. Angle decodes better than velocity (R² 0.16 vs 0.11); encoding pose → EMG fails (R² < 0) | ◐ |
| **4b Shared subspace** | Which latents do EMG and pose share? Are they the synergies? | [06](06_shared_subspace/06_shared_subspace.ipynb) | The shared subspace is **rank 2–3**: rank 3 beats the median by −1.51° [−2.60, −0.54] and shifted EMG by −1.46° [−2.16, −0.82] on new tasks. It is **not the synergies**: angles to NMF 47°/67°/88° (35% of random subspaces are as close), and it holds 9.8% of EMG variance (random 3-D: 18.4%). Delays 160–300 ms. "One shared finger latent" does **not** beat its null (the fingers' poses are correlated); only the thumb's separate direction does | ✅ |
| **5 Dynamics** | Is movement input-driven? What structure helps a decoder? | [07a](07_dynamics/07a_dynamics.ipynb), [07b](07_dynamics/07b_compositional_dynamics.ipynb) | **Linear (07a):** EMG-driven rollouts are worse than input off; "no change" wins at long horizons. Within recordings Kalman + velocity beats the recording median (−3.41° [−6.17, −0.94]), but on new tasks no decoder beats the median of 5 s of calibration; a per-task resting posture helps (20.0° → 18.3°); the shared-subspace Kalman (6-D state) matches the 40-D one. The physical model loses to Kalman + velocity. **Compositional (07b):** switching motifs are task-specific (motif–task information 0.23 bits vs null 0.006) but don't predict better than one linear model (K = 1 best). A multitask RNN is the best decoder so far (12.4°): it beats ridge + task input by −1.50° [−2.69, −0.54] and shifted EMG by −3.67°, and roughly ties the recording median (−2.55° [−5.58, +0.11]). **Transfer:** learning only a new task's input vector beats retraining all weights at 2–5 s (13.7° vs 17.4° at 2 s), on one held-out task | ✗ for input-driven dynamics; ◐ for the RNN and task-vector transfer |
| **6 Bottlenecks** | Do bottlenecks keep information and cut calibration? | [08](08_bottlenecks/08_bottlenecks.ipynb) | **Variance-based synergies (k = 2–4) are no better than random** (17.3–17.6° vs no bottleneck 14.9°). A **pose-informed (reduced-rank) bottleneck keeps almost everything** (15.4–15.5° at k = 2–4). Neither saves calibration data: the reduced-rank curve tracks the no-bottleneck one (−0.01° [−1.03, +0.75] at 5 s) | ✗ for the proposal's synergy bottleneck; ◐ for pose-informed bottlenecks |
| **7 People** | Do synergies and decoders transfer to new users? | — | Needs a multi-user subset of the full dataset | not run |
| **8 Simulation** | Does a simulated hand + controller reproduce the real EMG structure, and which ingredient causes it? | [10](10_goal_driven_simulation/10_goal_driven_simulation.ipynb) | A differentiable hand (20 joints, 32 lumped muscles, shared tendons, 16-electrode ring) driven by a Lazzari-style RNN, trained per fold on training tasks only; parameters from the literature, not fitted to the EMG. **Goal-driven** (pose only) imitates well (7.5° seen, 12.1° unseen tasks) but its EMG barely resembles the real EMG (r ≈ 0.05). It reproduces **1 of 3 signatures** (EMG more shared than pose) and partly a second (rank 2 shared subspace, but high-variance and close to the synergies); synergy count not reproduced. **Ablation:** tendon coupling changes nothing; wide crosstalk produces the sharing signature, narrow crosstalk the low-variance subspace, so neither setting reproduces both. **EMG-constrained** matches seen-task EMG (r 0.20) but not unseen (0.05), costs pose accuracy, and needs *less* co-contraction. Controllers: rotational slow points, ~6-D linear dynamics, mostly shared units (robust across 3 seeds) | ✗ — tendons + crosstalk don't explain the real EMG structure; the missing ingredient is likely EMG unrelated to finger pose (wrist muscles, co-contraction) |

### Reading the ladder as a whole
- **The synergies are real but are not what moves the hand.** They are low-dimensional beyond chance, stable and
  shared across tasks (stages 1–3), with crosstalk a live explanation for half of them. The part of the EMG that
  relates to pose is a separate 2–3 dimensional, low-variance subspace, delayed by about 200 ms (stage 4b).
  That is why a synergy bottleneck loses the pose information and a pose-informed one keeps it (stage 6).
- **Decoding mostly reads "which task is going on".** Within one person, the best decoders roughly tie with each
  recording's own typical posture (stages 4, 5).
- **The most promising lead for H1 is compositional:** in the multitask RNN, a new task is learned from 2–5 s by
  fitting only its input vector, without overfitting (stage 5). It is one held-out task so far.
- **The simulation (stage 8) rules out the simplest explanation:** shared tendons plus electrode crosstalk do not
  reproduce the real EMG's structure, so it is not just biomechanics and recording.
- **For the proposal:** variance-based synergy bottlenecks should not be the stage-7 design. Pose-informed
  bottlenecks and task-vector calibration should.

---

## 2 · Evaluation of every notebook

| Notebook | State | Main points |
|---|---|---|
| eda/00 getting_started | tutorial, not run | The pretrained vemg2pose model is the decoding **ceiling**; nothing is compared with it yet |
| eda/01 the_semg_dream | separate proposal | Hardware requirements (secondary aim); keep apart |
| eda/02 features_embeddings | figures saved | Participation ratio ≈ 2. The "hand side leaks into the embedding" result is expected: the hands are recorded together |
| eda/03 THANOS master pipeline | run | **Do not cite its effect sizes** (F = 51–151, d<sub>z</sub> = −6.5): windows of one person are treated as independent subjects, and its own permutation test gives p = 1.0. A warning cell sits above them |
| 01–09 experiment notebooks | all rebuilt | Same format everywhere (below). All run end to end with every ✅ check passing when the TODOs are filled in |

### Problems found and fixed
1. **Wrong unit of replication** (frames, windows, recordings, "pseudo-users") → blocks, in `common/protocol.py`.
2. **Hands paired by stage name** → paired by recording number (two blocks differ).
3. **Generous or missing references** → centred VAF, random-W and shifted-EMG nulls, per-recording median baseline.
4. **Decoding before understanding** → the ladder: existence, sharing and the EMG–pose link come first.
5. **Pose jitter and wobbly frames next to dropouts** → data v2. Smoothing makes one-step prediction trivially easy,
   so dynamics are judged on multi-step rollouts.
6. **A leak in the old calibration curves** (eigenpostures fitted on the held-out task) → refit per fold.

## 3 · Format of every learning notebook
Generated by one shared builder: title with ladder stage and "We expect / Wrong if" → goal and milestones with the
gate → setup and data (rebuilt automatically when out of date) → your prediction → per task: explanation,
`# TODO`, three hints, ✅ check against the reference module `refNN.py`, 📊 plot, ✍️ question → ⚠️ methodology →
Result (the gate). Notebook 09 collects the gates.

## 4 · What still needs doing (in order)
1. **Confirm the compositional transfer result** on every held-out task group (14 folds) with CIs, not one task.
   It is the strongest within-person evidence for H1.
2. **Test the co-activation explanation of stage 3:** remove the shared overall-activation direction from the EMG and
   recompute the sharing index; stabilise the pose sharing index.
3. **Pose-informed synergies in the bottleneck:** the supervised NMF in notebook 06 (`ref06.supervised_nmf`) as a
   non-negative, interpretable alternative to the reduced-rank bottleneck in notebook 08.
4. **Ceiling:** score the pretrained vemg2pose model (eda/00) on the same folds.
5. **Causal features** for any forecasting or online claim (zero-phase envelopes leak about 100 ms of future EMG).
6. **Confirm k = 5 vs 6** without the speed shortcuts.
7. **Stage 7:** a multi-user subset of the full dataset, testing a pose-informed bottleneck plus task/user-vector
   calibration.
8. Housekeeping: fill notebook 09 from the gate cells; the `log.md` files are empty; mark the `advanced_plan/`
   briefs that notebooks 04, 06 and 07 now cover.
