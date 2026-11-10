<!--
6-SLIDE DECK + 5-MINUTE SCRIPT
Paste each slide into Google Slides (widescreen 16:9).
Keep ≤6 bullets, ≥30 pt body, one idea per slide.
Speaker lines are timed to ~50 s each (5:00 total). Do not put speaker text on the slide.
-->

# Slide 1 — The bottleneck is a new person

**Title:** sEMG can drive the hand. A new user breaks it.

- Always-available hand input from wrist sEMG — if it generalizes.
- emg2pose (Salter et al., 2024; Meta): 193 users, 370 hours, 16 channels, 20 joint angles.
- Best baseline (vemg2pose): **12.2°** on held-out users → **15.8°** when users *and* movements are new.
- Every product pays for that gap with per-user calibration.

**Visual:** one bar pair, 12.2° vs 15.8°, labeled “new user” / “new user + new movement”. No extra series.

**Speaker (0:00–0:50).**
Wrist EMG could give us a keyboard we never take off. It does not, because a decoder trained on you fails on me. Meta’s emg2pose benchmark measures that failure: twelve degrees of angular error on a new person, almost sixteen when the movements are new too. Anatomy, electrode placement, the way each of us recruits muscle — the paper names those as the cause. We want to know which of those is actually doing the work.

---

# Slide 2 — Two manifolds, not one map

**Title:** Learn muscle and pose separately, then ask how they connect.

```
sEMG  →  muscle manifold  ─┐
                           ├──► aligned map ──► pose
 angles → kinematic man.  ─┘
```

- Muscle activity is low-dimensional (synergies). Hand pose is too (Santello: two PCs, >80% of 15-DoF grasp variance — the linear baseline).
- One end-to-end map hides the relationship. Two manifolds make it the object of study.
- Same move as neural-manifold alignment (Safaie 2023; Degenhart 2020) — untested for sEMG-to-pose.

**Visual:** two streams that meet. Detail lives on Slide 5.

**Speaker (0:50–1:40).**
The usual response is a bigger supervised model. We split the problem. EMG reflects muscle; muscle lives on a low-dimensional manifold. The hand lives on a kinematic manifold. If we learn each of those without the other modality, the only supervised piece left is the map between them. That is the thing we can actually test across people.

---

# Slide 3 — The question (say this twice)

**Title:** Shared coordination, or private behavior?

**Is the muscle → pose mapping shared across people, or private to each person?**

| | H1 Coordination | H2 Behavior |
|---|---|---|
| Bases (synergies, eigenpostures) | Match across people | Differ, or match only by chance |
| Map *A*: muscle latent → pose latent | *A* fitted on you predicts me | *A* does not transfer |
| What that means | Anatomy / coupling | Idiosyncratic recruitment |

- **Null we will report:** bases identical *and* a pooled decoder already transfers → factorization adds nothing.

**Visual:** the two-column table. Highlight the middle row — the map is the question, not the decoder score.

**Speaker (1:40–2:30).**
Here is the claim. If anatomy and biomechanical coupling dominate, synergies and eigenpostures will look alike across people, and a map fitted on one person will predict another. That is muscular coordination. If people solve the same tasks with private recruitment, the map will not transfer. That is behavior. Those are different scientific objects, and they imply different products: a shared decoder versus mandatory calibration. We will also say it if the whole question is moot — if a simple pooled model already works.

---

# Slide 4 — We already ran the muscle side

**Title:** One subject, already low-dimensional — and nonlinear.

- 80-D sEMG features (16 channels × RMS + four-band power), mini-dataset, one participant.
- **Participation ratio = 2.0.** PC1 = 69.8% of variance.
- Stage identity from a 2D embedding: **UMAP 40.0% vs PCA 24.2%** (chance 6.7%, 15 stages).
- Hand side: UMAP **67.1%**, but side is **<1%** of PCA variance — a nuisance we will control.
- **Not yet done:** the 20-angle eigenposture half. That is Month 1.

**Visual:** two numbers in huge type (2.0 and 40% vs 24%). Caption: “single participant; pose side unrun.”

**Speaker (2:30–3:20).**
This is not a literature review. On Meta’s mini-dataset we already measured a participation ratio of two in an eighty-dimensional EMG feature space. UMAP recovers task stage at forty percent from two dimensions; PCA only twenty-four. So the muscle side is low-dimensional, and not linearly. The embedding also organizes by left versus right hand, which nobody wants to decode — that is a confound we will hold out, not a finding we will celebrate. The pose side — twenty joint angles, Santello’s question on this skeleton — has not been run. That is the first month, and it is a sanity check, not a fishing expedition.

---

# Slide 5 — Methodology: individual manifold to alignment

**Title:** Individual manifold → alignment → fine-tune

**On the slide — flowchart only, plus one caption.**

```
sEMG signal  →  muscle manifold  →  hand-motion manifold  →  hand pose
```

Under that chain, three numbered stages (this is the process):

```
 ① Independent pretraining          ② Paired alignment         ③ Fine-tuning
 Learn a low-dim sEMG manifold      Align the two manifolds    Small supervised
 Learn a low-dim hand-pose          on the same time windows   readout → pose
 manifold
 (no labels from the other side)
```

Two stacked boxes on the left that only meet at ②:

```
 sEMG  ──►  muscle manifold     ──┐     NMF · UMAP · AE
                                   ├──► aligned map ──► fine-tune ──► pose
 20 angles ──► pose manifold  ──┘     PCA · UMAP · AE
                                      (PCA = linear baseline, not the only fit)
```

**Caption (one line under the diagram):**
Learning them separately reduces the supervised mapping from 80-D → 20-D to a map between two low-dimensional spaces.

**Do not put on this slide:** months, H1/H2, Hungarian, CKA. That is the next slide.

**Speaker (3:20–4:10).**
The method is a process. Surface EMG, a low-dimensional muscle manifold, a low-dimensional hand-motion manifold, then pose. We learn the two manifolds independently — EMG never sees pose, pose never sees EMG — so the only supervised piece is the alignment between them, then a small fine-tune. That is independent pretraining, paired alignment, fine-tuning. PCA is the Santello baseline on the pose side, not the only embedding.

---

# Slide 6 — Timeline: stages and milestones

**Title:** Six months, three stages, a result at Month 4

**On the slide — a horizontal bar, not a bullet list.**

```
 Nov          Jan              Mar               May
 |------ Stage 1 ------|------ Stage 2 ------|------ Stage 3 ------|
 Independent           Paired alignment      Fine-tune + write-up
 pretraining           + shared-vs-private
 Learn sEMG manifold   Align on paired       Readout → pose
 Learn pose manifold   windows               vs matched end-to-end
                       Test: does the map    Zenodo + seminar
                       transfer across people

        ▲ M1                 ▲ M2              ▲ M3                 ▲ M4
      Month 1              Month 2           Month 4              Month 6
```

**Milestone row (four short boxes under the bar):**

| | When | Gate |
|---|---|---|
| **M1** | end Month 1 | Pipeline on ≥50 users. Muscle PR and UMAP-vs-PCA replicate. First pose-manifold curve. |
| **M2** | end Month 2 | Per-subject muscle and pose manifolds for the full user set. |
| **M3** | end Month 4 | Map *A<sub>i</sub>* tested on person *j*. H1 vs H2 called. **Write this up even if Stage 3 slips.** |
| **M4** | end Month 6 | Fine-tuned readout vs matched-capacity end-to-end. Micropublication + 20-min seminar. |

**Footer:** Success is a decision on shared vs private, including a clean null — not beating vemg2pose.

**Speaker (4:10–5:00).**
Stage one, two months: learn each manifold alone. Stage two, months three and four: align them and ask whether that map is the same across people. That call, at Month 4, is the paper even if fine-tuning slips. Stage three is the engineering upside — a small readout, the write-up, the seminar. A null still counts. We already run the pipeline; we need the full 193-user set.

---

# 5-minute video — cut list

Record looking at camera, slides in the corner. One breath between slides. If you run long, cut Slide 2’s neural-manifold sentence and Slide 6’s mentor ask.

| Time | Slide | Must-say sentence |
|---|---|---|
| 0:00 | 1 | A new user breaks sEMG decoding: 12.2° → 15.8°. |
| 0:50 | 2 | Two manifolds, then a map — not one end-to-end model. |
| 1:40 | 3 | Shared coordination or private behavior? That is the claim. |
| 2:30 | 4 | PR = 2.0; UMAP 40% vs PCA 24%; pose side unrun. |
| 3:20 | 5 | Two manifolds learned apart, then aligned, then a small fine-tune. |
| 4:10 | 6 | Three stages. Month 4 is the paper. A null still counts. |

**Do not say:** “we will beat vemg2pose,” “deep learning,” “foundation model,” or any number you have not already computed. **Do say** H1, H2, and the null in the first 90 seconds (Slides 1–3).
