# Experiment 02: Multimodal reconstruction (sEMG ↔ kinematics)

**Status:** idea (raised 2026-11-08) · **Notebook:** `notebooks/exp02_multimodal/multimodal.ipynb` (to be written by the user)
**Builds on:** `notebooks/00_getting_started.ipynb` (pre-trained EMG→pose model, forward kinematics, IK-failure mask)

## Question
Do sEMG and hand kinematics share a common low-dimensional latent space? How much of each
modality can be reconstructed from that shared space?

## Ladder of approaches (simplest first)
1. **Decoding, EMG → pose.** Ridge from features to joint angles. The pre-trained vemg2pose model
   in notebook 00 is the ceiling.
2. **Encoding, pose → EMG.** Predict sEMG features from joint angles (and their velocities).
   What part of the EMG does posture *not* explain?
3. **Linear shared space: CCA / PLS.** Find paired directions in EMG and pose that covary.
   How many shared dimensions are there?
4. **Multimodal autoencoder.** One bottleneck that must reconstruct both EMG and pose. Compare with
   separate bottlenecks of the same total size.

## Decisions to make first
- [ ] Pose representation: joint angles (20-D) or 3-D landmarks via `emg2pose.kinematics.forward_kinematics`?
- [ ] Remove IK-failure frames (`Emg2PoseSessionData.no_ik_failure`) before fitting anything
- [ ] Do joint velocities belong in the pose vector? (EMG relates more to force and motion than to static posture)

## What to show
1. Decoding and encoding R² vs model complexity, with the pre-trained model as the ceiling
2. CCA canonical correlations: how many are clearly above a shuffled-time null?
3. Shared-AE reconstruction of each modality vs separate AEs at matched size

## Concepts to learn
Encoding vs decoding models · CCA/PLS · shared vs private latent spaces · inverse kinematics and IK failures
