# Experiment 04: Latent dynamics (self-sustaining or input-driven?)

**Status:** planning · **Notebook:** `notebooks/experiments/advanced_plan/exp04_latent_dynamics/latent_dynamics.ipynb` (to be written)
**Builds on:** Experiment 2 of the old `03_THANOS_DimRed` notebook (removed; see git history before commit `c62d0c0`)

## Question
Is hand movement better described as autonomous dynamics (the state evolves by itself) or as
input-driven dynamics (the state follows the sEMG drive and decays when it stops)?

## Key idea
Unlike the olfactory paper, the drive is *observed* here. Treat pose as the state z and sEMG as the input u:

    z(t+1) = A z(t) + B u(t)

## Ladder of models (simplest first)
1. **Linear dynamical system** fit by least squares. Eigenvalues of A show decay, oscillation or persistence.
2. **Switching / piecewise-linear** (rSLDS, PLRNN): different dynamics in different regions.
3. **LFADS-style** model with inferred inputs. Only after 1 and 2 are understood.

## Decisions to make first
- [ ] Sampling rate for the dynamics (50 Hz? 30 Hz?)
- [ ] Prediction horizon(s) for evaluation (e.g. 1, 5, 25 steps)

## What to show
1. k-step prediction error for: persistence baseline · autonomous model (B = 0) · input-driven model
2. Eigenvalue spectrum of A, converted to time constants in ms
3. Free rollouts after the input is switched off: decay, hold, or drift?

## Concepts to learn
Vector fields · fixed points and stability · eigenvalues as time constants · autonomous vs non-autonomous systems ·
teacher forcing vs free rollout
