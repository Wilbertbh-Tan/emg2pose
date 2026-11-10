# Experiment 03: Trajectories ("the skater")

**Status:** planning · **Notebook:** `notebooks/experiments/advanced_plan/exp03_trajectories/trajectories.ipynb` (to be written by the user)
**Builds on:** the embeddings from Experiment 01

## Question
Within a task, does the state move between regions of the manifold or stay inside one?
Are repeated movements the same path? Is task identity carried by *where* the state is or by *how it moves*?

## Approach (descriptive only, no dynamics model yet)
- Plot trajectories in the Experiment 01 embedding, coloured by time
- Speed along the trajectory; how long the state stays in one place
- Similarity of repeated movements within a stage (e.g. distance between aligned segments)
- Compare: stage decoding from *position* vs from *velocity* in the embedding

## Decisions to make first
- [ ] Which embedding (a parametric one is needed for new data: PCA, NMF, CEBRA; UMAP has `transform` but it is slow)
- [ ] How to define a "repeat" without trial markers (e.g. kinematic velocity thresholds)

## What to show
1. Example trajectories per stage
2. Position-only vs velocity-only stage decoding
3. Within-stage path similarity vs across-stage

## Concepts to learn
State space · trajectories and velocity · recurrence · why order in time matters (and PCA ignores it)
