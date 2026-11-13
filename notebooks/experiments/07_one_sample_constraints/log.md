# 07 one-sample constraints — log

**Expected:** under true n=1 recording, mean/ridge dominate; Box 5.4 + kinematic losses give small honest gains; shuffled/leaky protocol looks artificially good.

**Ran:** `run_one_sample_constraints.py` on `recording-1_right.hdf5` (stage 1).

**Observed (test MAE °):**
- mean pose 26.92
- honest ridge NMF 26.43 / linear ~26.45 / synergy RNN Box5.4 26.54
- train-only aug (all) 26.29 (small)
- dishonest shuffle+NMF-all 23.50 (**~3° fake win**)

**Learned:** with one file, RNN does not beat NMF/ridge; leakage can invent a “win”; keep gap splits + train-only bases.
