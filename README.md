# T.H.A.N.O.S.: Two-sided Hand-pose Adaptation with Non-negative and Orthogonal Structure

NMA Impact Scholars 2026 · Partner project with Meta Reality Labs on the [emg2pose](https://github.com/facebookresearch/emg2pose) benchmark ·
Andres Santiago Martinez Hernandez · Naveen Kumar Subramani · Wilbert Tan · Youssef Ibraheem

> **Hypothesis (H1):** if low-dimensional representations of sEMG (muscle synergies) and hand pose (eigenpostures)
> capture structure shared across people, then **constraining both sides of a decoder with them will need less
> calibration on a new user** to reach a given joint-angle error, matching or beating the baseline model.

**Research question:** does constraining both the input (muscle synergies) and the output (eigenpostures) of a
baseline decoder with low-dimensional, biologically grounded structure improve generalisation to held-out users?

## Why this matters

sEMG wristbands could read continuous hand pose without cameras, but decoders struggle on new people. On emg2pose's
held-out user + movement split, the vEMG2Pose model is off by **15.63°** per joint, not far from the **18.87°** you get
by ignoring the EMG and predicting each joint's median angle (Hadidi et al. 2026). A little calibration on the new user
helps, so **calibration burden is what limits deployment.**

Our bet: decoders are free to learn high-dimensional, user-specific features, even though both muscle activity and
hand pose can be described with far fewer dimensions. Forcing the decoder through that shared low-dimensional structure
might mean it needs less data from each new person.

## Research questions

| Aim | Question | How | Lead | Weeks |
|---|---|---|---|---|
| **Aim 1** | What are stable, low-dimensional representations of sEMG and pose? | **NMF** on sEMG envelopes (muscle synergies) and **PCA** on joint angles (eigenpostures); pick sizes by held-out error and stability | Wilbert | 1–5 |
| **Aim 2a** | Does each bottleneck help, and does combining them help more? | **2 × 2 design**: neither, sEMG only, pose only, both; same decoder throughout. Compare against **random** bottlenecks of the same size | Naveen | 5–12 |
| **Aim 2b** | Do the bottlenecks reduce calibration on new users? | Error vs calibration-data curves per held-out user; area under the curve; compare with REACT | Andres | 10–18 |
| *Secondary* | How few electrodes and how low a sampling rate still work? | Ablate channels (16 → 8 → 4) and rate (2 kHz → 250 Hz) on the best model | Youssef | 16–22 |
| | Write-up | Robustness checks, micropublication, presentation | Everyone | 21–26 |

**What each result would mean:** if the dual bottleneck beats both single ones *and* the random controls, H1 is
supported. If it helps compression but not calibration, low dimensions are useful for hardware but don't solve
cross-user transfer. If neither helps, unconstrained decoders win. All three are worth reporting.

**Where we are:** Aim 1, on the mini dataset (1 person). The notebooks in
[`notebooks/experiments/`](notebooks/experiments/) build and check the sEMG synergy side first.

## What we've found so far

From the [EDA](notebooks/eda/02_eda_features_embeddings.ipynb) on the mini dataset: **3 components explain 80% of the
sEMG variance and 7 explain 90%**, with the first component tracking overall activation rather than which muscles are
active. So the sEMG really is low-dimensional, which Aim 1 builds on.

**Data now:** the emg2pose *mini* split: 1 participant, 30 recordings (15 movement tasks × 2 hands), each about
1 minute, with 16 EMG channels at 2 kHz and 20 hand joint angles from motion capture. Cross-user questions
(Aim 2) need the full dataset (193 people).

## Quick start

**On Colab:** open a notebook from this repo in Colab (File → Open notebook → GitHub → `Wilbertbh-Tan/emg2pose`)
and run the first cell. It clones the repo and downloads the mini dataset (~617 MB) by itself.

**Locally:** follow [Setup](#setup) below, then open a notebook from inside the repo. The first cell finds the
repo root and downloads the dataset if `emg2pose_dataset_mini/` is missing.

**Run notebook 01 first.** It saves `data/processed_mini.npz`, which all the other notebooks load.

## Notebooks

✅ = in the proposal · 🧪 = small pilot of a proposal aim · ➕ = beyond the proposal (optional)

| # | Notebook | What it does | Proposal | Owner |
|---|---|---|---|---|
| 01 | [`01_data_prep`](notebooks/experiments/01_data_prep/01_data_prep.ipynb) | Clean, shared data for every other notebook (run first) | ✅ preprocessing | Person A |
| 02 | [`02_how_many_synergies`](notebooks/experiments/02_how_many_synergies/02_how_many_synergies.ipynb) | Aim 1: how many sEMG synergies (k) and eigenpostures (K_POSE)? | ✅ Aim 1 | Person B |
| 03 | [`03_are_synergies_real`](notebooks/experiments/03_are_synergies_real/03_are_synergies_real.ipynb) | Aim 1: are the synergies stable? Plus extra checks that they're real | ✅ stability · ➕ checks 1–4 | Person C |
| 04 | [`04_synergies_predict_pose`](notebooks/experiments/04_synergies_predict_pose/04_synergies_predict_pose.ipynb) | Mini 2 × 2 with ridge: do the bottlenecks keep the pose information? | 🧪 Aim 2 pilot | Person D |
| 05 | [`05_results_summary`](notebooks/experiments/05_results_summary/05_results_summary.ipynb) | Final figures and what they mean for H1 | ✅ | Everyone |
| 06 | [`06_synergy_dynamics`](notebooks/experiments/06_synergy_dynamics/06_synergy_dynamics.ipynb) | Does EMG drive the pose over time? | ➕ | Person A |

Notebook 01 is built. Notebooks 02–06 are starters with `# TODO` gaps to fill in. Each folder also has a
`log.md` for notes on what you expected, what happened and what you learned.

### Not started yet

These parts of the proposal need the full dataset (193 people) or a deep decoder, so no notebook covers them yet.
Santi's [master pipeline](notebooks/eda/03_THANOS_master_pipeline.ipynb) sketches most of them.

| Proposal step | Lead | Starting point |
|---|---|---|
| Download the full dataset and reproduce the vEMG2Pose baseline | Wilbert | Meta's code in [`emg2pose/`](emg2pose/) |
| Full 2 × 2 with a deep decoder, random non-negative / orthonormal controls, held-out users | Naveen | Master pipeline, Module 2 |
| Calibration curves, area under the curve, REACT comparison | Andres | Master pipeline, Module 3 |
| Electrode-count and sampling-rate ablations | Youssef | Master pipeline, Module 4 |

### Folder layout

| Folder | What's in it |
|---|---|
| [`notebooks/experiments/`](notebooks/experiments/) | The project notebooks above (01–06), plus [`advanced_plan/`](notebooks/experiments/advanced_plan/research_plan.md) with ideas for later |
| [`notebooks/eda/`](notebooks/eda/) | Exploratory notebooks: 00 Meta's tutorial · 01 first look at the data · 02 feature EDA and figures · 03 THANOS master pipeline |
| [`eda/`](eda/) | Helper code the EDA notebooks import (`features.py`), the figure script and saved figures |
| [`docs/`](docs/) | Slides and talk script |

## Team rules

1. **Test on data the model didn't see**, and hold out whole recordings, never random rows.
2. **Always compare with a simple baseline.** A number alone means nothing.
3. **Use the shared data from notebook 01.** No private preprocessing.
4. **One person per notebook.** Restart and "Run all" before committing; pull before you push.
5. **Write down what you expect before running.** Then compare.
6. **Never commit data** (`*.hdf5`, `*.npz`, dataset folders). `.gitignore` already blocks them.

## Earlier exploration

Windowed sEMG features embedded in 2D with PCA, UMAP and PHATE, coloured by stage (top) and by hand
(bottom). Task structure dominates, but hand side leaks in too. See [`eda/`](eda/) for the full analysis.
These 2D plots are for looking, not concluding.

![Windowed sEMG features in 2D](eda/figures/04_embeddings.png)

---

# Original emg2pose documentation

The rest of this README is Meta's original documentation for the dataset and the baseline models.

[ `[Paper](https://arxiv.org/abs/2412.02725)` ] [ `[Dataset](https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar)` ] [ `[Blog](https://ai.meta.com/blog/open-sourcing-surface-electromyography-datasets-neurips-2024/)` ] [ `[BibTeX](#citing-emg2pose)` ]

A dataset of Surface electromyography (sEMG) recordings paired with ground-truth, motion-capture recordings of the hands. Data loading, baseline model training, and baseline model evaluation code are provided.

## Data

The entire dataset has $25,253$ HDF5 files, each consisting of time-aligned, 2kHz sEMG and joint angles for a single hand in a single stage. Each stage is ~1 minute. There are $193$ participants, spanning $370$ hours and $29$ stages. `emg2pose.data.Emg2PoseSessionData` offers a programmatic read-only interface into the HDF5 session files.

The full dataset statistics are as follows:



The `metadata.csv` file includes the following information for each HDF5 file:


| Column           | Description                                                                                            |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| `user`           | Anonymized user ID                                                                                     |
| `session`        | Recording session (there are multiple stages per recording session)                                    |
| `stage`          | Name of stage                                                                                          |
| `side`           | Hand side (`left` or `right`)                                                                          |
| `moving_hand`    | Whether the hand is prompted to move during the stage                                                  |
| `held_out_user`  | Whether the user is held out from the training set                                                     |
| `held_out_stage` | Whether the stage is held out from the training set                                                    |
| `split`          | `train`, `test`, or `val`                                                                              |
| `generalization` | Type of generalization; across user (`user`), stage (`stage`), or across user and stage (`user_stage`) |




## Setup



### Environment and Dependencies

```shell
# Clone the repo, setup environment, and install local package
# Clone this fork (it has the project notebooks and eda/; upstream does not)
git clone https://github.com/Wilbertbh-Tan/emg2pose.git ~/emg2pose
cd ~/emg2pose
conda env create -f environment.yml

# Activate the environment
conda activate emg2pose

# Install the emg2pose package
pip install -e .

# Install the UmeTrack package (for forward kinematics and mesh skinning)
pip install -e emg2pose/UmeTrack
```



## Getting Started (Small, Sanity-Check Dataset)

The full dataset is $431$ GiB -- which can be cumbersome for a quick start. As a solution, we
also host a smaller (~ $600$ MiB) version of the dataset which can be downloaded and used to run
a sanity-check version of the train and eval logic.

### (Optional) Download Just the Metadata CSV (5 MiB)

The `emg2pose_metadata.csv` file described above can be downloaded on its own using the following endpoint.

NOTE: this metadata file is also included in each of the dataset downloads

```shell
# Download (just) the metadata.csv file to ~/emg2pose_metadata.csv
cd ~ && curl https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_metadata.csv -o emg2pose_metadata.csv
```



### Download a Smaller Version of the Dataset (~600 MiB)

```shell
# Download a mini (600 MiB) version of the dataset
cd ~ && curl "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar" -o emg2pose_dataset_mini.tar

# Unpack the tar to ~/emg2pose_dataset_mini
tar -xvf emg2pose_dataset_mini.tar
```



### Sanity Check Train / Eval

To run a sanity-check training workflow over the small, sanity-check version of the
dataset, please use the following command.

This runs training for the `tracking_vemg2pose` experiment for $5$ epochs as a sanity check.
It also runs evaluation on the validation and test splits -- again as a sanity check.

```shell
python -m emg2pose.train \
train=True \
eval=True \
experiment=tracking_vemg2pose \
trainer.max_epochs=5 \
data_split=mini_split \
data_location="${HOME}/emg2pose_dataset_mini"
```



## Getting Started (Full Dataset)

Above, we provided instructions for working with a smaller version of the dataset as a means
of sanity checking the main entrypoint (`train.py`). Here, we show how to get started with
the whole dataset.

### Download the Full Dataset (431 GiB)

```shell
# Download the full (431 GiB) version of the dataset, extract to ~/emg2pose_dataset
cd ~ && curl https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar -o emg2pose_dataset.tar

# Unpack the tar to ~/emg2pose_dataset
tar -xvf emg2pose_dataset.tar
```



### Train on the Full Dataset

To launch an example, full training run for the `vemg2pose (tracking)` setting, use the following:

```shell
python -m emg2pose.train \
train=True \
eval=True \
experiment=tracking_vemg2pose \
data_location="${HOME}/emg2pose_dataset"
```

The `experiment` CLI option supports the following experiments (see `config/experiment` files):

- `tracking_vemg2pose`
- `regression_vemg2pose`
- `regression_neuropose`



## Downloading Pre-trained Checkpoints

We provide pre-trained checkpoints (as `.ckpt` files) for the following:

1. vemg2pose (tracking, regression settings)
2. neuropose (regression setting)

To download and unpack these checkpoints, run the following.

```shell
# Download checkpoints
cd ~ && curl "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz" -o emg2pose_model_checkpoints.tar.gz

# Unpack to ~/emg2pose_model_checkpoints
tar -xvzf emg2pose_model_checkpoints.tar.gz
```



## Evaluation / Testing

To run basic evaluation for the validation / test splits, use the following:

Note that the `experiment` option to this script should match the checkpoint's experiment.

```shell
# Run train.py with train=False to isolate basic evaluation logic
python -m emg2pose.train \
train=False \
eval=True \
data_location="${HOME}/emg2pose_dataset" \
experiment=tracking_vemg2pose \
checkpoint="${HOME}/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt"
```

To run analyses for different modes of generalization and to generate a `.csv` file with results, use
the following script.

Note that the `experiment` option to this script should match the checkpoint's experiment.

```shell
python -m emg2pose.test_analysis \
data_location="${HOME}/emg2pose_dataset" \
experiment=tracking_vemg2pose \
checkpoint="${HOME}/emg2pose_model_checkpoints/tracking_vemg2pose.ckpt"
```



## Notebook and Visualization

Meta's walkthrough of data loading, inference and visualization is in
[`notebooks/eda/00_getting_started.ipynb`](notebooks/eda/00_getting_started.ipynb).

## License

emg2pose is CC-BY-NC-SA-4.0 licensed, as found in the LICENSE file.

emg2pose is also licensed subject to the licenses of its code dependencies.

UmeTrack is licensed under Attribution-NonCommercial 4.0 International, as found in the emg2pose/UmeTrack/LICENSE and [GitHub](https://github.com/facebookresearch/UmeTrack/blob/main/LICENSE).

## Citing emg2pose

```
@inproceedings{salteremg2pose,
  title={emg2pose: A Large and Diverse Benchmark for Surface Electromyographic Hand Pose Estimation},
  author={Salter, Sasha and Warren, Richard and Schlager, Collin and Spurr, Adrian and Han, Shangchen and Bhasin, Rohin and Cai, Yujun and Walkington, Peter and Bolarinwa, Anuoluwapo and Wang, Robert and others},
  booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track}
}
```

