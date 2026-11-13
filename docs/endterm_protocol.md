# Endterm protocol - fixed before ANN training

Owner: Khamza. Scope: feedforward ANN and shared reproducibility contract.
This continues the EPA combined fuel-efficiency regression project; the topic,
dataset and existing seed-42 split are unchanged. It does not implement Askhat's
clustering/evaluation or Aktore's PCA/final report.

## Dataset snapshot

- Source: U.S. EPA / DOE FuelEconomy.gov,
  <https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip>.
- Data dictionary: <https://www.fueleconomy.gov/feg/ws/index.shtml>.
- Local file: `data/raw/vehicles.csv`. No download or replacement is performed
  by the ANN commands. Do not use `download --force` to reproduce this run.
- SHA-256: `b1ff0e3071c46cdc1c99e0015878b98a01851462d7e447237ce2fc1294ccf303`.
- The midterm `data/README.md` records acquisition on **2026-10-08**. This is
  historical documentation, not an independently verified acquisition timestamp
  for these local bytes. The snapshot identity is the hash above; matching old
  row counts alone cannot prove historical byte identity.
- Existing scope: model year >=2015; existing filters exclude EV, fuel-cell,
  plug-in-hybrid and other non-comparable fuel entries. No new filtering.
- Prepared data: **13,975 rows, 14 columns** (including `row_id` and target).
- Existing `data.make_splits(seed=42)` / duplicate-grouped `GroupShuffleSplit`:
  **train 9,816; validation 2,080; previously used holdout 2,079**.
  No new split is introduced and existing split files are not overwritten.
- The existing duplicate-group key includes all project columns, including MPG;
  this inherited split definition is retained. MPG is never a predictor.
  Exact duplicates stay together; related make/model variants can cross splits.

The machine-readable reference is
`reports/results/endterm/ann_dataset_manifest.json`. Its SHA-256 fields include
the raw bytes, each ordered split's row IDs, and the full sorted `(row_id, split)`
assignment. The latter is `3104489e6deddeb5cad770545d0fd8192bb432428a0039494705ef136eca01c7`.
Hashes of ID lists use compact JSON integer lists; the assignment hash uses
compact JSON of `(integer ID, split name)` pairs sorted by ID. See
`neural_network.dataset_manifest` for the executable definition.

## Inputs, target and preprocessing

Raw numeric inputs: `model_year`, `cylinders`, `displacement`, `hybrid`, `turbo`,
`supercharged`. Categories: `drive`, `transmission`, `vehicle_class`, `fuel_type`.
The existing stateless feature `displacement_per_cylinder` is retained; this is
an inherited midterm choice, not a new feature search. `make`/`model` are for
identification/error analysis. `mpg`, `comb08`, `row_id` and target-derived
features are excluded from the design matrix.

Reuse `split_xy`, `add_features` and `build_preprocessor`:
numeric median imputation then StandardScaler; categorical most-frequent
imputation then one-hot encoding with unknown categories ignored. Every fitted
statistic/category comes from train only. Validation uses `transform` only.
The sparse/dense output is converted to contiguous float32 for the CPU ANN;
sparse values are cast before densifying. No features are selected using test.

Target: `mpg` (raw `comb08`), EPA combined **US MPG**. A separate StandardScaler
is fitted to **training targets only**. The loss is MSE in standardised target
units. Predictions are inverse-transformed; primary MAE and secondary RMSE are
in MPG, R-squared is unitless. A constant positive target scale means validation
MSE and validation RMSE in MPG rank epochs identically. No output clipping.

## Predeclared ANN experiment

- PyTorch **CPU**, float32, one thread; no alternative neural framework.
- Hidden sizes `(32,)` and `(64,32)` crossed with learning rates `0.001` and
  `0.0003`: exactly four candidates, in that order of nested iteration.
- Initial requested architecture `(64,32)` with `lr=0.001` is included.
- ReLU after hidden layers; one unrestricted linear regression output.
- MSE loss; Adam with its installed-version defaults, no weight decay.
- Batch size 64, maximum 200 epochs, shuffled training batches only.
- Early stopping: patience 20 consecutive epochs without a **strictly lower
  validation MSE**, no minimum-delta threshold. Copy the best state_dict and
  restore it; do not return the final epoch's weights by default.
- Within a trial, checkpoint/epoch selection uses validation loss. Across
  restored trial checkpoints, select the lowest **validation MAE**. Exact ties
  use the declared grid order. No extra trials added after inspecting results.
- Seeds: Python, NumPy, torch initialization and batch permutation generator 42.
  `torch.use_deterministic_algorithms(True)`; CPU only, one thread.
- Histories measure full train/validation loss and MAE after each epoch's
  updates, not a mixture of pre-update batch losses.

## Roles and interpretation of evaluation sets

Train is for fitting inputs/target transforms and network weights. Validation
is reused for early stopping and four-candidate selection; its score is
selection-biased and is not an independent generalisation estimate. The ANN
does not claim grouped cross-validation. Classical tuning uses Askhat's
training-only grouped CV; final supervised comparison must use identical
validation row IDs and the same metric definitions despite different tuners.

**Test is a previously used midterm holdout, not a newly untouched endterm
test.** Its historical Decision Tree result was published in midterm. ANN
development does not construct test X/y, predict test, compute test metrics or
inspect test residuals. Loading raw data and recreating split membership is
needed for the inherited split; test row IDs/counts are used only as metadata.

Askhat may load the frozen ANN and evaluate it in the agreed final evaluation
stage without retraining, after all models/choices are fixed. Document prior
holdout use and do not tune in response to those scores. No claim of a new
independent endterm test is warranted. No new split or nested strategy is
introduced here. This module intentionally has no `--test` option.

## Artifact and integration contract

- The initial experiment preserved the midterm notebooks and figures. During final
  presentation cleanup, the obsolete midterm report/plan and two old evaluation
  figures were removed from the working tree; their originals remain in Git.
  Required EDA now lives in `notebooks/endterm/00_eda.ipynb` and its four figures
  in `reports/figures/endterm/`. Historical top-level result files remain for
  provenance; final comparisons read `reports/results/endterm/`.
- New notebook: `notebooks/endterm/03_ann.ipynb`.
- New tables/metadata/bundle: `reports/results/endterm/ann_*`.
- New plots: `reports/figures/endterm/ann_*`.
- Save unrounded validation predictions with `row_id,y_true,y_pred`.
- History contains all trials, losses, MAEs, candidate ID, selected flag,
  best epoch and an `is_best_epoch` marker. Loss is standardised MSE; MAE is MPG.
- Search contains restored-checkpoint train/validation metrics, epochs run,
  stopping status, parameters and selected flag.
- Bundle: `ann_model.pt` (state_dict), `ann_preprocessing.joblib` (fitted input
  pipeline and target scaler), `ann_config.json` (architecture, checkpoint,
  metadata and bundle checksums). Load only the team's trusted joblib artifacts.
- Metrics/config record dataset/split/features, seeds, actual versions,
  deterministic settings and source hashes. `test_evaluated` is false.

Before merging results, each participant must compute
`dataset_manifest(df, make_splits(df, seed=42))` using the same local snapshot,
then call `verify_manifest(reference, actual)`. A mismatch in snapshot, split
IDs/order, seed or feature contract must stop comparison; do not silently
replace the reference manifest. Align predictions by unique row_id, confirm
the exact validation ID set and unchanged y_true, then use shared
`evaluation.compute_metrics`. Use the same train-only fitting design for the
classical models. Old saved metrics without equivalent provenance are
historical context, not a substitute for Askhat's verified endterm comparison.

`load_ann(expected_manifest=manifest).predict(X)` requires shared raw inputs
and applies the stored transforms and target inverse transform without fit.
`--verify` checks snapshot, bundle checksums, validation rows/targets and
recomputed predictions/metrics. It does not access test outcomes.

## Reproduction and limits

See `docs/endterm_ann_handoff.md` for commands, actual results and defense notes;
`requirements-endterm.txt` pins the tested package versions.
The original snapshot is Git-ignored: teammates must obtain the same real CSV
and verify its hash; downloading the current EPA file may produce a mismatch.
Do not regenerate the reference manifest to conceal that difference.

Deterministic CPU operations and seeds support repeatability in the tested
environment. Bitwise equality across OS/CPU/library/PyTorch releases is not
guaranteed. See [PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.15/notes/randomness.html).
Best weights are deep-copied and saved as a state_dict following
[PyTorch saving/loading guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
Scientific conclusions concern EPA ratings for this scoped dataset; missing
weight/horsepower/hybrid-system detail and related variants limit transfer.
