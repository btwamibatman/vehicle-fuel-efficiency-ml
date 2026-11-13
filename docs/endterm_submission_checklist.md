# Endterm submission checklist

## Completed now

- [x] Shared snapshot and seed-42 split are documented by the ANN protocol.
- [x] PCA fits median imputation, standardisation and PCA only on 9,816 training rows.
- [x] PCA artifacts include variance, loadings, coordinates, configuration and two figures.
- [x] PCA source tests verify component output, train-only fitting and strict cluster joins.
- [x] ANN handoff is present with its validation metrics and reproducibility metadata.
- [x] Full project test suite passed: 34 tests with PyTorch 2.15.0+cpu on Python 3.12.

## Required from Askhat before final submission

- [x] `clustering_assignments_train.csv` with unique train `row_id,cluster` pairs.
- [x] `clustering_search.csv`, profiles, configuration and labelled clustering figures.
- [x] Endterm classical-model tuning records and a validation comparison on the shared split.
- [x] Unified validation predictions and error-analysis outputs for ANN and the selected classical model.
- [x] Final fixed-model test table, explicitly labelled as the previously used midterm holdout.

## Final assembly after those files arrive

- [x] Generate the PCA/K-Means overlay with an exact row-ID match.
- [ ] Complete `notebooks/endterm/project.ipynb` without rerunning selection on test data.
- [ ] Replace all pending markers in README and the presentation with verified artifact values.
- [ ] Run the complete test suite in the documented environment and execute notebooks from a fresh kernel.
- [ ] Visually inspect the 10–15 slide deck and verify every metric's split and units.
