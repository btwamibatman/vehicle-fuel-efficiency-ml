# Endterm submission checklist

## Completed now

- [x] Shared snapshot and seed-42 split are documented by the ANN protocol.
- [x] PCA fits median imputation, standardisation and PCA only on 9,816 training rows.
- [x] PCA artifacts include variance, loadings, coordinates, configuration and two figures.
- [x] PCA source tests verify component output, train-only fitting and strict cluster joins.
- [x] ANN handoff is present with its validation metrics and reproducibility metadata.
- [x] Full project test suite passed: 38 tests with PyTorch 2.15.0+cpu on Python 3.12.

## Integrated clustering and evaluation

- [x] `clustering_assignments_train.csv` with unique train `row_id,cluster` pairs.
- [x] `clustering_search.csv`, profiles, configuration and labelled clustering figures.
- [x] Endterm classical-model tuning records and a validation comparison on the shared split.
- [x] Unified validation predictions and error-analysis outputs for ANN and the selected classical model.
- [x] Final fixed-model test table, explicitly labelled as the previously used midterm holdout.

## Final assembly and remaining defense preparation

- [x] Generate the PCA/K-Means overlay with an exact row-ID match.
- [x] Complete `notebooks/endterm/project.ipynb` with training EDA, preparation, methods, frozen comparisons and limitations.
- [x] Update README and presentation outline to the completed endterm artifacts.
- [x] Run 38 tests and all six endterm notebooks from fresh kernels; reproduce ANN validation metrics within 1e-10 in an isolated copy without replacing official model artifacts.
- [ ] Visually inspect the 10–15 slide deck and verify every metric's split and units.
