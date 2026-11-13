# Add reproducible CPU ANN for endterm MPG regression

> Historical ANN development/handoff record. The completed endterm
> comparison, current notebook paths and defense entry point are in README
> and `notebooks/endterm/project.ipynb`. Statements below about pending
> integration or preserved midterm files describe the earlier stage.


The endterm project needs a tabular ANN alongside the existing supervised
models. Add a train/validation-only PyTorch CPU workflow with the shared
preprocessing and original seed-42 split, a four-configuration search, early
stopping and restored checkpoints. The selected 64/32 network obtains validation
MAE 1.082715 MPG, RMSE 1.522971 MPG and R-squared 0.939596.

Include the executable notebook, learning curves, unrounded row-level
predictions, dataset/split fingerprints, a load/predict bundle, tested package
pins, protocol and team/defense handoff. Historical midterm artifacts and the
existing data/model/evaluation modules are preserved. No ANN test evaluation;
the protocol discloses prior midterm holdout use.

The Russian implementation guide explains the actual feature dimensions,
target inverse transform and the distinction between loss-selected epochs and
MAE-selected configurations. Include a four-minute defense script/Q&A and an
inference-only walkthrough of a real validation Mazda record.

Validation: 27 pytest checks; real-data three-epoch smoke run; complete
four-candidate run repeated from a fresh notebook kernel with identical
histories, search and predictions; saved-bundle verification; `pip check`;
rendered notebook/figure inspection; preservation audit of existing files.
See `reports/results/endterm/ann_validation_checks.json` for the delivery record.
The walkthrough was also executed and checked against the saved validation
prediction; stored training artifacts remained unchanged.

Limits: validation is reused for early stopping and configuration selection;
one seed, no ANN CV or no-early-stopping ablation, no fresh independent test.
Askhat's final verified classical/ANN comparison and Aktore's PCA/final assembly
remain separate team tasks.
