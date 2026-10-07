# Team Tasks

Branch naming: `aktore/<topic>`, `khamza/<topic>`, `askhat/<topic>`. See [CONTRIBUTING.md](../CONTRIBUTING.md).

## Already done (initial setup - committed by whoever merges it)

- Raw data download, loading, schema checks, duplicate-safe 70/15/15 split (`data.py`)
- Preprocessing `ColumnTransformer`, one engineered feature (`preprocessing.py`)
- Metrics, CV helper (`evaluation.py`), `make_pipeline` helper, DummyRegressor baseline (`models.py`)
- Command `python -m vehicle_efficiency.train` that saves baseline validation metrics
- Tests (`pytest`) and notebook skeletons

Everyone: run the setup in the README and `pytest` before starting.

---

## Khamza - data pipeline, baseline, Linear Regression, environment

**Owned files:** `src/vehicle_efficiency/data.py`, `preprocessing.py`, `pyproject.toml`, `tests/test_data.py`, `tests/test_pipeline.py`, `build_linear_regression` in `models.py`, notebook sections 4-7 and the Linear Regression cell of `project.ipynb`.

| # | Task | Acceptance criteria | Status |
|---|---|---|---|
| K1 | Review the scaffolded cleaning code (`data.prepare` scope filter); check `train.describe()` for implausible values; write findings and removed-row counts in notebook section 4 | Section 4 lists every cleaning decision with a reason; no code duplicated from `src/` | ✅ Done (awaiting review) |
| K2 | Test whether `displacement_per_cylinder` (and anything else you add) helps | A validation-set comparison with/without; kept or dropped based on the numbers; justified in section 5 | ✅ Done (awaiting review) |
| K3 | Implement `build_linear_regression` (scaled; try Ridge optionally) | `python -m vehicle_efficiency.train --model linear_regression --cv` saves metrics; baseline vs LR in the notebook | ✅ Done (awaiting review) |
| K4 | Interpret baseline and LR results (coefficients, sign sanity check) | Written from your real output | ✅ Done (awaiting review) |
| K5 | Confirm a fresh clone works: README setup commands | A teammate follows them on their machine without help | ✅ Done |
| K6 | Add tests for anything new you write | `pytest` passes | ✅ Done (18 passing tests) |

**Dependencies:** none to start. Askhat needs your pipeline to remain stable - announce interface changes in a PR.

## Askhat - Decision Tree, KNN, cross-validation, tuning, comparison, error analysis

**Owned files:** `src/vehicle_efficiency/evaluation.py` (extensions), `build_decision_tree`, `build_knn` in `models.py` (coordinate with Khamza: separate functions, small PRs), notebook sections 8 (tree, KNN, CV) and 9, `reports/results/`.

| # | Task | Acceptance criteria |
|---|---|---|
| A1 | Implement `build_decision_tree` | `--model decision_tree` runs; train vs validation gap discussed (overfitting) |
| A2 | Implement `build_knn` with scaling | `--model knn` runs; pipeline contains a scaler |
| A3 | Cross-validation for at least one model | Mean ± std of MAE reported; uses `cross_validate_pipeline` |
| A4 | Initial tuning (`max_depth`, `min_samples_leaf`, `n_neighbors`) | Tuned with training CV or validation only, never test; grid and results recorded |
| A5 | Comparison table: baseline + LR + tree + KNN | MAE, RMSE, R² on validation; units stated |
| A6 | Choose final model, evaluate once on test | One test run, after the choice is fixed |
| A7 | Error analysis | Residual plot, worst 10 predictions, interpretation from the real output |

**Dependencies:** pipeline from Khamza (done); Linear Regression result (K3) for the comparison table.

## Aktore - dataset documentation, EDA, README, presentation, final-stage plan

**Owned files:** `data/README.md`, `notebooks/01_eda.ipynb`, `README.md` (except the status/commands block - Khamza keeps it accurate), `docs/presentation_outline.md`, `docs/final_stage_plan.md`, `reports/figures/`, notebook sections 1-3 and 11.

| # | Task | Acceptance criteria |
|---|---|---|
| E1 | Verify and finish dataset documentation (licence text, units, citation) | No remaining "verify" notes in `data/README.md` |
| E2 | Write problem statement | In README and notebook section 1, own words |
| E3 | EDA with at least 4 plots in `01_eda.ipynb` | You wrote the Python; each plot has a title, labelled axes with units, saved to `reports/figures/`, and a written interpretation based on the plot |
| E4 | EDA summary for modelling | 3-5 bullets Khamza/Askhat can act on |
| E5 | Copy final EDA into `project.ipynb` section 3 | Uses training split only |
| E6 | Presentation slides from `docs/presentation_outline.md` | 7-12 slides, ~10 min, fill in real results at the end |
| E7 | Final-stage plan | `docs/final_stage_plan.md` completed |
| E8 | README: contribution table and final status | Real commit/PR links only |

**Dependencies:** results from K3/A5 for slides; none for E1-E4.

---

## Shared

- Every PR is reviewed by one teammate (rotate: Aktore→Khamza→Askhat→Aktore).
- Final integration of `project.ipynb`: run top to bottom (`Restart & Run All`) with no errors before the defense.
- All three practise the full workflow and speak in both defenses.
- To limit merge conflicts: edit only your owned files; notebooks are edited by one person at a time (announce in chat).
