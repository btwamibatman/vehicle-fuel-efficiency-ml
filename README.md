# Vehicle Fuel Efficiency Prediction

## Project question

Can we predict the EPA combined fuel efficiency of a recent vehicle from its specifications? We predict **combined US miles per gallon (MPG)**: larger values mean greater fuel efficiency. This is a supervised **regression** problem. Our primary metric is **mean absolute error (MAE), in MPG**: the average absolute distance between a predicted and EPA-rated MPG, so an MAE of 2 means predictions are off by two MPG on average. RMSE (MPG) and R-squared (unitless) are secondary metrics.

## Team roles

| Member | Responsibility |
|---|---|
| Aktore | Problem and dataset documentation, EDA, README, presentation outline, final-stage plan |
| Khamza | Shared data loading/split, cleaning, preprocessing, feature engineering, baseline and linear regression |
| Askhat | Decision Tree/KNN, cross-validation, tuning, model comparison, error analysis |

See [docs/team_tasks.md](docs/team_tasks.md) for task ownership. The contribution table is intentionally left until real commits/PRs can be linked.

## Dataset

We use the public U.S. EPA/DOE FuelEconomy.gov vehicle data, `vehicles.csv`, restricted to model years 2015+ and vehicles with comparable MPG ratings. The 2026-10-08 download contains 50,407 raw rows and 84 columns; the project subset contains 13,975 rows and 14 columns. The target is raw `comb08` (EPA combined MPG). Core features include model year, cylinder count, engine displacement in litres, drive, vehicle class, fuel type, and transmission.

Source, direct download, public-domain usage terms, filtering rules, schema, data-quality checks, and limitations are documented in [data/README.md](data/README.md). Weight and horsepower are not available in this selected source; the EDA substitutes available engine displacement and model year rather than claiming those features exist.

## Setup and verified commands

Python 3.10+ is required. The commands below were verified on 2026-10-08 with Python 3.12 after installing the package editable.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m vehicle_efficiency.data download
python -m pytest
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.executed.ipynb
```

On macOS/Linux, activate with `source .venv/bin/activate`; use `python` in place of `py -3.12` where appropriate. To run the modelling commands after setup:

```bash
python -m vehicle_efficiency.train
python -m vehicle_efficiency.train --cv
jupyter lab
```

## Reproducibility design

- The shared seed-42 split is approximately 70/15/15 (train/validation/test). EDA and feature decisions use the training split only.
- Exact duplicate project records are grouped before splitting. All fitted preprocessing remains inside scikit-learn pipelines.
- The test set remains reserved for a single final evaluation once the team selects a model.

## Current status

| Item | Status |
|---|---|
| Dataset download and schema | Ready; source/usage terms documented |
| Aktore EDA | Ready: executable training-only notebook, four figures, observations, and preprocessing notes |
| Baseline | Existing saved validation result: MAE 4.66 MPG, RMSE 6.20 MPG, R-squared 0.00 |
| Linear regression | Existing saved validation result: MAE 2.02 MPG, R-squared 0.80; 5-fold training CV MAE 1.99 +/- 0.007 MPG |
| Decision Tree and KNN | TODO: Askhat implementation, CV, and tuning |
| Final test result | TODO: do not run until model selection is complete |

The EDA observations are descriptive, not model results. See [docs/presentation_outline.md](docs/presentation_outline.md) and [docs/final_stage_plan.md](docs/final_stage_plan.md) for the current defense and final-stage plan.

## Repository layout

```text
src/vehicle_efficiency/   shared data, preprocessing, models, evaluation and train code
notebooks/01_eda.ipynb    Aktore's executable training-only EDA
notebooks/project.ipynb   integrated final report notebook
data/README.md            dataset source, terms, schema and limitations
reports/figures/          generated EDA figures
reports/results/          saved modelling results
docs/                     team task, presentation and final-stage documents
```


