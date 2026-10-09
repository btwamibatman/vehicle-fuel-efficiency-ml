# Presentation Outline: Vehicle Fuel Efficiency Prediction

Ten minutes total: Aktore about 3:00, Khamza about 3:00, Askhat about 4:00. All model numbers below are taken from committed `reports/results/` artifacts.

| # | Slide | Speaker | Time | Verified content / visual |
|---|---|---:|---:|---|
| 1 | Title, team, question | Aktore | 0:20 | Predict EPA combined US MPG of a recent vehicle from its specifications; regression task. |
| 2 | Data and scope | Aktore | 0:45 | EPA/DOE FuelEconomy.gov `vehicles.csv`; 50,407 raw rows / 84 columns on 2026-10-08; filtered 2015+ MPG-compatible subset: 13,975 rows / 14 columns. Cite source and public-domain terms. |
| 3 | Target, metric, and split | Aktore | 0:30 | Target is `comb08`, combined MPG. MAE is average absolute prediction error in MPG. Shared approximate 70/15/15 split; training-only EDA; test held back until model choice. |
| 4 | EDA observations | Aktore | 1:25 | Show target distribution plus displacement and cylinder plots. Median MPG 22, mean 23.2, range 9--59; displacement-MPG correlation -0.705; cylinders-MPG correlation -0.686. Mention the model-year plot's weak correlation (0.034) and the missing weight/horsepower limitation. |
| 5 | Cleaning and derived features | Khamza | 1:00 | Scope filter; transmission, hybrid, turbo, supercharged flags; categorical encoding; `displacement_per_cylinder`. State that no used training-column values are missing. |
| 6 | Leakage controls | Khamza | 0:45 | Exact duplicates grouped before splitting; median imputation/scaling/one-hot encoding fitted inside pipelines on train folds only. |
| 7 | Baseline and linear regression | Khamza | 1:15 | Validation: DummyRegressor MAE 4.66, RMSE 6.20, R-squared 0.00; engineered linear regression MAE 2.02, R-squared 0.80. Five-fold train CV MAE 1.99 +/- 0.04. |
| 8 | Decision Tree and KNN | Askhat | 0:55 | Both were tuned with grouped 5-fold training CV. Tree: validation MAE 1.059 MPG; KNN: 1.093 MPG. Explain that KNN scaling is inside its pipeline. |
| 9 | Validation comparison and error analysis | Askhat | 1:25 | Decision Tree had the lowest validation MAE (1.059 MPG), ahead of KNN (1.093), Linear Regression (2.019), and baseline (4.662). Show the committed actual-vs-predicted and residual figures. |
| 10 | Final choice, test, conclusions | Askhat | 1:40 | The Decision Tree was selected by lowest validation MAE and then evaluated once on the 2,079-row test set: MAE 1.038 MPG, RMSE 1.668 MPG, R-squared 0.928. Re-state scope and data limitations. |

## Slide-production notes

- Use saved `reports/figures/eda_target_distribution.png`, `eda_displacement_vs_mpg.png`, `eda_mpg_by_model_year.png`, and `eda_mpg_by_cylinders.png` rather than redrawing figures manually.
- Footnote Slide 2 with FuelEconomy.gov, download date, and the exclusion of MPGe-only vehicles.
- Do not describe displacement or cylinder associations as causal. State that the held-out test result was produced only after the validation-based model choice.
