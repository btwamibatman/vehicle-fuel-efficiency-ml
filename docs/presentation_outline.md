# Presentation Outline: Vehicle Fuel Efficiency Prediction

Ten minutes total: Aktore about 3:00, Khamza about 3:00, Askhat about 4:00. Use only results saved in `reports/results/` or reproduced by the notebooks. Items marked TODO are intentionally not claims.

| # | Slide | Speaker | Time | Verified content / visual |
|---|---|---:|---:|---|
| 1 | Title, team, question | Aktore | 0:20 | Predict EPA combined US MPG of a recent vehicle from its specifications; regression task. |
| 2 | Data and scope | Aktore | 0:45 | EPA/DOE FuelEconomy.gov `vehicles.csv`; 50,407 raw rows / 84 columns on 2026-10-08; filtered 2015+ MPG-compatible subset: 13,975 rows / 14 columns. Cite source and public-domain terms. |
| 3 | Target, metric, and split | Aktore | 0:30 | Target is `comb08`, combined MPG. MAE is average absolute prediction error in MPG. Shared approximate 70/15/15 split; training-only EDA; test held back. |
| 4 | EDA observations | Aktore | 1:25 | Show target distribution plus displacement and cylinder plots. Median MPG 22, mean 23.2, range 9--59; displacement-MPG correlation -0.705; cylinders-MPG correlation -0.686. Mention the model-year plot's weak correlation (0.034) and the missing weight/horsepower limitation. |
| 5 | Cleaning and derived features | Khamza | 1:00 | Scope filter; transmission, hybrid, turbo, supercharged flags; categorical encoding; `displacement_per_cylinder`. State that no used training-column values are missing. |
| 6 | Leakage controls | Khamza | 0:45 | Exact duplicates grouped before splitting; median imputation/scaling/one-hot encoding fitted inside pipelines on train folds only. |
| 7 | Baseline and linear regression | Khamza | 1:15 | Saved validation: DummyRegressor MAE 4.66, RMSE 6.20, R-squared 0.00; engineered linear regression MAE 2.02, R-squared 0.80. Saved 5-fold train CV MAE 1.99 +/- 0.007. |
| 8 | Candidate nonlinear models | Askhat | 0:55 | TODO: Decision Tree and KNN design, scaling rationale for KNN, selected parameters. Do not fill until executed. |
| 9 | Validation comparison and error analysis | Askhat | 1:25 | TODO: validation/CV comparison and where predictions fail. Use plots/tables generated from verified runs only. |
| 10 | Final choice, test, conclusions | Askhat | 1:40 | TODO: select by validation/CV, evaluate untouched test set once, report final MAE and limitations. No test number before that run. |

## Slide-production notes

- Use saved `reports/figures/eda_target_distribution.png`, `eda_displacement_vs_mpg.png`, `eda_mpg_by_model_year.png`, and `eda_mpg_by_cylinders.png` rather than redrawing figures manually.
- Footnote Slide 2 with FuelEconomy.gov, download date, and the exclusion of MPGe-only vehicles.
- Do not describe displacement or cylinder associations as causal. Do not claim performance for tree/KNN or the test set until results exist.
