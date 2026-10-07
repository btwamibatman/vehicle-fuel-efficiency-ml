# Presentation Outline (10 minutes: Aktore ~3, Khamza ~3, Askhat ~4)

Fill in numbers only from saved results in `reports/results/`. Everyone should be able to answer questions on every slide.

| # | Slide | Speaker | Time | Content |
|---|---|---|---|---|
| 1 | Title and question | Aktore | 0:20 | "Can we predict a car's MPG from its specifications?" Team and roles |
| 2 | Dataset | Aktore | 0:50 | EPA fueleconomy.gov data, model years 2015+: source, ~14k rows, features, target units (combined MPG), scope filter (no EV/PHEV), limitations (no weight/horsepower) |
| 3 | EDA findings | Aktore | 1:30 | The 2-3 most informative plots with takeaways |
| 4 | Prediction setup and metrics | Aktore | 0:20 | 70/15/15 split, why; MAE primary (in MPG), RMSE, R² |
| 5 | Cleaning and features | Khamza | 1:00 | Scope filter, transmission/hybrid flags, one-hot categoricals, `displacement_per_cylinder` and whether it helped |
| 6 | No-leakage pipeline | Khamza | 1:00 | Pipelines, preprocessing fitted on train only, duplicates grouped, shared split |
| 7 | Baseline and Linear Regression | Khamza | 1:00 | DummyRegressor vs LR numbers |
| 8 | Tree and KNN | Askhat | 1:15 | Models, scaling for KNN, tuned parameters |
| 9 | Cross-validation and comparison | Askhat | 1:15 | CV mean ± std; comparison table on validation |
| 10 | Final test result and error analysis | Askhat | 1:30 | Single test evaluation; where the model fails |
| 11 | Conclusions and next steps | Askhat | 0:10 + Q&A | Supported conclusions, final-stage plan pointer |

Timing: 3:10 / 3:00 / 4:00 - trim slide 3 or 4 if over. Slides 9-11 stay as drafts until results exist.
