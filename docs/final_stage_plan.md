# Final-stage Plan

## Current evidence and open problems

- The executable training-only EDA is complete. Engine displacement and cylinder count are strongly negatively associated with MPG, while model year alone is weak in this subset. These are descriptive findings, not a model selection result.
- Baseline and linear-regression validation results are saved. Decision Tree and KNN implementation, tuning, and comparison are still TODO.
- The test partition must remain untouched until the team chooses a candidate using validation/CV evidence.
- The selected EPA source lacks vehicle weight and horsepower. The model may miss important physical drivers of fuel economy, and the original requested plots were transparently adapted.
- Exact duplicates are grouped, but related make/model variants can still occur across partitions. A make+model grouped split is a possible robustness check if course time permits.
- EPA updates the source periodically, so the raw file and row count must be dated whenever results are reproduced.

## Plan to final submission

| Order | Deliverable | Owner | Acceptance criterion |
|---:|---|---|---|
| 1 | Implement Decision Tree and KNN pipelines | Askhat | Both run through the shared split and report validation metrics; KNN uses pipeline scaling. |
| 2 | Cross-validate and tune only on training data | Askhat, with Khamza review | Parameters/search space and mean +/- standard deviation are recorded; no test access. |
| 3 | Compare candidates | Askhat | One table with baseline, linear regression, tree, and KNN validation/CV MAE (primary), with RMSE/R-squared where available. |
| 4 | Check pipeline/data assumptions | Khamza | Confirm no leakage, preprocessing fitted per fold, and any change is justified by a reproducible result. |
| 5 | Select model and run final test once | Team | Decision is recorded from validation/CV; exactly one final test evaluation is saved. |
| 6 | Error analysis and limitations | Askhat | Analyze residuals/subgroups of the chosen model; distinguish evidence from speculation. |
| 7 | Final report and slides | Aktore | Replace TODOs only with verified results, include source/terms, EDA figures, limitations, and contribution links. |

## Decision rule

Choose the model with the strongest validation/CV evidence on **MAE in MPG**, considering instability and complexity. Report RMSE and R-squared as supporting context. Do not select based on the held-out test result.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Unfinished nonlinear models | Time-box tuning and retain the verified linear-regression reference; do not invent comparison results. |
| Leakage or inconsistent feature treatment | Reuse the shared loader/split and pipeline functions; run the existing tests before final evaluation. |
| Overconfident generalisation | State that the sample excludes MPGe vehicle types and lacks weight/horsepower; present EPA ratings as laboratory estimates. |
| Data changes upstream | Keep the download date, rerun the notebook, and update recorded row counts/results together. |
