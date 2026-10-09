# Final-stage Plan

## Completed decision path

- Training-only EDA is complete. Engine displacement and cylinder count are strongly negatively associated with MPG, while model year alone is weak in this subset. These are descriptive findings, not causal claims.
- Baseline, Linear Regression, Decision Tree, and KNN were compared with grouped 5-fold training CV and the shared validation set. The Decision Tree had the lowest validation MAE: **1.059 MPG**.
- The Decision Tree was then evaluated once on the held-out 2,079-row test set: **MAE 1.038 MPG, RMSE 1.668 MPG, R-squared 0.928**. The test result is recorded in `reports/results/final_test.json`.

## Final deliverables

| Order | Deliverable | Owner | Status / acceptance criterion |
|---:|---|---|---|
| 1 | Preserve reproducibility | Khamza | Shared loader/split and pipeline tests pass; do not change the selected model after reading the test result. |
| 2 | Final error analysis | Askhat | Use committed validation residual, actual-vs-predicted, vehicle-class, and largest-error artifacts; distinguish observed patterns from explanations. |
| 3 | Final report notebook | Team | Run `project.ipynb` top-to-bottom in the final environment and resolve only real execution issues. |
| 4 | Slides and defense rehearsal | Aktore / all | Use `presentation_outline.md`; present only committed metrics and label the EPA/source limitations. |
| 5 | Contribution record | All | Add only real PR/commit links and reviewer information after review/merge. |

## Remaining limitations and risks

- The selected EPA source lacks vehicle weight, horsepower, aerodynamic characteristics, and real-world driving conditions. Predictions concern EPA laboratory ratings, not a driver's observed fuel use.
- EVs, fuel-cell vehicles, plug-in hybrids, and CNG/bi-fuel vehicles are out of scope because their reported efficiency is not directly comparable to MPG here.
- Exact duplicates are grouped, but related make/model variants may still cross record-level partitions; results may be optimistic for completely unseen models.
- EPA updates the source periodically. Preserve the dataset download date with any reproduced numbers.
