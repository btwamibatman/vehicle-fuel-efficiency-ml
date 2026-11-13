# Vehicle Fuel Efficiency - 15-slide presentation skeleton

This is the content specification for a 12-minute endterm defense. It uses the
feature-extension results merged in `main` on 2026-11-14. Create the final
PowerPoint as a separate file; preserve the graphs and numbers below.

## 1. Title and research question - Aktore, 25 seconds

**Title:** Predicting EPA Vehicle Fuel Efficiency

**Question:** How much do start-stop and gearbox details improve prediction of
EPA combined MPG?

**Footer:** Aktore / Khamza / Askhat, Endterm Course Project

**Visual:** clean title treatment; no chart.

**Note:** Define MPG as the EPA combined laboratory rating, not an individual
driver's real fuel consumption.

## 2. Dataset and scope - Aktore, 35 seconds

**On-slide facts:**

- FuelEconomy.gov / EPA snapshot; model years 2015 onward.
- 13,975 prepared vehicle records.
- 9,816 train / 2,080 validation / 2,079 previously used midterm holdout.
- EV, fuel-cell and plug-in-hybrid entries excluded from this MPG comparison.

**Visual:** three large dataset counts.

**Note:** The holdout was used in midterm, so new feature-extension results are
reported on validation only.

## 3. Training-only EDA - Aktore, 40 seconds

**Message:** Larger displacement and more cylinders generally align with lower
MPG, but vehicles with similar engines can still differ materially.

**Visual:** `reports/figures/endterm/eda_displacement_vs_mpg.png`.

**Note:** This motivates engine features but does not claim engine size alone
causes fuel efficiency.

## 4. Leakage prevention and fixed protocol - Khamza, 45 seconds

**On-slide facts:**

- Same raw CSV, filters, seed-42 split and row IDs for all variants.
- Exact duplicates grouped before splitting.
- Imputers, encoders and scalers fit on training rows only.
- MAE is primary, RMSE and R-squared are secondary.

**Visual:** simple train -> validation workflow.

**Note:** The original holdout is historical context, not a fresh independent
endterm test.

## 5. Controlled feature extension - Khamza, 45 seconds

**Title:** Two added EPA fields

**On-slide facts:**

- `start_stop`: EPA yes/no engine automatic shutoff while stopped.
- `transmission_detail`: full cleaned gearbox label, replacing broad
  automatic/manual/other.
- ANN input dimension: 40 -> 41 -> 72 after preprocessing.

**Visual:** three-step feature-set progression.

**Note:** No external joins or synthetic horsepower/weight variables were
added. Classical hyperparameters stayed fixed.

## 6. Classical-model protocol - Askhat, 40 seconds

**On-slide facts:**

- Baseline, Linear Regression, Ridge, Decision Tree and KNN.
- GroupKFold(5) inside training rows.
- Fixed prior settings across the three feature variants.
- Classical CV supports comparison; validation selects final performance.

**Visual:** compact model-family list with validation criterion.

## 7. K-Means engine groups - Askhat, 45 seconds

**On-slide table:**

| Group | n | Mean cylinders | Mean litres | Mean MPG |
|---|---:|---:|---:|---:|
| Smaller engines | 4,581 | 3.97 | 2.04 | 27.64 |
| Medium engines | 3,177 | 6.00 | 3.30 | 21.02 |
| Larger engines | 2,058 | 8.46 | 5.22 | 16.72 |

**Visual:** `reports/figures/endterm/clustering_scatter.png`.

**Note:** k=3 is an interpretable choice. k=4 has slightly higher silhouette
but creates a small 205-row group. Cluster IDs are not quality rankings.

## 8. PCA variance and loadings - Aktore, 45 seconds

**On-slide facts:**

- PCA uses only cylinders and displacement after training-only scaling.
- PC1 retains 96.06% of variance; PC1 + PC2 retain 100%.
- Similar PC1 loadings reflect jointly larger versus smaller engines.

**Visual:** `reports/figures/endterm/pca_explained_variance.png`.

**Note:** PCA maximises feature variance, not prediction accuracy.

## 9. PCA scatter with clusters - Aktore, 35 seconds

**Visual:** `reports/figures/endterm/pca_clusters_train.png`.

**On-slide statement:** The PCA scatter rotates the two engine features and
shows Askhat's fixed K-Means labels. It does not prove natural vehicle classes.

## 10. ANN configuration and learning - Khamza, 45 seconds

**On-slide facts:**

- 72 encoded inputs -> 64 ReLU -> 32 ReLU -> linear MPG output.
- Adam, learning rate 0.001, batch size 64, MSE on training-standardised MPG.
- Best checkpoint: epoch 134; stopped: epoch 154.

**Visual:** `reports/figures/endterm/feature_extension/ann_learning_curves.png`.

**Note:** Validation controls checkpoint selection, so its score is not an
independent estimate.

## 11. Feature ablation - Khamza, 50 seconds

**Visual:** `reports/figures/endterm/feature_extension/feature_ablation.png`.

**On-slide takeaway:** ANN validation MAE: 1.083 original -> 1.010 with
start-stop -> 0.805 with start-stop plus detailed transmission.

**Note:** Same validation rows and fixed settings enable a controlled
comparison, but this remains post-hoc development on one split.

## 12. Final validation comparison - Askhat, 50 seconds

**On-slide table:**

| Model | MAE MPG | RMSE MPG | R-squared |
|---|---:|---:|---:|
| Feedforward ANN | **0.805** | **1.197** | **0.963** |
| KNN | 0.834 | 1.380 | 0.950 |
| Decision Tree | 0.837 | 1.417 | 0.948 |
| Linear Regression | 1.749 | 2.388 | 0.851 |
| Ridge | 1.751 | 2.388 | 0.851 |
| Baseline | 4.662 | 6.197 | approximately 0 |

**Note:** ANN has the lowest validation MAE. KNN and tree lie within 0.032 MPG;
no statistical superiority claim follows from this one split.

## 13. Error analysis - Askhat, 45 seconds

**Visual:** `reports/figures/endterm/feature_extension/evaluation_validation.png`.

**On-slide facts:** ANN MAE: 1.421 MPG for 241 hybrid rows and 0.725 MPG for
1,839 non-hybrid rows.

**Note:** This is descriptive. Weight, horsepower and detailed hybrid system
information are missing, so the subgroup gap does not identify a cause.

## 14. Holdout caveat and limitations - Askhat, 35 seconds

**On-slide facts:**

- Original-model holdout values came from a previously used midterm partition.
- The new feature variants did not receive a new test evaluation.
- Validation informed ANN checkpoint selection and feature comparison.
- EPA lab ratings, missing variables and related make/model variants limit
  generalisation.

**Visual:** small labelled “validation evidence” callout, not a new test table.

## 15. Conclusion and contributions - Aktore, 35 seconds

**On-slide facts:**

- Detailed transmission plus start-stop lowered the best validation MAE to
  0.805 MPG.
- ANN leads this fixed validation comparison; KNN and tree remain close.
- PCA and K-Means give an interpretable engine-focused view of the dataset.
- Aktore: dataset/EDA, PCA, final integration. Khamza: preprocessing and ANN.
  Askhat: tuning, clustering, comparison and error analysis.

**Final line:** Future work needs a prospectively held-out evaluation on newer
vehicles.
