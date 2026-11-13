# Endterm presentation outline

Target: 14 slides, approximately 12 minutes. Claims and values must come from
the committed endterm artifacts.

| Slides | Speaker | Content | Current source |
|---|---|---|---|
| 1–3 | Aktore | question, EPA dataset and training-only EDA | `00_eda.ipynb`, final report and protocol |
| 4 | Khamza | preprocessing, fixed split and added features | original protocol and `feature_extension_results_ru.md` |
| 5–7 | Askhat | classical tuning and K-Means | `supervised_tuning.csv`, comparison and error-analysis artifacts |
| 8–9 | Aktore | PCA variance/loadings; PCA scatter with clusters | PCA artifacts and `pca_clusters_train.png` |
| 10–11 | Khamza | original versus 72-input ANN; new learning curves | feature-extension ablation and ANN figures |
| 12–13 | Askhat | new six-model comparison, validation error analysis and original holdout caveat | feature-extension tables; old test scores apply only to original models |
| 14 | Aktore | conclusions, limitations and contributions | final report and README |

## Aktore's PCA notes

PCA uses only engine cylinders and displacement, after median imputation and
standardisation learned on training data. PC1 retains 96.06% of their variance;
PC1 plus PC2 retains 100%. The similar PC1 loadings for both inputs show that
PC1 broadly follows jointly larger or smaller engines. PCA maximises feature
variance, not MPG prediction accuracy, and does not establish natural clusters.
