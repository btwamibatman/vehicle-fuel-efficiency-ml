# Endterm presentation outline

Target: 14 slides, approximately 12 minutes. Claims and values must come from
the committed endterm artifacts.

| Slides | Speaker | Content | Current source |
|---|---|---|---|
| 1–3 | Aktore | question, EPA dataset and training-only EDA | PCA notebook and protocol |
| 4 | Khamza | preprocessing, split and leakage prevention | `endterm_protocol.md` |
| 5–7 | Askhat | classical tuning and K-Means | pending Askhat artifacts |
| 8–9 | Aktore | PCA variance/loadings; PCA scatter with clusters | PCA artifacts; overlay pending assignments |
| 10–11 | Khamza | ANN architecture and learning curves | ANN handoff and figures |
| 12–13 | Askhat | verified comparison and validation error analysis | pending Askhat artifacts |
| 14 | Aktore | conclusions, limitations and contributions | complete after integration |

## Aktore's PCA notes

PCA uses only engine cylinders and displacement, after median imputation and
standardisation learned on training data. PC1 retains 96.06% of their variance;
PC1 plus PC2 retains 100%. The similar PC1 loadings for both inputs show that
PC1 broadly follows jointly larger or smaller engines. PCA maximises feature
variance, not MPG prediction accuracy, and does not establish natural clusters.
