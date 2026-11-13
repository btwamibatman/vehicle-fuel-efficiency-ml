# Endterm team tasks

This record separates completed technical work from integration work that still
requires an owner artifact. All contributors use the same SHA-256-identified
EPA snapshot and seed-42 duplicate-grouped split in `docs/endterm_protocol.md`.

| Member | Responsibility | Status on 2026-11-13 | Required handoff |
|---|---|---|---|
| Aktore | Training-only PCA of `cylinders` and `displacement`; final notebook, documentation and deck | PCA complete; final assembly pending upstream artifacts | `pca_*.csv`, figures, PCA module/notebook |
| Khamza | Shared protocol and reproducibility; feedforward ANN | Complete and merged | `ann_*` bundle, validation predictions, curves |
| Askhat | K-Means, classical-model refinement, validated comparison and error analysis | Pending | `clustering_*`, `model_comparison_*`, `validation_predictions.csv`, error-analysis tables/figures |

## Integration gate

Aktore can complete PCA independently. The following items must not be filled
with historical or invented values: the PCA cluster overlay, cluster profiles,
final comparison, error analysis, and the corresponding presentation slides.
They require Askhat's files under `reports/results/endterm/` and
`reports/figures/endterm/`.

Before integrating, Askhat must verify the raw snapshot and split against
`ann_dataset_manifest.json`, use unique `row_id` values, and provide the
assignment file with exactly the 9,816 training IDs. Aktore's PCA command then
creates the overlay with:

```powershell
py -3.12 -m vehicle_efficiency.dimensionality_reduction --overlay reports/results/endterm/clustering_assignments_train.csv
```
