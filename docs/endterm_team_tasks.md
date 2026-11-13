# Endterm team tasks

This record separates completed technical work from integration work that still
requires an owner artifact. All contributors use the same SHA-256-identified
EPA snapshot and seed-42 duplicate-grouped split in `docs/endterm_protocol.md`.

| Member | Responsibility | Status on 2026-11-13 | Required handoff |
|---|---|---|---|
| Aktore | Training-only PCA of `cylinders` and `displacement`; final notebook, documentation and deck | PCA and report integrated; deck still to prepare | `pca_*.csv`, figures, PCA module/notebook |
| Khamza | Shared protocol and reproducibility; feedforward ANN | Complete and merged | `ann_*` bundle, validation predictions, curves |
| Askhat | K-Means, classical-model refinement, validated comparison and error analysis | Complete and merged | `clustering_*`, `model_comparison_*`, `validation_predictions.csv`, error-analysis tables/figures |

## Integration gate

Aktore completed the strict PCA/K-Means row-ID join after importing Askhat's
committed artifacts. The cluster overlay is
`reports/figures/endterm/pca_clusters_train.png`. Final slides and prose must
continue to use the committed artifacts rather than historical midterm values.

For any future reproduction, verify the raw snapshot and split against
`ann_dataset_manifest.json`, use unique `row_id` values, and provide the
assignment file with exactly the 9,816 training IDs. The PCA command then
creates the overlay with:

```powershell
py -3.12 -m vehicle_efficiency.dimensionality_reduction --overlay reports/results/endterm/clustering_assignments_train.csv
```
