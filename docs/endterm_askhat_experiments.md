# Endterm experiments: clustering and supervised models

Before running: K-Means uses only cylinders and displacement. Compare k=2–8 by inertia, a seed-42 sample silhouette, cluster sizes and interpretability; k=3 is a compact candidate to be assessed against those diagnostics. The groups describe engine characteristics, not complete vehicle types. MPG is excluded from fitting and k selection.

For classical refinement, the inherited unrestricted tree may overfit, so compare nearby depths and leaf sizes; three-neighbor KNN may benefit from neighboring k and distance weighting; Ridge may reduce coefficient instability compared with ordinary least squares. The declared grids are tree depth [12, 16, None], leaf [1, 2, 5]; KNN neighbors [2, 3, 4, 5], weights [uniform, distance], distance p [1, 2]; Ridge alpha [0.01, 0.1, 1, 10], plus the original linear model. All use the same training-only duplicate-grouped five folds. Model choice uses validation MAE. The historical test was used in the midterm and is not an untouched endterm holdout.

## Clustering findings

For k=2, 3 and 4, sampled silhouette is 0.663, 0.727 and 0.753; inertia drops from 6525 to 2517 to 1381. We retain the predeclared compact k=3 choice: the large improvement from two groups yields groups of 4,581, 3,177 and 2,058 rows, while k=4 introduces a 205-row group. The latter is a reasonable alternative, not proof that three is objectively optimal. Cluster centers are approximately (3.97 cylinders, 2.04 L), (6.00, 3.30 L), and (8.46, 5.22 L). Their descriptive mean MPG values are 27.64, 21.02 and 16.72, respectively. The observed differences are associations, not evidence of causal effects. K-Means creates a partition whether or not sharply separated natural groups exist.

## Notes for the defense / Для защиты

- **Why k=3?** The elbow improves strongly from two to three and the sampled silhouette is 0.727. Four yields 0.753 but creates a smaller group. This is an interpretability tradeoff, not a unique optimum. / Три группы понятны и достаточно крупные; четыре тоже допустимо, но создаёт меньшую группу.
- **Silhouette:** measures, for each point, relative closeness to its own cluster compared with another cluster; near 1 means better separation, around 0 means overlap, negative suggests a poor assignment. / Показатель показывает, насколько объект ближе к своей группе, чем к соседней.
- **Scaling:** K-Means uses distances, so unequal units and spreads can dominate. The scaler and imputer learn only from train. / Масштабирование делает вклады объёма и числа цилиндров сопоставимыми.
- **Leakage:** preprocessing and hyperparameter tuning are fitted inside each training fold; validation chooses the model. The test was previously used in the midterm, so it is not an independent new holdout. / Предобработка обучается внутри каждого train-фолда; validation выбирает модель; test уже использовался.
- **Why might the ANN lose?** A tree can exploit thresholds and interactions efficiently for structured data, while ANN performance depends on optimization, architecture, dataset size and regularization. This is a possible explanation, not a causal finding from the comparison. / Дерево может хорошо использовать пороги и взаимодействия в табличных данных; нейросеть зависит от настройки и обучения.

## Four-slide outline

1. **Tuning:** shared train/validation split, duplicate-grouped five-fold CV, tree/KNN/Ridge search spaces and validation MAE. Explain how the endterm results compare with midterm.
2. **Clustering:** inputs, scaling, elbow/silhouette, chosen k and engine profiles; note descriptive MPG and no causal claim.
3. **Model comparison:** validation MAE, RMSE and R² for all candidates and ANN. Identify winner using the predeclared primary metric and distinguish ANN's validation-based selection from classical CV.
4. **Error analysis:** actual vs predicted, residual sign, hybrid and class errors with counts, limitations of EPA ratings and repeated/related vehicle variants; identify the holdout's prior use.

## Results and interpretation

On the verified shared validation split, the tree selected by grouped CV (`max_depth=None`, `min_samples_leaf=2`) has MAE 1.051 MPG, RMSE 1.636 MPG and R² 0.930. The ANN has MAE 1.083, RMSE 1.523 and R² 0.940. KNN has MAE 1.089, RMSE 1.672 and R² 0.927; linear regression has MAE 2.019 and Ridge 2.021. The tree wins the declared primary metric; the ANN's better RMSE indicates smaller squared errors overall, but does not change the selection rule. The earlier midterm report recorded tree validation MAE 1.059 with minimum leaf 1. That historical comparison provides context; its provenance cannot independently establish identical raw bytes and software, so the small difference is not proof of a robust improvement.

The tree's training MAE is 0.686 versus 1.051 on validation, indicating a generalization gap. On validation hybrids (241 rows), its MAE is 1.827 versus 0.949 on non-hybrids (1,839 rows); the ANN's corresponding values are 1.667 and 1.006. These differences do not establish a cause, and vehicle-class results with small sample counts need caution. The source does not include vehicle weight, horsepower, full hybrid technology or actual driving conditions. Similar vehicle variants may appear across partitions.

After configurations were fixed, the previously used midterm holdout was evaluated once: tree MAE 1.074, RMSE 1.664, R² 0.928; ANN MAE 1.078, RMSE 1.473, R² 0.944. This holdout has a known history of prior use and is not an independent endterm confirmation. Scores did not drive another model selection.

## Reproduction and handoff

Use the exact raw snapshot identified in `docs/endterm_protocol.md` and install the endterm dependencies. From the repository root, `python -m vehicle_efficiency.clustering` verifies the manifest and regenerates training-only clustering outputs. `python -m vehicle_efficiency.endterm_compare` performs training CV and validation plus ANN verification; it intentionally refuses to run when a final holdout table exists, to avoid tuning after reviewing the holdout. The `--test` stage also refuses to overwrite an existing table. These safeguards are workflow checks, not access controls. Both notebooks load the committed artifacts and execute in order.

For Aktore: `reports/results/endterm/clustering_assignments_train.csv` contains one `row_id,cluster` per training row. Join one-to-one on `row_id`; use the same `cylinders` and `displacement` features, training split and snapshot. Preserve cluster labels rather than fitting K-Means again on PCA coordinates.

On this macOS/ARM run, the ANN bundle checksums, manifest, validation IDs and targets match. Recomputed validation predictions agree with the Windows-saved predictions within 0.000008 MPG (maximum absolute difference 0.00000763), below the comparison's declared 0.0001 absolute tolerance. Khamza's separate `python -m vehicle_efficiency.neural_network --verify` uses an absolute tolerance of 1e-10 and therefore fails across these environments; it should not be reported as passing here. The differences do not change the displayed metrics or validation ranking.
