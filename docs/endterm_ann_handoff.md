# Khamza's endterm ANN - results, reproduction and team handoff

## Delivered scope

A train/validation-only PyTorch CPU feedforward regressor, four-candidate
selection, checkpoint restoration, stored preprocessing/weights, row-level
validation predictions, reproducibility checks, an executable experiment
notebook and defense material. The scientific contract is in
[`endterm_protocol.md`](endterm_protocol.md).

Midterm notebooks/results/figures are preserved. No ANN test evaluation, new
split, clustering, PCA, final team comparison or presentation deck is included
in Khamza's scope. Askhat owns the comparison; Aktore owns PCA and final assembly.

## Actual results

Same local raw SHA-256 and seed-42 split: 9,816 train / 2,080 validation /
2,079 previously used holdout records. Raw SHA-256:
`b1ff0e3071c46cdc1c99e0015878b98a01851462d7e447237ce2fc1294ccf303`.

Selected architecture: **40 inputs -> 64 ReLU -> 32 ReLU -> 1 linear output**.
Adam learning rate 0.001; MSE of train-standardised targets; batch size 64.
Maximum 200 epochs; patience 20; best checkpoint **113**, stopped at **133**.

| Split | MAE (MPG) | RMSE (MPG) | R-squared |
|---|---:|---:|---:|
| Train | 0.946014 | 1.311053 | 0.955774 |
| Validation | **1.082715** | **1.522971** | **0.939596** |
| Previously used holdout | Not evaluated for ANN | Not evaluated | Not evaluated |

| Hidden units | Learning rate | Best epoch | Epochs run | Validation MAE | Validation RMSE |
|---|---:|---:|---:|---:|---:|
| 64, 32 | 0.001 | 113 | 133 | **1.082715** | 1.522971 |
| 64, 32 | 0.0003 | 195 | 200 | 1.108980 | 1.557950 |
| 32 | 0.001 | 95 | 115 | 1.206684 | 1.656717 |
| 32 | 0.0003 | 200 | 200 | 1.240870 | 1.690580 |

These are restored-checkpoint results. Architecture/lr selection uses
validation MAE; epoch selection within each trial uses validation MSE.
These are four predeclared configurations, not a search expanded until success.

### Convergence and overfitting

For the selected run, validation standardised MSE decreases from 0.117109
after epoch 1 to 0.059679 at epoch 113. At the last trained epoch, 133, it is
0.060805. Training/validation MAE at the restored checkpoint are 0.946014 and
1.082715: a gap of 0.136701 MPG, consistent with some training-specific fitting.
The curves show improvement followed by a noisy plateau rather than divergence.
They do not establish performance on truly unseen vehicle designs.

Restoring the checkpoint changes final-epoch validation MAE from 1.086470 to
1.082715 MPG. Patience stops this run 67 epochs before the cap. This is a
within-run observation, not an ablation measuring the causal benefit of early
stopping; no no-early-stopping run was performed. Both slower-lr trials reached
200 epochs, so their performance under a larger budget remains unknown.
The deeper network has lower validation MAE at each declared learning rate.

The historical midterm Decision Tree validation MAE is 1.059358, slightly below
ANN's 1.082715. ANN's RMSE is below that historical tree's 1.690783, but MAE is
primary. Do not claim an ANN win or statistical significance. Historical
classical artifacts have no recorded snapshot hash, so Askhat must verify or
reproduce them under the endterm contract before the final comparison.

## Installation and execution

Run commands from the repository root. The tested environment is **Python
3.12.3, Windows x86-64, PyTorch 2.15.0+cpu**. Relevant package versions are pinned
in [`requirements-endterm.txt`](../requirements-endterm.txt) and recorded in
the ANN metrics/config. The pin file lists direct analysis/notebook packages;
it is not a complete transitive-dependency or OS image lock.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-endterm.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ann]" --no-deps

# Three epochs on 512 actual train / 128 actual validation rows; no files written.
.\.venv\Scripts\python.exe -m vehicle_efficiency.neural_network --smoke

# Full four-candidate experiment, train/validation only.
.\.venv\Scripts\python.exe -m vehicle_efficiency.neural_network

# Validate saved model, snapshot, split, row predictions and metrics without tuning.
.\.venv\Scripts\python.exe -m vehicle_efficiency.neural_network --verify
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

If `.venv` already exists, retain it and install the required packages there.
Do not delete an existing environment. For a flexible installation rather than
the tested pins, first install CPU torch from
`https://download.pytorch.org/whl/cpu`, then `pip install -e ".[dev,ann]"`.
Using different versions may change exact predictions.

The local raw CSV must match the manifest. It is Git-ignored and not distributed
with this branch. Obtain the original real snapshot from the team; the current
EPA download can differ. ANN commands never download or replace raw data.

### Notebook

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name vehicle-efficiency-ann --display-name "Vehicle Efficiency (ANN)"
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/endterm/03_ann.ipynb
```

Alternatively open it in Jupyter and select the same environment, then Restart
Kernel and Run All. The notebook's full run retrains all four configurations
and replaces only its own ANN endterm artifacts. The saved notebook contains
executed tables and two embedded charts; it can be read without the CSV/kernel.
Root discovery supports launching from the root or `notebooks/endterm/`.

## Interface for Askhat

```python
import pandas as pd
from vehicle_efficiency.neural_network import (
    load_training_data, load_ann, verify_saved_run,
)
from vehicle_efficiency.preprocessing import split_xy
from vehicle_efficiency.evaluation import compute_metrics

train, val, manifest = load_training_data()  # verifies the committed reference
ann = load_ann(expected_manifest=manifest)  # restores weights AND fitted transforms
X_val, y_val = split_xy(val)
prediction = ann.predict(X_val)             # unrounded MPG, numpy vector
metrics = compute_metrics(y_val, prediction)
print(verify_saved_run())

# Align by unique row_id when combining with classical-model predictions.
ann_rows = pd.DataFrame({
    "row_id": val.row_id.to_numpy(),
    "y_true": y_val.to_numpy(), "y_pred": prediction,
})
```

For programmatic training, `train_ann(X_train,y_train,X_val,y_val,config)`
returns `(ANNModel, history)`. `select_ann(...)` returns `(model,history,search)`
for the fixed grid. Neither accepts a test set. `run_experiment()` returns
`(model,history,search,metrics)` and writes the designated ANN artifacts.

Do not call training again just to include ANN in a comparison. Freeze all
models first, then load this bundle. `ANNModel.predict` selects the exact raw
feature whitelist; extra columns such as row_id/MPG cannot become inputs.
Within the agreed final stage Askhat can pass another shared raw-feature frame
to `predict`; no test-scoring call is included in this handoff or notebook.
The test is a **previously used midterm holdout**, not a fresh endterm test.

Before combining artifacts, verify raw hash, feature contract, split sizes,
ordered split-ID hashes and assignment hash. Use
`dataset_manifest(df,make_splits(df,seed=42))` and
`verify_manifest(reference,actual)` if loading data outside `load_training_data`.
Join by row_id with one-to-one validation; check both complete ID coverage and
unchanged y_true. Do not silently accept mismatched manifests.

Bundle files are coupled by SHA-256 checksums in `ann_config.json`.
`torch.load` uses `weights_only=True` and CPU placement; sklearn objects use
joblib and must come from a trusted team artifact. No preprocessing refit occurs
on load or predict. Keep the bundle on the tested sklearn/PyTorch versions.

## Files for Aktore

| File | Purpose |
|---|---|
| `notebooks/endterm/03_ann.ipynb` | Executed experiment, interpretation, integration example |
| `reports/figures/endterm/ann_learning_curves.png` | Train/validation MSE and MAE by epoch |
| `reports/figures/endterm/ann_search_validation.png` | Four-candidate validation comparison |
| `reports/results/endterm/ann_search.csv` | Full parameter/checkpoint/metric table |
| `reports/results/endterm/ann_metrics.json` | Train/validation metrics and provenance |
| `reports/results/endterm/ann_history.csv` | All candidate histories, checkpoint/selected flags |
| `reports/results/endterm/ann_predictions_validation.csv` | 2,080 unique IDs with true/predicted MPG |
| `reports/results/endterm/ann_dataset_manifest.json` | Shared snapshot and split reference |
| `reports/results/endterm/ann_config.json` | Architecture, settings, metadata, bundle checksums |
| `reports/results/endterm/ann_model.pt` | Selected CPU state_dict |
| `reports/results/endterm/ann_preprocessing.joblib` | Fitted input pipeline and target scaler |

### English report text

We extended the EPA fuel-efficiency regression project with a CPU feedforward
ANN using the same local dataset snapshot and seed-42 duplicate-grouped split
as the existing workflow. Ten raw specifications and the inherited
displacement-per-cylinder feature were transformed into 40 inputs. Numeric
median imputation and standardisation, categorical imputation and one-hot
encoding, and target standardisation were fitted on training rows only.
Unknown validation categories were ignored by the fitted encoder.

The predeclared search crossed one or two hidden layers, (32) and (64,32), with
Adam learning rates 0.001 and 0.0003. ReLU hidden activations and a single linear
output were used with mean squared error, batch size 64, a 200-epoch cap and
early-stopping patience 20. Within each trial the minimum-validation-loss
checkpoint was restored; final configuration selection used validation MAE.

The selected 40-64-32-1 network with learning rate 0.001 restored epoch 113 after
stopping at epoch 133. On 2,080 validation records it achieved MAE 1.082715 MPG,
RMSE 1.522971 MPG and R-squared 0.939596, compared with training MAE 0.946014 MPG.
Learning curves show substantial initial improvement and a noisy plateau with
some train/validation separation. The ANN does not establish a primary-metric
win over the historical Decision Tree; the final verified comparison is
reported separately by the evaluation owner.

Validation was reused for checkpoint and hyperparameter selection, so these
scores are not independent estimates. No ANN test predictions or test metrics
were computed during development. The existing test partition had already
been used in midterm and must be described as a previously used holdout.
Missing vehicle weight, horsepower and hybrid-system detail, related models
across splits, and the difference between EPA ratings and real-world fuel use
limit the conclusions. Results are from one seed and one scoped snapshot;
statistical superiority or transfer to new designs is not established.

## Объяснение Хамзе

Вход ANN — те же характеристики машины, что у классических моделей. После
one-hot получается 40 чисел. Два скрытых слоя, 64 и 32 нейрона, учатся находить
нелинейные зависимости. Последний нейрон выдаёт одно число: после обратного
scaling это прогноз combined MPG.

Мы проверили четыре варианта, заранее заданные в protocol. Победил 64/32 с
learning rate 0.001. Ошибка на validation в среднем около **1.08 MPG**; это
ошибка EPA rating, а не литры на 100 км и не гарантия расхода на дороге.

На train ошибка 0.95 MPG: ниже, чем на validation 1.08. Значит, модель частично
подстраивается под обучение. При этом кривые не расходятся резко. После лучшей
эпохи 113 не было улучшения validation loss 20 эпох; обучение закончилось на
133, а веса вернулись к 113. Выбираем checkpoint по MSE, конфигурацию по MAE —
это разные уровни выбора, оба используют только validation.

ANN чуть уступает старому дереву по MAE (1.083 против исторических 1.059).
Не надо объявлять победу по RMSE, если основной metric заранее выбрали MAE.
Качество ANN хорошее для этой validation, но нейросеть не обязана превосходить
дерево на таблице. Асхат проверит общее сравнение на том же snapshot и строках.

## Content for three slides (English)

### Slide 1 - Shared preprocessing and evaluation protocol

- Real EPA data; fixed local SHA-256; original seed-42 split.
- Train 9,816 / validation 2,080 / previously used holdout 2,079.
- Same raw features plus displacement per cylinder; 40 encoded inputs.
- Input and target transforms fit only on train.
- MAE primary (MPG), RMSE secondary (MPG), R-squared unitless.
- Validation selects ANN; no ANN test evaluation in development.

Speaker note (~60 seconds): describe what the model predicts, separate MPG from
real driving consumption, and explain that scaling uses training statistics.
Disclose prior midterm holdout use. Do not call it an untouched endterm test.

### Slide 2 - Feedforward ANN and bounded search

- 40 -> 64 ReLU -> 32 ReLU -> 1 linear output.
- Adam; selected lr=0.001; MSE on standardised target; batch 64.
- Four candidates: (32) / (64,32) x 0.001 / 0.0003.
- Max 200 epochs; patience 20; restore minimum validation-loss checkpoint.
- Rank restored candidates by validation MAE; Python/NumPy/torch seed 42.

Speaker note (~75 seconds): explain nonlinearity, unrestricted regression
output, gradient-based optimisation and why the search budget was kept small.
The source code/network repr is available in the notebook; avoid code dumps.

### Slide 3 - Learning and validation results

Use `ann_learning_curves.png` and a compact train/validation metrics table.

- Best epoch 113; stopped at 133.
- Train MAE 0.946; validation MAE **1.083 MPG**.
- Validation RMSE **1.523 MPG**; R-squared **0.940**.
- Some train/validation gap; plateau rather than divergent training.
- No claim that ANN wins primary MAE; final comparison is Askhat's section.
- Selection-biased validation; previously used holdout; missing features.

Speaker note (~90 seconds): interpret the curves and metrics, describe the
checkpoint rather than simply saying more epochs are better, and be candid
that the old tree's MAE is slightly lower. No significance claim.

## Вопросы для защиты

**Почему ReLU?** Даёт нелинейность и простой градиент; без нелинейных активаций
последовательность линейных слоёв оставалась бы линейным преобразованием.

**Почему линейный выход?** Это regression с одним непрерывным значением.
Sigmoid ограничил бы выход интервалом 0..1, softmax нужен для распределения
вероятностей классов. Standardised MPG может быть отрицательным; это нормально.

**Зачем scaling?** Год и объём двигателя имеют разные масштабы. StandardScaler
выравнивает numeric inputs для оптимизации; target scaler упрощает масштаб
градиентов. Оба fit только на train, обратно переводим prediction в MPG.

**Чем loss отличается от MAE?** Loss здесь — средний квадрат ошибки standardised
target: его минимизирует Adam, он сильнее штрафует большие промахи. MAE —
средняя абсолютная ошибка после inverse transform, в MPG, понятная для отчёта
и выбора итоговой конфигурации. Validation loss выбирает epoch внутри trial.

**Как работает early stopping?** После каждой эпохи считаем validation MSE.
Если он строго меньше предыдущего лучшего, копируем state_dict и обнуляем
счётчик ожидания. Через 20 эпох без улучшения останавливаемся и возвращаем
лучшую копию весов, а не последние веса.

**Validation тоже участвует в обучении?** Градиенты и fit transforms на нём не
считаются. Но он влияет на остановку и выбор конфигурации, поэтому его score
оптимистичен и не является независимой оценкой generalisation.

**Почему ANN может быть хуже дерева?** Деревья хорошо работают с табличными
условиями и локальными закономерностями; здесь также могут повторяться
близкие варианты машин. Это возможные объяснения, а не доказанная причина.
По одной маленькой разнице нельзя заявлять статистическое превосходство.

**Зачем фиксировать hash, если seed одинаковый?** Seed повторяет выбор индексов
при том же содержимом и порядке данных. Другой CSV может дать другие строки,
категории и splits при том же seed. Поэтому нужны hash CSV и hash назначений.

**Почему не CNN/NLP?** В этом dataset обычная таблица характеристик без изображений
и содержательного текстового корпуса; feedforward ANN соответствует задаче.

## Validation and remaining limits

- Existing baseline: 18 tests passed before changes.
- Expanded suite: 27 tests, including real-data ANN training, leakage checks,
  unknown categories, target inverse scaling, checkpoint restoration,
  patience, seed repeatability, bundle roundtrip and manifest rejection.
- Smoke training: three epochs on 512 real train / 128 real validation rows.
- Full four-candidate run completed twice: once through the CLI and once in a
  fresh notebook kernel. All trial histories, search rows and validation
  predictions matched exactly. All eight code cells ran; both charts are
  embedded. The rendered notebook summary, results table, learning curves and
  takeaways were visually inspected, along with both exported figures.
- The delivery record is in `reports/results/endterm/ann_validation_checks.json`.
- `--verify` recomputes all 2,080 validation predictions and metrics from the
  stored bundle. `pip check` checks environment dependency consistency.
- File-hash audit checks that all pre-existing tracked files except the
  intentionally updated pyproject, and the local raw CSV, remain unchanged.

The final test suite passed **27/27**, `pip check` found no broken requirements,
and independent NumPy calculations reproduced all three validation metrics.
The preservation audit checked 39 pre-existing files (including raw CSV): only
`pyproject.toml` changed. Existing `models.py`, `evaluation.py`, `data.py`,
`preprocessing.py`, README, tests and midterm notebooks/artifacts are unchanged.

Single seed, one validation split and four candidates do not quantify run-to-run
uncertainty. No ANN grouped CV, no early-stopping ablation and no fresh external
test were performed. No new split was introduced. CPU deterministic flags and
fixed seeds do not guarantee identical outputs on different platforms/releases.
See [PyTorch reproducibility](https://docs.pytorch.org/docs/2.15/notes/randomness.html)
and [state_dict persistence](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
