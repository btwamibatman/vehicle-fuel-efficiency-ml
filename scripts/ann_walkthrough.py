r"""Explain the stored ANN on one real validation car; no fit or test prediction.

From the repository root:
    .\.venv\Scripts\python.exe -X utf8 scripts/ann_walkthrough.py
"""
import json

from vehicle_efficiency.neural_network import (
    RESULTS_DIR, load_ann, load_training_data, verify_saved_run,
)
from vehicle_efficiency.preprocessing import split_xy


def main() -> None:
    _, validation, manifest = load_training_data()
    model = load_ann(expected_manifest=manifest)
    row = validation.iloc[:1]
    X, y = split_xy(row)
    prep = model.preprocessor.named_steps["prep"]
    numeric_count = len(prep.transformers_[0][2])
    encoder = prep.named_transformers_["cat"].named_steps["onehot"]
    category_counts = {column: len(values) for column, values in
                       zip(prep.transformers_[1][2], encoder.categories_)}
    predicted = float(model.predict(X)[0])
    actual = float(y.iloc[0])
    mean = float(model.target_scaler.mean_[0])
    sd = float(model.target_scaler.scale_[0])
    metrics = json.loads((RESULTS_DIR / "ann_metrics.json").read_text(encoding="utf-8"))

    print("1. Что прогнозируем")
    print("EPA combined US MPG: сколько миль машина проходит на одном US gallon.")
    print("Чем выше MPG, тем выше топливная экономичность по рейтингу EPA.\n")
    print("2. Реальный пример из validation, а не из test")
    print(row[["row_id", "make", "model", "model_year"]].to_string(index=False))
    print("Входные признаки:")
    print(X.to_string(index=False))
    ratio = float(X.displacement.iloc[0] / X.cylinders.iloc[0])
    print(f"displacement_per_cylinder = {ratio:.3f} литра на цилиндр.")
    print("mpg, row_id, make и model не передаются как признаки сети.\n")
    print("3. Как получаются входы сети")
    print(f"Числовых признаков с engineered feature: {numeric_count}.")
    print(f"One-hot категорий, выученных только на train: {category_counts}.")
    print(f"Итого: {numeric_count} + {sum(category_counts.values())} = "
          f"{model.network[0].in_features} входов.")
    print("Imputer, scaler и encoder уже обучены на train; здесь только transform.\n")
    print("4. Какая сеть сохранена")
    print(model.network)
    print(f"Обучаемых параметров: {sum(p.numel() for p in model.network.parameters())}.")
    print(f"Learning rate: {model.config.learning_rate}; batch: {model.config.batch_size}; "
          f"patience: {model.config.patience}; restored epoch: {model.best_epoch}.\n")
    print("5. Как результат возвращается в MPG")
    print(f"Среднее training MPG: {mean:.6f}; стандартное отклонение: {sd:.6f}.")
    scaled = (predicted - mean) / sd
    print(f"Стандартизованный прогноз примерно {scaled:.6f}.")
    print(f"Обратно: {scaled:.6f} * {sd:.6f} + {mean:.6f} = {predicted:.6f} MPG.")
    print(f"В EPA записано: {actual:.0f} MPG; абсолютная ошибка этой строки: "
          f"{abs(predicted - actual):.6f} MPG.\n")
    print("6. Результат на всей validation, а не на одной машине")
    scores = metrics["validation"]
    print(f"MAE = {scores['mae']:.6f} MPG; RMSE = {scores['rmse']:.6f} MPG; "
          f"R-squared = {scores['r2']:.6f}.")
    print("MAE — средняя абсолютная ошибка; это не максимум ошибки и не процент accuracy.")
    print("Epoch выбирался по validation MSE, конфигурация — по validation MAE.\n")
    print("7. Проверка сохранённой модели и всех validation predictions")
    print(json.dumps(verify_saved_run(), ensure_ascii=False, indent=2))
    print("Нового обучения не было. ANN на test не оценивалась.")


if __name__ == "__main__":
    main()
