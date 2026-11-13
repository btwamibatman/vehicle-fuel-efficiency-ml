"""Show the stored extended models on one real validation car, without fitting."""
import json
import joblib
from vehicle_efficiency.feature_extension import (
    RESULTS, VARIANTS, input_columns, load_extension_data, variant_manifest,
)
from vehicle_efficiency.neural_network import load_ann


def main():
    _, validation, manifest = load_extension_data()
    variant = VARIANTS[-1]
    row = validation.iloc[:1]
    metadata = json.loads((RESULTS / 'experiment_metadata.json').read_text())
    ann = load_ann(RESULTS / f'ann_{variant}', expected_manifest=variant_manifest(manifest, variant))
    classical = joblib.load(RESULTS / 'classical_models.joblib')
    name = metadata['best_classical_full']
    print('Реальный validation-пример; нового обучения и test predictions нет.')
    print(row[['make', 'model', 'model_year', 'transmission', 'transmission_detail', 'start_stop']].to_string(index=False))
    print('Start-stop: 1 = есть, 0 = нет. Подробная коробка заменяет грубую категорию.')
    print(f'Входов сохранённой ANN после preprocessing: {ann.network[0].in_features}')
    print(f'ANN prediction: {ann.predict(row)[0]:.6f} MPG')
    print(f'{name} prediction: {classical[(variant, name)].predict(row[input_columns(variant)])[0]:.6f} MPG')
    print(f'EPA actual: {row.mpg.iloc[0]:.0f} MPG')
    print('Один пример не заменяет метрики всех 2,080 validation rows.')
    print('Итоговый ANN validation MAE: 0.805409 MPG. Независимого нового test score нет.')


if __name__ == '__main__':
    main()
