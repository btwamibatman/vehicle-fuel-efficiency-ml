"""Controlled start-stop/transmission ablation on the frozen train/validation split.

    python -m vehicle_efficiency.feature_extension
    python -m vehicle_efficiency.feature_extension --verify

No new download, splitting, tuning grid or test evaluation is performed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_validate

from .data import PROJECT_ROOT, RAW_FILE, _duplicate_groups
from .endterm_compare import SPECS
from .evaluation import compute_metrics
from .neural_network import (
    ANNConfig, file_sha256, library_versions, load_ann, load_training_data,
    save_ann, train_ann,
)
from .preprocessing import ENGINEERED, RAW_FEATURES

VARIANTS = ('original', 'start_stop', 'start_stop_transmission')
PARENT = PROJECT_ROOT / 'reports/results/endterm'
RESULTS = PARENT / 'feature_extension'
FIGURES = PROJECT_ROOT / 'reports/figures/endterm/feature_extension'
PROTOCOL = PROJECT_ROOT / 'docs/feature_extension_protocol.json'


def input_columns(variant: str) -> list[str]:
    """The detailed category replaces, rather than duplicates, coarse transmission."""
    if variant not in VARIANTS:
        raise ValueError(f'Unknown feature variant: {variant}')
    columns = list(RAW_FEATURES)
    if variant != 'original':
        columns.append('start_stop')
    if variant == 'start_stop_transmission':
        columns[columns.index('transmission')] = 'transmission_detail'
    return columns


def add_extension_columns(rows: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Attach source fields by immutable raw row ID; never regroup or reorder rows."""
    if raw.index.has_duplicates or rows.row_id.duplicated().any():
        raise ValueError('Extension attachment requires unique raw row IDs')
    source = raw.loc[rows.row_id]
    out = rows.copy()
    # Unknown/blank values remain missing instead of silently meaning "no".
    flags = source.startStop.astype('string').str.strip().str.upper()
    out['start_stop'] = flags.map({'Y': 1.0, 'N': 0.0}).to_numpy(dtype=float, na_value=np.nan)
    detail = source.trany.astype('string').str.strip().replace('', pd.NA)
    out['transmission_detail'] = detail.astype(object).where(detail.notna(), np.nan).to_numpy()
    return out


def load_extension_data():
    """Verify the original snapshot/splits and expose only train and validation."""
    train, validation, original_manifest = load_training_data()
    raw = pd.read_csv(RAW_FILE, usecols=['startStop', 'trany'], low_memory=False)
    return (add_extension_columns(train, raw),
            add_extension_columns(validation, raw), original_manifest)


def variant_manifest(original: dict, variant: str) -> dict:
    manifest = copy.deepcopy(original)
    manifest['raw_features'] = input_columns(variant)
    manifest['feature_set'] = input_columns(variant) + ENGINEERED
    manifest['feature_variant'] = variant
    return manifest


def build_variant_pipeline(name: str, variant: str, parameters: dict):
    """Reuse the original estimator, feature engineering and fitted-fold design."""
    pipeline = SPECS[name][0]().set_params(**parameters)
    preprocessor = pipeline.named_steps['prep']
    for i, (kind, transformer, columns) in enumerate(preprocessor.transformers):
        columns = list(columns)  # do not mutate shared CATEGORICAL definitions
        if kind == 'num' and variant != 'original':
            columns.append('start_stop')
        if kind == 'cat' and variant == 'start_stop_transmission':
            columns[columns.index('transmission')] = 'transmission_detail'
        preprocessor.transformers[i] = (kind, transformer, columns)
    return pipeline


def protected_hashes() -> dict:
    """Fingerprint the completed parent experiment and raw bytes before/after."""
    paths = [RAW_FILE, *[p for p in PARENT.iterdir() if p.is_file()]]
    return {str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in paths}


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run() -> None:
    if (RESULTS / 'experiment_metadata.json').exists():
        raise RuntimeError('Extension already recorded; use --verify, not post-hoc reselection')
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    before = protected_hashes()
    train, validation, original = load_extension_data()
    if original != protocol['original_manifest']:
        raise ValueError('Source snapshot/split contract differs from the declared experiment')
    RESULTS.mkdir(parents=True, exist_ok=True)
    groups = _duplicate_groups(train)
    rows, predictions, folds, histories, models = [], [], [], [], {}
    historical_differences = []
    old_comparison = pd.read_csv(PARENT / 'model_comparison_validation.csv').set_index('model')

    for variant in VARIANTS:
        columns = input_columns(variant)
        X, y = train[columns], train.mpg
        Xv, yv = validation[columns], validation.mpg
        for name in SPECS:
            parameters = protocol['classical_parameters'][name]
            pipeline = build_variant_pipeline(name, variant, parameters)
            cv = cross_validate(pipeline, X, y, groups=groups, cv=GroupKFold(5),
                                scoring='neg_mean_absolute_error', n_jobs=1,
                                error_score='raise')
            scores = -cv['test_score']
            for fold, score in enumerate(scores, 1):
                folds.append({'variant': variant, 'model': name, 'fold': fold, 'mae': score})
            pipeline.fit(X, y)
            predicted = pipeline.predict(Xv)
            metrics = compute_metrics(yv, predicted)
            rows.append({'variant': variant, 'model': name,
                         'parameters': json.dumps(parameters, sort_keys=True),
                         'cv_mae': scores.mean(), 'cv_std': scores.std(),
                         'train_mae': compute_metrics(y, pipeline.predict(X))['mae'],
                         **{f'validation_{k}': v for k, v in metrics.items()}})
            predictions.append(pd.DataFrame({'variant': variant, 'model': name,
                                             'row_id': validation.row_id.to_numpy(),
                                             'y_true': yv.to_numpy(), 'y_pred': predicted}))
            models[(variant, name)] = pipeline
            if variant == 'original':
                if name == 'KNN':
                    historical = pd.read_csv(PARENT / 'validation_predictions.csv')
                    historical = historical[historical.model == name].set_index('row_id').loc[validation.row_id]
                    np.testing.assert_array_equal(historical.y_true, yv)
                    changed = ~np.isclose(predicted, historical.y_pred.to_numpy(), rtol=0, atol=1e-10)
                    distances = pipeline.named_steps['model'].kneighbors(
                        pipeline[:-1].transform(Xv), n_neighbors=4)[0]
                    boundary_ties = np.isclose(distances[:, 2], distances[:, 3], rtol=0, atol=1e-10)
                    if not boundary_ties[changed].all():
                        raise ValueError('Historical KNN mismatch is not explained by boundary ties')
                    _write_json(RESULTS / 'knn_historical_ties.json', {
                        'changed_prediction_rows': int(changed.sum()),
                        'all_changed_rows_have_equal_third_fourth_distance': True,
                        'total_boundary_tie_rows': int(boundary_ties.sum()),
                        'interpretation': 'Equal-distance neighbor selection differs; compare variants against the rerun current-environment control'})
                for key, value in metrics.items():
                    if name != 'KNN':
                        np.testing.assert_allclose(value, old_comparison.loc[name, f'validation_{key}'],
                                                   rtol=0, atol=protocol['historical_classical_metric_atol'])
                    historical_differences.append({'model': name, 'metric': key,
                        'historical': old_comparison.loc[name, f'validation_{key}'],
                        'current_control': value,
                        'difference': value - old_comparison.loc[name, f'validation_{key}']})
            print(variant, name, f"validation MAE={metrics['mae']:.6f}", flush=True)

        # Architecture/lr are frozen at the original selected values. This is
        # one deterministic training run per feature variant, not a new grid.
        config = ANNConfig(**{**protocol['ann_parameters'],
                              'hidden_sizes': tuple(protocol['ann_parameters']['hidden_sizes'])})
        prep = build_variant_pipeline('Linear Regression', variant, {})[:-1]
        ann, history = train_ann(X, y, Xv, yv, config, preprocessor=prep)
        ann.metadata.update({
            'dataset': variant_manifest(original, variant), 'versions': library_versions(),
            'candidate_selection': 'Frozen architecture and learning rate; checkpoint by validation MSE',
            'feature_variant': variant, 'test_evaluated': False,
        })
        directory = RESULTS / f'ann_{variant}'
        save_ann(ann, directory)
        predicted = ann.predict(Xv)
        metrics = compute_metrics(yv, predicted)
        rows.append({'variant': variant, 'model': 'Feedforward ANN',
                     'parameters': json.dumps(protocol['ann_parameters'], sort_keys=True),
                     'cv_mae': np.nan, 'cv_std': np.nan,
                     'train_mae': compute_metrics(y, ann.predict(X))['mae'],
                     **{f'validation_{k}': v for k, v in metrics.items()},
                     'best_epoch': ann.best_epoch, 'epochs_run': len(history),
                     'input_dim': ann.network[0].in_features})
        predictions.append(pd.DataFrame({'variant': variant, 'model': 'Feedforward ANN',
                                         'row_id': validation.row_id.to_numpy(),
                                         'y_true': yv.to_numpy(), 'y_pred': predicted}))
        histories.append(history.assign(variant=variant))
        if variant == 'original':
            saved = load_ann(expected_manifest=original).predict(Xv)
            np.testing.assert_allclose(predicted, saved, rtol=0, atol=1e-10)
        print(variant, 'Feedforward ANN', f"validation MAE={metrics['mae']:.6f}", flush=True)

    comparison = pd.DataFrame(rows)
    all_predictions = pd.concat(predictions, ignore_index=True)
    comparison.to_csv(RESULTS / 'ablation_validation.csv', index=False)
    pd.DataFrame(historical_differences).to_csv(RESULTS / 'historical_reconciliation.csv', index=False)
    pd.DataFrame(folds).to_csv(RESULTS / 'cv_folds.csv', index=False)
    all_predictions.to_csv(RESULTS / 'validation_predictions.csv', index=False)
    pd.concat(histories, ignore_index=True).to_csv(RESULTS / 'ann_history.csv', index=False)
    joblib.dump(models, RESULTS / 'classical_models.joblib', compress=3)
    final = comparison[comparison.variant == VARIANTS[-1]].sort_values('validation_mae')
    final.to_csv(RESULTS / 'model_comparison_validation.csv', index=False)

    best_classical = final[final.model != 'Feedforward ANN'].iloc[0].model
    errors, largest = [], []
    for name in [best_classical, 'Feedforward ANN']:
        p = all_predictions[(all_predictions.variant == VARIANTS[-1]) &
                            (all_predictions.model == name)].merge(
            validation[['row_id', 'make', 'model', 'vehicle_class', 'hybrid',
                        'start_stop', 'transmission_detail']],
            on='row_id', validate='one_to_one', suffixes=('_predictor', '_vehicle'))
        p['error'] = p.y_pred - p.y_true
        p['absolute_error'] = p.error.abs()
        largest.append(p.nlargest(10, 'absolute_error'))
        for column in ['hybrid', 'vehicle_class', 'start_stop', 'transmission_detail']:
            g = p.groupby(column, dropna=False).agg(n=('error', 'size'),
                    mae=('absolute_error', 'mean'), mean_error=('error', 'mean')).reset_index()
            errors.append(g.rename(columns={column: 'group'}).assign(model=name, group_type=column))
    pd.concat(errors).to_csv(RESULTS / 'error_by_group_validation.csv', index=False)
    pd.concat(largest).to_csv(RESULTS / 'largest_errors_validation.csv', index=False)
    after = protected_hashes()
    if before != after:
        raise ValueError('The extension changed a protected original experiment artifact')
    metadata = {
        'protocol': protocol, 'versions': library_versions(), 'protected_sha256': before,
        'validation_winner_full': final.iloc[0].model, 'best_classical_full': best_classical,
        'test_evaluated': False, 'selection_metric': 'validation MAE (US MPG)',
        'scope': 'Post-hoc feature study after original holdout use; no independent confirmation',
        'limitations': ['One seed/snapshot; validation reused for checkpoints and comparisons',
                       'Frozen hyperparameters were originally selected using the coarse features',
                       'Do not infer causal start-stop or transmission effects from this ablation'],
        'source_sha256': {name: file_sha256(Path(__file__).with_name(name)) for name in
                          ['feature_extension.py', 'neural_network.py', 'data.py', 'preprocessing.py']},
    }
    metadata['bundle_sha256'] = {str(p.relative_to(RESULTS)): file_sha256(p)
        for p in RESULTS.rglob('*') if p.is_file()}
    _write_json(RESULTS / 'experiment_metadata.json', metadata)
    plot_results()
    print(final[['model', 'validation_mae', 'validation_rmse', 'validation_r2']].to_string(index=False))


def plot_results() -> None:
    """Regenerate portable figures from the recorded, unrounded result tables."""
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(RESULTS / 'ablation_validation.csv')
    colors = ['#737b85', '#b87818', '#236bb0']
    labels = ['Original', '+ start-stop', '+ start-stop + detailed transmission']
    names = ['Linear Regression', 'Ridge', 'KNN', 'Decision Tree', 'Feedforward ANN']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), sharey=True)
    for ax, metric, title in zip(axes, ['cv_mae', 'validation_mae'],
                                 ['Training-only grouped 5-fold CV', 'Shared validation (n=2,080)']):
        for j, variant in enumerate(VARIANTS):
            values = table[table.variant == variant].set_index('model').reindex(names)[metric]
            y = np.arange(len(names)) + (j - 1) * .19
            ax.scatter(values, y, color=colors[j], marker=['o', 's', '^'][j], s=55, label=labels[j])
        ax.set(yticks=np.arange(len(names)), yticklabels=names, xlabel='MAE (US MPG), lower is better', title=title)
        ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.5, .055),
               ncol=3, fontsize=9, frameon=False)
    fig.suptitle('Feature ablation: frozen estimator hyperparameters, unchanged row IDs')
    fig.text(.5, .01, 'Mean baseline omitted for scale (validation MAE 4.662). ANN has no CV score.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .14, 1, .96])
    fig.savefig(FIGURES / 'feature_ablation.png', dpi=160)
    plt.close(fig)

    history = pd.read_csv(RESULTS / 'ann_history.csv')
    chosen = history[history.variant == VARIANTS[-1]]
    best_epoch = int(chosen.best_epoch.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, ylabel in zip(axes, ['loss', 'mae'], ['MSE (standardized target)', 'MAE (US MPG)']):
        ax.plot(chosen.epoch, chosen[f'train_{metric}'], color='#236bb0', label='Train')
        ax.plot(chosen.epoch, chosen[f'val_{metric}'], color='#b87818', linestyle='--', label='Validation')
        ax.axvline(best_epoch, color='#555555', linestyle=':', label=f'Restored epoch {best_epoch}')
        ax.set(xlabel='Epoch', ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('ANN: start-stop and detailed transmission; frozen 64/32, lr=0.001')
    fig.tight_layout()
    fig.savefig(FIGURES / 'ann_learning_curves.png', dpi=160)
    plt.close(fig)

    meta = json.loads((RESULTS / 'experiment_metadata.json').read_text())
    predictions = pd.read_csv(RESULTS / 'validation_predictions.csv')
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, name in enumerate([meta['best_classical_full'], 'Feedforward ANN']):
        p = predictions[(predictions.variant == VARIANTS[-1]) & (predictions.model == name)]
        axes[0, col].scatter(p.y_true, p.y_pred, s=8, alpha=.3, color='#236bb0')
        bounds = [min(p.y_true.min(), p.y_pred.min()), max(p.y_true.max(), p.y_pred.max())]
        axes[0, col].plot(bounds, bounds, '--', color='#555555')
        axes[0, col].set(title=name, xlabel='Actual MPG', ylabel='Predicted MPG')
        axes[1, col].hist(p.y_pred - p.y_true, bins=40, color='#236bb0')
        axes[1, col].set(xlabel='Residual: predicted - actual (MPG)', ylabel='Records')
    fig.suptitle('Extended features: validation diagnostics (n=2,080)')
    fig.tight_layout()
    fig.savefig(FIGURES / 'evaluation_validation.png', dpi=160)
    plt.close(fig)


def verify() -> dict:
    """Reconcile all 18 row-level predictions and load/predict every saved model."""
    meta = json.loads((RESULTS / 'experiment_metadata.json').read_text())
    if meta['test_evaluated'] or meta['protected_sha256'] != protected_hashes():
        raise ValueError('Protected experiment changed or unexpected holdout evaluation')
    for relative, expected in meta['bundle_sha256'].items():
        if file_sha256(RESULTS / relative) != expected:
            raise ValueError(f'Extension artifact checksum mismatch: {relative}')
    train, validation, original = load_extension_data()
    assert original == meta['protocol']['original_manifest']
    table = pd.read_csv(RESULTS / 'ablation_validation.csv')
    predictions = pd.read_csv(RESULTS / 'validation_predictions.csv')
    models = joblib.load(RESULTS / 'classical_models.joblib')
    assert len(table) == 18 and len(predictions) == 18 * len(validation)
    for variant in VARIANTS:
        ann = load_ann(RESULTS / f'ann_{variant}', expected_manifest=variant_manifest(original, variant))
        Xv = validation[input_columns(variant)]
        for name in [*SPECS, 'Feedforward ANN']:
            p = predictions[(predictions.variant == variant) & (predictions.model == name)]
            assert p.row_id.is_unique
            np.testing.assert_array_equal(p.row_id, validation.row_id)
            np.testing.assert_array_equal(p.y_true, validation.mpg)
            model = ann if name == 'Feedforward ANN' else models[(variant, name)]
            np.testing.assert_allclose(model.predict(Xv), p.y_pred, rtol=0, atol=1e-10)
            row = table[(table.variant == variant) & (table.model == name)].iloc[0]
            for key, value in compute_metrics(p.y_true, p.y_pred).items():
                np.testing.assert_allclose(value, row[f'validation_{key}'], rtol=0, atol=1e-10)
    cv = pd.read_csv(RESULTS / 'cv_folds.csv')
    assert len(cv) == 75
    for (variant, name), group in cv.groupby(['variant', 'model']):
        row = table[(table.variant == variant) & (table.model == name)].iloc[0]
        np.testing.assert_allclose(group.mae.mean(), row.cv_mae, rtol=0, atol=1e-10)
        np.testing.assert_allclose(group.mae.std(ddof=0), row.cv_std, rtol=0, atol=1e-10)
    return {'variants': 3, 'models_per_variant': 6, 'validation_rows': len(validation),
            'predictions_and_metrics': 'match', 'original_artifacts': 'unchanged', 'test_evaluated': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify(), indent=2))
    else:
        run()


if __name__ == '__main__':
    main()
