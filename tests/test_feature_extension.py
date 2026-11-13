"""Feature attachment, unchanged split identity and train-only extended models."""
import numpy as np
import pandas as pd
import pytest

from vehicle_efficiency.data import RAW_FILE, _duplicate_groups
from vehicle_efficiency.feature_extension import (
    VARIANTS, add_extension_columns, build_variant_pipeline, input_columns,
    load_extension_data, variant_manifest,
)
from vehicle_efficiency.neural_network import ANNConfig, load_ann, load_training_data, save_ann, train_ann
from vehicle_efficiency.preprocessing import CATEGORICAL, RAW_FEATURES


def test_attachment_uses_row_ids_preserves_order_and_unknown_start_stop():
    rows = pd.DataFrame({'row_id': [2, 0, 1]}, index=[80, 30, 50])
    raw = pd.DataFrame({'startStop': ['Y', 'unknown', 'N'],
                        'trany': ['Automatic 8-spd', None, 'Manual 6-spd']})
    result = add_extension_columns(rows, raw)
    pd.testing.assert_series_equal(result.row_id, rows.row_id)
    np.testing.assert_allclose(result.start_stop, [0, 1, np.nan], equal_nan=True)
    assert result.transmission_detail.iloc[:2].tolist() == ['Manual 6-spd', 'Automatic 8-spd']
    assert pd.isna(result.transmission_detail.iloc[2])
    assert list(rows.columns) == ['row_id']  # source frame must stay unchanged


def test_duplicate_raw_id_attachment_is_rejected():
    with pytest.raises(ValueError, match='unique'):
        add_extension_columns(pd.DataFrame({'row_id': [0, 0]}),
                              pd.DataFrame({'startStop': ['Y'], 'trany': ['Manual 6-spd']}))


@pytest.mark.skipif(not RAW_FILE.exists(), reason='Fixed real EPA snapshot required')
def test_enrichment_keeps_original_rows_targets_groups_and_manifest():
    train, val, manifest = load_extension_data()
    old_train, old_val, old_manifest = load_training_data()
    assert manifest == old_manifest
    for current, old in [(train, old_train), (val, old_val)]:
        pd.testing.assert_frame_equal(current[old.columns], old)
        pd.testing.assert_series_equal(_duplicate_groups(current), _duplicate_groups(old))
        assert current.start_stop.isin([0., 1.]).all()
        assert current.transmission_detail.notna().all()
    assert set(train.row_id).isdisjoint(val.row_id)


@pytest.mark.parametrize('variant', VARIANTS)
def test_predictor_contract_excludes_target_and_replaces_coarse_transmission(variant):
    columns = input_columns(variant)
    assert not {'mpg', 'comb08', 'row_id', 'make', 'model'} & set(columns)
    assert ('start_stop' in columns) == (variant != 'original')
    assert ('transmission_detail' in columns) == (variant == VARIANTS[-1])
    assert ('transmission' in columns) != (variant == VARIANTS[-1])
    assert RAW_FEATURES == input_columns('original')


@pytest.fixture
def small_rows():
    if not RAW_FILE.exists():
        pytest.skip('Fixed real EPA snapshot required')
    train, val, manifest = load_extension_data()
    return train.iloc[:256].copy(), val.iloc[:80].copy(), manifest


def test_extended_scaler_and_encoder_fit_only_train_and_handle_unknown_category(small_rows):
    train, val, _ = small_rows
    variant = VARIANTS[-1]
    columns = input_columns(variant)
    pipeline = build_variant_pipeline('Linear Regression', variant, {})
    Xv = val[columns].copy()
    Xv['transmission_detail'] = 'VALIDATION_ONLY_CVT'
    Xv['start_stop'] = 123.0  # must not contaminate fitted training statistics
    pipeline.fit(train[columns], train.mpg)
    prep = pipeline.named_steps['prep']
    encoder = prep.named_transformers_['cat'].named_steps['onehot']
    assert all('VALIDATION_ONLY_CVT' not in categories for categories in encoder.categories_)
    index = list(prep.transformers_[0][2]).index('start_stop')
    scaler = prep.named_transformers_['num'].named_steps['scale']
    assert scaler.mean_[index] == pytest.approx(train.start_stop.mean())
    before = scaler.mean_.copy()
    assert np.isfinite(pipeline.predict(Xv)).all()
    np.testing.assert_array_equal(before, scaler.mean_)
    assert CATEGORICAL == ['drive', 'transmission', 'vehicle_class', 'fuel_type']


def test_extended_ann_roundtrip_input_contract_and_unseen_transmission(small_rows, tmp_path):
    train, val, original = small_rows
    variant = VARIANTS[-1]
    columns = input_columns(variant)
    prep = build_variant_pipeline('Linear Regression', variant, {})[:-1]
    ann, history = train_ann(train[columns], train.mpg, val[columns], val.mpg,
                             ANNConfig(max_epochs=3), preprocessor=prep)
    # Caller-owned preprocessing stays unfitted because training clones it.
    assert not hasattr(prep.named_steps['prep'], 'transformers_')
    manifest = variant_manifest(original, variant)
    ann.metadata.update({'dataset': manifest, 'test_evaluated': False})
    save_ann(ann, tmp_path)
    restored = load_ann(tmp_path, expected_manifest=manifest)
    np.testing.assert_array_equal(ann.predict(val), restored.predict(val))
    changed = val.copy()
    changed['mpg'] = -999999
    changed['transmission'] = 'IGNORED_COARSE_CATEGORY'
    np.testing.assert_array_equal(ann.predict(changed), ann.predict(val))
    changed['transmission_detail'] = 'UNSEEN_TEST_CATEGORY'
    assert np.isfinite(restored.predict(changed)).all()
    assert ann.target_scaler.mean_[0] == pytest.approx(train.mpg.mean())
    assert len(history) == 3
    assert original['raw_features'] == RAW_FEATURES


def test_invalid_variant_rejected():
    with pytest.raises(ValueError, match='Unknown'):
        input_columns('target_leakage')
