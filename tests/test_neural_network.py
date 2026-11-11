"""Leakage, checkpoint, persistence and snapshot checks for Khamza's ANN."""
import copy

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from vehicle_efficiency.data import RAW_FILE
from vehicle_efficiency.evaluation import compute_metrics
from vehicle_efficiency.neural_network import (
    ANNConfig, _dense_float32, load_ann, load_training_data,
    save_ann, train_ann, verify_manifest,
)
from vehicle_efficiency.preprocessing import RAW_FEATURES, split_xy


@pytest.fixture(scope="module")
def real_rows():
    if not RAW_FILE.exists():
        pytest.skip("The fixed real EPA snapshot is needed for ANN integration tests")
    train, val, manifest = load_training_data()
    return train.iloc[:256].copy(), val.iloc[:80].copy(), manifest


@pytest.fixture(scope="module")
def trained(real_rows):
    train, val, manifest = real_rows
    Xt, yt = split_xy(train)
    Xv, yv = split_xy(val)
    model, history = train_ann(Xt, yt, Xv, yv, ANNConfig(max_epochs=4))
    model.metadata = {"dataset": manifest, "test_evaluated": False}
    return model, history


def test_dense_conversion_handles_sparse_and_rejects_nonfinite():
    matrix = sparse.csr_matrix([[0.0, 2.0], [3.0, 0.0]])
    dense = _dense_float32(matrix)
    assert dense.dtype == np.float32 and dense.flags.c_contiguous
    np.testing.assert_array_equal(dense, matrix.toarray())
    np.testing.assert_array_equal(_dense_float32(matrix.toarray()), dense)
    with pytest.raises(ValueError, match="non-finite"):
        _dense_float32(np.array([[np.inf]]))


def test_training_only_imputation_scaling_and_unknown_categories(real_rows):
    train, val, _ = real_rows
    Xt, yt = split_xy(train)
    Xv, yv = split_xy(val)
    Xt.loc[Xt.index[0], "displacement"] = np.nan
    Xv["displacement"] = 1000.0
    Xv["vehicle_class"] = "UNSEEN_VALIDATION_ONLY_CLASS"
    model, _ = train_ann(Xt, yt, Xv, yv + 1000, ANNConfig(max_epochs=2))
    prep = model.preprocessor.named_steps["prep"]
    numeric = prep.named_transformers_["num"]
    index = list(prep.transformers_[0][2]).index("displacement")
    median = Xt["displacement"].median()
    assert numeric.named_steps["impute"].statistics_[index] == pytest.approx(median)
    assert numeric.named_steps["scale"].mean_[index] == pytest.approx(
        Xt["displacement"].fillna(median).mean())
    encoder = prep.named_transformers_["cat"].named_steps["onehot"]
    assert all("UNSEEN_VALIDATION_ONLY_CLASS" not in values for values in encoder.categories_)
    assert model.target_scaler.mean_[0] == pytest.approx(yt.mean())
    assert model.target_scaler.var_[0] == pytest.approx(yt.var(ddof=0))
    before = numeric.named_steps["scale"].mean_.copy()
    assert np.isfinite(model.predict(Xv)).all()
    np.testing.assert_array_equal(before, numeric.named_steps["scale"].mean_)


def test_restored_checkpoint_matches_minimum_validation_loss(real_rows):
    train, val, _ = real_rows
    Xt, yt = split_xy(train)
    Xv, yv = split_xy(val)
    model, history = train_ann(Xt, yt, Xv, yv,
                               ANNConfig(learning_rate=0.02, max_epochs=40, patience=4))
    assert model.best_epoch == int(history.loc[history["val_loss"].idxmin(), "epoch"])
    scaled_true = model.target_scaler.transform(yv.to_numpy().reshape(-1, 1)).ravel()
    scaled_pred = model.target_scaler.transform(model.predict(Xv).reshape(-1, 1)).ravel()
    assert np.mean((scaled_pred - scaled_true) ** 2) == pytest.approx(
        history["val_loss"].min(), rel=2e-6)
    assert history["is_best_epoch"].sum() == 1


def test_early_stopping_waits_patience_epochs_without_improvement(real_rows):
    train, val, _ = real_rows
    Xt, yt = split_xy(train)
    Xv, yv = split_xy(val)
    # Zero learning rate deliberately produces identical losses (test only).
    model, history = train_ann(Xt, yt, Xv, yv,
                               ANNConfig(learning_rate=0, max_epochs=20, patience=3))
    assert model.best_epoch == 1
    assert len(history) == 4


def test_prediction_is_in_mpg_and_target_columns_cannot_influence_it(real_rows, trained):
    train, val, _ = real_rows
    model, _ = trained
    Xv, _ = split_xy(val)
    expected = model.predict(Xv)
    assert expected.shape == (len(val),) and np.isfinite(expected).all()
    changed = val.copy()
    changed["mpg"] = -1000000
    changed["row_id"] = -1
    np.testing.assert_array_equal(model.predict(changed), expected)
    assert not {"mpg", "row_id", "comb08", "make", "model"} & set(RAW_FEATURES)
    constant = copy.deepcopy(model)
    for parameter in constant.network.parameters():
        parameter.data.zero_()
    np.testing.assert_allclose(constant.predict(Xv), train["mpg"].mean(), atol=1e-5)


def test_same_seed_reproduces_history_and_predictions(real_rows, trained):
    train, val, _ = real_rows
    first, history = trained
    Xt, yt = split_xy(train)
    Xv, yv = split_xy(val)
    again, other = train_ann(Xt, yt, Xv, yv, ANNConfig(max_epochs=4))
    pd.testing.assert_frame_equal(history, other)
    np.testing.assert_array_equal(first.predict(Xv), again.predict(Xv))


def test_saved_bundle_roundtrip_and_mismatched_snapshot_rejected(tmp_path, real_rows, trained):
    _, val, manifest = real_rows
    model, _ = trained
    save_ann(model, tmp_path)
    loaded = load_ann(tmp_path, expected_manifest=manifest)
    Xv, _ = split_xy(val)
    np.testing.assert_array_equal(loaded.predict(Xv), model.predict(Xv))
    wrong = copy.deepcopy(manifest)
    wrong["raw_sha256"] = "different-snapshot"
    with pytest.raises(ValueError, match="raw_sha256"):
        load_ann(tmp_path, expected_manifest=wrong)
    weights = tmp_path / "ann_model.pt"
    weights.write_bytes(weights.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_ann(tmp_path)


def test_manifest_rejects_changed_split_and_features(real_rows):
    _, _, manifest = real_rows
    for key, value in (("split_seed", 99), ("feature_set", ["mpg"]),
                       ("split_row_id_sha256", {})):
        changed = copy.deepcopy(manifest)
        changed[key] = value
        with pytest.raises(ValueError, match=key):
            verify_manifest(manifest, changed)


def test_validation_mae_history_is_in_original_units(real_rows, trained):
    _, val, _ = real_rows
    model, history = trained
    Xv, yv = split_xy(val)
    restored = history.loc[history["is_best_epoch"]].iloc[0]
    actual = compute_metrics(yv, model.predict(Xv))
    assert restored["val_mae"] == pytest.approx(actual["mae"], rel=2e-6)
