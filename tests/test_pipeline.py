"""Checks for training-only preprocessing and metric calculations."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from vehicle_efficiency.evaluation import compute_metrics, cross_validate_pipeline
from vehicle_efficiency.models import build_baseline, make_pipeline
from vehicle_efficiency.preprocessing import RAW_FEATURES, add_features


def frame(displacement, cylinders=None):
    n = len(displacement)
    return pd.DataFrame({
        "model_year": [2018] * n,
        "cylinders": cylinders if cylinders is not None else [4.0] * n,
        "displacement": displacement,
        "hybrid": [0] * n, "turbo": [0] * n, "supercharged": [0] * n,
        "drive": ["Front-Wheel Drive"] * n,
        "transmission": ["Automatic"] * n,
        "vehicle_class": ["Compact Cars"] * n,
        "fuel_type": ["Regular Gasoline"] * n,
    })[RAW_FEATURES]


def _num(pipe):
    return pipe.named_steps["prep"].named_transformers_["num"]


def test_imputer_and_scaler_learn_from_training_only():
    X_train = frame([2.0, 2.0, np.nan, 3.0])   # train median displacement = 2.0
    X_val = frame([9.0, 9.0, 9.0, 9.0])         # must not influence anything
    pipe = make_pipeline(LinearRegression(), scale=True)
    pipe.fit(X_train, [10, 11, 12, 13])
    idx = list(pipe.named_steps["prep"].transformers_[0][2]).index("displacement")
    assert _num(pipe).named_steps["impute"].statistics_[idx] == 2.0
    assert _num(pipe).named_steps["scale"].mean_[idx] == pytest.approx(np.mean([2, 2, 2, 3]))
    before = _num(pipe).named_steps["scale"].mean_.copy()
    pipe.predict(X_val)  # transforming new data must not refit anything
    np.testing.assert_array_equal(before, _num(pipe).named_steps["scale"].mean_)


def test_unseen_category_does_not_break_prediction():
    pipe = make_pipeline(LinearRegression(), scale=True).fit(frame([2.0, 2.5, 3.0, 3.5]), [1, 2, 3, 4])
    new = frame([2.0])
    new["vehicle_class"] = "Brand New Class"
    assert np.isfinite(pipe.predict(new)).all()


def test_cross_validation_refits_preprocessing_per_fold():
    rng = np.random.default_rng(0)
    X = frame(list(rng.uniform(1.0, 6.0, 40)))
    y = pd.Series(rng.uniform(10, 40, 40))
    res = cross_validate_pipeline(make_pipeline(LinearRegression(), scale=True),
                                  X, y, n_splits=4, return_estimators=True)
    idx = list(res["estimators"][0].named_steps["prep"].transformers_[0][2]).index("displacement")
    means = [_num(e).named_steps["scale"].mean_[idx] for e in res["estimators"]]
    assert len(set(np.round(means, 6))) > 1  # one scaler per fold, different stats


def test_add_features_is_stateless_and_nan_safe():
    out = add_features(frame([2.0, np.nan], [4.0, 4.0]))
    assert out["displacement_per_cylinder"].iloc[0] == pytest.approx(0.5)
    assert np.isnan(out["displacement_per_cylinder"].iloc[1])


def test_baseline_predicts_training_mean():
    pipe = build_baseline().fit(frame([2.0, 2.5, 3.0]), [10.0, 20.0, 30.0])
    assert pipe.predict(frame([9.0])) == pytest.approx([20.0])


def test_metrics_known_values():
    m = compute_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(4 / 3))
    assert m["r2"] == pytest.approx(-1.0)  # 1 - SSE(4) / SST(2)


def test_perfect_prediction_metrics():
    m = compute_metrics([5.0, 7.0, 9.0], [5.0, 7.0, 9.0])
    assert m == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}


def test_linear_regression_builder_fits_scales_and_supports_ridge():
    from sklearn.linear_model import Ridge
    from vehicle_efficiency.models import build_linear_regression
    X = frame([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    y = [30, 28, 26, 24, 22, 20]
    for pipe in (build_linear_regression(), build_linear_regression(engineered=True),
                 build_linear_regression(alpha=1.0)):
        pipe.fit(X, y)
        assert np.isfinite(pipe.predict(X)).all()
        assert "scale" in _num(pipe).named_steps
    assert isinstance(build_linear_regression(alpha=1.0).named_steps["model"], Ridge)
