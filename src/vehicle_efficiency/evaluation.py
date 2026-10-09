"""Metrics, cross-validation and tuning helpers shared by every model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_validate

from .data import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "reports" / "results"


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE (primary), RMSE and R^2.

    MAE and RMSE are in MPG (average / outlier-sensitive size of the error; lower
    is better). R^2 is unitless: 1 is perfect, 0 equals always predicting the
    mean of the data it was computed on, negative is worse than that.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate(pipeline: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Metrics of an already fitted pipeline on (X, y)."""
    return compute_metrics(y, pipeline.predict(X))


def cross_validate_pipeline(pipeline: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                            n_splits: int = 5, seed: int = 42,
                            return_estimators: bool = False) -> dict:
    """K-fold CV on training data only.

    The whole pipeline is cloned and refitted per fold, so imputers, scalers and
    encoders learn from that fold's training part only. Returns mean and std per
    metric (MAE/RMSE in MPG).
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scoring = {"mae": "neg_mean_absolute_error",
               "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    out = cross_validate(pipeline, X, y, cv=cv, scoring=scoring,
                         return_estimator=return_estimators)
    summary: dict = {"n_splits": n_splits}
    for m in ("mae", "rmse", "r2"):
        scores = out[f"test_{m}"] * (-1 if m != "r2" else 1)
        summary[f"{m}_mean"] = float(scores.mean())
        summary[f"{m}_std"] = float(scores.std())
    if return_estimators:
        summary["estimators"] = out["estimator"]
    return summary


def save_results(result: dict, filename: str) -> Path:
    """Write a result dict as JSON under ``reports/results/``."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(result, indent=2))
    return path


# ---------------------------------------------------------------------------
# Hyperparameter tuning (Askhat)
# ---------------------------------------------------------------------------

def grouped_cv(n_splits: int = 5) -> GroupKFold:
    """Folds that keep exact duplicate records together.

    The shared split already groups duplicates (``data.make_splits``); plain KFold
    inside the training split would not, and a duplicate in another fold is a free
    exact match for KNN and a deep tree, which flatters their CV score.
    """
    return GroupKFold(n_splits=n_splits)


def tune_pipeline(pipeline: BaseEstimator, param_grid: dict, X: pd.DataFrame,
                  y: pd.Series, groups: pd.Series | None = None,
                  n_splits: int = 5, seed: int = 42) -> tuple[BaseEstimator, dict, pd.DataFrame]:
    """Grid search on training data only; MAE selects the best setting.

    The whole pipeline (imputer, scaler, encoder, model) is refitted inside every
    fold. With ``groups`` the folds keep duplicates together (:func:`grouped_cv`),
    otherwise shuffled KFold. Returns the best pipeline refitted on all of
    ``X``, a summary dict and the full grid table (MAE/RMSE in MPG).
    """
    cv = grouped_cv(n_splits) if groups is not None else KFold(n_splits, shuffle=True,
                                                              random_state=seed)
    scoring = {"mae": "neg_mean_absolute_error",
               "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, refit="mae",
                          return_train_score=True, n_jobs=-1)
    search.fit(X, y, groups=groups)
    res = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({
        "params": res["params"].map(lambda p: {k.removeprefix("model__"): v for k, v in p.items()}),
        "cv_mae_mean": -res["mean_test_mae"], "cv_mae_std": res["std_test_mae"],
        "cv_rmse_mean": -res["mean_test_rmse"], "cv_r2_mean": res["mean_test_r2"],
        "train_mae_mean": -res["mean_train_mae"],
    }).sort_values("cv_mae_mean").reset_index(drop=True)
    best = table.iloc[0]
    summary = {
        "n_splits": n_splits,
        "cv": "GroupKFold (duplicates grouped)" if groups is not None else "KFold (shuffled)",
        "grid": {k.removeprefix("model__"): v for k, v in param_grid.items()},
        "n_candidates": len(table),
        "best_params": best["params"],
        "cv_mae_mean": float(best["cv_mae_mean"]), "cv_mae_std": float(best["cv_mae_std"]),
        "cv_rmse_mean": float(best["cv_rmse_mean"]), "cv_r2_mean": float(best["cv_r2_mean"]),
    }
    return search.best_estimator_, summary, table
