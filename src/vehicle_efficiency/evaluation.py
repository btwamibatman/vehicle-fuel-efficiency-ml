"""Metrics, cross-validation, tuning, comparison and error-analysis helpers shared by every model."""
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
# Tuning, model comparison and error analysis (Askhat)
# ---------------------------------------------------------------------------

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRIC_LABELS = {"mae": "MAE (MPG)", "rmse": "RMSE (MPG)", "r2": "R2"}

# Plot styling: one series per panel, so a single categorical hue; text stays in ink.
_POINT_COLOR = "#2a78d6"
_REFERENCE_COLOR = "#8a8984"
_INK = "#0b0b0b"
_INK_MUTED = "#52514e"


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


def comparison_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Models as rows, MAE/RMSE/R2 as columns, best (lowest MAE) first."""
    return (pd.DataFrame(results).T[list(METRIC_LABELS)]
            .sort_values("mae").rename(columns=METRIC_LABELS))


def largest_errors(rows: pd.DataFrame, y_true: pd.Series, y_pred, n: int = 10) -> pd.DataFrame:
    """The ``n`` predictions with the largest absolute error, with the vehicle's details.

    ``rows`` is the split frame (it still has make/model for identification).
    ``error`` = predicted - actual, so positive means the model over-estimated MPG.
    """
    out = rows.copy()
    out["actual_mpg"] = np.asarray(y_true)
    out["predicted_mpg"] = np.round(np.asarray(y_pred), 1)
    out["error"] = (out["predicted_mpg"] - out["actual_mpg"]).round(1)
    cols = ["make", "model", "model_year", "vehicle_class", "cylinders", "displacement",
            "drive", "transmission", "fuel_type", "hybrid", "turbo",
            "actual_mpg", "predicted_mpg", "error"]
    return out.loc[out["error"].abs().nlargest(n).index, cols]


def error_by_group(rows: pd.DataFrame, y_true: pd.Series, y_pred, column: str) -> pd.DataFrame:
    """MAE and mean signed error (predicted - actual) per value of ``column``."""
    err = pd.Series(np.asarray(y_pred) - np.asarray(y_true), index=rows.index)
    g = pd.DataFrame({column: rows[column], "error": err}).groupby(column)["error"]
    return (pd.DataFrame({"n": g.size(), "mae": g.apply(lambda e: e.abs().mean()),
                          "mean_error": g.mean()})
            .sort_values("mae", ascending=False).round(2))


def _panels(n: int):
    import matplotlib.pyplot as plt
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.6 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=_INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=_INK_MUTED)
    ax.set_ylabel(ylabel, color=_INK_MUTED)
    ax.tick_params(colors=_INK_MUTED)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_actual_vs_predicted(y_true: pd.Series, predictions: dict[str, np.ndarray],
                             path: Path | None = None):
    """One panel per model: predicted vs actual MPG; the diagonal is a perfect prediction."""
    fig, axes = _panels(len(predictions))
    lo, hi = float(np.min(y_true)) - 2, float(np.max(y_true)) + 2
    for ax, (name, pred) in zip(axes, predictions.items()):
        ax.plot([lo, hi], [lo, hi], color=_REFERENCE_COLOR, lw=1.2, ls="--")
        ax.scatter(y_true, pred, s=10, alpha=0.4, color=_POINT_COLOR, edgecolors="none")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        mae = mean_absolute_error(y_true, pred)
        _style(ax, f"{name}  (MAE {mae:.2f} MPG)", "actual MPG", "predicted MPG")
    fig.suptitle("Validation set: predicted vs actual MPG", color=_INK, x=0.01, ha="left")
    fig.tight_layout()
    if path is not None:
        _save(fig, path)
    return fig


def plot_residuals(y_true: pd.Series, predictions: dict[str, np.ndarray],
                   path: Path | None = None):
    """One panel per model: residual (actual - predicted) against predicted MPG."""
    fig, axes = _panels(len(predictions))
    for ax, (name, pred) in zip(axes, predictions.items()):
        resid = np.asarray(y_true) - np.asarray(pred)
        ax.axhline(0, color=_REFERENCE_COLOR, lw=1.2, ls="--")
        ax.scatter(pred, resid, s=10, alpha=0.4, color=_POINT_COLOR, edgecolors="none")
        _style(ax, name, "predicted MPG", "residual: actual - predicted (MPG)")
    fig.suptitle("Validation set: residuals (above 0 = model under-predicts MPG)",
                 color=_INK, x=0.01, ha="left")
    fig.tight_layout()
    if path is not None:
        _save(fig, path)
    return fig


def _save(fig, path: Path | str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = FIGURES_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
