"""Model builders. Every model is a full Pipeline: features -> preprocessing -> estimator.

Status
------
- build_baseline          DONE (Khamza)
- build_linear_regression DONE (Khamza)
- build_decision_tree     DONE (Askhat)
- build_knn               TODO Askhat
"""
from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeRegressor

from .preprocessing import add_features, build_preprocessor


def make_pipeline(estimator: BaseEstimator, scale: bool = False,
                  engineered: bool = False) -> Pipeline:
    """Wrap an estimator with the shared preprocessing.

    Use ``scale=True`` for distance-based or regularised models (KNN requires it).
    Trees do not need scaling. Always use this helper so every model gets
    identical, leak-free preprocessing.
    """
    steps = []
    if engineered:
        steps.append(("features", FunctionTransformer(add_features)))
    steps.append(("prep", build_preprocessor(scale=scale, engineered=engineered)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def build_baseline() -> Pipeline:
    """DummyRegressor predicting the training-set mean MPG. Every real model must beat it."""
    return make_pipeline(DummyRegressor(strategy="mean"))


def build_linear_regression(engineered: bool = False, alpha: float | None = None) -> Pipeline:
    """Linear Regression on scaled, one-hot encoded features.

    ``engineered=True`` adds ``displacement_per_cylinder``. ``alpha`` switches to
    Ridge regression with that penalty (``None`` = ordinary least squares).
    Whether the engineered feature helps is decided on validation data in
    ``project.ipynb`` section 5, not assumed here.
    """
    estimator = LinearRegression() if alpha is None else Ridge(alpha=alpha)
    return make_pipeline(estimator, scale=True, engineered=engineered)


def build_decision_tree(engineered: bool = False, **params) -> Pipeline:
    """Decision Tree regressor; no scaling (splits are thresholds on one feature).

    ``params`` go to ``DecisionTreeRegressor`` (e.g. ``max_depth``,
    ``min_samples_leaf``). Unlimited depth memorises the training set, so these
    two are tuned with training CV (``TREE_PARAM_GRID``).
    """
    return make_pipeline(DecisionTreeRegressor(random_state=42, **params),
                         engineered=engineered)


def build_knn(**params) -> Pipeline:
    """TODO (Askhat): KNeighborsRegressor(**params) with ``scale=True`` (mandatory for KNN)."""
    raise NotImplementedError("KNN is Askhat's task - see docs/team_tasks.md")


# Small hyperparameter grids searched with CV on the training split only.
# Keys use the Pipeline step prefix ``model__`` so GridSearchCV can set them.
TREE_PARAM_GRID = {
    "model__max_depth": [4, 6, 8, 10, 12, None],
    "model__min_samples_leaf": [1, 5, 10, 20],
}


MODEL_BUILDERS = {
    "baseline": build_baseline,
    "linear_regression": build_linear_regression,
    "decision_tree": build_decision_tree,
    "knn": build_knn,
}
