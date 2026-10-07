"""Feature selection, feature engineering and preprocessing (Khamza).

Design rule: nothing in here learns from data except scikit-learn transformers
inside a Pipeline, which are fitted on training rows only (and, in
cross-validation, on each fold's training portion only).
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import TARGET

BASE_NUMERIC = ["model_year", "cylinders", "displacement",
                "hybrid", "turbo", "supercharged"]  # last three are 0/1 flags
CATEGORICAL = ["drive", "transmission", "vehicle_class", "fuel_type"]
ENGINEERED = ["displacement_per_cylinder"]
# make/model are excluded: very many distinct values, they would mostly memorise
# the training cars. They stay in the data for grouping and error analysis.
RAW_FEATURES = BASE_NUMERIC + CATEGORICAL


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return raw feature columns and the MPG target."""
    return df[RAW_FEATURES].copy(), df[TARGET].copy()


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature engineering (safe before splitting: nothing is fitted).

    displacement_per_cylinder = displacement (litres) / cylinders. Hypothesis:
    engine size per cylinder separates large-bore engines from small turbo
    engines better than displacement and cylinder count separately. Missing
    inputs stay NaN and are imputed later, inside the pipeline. Whether it helps
    must be tested on the validation set, not assumed.
    """
    X = X.copy()
    X["displacement_per_cylinder"] = X["displacement"] / X["cylinders"]
    return X


def build_preprocessor(scale: bool = False, engineered: bool = False) -> ColumnTransformer:
    """Unfitted ColumnTransformer.

    - numeric: median imputation, optional StandardScaler (needed for KNN and
      helpful for Linear Regression).
    - categorical: most-frequent imputation + one-hot; categories unseen in
      training are ignored.
    ``engineered=True`` expects columns produced by :func:`add_features`.
    """
    numeric = BASE_NUMERIC + (ENGINEERED if engineered else [])
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
    return ColumnTransformer(
        [("num", Pipeline(num_steps), numeric), ("cat", Pipeline(cat_steps), CATEGORICAL)]
    )


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per raw feature (for the cleaning section)."""
    n = df[RAW_FEATURES + [TARGET]].isna().sum()
    return pd.DataFrame({"missing": n, "percent": (n / len(df) * 100).round(2)})
