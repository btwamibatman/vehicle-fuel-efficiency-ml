"""Tests for Aktore's training-only PCA helpers."""
import numpy as np
import pandas as pd
import pytest

from vehicle_efficiency.dimensionality_reduction import (
    PCA_FEATURES, build_pca_pipeline, fit_pca, merge_cluster_assignments,
)


def rows():
    return pd.DataFrame({
        "row_id": [11, 12, 13, 14],
        "cylinders": [4.0, 6.0, np.nan, 8.0],
        "displacement": [1.8, 3.0, 2.1, 5.0],
        "mpg": [35.0, 25.0, 30.0, 18.0],
    })


def test_pca_uses_engine_features_and_returns_two_components():
    pipeline, variance, loadings, coordinates = fit_pca(rows())
    assert list(loadings.index) == PCA_FEATURES
    assert list(variance["component"]) == ["PC1", "PC2"]
    assert variance["explained_variance_ratio"].sum() == pytest.approx(1.0)
    assert len(coordinates) == 4
    assert list(coordinates.columns) == ["row_id", "PC1", "PC2"]
    assert np.isfinite(coordinates[["PC1", "PC2"]].to_numpy()).all()
    assert "impute" in pipeline.named_steps and "scale" in pipeline.named_steps


def test_pca_scaler_is_fit_only_on_the_provided_training_rows():
    pipeline = build_pca_pipeline().fit(rows()[PCA_FEATURES])
    mean_before = pipeline.named_steps["scale"].mean_.copy()
    pipeline.transform(pd.DataFrame({"cylinders": [100.0], "displacement": [100.0]}))
    np.testing.assert_array_equal(mean_before, pipeline.named_steps["scale"].mean_)


def test_cluster_overlay_requires_exact_unique_row_id_match():
    _, _, _, coordinates = fit_pca(rows())
    assignments = pd.DataFrame({"row_id": [11, 12, 13, 14], "cluster": [0, 0, 1, 1]})
    merged = merge_cluster_assignments(coordinates, assignments)
    assert list(merged["cluster"]) == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="sets differ"):
        merge_cluster_assignments(coordinates, assignments.iloc[:-1])
    with pytest.raises(ValueError, match="unique"):
        merge_cluster_assignments(coordinates, pd.concat([assignments, assignments.iloc[:1]]))
