"""Training-only PCA for the endterm engine-characteristic analysis.

The analysis deliberately uses only ``cylinders`` and ``displacement``.  It
describes variation among engines; it is not a MPG prediction model and it does
not fit, select components, or inspect the validation/test targets.

Run ``python -m vehicle_efficiency.dimensionality_reduction`` from the project
root to write the PCA tables and figures.  The optional cluster overlay remains
separate because Askhat supplies the K-Means assignments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import PROJECT_ROOT, RANDOM_SEED, RAW_FILE, load_dataset, make_splits

PCA_FEATURES = ["cylinders", "displacement"]
RESULTS_DIR = PROJECT_ROOT / "reports" / "results" / "endterm"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "endterm"
MANIFEST_FILE = RESULTS_DIR / "ann_dataset_manifest.json"


def build_pca_pipeline() -> Pipeline:
    """Return an unfitted median-imputation, scaling and two-component PCA pipeline."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=len(PCA_FEATURES))),
    ])


def fit_pca(train_rows: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit PCA on training rows and return pipeline, variance, loadings and coordinates."""
    missing = set(PCA_FEATURES + ["row_id"]) - set(train_rows.columns)
    if missing:
        raise ValueError(f"PCA input is missing columns: {sorted(missing)}")
    pipeline = build_pca_pipeline().fit(train_rows[PCA_FEATURES])
    pca = pipeline.named_steps["pca"]
    transformed = pipeline.transform(train_rows[PCA_FEATURES])
    components = [f"PC{i}" for i in range(1, len(PCA_FEATURES) + 1)]
    variance = pd.DataFrame({
        "component": components,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    loadings = pd.DataFrame(pca.components_.T, index=PCA_FEATURES, columns=components)
    loadings.index.name = "feature"
    coordinates = pd.DataFrame(transformed, columns=components)
    coordinates.insert(0, "row_id", train_rows["row_id"].to_numpy())
    return pipeline, variance, loadings, coordinates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Shared endterm manifest is missing: {MANIFEST_FILE}")
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))


def _verify_snapshot(manifest: dict) -> None:
    observed = _sha256(RAW_FILE)
    if observed != manifest["raw_sha256"]:
        raise ValueError("Local vehicles.csv does not match the shared endterm snapshot")


def _plot_variance(variance: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(variance))
    ratios = variance["explained_variance_ratio"].to_numpy() * 100
    cumulative = variance["cumulative_explained_variance"].to_numpy() * 100
    ax.bar(x, ratios, color="#2a78d6", label="individual explained variance")
    ax.plot(x, cumulative, marker="o", color="#d35f3d", label="cumulative explained variance")
    ax.set_xticks(x, variance["component"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("Training data: PCA explained variance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_coordinates(coordinates: pd.DataFrame, variance: pd.DataFrame, path: Path,
                      clusters: pd.Series | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    x_label = f"PC1 ({variance.iloc[0]['explained_variance_ratio'] * 100:.1f}% variance)"
    y_label = f"PC2 ({variance.iloc[1]['explained_variance_ratio'] * 100:.1f}% variance)"
    if clusters is None:
        ax.scatter(coordinates["PC1"], coordinates["PC2"], s=9, alpha=0.35,
                   color="#2a78d6", edgecolors="none")
        title = "Training data: engine PCA coordinates"
    else:
        scatter = ax.scatter(coordinates["PC1"], coordinates["PC2"], c=clusters,
                             cmap="tab10", s=10, alpha=0.5, edgecolors="none")
        fig.colorbar(scatter, ax=ax, label="K-Means cluster")
        title = "Training data: engine PCA coordinates by K-Means cluster"
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def merge_cluster_assignments(coordinates: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """Strictly join Askhat's training assignments to PCA coordinates by ``row_id``."""
    expected = {"row_id", "cluster"}
    if not expected <= set(assignments.columns):
        raise ValueError("Cluster assignments must contain row_id and cluster")
    if coordinates["row_id"].duplicated().any() or assignments["row_id"].duplicated().any():
        raise ValueError("PCA coordinates and cluster assignments require unique row_id values")
    merged = coordinates.merge(assignments[["row_id", "cluster"]], on="row_id",
                               how="outer", validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError("PCA and cluster row_id sets differ; do not plot a partial overlay")
    return merged.drop(columns="_merge").sort_values("row_id").reset_index(drop=True)


def run_pca() -> dict[str, Path]:
    """Create the PCA artifacts from the shared snapshot and training split only."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    _verify_snapshot(manifest)
    dataset = load_dataset()
    splits = make_splits(dataset, seed=RANDOM_SEED)
    train = splits["train"]
    pipeline, variance, loadings, coordinates = fit_pca(train)
    config = {
        "features": PCA_FEATURES,
        "fit_split": "train",
        "n_train": len(train),
        "imputation": "median fitted on train only",
        "scaling": "StandardScaler fitted on train only",
        "components": 2,
        "split_seed": RANDOM_SEED,
        "snapshot": {key: manifest[key] for key in (
            "raw_sha256", "n_prepared", "split_seed", "split_sizes",
            "split_row_id_sha256", "split_assignments_sha256",
        )},
        "interpretation_limit": "PCA describes engine-feature variance, not MPG prediction or natural clusters.",
    }
    paths = {
        "variance": RESULTS_DIR / "pca_variance.csv",
        "loadings": RESULTS_DIR / "pca_loadings.csv",
        "coordinates": RESULTS_DIR / "pca_coordinates_train.csv",
        "config": RESULTS_DIR / "pca_config.json",
        "variance_plot": FIGURES_DIR / "pca_explained_variance.png",
        "scatter_plot": FIGURES_DIR / "pca_train_scatter.png",
    }
    variance.to_csv(paths["variance"], index=False)
    loadings.reset_index().to_csv(paths["loadings"], index=False)
    coordinates.to_csv(paths["coordinates"], index=False)
    paths["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")
    _plot_variance(variance, paths["variance_plot"])
    _plot_coordinates(coordinates, variance, paths["scatter_plot"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the training-only engine PCA analysis")
    parser.add_argument("--overlay", type=Path, help="Askhat's clustering_assignments_train.csv")
    args = parser.parse_args()
    paths = run_pca()
    if args.overlay:
        assignments = pd.read_csv(args.overlay)
        coordinates = pd.read_csv(paths["coordinates"])
        variance = pd.read_csv(paths["variance"])
        merged = merge_cluster_assignments(coordinates, assignments)
        overlay_path = FIGURES_DIR / "pca_clusters_train.png"
        _plot_coordinates(merged, variance, overlay_path, merged["cluster"])
        print(f"cluster_overlay={overlay_path}")
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
