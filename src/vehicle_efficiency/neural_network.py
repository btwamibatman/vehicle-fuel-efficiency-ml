"""CPU feedforward ANN: train/validation selection, persistence and MPG prediction.

    python -m vehicle_efficiency.neural_network --smoke
    python -m vehicle_efficiency.neural_network
    python -m vehicle_efficiency.neural_network --verify

There is deliberately no test-evaluation command. The existing split is recreated
only to verify row identities; no test features/targets enter ANN development.
Owner: Khamza.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
import torch
from torch import nn

from .data import PROJECT_ROOT, RANDOM_SEED, RAW_FILE, RAW_URL, load_dataset, make_splits
from .evaluation import compute_metrics
from .preprocessing import (ENGINEERED, RAW_FEATURES, add_features,
                            build_preprocessor, split_xy)

RESULTS_DIR = PROJECT_ROOT / "reports" / "results" / "endterm"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "endterm"
MANIFEST_FILE = RESULTS_DIR / "ann_dataset_manifest.json"
FEATURE_SET = RAW_FEATURES + ENGINEERED
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "torch", "joblib",
            "matplotlib", "seaborn", "jupyter", "nbconvert", "nbformat",
            "nbclient", "ipykernel", "pytest")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                     sort_keys=True).encode("utf-8")).hexdigest()


def library_versions() -> dict:
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in PACKAGES}}


def dataset_manifest(df: pd.DataFrame, splits: dict[str, pd.DataFrame],
                     raw_path: Path = RAW_FILE) -> dict:
    """Fingerprint the raw bytes and the existing split's IDs, not test outcomes."""
    assignments = sorted((int(row_id), name) for name, rows in splits.items()
                         for row_id in rows["row_id"])
    return {
        "source_url": RAW_URL,
        "raw_sha256": file_sha256(raw_path),
        "n_prepared": len(df), "split_seed": RANDOM_SEED,
        "split_method": "existing data.make_splits / GroupShuffleSplit",
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "split_row_id_sha256": {
            name: _json_hash([int(i) for i in rows["row_id"]])
            for name, rows in splits.items()},
        "split_assignments_sha256": _json_hash(assignments),
        "raw_features": RAW_FEATURES, "feature_set": FEATURE_SET,
        "target": "mpg", "target_source": "comb08", "target_unit": "US MPG",
    }


def verify_manifest(expected: dict, actual: dict) -> None:
    """Fail before combining artifacts from different snapshots, splits or features."""
    for key in ("raw_sha256", "n_prepared", "split_seed", "split_method",
                "split_sizes", "split_row_id_sha256", "split_assignments_sha256",
                "raw_features", "feature_set", "target", "target_unit"):
        if expected.get(key) != actual.get(key):
            raise ValueError(f"Endterm dataset/protocol mismatch: {key}")


def load_training_data(manifest_path: Path = MANIFEST_FILE):
    """Return only train, validation and checked metadata; never return test data.

    Requires the existing local CSV. It never downloads or replaces a snapshot.
    The committed manifest is the reference, not a newly accepted fingerprint.
    """
    expected = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    df = load_dataset()
    splits = make_splits(df, seed=RANDOM_SEED)
    actual = dataset_manifest(df, splits)
    verify_manifest(expected, actual)
    return splits["train"], splits["val"], actual


@dataclass(frozen=True)
class ANNConfig:
    hidden_sizes: tuple[int, ...] = (64, 32)
    learning_rate: float = 0.001
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20
    seed: int = 42


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _network(n_features: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    layers = []
    for width in hidden_sizes:
        layers.extend([nn.Linear(n_features, width), nn.ReLU()])
        n_features = width
    return nn.Sequential(*layers, nn.Linear(n_features, 1))


def _dense_float32(matrix) -> np.ndarray:
    # Convert before densifying to avoid a temporary dense float64 matrix.
    # The current 40-column design needs about 1.5 MiB for the entire train split.
    matrix = matrix.astype(np.float32)
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.ascontiguousarray(matrix)
    if not np.isfinite(matrix).all():
        raise ValueError("ANN input contains non-finite values after preprocessing")
    return matrix


@dataclass
class ANNModel:
    network: nn.Sequential
    preprocessor: Pipeline
    target_scaler: StandardScaler
    config: ANNConfig
    best_epoch: int
    metadata: dict

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict unrounded MPG from the shared raw feature columns, without refit."""
        matrix = _dense_float32(self.preprocessor.transform(X[RAW_FEATURES]))
        self.network.eval()
        with torch.inference_mode():
            scaled = self.network(torch.from_numpy(matrix)).numpy()
        return self.target_scaler.inverse_transform(scaled).ravel().astype(float)


def train_ann(X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val,
              config: ANNConfig = ANNConfig()) -> tuple[ANNModel, pd.DataFrame]:
    """Fit on train; stop/restore by validation scaled MSE. No test argument.

    Epoch metrics are evaluated on whole splits AFTER the epoch update. Loss is
    MSE of train-standardised targets; MAE is inverse-transformed to MPG.
    """
    if min(config.max_epochs, config.patience, config.batch_size) < 1:
        raise ValueError("epochs, patience and batch_size must be positive")
    set_seeds(config.seed)
    prep = Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", build_preprocessor(scale=True, engineered=True)),
    ])
    train_array = _dense_float32(prep.fit_transform(X_train[RAW_FEATURES]))
    val_array = _dense_float32(prep.transform(X_val[RAW_FEATURES]))
    target_scaler = StandardScaler().fit(np.asarray(y_train).reshape(-1, 1))
    train_target = target_scaler.transform(np.asarray(y_train).reshape(-1, 1))
    val_target = target_scaler.transform(np.asarray(y_val).reshape(-1, 1))
    Xt, Xv = torch.from_numpy(train_array), torch.from_numpy(val_array)
    yt = torch.tensor(train_target, dtype=torch.float32)
    yv = torch.tensor(val_target, dtype=torch.float32)
    net = _network(train_array.shape[1], config.hidden_sizes).cpu()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    best_loss, best_epoch, wait = float("inf"), 0, 0
    best_state, records = None, []
    target_sd = float(target_scaler.scale_[0])

    for epoch in range(1, config.max_epochs + 1):
        net.train()
        indices = torch.randperm(len(Xt), generator=generator)
        for batch in indices.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(Xt[batch]), yt[batch])
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.inference_mode():
            train_pred, val_pred = net(Xt), net(Xv)
            train_loss = float(criterion(train_pred, yt))
            val_loss = float(criterion(val_pred, yv))
            train_mae = float(torch.abs(train_pred - yt).mean()) * target_sd
            val_mae = float(torch.abs(val_pred - yv).mean()) * target_sd
        if not np.isfinite([train_loss, val_loss, train_mae, val_mae]).all():
            raise ValueError("ANN training produced non-finite metrics")
        records.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "train_mae": train_mae,
                        "val_mae": val_mae})
        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            wait += 1
        if wait >= config.patience:
            break

    net.load_state_dict(best_state)
    net.eval()
    history = pd.DataFrame(records)
    history["best_epoch"] = best_epoch
    history["is_best_epoch"] = history["epoch"].eq(best_epoch)
    return ANNModel(net, prep, target_scaler, config, best_epoch, {}), history


def select_ann(X_train, y_train, X_val, y_val, *, max_epochs: int = 200):
    """Run the four predeclared candidates; select restored checkpoint by val MAE."""
    best, best_mae, histories, rows = None, float("inf"), [], []
    for hidden in ((32,), (64, 32)):
        for lr in (0.001, 0.0003):
            candidate = f"h{'-'.join(map(str, hidden))}_lr{lr:g}"
            config = ANNConfig(hidden_sizes=hidden, learning_rate=lr,
                               max_epochs=max_epochs)
            model, history = train_ann(X_train, y_train, X_val, y_val, config)
            train_metrics = compute_metrics(y_train, model.predict(X_train))
            val_metrics = compute_metrics(y_val, model.predict(X_val))
            rows.append({"candidate": candidate,
                         "hidden_sizes": json.dumps(hidden), "learning_rate": lr,
                         "best_epoch": model.best_epoch, "epochs_run": len(history),
                         "stopped_early": len(history) < max_epochs,
                         **{f"train_{k}": v for k, v in train_metrics.items()},
                         **{f"val_{k}": v for k, v in val_metrics.items()}})
            histories.append(history.assign(candidate=candidate))
            if val_metrics["mae"] < best_mae:
                best, best_mae = model, val_metrics["mae"]
    search = pd.DataFrame(rows).sort_values("val_mae", kind="stable").reset_index(drop=True)
    selected = search.iloc[0]["candidate"]
    search["selected"] = search["candidate"].eq(selected)
    history = pd.concat(histories, ignore_index=True)
    history["selected"] = history["candidate"].eq(selected)
    return best, history, search


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_ann(model: ANNModel, directory: Path = RESULTS_DIR) -> None:
    """Save CPU state_dict and fitted sklearn objects; never refit on load."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / "ann_model.pt"
    preprocessing = directory / "ann_preprocessing.joblib"
    torch.save(model.network.state_dict(), weights)
    joblib.dump({"preprocessor": model.preprocessor,
                 "target_scaler": model.target_scaler}, preprocessing)
    first_layer = model.network[0]
    _write_json(directory / "ann_config.json", {
        "training": asdict(model.config), "input_dim": first_layer.in_features,
        "best_epoch": model.best_epoch, "activation": "ReLU",
        "output_activation": "linear", "loss": "MSE (train-standardised target)",
        "optimizer": "Adam", "target_scaling": "StandardScaler fitted on train",
        "early_stopping": "strictly lowest validation loss; patience epochs without improvement",
        "candidate_selection": "lowest validation MAE at restored checkpoint; fixed grid order breaks ties",
        "metadata": model.metadata,
        "file_sha256": {weights.name: file_sha256(weights),
                        preprocessing.name: file_sha256(preprocessing)},
    })


def load_ann(directory: Path = RESULTS_DIR, *, expected_manifest: dict | None = None) -> ANNModel:
    """Load our trusted local bundle. joblib must not load untrusted artifacts."""
    directory = Path(directory)
    saved = json.loads((directory / "ann_config.json").read_text(encoding="utf-8"))
    if expected_manifest is not None:
        verify_manifest(expected_manifest, saved["metadata"]["dataset"])
    for name in ("ann_model.pt", "ann_preprocessing.joblib"):
        if file_sha256(directory / name) != saved["file_sha256"][name]:
            raise ValueError(f"ANN bundle checksum mismatch: {name}")
    params = saved["training"]
    params["hidden_sizes"] = tuple(params["hidden_sizes"])
    config = ANNConfig(**params)
    # Inference also uses one CPU thread, as in the recorded training run.
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        net = _network(saved["input_dim"], config.hidden_sizes).cpu()
    net.load_state_dict(torch.load(directory / "ann_model.pt", map_location="cpu", weights_only=True))
    net.eval()
    objects = joblib.load(directory / "ann_preprocessing.joblib")
    return ANNModel(net, objects["preprocessor"], objects["target_scaler"],
                    config, saved["best_epoch"], saved["metadata"])


def plot_learning_curves(history: pd.DataFrame, search: pd.DataFrame,
                         directory: Path = FIGURES_DIR) -> None:
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    chosen = history[history["selected"]]
    best_epoch = int(chosen["best_epoch"].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, label in zip(axes, ("loss", "mae"),
                                  ("MSE (standardised target)", "MAE (MPG)")):
        ax.plot(chosen["epoch"], chosen[f"train_{metric}"], label="Train", color="#2878b5")
        ax.plot(chosen["epoch"], chosen[f"val_{metric}"], label="Validation",
                color="#d46b36", linestyle="--")
        ax.axvline(best_epoch, color="#555555", linestyle=":", label=f"Restored epoch {best_epoch}")
        ax.set(xlabel="Epoch", ylabel=label,
               title="Loss history" if metric == "loss" else "Error in original units")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"ANN learning curves: {chosen['candidate'].iloc[0]}")
    fig.tight_layout()
    fig.savefig(directory / "ann_learning_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh(search["candidate"], search["val_mae"],
            color=["#2878b5" if selected else "#a8b9c9" for selected in search["selected"]])
    ax.invert_yaxis()
    ax.set(xlabel="Validation MAE (MPG), lower is better",
           title="Four predeclared ANN configurations (restored checkpoints)")
    for i, value in enumerate(search["val_mae"]):
        ax.text(value + 0.015, i, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, float(search["val_mae"].max()) * 1.15)
    fig.tight_layout()
    fig.savefig(directory / "ann_search_validation.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_experiment():
    """Full fixed-grid experiment. Writes only new endterm ANN artifacts."""
    train, val, manifest = load_training_data()
    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    model, history, search = select_ann(X_train, y_train, X_val, y_val)
    model.metadata = {
        "dataset": manifest, "versions": library_versions(),
        "seeds": {"python": 42, "numpy": 42, "torch": 42, "batch_shuffle": 42},
        "device": "cpu", "torch_threads": 1, "deterministic_algorithms": True,
        "source_sha256": {name: file_sha256(Path(__file__).with_name(name)) for name in
                          ("neural_network.py", "data.py", "preprocessing.py", "evaluation.py")},
        "test_evaluated": False,
        "test_history": "Previously used midterm holdout; not a new independent endterm test",
    }
    predictions = model.predict(X_val)
    metrics = {
        "model": "Feedforward ANN", "selected_candidate": search.iloc[0]["candidate"],
        "best_epoch": model.best_epoch,
        "primary_metric": "mae", "units": {"mae": "MPG", "rmse": "MPG", "r2": "unitless"},
        "train": compute_metrics(y_train, model.predict(X_train)),
        "validation": compute_metrics(y_val, predictions),
        "n_train": len(train), "n_val": len(val), "test_evaluated": False,
        "metadata": model.metadata,
    }
    save_ann(model)
    history.to_csv(RESULTS_DIR / "ann_history.csv", index=False)
    search.to_csv(RESULTS_DIR / "ann_search.csv", index=False)
    pd.DataFrame({"row_id": val["row_id"].to_numpy(), "y_true": y_val.to_numpy(),
                  "y_pred": predictions}).to_csv(RESULTS_DIR / "ann_predictions_validation.csv", index=False)
    _write_json(RESULTS_DIR / "ann_metrics.json", metrics)
    plot_learning_curves(history, search)
    return model, history, search, metrics


def verify_saved_run() -> dict:
    """Check fingerprint, restored bundle, all validation row IDs/targets/metrics."""
    train, val, manifest = load_training_data()
    model = load_ann(expected_manifest=manifest)
    X_val, y_val = split_xy(val)
    saved = pd.read_csv(RESULTS_DIR / "ann_predictions_validation.csv")
    if saved["row_id"].duplicated().any():
        raise ValueError("Duplicate validation prediction row_id")
    np.testing.assert_array_equal(saved["row_id"], val["row_id"])
    np.testing.assert_array_equal(saved["y_true"], y_val)
    predictions = model.predict(X_val)
    np.testing.assert_allclose(saved["y_pred"], predictions, rtol=0, atol=1e-10)
    metrics = json.loads((RESULTS_DIR / "ann_metrics.json").read_text(encoding="utf-8"))
    verify_manifest(manifest, metrics["metadata"]["dataset"])
    if metrics["test_evaluated"] or model.metadata["test_evaluated"]:
        raise ValueError("Development bundle must not contain ANN test evaluation")
    for name, actual in compute_metrics(y_val, predictions).items():
        np.testing.assert_allclose(metrics["validation"][name], actual, rtol=0, atol=1e-10)
    return {"snapshot_and_split": "match", "validation_rows": len(val),
            "predictions_and_metrics": "match", "test_evaluated": False}


def smoke_check() -> dict:
    """Three epochs on actual train/validation rows; no official artifacts written."""
    train, val, _ = load_training_data()
    X_train, y_train = split_xy(train.iloc[:512])
    X_val, y_val = split_xy(val.iloc[:128])
    model, history = train_ann(X_train, y_train, X_val, y_val,
                               ANNConfig(max_epochs=3))
    predictions = model.predict(X_val)
    assert len(predictions) == len(y_val) and np.isfinite(predictions).all()
    return {"smoke_epochs": len(history), "train_rows": len(X_train),
            "val_rows": len(X_val), "test_evaluated": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        result = smoke_check()
    elif args.verify:
        result = verify_saved_run()
    else:
        _, _, search, metrics = run_experiment()
        print(search.to_string(index=False))
        result = {"validation": metrics["validation"], "train": metrics["train"],
                  "best_epoch": metrics["best_epoch"], "test_evaluated": False}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
