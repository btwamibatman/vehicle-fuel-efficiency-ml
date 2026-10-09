"""Tune Decision Tree and KNN, compare all models on validation, analyse errors (Askhat).

    python -m vehicle_efficiency.compare            # tuning + validation comparison + plots
    python -m vehicle_efficiency.compare --test     # also the single final test evaluation

Tuning uses 5-fold CV on the training split only (duplicates kept in one fold).
Model selection uses the validation split. ``--test`` evaluates the model with
the lowest validation MAE on the test split; run it once, after the choice is fixed.
"""
from __future__ import annotations

import argparse
import sys

from .data import (RANDOM_SEED, _duplicate_groups, download_raw, load_dataset,
                   make_splits, save_split_assignments)
from .evaluation import (RESULTS_DIR, comparison_table, error_by_group, evaluate,
                         largest_errors, plot_actual_vs_predicted, plot_residuals,
                         save_results, tune_pipeline)
from .models import (KNN_PARAM_GRID, TREE_PARAM_GRID, build_baseline, build_decision_tree,
                     build_knn, build_linear_regression)
from .preprocessing import split_xy

# Khamza kept displacement_per_cylinder (project.ipynb section 5), so every real
# model gets the same engineered feature set.
CANDIDATES = {
    "Baseline (mean)": (build_baseline, {}),
    "Linear Regression": (lambda: build_linear_regression(engineered=True), {}),
    "Decision Tree": (lambda: build_decision_tree(engineered=True), TREE_PARAM_GRID),
    "KNN": (lambda: build_knn(engineered=True), KNN_PARAM_GRID),
}


def run(seed: int = RANDOM_SEED, test: bool = False) -> dict:
    download_raw()
    df = load_dataset()
    splits = make_splits(df, seed)
    save_split_assignments(df, splits)
    train, val = splits["train"], splits["val"]
    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    groups = _duplicate_groups(train)

    fitted, val_metrics, summary = {}, {}, {}
    for name, (builder, grid) in CANDIDATES.items():
        # An empty grid is a single candidate: plain CV on the same folds as the tuned models.
        pipe, cv, table = tune_pipeline(builder(), grid, X_train, y_train, groups=groups)
        fitted[name] = pipe
        val_metrics[name] = evaluate(pipe, X_val, y_val)
        summary[name] = {"cv_train": cv, "train": evaluate(pipe, X_train, y_train),
                         "validation": val_metrics[name]}
        if grid:
            slug = name.lower().replace(" ", "_")
            table.assign(params=table["params"].astype(str)).to_csv(
                RESULTS_DIR / f"{slug}_tuning.csv", index=False)
        print(f"[{name}] CV MAE {cv['cv_mae_mean']:.3f} +/- {cv['cv_mae_std']:.3f} | "
              f"validation MAE {val_metrics[name]['mae']:.3f} MPG"
              + (f" | best {cv['best_params']}" if grid else ""))

    table = comparison_table(val_metrics)
    table.round(3).to_csv(RESULTS_DIR / "model_comparison_validation.csv")
    best = table.index[0]
    preds = {n: p.predict(X_val) for n, p in fitted.items() if n != "Baseline (mean)"}
    plot_actual_vs_predicted(y_val, preds, "actual_vs_predicted_validation.png")
    plot_residuals(y_val, preds, "residuals_validation.png")
    largest_errors(val, y_val, preds[best]).to_csv(
        RESULTS_DIR / "largest_errors_validation.csv", index=False)
    error_by_group(val, y_val, preds[best], "vehicle_class").to_csv(
        RESULTS_DIR / "error_by_vehicle_class_validation.csv")

    result = {"seed": seed, "n_train": len(X_train), "n_val": len(X_val),
              "units": "MAE and RMSE in miles per gallon; R2 unitless",
              "selected_by": "lowest validation MAE", "best_model": best,
              "models": summary}
    if test:
        X_test, y_test = split_xy(splits["test"])
        result["test"] = {"model": best, "n_test": len(X_test),
                          "metrics": evaluate(fitted[best], X_test, y_test)}
        save_results(result["test"], "final_test.json")
    save_results(result, "model_comparison.json")
    print(table.round(3).to_string())
    print(f"Best on validation: {best}")
    if test:
        t = result["test"]["metrics"]
        print(f"[{best}] TEST  MAE={t['mae']:.3f} MPG  RMSE={t['rmse']:.3f} MPG  R2={t['r2']:.3f}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--test", action="store_true",
                        help="evaluate the validation-selected model once on the test split")
    args = parser.parse_args(argv)
    run(args.seed, args.test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
