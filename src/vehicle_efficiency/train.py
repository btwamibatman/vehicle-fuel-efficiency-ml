"""Command line entry point.

    python -m vehicle_efficiency.train                    # baseline, validation metrics
    python -m vehicle_efficiency.train --model linear_regression --cv [--engineered]
    python -m vehicle_efficiency.train --model knn --cv   # once a model is implemented

Fits on the training split, reports on the validation split. The test split is
intentionally not used here; it is evaluated once, at the end, in project.ipynb.
"""
from __future__ import annotations

import argparse
import sys

from .data import (RANDOM_SEED, download_raw, load_dataset, make_splits,
                   save_split_assignments)
from .evaluation import cross_validate_pipeline, evaluate, save_results
from .models import MODEL_BUILDERS
from .preprocessing import split_xy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODEL_BUILDERS), default="baseline")
    parser.add_argument("--cv", action="store_true", help="also run 5-fold CV on train")
    parser.add_argument("--engineered", action="store_true",
                        help="linear_regression only: add displacement_per_cylinder")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args(argv)

    download_raw()  # no-op if the raw file already exists
    df = load_dataset()  # scope filter + schema check
    splits = make_splits(df, args.seed)
    save_split_assignments(df, splits)
    print({k: len(v) for k, v in splits.items()}, "rows per split")

    if args.engineered and args.model != "linear_regression":
        parser.error("--engineered currently applies to --model linear_regression only")
    kwargs = {"engineered": True} if args.engineered else {}
    label = args.model + ("_engineered" if args.engineered else "")
    try:
        pipeline = MODEL_BUILDERS[args.model](**kwargs)
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        return 1

    X_train, y_train = split_xy(splits["train"])
    X_val, y_val = split_xy(splits["val"])
    pipeline.fit(X_train, y_train)

    result = {
        "model": label,
        "seed": args.seed,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "units": "MAE and RMSE in miles per gallon; R2 unitless",
        "train": evaluate(pipeline, X_train, y_train),
        "validation": evaluate(pipeline, X_val, y_val),
    }
    if args.cv:
        cv = cross_validate_pipeline(pipeline, X_train, y_train, seed=args.seed)
        result["cv_train"] = cv
    path = save_results(result, f"{label}_validation.json")

    v = result["validation"]
    print(f"[{label}] validation  MAE={v['mae']:.3f} MPG  "
          f"RMSE={v['rmse']:.3f} MPG  R2={v['r2']:.3f}")
    if args.cv:
        c = result["cv_train"]
        print(f"[{label}] 5-fold CV (train)  MAE={c['mae_mean']:.3f} +/- {c['mae_std']:.3f} MPG")
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
