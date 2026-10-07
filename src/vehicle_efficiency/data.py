"""Data acquisition, loading, schema checks and splitting.

Dataset: US EPA / DOE fueleconomy.gov "vehicles.csv" (recent model years only).
Run ``python -m vehicle_efficiency.data download`` to fetch the raw file into
``data/raw/``. Raw files are never modified by this project.

Owner: Khamza (Aktore documents the dataset in ``data/README.md``).
"""
from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_FILE = RAW_DIR / "vehicles.csv"

RAW_URL = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv.zip"

# Scope decisions (stateless, applied before splitting, documented in data/README.md)
MIN_YEAR = 2015  # "recent vehicles only"; change here to widen/narrow the range
EXCLUDED_ATV_TYPES = {"EV", "FCV", "eFCV", "Plug-in Hybrid", "Bifuel (CNG)", "CNG"}
EXCLUDED_FUEL_TYPES = {"Electricity", "Hydrogen", "Natural Gas"}

# Raw EPA column -> project column name
RAW_TO_PROJECT = {
    "year": "model_year",
    "make": "make",
    "model": "model",
    "comb08": "mpg",
    "cylinders": "cylinders",
    "displ": "displacement",
    "drive": "drive",
    "trany": "trany",
    "VClass": "vehicle_class",
    "fuelType1": "fuel_type",
    "atvType": "atv_type",
    "tCharger": "t_charger",
    "sCharger": "s_charger",
}
TARGET = "mpg"  # EPA combined MPG (55% city / 45% highway); higher = more efficient

# Columns the final (cleaned) frame must have
COLUMNS = [
    "model_year", "make", "model", "mpg", "cylinders", "displacement", "drive",
    "vehicle_class", "fuel_type", "transmission", "hybrid", "turbo", "supercharged",
]
NUMERIC_COLUMNS = ["model_year", "mpg", "cylinders", "displacement",
                   "hybrid", "turbo", "supercharged"]
# Rows identical on all of these are "exact duplicates" for the split
GROUP_COLUMNS = COLUMNS

RANDOM_SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
SPLIT_NAMES = ("train", "val", "test")


def download_raw(force: bool = False) -> Path:
    """Download and unzip the EPA ``vehicles.csv`` into ``data/raw/`` (unchanged)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and not force:
        return RAW_FILE
    with urllib.request.urlopen(RAW_URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        RAW_FILE.write_bytes(zf.read("vehicles.csv"))
    return RAW_FILE


def load_raw(path: Path | str = RAW_FILE) -> pd.DataFrame:
    """Read the raw file, keeping only the columns this project uses.

    No rows are dropped and no values are changed. ``row_id`` is the 0-based row
    number in the raw file, so every model sees exactly the same rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m vehicle_efficiency.data download"
        )
    df = pd.read_csv(path, usecols=list(RAW_TO_PROJECT), low_memory=False)
    df.insert(0, "row_id", range(len(df)))
    return df.rename(columns=RAW_TO_PROJECT)


def prepare(df: pd.DataFrame, min_year: int = MIN_YEAR) -> pd.DataFrame:
    """Apply the scope filter and derive simple flags. Stateless: learns nothing.

    Keeps model years >= ``min_year`` and drops vehicles whose efficiency is
    reported in MPGe instead of MPG (EV, fuel cell, plug-in hybrid) or that run
    on CNG/hydrogen. Original ``row_id`` values are preserved.
    """
    df = df[df["model_year"] >= min_year]
    df = df[~df["atv_type"].isin(EXCLUDED_ATV_TYPES)]
    df = df[~df["fuel_type"].isin(EXCLUDED_FUEL_TYPES)].copy()

    trany = df["trany"].fillna("")
    df["transmission"] = np.select(
        [trany.str.startswith("Manual"), trany.str.startswith("Automatic")],
        ["Manual", "Automatic"], default="Other",
    )
    df["hybrid"] = (df["atv_type"] == "Hybrid").astype(int)
    df["turbo"] = (df["t_charger"] == "T").astype(int)
    df["supercharged"] = (df["s_charger"] == "S").astype(int)
    return df[["row_id"] + COLUMNS].reset_index(drop=True)


def validate_schema(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` listing every schema problem found."""
    missing_cols = [c for c in COLUMNS + ["row_id"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    if len(df) == 0:
        raise ValueError("Dataset is empty")
    problems: list[str] = []
    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"{col} is not numeric (dtype={df[col].dtype})")
    if df[TARGET].isna().any():
        problems.append(f"target '{TARGET}' contains missing values")
    if (df[TARGET].dropna() <= 0).any():
        problems.append(f"target '{TARGET}' has non-positive values")
    if (df["displacement"].dropna() <= 0).any():
        problems.append("displacement has non-positive values")
    if (df["cylinders"].dropna() <= 0).any():
        problems.append("cylinders has non-positive values")
    if df["fuel_type"].isin(EXCLUDED_FUEL_TYPES).any():
        problems.append("out-of-scope fuel types present (MPGe vehicles)")
    if problems:
        raise ValueError("Schema check failed: " + "; ".join(problems))


def _duplicate_groups(df: pd.DataFrame) -> pd.Series:
    """Group id per row; rows identical on every project column share one id."""
    return df.groupby(GROUP_COLUMNS, dropna=False, sort=False).ngroup()


def make_splits(df: pd.DataFrame, seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    """Reproducible ~70/15/15 train/val/test split.

    Exact duplicate records are kept together (GroupShuffleSplit) so a duplicate
    can never appear in two splits. Because whole groups are moved, sizes can
    differ slightly from the targets.
    """
    groups = _duplicate_groups(df)
    outer = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRAC, random_state=seed)
    train_idx, rest_idx = next(outer.split(df, groups=groups))
    rest = df.iloc[rest_idx]
    inner = GroupShuffleSplit(
        n_splits=1, train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC), random_state=seed
    )
    val_rel, test_rel = next(inner.split(rest, groups=groups.iloc[rest_idx]))
    return {
        "train": df.iloc[train_idx].copy(),
        "val": rest.iloc[val_rel].copy(),
        "test": rest.iloc[test_rel].copy(),
    }


def load_dataset() -> pd.DataFrame:
    """Raw file -> scoped, derived, schema-checked frame (all rows, no splitting)."""
    df = prepare(load_raw())
    validate_schema(df)
    return df


def load_split(name: str, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return one split ('train', 'val' or 'test') of the prepared dataset.

    Use 'train' for EDA and modelling decisions, 'val' for model selection and
    keep 'test' for the very end.
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    return make_splits(load_dataset(), seed)[name]


def save_split_assignments(df: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> Path:
    """Write ``data/processed/split_assignments.csv`` (raw row_id -> split)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.concat(
        [pd.DataFrame({"row_id": s["row_id"], "split": n}) for n, s in splits.items()]
    ).sort_values("row_id")
    path = PROCESSED_DIR / "split_assignments.csv"
    out.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset utilities")
    parser.add_argument("command", choices=["download"])
    parser.add_argument("--force", action="store_true", help="re-download")
    args = parser.parse_args()
    if args.command == "download":
        path = download_raw(force=args.force)
        print(f"Raw data ready: {path}")


if __name__ == "__main__":
    main()
