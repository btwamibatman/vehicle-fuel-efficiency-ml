"""Checks for schema validation, scope filtering, split overlap and the real dataset."""
import pandas as pd
import pytest

from vehicle_efficiency.data import (COLUMNS, GROUP_COLUMNS, MIN_YEAR, RAW_FILE,
                                     RAW_TO_PROJECT, load_dataset, make_splits,
                                     prepare, validate_schema)


def tiny_frame(n=200, n_duplicates=20):
    """Small hand-made prepared frame (test fixture only; not used for any result)."""
    df = pd.DataFrame({
        "model_year": [2015 + i % 10 for i in range(n)],
        "make": [f"make{i % 7}" for i in range(n)],
        "model": [f"model{i}" for i in range(n)],
        "mpg": [15.0 + i % 30 for i in range(n)],
        "cylinders": [4.0 + 2 * (i % 3) for i in range(n)],
        "displacement": [1.5 + (i % 40) / 10 for i in range(n)],
        "drive": ["Front-Wheel Drive", "All-Wheel Drive"] * (n // 2),
        "vehicle_class": ["Compact Cars", "Midsize Cars", "Standard SUV 4WD"] * (n // 3) + ["Compact Cars"] * (n % 3),
        "fuel_type": ["Regular Gasoline"] * n,
        "transmission": ["Automatic", "Manual"] * (n // 2),
        "hybrid": [i % 5 == 0 for i in range(n)],
        "turbo": [i % 4 == 0 for i in range(n)],
        "supercharged": [0] * n,
    })
    for c in ("hybrid", "turbo"):
        df[c] = df[c].astype(int)
    df = pd.concat([df, df.iloc[:n_duplicates]], ignore_index=True)  # exact duplicates
    df.insert(0, "row_id", range(len(df)))
    return df


def raw_like_frame():
    """Frame shaped like the renamed raw EPA columns, including out-of-scope rows."""
    return pd.DataFrame({
        "row_id": range(6),
        "model_year": [1990, 2016, 2020, 2020, 2021, 2022],
        "make": list("abcdef"), "model": list("ghijkl"),
        "mpg": [20, 30, 120, 25, 90, 35],
        "cylinders": [4, 4, None, 6, None, 4],
        "displacement": [2.0, 2.0, None, 3.0, None, 2.5],
        "drive": ["x"] * 6, "trany": ["Manual 5-spd", "Automatic 8-spd", None,
                                      "Automatic (S6)", None, "Auto(AV-S6)"],
        "vehicle_class": ["c"] * 6,
        "fuel_type": ["Regular Gasoline", "Regular Gasoline", "Electricity",
                      "Premium Gasoline", "Regular Gasoline", "Regular Gasoline"],
        "atv_type": [None, "Hybrid", "EV", None, "Plug-in Hybrid", None],
        "t_charger": [None, None, None, "T", None, None],
        "s_charger": [None] * 6,
    })


def test_prepare_filters_scope_and_derives_flags():
    out = prepare(raw_like_frame())
    assert list(out["row_id"]) == [1, 3, 5]  # old, EV and plug-in rows removed
    assert (out["model_year"] >= MIN_YEAR).all()
    assert list(out["hybrid"]) == [1, 0, 0]
    assert list(out["turbo"]) == [0, 1, 0]
    assert list(out["transmission"]) == ["Automatic", "Automatic", "Automatic"]


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("Automatic 8-spd", "Automatic"),
        ("Auto(AV-S6)", "Automatic"),
        ("Manual 5-spd", "Manual"),
        (None, "Other"),
    ],
)
def test_prepare_normalizes_epa_transmission_labels(raw_value, expected):
    df = raw_like_frame().iloc[[1]].copy()
    df["trany"] = raw_value

    assert prepare(df).iloc[0]["transmission"] == expected


def test_raw_to_project_names_cover_required_columns():
    for col in ("model_year", "mpg", "cylinders", "displacement"):
        assert col in RAW_TO_PROJECT.values()


def test_schema_accepts_valid_frame():
    validate_schema(tiny_frame())


def test_schema_rejects_missing_column():
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(tiny_frame().drop(columns=["displacement"]))


def test_schema_rejects_missing_target_and_out_of_scope_fuel():
    df = tiny_frame()
    df.loc[0, "mpg"] = None
    df.loc[1, "fuel_type"] = "Electricity"
    with pytest.raises(ValueError) as exc:
        validate_schema(df)
    assert "target" in str(exc.value) and "fuel types" in str(exc.value)


def test_splits_do_not_overlap_and_cover_everything():
    df = tiny_frame()
    s = make_splits(df)
    ids = {k: set(v["row_id"]) for k, v in s.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    assert sum(len(v) for v in ids.values()) == len(df)


def _keys(split):
    return set(map(tuple, split[GROUP_COLUMNS].astype(str).values))


def test_exact_duplicates_never_cross_splits():
    s = make_splits(tiny_frame())
    assert not _keys(s["train"]) & _keys(s["val"])
    assert not _keys(s["train"]) & _keys(s["test"])
    assert not _keys(s["val"]) & _keys(s["test"])


def test_split_handles_missing_values_and_duplicates_with_nan():
    df = tiny_frame()
    df.loc[0, "displacement"] = float("nan")  # row 0 and its duplicate both NaN
    df.loc[len(df) - 20, "displacement"] = float("nan")
    s = make_splits(df)
    assert not _keys(s["train"]) & _keys(s["val"])
    assert not _keys(s["train"]) & _keys(s["test"])
    assert sum(len(v) for v in s.values()) == len(df)


def test_split_is_reproducible_and_roughly_70_15_15():
    df = tiny_frame()
    a, b = make_splits(df, seed=1), make_splits(df, seed=1)
    assert list(a["train"]["row_id"]) == list(b["train"]["row_id"])
    n = len(df)
    assert abs(len(a["train"]) / n - 0.70) < 0.05
    assert abs(len(a["val"]) / n - 0.15) < 0.05


@pytest.mark.skipif(not RAW_FILE.exists(), reason="run: python -m vehicle_efficiency.data download")
def test_real_dataset_is_recent_and_valid():
    df = load_dataset()  # includes validate_schema
    assert len(df) > 1000
    assert df["model_year"].min() >= MIN_YEAR
    assert set(COLUMNS) <= set(df.columns)
