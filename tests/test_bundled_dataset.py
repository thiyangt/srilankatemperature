from __future__ import annotations

import pandas as pd

from slmet import load_temperatures, to_wide
from slmet.dataset import TIDY_COLUMNS


def test_bundled_dataset_is_tidy():
    df = load_temperatures()
    assert list(df.columns) == TIDY_COLUMNS
    assert set(df["measure"]) == {"Minimum", "Maximum"}
    assert df["date"].min() == "2026-01-01"
    # one row per (date, measure, station)
    assert not df.duplicated(["date", "measure", "station"]).any()


def test_bundled_dataset_wide_view():
    wide = load_temperatures(wide=True)
    assert {"date", "station", "tmin_c", "tmax_c"} <= set(wide.columns)
    assert not wide.duplicated(["date", "station"]).any()
    both = wide.dropna(subset=["tmin_c", "tmax_c"])
    assert len(both) > 0
    assert (both["tmax_c"] >= both["tmin_c"]).mean() > 0.99


def test_to_wide_joins_min_max_despite_coordinate_drift():
    df = pd.DataFrame([
        ("2026-01-01", "Minimum", "Colombo", 6.9006, 79.8996, 24.0, "a.jpg"),
        ("2026-01-01", "Maximum", "Colombo", 6.9012, 79.8994, 31.0, "b.jpg"),
        ("2026-01-02", "Maximum", "Colombo", 6.9010, 79.8995, 30.5, "c.jpg"),
    ], columns=TIDY_COLUMNS)
    wide = to_wide(df)
    assert list(wide.columns) == ["date", "station", "latitude", "longitude", "tmin_c", "tmax_c"]
    assert len(wide) == 2
    day1 = wide.iloc[0]
    assert (day1["tmin_c"], day1["tmax_c"]) == (24.0, 31.0)
    assert day1["latitude"] == 6.9009
    assert pd.isna(wide.iloc[1]["tmin_c"])
