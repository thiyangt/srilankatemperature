"""Find which (date, measure) maps haven't been downloaded yet.

Works two ways:

* :func:`find_missing_in_dir` -- scans a directory of images already renamed
  by :func:`slmet.ingest.rename_and_copy` (``"<Measure> Temperature -
  YYYY-MM-DD.jpg"``) and diffs the parsed dates against the full expected
  date range.
* :func:`find_missing_in_dataset` -- same idea, but against a tidy dataset
  DataFrame (e.g. from :func:`slmet.dataset.build_dataset`).
"""
from __future__ import annotations

import datetime as _dt
import os
import re

import pandas as pd

from .config import MEASURES

_NAME_RE = re.compile(r"^(Minimum|Maximum) Temperature - (\d{4}-\d{2}-\d{2})\.jpe?g$")


def _date_range(start: str | _dt.date, end: str | _dt.date) -> list[_dt.date]:
    if isinstance(start, str):
        start = _dt.date.fromisoformat(start)
    if isinstance(end, str):
        end = _dt.date.fromisoformat(end)
    n = (end - start).days
    return [start + _dt.timedelta(days=i) for i in range(n + 1)]


def _expected_grid(start, end) -> pd.DataFrame:
    dates = _date_range(start, end)
    return pd.DataFrame(
        [(d.isoformat(), m) for d in dates for m in MEASURES],
        columns=["date", "measure"],
    )


def find_missing_in_dir(renamed_dir: str, start: str, end: str) -> pd.DataFrame:
    """Diff the renamed-image directory against the full expected
    (date, measure) grid for [start, end] inclusive."""
    present = set()
    for fname in os.listdir(renamed_dir):
        m = _NAME_RE.match(fname)
        if m:
            present.add((m.group(2), m.group(1)))

    expected = _expected_grid(start, end)
    missing_mask = ~expected.apply(lambda r: (r["date"], r["measure"]) in present, axis=1)
    return expected[missing_mask].reset_index(drop=True)


def find_missing_in_dataset(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Same as :func:`find_missing_in_dir` but against a tidy dataset with
    ``date``/``measure`` columns."""
    present = set(zip(df["date"].astype(str), df["measure"]))
    expected = _expected_grid(start, end)
    missing_mask = ~expected.apply(lambda r: (r["date"], r["measure"]) in present, axis=1)
    return expected[missing_mask].reset_index(drop=True)
