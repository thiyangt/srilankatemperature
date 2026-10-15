"""Build and persist the full tidy dataset across a directory of map images."""
from __future__ import annotations

import os

import pandas as pd

from .extract import DIAGNOSTIC_COLUMNS, Readers, extract_image
from .stations import discover_image_paths, load_registry

TIDY_COLUMNS = [
    "date", "measure", "station", "latitude", "longitude", "temperature_c",
    "source_file",
]


def build_dataset(image_dir: str, registry: pd.DataFrame | None = None,
                   cache_path: str | None = None, verbose: bool = True,
                   diagnostics: bool = False) -> pd.DataFrame:
    """Extract every image in ``image_dir`` into one long, tidy DataFrame.

    ``diagnostics=True`` adds the per-record QA columns described in
    :func:`slmet.extract.extract_image`.
    """
    registry = registry if registry is not None else load_registry()
    readers = Readers()
    paths = discover_image_paths(image_dir)

    rows = []
    errors = []
    for i, path in enumerate(paths):
        try:
            rows.extend(extract_image(path, registry=registry, readers=readers,
                                      diagnostics=diagnostics))
        except Exception as e:
            errors.append((path, str(e)))
        if verbose and (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(paths)} images "
                  f"({len(errors)} errors so far)")

    df = pd.DataFrame(rows, columns=TIDY_COLUMNS + (DIAGNOSTIC_COLUMNS if diagnostics else []))
    df = df.sort_values(["date", "measure", "station"]).reset_index(drop=True)

    if errors and verbose:
        print(f"{len(errors)} image(s) failed to extract:")
        for path, msg in errors[:20]:
            print(f"  {os.path.basename(path)}: {msg}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_csv(cache_path, index=False)

    return df


def load_dataset(cache_path: str) -> pd.DataFrame:
    return pd.read_csv(cache_path)


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience view: one row per (date, station), min/max as columns."""
    pivot = df.pivot_table(index=["date", "station", "latitude", "longitude"],
                            columns="measure", values="temperature_c", aggfunc="first")
    pivot = pivot.reset_index()
    pivot.columns.name = None
    rename = {"Minimum": "tmin_c", "Maximum": "tmax_c"}
    return pivot.rename(columns=rename)
