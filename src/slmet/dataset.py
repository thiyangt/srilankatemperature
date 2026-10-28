"""Build and persist the full tidy dataset across a directory of map images."""
from __future__ import annotations

import os

import pandas as pd

from .extract import DIAGNOSTIC_COLUMNS, Readers, extract_image
from .ingest import _file_hash
from .stations import discover_image_paths, load_registry

TIDY_COLUMNS = [
    "date", "measure", "station", "latitude", "longitude", "temperature_c",
    "source_file",
]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BUNDLED_DATASET_PATH = os.path.join(_DATA_DIR, "temperatures.csv")


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
    skipped = []
    seen_hashes: set[str] = set()
    for i, path in enumerate(paths):
        # repeat downloads of the same map (e.g. "x (1).jpg") would
        # otherwise double that day's rows
        content_hash = _file_hash(path)
        if content_hash in seen_hashes:
            skipped.append(path)
            continue
        seen_hashes.add(content_hash)
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

    if skipped and verbose:
        print(f"Skipped {len(skipped)} duplicate image(s): "
              + ", ".join(os.path.basename(p) for p in skipped))
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


def load_temperatures(wide: bool = False) -> pd.DataFrame:
    """Load the tidy dataset that ships with the package, extracted from
    every published map so far -- no images or CSV download needed.

    ``wide=True`` returns the :func:`to_wide` view (``tmin_c``/``tmax_c``
    columns) instead of the long format.
    """
    df = load_dataset(BUNDLED_DATASET_PATH)
    return to_wide(df) if wide else df


def to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience view: one row per (date, station), min/max as columns.

    Latitude/longitude are fitted per image, so they differ slightly between
    a day's Minimum and Maximum maps; they're averaged here rather than used
    as pivot keys (which would split min and max onto separate rows).
    """
    pivot = df.pivot_table(index=["date", "station"], columns="measure",
                            values="temperature_c", aggfunc="first")
    pivot.columns.name = None
    pivot = pivot.rename(columns={"Minimum": "tmin_c", "Maximum": "tmax_c"})
    coords = df.groupby(["date", "station"])[["latitude", "longitude"]].mean().round(4)
    out = coords.join(pivot).reset_index()
    for col in ("tmin_c", "tmax_c"):
        if col not in out:
            out[col] = float("nan")
    return out[["date", "station", "latitude", "longitude", "tmin_c", "tmax_c"]]
