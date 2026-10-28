"""Canonical station registry.

Every map is generated from the same fixed table of ~24 weather collection
centres, so a given station's bubble lands at (very nearly) the same
longitude/latitude every single day. That regularity means station identity
is far more reliably recovered from *position*, pooled across the whole
corpus, than from OCR-ing the station-name label on any single image (label
placement varies side-to-side and is the hardest text on the page to read).

:func:`build_registry` samples bubbles across many images, clusters them by
position (k-means, k = number of known stations), and names each cluster by
a majority vote of per-image label OCR over its members. The result is
persisted once as ``data/stations.csv`` and shipped with the package;
:func:`assign_station` then just does nearest-cluster lookup per bubble.
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .bubbles import detect_bubbles, read_bubble_label
from .calibration import calibrate
from .config import IMAGE_EXTENSIONS, KNOWN_STATIONS
from .imaging import detect_frame, load_gray_rgb
from .ocr import DigitReader, VocabularyReader

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_REGISTRY_PATH = os.path.join(_DATA_DIR, "stations.csv")


def build_registry(image_paths: list[str], n_stations: int = len(KNOWN_STATIONS),
                    max_images: int | None = 60, seed: int = 0) -> pd.DataFrame:
    """Cluster bubble positions across a sample of images into a named
    station registry with columns: station, longitude, latitude, n_obs,
    name_votes, name_confidence.
    """
    rng = np.random.default_rng(seed)
    paths = list(image_paths)
    if max_images is not None and len(paths) > max_images:
        paths = list(rng.choice(paths, size=max_images, replace=False))

    digit_reader = DigitReader()
    vocab_reader = VocabularyReader(KNOWN_STATIONS)

    points = []  # (lon, lat)
    labels = []  # OCR-guessed name per point
    label_scores = []

    for path in paths:
        try:
            gray, rgb = load_gray_rgb(path)
            frame = detect_frame(gray)
            gt = calibrate(gray, frame, digit_reader)
            bubbles = detect_bubbles(gray, frame)
        except Exception:
            continue
        for b in bubbles:
            lon, lat = gt.to_lonlat(b.cx, b.cy)
            word, score = read_bubble_label(gray, b, vocab_reader)
            points.append((lon, lat))
            labels.append(word)
            label_scores.append(score)

    if len(points) < n_stations:
        raise ValueError(f"Only found {len(points)} bubbles across the sample; "
                          f"need at least {n_stations} to cluster stations.")

    X = np.array(points)
    km = KMeans(n_clusters=n_stations, n_init=10, random_state=seed).fit(X)

    rows = []
    for cid in range(n_stations):
        idx = np.where(km.labels_ == cid)[0]
        centroid = X[idx].mean(axis=0)
        votes = Counter(labels[i] for i in idx if label_scores[i] > 0.3 and labels[i])
        if votes:
            name, n_votes = votes.most_common(1)[0]
            confidence = n_votes / len(idx)
        else:
            name, confidence = f"unknown_{cid}", 0.0
        rows.append({
            "station": name,
            "longitude": round(float(centroid[0]), 4),
            "latitude": round(float(centroid[1]), 4),
            "n_obs": len(idx),
            "name_confidence": round(confidence, 3),
        })

    df = pd.DataFrame(rows).sort_values("latitude", ascending=False).reset_index(drop=True)
    _dedupe_names(df)
    return df


def _dedupe_names(df: pd.DataFrame) -> None:
    """If two clusters land on the same OCR'd name (or both unresolved),
    disambiguate in place rather than silently colliding."""
    seen: dict[str, int] = {}
    for i, name in enumerate(df["station"]):
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            df.loc[i, "station"] = f"{name}_{seen[name]}"


def save_registry(df: pd.DataFrame, path: str = DEFAULT_REGISTRY_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def load_registry(path: str = DEFAULT_REGISTRY_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No station registry at {path}. Build one first with "
            f"slmet.stations.build_registry() and save_registry()."
        )
    return pd.read_csv(path)


def assign_station(lon: float, lat: float, registry: pd.DataFrame,
                    max_distance_deg: float = 0.15) -> str | None:
    """Nearest-cluster lookup: which known station is this bubble?"""
    d = np.hypot(registry["longitude"].to_numpy() - lon, registry["latitude"].to_numpy() - lat)
    i = int(np.argmin(d))
    if d[i] > max_distance_deg:
        return None
    return str(registry.iloc[i]["station"])


def discover_image_paths(image_dir: str) -> list[str]:
    return sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir)
                  if f.lower().endswith(IMAGE_EXTENSIONS))
