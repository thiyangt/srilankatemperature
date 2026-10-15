"""Top-level per-image extraction: image path in, tidy records out.

This is the function referenced in the brief: "you should write a function
to extract location and temperature from the above maps [...] the function
should be able to store data in tidy format." :func:`extract_image` reads
one map end-to-end (title -> date/measure, axes -> geo transform, bubbles ->
station/temperature) and returns one row per station as plain dicts, ready
to hand to :class:`pandas.DataFrame`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from .bubbles import detect_bubbles, read_bubble_label, read_bubble_value
from .calibration import calibrate
from .config import KNOWN_STATIONS
from .imaging import detect_frame, load_gray_rgb
from .ocr import DigitReader, VocabularyReader
from .stations import assign_station, load_registry
from .title import read_title

# Per-record QA columns, only included when ``diagnostics=True``.
DIAGNOSTIC_COLUMNS = ["label_ocr_guess", "label_ocr_score", "station_source"]


@dataclass
class Readers:
    """Bundle of OCR/vocabulary readers, built once and reused across many
    images (building font templates per image would be wasteful)."""

    digit: DigitReader = field(default_factory=DigitReader)
    station_vocab: VocabularyReader = field(
        default_factory=lambda: VocabularyReader(KNOWN_STATIONS))


def extract_image(path: str, registry: pd.DataFrame | None = None,
                   readers: Readers | None = None,
                   min_label_confidence: float = 0.0,
                   diagnostics: bool = False) -> list[dict]:
    """Extract one tidy record per detected station bubble in ``path``.

    Each record has: date, measure, station, latitude, longitude,
    temperature_c, source_file. With ``diagnostics=True`` it also has
    label_ocr_guess / label_ocr_score (the station name read from the text
    beside the bubble, and its match score) and station_source ("position"
    or "ocr_fallback") -- useful for auditing, but the station itself is
    resolved by position, so these aren't needed for analysis.
    """
    readers = readers or Readers()
    if registry is None:
        registry = load_registry()

    gray, rgb = load_gray_rgb(path)
    frame = detect_frame(gray)
    title = read_title(gray, frame, readers.digit)
    geo = calibrate(gray, frame, readers.digit)
    bubbles = detect_bubbles(gray, frame)

    records = []
    for b in bubbles:
        lon, lat = geo.to_lonlat(b.cx, b.cy)
        station = assign_station(lon, lat, registry)
        temp = read_bubble_value(rgb, b, readers.digit)

        # The label read only matters as a fallback (or when asked for), so
        # skip it for bubbles that matched a known position.
        label_word, label_score = "", -1.0
        if station is None or diagnostics:
            label_word, label_score = read_bubble_label(gray, b, readers.station_vocab)

        station_source = "position"
        if station is None:
            # Fall back to the per-image OCR guess if this bubble's position
            # doesn't match any known cluster (e.g. a station added later).
            station = label_word or f"unresolved_{round(lon, 3)}_{round(lat, 3)}"
            station_source = "ocr_fallback"

        records.append({
            "date": title.date.isoformat(),
            "measure": title.measure,
            "station": station,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "temperature_c": temp,
            "source_file": os.path.basename(path),
            "label_ocr_guess": label_word,
            "label_ocr_score": round(label_score, 3),
            "station_source": station_source,
        })

    if min_label_confidence > 0:
        records = [r for r in records if r["label_ocr_score"] >= min_label_confidence
                   or r["station_source"] == "position"]
    if not diagnostics:
        for r in records:
            for col in DIAGNOSTIC_COLUMNS:
                del r[col]
    return records
