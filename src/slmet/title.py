"""Read the measure ("Minimum"/"Maximum") and date out of a map's title.

The title is always ``"<Minimum|Maximum> Temperatures Reported on
DD-MM-YYYY"`` on one line, with a smaller ``"(Issued by the Department of
Meteorology)"`` line underneath. Both are read from the strip of the image
above the plot's axes frame.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass

import numpy as np

from .config import FONT_REGULAR, MEASURES
from .imaging import Frame
from .ocr import DigitReader, VocabularyReader, _segment_components, dark_mask_from_gray


@dataclass
class TitleInfo:
    measure: str  # "Minimum" or "Maximum"
    date: _dt.date
    measure_score: float


def _title_line_band(gray: np.ndarray, frame: Frame) -> np.ndarray:
    """The larger of the two text lines above the axes frame is the title."""
    strip = gray[0:frame.top, :]
    dark = dark_mask_from_gray(strip)
    row_counts = dark.sum(axis=1)
    nonzero = row_counts > 0
    bands = []
    start = None
    for i, v in enumerate(nonzero):
        if v and start is None:
            start = i
        elif not v and start is not None:
            bands.append((start, i))
            start = None
    if start is not None:
        bands.append((start, len(nonzero)))
    if not bands:
        raise ValueError("No title text found above axes frame")
    # the title line has taller glyphs than the "(Issued by ...)" subtitle
    best = max(bands, key=lambda b: b[1] - b[0])
    return dark[best[0]:best[1], :]


def _first_word_crop(dark: np.ndarray, max_gap_ratio: float = 0.6) -> np.ndarray:
    """Crop to the first word, using the same letter- vs word-spacing
    threshold logic as :func:`_rightmost_run_crop`."""
    comps = _segment_components(dark, min_area=4)
    if not comps:
        return dark
    heights = [c[3] - c[1] for c in comps]
    med_h = float(np.median(heights))
    comps.sort(key=lambda c: c[0])
    keep_to = 0
    for i in range(1, len(comps)):
        gap = comps[i][0] - comps[i - 1][2]
        if gap > max_gap_ratio * med_h:
            break
        keep_to = i
    pad = 4
    x0 = max(0, comps[0][0] - pad)
    x1 = min(dark.shape[1], comps[keep_to][2] + pad)
    return dark[:, x0:x1]


def _rightmost_run_crop(dark: np.ndarray, max_gap_ratio: float = 0.6) -> np.ndarray:
    """Crop to the trailing run of glyphs with normal letter-spacing -- i.e.
    the date, which is the last thing on the title line."""
    comps = _segment_components(dark, min_area=4)
    if not comps:
        return dark
    heights = [c[3] - c[1] for c in comps]
    med_h = float(np.median(heights))
    comps.sort(key=lambda c: c[0])
    keep_from = len(comps) - 1
    for i in range(len(comps) - 1, 0, -1):
        gap = comps[i][0] - comps[i - 1][2]
        if gap > max_gap_ratio * med_h:
            keep_from = i
            break
        keep_from = i - 1
    pad = 4
    x0 = max(0, min(c[0] for c in comps[keep_from:]) - pad)
    x1 = min(dark.shape[1], max(c[2] for c in comps[keep_from:]) + pad)
    return dark[:, x0:x1]


def read_title(gray: np.ndarray, frame: Frame,
               digit_reader: DigitReader | None = None,
               measure_reader: VocabularyReader | None = None) -> TitleInfo:
    digit_reader = digit_reader or DigitReader()
    measure_reader = measure_reader or VocabularyReader(list(MEASURES), font_size=48,
                                                          canvas=(60, 260))

    line = _title_line_band(gray, frame)

    word_crop = _first_word_crop(line)
    measure, m_score = measure_reader.match((~word_crop).astype(np.uint8) * 255)

    date_crop = _rightmost_run_crop(line)
    # The date is always DD-MM-YYYY; the dash is a low, thin glyph that is
    # unreliable to classify individually, so just keep the digits (8 of
    # them) and use the known fixed positions instead of parsing dashes.
    digits = "".join(ch for ch in digit_reader.read_string(date_crop, border_margin=0)
                      if ch.isdigit())
    if len(digits) != 8:
        raise ValueError(f"Could not parse date from title (got digits={digits!r})")
    day, month, year = int(digits[0:2]), int(digits[2:4]), int(digits[4:8])
    date = _dt.date(year, month, day)

    return TitleInfo(measure=measure, date=date, measure_score=m_score)
