"""Detect the temperature bubbles on a map and read what's on/near them.

Each bubble is a filled, black-edged circle (matplotlib scatter marker) with
a colour that encodes temperature and a black numeric label drawn at its
centre. Bubbles frequently touch the coastline outline or a neighbouring
station's outline, which would otherwise merge them into one connected
component -- so detection works on an *eroded* mask: eroding away thin
strokes (coastline, bubble ring, text) by a fixed radius breaks those
incidental connections while a solid bubble disk survives as a smaller disk,
which is then filtered by area and roundness and re-inflated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import measure, morphology

from .imaging import Frame
from .ocr import DigitReader, VocabularyReader, dark_mask_from_rgb

EROSION_RADIUS = 10
MIN_ERODED_AREA = 1200
MAX_ERODED_AREA = 6500
MAX_ASPECT = 1.8


@dataclass
class Bubble:
    cx: float  # pixel column, in full-image coordinates
    cy: float  # pixel row, in full-image coordinates
    radius: float
    fill_rgb: tuple[int, int, int]
    temperature_c: float | None = None
    label: str | None = None
    label_score: float = -1.0


def detect_bubbles(gray: np.ndarray, frame: Frame, scale: float | None = None) -> list[Bubble]:
    """Find bubble circles within the plot's axes frame."""
    scale = scale if scale is not None else gray.shape[0] / 2048.0
    erosion_r = max(3, round(EROSION_RADIUS * scale))
    min_area = MIN_ERODED_AREA * scale ** 2
    max_area = MAX_ERODED_AREA * scale ** 2

    sub = gray[frame.top:frame.bottom, frame.left:frame.right]
    notwhite = sub < 250
    eroded = morphology.erosion(notwhite, morphology.disk(erosion_r))
    lbl = measure.label(eroded, connectivity=2)

    bubbles = []
    for r in measure.regionprops(lbl):
        if r.area < min_area or r.area > max_area:
            continue
        y0, x0, y1, x1 = r.bbox
        h, w = y1 - y0, x1 - x0
        if max(h, w) / max(1, min(h, w)) > MAX_ASPECT:
            continue
        cy, cx = r.centroid
        eroded_radius = float(np.sqrt(r.area / np.pi))
        radius = eroded_radius + erosion_r
        bubbles.append(Bubble(
            cx=cx + frame.left, cy=cy + frame.top, radius=radius,
            fill_rgb=(0, 0, 0),
        ))
    return bubbles


def _sample_fill_color(rgb: np.ndarray, bubble: Bubble) -> tuple[int, int, int]:
    """Median colour near the bubble centre, off-axis from the text row."""
    samples = []
    r = bubble.radius * 0.55
    for dx, dy in [(-r, -r), (r, -r), (-r, r), (r, r), (0, -r * 0.9), (0, r * 0.9)]:
        y = int(round(bubble.cy + dy))
        x = int(round(bubble.cx + dx))
        if 0 <= y < rgb.shape[0] and 0 <= x < rgb.shape[1]:
            px = rgb[y, x]
            if not (px.max() < 60 or px.min() > 245):  # skip near-black/near-white
                samples.append(px)
    if not samples:
        y, x = int(bubble.cy), int(bubble.cx)
        return tuple(int(v) for v in rgb[y, x])
    arr = np.array(samples)
    return tuple(int(v) for v in np.median(arr, axis=0))


def read_bubble_value(rgb: np.ndarray, bubble: Bubble, reader: DigitReader,
                       inner_fraction: float = 1.2) -> float | None:
    """Read the numeric label at the bubble's centre.

    Uses an RGB (not grayscale) text mask because low temperatures are
    plotted on a near-navy fill, on which the black text has almost the same
    grayscale luminance as the background -- see
    :func:`slmet.ocr.dark_mask_from_rgb`. The crop is taken slightly larger
    than the bubble radius (further than its ring) because text sometimes
    sits close to the ring; components touching the crop border are dropped
    by the reader as noise, which discards the little that leaks in from the
    square crop's corners. (A small minority of bubbles -- mostly coastal
    stations where the coastline runs right up against the marker -- have
    text touching the ring itself; those are a known, documented gap rather
    than something this crop geometry alone can resolve.)
    """
    r = int(bubble.radius * inner_fraction)
    y0 = max(0, int(bubble.cy - r))
    y1 = min(rgb.shape[0], int(bubble.cy + r))
    x0 = max(0, int(bubble.cx - r))
    x1 = min(rgb.shape[1], int(bubble.cx + r))
    crop = rgb[y0:y1, x0:x1]
    digits = reader.read_string(dark_mask_from_rgb(crop))
    return _parse_temperature(digits)


def _parse_temperature(digits: str) -> float | None:
    """Parse a bubble's OCR'd digit string as a temperature.

    Every value on these maps has exactly one decimal place (e.g. "8.3",
    "24.6"). The decimal point is a tiny, low-contrast glyph and is
    occasionally missed by segmentation even though the surrounding digits
    read correctly (e.g. "31.0" -> "310"); since the format is fixed, a
    missing point is unambiguous to restore rather than losing the reading
    outright.
    """
    if "." not in digits and digits.lstrip("-").isdigit() and 2 <= len(digits.lstrip("-")) <= 3:
        digits = digits[:-1] + "." + digits[-1]
    try:
        value = float(digits)
    except ValueError:
        return None
    # Sri Lanka's published min/max temperatures never fall outside this
    # range; a value beyond it means a digit was mis-segmented (e.g. a
    # fragment of a digit's serif mistaken for a second decimal point) --
    # better to report a clean miss than a confidently wrong number.
    if not (-5.0 <= value <= 50.0):
        return None
    return value


def _label_box(bubble: Bubble, side: str, gap: float, along: float, across: float
               ) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) flush against one edge of the bubble,
    extending outward -- deliberately excludes the ring itself so its
    text-detection threshold isn't confused by bubble outline pixels."""
    r = bubble.radius
    if side == "right":
        x0, x1 = bubble.cx + r * gap, bubble.cx + r * (gap + along)
        y0, y1 = bubble.cy - r * across / 2, bubble.cy + r * across / 2
    elif side == "left":
        x0, x1 = bubble.cx - r * (gap + along), bubble.cx - r * gap
        y0, y1 = bubble.cy - r * across / 2, bubble.cy + r * across / 2
    elif side == "below":
        x0, x1 = bubble.cx - r * along / 2, bubble.cx + r * along / 2
        y0, y1 = bubble.cy + r * gap, bubble.cy + r * (gap + across)
    else:  # above
        x0, x1 = bubble.cx - r * along / 2, bubble.cx + r * along / 2
        y0, y1 = bubble.cy - r * (gap + across), bubble.cy - r * gap
    return int(x0), int(y0), int(x1), int(y1)


def read_bubble_label(gray: np.ndarray, bubble: Bubble, reader: VocabularyReader,
                       gap: float = 1.08, along: float = 4.2, across: float = 1.3
                       ) -> tuple[str, float]:
    """Try each side of the bubble (labels aren't placed on a fixed side) and
    keep whichever crop best matches a known station name."""
    best_word, best_score = "", -2.0
    h, w = gray.shape
    for side in ("right", "left", "below", "above"):
        x0, y0, x1, y1 = _label_box(bubble, side, gap, along, across)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = gray[y0:y1, x0:x1]
        word, score = reader.match(crop)
        if score > best_score:
            best_word, best_score = word, score
    return best_word, best_score


def analyze_bubbles(gray: np.ndarray, rgb: np.ndarray, frame: Frame,
                     digit_reader: DigitReader, vocab_reader: VocabularyReader) -> list[Bubble]:
    bubbles = detect_bubbles(gray, frame)
    for b in bubbles:
        b.fill_rgb = _sample_fill_color(rgb, b)
        b.temperature_c = read_bubble_value(rgb, b, digit_reader)
        b.label, b.label_score = read_bubble_label(gray, b, vocab_reader)
    return bubbles
