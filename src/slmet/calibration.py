"""Turn pixel coordinates on a map image into (longitude, latitude).

This is the function referenced in the brief: "To identify longitude and
latitude, you should also write a function." Rather than hard-coding the
axis data-limits (which would silently break if the department ever changes
them), calibration is done per image by:

1. detecting the plot's axes frame (see :mod:`slmet.imaging`),
2. locating the tick marks matplotlib draws just outside that frame,
3. OCR-reading each tick's numeric label with :class:`slmet.ocr.DigitReader`,
4. fitting a pixel -> data linear map (least squares) from the (tick pixel,
   tick value) pairs, for x (longitude) and y (latitude) independently.

Because ticks are read from the image every time, this is robust to the one
odd-sized image in the corpus (a uniformly downscaled copy) and would keep
working if the plotted extent ever changed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .imaging import Frame
from .ocr import DigitReader, dark_mask_from_gray


@dataclass
class GeoTransform:
    """Affine pixel->data mapping fitted independently per axis."""

    x_slope: float
    x_intercept: float
    y_slope: float
    y_intercept: float
    n_x_ticks: int
    n_y_ticks: int

    def lon(self, px: np.ndarray | float) -> np.ndarray | float:
        return self.x_slope * px + self.x_intercept

    def lat(self, py: np.ndarray | float) -> np.ndarray | float:
        return self.y_slope * py + self.y_intercept

    def to_lonlat(self, px: float, py: float) -> tuple[float, float]:
        return float(self.lon(px)), float(self.lat(py))


def _cluster_columns(cols: np.ndarray, gap: int = 4) -> list[float]:
    if len(cols) == 0:
        return []
    cols = np.sort(cols)
    groups = [[cols[0]]]
    for c in cols[1:]:
        if c - groups[-1][-1] <= gap:
            groups[-1].append(c)
        else:
            groups.append([c])
    return [float(np.mean(g)) for g in groups]


def _find_x_ticks(gray: np.ndarray, frame: Frame, scale: float) -> list[float]:
    band_h = max(4, round(15 * scale))
    r0 = frame.bottom + 1
    r1 = min(gray.shape[0], r0 + band_h)
    band = gray[r0:r1, frame.left:frame.right] < 100
    persistent = band.mean(axis=0) > 0.4
    cols = np.where(persistent)[0]
    return [c + frame.left for c in _cluster_columns(cols, gap=max(2, round(4 * scale)))]


def _find_y_ticks(gray: np.ndarray, frame: Frame, scale: float) -> list[float]:
    band_w = max(4, round(15 * scale))
    c1 = frame.left - 1
    c0 = max(0, c1 - band_w)
    band = gray[frame.top:frame.bottom, c0:c1] < 100
    persistent = band.mean(axis=1) > 0.4
    rows = np.where(persistent)[0]
    return [r + frame.top for r in _cluster_columns(rows, gap=max(2, round(4 * scale)))]


def _read_x_tick_label(gray: np.ndarray, frame: Frame, tick_px: float, scale: float,
                        reader: DigitReader) -> float | None:
    half_w = round(45 * scale)
    y0 = frame.bottom + round(16 * scale)
    y1 = frame.bottom + round(70 * scale)
    x0 = max(0, int(tick_px - half_w))
    x1 = min(gray.shape[1], int(tick_px + half_w))
    crop = gray[y0:min(gray.shape[0], y1), x0:x1]
    return reader.read_float(dark_mask_from_gray(crop))


def _read_y_tick_label(gray: np.ndarray, frame: Frame, tick_px: float, scale: float,
                        reader: DigitReader) -> float | None:
    half_h = round(20 * scale)
    x1 = frame.left - round(16 * scale)
    x0 = frame.left - round(110 * scale)
    y0 = max(0, int(tick_px - half_h))
    y1 = min(gray.shape[0], int(tick_px + half_h))
    crop = gray[y0:y1, max(0, int(x0)):max(1, int(x1))]
    return reader.read_float(dark_mask_from_gray(crop))


def calibrate(gray: np.ndarray, frame: Frame, reader: DigitReader | None = None) -> GeoTransform:
    """Fit a pixel->(longitude, latitude) transform for one map image."""
    reader = reader or DigitReader()
    scale = gray.shape[0] / 2048.0

    x_ticks_px = _find_x_ticks(gray, frame, scale)
    y_ticks_px = _find_y_ticks(gray, frame, scale)

    x_pairs = []
    for px in x_ticks_px:
        val = _read_x_tick_label(gray, frame, px, scale, reader)
        if val is not None:
            x_pairs.append((px, val))

    y_pairs = []
    for py in y_ticks_px:
        val = _read_y_tick_label(gray, frame, py, scale, reader)
        if val is not None:
            y_pairs.append((py, val))

    if len(x_pairs) < 2 or len(y_pairs) < 2:
        raise ValueError(
            f"Calibration failed: read {len(x_pairs)} usable x-ticks and "
            f"{len(y_pairs)} usable y-ticks (need >= 2 of each)."
        )

    xs = np.array([p[0] for p in x_pairs])
    xv = np.array([p[1] for p in x_pairs])
    ys = np.array([p[0] for p in y_pairs])
    yv = np.array([p[1] for p in y_pairs])

    x_slope, x_intercept = np.polyfit(xs, xv, 1)
    y_slope, y_intercept = np.polyfit(ys, yv, 1)

    return GeoTransform(
        x_slope=float(x_slope), x_intercept=float(x_intercept),
        y_slope=float(y_slope), y_intercept=float(y_intercept),
        n_x_ticks=len(x_pairs), n_y_ticks=len(y_pairs),
    )
