"""Low-level image loading and axes-frame detection.

All the maps share one layout: a rectangular plot frame (black spines) drawn
by matplotlib, inset from the JPEG edges by title/label whitespace. Detecting
that frame per image (rather than hard-coding pixel coordinates) makes the
rest of the pipeline robust to the JPEGs being different sizes (the corpus
has one image scaled down relative to the other 500).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Frame:
    """Pixel bounding box of the plot axes (inclusive of the spine lines)."""

    top: int
    bottom: int
    left: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left


# The size shared by (almost) every image in the corpus. One known outlier
# is a uniformly downscaled copy of the same figure; resizing every image up
# to this canonical size means every pixel-based constant tuned elsewhere in
# the package (erosion radii, crop margins, tick-band heights, ...) applies
# unchanged regardless of the source JPEG's actual resolution.
CANONICAL_SIZE = (1482, 2048)  # (width, height), PIL convention


def load_gray_rgb(path: str, canonical_size: tuple[int, int] | None = CANONICAL_SIZE
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return (grayscale, rgb) uint8 arrays for an image path.

    Resizes to ``canonical_size`` first (pass ``None`` to disable) so all
    downstream pixel-based heuristics see a consistent scale.
    """
    im = Image.open(path).convert("RGB")
    if canonical_size is not None and im.size != canonical_size:
        im = im.resize(canonical_size, Image.LANCZOS)
    rgb = np.array(im)
    gray = rgb.mean(axis=2)
    return gray, rgb


def detect_frame(gray: np.ndarray, dark_thresh: int = 80) -> Frame:
    """Locate the plot's rectangular axes frame via long runs of dark pixels.

    The spine lines are the only features in the image that span (close to)
    the full width/height of the axes, so thresholding on "mostly dark rows
    / columns" isolates them cleanly from text, bubbles and the coastline.
    """
    dark = gray < dark_thresh
    h, w = dark.shape
    row_counts = dark.sum(axis=1)
    col_counts = dark.sum(axis=0)
    rows = np.where(row_counts > 0.5 * w)[0]
    cols = np.where(col_counts > 0.3 * h)[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("Could not detect axes frame (no long dark runs found)")
    return Frame(top=int(rows.min()), bottom=int(rows.max()),
                 left=int(cols.min()), right=int(cols.max()))
