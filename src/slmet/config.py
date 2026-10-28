"""Constants shared across the package.

The Department of Meteorology publishes these maps from a single fixed
plotting script: same figure layout, same font, same axis frame, same
~24 weather collection centres every day. That regularity is what makes
a from-scratch, dependency-light OCR feasible, and is exploited throughout
this package instead of hard-coding pixel positions everywhere.
"""
from __future__ import annotations

import os

import matplotlib

# DejaVu Sans ships with matplotlib itself, so no extra font install is
# required to build glyph/word templates.
_MPL_DATA = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
FONT_REGULAR = os.path.join(_MPL_DATA, "DejaVuSans.ttf")
FONT_BOLD = os.path.join(_MPL_DATA, "DejaVuSans-Bold.ttf")

# Known Sri Lanka Dept. of Meteorology surface weather / climate collection
# centres that appear as bubble labels on these maps. Used to (a) resolve
# bubble station identity via whole-word template matching and (b) sanity
# check/label the auto-clustered station registry.
KNOWN_STATIONS = [
    "Jaffna", "Mullativu", "Mannar", "Vavuniya", "Trincomalee",
    "Anuradhapura", "Polonnaruwa", "Maha Illuppallama", "Batticaloa",
    "Puttalam", "Kurunagala", "Katugasthota", "Katunayake", "Colombo",
    "Rathmalana", "Nuwara Eliya", "Badulla", "Bandarawela", "Pothuvil",
    "Rathnapura", "Monaragala", "Mattala", "Hambanthota", "Galle",
]

MEASURES = ("Minimum", "Maximum")
MEASURE_SHORT = {"Minimum": "Min", "Maximum": "Max"}

# Digits + punctuation actually needed to read tick labels, bubble values,
# and the title date -- kept small and closed on purpose for accuracy.
DIGIT_CHARSET = "0123456789.-:"

DEFAULT_DATE_START = "2026-01-01"

# Source image extensions picked up from a folder. Facebook downloads are
# sometimes saved as ".png" even when the content is a JPEG.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
