"""Shared test fixtures: a synthetic map image generated the same way the
Dept. of Meteorology's plotting script would, so tests don't depend on the
real (private) image corpus.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


SYNTH_STATIONS = [
    ("Jaffna", 80.02, 9.70, 24.6),
    ("Colombo", 79.87, 6.90, 25.0),
    ("Nuwara Eliya", 80.77, 6.97, 13.9),
    ("Trincomalee", 81.23, 8.58, 23.4),
    ("Galle", 80.22, 6.03, 25.5),
]


def make_synthetic_map(path: str, measure: str = "Minimum", date: str = "02-01-2026"):
    fig, ax = plt.subplots(figsize=(7.41, 10.24), dpi=200)
    lons = [s[1] for s in SYNTH_STATIONS]
    lats = [s[2] for s in SYNTH_STATIONS]
    temps = [s[3] for s in SYNTH_STATIONS]
    sc = ax.scatter(lons, lats, c=temps, cmap="coolwarm", vmin=5, vmax=40,
                     s=900, edgecolors="black", zorder=3)
    for name, lon, lat, t in SYNTH_STATIONS:
        ax.text(lon, lat, f"{t:.1f}", ha="center", va="center", zorder=4)
        ax.text(lon + 0.09, lat, name, ha="left", va="center", fontsize=11)
    ax.set_xlim(79.5, 81.9)
    ax.set_ylim(5.9, 10.0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(sc, ax=ax, label=f"{measure[:3].lower()} (°C)")
    fig.suptitle(f"{measure} Temperatures Reported on {date}", fontsize=18, y=0.97)
    ax.set_title("(Issued by the Department of Meteorology)", fontsize=13)
    fig.savefig(path, format="jpg")
    plt.close(fig)


@pytest.fixture
def synthetic_map(tmp_path):
    path = tmp_path / "synthetic_min.jpg"
    make_synthetic_map(str(path))
    return str(path)
