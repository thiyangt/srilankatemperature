from __future__ import annotations

import datetime as dt

from slmet.bubbles import detect_bubbles
from slmet.calibration import calibrate
from slmet.imaging import detect_frame, load_gray_rgb
from slmet.stations import build_registry
from slmet.title import read_title
from slmet.extract import extract_image

from conftest import SYNTH_STATIONS


def test_detect_frame(synthetic_map):
    gray, _rgb = load_gray_rgb(synthetic_map)
    frame = detect_frame(gray)
    assert frame.width > 0 and frame.height > 0
    assert frame.top < frame.bottom and frame.left < frame.right


def test_calibrate_matches_known_axis_limits(synthetic_map):
    gray, _rgb = load_gray_rgb(synthetic_map)
    frame = detect_frame(gray)
    geo = calibrate(gray, frame)
    lon_left = geo.lon(frame.left)
    lon_right = geo.lon(frame.right)
    lat_top = geo.lat(frame.top)
    lat_bottom = geo.lat(frame.bottom)
    # synthetic map's xlim/ylim are (79.5, 81.9) / (5.9, 10.0); the axes
    # frame sits a hair outside the outermost ticks, so allow a small margin
    assert 79.3 < lon_left < 79.7
    assert 81.9 < lon_right < 82.4
    assert 9.8 < lat_top < 10.2
    assert 5.7 < lat_bottom < 6.1


def test_detect_bubbles_finds_all_stations(synthetic_map):
    gray, _rgb = load_gray_rgb(synthetic_map)
    frame = detect_frame(gray)
    bubbles = detect_bubbles(gray, frame)
    assert len(bubbles) == len(SYNTH_STATIONS)


def test_read_title(synthetic_map):
    gray, _rgb = load_gray_rgb(synthetic_map)
    frame = detect_frame(gray)
    info = read_title(gray, frame)
    assert info.measure == "Minimum"
    assert info.date == dt.date(2026, 1, 2)
    assert info.measure_score > 0.5


def test_extract_image_end_to_end(synthetic_map):
    registry = build_registry([synthetic_map], n_stations=len(SYNTH_STATIONS),
                               max_images=1)
    records = extract_image(synthetic_map, registry=registry)
    assert len(records) == len(SYNTH_STATIONS)

    known_temps = {name: temp for name, _lon, _lat, temp in SYNTH_STATIONS}
    matched = 0
    for r in records:
        assert r["date"] == "2026-01-02"
        assert r["measure"] == "Minimum"
        if r["station"] in known_temps and r["temperature_c"] is not None:
            assert abs(r["temperature_c"] - known_temps[r["station"]]) < 0.05
            matched += 1
    # every station's value should have been read correctly at minimum
    assert all(r["temperature_c"] is not None for r in records)


def test_extract_image_diagnostics_columns(synthetic_map):
    registry = build_registry([synthetic_map], n_stations=len(SYNTH_STATIONS),
                               max_images=1)
    core = {"date", "measure", "station", "latitude", "longitude",
            "temperature_c", "source_file"}
    diag = {"label_ocr_guess", "label_ocr_score", "station_source"}

    plain = extract_image(synthetic_map, registry=registry)
    assert all(set(r) == core for r in plain)

    full = extract_image(synthetic_map, registry=registry, diagnostics=True)
    assert all(set(r) == core | diag for r in full)
    # diagnostics only add columns; the core values are identical
    assert [{k: r[k] for k in core} for r in full] == plain
