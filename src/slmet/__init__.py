"""slmet: extract tidy min/max temperature data from Sri Lanka Dept. of
Meteorology map images -- no external OCR engine required.

Typical usage::

    from slmet import extract_image, build_dataset, calibrate, load_registry
    from slmet import load_temperatures, download_images
    from slmet.imaging import load_gray_rgb, detect_frame

    # the bundled dataset, no images needed
    df = load_temperatures()

    # fetch map images from direct URLs into a folder
    report = download_images(["https://.../map.jpg"], "Downloads")

    # one map -> tidy records
    records = extract_image("path/to/map.jpg")

    # a whole folder -> one tidy DataFrame
    df = build_dataset("Original Images", cache_path="temperatures.csv")

    # the longitude/latitude function, standalone
    gray, rgb = load_gray_rgb("path/to/map.jpg")
    frame = detect_frame(gray)
    geo = calibrate(gray, frame)
    lon, lat = geo.to_lonlat(px=800, py=500)
"""
from .calibration import GeoTransform, calibrate
from .dataset import build_dataset, load_dataset, load_temperatures, to_wide
from .download import download_images
from .extract import Readers, extract_image
from .ingest import rename_and_copy
from .missing import find_missing_in_dataset, find_missing_in_dir
from .stations import assign_station, build_registry, load_registry, save_registry
from .title import TitleInfo, read_title

__version__ = "0.1.0"

__all__ = [
    "GeoTransform", "calibrate",
    "build_dataset", "load_dataset", "load_temperatures", "to_wide",
    "download_images",
    "Readers", "extract_image",
    "rename_and_copy",
    "find_missing_in_dataset", "find_missing_in_dir",
    "assign_station", "build_registry", "load_registry", "save_registry",
    "TitleInfo", "read_title",
]
