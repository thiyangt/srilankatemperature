# slmet

Extract a tidy min/max temperature dataset from the Sri Lanka Department of
Meteorology's daily map images — **no external OCR engine required**
(no Tesseract, no EasyOCR, no cloud API). Everything, including the text and
number reading, is implemented from scratch on top of numpy/Pillow/
scikit-image, because every map is rendered by the department's own fixed
matplotlib script: same layout, same DejaVu Sans font, same ~24 weather
collection centres, every day. That regularity is what the package exploits.

## Install

Straight from GitHub:

```bash
pip install git+https://github.com/thiyangt/srilankatemperature.git
```

Or from a local clone (editable, for development):

```bash
git clone https://github.com/thiyangt/srilankatemperature.git
cd srilankatemperature
pip install -e ".[dev]"
pytest
```

(installs `numpy`, `pillow`, `scikit-image`, `pandas`, `matplotlib`,
`scipy`, `scikit-learn`)

## What it does

1. **Ingest / rename** — copy raw downloaded images into
   `<Minimum|Maximum> Temperature - YYYY-MM-DD.jpg`, read straight off each
   image's own title (no manual renaming, no relying on the download
   filename).
2. **Find missing days** — diff what's on disk (or in the dataset) against
   the full expected `[start, end] x {Minimum, Maximum}` grid.
3. **Extract location + temperature from a map image** — the core ask:
   `extract_image()` reads one JPEG and returns one row per weather station
   with its **name, latitude, longitude, and temperature**, tidy and ready
   for `pandas`. `calibrate()` is the standalone "identify longitude and
   latitude" function, fitted per image from the plot's own axis ticks
   rather than hard-coded.
4. **Build the whole dataset** — run extraction over every downloaded image
   and get one long, tidy `DataFrame` / CSV covering everything published so
   far.

## Quickstart (Python API)

```python
from slmet import rename_and_copy, find_missing_in_dir, extract_image, build_dataset

# 1. rename/copy raw downloads
report = rename_and_copy("Original Images", "Renamed")

# 2. what's missing between 2026-01-01 and 2026-08-07?
missing = find_missing_in_dir("Renamed", start="2026-01-01", end="2026-08-07")

# 3. one map -> tidy rows: station, lat, lon, temperature_c, ...
records = extract_image("Renamed/Minimum Temperature - 2026-01-02.jpg")

# 4. the whole corpus -> one tidy CSV
df = build_dataset("Renamed", cache_path="temperatures.csv")
```

### The longitude/latitude function on its own

```python
from slmet.imaging import load_gray_rgb, detect_frame
from slmet.calibration import calibrate

gray, rgb = load_gray_rgb("map.jpg")
frame = detect_frame(gray)
geo = calibrate(gray, frame)          # fits pixel -> (lon, lat) from the axis ticks
lon, lat = geo.to_lonlat(px=800, py=500)
```

## CLI

```bash
slmet ingest "Original Images" "Renamed"
slmet missing "Renamed" --start 2026-01-01 --end 2026-08-07
slmet build-stations "Original Images" --out src/slmet/data/stations.csv
slmet build-dataset "Original Images" --out temperatures.csv
```

## How the OCR works (no Tesseract/EasyOCR)

- **`DigitReader`** — per-glyph template matching restricted to the closed
  charset `0-9 . -` (tick labels, bubble values, dates). A tiny known
  charset makes this reliable even on JPEG-compressed anti-aliased text.
  Bubble-value text is read from an **RGB** mask (not grayscale), because
  the coldest bubbles are filled near-navy, where black text and background
  have almost the same grayscale luminance but very different RGB.
- **`VocabularyReader`** — whole-word template matching against a *finite*
  known vocabulary (station names, "Minimum"/"Maximum"). Since the space of
  possible answers is small and known ahead of time, matching a crop against
  every candidate's rendered reference beats segmenting individual letters.
- **Station identity is resolved by position, not per-image OCR.** A given
  station's bubble lands in almost exactly the same spot every day, so
  `slmet.stations.build_registry()` clusters bubble positions across many
  images (k-means) and names each cluster by a majority vote of per-image
  label OCR — far more robust than trusting any single image's label
  reading. Per-day extraction then just does nearest-cluster lookup.

## Known data gaps

- **2026-07-07 Minimum** — no source image was ever published for this
  day/measure (only the Maximum map exists), so `temperatures.csv` has no
  `2026-07-07,Minimum` rows. This is a gap in the department's postings, not
  an extraction failure. The dataset otherwise covers every
  `[2026-01-01, 2026-09-07] x {Minimum, Maximum}` day for which an image
  exists.

## Known limitations

- Tuned against the corpus's dominant image size (1482x2048); other sizes
  are upscaled to that canonical size before processing, which works well
  but degrades OCR precision on notably lower-resolution source images.
- The bundled `data/stations.csv` registry was built from this corpus. If
  the department adds/moves a station, rebuild it with `slmet build-stations`.
- Digit OCR occasionally misreads a bubble in a very small minority of
  images (e.g. adjacent-bubble crowding); `extract_image()` leaves
  `temperature_c` as `None` when reading fails or gives a clearly implausible
  string rather than guessing.

## Package layout

```
src/slmet/
  imaging.py       image loading + axes-frame detection
  ocr.py           from-scratch digit + vocabulary OCR
  calibration.py   pixel -> (longitude, latitude) — the "identify lon/lat" function
  bubbles.py       bubble (circle) detection + value/label reading
  title.py         title parsing -> (measure, date)
  stations.py      canonical station registry (clustering + naming)
  extract.py       extract_image() — one map -> tidy records
  ingest.py        rename_and_copy() — raw downloads -> named copies
  missing.py       find_missing_in_dir() / find_missing_in_dataset()
  dataset.py       build_dataset() — whole corpus -> tidy DataFrame/CSV
  cli.py           `slmet` command-line entry point
  data/stations.csv  bundled canonical station registry
```
