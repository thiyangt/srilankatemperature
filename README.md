# slmet

Extract a tidy min/max temperature dataset from the Sri Lanka Department of
Meteorology's daily map images — **no external OCR engine required**
(no Tesseract, no EasyOCR, no cloud API). Everything, including the text and
number reading, is implemented from scratch on top of numpy/Pillow/
scikit-image, because every map is rendered by the department's own fixed
matplotlib script: same layout, same DejaVu Sans font, same ~24 weather
collection centres, every day. That regularity is what the package exploits.

## Getting started: a step-by-step guide

### Step 1: Check what you need

- **Python 3.9 or newer**: check with `python --version`.
- **Git**: needed to install straight from GitHub. Check with `git --version`.
- **Map images**: this repo already includes every map from 2026-01-01 to
  2026-09-07 (`Original Images/`). Maps you download yourself also work.

### Step 2: Install

Choose one option. A virtual environment (`python -m venv .venv`) or a conda
environment is recommended.

**A. Just use the library:**

```bash
pip install git+https://github.com/thiyangt/srilankatemperature.git
```

**B. Get the code, the images and the dataset too:** the clone is about
200 MB because it includes the images.

```bash
git clone https://github.com/thiyangt/srilankatemperature.git
cd srilankatemperature
pip install -e ".[dev]"
```

Both options install the dependencies automatically: `numpy`, `pillow`,
`scikit-image`, `pandas`, `matplotlib`, `scipy` and `scikit-learn`.

### Step 3: Check that it works

```bash
slmet --help
python -c "import slmet; print(slmet.__version__)"   # 0.1.0
```

If you chose option B, you can also run `pytest`. All tests should pass.

### Step 4: Use it

**If you only want the data**, you don't need to run anything.
`temperatures.csv` in this repo already covers 2026-01-01 to 2026-09-07:

```python
import pandas as pd
df = pd.read_csv("temperatures.csv")
```

**Feature 1: Read a single map.**

```python
from slmet import extract_image
rows = extract_image("Maximum Temperature - 2026-01-01.jpg")
```

You get one record per station with `date`, `measure` (Minimum or Maximum),
`station`, `latitude`, `longitude`, `temperature_c` and `source_file`. If a
value can't be read reliably, `temperature_c` is `None` rather than a guess.

For auditing, pass `diagnostics=True` (to `extract_image` or
`build_dataset`, or `--diagnostics` on the command line) to also get
`label_ocr_guess` / `label_ocr_score` (the station name read from the text
beside each bubble, and how well it matched) and `station_source`. The
station is identified by the bubble's position, not by this label reading,
so these columns aren't needed for analysis.

**Feature 2: Convert a whole folder into one table.**

```python
from slmet import build_dataset
df = build_dataset("Original Images", cache_path="temperatures.csv")
```

Or from the command line:
`slmet build-dataset "Original Images" --out temperatures.csv`

**Feature 3: Rename raw downloads.** Downloaded images have meaningless names
like `606000882_..._n.jpg`. This reads each map's title and copies it as
`Maximum Temperature - 2026-01-01.jpg`, skipping duplicate downloads.

```bash
slmet ingest "Original Images" "Renamed"
```

In Python, `rename_and_copy(src, dst)` does the same and returns a report of
which files were copied and which were skipped.

**Feature 4: Find missing days.**

```bash
slmet missing "Renamed" --start 2026-01-01 --end 2026-09-07
```

This lists every (date, Minimum/Maximum) pair with no image. In Python, use
`find_missing_in_dir(...)`, or `find_missing_in_dataset(df, start, end)` to
check a dataset you've already built.

**Feature 5: Put min and max side by side.**

```python
from slmet import to_wide
wide = to_wide(df)   # one row per date + station, with tmin_c and tmax_c columns
```

**Feature 6: Convert pixel positions to longitude/latitude.** This uses the
map's own axis labels, so nothing is hard-coded.

```python
from slmet import calibrate
from slmet.imaging import load_gray_rgb, detect_frame

gray, rgb = load_gray_rgb("map.jpg")
geo = calibrate(gray, detect_frame(gray))
lon, lat = geo.to_lonlat(px=800, py=500)
```

**Feature 7: Rebuild the station list.** The package comes with the 24
stations, from Jaffna to Galle. You only need this if the department adds or
moves a station:

```bash
slmet build-stations "Original Images" --out stations.csv
```

### Good to know

- **No OCR software is needed.** No Tesseract, EasyOCR or online service;
  everything runs locally.
- **New images aren't downloaded for you.** To extend the dataset, download
  new maps, put them in a folder and run Feature 2.
- See [Known data gaps](#known-data-gaps) and
  [Known limitations](#known-limitations) below.

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
- Bubble values are read several ways (text thresholded relative to each
  bubble's own fill colour, at a few clip radii, plus an absolute black
  mask) and the majority reading wins. This fixed a systematic error on hot,
  deep-red bubbles, where JPEG blur broke up the digits (e.g. "34.4" read
  as "3.4"). Bubble size is also taken as the image's typical marker size,
  since per-bubble estimates ran small and clipped the leading digit on pale
  bubbles ("22.4" read as "2.4"). On 96 hand-checked values from four maps
  (hot and cold), all are read correctly. Misreads are still possible; `extract_image()` leaves
  `temperature_c` as `None` when no reading is plausible rather than
  guessing.

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
