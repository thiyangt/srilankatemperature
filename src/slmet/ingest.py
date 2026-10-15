"""Rename/copy raw downloaded map images into a predictable naming scheme.

Reads only the title of each image (cheap -- no calibration or bubble
detection needed) and copies it to
``<dst_dir>/<Minimum|Maximum> Temperature - YYYY-MM-DD.jpg``.
"""
from __future__ import annotations

import hashlib
import os
import shutil

import pandas as pd

from .imaging import detect_frame, load_gray_rgb
from .ocr import DigitReader
from .title import read_title


def _file_hash(path: str, chunk: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def rename_and_copy(src_dir: str, dst_dir: str, overwrite: bool = False,
                     digit_reader: DigitReader | None = None) -> pd.DataFrame:
    """Copy every ``*.jpg``/``*.jpeg`` in ``src_dir`` into ``dst_dir`` under
    the ``"<Measure> Temperature - YYYY-MM-DD.jpg"`` naming scheme.

    Returns a report DataFrame with one row per source file: the new name,
    whether it was written/skipped, and why (e.g. a duplicate content hash,
    or a name collision with a different source file already copied).
    """
    digit_reader = digit_reader or DigitReader()
    os.makedirs(dst_dir, exist_ok=True)

    src_files = sorted(
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg"))
    )

    seen_hashes: dict[str, str] = {}  # content hash -> first source file
    rows = []
    for fname in src_files:
        src_path = os.path.join(src_dir, fname)
        row = {"source_file": fname, "new_file": None, "measure": None,
               "date": None, "status": None}
        try:
            content_hash = _file_hash(src_path)
            if content_hash in seen_hashes:
                row["status"] = f"skipped_duplicate_of:{seen_hashes[content_hash]}"
                rows.append(row)
                continue
            seen_hashes[content_hash] = fname

            gray, _rgb = load_gray_rgb(src_path)
            frame = detect_frame(gray)
            info = read_title(gray, frame, digit_reader)

            new_name = f"{info.measure} Temperature - {info.date.isoformat()}.jpg"
            dst_path = os.path.join(dst_dir, new_name)
            row["measure"] = info.measure
            row["date"] = info.date.isoformat()
            row["new_file"] = new_name

            if os.path.exists(dst_path) and not overwrite:
                row["status"] = "skipped_exists"
                rows.append(row)
                continue

            shutil.copy2(src_path, dst_path)
            row["status"] = "copied"
        except Exception as e:
            row["status"] = f"error:{e}"
        rows.append(row)

    return pd.DataFrame(rows)
