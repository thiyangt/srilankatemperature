"""Download map images from a list of URLs.

The counterpart to working from a folder of manually downloaded images:
give it direct image links (e.g. a map's "open image in new tab" address,
or a Dept. of Meteorology page if they ever publish an archive) and it
saves each one into a folder that :func:`slmet.ingest.rename_and_copy` and
:func:`slmet.dataset.build_dataset` can then process as usual.

Uses only the standard library (``urllib``), so ``file://`` URLs work too.
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import urllib.parse
import urllib.request

import pandas as pd
from PIL import Image

# Some image CDNs reject urllib's default "Python-urllib/x.y" agent.
_USER_AGENT = "Mozilla/5.0 (compatible; slmet image downloader)"


def _filename_for(url: str) -> str:
    """Derive a safe ``.jpg`` filename from the URL's path, falling back to
    a hash of the URL when the path has no usable basename."""
    base = os.path.basename(urllib.parse.urlparse(url).path)
    stem, ext = os.path.splitext(base)
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
    if not stem or ext.lower() not in (".jpg", ".jpeg", ".png", ".webp", ""):
        stem = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
    return f"{stem}.jpg"


def _to_jpeg_bytes(data: bytes) -> bytes:
    """Return ``data`` unchanged if it is already a JPEG, otherwise re-encode
    it as one (the rest of the pipeline only globs ``*.jpg``). Raises if the
    payload is not an image at all, e.g. an HTML login/error page."""
    img = Image.open(io.BytesIO(data))
    img.verify()
    if img.format == "JPEG":
        return data
    img = Image.open(io.BytesIO(data)).convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=95)
    return out.getvalue()


def download_images(urls, dest_dir: str, overwrite: bool = False,
                    timeout: float = 30.0) -> pd.DataFrame:
    """Download each URL in ``urls`` (a single string or an iterable of
    strings) into ``dest_dir`` as a ``.jpg``.

    Returns a report DataFrame with one row per URL: the saved file name
    and a status of ``downloaded``, ``skipped_exists`` or ``error:<reason>``.
    Failures (network errors, non-image responses) are reported rather than
    raised, so one bad link doesn't stop the batch.
    """
    if isinstance(urls, str):
        urls = [urls]
    os.makedirs(dest_dir, exist_ok=True)

    rows = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        fname = _filename_for(url)
        dst_path = os.path.join(dest_dir, fname)
        row = {"url": url, "file": fname, "status": None}
        try:
            if os.path.exists(dst_path) and not overwrite:
                row["status"] = "skipped_exists"
                rows.append(row)
                continue

            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            data = _to_jpeg_bytes(data)

            with open(dst_path, "wb") as f:
                f.write(data)
            row["status"] = "downloaded"
        except Exception as e:
            row["status"] = f"error:{e}"
        rows.append(row)

    return pd.DataFrame(rows, columns=["url", "file", "status"])
