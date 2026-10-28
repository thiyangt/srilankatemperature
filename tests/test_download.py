from __future__ import annotations

import functools
import http.server
import os
import threading
from pathlib import Path

import pytest
from PIL import Image

from slmet.download import _filename_for, download_images
from slmet.ingest import rename_and_copy

from conftest import make_synthetic_map


@pytest.fixture
def http_dir(tmp_path):
    """Serve a temp directory over HTTP on localhost; yields (dir, base_url)."""
    served = tmp_path / "served"
    served.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(served))
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield served, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_filename_for_cdn_style_url():
    url = "https://scontent.example.net/v/t39/606000882_1189_n.jpg?stp=dst&_nc_ht=x"
    assert _filename_for(url) == "606000882_1189_n.jpg"
    # no usable basename -> stable hash-based name
    assert _filename_for("https://example.net/") == _filename_for("https://example.net/")
    assert _filename_for("https://example.net/").endswith(".jpg")


def test_download_then_ingest_over_http(http_dir, tmp_path):
    served, base = http_dir
    make_synthetic_map(str(served / "raw_min.jpg"), measure="Minimum", date="02-01-2026")
    make_synthetic_map(str(served / "raw_max.jpg"), measure="Maximum", date="02-01-2026")

    dl = tmp_path / "downloads"
    report = download_images([f"{base}/raw_min.jpg", f"{base}/raw_max.jpg"], str(dl))
    assert list(report["status"]) == ["downloaded", "downloaded"]
    assert sorted(os.listdir(dl)) == ["raw_max.jpg", "raw_min.jpg"]

    # downloaded files feed straight into the manual-image pipeline
    renamed = tmp_path / "renamed"
    ingest = rename_and_copy(str(dl), str(renamed))
    assert set(ingest["status"]) == {"copied"}
    assert sorted(os.listdir(renamed)) == [
        "Maximum Temperature - 2026-01-02.jpg",
        "Minimum Temperature - 2026-01-02.jpg",
    ]


def test_download_skips_existing_and_reports_errors(http_dir, tmp_path):
    served, base = http_dir
    make_synthetic_map(str(served / "map.jpg"))
    (served / "login.html").write_text("<html>please log in</html>")

    dl = tmp_path / "downloads"
    download_images(f"{base}/map.jpg", str(dl))
    report = download_images(
        [f"{base}/map.jpg", f"{base}/login.html", f"{base}/missing.jpg"], str(dl))
    statuses = list(report["status"])
    assert statuses[0] == "skipped_exists"
    assert statuses[1].startswith("error:")   # not an image
    assert statuses[2].startswith("error:")   # 404
    assert sorted(os.listdir(dl)) == ["map.jpg"]


def test_download_converts_non_jpeg_via_file_url(tmp_path):
    png = tmp_path / "map.png"
    Image.new("RGB", (40, 30), "red").save(png)

    dl = tmp_path / "downloads"
    report = download_images(Path(png).as_uri(), str(dl))
    assert list(report["status"]) == ["downloaded"]
    with Image.open(dl / "map.jpg") as img:
        assert img.format == "JPEG"
        assert img.size == (40, 30)


def test_ingest_accepts_png_files(tmp_path):
    src = tmp_path / "raw"
    src.mkdir()
    # a JPEG saved with a ".png" name (as Facebook downloads sometimes are)
    make_synthetic_map(str(src / "mislabelled.png"), measure="Minimum", date="02-01-2026")
    # and a genuine PNG
    jpg = tmp_path / "max.jpg"
    make_synthetic_map(str(jpg), measure="Maximum", date="02-01-2026")
    Image.open(jpg).save(src / "real.png", format="PNG")

    renamed = tmp_path / "renamed"
    report = rename_and_copy(str(src), str(renamed))
    assert set(report["status"]) == {"copied"}
    for name in ["Maximum Temperature - 2026-01-02.jpg",
                 "Minimum Temperature - 2026-01-02.jpg"]:
        with Image.open(renamed / name) as img:
            assert img.format == "JPEG"
