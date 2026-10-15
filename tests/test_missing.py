from __future__ import annotations

import os

from slmet.missing import find_missing_in_dir


def test_find_missing_in_dir(tmp_path):
    d = tmp_path / "renamed"
    d.mkdir()
    for name in [
        "Minimum Temperature - 2026-01-01.jpg",
        "Maximum Temperature - 2026-01-01.jpg",
        "Minimum Temperature - 2026-01-02.jpg",
        # 2026-01-02 Maximum is missing
        # 2026-01-03 is entirely missing
    ]:
        open(os.path.join(d, name), "w").close()

    missing = find_missing_in_dir(str(d), start="2026-01-01", end="2026-01-03")
    pairs = set(zip(missing["date"], missing["measure"]))
    assert pairs == {
        ("2026-01-02", "Maximum"),
        ("2026-01-03", "Minimum"),
        ("2026-01-03", "Maximum"),
    }
