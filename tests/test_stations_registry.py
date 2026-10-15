from __future__ import annotations

from slmet.stations import load_registry


def test_bundled_registry_is_well_formed():
    df = load_registry()
    assert len(df) == 24
    assert df["station"].is_unique
    assert not df["station"].str.startswith("unknown_").any()
    assert df["longitude"].between(79, 82.5).all()
    assert df["latitude"].between(5.5, 10.5).all()
