"""Tests for sx_phot.image_index."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sx_phot.image_index import (
    decode_image_key,
    encode_image_key,
    images_containing,
    parse_filename,
    parse_obscore_snapshot,
    scan_local_store,
    shard_name,
)


def test_image_key_roundtrip():
    """Keys decode back to their components and stay unique."""
    cases = [
        ("2025W19_2B_0422_1", 1, 2),
        ("2026W30_1A_9999_9", 6, 3),
        ("2025W17_4D_0001_0", 4, 2),
    ]
    keys = []
    for obs_id, det, qr in cases:
        key = encode_image_key(obs_id, det, qr)
        dec = decode_image_key(key)
        assert dec["obs_id"] == obs_id
        assert dec["det"] == det
        assert dec["qr"] == qr
        assert dec["week_cam"] == obs_id[:10]
        keys.append(key)
    assert len(set(keys)) == len(keys)
    assert max(keys) < 2**62


def test_image_key_rejects_bad_obs_id():
    """Malformed observation ids raise."""
    with pytest.raises(ValueError):
        encode_image_key("2025W19_2B_422_1", 1, 2)


def test_parse_filename():
    """Level-2 filenames parse into (obs_id, det, version)."""
    name = "level2_2025W19_2B_0422_1D1_spx_l2b-v20-2025-247.fits"
    assert parse_filename(name) == ("2025W19_2B_0422_1", 1, "l2b-v20-2025-247")
    retry = "level2_2026W30_1B_0001_1D3_spx_l2b_retry-v27-2026-223.fits"
    assert parse_filename(retry) == (
        "2026W30_1B_0001_1", 3, "l2b_retry-v27-2026-223"
    )
    assert parse_filename("something_else.fits") is None
    assert shard_name("2025W19_2B", "l2b-v20-2025-247", 1) == (
        "2025W19_2B__l2b-v20-2025-247__D1"
    )


def test_scan_local_store(tmp_path: Path):
    """The store scanner finds files at the expected depth."""
    d = tmp_path / "2025W19_2B" / "l2b-v20-2025-247" / "1"
    d.mkdir(parents=True)
    (d / "level2_2025W19_2B_0422_1D1_spx_l2b-v20-2025-247.fits").touch()
    (d / "level2_2025W19_2B_0422_2D1_spx_l2b-v20-2025-247.fits").touch()
    (d / "junk.fits").touch()
    df = scan_local_store(tmp_path)
    assert len(df) == 2
    assert set(df["obs_id"]) == {"2025W19_2B_0422_1", "2025W19_2B_0422_2"}
    assert (df["det"] == 1).all()
    assert (df["ver"] == "l2b-v20-2025-247").all()
    assert (df["week_cam"] == "2025W19_2B").all()


def _square_polygon(ra0: float, dec0: float, half: float) -> str:
    corners = [
        (ra0 - half, dec0 - half), (ra0 + half, dec0 - half),
        (ra0 + half, dec0 + half), (ra0 - half, dec0 + half),
        (ra0 - half, dec0 - half),
    ]
    return "POLYGON ICRS " + " ".join(f"{a} {d}" for a, d in corners)


def test_parse_obscore_and_images_containing(tmp_path: Path):
    """ObsCore rows parse, calibration rows drop, and footprints match."""
    rows = [
        ("2025W18_1B_0001_1", "spherex_qr2_deep", "SPHEREx-D1", 10.0, 0.0,
         60793.5, 60793.6, _square_polygon(10.0, 0.0, 1.5)),
        ("2025W18_1B_0001_1", "spherex_qr2_deep", "SPHEREx-D2", 20.0, 0.0,
         60793.5, 60793.6, _square_polygon(20.0, 0.0, 1.5)),
        ("2025W18_1B_0001_1", "spherex_qr2_cal", "SPHEREx-All", np.nan,
         np.nan, np.nan, np.nan, ""),
    ]
    df = pd.DataFrame(rows, columns=[
        "obs_id", "obs_collection", "energy_bandpassname", "s_ra", "s_dec",
        "t_min", "t_max", "s_region",
    ])
    csv = tmp_path / "obscore.csv"
    df.to_csv(csv, index=False)
    parsed = parse_obscore_snapshot(csv)
    assert len(parsed) == 2
    assert list(parsed["det"]) == [1, 2]
    assert (parsed["qr"] == 2).all()
    assert parsed["ra1"].iloc[0] == pytest.approx(8.5)
    assert parsed["dec3"].iloc[0] == pytest.approx(1.5)

    hit = images_containing(parsed, 10.5, 0.5)
    assert list(hit) == [0]
    hit = images_containing(parsed, 21.4, -1.4)
    assert list(hit) == [1]
    assert images_containing(parsed, 15.0, 0.0).size == 0
    assert images_containing(parsed, 10.0, 1.6).size == 0
