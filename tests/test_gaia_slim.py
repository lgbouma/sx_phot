"""Tests for sx_phot.gaia_slim (synthetic parts, fake TAP service)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.table import MaskedColumn, Table

import sx_phot.gaia_slim as gs
from sx_phot.gaia_slim import (
    COLUMNS,
    HEALPIX_SHIFT,
    GaiaSlim,
    cone_id_ranges,
    fetch_all,
    read_manifest,
)


def _synthetic_sky(n: int, seed: int = 0) -> pd.DataFrame:
    """Random sources with source_id consistent with their position."""
    import astropy.units as u
    from astropy_healpix import HEALPix

    rng = np.random.default_rng(seed)
    ra = rng.uniform(44.0, 46.0, n)
    dec = rng.uniform(-1.0, 1.0, n)
    hp = HEALPix(nside=1 << 12, order="nested")
    pix = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg).astype(np.int64)
    sid = (pix << HEALPIX_SHIFT) + rng.integers(1, 1 << 30, n)
    df = pd.DataFrame({
        "source_id": sid, "ra": ra, "dec": dec,
        "pmra": rng.normal(size=n).astype(np.float32),
        "pmdec": rng.normal(size=n).astype(np.float32),
        "phot_g_mean_mag": rng.uniform(10, 21, n).astype(np.float32),
    })
    return df.sort_values("source_id").reset_index(drop=True)


def _write_parts(out: Path, df: pd.DataFrame, rows_per_part: int) -> None:
    (out / "parts").mkdir(parents=True)
    rows = []
    for k, start in enumerate(range(0, len(df), rows_per_part)):
        part = df.iloc[start:start + rows_per_part]
        part.to_parquet(out / "parts" / f"part_{k:05d}.parquet",
                        index=False, row_group_size=50)
        rows.append({"part": k, "min_id": int(part.source_id.iloc[0]),
                     "max_id": int(part.source_id.iloc[-1]),
                     "n_rows": len(part)})
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)


def test_cone_id_ranges_merge_neighbours():
    """Adjacent pixels merge into one range; ranges are pixel aligned."""
    ranges = cone_id_ranges(45.0, 0.0, 120.0)
    assert ranges
    for lo, hi in ranges:
        assert lo % (1 << HEALPIX_SHIFT) == 0 and hi > lo
    assert all(ranges[i][1] < ranges[i + 1][0] for i in range(len(ranges) - 1))


def test_cone_matches_brute_force(tmp_path: Path):
    """Cone search reproduces a brute-force separation cut."""
    df = _synthetic_sky(5000)
    _write_parts(tmp_path, df, rows_per_part=1200)
    g = GaiaSlim(tmp_path)
    rng = np.random.default_rng(1)
    for _ in range(20):
        ra0, dec0 = rng.uniform(44.2, 45.8), rng.uniform(-0.8, 0.8)
        got = g.cone(ra0, dec0, 300.0)
        ra, dec = np.radians(df.ra), np.radians(df.dec)
        a, d = np.radians(ra0), np.radians(dec0)
        cosd = (np.sin(dec) * np.sin(d)
                + np.cos(dec) * np.cos(d) * np.cos(ra - a))
        sep = np.degrees(np.arccos(np.clip(cosd, -1, 1))) * 3600
        want = set(df.source_id[sep <= 300.0])
        assert set(got.source_id) == want
        assert got.sep_arcsec.is_monotonic_increasing
        assert list(got.columns) == COLUMNS + ["sep_arcsec"]


def test_fetch_all_resumes_and_stops(tmp_path: Path, monkeypatch):
    """Chunks follow the cursor, parts land in the manifest, resume works."""
    sky = _synthetic_sky(250, seed=2)
    calls = []

    def fake_chunk(service, last_id, max_rows, timeout_s=0):
        calls.append(last_id)
        return sky[sky.source_id > last_id].head(max_rows).reset_index(
            drop=True)

    class FakeTAP:
        def __init__(self, url):
            self.url = url

    import pyvo

    monkeypatch.setattr(pyvo.dal, "TAPService", FakeTAP)
    monkeypatch.setattr(gs, "fetch_chunk", fake_chunk)
    man = fetch_all(tmp_path, max_rows=100, pause_s=0, max_parts=1)
    assert len(man) == 1 and calls == [-1]
    man = fetch_all(tmp_path, max_rows=100, pause_s=0)
    assert list(man.n_rows) == [100, 100, 50]
    assert calls[1] == int(sky.source_id.iloc[99])
    got = pd.concat([pd.read_parquet(p) for p in
                     sorted((tmp_path / "parts").glob("*.parquet"))])
    assert np.array_equal(got.source_id.to_numpy(), sky.source_id.to_numpy())
    n = len(calls)
    assert len(fetch_all(tmp_path, max_rows=100, pause_s=0)) == 3
    assert len(calls) == n       # complete copies are not re-queried
    assert list(read_manifest(tmp_path).columns) == [
        "part", "min_id", "max_id", "n_rows"]


def test_to_frame_masks_to_nan():
    """Masked proper motions become NaN float32; ids stay int64."""
    t = Table({
        "source_id": [5, 3], "ra": [1.0, 2.0], "dec": [0.0, 0.5],
        "pmra": MaskedColumn([1.0, 2.0], mask=[True, False]),
        "pmdec": MaskedColumn([1.0, 2.0], mask=[False, True]),
        "phot_g_mean_mag": [15.0, 16.0],
    })
    df = gs._to_frame(t)
    assert list(df.source_id) == [3, 5]
    assert df.source_id.dtype == np.int64 and df.pmra.dtype == np.float32
    assert np.isnan(df.pmra.iloc[1]) and np.isnan(df.pmdec.iloc[0])
    assert df.phot_g_mean_mag.iloc[0] == pytest.approx(16.0)
