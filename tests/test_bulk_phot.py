"""Tests for sx_phot.bulk_phot (unit tests plus an on-disk check)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sx_phot.bulk_phot import (
    PhotConfig,
    jyear_to_mjd,
    load_catalog,
    photometer_image,
    plan_shards,
    processed_keys,
    propagate_linear,
)
from sx_phot.circphot import BITMASK

TEST_RESULTS = Path(__file__).resolve().parents[1] / "drivers" / "test_results"
REFERENCE_CSV = (
    "sxphot_cache_TIC_141146667_ra166p31182521_dec59p25177957_annulus.csv"
)


def test_jyear_to_mjd():
    """J2016.0 is MJD 57388.5."""
    assert jyear_to_mjd(2016.0) == pytest.approx(57388.5, abs=1e-6)


def test_propagate_linear():
    """Linear propagation moves by pm * dt with the cos(dec) factor."""
    ra, dec = propagate_linear(
        np.array([10.0]), np.array([60.0]), np.array([3600.0]),
        np.array([-3600.0]), epoch_mjd=0.0, mjd=365.25 * 10,
    )
    # 3600 mas/yr for 10 yr = 36 arcsec = 0.01 deg.
    assert dec[0] == pytest.approx(60.0 - 0.01, abs=1e-9)
    assert ra[0] == pytest.approx(10.0 + 0.01 / np.cos(np.deg2rad(60.0)),
                                  abs=1e-9)
    ra, _ = propagate_linear(
        np.array([359.999]), np.array([0.0]), np.array([3.6e6 * 0.01]),
        np.array([0.0]), epoch_mjd=0.0, mjd=365.25,
    )
    assert ra[0] == pytest.approx(0.009, abs=1e-9)


def test_load_catalog_csv(tmp_path: Path):
    """CSV catalogs load with zero-filled proper motions."""
    p = tmp_path / "cat.csv"
    pd.DataFrame({
        "TICID": [5, 7], "ra": [1.0, 2.0], "dec": [3.0, 4.0],
        "pmra": [np.nan, 1.0], "pmdec": [2.0, np.nan],
    }).to_csv(p, index=False)
    cat = load_catalog(p)
    assert list(cat["star_idx"]) == [0, 1]
    assert list(cat["ticid"]) == [5, 7]
    assert cat["pmra"].tolist() == [0.0, 1.0]
    assert cat["pmdec"].tolist() == [2.0, 0.0]


def test_plan_shards_skips_processed(tmp_path: Path):
    """Only unprocessed local files are planned."""
    index = pd.DataFrame({
        "image_key": [1, 2, 3, 4],
        "path": ["/a/1", "/a/2", "/b/3", None],
        "shard": ["A", "A", "B", "B"],
        "s_ra": [0.0] * 4, "s_dec": [0.0] * 4,
    })
    shard_a = tmp_path / "phot" / "A"
    shard_a.mkdir(parents=True)
    np.save(shard_a / "pass_x.keys.npy", np.array([1], dtype=np.int64))
    assert list(processed_keys(shard_a)) == [1]
    plan = plan_shards(tmp_path, index)
    assert [s for s, _ in plan] == ["A", "B"]
    assert list(plan[0][1]["image_key"]) == [2]
    assert list(plan[1][1]["image_key"]) == [3]
    plan = plan_shards(tmp_path, index, shard_glob="B*")
    assert [s for s, _ in plan] == ["B"]


def _reference_rows() -> tuple[float, float, pd.DataFrame]:
    """Return (ra, dec, rows) from a cached old-pipeline CSV on disk."""
    csv = TEST_RESULTS / REFERENCE_CSV
    if not csv.exists():
        pytest.skip("reference CSV not available")
    m = re.search(r"_ra(-?\d+)p(\d+)_dec(-?\d+)p(\d+)_", csv.name)
    ra = float(f"{m.group(1)}.{m.group(2)}")
    dec = float(f"{m.group(3)}.{m.group(4)}")
    ref = pd.read_csv(csv)
    ref = ref[ref["file"].apply(os.path.exists)].head(3)
    if ref.empty:
        pytest.skip("reference Level-2 files not on this host")
    return ra, dec, ref


def test_photometer_image_matches_circphot():
    """Exact-statistics mode reproduces the cached circphot numbers."""
    ra, dec, ref = _reference_rows()
    cfg = PhotConfig(stats_stride=1, min_file_age_sec=0.0)
    for r in ref.itertuples():
        rows, meta = photometer_image(
            r.file, np.array([ra]), np.array([dec]), config=cfg,
        )
        assert rows["flux_jy"].size == 1
        assert meta["n_in"] == 1
        assert rows["flux_jy"][0] == pytest.approx(r.flux_jy, rel=1e-5)
        assert rows["flux_err_jy"][0] == pytest.approx(r.flux_err_jy,
                                                       rel=1e-5)
        assert rows["wavelength_um"][0] == pytest.approx(r.wavelength_um,
                                                         rel=1e-6)
        assert rows["bandwidth_um"][0] == pytest.approx(r.bandwidth_um,
                                                        rel=1e-6)
        assert rows["mjd_avg"][0] == pytest.approx(r.mjd_avg, abs=1e-9)
        masked = bool(int(rows["flag_or"][0]) & BITMASK)
        assert masked == bool(r.masked)
        assert rows["flux_raw_jy"][0] - rows["bkg_jy"][0] == pytest.approx(
            rows["flux_jy"][0], rel=1e-5,
        )
        assert np.isfinite(rows["zodi_jy"][0])


def test_photometer_image_subsampled_stats_close():
    """The subsampled star mask changes fluxes by well under one sigma."""
    ra, dec, ref = _reference_rows()
    cfg = PhotConfig(stats_stride=8, min_file_age_sec=0.0)
    for r in ref.itertuples():
        rows, _ = photometer_image(
            r.file, np.array([ra]), np.array([dec]), config=cfg,
        )
        assert abs(rows["flux_jy"][0] - r.flux_jy) < 0.1 * r.flux_err_jy


def test_photometer_image_no_candidates():
    """Images with no candidate stars return empty rows quickly."""
    _, _, ref = _reference_rows()
    rows, meta = photometer_image(
        ref["file"].iloc[0], np.zeros(0), np.zeros(0),
        config=PhotConfig(min_file_age_sec=0.0),
    )
    assert rows["flux_jy"].size == 0
    assert meta["n_cand"] == 0


BAND_EDGE_FILE = (
    "/ar0/SPHEREX/spherex_l2/2025W17_4B/l2b-v20-2025-241/1/"
    "level2_2025W17_4B_0101_4D1_spx_l2b-v20-2025-241.fits"
)


def test_band_edge_annulus_fallback():
    """A star on the bright D1 band edge gets a non-zero background.

    The global star mask flags the whole band-edge region (sky several
    times brighter than the image median), which used to empty the
    annulus and silently set the background to zero.
    """
    if not os.path.exists(BAND_EDGE_FILE):
        pytest.skip("band-edge Level-2 file not on this host")
    ra, dec = 271.9293008, 65.3522887  # TIC 233085422, epoch 2016.0
    pmra, pmdec = -25.4304, 11.7161
    cfg = PhotConfig(min_file_age_sec=0.0)
    rows, meta = photometer_image(
        BAND_EDGE_FILE, np.array([ra]), np.array([dec]),
        pmra_masyr=np.array([pmra]), pmdec_masyr=np.array([pmdec]),
        config=cfg,
    )
    assert meta["n_in"] == 1
    assert rows["bkg_mode"][0] == 1
    assert rows["n_ann_pix"][0] >= cfg.min_ann_pix
    assert rows["bkg_jy"][0] > 0.0
    # The sky here is ~4x the image median; the raw aperture sum would
    # overestimate the star flux by about a factor of two.
    assert rows["flux_jy"][0] < 0.7 * rows["flux_raw_jy"][0]

    # An empty annulus always falls back, whatever the threshold.
    strict = PhotConfig(min_file_age_sec=0.0, min_ann_pix=0)
    rows0, _ = photometer_image(
        BAND_EDGE_FILE, np.array([ra]), np.array([dec]),
        pmra_masyr=np.array([pmra]), pmdec_masyr=np.array([pmdec]),
        config=strict,
    )
    assert rows0["bkg_mode"][0] == 1
    assert rows0["bkg_jy"][0] == pytest.approx(rows["bkg_jy"][0])
