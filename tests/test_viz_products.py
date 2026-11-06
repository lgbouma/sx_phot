"""Tests for sx_phot.viz_products (binning and per-star products)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sx_phot.sedmodels import spherex_bin_edges
from sx_phot.viz_products import (MASK_EDGE, MASK_HARD, MASK_OUTLIER,
                                  MASK_SOFT, SPECTRUM_COLUMNS, bin_spectrum,
                                  gross_outliers, quality_masks,
                                  star_products)

CFG = {"aperture_radius": 2.0, "annulus_r_in": 6.0, "annulus_r_out": 8.0}


def _fake_rows(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lam = np.sort(rng.uniform(0.75, 5.0, n)).astype(np.float32)
    flux = (0.01 * (lam / 1.0) ** -1.5).astype(np.float32)
    err = (0.02 * flux).astype(np.float32)
    flux = (flux + rng.normal(0, 1, n).astype(np.float32) * err)
    return pd.DataFrame({
        "star_idx": np.zeros(n, np.int32), "ticid": np.full(n, 123, np.int64),
        "image_key": np.arange(n, dtype=np.int64),
        "x": rng.uniform(10, 2030, n).astype(np.float32),
        "y": rng.uniform(10, 2030, n).astype(np.float32),
        "flux_jy": flux, "flux_err_jy": err, "flux_raw_jy": flux,
        "bkg_jy": np.zeros(n, np.float32), "zodi_jy": np.zeros(n, np.float32),
        "flag_or": np.where(rng.uniform(size=n) < 0.05, 1, 0).astype(np.int32),
        "n_bad_pix": np.zeros(n, np.int16),
        "n_ap_pix": np.full(n, 12, np.int16),
        "n_ann_pix": np.full(n, 80, np.int16),
        "bkg_mode": np.zeros(n, np.int8),
        "wavelength_um": lam, "bandwidth_um": (lam / 40).astype(np.float32),
        "mjd_avg": 60800 + rng.uniform(0, 400, n),
        "path": ["/nonexistent/img.fits"] * n,
    })


def test_bin_spectrum_weighted_mean():
    edges = spherex_bin_edges()
    lam = np.array([0.76, 0.76, 0.76])
    flux = np.array([1.0, 3.0, 100.0])
    err = np.array([1.0, 1.0, 1.0])
    good = np.array([True, True, False])
    out = bin_spectrum(lam, flux, err, good, edges)
    assert out["n"][0] == 2
    assert out["flux"][0] == pytest.approx(2.0)
    assert out["err"][0] == pytest.approx(1 / np.sqrt(2))
    assert out["std"][0] == pytest.approx(1.0)
    assert np.isnan(out["flux"][1])
    assert np.isnan(out["model"][0])


def test_bin_spectrum_outside_range_ignored():
    edges = spherex_bin_edges()
    out = bin_spectrum(np.array([0.5, 6.0]), np.ones(2), np.ones(2),
                       np.ones(2, bool), edges)
    assert out["n"].sum() == 0


def test_star_products_shapes_and_summary():
    edges = spherex_bin_edges()
    rows = _fake_rows()
    spectrum, summary, binned = star_products(rows, CFG, edges)
    assert len(spectrum) == len(rows)
    assert list(spectrum.columns) == list(SPECTRUM_COLUMNS)
    for col, dt in SPECTRUM_COLUMNS.items():
        assert spectrum[col].dtype == dt, col
    assert summary["n_rows"] == len(rows)
    # Bit 0 (TRANSIENT) is soft: consistent points are used, not masked.
    n_soft = int((rows["flag_or"] & 1).astype(bool).sum())
    assert summary["n_soft"] == n_soft
    assert summary["n_soft_used"] > 0.8 * n_soft
    assert summary["n_masked"] == n_soft - summary["n_soft_used"]
    assert summary["n_used"] <= summary["n_good"] <= summary["n_rows"]
    assert summary["spline_ok"]
    assert summary["med_snr"] > 10
    assert binned["flux"].shape == (edges.size - 1,)
    assert np.isfinite(binned["flux"]).sum() > 50
    # The binned spline model tracks the binned flux.
    ok = np.isfinite(binned["model"]) & np.isfinite(binned["flux"])
    ratio = binned["model"][ok] / binned["flux"][ok]
    assert np.nanmedian(np.abs(ratio - 1)) < 0.05
    assert np.isfinite(summary["c_3p6_1p6"])


def test_star_products_tiny_input():
    edges = spherex_bin_edges()
    rows = _fake_rows(n=3)
    spectrum, summary, binned = star_products(rows, CFG, edges)
    assert len(spectrum) == 3
    assert not summary["spline_ok"]
    assert summary["n_used"] == summary["n_good"]


def test_star_products_tolerates_missing_band():
    edges = spherex_bin_edges()
    rows = _fake_rows(n=400)
    rows = rows[rows["wavelength_um"] < 4.3].reset_index(drop=True)
    rows.loc[rows["wavelength_um"] > 3.9, "flag_or"] = 0
    spectrum, summary, binned = star_products(rows, CFG, edges)
    assert summary["spline_ok"]
    band6 = spectrum["wavelength_um"] > 4.42
    assert band6.sum() == 0
    # Band 5 has few points, so it may be unfit: those points stay used.
    unfit = ~np.isfinite(spectrum["model_flux"]) & ~spectrum["masked"]
    assert spectrum.loc[unfit, "fit_mask"].all()
    assert summary["n_used"] == summary["n_good"] - summary["n_clipped"]


def test_quality_masks_edge_hard_soft_outlier():
    rows = _fake_rows(n=200)
    rows["flag_or"] = 0
    rows.loc[0, "flag_or"] = 1 << 0          # soft (TRANSIENT)
    rows.loc[1, "flag_or"] = 1 << 6          # hard (NONFUNC)
    rows.loc[2, "flag_or"] = (1 << 1) | (1 << 2)  # soft + hard -> hard
    rows.loc[3, "x"] = 2039.4                # aperture off the array
    rows.loc[4, "y"] = 1.5
    rows.loc[5, "x"] = 2.5                   # just inside
    rows.loc[6, "flux_jy"] = rows.loc[6, "flux_jy"] * 5  # gross outlier
    bits, masked = quality_masks(rows, include_soft=True)
    assert bits[0] == MASK_SOFT and not masked[0]
    assert bits[1] == MASK_HARD and masked[1]
    assert bits[2] == MASK_HARD and masked[2]
    assert bits[3] == MASK_EDGE and masked[3]
    assert bits[4] == MASK_EDGE and masked[4]
    assert bits[5] == 0 and not masked[5]
    assert bits[6] == MASK_OUTLIER and masked[6]
    assert masked.sum() == 5
    _, masked_strict = quality_masks(rows)
    assert masked_strict[0] and masked_strict.sum() == 6


def test_gross_outliers_spares_real_features():
    rng = np.random.default_rng(5)
    lam = np.sort(rng.uniform(0.75, 5.0, 400))
    flux = 0.01 * lam ** -1.5
    # A 35 percent absorption band (like water at 1.4 um) must survive.
    flux *= 1 - 0.35 * np.exp(-0.5 * ((lam - 1.4) / 0.05) ** 2)
    err = 0.02 * flux
    flux[100] *= 3.0
    flux[200] *= 0.2
    bad = gross_outliers(lam, flux, err, np.ones(lam.size, bool))
    assert set(np.flatnonzero(bad)) == {100, 200}


def test_star_products_rejects_inconsistent_soft_points():
    edges = spherex_bin_edges()
    rows = _fake_rows(n=400)
    rows["flag_or"] = 0
    rows.loc[10:29, "flag_or"] = 1 << 1               # soft OVERFLOW
    rows.loc[10:19, "flux_jy"] = rows.loc[10:19, "flux_jy"] * 1.6  # bad
    spectrum, summary, binned = star_products(rows, CFG, edges)
    assert summary["n_soft"] == 20
    assert 8 <= summary["n_soft_used"] <= 10
    assert spectrum.loc[10:19, "masked"].all()
    assert not spectrum.loc[10:19, "fit_mask"].any()
    assert spectrum.loc[20:29, "fit_mask"].sum() >= 8
    assert summary["n_used"] == summary["n_good"] - summary["n_clipped"]
    assert (spectrum.loc[10:29, "mask_bits"] & MASK_SOFT).all()


def test_spectrum_store_bounds_open_files(tmp_path: Path):
    """Reads across many buckets keep at most ``max_open`` handles."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from sx_phot.viz_products import SpectrumStore

    spectra = tmp_path / "spectra"
    spectra.mkdir()
    rows = []
    n_buckets, per_bucket = 6, 3
    for b in range(n_buckets):
        with pq.ParquetWriter(
                spectra / f"bucket={b}.parquet",
                pa.schema([("ticid", pa.int64()),
                           ("mjd_avg", pa.float64())])) as w:
            for r in range(per_bucket):
                tic = 1000 * b + r
                w.write_table(pa.table({"ticid": [tic, tic],
                                        "mjd_avg": [60000.0, 60001.0]}))
                rows.append((tic, b, r))
    summary = pd.DataFrame(rows, columns=["ticid", "bucket", "row_group"])
    store = SpectrumStore(tmp_path, summary, max_open=2)
    assert store.prewarm() == 2
    for tic, b, r in rows:
        df = store.read(tic)
        assert len(df) == 2 and int(df["ticid"].iloc[0]) == tic
        assert len(store._files) <= 2
    # Most recently used bucket is kept, the oldest evicted.
    assert list(store._files) == [n_buckets - 2, n_buckets - 1]
    assert 9999 not in store
