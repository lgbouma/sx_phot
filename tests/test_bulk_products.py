"""Tests for sx_phot.bulk_products on a synthetic run directory."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sx_phot.bulk_phot import ROW_DTYPES, PhotConfig
from sx_phot.bulk_products import (
    LEGACY_COLUMNS,
    gather_by_star,
    load_star_rows,
    materialize_star,
    supplement_with_spline,
    to_legacy_frame,
)


def _rows(star_idx, ticid, image_key, lam, flux) -> dict:
    flux = np.asarray(flux, dtype=float)
    n = len(star_idx)
    rows = {k: np.zeros(n, dtype=v) for k, v in ROW_DTYPES.items()}
    rows["star_idx"][:] = star_idx
    rows["ticid"][:] = ticid
    rows["image_key"][:] = image_key
    rows["wavelength_um"][:] = lam
    rows["bandwidth_um"][:] = 0.02
    rows["flux_jy"][:] = flux
    rows["flux_err_jy"][:] = 0.01
    rows["flux_raw_jy"][:] = flux + 0.5
    rows["bkg_jy"][:] = 0.5
    rows["mjd_avg"][:] = 61000.0
    rows["flag_or"][:] = 0
    return rows


def _make_run(tmp_path: Path, n_stars: int = 5) -> Path:
    run = tmp_path / "run"
    (run / "phot" / "A").mkdir(parents=True)
    (run / "phot" / "B").mkdir(parents=True)
    cfg = {"phot": asdict(PhotConfig()), "n_stars": n_stars}
    (run / "config.json").write_text(json.dumps(cfg))
    pd.DataFrame({
        "star_idx": np.arange(n_stars, dtype=np.int32),
        "ticid": np.arange(100, 100 + n_stars, dtype=np.int64),
        "ra": np.linspace(10, 11, n_stars), "dec": np.zeros(n_stars),
        "pmra": np.zeros(n_stars), "pmdec": np.zeros(n_stars),
    }).to_parquet(run / "catalog.parquet", index=False)
    pd.DataFrame({
        "image_key": [11, 12, 13], "path": ["/p/11", "/p/12", "/p/13"],
        "qr": [2, 2, 3], "ver": ["v1", "v1", "v2"],
        "week_cam": ["2025W18_1B"] * 3,
        "obs_id": ["2025W18_1B_0001_1"] * 3, "det": [1, 2, 3],
    }).to_parquet(run / "index.parquet", index=False)
    # Star 2 (TIC 102) appears on all three images, star 0 on one.
    a = _rows([2, 0], [102, 100], [11, 11], [0.8, 0.9], [1.0, 2.0])
    b = _rows([2, 2], [102, 102], [12, 13], [1.5, 1.1], [1.2, 1.1])
    for shard, rows in (("A", a), ("B", b)):
        table = pa.table({k: pa.array(v) for k, v in rows.items()})
        pq.write_table(table, run / "phot" / shard / "pass_1.parquet")
    return run


def test_gather_and_load(tmp_path: Path):
    """Gathering buckets rows by star and loading returns sorted rows."""
    run = _make_run(tmp_path)
    meta = gather_by_star(run, stars_per_bucket=2)
    assert meta["n_rows"] == 4
    assert meta["n_buckets"] == 3
    assert (run / "bystar" / "bucket=1").exists()
    rows = load_star_rows(run, 102)
    assert len(rows) == 3
    assert list(rows["image_key"]) == [11, 13, 12]
    assert rows["wavelength_um"].is_monotonic_increasing
    assert list(rows["path"]) == ["/p/11", "/p/13", "/p/12"]
    assert list(rows["qr"]) == [2, 3, 2]
    assert len(load_star_rows(run, 100)) == 1
    assert len(load_star_rows(run, 101)) == 0
    with pytest.raises(KeyError):
        load_star_rows(run, 999)


def test_gather_rejects_bad_bucket_size(tmp_path: Path):
    """stars_per_bucket and buckets_per_group must be powers of two."""
    run = _make_run(tmp_path)
    with pytest.raises(ValueError):
        gather_by_star(run, stars_per_bucket=3)
    with pytest.raises(ValueError):
        gather_by_star(run, stars_per_bucket=2, buckets_per_group=3)


def test_gather_many_groups_matches_pass_files(tmp_path: Path):
    """Multi-group gather keeps every row, sorted by star within bucket."""
    run = _make_run(tmp_path)
    meta = gather_by_star(run, stars_per_bucket=1, buckets_per_group=2,
                          nproc=2)
    assert meta["n_rows"] == 4
    assert meta["n_buckets"] == 5
    assert not (run / "bystar.groups").exists()
    assert not (run / "bystar.part").exists()
    assert sorted(p.name for p in (run / "bystar").iterdir()) == [
        "bucket=0", "bucket=2"]
    got = pd.concat(
        pd.read_parquet(p) for p in (run / "bystar").glob("*/*.parquet")
    )
    want = pd.concat(
        pd.read_parquet(p) for p in (run / "phot").glob("*/*.parquet")
    )
    assert list(got.columns) == list(ROW_DTYPES)
    key = ["star_idx", "image_key"]
    pd.testing.assert_frame_equal(
        got.sort_values(key).reset_index(drop=True),
        want.sort_values(key).reset_index(drop=True),
    )
    assert len(load_star_rows(run, 102)) == 3


def test_legacy_frame_and_materialize(tmp_path: Path):
    """Legacy columns are produced and the CSV products are written."""
    run = _make_run(tmp_path)
    gather_by_star(run, stars_per_bucket=2)
    rows = load_star_rows(run, 102)
    cfg = json.loads((run / "config.json").read_text())["phot"]
    legacy = to_legacy_frame(rows, cfg)
    for col in LEGACY_COLUMNS:
        assert col in legacy.columns
    assert (~legacy["masked"]).all()
    assert legacy["tjd_avg"].iloc[0] == pytest.approx(61000.0 - 56999.5)
    assert legacy["aperture_radius_pix"].iloc[0] == 2.0
    paths = materialize_star(run, 102, tmp_path / "out", supplement=False)
    assert paths["cache"].exists()
    assert "supp" not in paths
    back = pd.read_csv(paths["cache"])
    assert len(back) == 3
    assert "ra" in paths["cache"].name


def test_supplement_with_spline_columns():
    """The spline supplement adds the six model columns."""
    rng = np.random.default_rng(0)
    lam = np.linspace(0.75, 5.0, 300)
    base = pd.DataFrame({
        "wavelength_um": lam,
        "bandwidth_um": np.full(lam.size, 0.02),
        "flux_jy": 1.0 + 0.01 * rng.standard_normal(lam.size),
        "flux_err_jy": np.full(lam.size, 0.01),
        "masked": np.zeros(lam.size, dtype=bool),
    })
    supp = supplement_with_spline(base)
    for col in ("model_flux", "fit_mask", "residuals", "res_valid",
                "res_masked", "res_clipped"):
        assert col in supp.columns
    assert np.isfinite(supp["model_flux"]).mean() > 0.9
    assert supp["res_valid"].sum() > 200


def test_materialize_run_writes_all_stars(tmp_path: Path):
    """materialize_run writes one CSV per star with rows and a summary."""
    from sx_phot.bulk_products import materialize_run

    run = _make_run(tmp_path)
    gather_by_star(run, stars_per_bucket=2)
    summary = materialize_run(run, tmp_path / "csv", id_prefix="GaiaDR3",
                              supplement=False, nproc=2)
    assert len(summary) == 5
    assert set(summary.loc[summary["n"] > 0, "ticid"]) == {100, 102}
    assert summary.set_index("ticid").loc[102, "n"] == 3
    assert summary.set_index("ticid").loc[102, "star_id"] == "GaiaDR3_102"
    written = sorted(p.name for p in (tmp_path / "csv").glob("*.csv"))
    assert len(written) == 3  # two stars + summary.csv
    assert any(n.startswith("sxphot_cache_GaiaDR3_102_") for n in written)
    assert (tmp_path / "csv" / "summary.csv").exists()
    one = pd.read_csv(summary.set_index("ticid").loc[102, "cache"])
    assert list(one["wavelength_um"]) == sorted(one["wavelength_um"])


def test_materialize_run_subset(tmp_path: Path):
    """ticids restricts the output to the requested stars."""
    from sx_phot.bulk_products import materialize_run

    run = _make_run(tmp_path)
    gather_by_star(run, stars_per_bucket=2)
    out = tmp_path / "csv"
    summary = materialize_run(run, out, nproc=1, ticids=[102, 999])
    assert list(summary["ticid"]) == [102]
    assert int(summary["n"].iloc[0]) == 3
    files = sorted(p.name for p in out.glob("*.csv"))
    assert len(files) == 2  # cache (3 rows: no spline), summary
    assert all("TIC_102" in f or f == "summary.csv" for f in files)
