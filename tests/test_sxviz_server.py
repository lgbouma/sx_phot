"""Tests for the sx-viz server's data layer on a synthetic viz build."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "sx-viz"))

from sx_phot.sedmodels import spherex_bin_edges  # noqa: E402
from sx_phot.viz_products import SpectrumStore, star_products  # noqa: E402
from test_viz_products import CFG, _fake_rows  # noqa: E402

server = pytest.importorskip("server")


@pytest.fixture(scope="module")
def viz_build(tmp_path_factory):
    """A run dir with a two-star viz build and a matching TARS table."""
    run = tmp_path_factory.mktemp("run")
    viz = run / "viz"
    (viz / "spectra").mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"phot": CFG}))
    edges = spherex_bin_edges()
    tics = [111, 222]
    specs, summaries, binned = [], [], []
    for rg, tic in enumerate(tics):
        rows = _fake_rows(n=300, seed=tic)
        rows["ticid"] = tic
        spectrum, summary, b = star_products(rows, CFG, edges)
        spectrum.insert(0, "ticid", np.int64(tic))
        summary.update({"ticid": tic, "star_idx": rg, "bucket": 0,
                        "row_group": rg, "ra": 10.0 + rg, "dec": -5.0})
        specs.append(spectrum)
        summaries.append(summary)
        binned.append(b)
    schema = pa.Schema.from_pandas(specs[0], preserve_index=False)
    with pq.ParquetWriter(viz / "spectra" / "bucket=0.parquet",
                          schema) as w:
        for sp in specs:
            w.write_table(pa.Table.from_pandas(sp, schema=schema,
                                               preserve_index=False),
                          row_group_size=len(sp))
    pd.DataFrame(summaries).to_parquet(viz / "summary.parquet", index=False)
    np.save(viz / "binned_ticid.npy", np.array(tics, np.int64))
    for key in ("flux", "err", "std", "n", "model"):
        np.save(viz / f"binned_{key}.npy", np.stack([b[key] for b in binned]))
    np.save(viz / "bin_edges.npy", edges)
    pd.DataFrame({"image_key": [0], "path": ["/nonexistent.fits"],
                  "obs_id": ["x"], "det": [1]}).to_parquet(
        run / "index.parquet", index=False)
    tars = pd.DataFrame({
        "TICID": tics + [333], "ra": [0, 0, 0], "dec": [0, 0, 0],
        "parallax": [10.0, 20.0, 5.0], "teff": [3500.0, 5500.0, 4000.0],
        "Tmag": [10.0, 12.0, 9.0], "adopted_period": [1.0, 2.0, 3.0],
        "ruwe": [1.0, 1.5, 1.1], "median_amplitude": [1.0, 2.0, 3.0],
        "flag_possible_binary": pd.array([True, False, None],
                                         dtype="boolean"),
        "sectors": ["1,2", "3", "4"],
    })
    tars.to_feather(run / "tars.feather")
    return run


@pytest.fixture(scope="module")
def data(viz_build):
    from sx_phot.sedmodels import SPHINX_DIR

    if not any(SPHINX_DIR.glob("Teff_*.txt")):
        pytest.skip("SPHINX model grid not on this host")
    return server.Data(viz_build, None, viz_build / "tars.feather")


def test_table_join_and_order(data):
    assert len(data.table) == 2  # stars without products are dropped
    assert list(data.table.columns[:3]) == ["ticid", "ra", "dec"]
    assert "teff" in data.numeric
    assert "flag_possible_binary" in data.boolean


def test_query_filters_and_sort(data):
    out = data.query({"sort": ["teff"], "desc": ["1"], "limit": ["10"]})
    assert out["total"] == 2
    assert [r["ticid"] for r in out["rows"]] == [222, 111]
    out = data.query({"min_teff": ["4000"]})
    assert [r["ticid"] for r in out["rows"]] == [222]
    out = data.query({"flag_flag_possible_binary": ["1"]})
    assert [r["ticid"] for r in out["rows"]] == [111]
    out = data.query({"q": ["11"], "ids_only": ["1"]})
    assert out["ids"] == [111]
    out = data.query({"cols": ["teff,Tmag"], "offset": ["1"], "limit": ["1"]})
    assert out["columns"] == ["ticid", "teff", "Tmag"]
    assert len(out["rows"]) == 1
    out = data.query({"sort": ["not_a_column"]})
    assert out["sort"] == "ticid"


def test_star_payloads(data):
    row = data.star_row(111)
    assert row["ticid"] == 111 and row["parallax"] == 10.0
    assert row["sectors"] == "1,2"
    spec = data.spectrum_json(111)
    assert len(spec["wavelength_um"]) == 300
    assert len(spec["image_key"]) == 300
    assert all(isinstance(k, str) for k in spec["image_key"][:3])
    binned = data.binned_json(222)
    assert len(binned["lam"]) == len(binned["flux"]) == 76
    picks = data.default_cutout_keys(111)
    assert 1 <= len(picks) <= 6
    assert set(picks[0]) == {"image_key", "wavelength_um", "tjd"}


def test_sed_fit_endpoint_logic(data):
    res = data.sed_fit(111, {"grids": ["sphinx"], "logg": ["5.0"],
                             "binary": ["0"]})
    fit = res["fit"]
    assert fit["n_bins_used"] > 50
    assert fit["grid_full"] == "sphinx"
    assert "full" in res["curves"] and "blue" in res["curves"]
    assert "teff_bin1" not in fit
    assert res["options"]["logg"] == "5.0"
    res2 = data.sed_fit(222, {})
    assert res2["fit"]["teff_full"] > 0
    assert "binary" in res2["curves"]


def test_spectrum_store_roundtrip(viz_build):
    store = SpectrumStore(viz_build / "viz")
    df = store.read(222)
    assert int(df["ticid"].iloc[0]) == 222
    assert "tjd_avg" in df
    assert 222 in store and 333 not in store


def test_record_store(tmp_path):
    store = server.RecordStore(tmp_path / "verdicts.json")
    rec = server.parse_verdict({"verdict": "binary", "notes": "hi"})
    saved = store.save(5, rec)
    assert saved["verdict"] == "binary"
    assert store.load()["5"]["notes"] == "hi"
    store.save(5, server.parse_verdict({"verdict": "", "notes": ""}))
    assert store.load() == {}
    assert (tmp_path / "verdicts_history.csv").read_text().count("\n") == 3
    with pytest.raises(ValueError):
        server.parse_verdict({"verdict": "nope"})


def test_tars_vetting_unreachable(data, monkeypatch):
    monkeypatch.setattr(server, "LCVIZ_URL", "http://127.0.0.1:9")
    out = data.tars_vetting(111)
    assert out["images"] == []
    assert "error" in out
    assert out["star_url"].endswith("ticid=111")


def test_dss_cutout_recentres_on_proper_motion(data, tmp_path, monkeypatch):
    """A fake STScI DSS plate: the tile centre follows the proper motion."""
    import io

    import requests
    from astropy.io import fits
    from astropy.wcs import WCS
    from PIL import Image

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [10.0, -5.0]  # TIC 111 in the fixture
    w.wcs.crpix = [150.0, 150.0]
    w.wcs.cd = np.array([[-1.0, 0.0], [0.0, 1.0]]) / 3600  # 1"/px, N up
    img = np.full((300, 300), 10.0)
    # Star drawn at its 1990 position: it reaches the 2016 position
    # (0-based pixel 149, 149) after 26 years of +1000 mas/yr eastward,
    # so in 1990 it was 26" west = 26 px to the right (CD1_1 < 0).
    img[149, 149 + 26] = 1000.0
    hdr = w.to_header()
    hdr["DATE-OBS"] = "1990-01-01T00:00:00"
    buf = io.BytesIO()
    fits.PrimaryHDU(img, header=hdr).writeto(buf)

    class Resp:
        status_code = 200
        content = buf.getvalue()

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    data.table.loc[data.row_of[111], ["pmra", "pmdec"]] = [1000.0, 0.0]
    old_dir = data.viz_dir
    data.viz_dir = tmp_path
    try:
        jpeg = data.sky_cutout(111, "dss2r")
        meta = data.sky_cutout_meta(111, "dss2r")
    finally:
        data.viz_dir = old_dir
    assert jpeg is not None and jpeg.startswith(b"\xff\xd8")
    assert meta["source"] == "stsci-dss" and meta["epoch"] == 1990.0
    assert meta["shift_arcsec"] == pytest.approx(26.0, abs=0.1)
    assert meta["offset_east_arcsec"] == pytest.approx(26.0, abs=0.1)
    arr = np.asarray(Image.open(io.BytesIO(jpeg)))
    iy, ix = np.unravel_index(np.argmax(arr), arr.shape)
    assert abs(ix - 150) <= 1 and abs(iy - 150) <= 1
