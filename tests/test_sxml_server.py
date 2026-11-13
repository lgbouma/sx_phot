"""Tests for the sx-ml server's data layer on a synthetic build."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

from sx_phot import empfit, specml  # noqa: E402
from sx_phot.sedmodels import spherex_bin_edges  # noqa: E402
from test_empfit import _toy_library  # noqa: E402


def _load_server():
    # Both apps have a server.py; load this one under its own name so
    # that test_sxviz_server's import is not shadowed (and vice versa).
    import importlib.util
    path = ROOT / "apps" / "sx-ml" / "server.py"
    spec = importlib.util.spec_from_file_location("sxml_server", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sxml_server"] = mod
    spec.loader.exec_module(mod)
    return mod


server = _load_server()

N_STARS = 120


@pytest.fixture(scope="module")
def build(tmp_path_factory):
    """A run dir with binned spectra, a toy library, fits and an embedding."""
    run = tmp_path_factory.mktemp("run")
    viz = run / "viz"
    ml = run / "ml"
    viz.mkdir()
    ml.mkdir()
    edges = spherex_bin_edges()
    lam = np.sqrt(edges[:-1] * edges[1:])
    lib = _toy_library()
    em = empfit.EmpiricalModels(lib, n_grid=512)
    rng = np.random.default_rng(0)
    tics = np.arange(1000, 1000 + N_STARS, dtype=np.int64)
    t_true = rng.uniform(0.05, 0.95, N_STARS)
    plx = rng.uniform(5, 30, N_STARS)
    flux = em.abs_flux_at(t_true) * (plx / 100.0)[:, None] ** 2
    err = flux / 100.0
    flux = flux + rng.normal(0, 1, flux.shape) * err
    nper = np.full(flux.shape, 5, np.int16)
    nper[0, 50:] = 0
    np.save(viz / "binned_flux.npy", flux.astype(np.float32))
    np.save(viz / "binned_err.npy", err.astype(np.float32))
    np.save(viz / "binned_n.npy", nper)
    np.save(viz / "binned_ticid.npy", tics)
    np.save(viz / "bin_edges.npy", edges)
    pd.DataFrame({
        "ticid": tics, "med_snr": np.full(N_STARS, 100.0),
        "n_used": np.full(N_STARS, 300), "n_good": np.full(N_STARS, 300),
        "n_masked": np.zeros(N_STARS, int), "spline_ok": np.ones(N_STARS, bool),
        "resid_mad": np.full(N_STARS, 0.01), "n_epochs": np.full(N_STARS, 3),
        "ra": np.linspace(0, 10, N_STARS), "dec": np.zeros(N_STARS),
    }).to_parquet(viz / "summary.parquet", index=False)
    tars = pd.DataFrame({
        "TICID": tics, "dr3_source_id": tics * 7, "ra": np.zeros(N_STARS),
        "dec": np.zeros(N_STARS), "parallax": plx,
        "teff": 7000 - 4000 * t_true, "BpmRp0": 0.5 + 2.5 * t_true,
        "phot_g_mean_mag_0": 3 + 8 * t_true, "phot_g_mean_mag": 12.0,
        "extinction_a0": np.zeros(N_STARS), "ruwe": np.ones(N_STARS),
        "non_single_star": np.zeros(N_STARS, np.int16), "Tmag": 11.0,
        "adopted_period": np.full(N_STARS, 2.0),
        "median_amplitude": np.full(N_STARS, 0.01),
        "final_n_contams": np.zeros(N_STARS, np.float32),
        "flag_possible_binary": np.zeros(N_STARS, bool),
        "flag_multiple_periods": np.zeros(N_STARS, bool),
        "n_secs": np.ones(N_STARS, int),
    })
    tars_path = run / "tars.parquet"
    tars.to_parquet(tars_path, index=False)
    lib.save(ml / "sequence")
    fits = empfit.fit_chunk(flux, err, nper, plx, None, em)
    fits.insert(0, "row", np.arange(N_STARS))
    fits.insert(0, "ticid", tics)
    fits = empfit.add_population_columns(fits, n_bins=4)
    fits.to_parquet(ml / "empfit.parquet", index=False)
    pd.DataFrame({"ticid": tics[:20], "row": np.arange(20), "t": t_true[:20],
                  "chi2": np.ones(20), "inlier": np.ones(20, bool)}
                 ).to_parquet(ml / "sequence_train.parquet", index=False)
    emb = np.column_stack([t_true, rng.normal(0, 0.1, N_STARS)]).astype(
        np.float32)
    emb[1] = np.nan
    np.save(ml / "umap_shape.npy", emb)
    (ml / "umap.json").write_text(json.dumps({"settings": {"n_pc": 5}}))
    (run / "config.json").write_text("{}")
    return run, tars_path, t_true, tics


@pytest.fixture(scope="module")
def data(build):
    run, tars_path, _, _ = build
    return server.Data(run, None, None, tars_path, sxviz_url="http://x:1")


def test_meta_and_columns(data):
    m = data.meta()
    assert m["n"] == N_STARS
    assert m["name"] == "default" and m["label"] == "default"
    names = {c["name"] for c in m["columns"]}
    assert {"teff_emp", "t_single", "bprp0", "dms", "chi2_red",
            "sequence_train", "dbic_binary_pct"} <= names
    assert m["embeddings"][0]["name"] == "shape"
    assert m["embeddings"][0]["n"] == N_STARS - 1
    assert m["umap"]["n_pc"] == 5
    col = data.col("teff_emp")
    assert col.dtype == np.float32 and len(col) == N_STARS
    with pytest.raises(KeyError):
        data.col("nope")
    # Identifiers keep their precision; served floats are float32.
    assert data.table["ticid"].dtype == np.int64
    assert data.table["bprp0"].dtype == np.float32


def test_main_sequence_curve(build, data):
    _, _, t_true, _ = build
    ms = data.meta()["mainseq"]
    assert len(ms["x"]) == len(ms["y"]) > 10
    # The synthetic sample is a straight line mg0 = 3 + 8 t, bprp0 = 0.5
    # + 2.5 t: the running median follows it and the offsets are ~0.
    x = np.array(ms["x"])
    y = np.array(ms["y"])
    assert np.allclose(y, 3 + 8 * (x - 0.5) / 2.5, atol=0.3)
    dms = data.col("dms")
    assert np.nanmedian(np.abs(dms)) < 0.2


def test_star_payload(build, data):
    _, _, t_true, tics = build
    s = data.star(3)
    assert s["ticid"] == int(tics[3])
    assert len(s["lam"]) == 76 and len(s["flux_dered"]) == 76
    assert "single" in s["curves"] and "binary" in s["curves"]
    assert abs(s["info"]["t_single"] - t_true[3]) < 0.02
    assert s["links"]["sxviz"].endswith(f"ticid={int(tics[3])}")
    assert s["links"]["tarviz"].endswith(f"ticid={int(tics[3])}")
    assert s["embedding"]["shape"][0] is not None
    assert data.star(1)["embedding"]["shape"][0] is None
    # Missing bins come back as None.
    s0 = data.star(0)
    assert s0["flux"][60] is None and s0["n"][60] == 0
    assert data.row_of[int(tics[5])] == 5


def test_selection(data):
    out = data.selection(np.array([2, 3, 4, 4, 9999]))
    assert out["n"] == 3
    assert len(out["resid_med"]) == 76
    assert abs(np.nanmedian([v for v in out["resid_med"] if v is not None])
               ) < 0.02
    assert "teff_emp" in out["columns"]
    assert out["ticids"] == [1002, 1003, 1004]
    assert data.selection(np.array([]))["n"] == 0


def test_sequence_endpoints(data):
    seq = data.sequence()
    assert len(seq["nodes"]) == 48 and len(seq["templates"][0]) == 76
    assert len(seq["mass_ms"]) == 48
    assert seq["summary"]["teff_tars_med"][0] > seq["summary"]["teff_tars_med"][-1]
    at = data.sequence_at(0.5, n_examples=5)
    assert abs(at["teff"] - 5000) < 100
    assert len(at["template"]) == 76
    assert len(at["resid_comps"]) == 2
    assert at["n_near"] >= 0
    at_all = data.sequence_at(0.5, n_examples=5, source="all")
    assert isinstance(at_all["examples"], list)
    for ex in at_all["examples"]:
        assert len(ex["y"]) == 76


def test_datasets(build, data):
    """Two datasets from one build: routing by name, sx-viz link gating."""
    run, tars_path, _, tics = build
    other = server.Data(run, None, None, tars_path, sxviz_url="http://x:1",
                        name="other", label="the other one",
                        sxviz_rows={int(tics[3])})
    sets = server.Datasets([data, other])
    assert sets.get(None) is data and sets.get("other") is other
    with pytest.raises(KeyError):
        sets.get("nope")
    assert [d["name"] for d in sets.summary()] == ["default", "other"]
    assert sets.summary()[1]["label"] == "the other one"
    # Links only for the stars sx-viz has.
    assert "sxviz" in other.star(3)["links"]
    assert "tarviz" in other.star(3)["links"]
    assert other.star(4)["links"] == {}
    assert "sxviz" in data.star(4)["links"]


def test_load_datasets_skips_missing(build, tmp_path):
    run, tars_path, _, _ = build
    cfg = {"sxviz_dataset": "a", "datasets": [
        {"name": "a", "label": "A", "run_dir": str(run),
         "tars": str(tars_path)},
        {"name": "b", "label": "B", "run_dir": str(tmp_path / "nope"),
         "tars": str(tars_path)},
        {"name": "c", "label": "C", "run_dir": str(run),
         "tars": str(tars_path)},
    ]}
    sets = server.load_datasets(cfg, "http://x:1")
    assert list(sets.items) == ["a", "c"]
    assert sets.default.name == "a"
    assert sets.pending() == ["b"]
    assert [d["ready"] for d in sets.summary()] == [True, False, True]
    assert sets.summary()[1]["label"] == "B"
    assert sets.items["a"].sxviz_rows is None
    assert sets.items["c"].sxviz_rows is sets.items["a"].row_of
    with pytest.raises(ValueError):
        server.load_datasets({"datasets": []})
    # Nothing changed: refresh loads nothing.  A rebuilt product set is
    # reloaded once it has been quiet long enough.
    assert sets.refresh(quiet=0.0) == []
    stamp = server.products_stamp(run / "viz", run / "ml")
    assert stamp is not None
    import os
    os.utime(run / "ml" / "empfit.parquet", None)
    assert sets.refresh(quiet=3600.0) == []
    assert sets.refresh(quiet=0.0) == ["a", "c"]
    assert sets.items["c"].sxviz_rows is sets.items["a"].row_of
    # A pending entry becomes available.
    cfg2 = {"datasets": [{"name": "late", "label": "L",
                          "run_dir": str(tmp_path / "late"),
                          "tars": str(tars_path)}]}
    late = server.load_datasets(cfg2, "http://x:1")
    assert late.pending() == ["late"]
    with pytest.raises(KeyError):
        late.default
    (tmp_path / "late").symlink_to(run)
    assert late.refresh(quiet=0.0) == ["late"]
    assert late.default.name == "late"


def test_http_roundtrip(build, data):
    """Exercise the handler through a real socket."""
    import threading
    import urllib.request
    from functools import partial
    from http.server import ThreadingHTTPServer

    run, tars_path, _, _ = build
    other = server.Data(run, None, None, tars_path, sxviz_url="http://x:1",
                        name="other", label="B")
    sets = server.Datasets([data, other])
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(server.Handler,
                                                        data=sets))
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        base = f"http://127.0.0.1:{port}"
        meta = json.loads(urllib.request.urlopen(base + "/api/meta").read())
        assert meta["n"] == N_STARS
        assert meta["name"] == "default" and meta["default"] == "default"
        assert [d["name"] for d in meta["datasets"]] == ["default", "other"]
        assert len(meta["mainseq"]["x"]) > 10
        m2 = json.loads(urllib.request.urlopen(
            base + "/api/meta?ds=other").read())
        assert m2["name"] == "other" and m2["label"] == "B"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/api/meta?ds=nope")
        s2 = json.loads(urllib.request.urlopen(
            base + "/api/star/1005?by=ticid&ds=other").read())
        assert s2["row"] == 5
        req2 = urllib.request.Request(
            base + "/api/selection?ds=other",
            data=json.dumps({"rows": [1, 2, 3]}).encode(),
            headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req2).read())["n"] == 3
        pts = urllib.request.urlopen(base + "/api/points?emb=shape").read()
        arr = np.frombuffer(pts, dtype="<f4")
        assert arr.shape == (2 * N_STARS,)
        tic = np.frombuffer(urllib.request.urlopen(base + "/api/ticids").read(),
                            dtype="<u4")
        assert tic[0] == 1000
        col = np.frombuffer(urllib.request.urlopen(
            base + "/api/col/teff_emp").read(), dtype="<f4")
        assert len(col) == N_STARS
        star = json.loads(urllib.request.urlopen(
            base + "/api/star/1005?by=ticid").read())
        assert star["row"] == 5
        req = urllib.request.Request(
            base + "/api/selection", data=json.dumps({"rows": [1, 2]}).encode(),
            headers={"Content-Type": "application/json"})
        sel = json.loads(urllib.request.urlopen(req).read())
        assert sel["n"] == 2
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/api/col/nope")
        page = urllib.request.urlopen(base + "/").read()
        assert b"sx-ml" in page
    finally:
        srv.shutdown()
        srv.server_close()
