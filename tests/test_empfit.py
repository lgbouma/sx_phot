"""Tests for the empirical template fits (sx_phot.empfit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sx_phot import empfit, specml
from sx_phot.sedmodels import spherex_bin_edges

EDGES = spherex_bin_edges()
LAM = np.sqrt(EDGES[:-1] * EDGES[1:])


def _bands(t: float) -> np.ndarray:
    """Absorption bands (dex) that deepen along the sequence."""
    # Bands switch on beyond t = 0.55 so that a hot star plus a cool
    # companion (blue slope with bands) is not a single template.
    depth = 0.35 * np.clip((t - 0.55) / 0.45, 0, 1) ** 2
    g = (np.exp(-((LAM - 1.4) / 0.08) ** 2) + np.exp(-((LAM - 1.9) / 0.1) ** 2)
         + 0.6 * np.exp(-((LAM - 2.7) / 0.15) ** 2)
         + 0.4 * np.exp(-((LAM - 4.6) / 0.2) ** 2))
    return -depth * g


def _toy_library(n_nodes: int = 48) -> specml.SpectralSequence:
    """Blackbodies with water-like bands, 7000 K (t=0) to 3000 K (t=1)."""
    nodes = specml.node_positions(n_nodes)
    teff = 7000 - 4000 * nodes
    templates = []
    abs_off = []
    for t_node, te in zip(nodes, teff):
        bb = empfit.planck_jy_sr(LAM, te)
        logbb = np.log10(bb) + _bands(t_node)
        templates.append(logbb - logbb.mean())
        # Dwarf radius shrinks with Teff: (R/d)^2 at 10 pc.
        r_rsun = 0.3 + 0.7 * (te - 3000) / 4000
        omega = np.pi * (r_rsun * empfit.RSUN_M / (10 * empfit.PC_M)) ** 2
        abs_off.append(logbb.mean() + np.log10(omega))
    templates = np.array(templates, np.float32)
    arrays = {
        "nodes": nodes, "lam_um": LAM, "templates": templates,
        "scatter": np.full(templates.shape, 0.002, np.float32),
        "scatter_int": np.full(templates.shape, 0.002, np.float32),
        "abs_offset": np.array(abs_off), "abs_scatter": np.full(n_nodes, 0.05),
        "resid_comps": np.zeros((n_nodes, 3, len(LAM)), np.float32),
        "resid_amp": np.zeros((n_nodes, 3), np.float32),
        "arc": specml.arc_length(templates),
    }
    arrays["resid_comps"][:, 0, -1] = 1.0
    summary = pd.DataFrame({"node": np.arange(n_nodes), "t": nodes,
                            "n": 100, "teff_tars_med": teff,
                            "mg0_med": 3 + 8 * nodes, "counts": 100})
    return specml.SpectralSequence(arrays, summary, {"toy": True})


@pytest.fixture(scope="module")
def models():
    return empfit.EmpiricalModels(_toy_library(), n_grid=512)


def _observe(em, t, plx_mas, scale_mult=1.0, snr=200.0, seed=0,
             extra=None):
    """Flux of a main-sequence star at ``t`` seen at ``plx_mas``."""
    rng = np.random.default_rng(seed)
    f = em.abs_flux_at(np.array([t]))[0] * (plx_mas / 100.0) ** 2 * scale_mult
    if extra is not None:
        f = f + extra
    err = f / snr
    return f + rng.normal(0, 1, len(f)) * err, err


def test_single_star_recovered(models):
    em = models
    f, e = _observe(em, 0.4, 20.0)
    n = np.full((1, len(f)), 5)
    df = empfit.fit_chunk(f[None], e[None], n, np.array([20.0]), None, em,
                          do_binary=False, do_disk=False)
    r = df.iloc[0]
    assert abs(r["t_single"] - 0.4) < 0.01
    assert abs(r["teff_emp"] - 5400) < 60
    assert abs(r["lum_ratio"] - 1.0) < 0.05
    assert abs(r["dmag_ms"]) < 0.06
    assert r["chi2_single"] / r["dof_single"] < 2.0
    assert abs(r["red_excess"]) < 0.01
    assert abs(r["t_ms"] - 0.4) < 0.02


def test_missing_bins_and_dereddening(models):
    em = models
    f, e = _observe(em, 0.6, 10.0)
    n = np.full((1, len(f)), 5)
    n[0, 60:] = 0                      # no data beyond ~3.5 um
    a0 = np.array([0.5])
    fac = specml.deredden_factor(LAM, a0)[0]
    f_red = f / fac                    # redden the observation
    df = empfit.fit_chunk(f_red[None], e[None], n, np.array([10.0]), a0, em,
                          do_binary=False, do_disk=False)
    r = df.iloc[0]
    assert r["n_bins_used"] == 60
    assert abs(r["t_single"] - 0.6) < 0.01
    assert np.isnan(r["excess_b6"])
    # Too few bins -> NaN.
    n[0, 10:] = 0
    df = empfit.fit_chunk(f[None], e[None], n, np.array([10.0]), None, em,
                          do_binary=False, do_disk=False)
    assert np.isnan(df.iloc[0]["t_single"])


def test_binary_recovered(models):
    em = models
    plx = 15.0
    f1 = em.abs_flux_at(np.array([0.3]))[0] * (plx / 100) ** 2
    f2 = em.abs_flux_at(np.array([0.8]))[0] * (plx / 100) ** 2
    rng = np.random.default_rng(3)
    f = f1 + f2
    e = f / 300.0
    f = f + rng.normal(0, 1, len(f)) * e
    n = np.full((1, len(f)), 5)
    df = empfit.fit_chunk(f[None], e[None], n, np.array([plx]), None, em,
                          do_disk=False)
    r = df.iloc[0]
    assert r["dchi2_binary"] > 30
    assert r["dbic_binary"] > 15
    assert abs(r["t_bin1"] - 0.3) < 0.03
    assert abs(r["t_bin2"] - 0.8) < 0.08
    assert abs(r["t_bin1_ms"] - 0.3) < 0.03
    assert r["dchi2_binary_ms"] > r["dchi2_binary"]
    assert r["lum_ratio"] > 1.05       # overluminous for its colour
    # A single star shows no binary preference.
    fs, es = _observe(em, 0.3, plx, seed=4, snr=300.0)
    df1 = empfit.fit_chunk(fs[None], es[None], n, np.array([plx]), None, em,
                           do_disk=False)
    assert df1.iloc[0]["dbic_binary"] < 10


def test_disk_recovered(models):
    em = models
    plx = 25.0
    star = em.abs_flux_at(np.array([0.5]))[0] * (plx / 100) ** 2
    bb = empfit.planck_jy_sr(LAM, 600.0)
    omega = 0.5 * star[-1] / bb[-1]    # 50 % excess at 5 um
    rng = np.random.default_rng(5)
    f = star + omega * bb
    e = f / 200.0
    f = f + rng.normal(0, 1, len(f)) * e
    n = np.full((1, len(f)), 5)
    df = empfit.fit_chunk(f[None], e[None], n, np.array([plx]), None, em,
                          do_binary=True, do_disk=True)
    r = df.iloc[0]
    assert r["dchi2_disk"] > 100
    assert 400 < r["teff_disk"] < 900
    assert abs(r["t_disk"] - 0.5) < 0.03
    assert 0.3 < r["disk_frac_b6"] < 0.7
    assert r["dbic_disk_vs_binary"] > 0
    assert r["red_excess"] > 0.05
    assert r["excess_b6_sig"] > 5
    curves = empfit.model_curves(em, r.to_dict())
    assert len(curves["disk"]) == len(LAM)
    assert curves["disk"][-1] > curves["disk_star"][-1]


def test_population_columns():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"t_single": rng.uniform(0, 1, 500),
                       "dbic_binary": rng.normal(0, 1, 500),
                       "red_excess": rng.normal(0, 1, 500)})
    df.loc[0, "dbic_binary"] = np.nan
    out = empfit.add_population_columns(df, n_bins=5,
                                        columns=("dbic_binary",
                                                 "red_excess", "missing"))
    assert "dbic_binary_pct" in out and "missing_pct" not in out
    assert np.isnan(out.loc[0, "dbic_binary_pct"])
    v = out["dbic_binary_pct"].dropna()
    assert 0 <= v.min() and v.max() <= 1
    assert abs(v.mean() - 0.5) < 0.05
