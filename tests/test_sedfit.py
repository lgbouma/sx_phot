"""Tests for sx_phot.sedmodels and sx_phot.sedfit on synthetic grids."""

from __future__ import annotations

import numpy as np
import pytest

from sx_phot.sedfit import (MainSequence, ModelSet, add_population_columns,
                            chi2_pairs, chi2_single, fit_star, radius_rsun)
from sx_phot.sedmodels import (ModelGrid, bin_average, flam_si_to_jy,
                               spherex_bin_edges)

H = 6.62607015e-34
C = 299792458.0
K = 1.380649e-23


def _planck_fnu_jy(lam_um: np.ndarray, teff: float) -> np.ndarray:
    """Emergent blackbody F_nu (pi B_nu) in Jy."""
    nu = C / (lam_um * 1e-6)
    b = 2 * H * nu ** 3 / C ** 2 / np.expm1(H * nu / (K * teff))
    return np.pi * b / 1e-26


def _blackbody_set(teffs, edges) -> ModelSet:
    centers = np.sqrt(edges[:-1] * edges[1:])
    flux = np.array([_planck_fnu_jy(centers, t) for t in teffs])
    return ModelSet(teff=np.array(teffs, float),
                    grid=np.array(["bb"] * len(teffs)),
                    logg=np.full(len(teffs), 4.5), mh=np.zeros(len(teffs)),
                    flux=flux, bin_edges=edges)


def test_bin_edges_resolution():
    edges = spherex_bin_edges()
    assert edges[0] == pytest.approx(0.75)
    assert edges[-1] >= 5.0
    ratio = edges[1:] / edges[:-1]
    assert np.allclose(ratio, np.exp(1 / 40.0))


def test_flam_to_jy_roundtrip():
    lam = np.array([1.0, 2.0])
    fnu = flam_si_to_jy(lam, np.array([1.0, 1.0]))
    # F_nu = F_lam * lam^2 / c; lam = 1e-6 m -> 1e-12 / 3e8 / 1e-26
    assert fnu[0] == pytest.approx(1e-12 / C / 1e-26)
    assert fnu[1] == pytest.approx(4 * fnu[0])


def test_bin_average_constant():
    lam = np.linspace(0.5, 6.0, 5000)
    edges = spherex_bin_edges()
    out = bin_average(lam, np.full_like(lam, 3.0), edges)
    assert np.allclose(out, 3.0)


def test_chi2_single_recovers_scale():
    edges = spherex_bin_edges()
    models = _blackbody_set([3000, 4000, 5000], edges)
    truth = models.flux[1] * 2.5e-20
    err = 0.01 * truth
    chi2, scale = chi2_single(truth, err, models.flux,
                              np.ones(truth.size, bool))
    assert np.argmin(chi2) == 1
    assert scale[1] == pytest.approx(2.5e-20)
    assert chi2[1] == pytest.approx(0.0, abs=1e-6)


def test_chi2_pairs_recovers_binary():
    edges = spherex_bin_edges()
    models = _blackbody_set([3000, 3500, 4000, 5000, 6000], edges)
    truth = models.flux[4] * 1e-20 + models.flux[0] * 3e-20
    err = 0.002 * truth
    use = np.ones(truth.size, bool)
    chi2, s1, s2 = chi2_pairs(truth, err, models.flux, use)
    i, j = np.unravel_index(np.argmin(chi2), chi2.shape)
    assert (i, j) in ((4, 0), (0, 4))
    assert chi2[i, j] == pytest.approx(0.0, abs=1e-6)
    scales = sorted([s1[i, j], s2[i, j]])
    assert scales[0] == pytest.approx(1e-20, rel=1e-6)
    assert scales[1] == pytest.approx(3e-20, rel=1e-6)
    # Subset form gives the same answer.
    chi2s, s1s, s2s = chi2_pairs(truth, err, models.flux, use,
                                 primary=np.array([4]),
                                 secondary=np.array([0]))
    assert chi2s.shape == (1, 1)
    assert chi2s[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_fit_star_red_excess_for_binary():
    edges = spherex_bin_edges()
    teffs = list(range(2500, 6501, 250))
    models = _blackbody_set(teffs, edges)
    ms = MainSequence()
    single = models.flux[teffs.index(5000)] * 1e-20
    err = 0.01 * single
    res = fit_star(single, err, models, parallax_mas=10.0, main_sequence=ms)
    assert res["teff_full"] == 5000
    assert abs(res["red_excess"]) < 0.02
    # Add a cool companion: red excess becomes positive and the binary
    # fit improves on the single fit.
    binary = single + models.flux[teffs.index(3000)] * 2e-20
    res2 = fit_star(binary, 0.01 * binary, models, parallax_mas=10.0,
                    main_sequence=ms)
    assert res2["red_excess"] > 0.05
    assert res2["dchi2_binary"] > 10
    assert res2["teff_bin2"] <= 3500
    assert res2["n_bins_used"] == edges.size - 1


def test_radius_from_scale():
    # R = 1 Rsun at 10 pc: scale = (Rsun / 10 pc)^2
    scale = (6.957e8 / (10 * 3.0856775814913674e16)) ** 2
    assert radius_rsun(scale, 100.0) == pytest.approx(1.0)


def test_main_sequence_monotonic():
    ms = MainSequence()
    r = ms.radius_at([3000, 4000, 5800, 7000])
    assert np.all(np.diff(r) > 0)
    assert 0.9 < ms.radius_at(5772) < 1.1
    assert 4.3 < ms.logg_at(5772) < 4.6


def test_model_grid_teff_sequence_interpolates():
    edges = spherex_bin_edges()
    centers = np.sqrt(edges[:-1] * edges[1:])
    fine = np.linspace(0.7, 5.1, 50)
    teff = np.array([3000.0, 3100.0, 3000.0])
    logg = np.array([5.0, 5.0, 4.5])
    flux = np.array([_planck_fnu_jy(centers, t) for t in teff])
    ffine = np.array([_planck_fnu_jy(fine, t) for t in teff])
    g = ModelGrid(name="t", teff=teff, logg=logg, mh=np.zeros(3),
                  bin_edges=edges, bin_flux_jy=flux, fine_lam_um=fine,
                  fine_flux_jy=ffine)
    t, f = g.teff_sequence(5.0, 0.0, teff_step=50.0)
    assert list(t) == [3000.0, 3050.0, 3100.0]
    assert np.all(f[1] > f[0]) and np.all(f[1] < f[2])
    with pytest.raises(ValueError):
        g.teff_sequence(4.5, 0.0)


def test_add_population_columns_removes_teff_trend():
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 4000
    teff = rng.uniform(3000, 6000, n)
    offset = -0.1 + (teff - 3000) / 30000.0  # linear trend in Teff
    excess = offset + rng.normal(0, 0.02, n)
    excess[:5] += 0.5  # outliers
    df = pd.DataFrame({
        "teff_blue": teff, "teff_full": teff, "red_excess": excess,
        "red_excess_err": np.full(n, 0.005), "radius_ratio_full": 1.0 + offset,
        "n_used": np.full(n, 300),
    })
    df.loc[10, "n_used"] = 10  # excluded from the statistics, still scored
    add_population_columns(df)
    rel = df["red_excess_rel"].to_numpy()
    assert abs(np.median(rel[5:])) < 0.01
    assert np.std(rel[5:]) < 0.03
    assert (df["red_excess_rel_sig"].to_numpy()[:5] > 10).all()
    assert np.isfinite(rel[10])
    assert abs(np.median(df["radius_ratio_full_rel"]) - 1) < 0.02
    # Percentile ranks: a Teff-dependent offset in dbic is removed.
    df["dbic_binary"] = 100 * (teff - 3000) / 3000 + rng.normal(0, 5, n)
    add_population_columns(df)
    pct = df["dbic_binary_pct"].to_numpy()
    assert np.all((pct >= 0) & (pct <= 1))
    hi = pct > 0.9
    assert 0.08 < hi.mean() < 0.12
    # ... so the selected fraction is the same for cool and hot stars.
    cool = teff < 4000
    assert abs(hi[cool].mean() - hi[~cool].mean()) < 0.03


def test_fit_star_disk_versus_binary():
    from sx_phot.sedfit import planck_fnu_jy

    edges = spherex_bin_edges()
    centers = np.sqrt(edges[:-1] * edges[1:])
    teffs = list(range(3000, 6501, 250))
    models = _blackbody_set(teffs, edges)
    ms = MainSequence()
    star = models.flux[teffs.index(5000)] * 1e-20
    # Hot dust: a 900 K blackbody with ten times the stellar emitting
    # area, i.e. a 35 percent excess at 5 um and 8 percent at 3 um.
    dust = planck_fnu_jy(centers, 900.0) * 1e-19
    obs = star + dust
    res = fit_star(obs, 0.01 * obs, models, parallax_mas=10.0,
                   main_sequence=ms)
    assert res["teff_disk_star"] == 5000
    assert 700 < res["teff_disk"] < 1200
    assert res["dbic_disk"] > res["dbic_binary"]
    assert res["dbic_disk_vs_binary"] > 0
    assert res["f_disk"] > 0
    assert res["bic_single"] > res["bic_disk"]
    # A pure star prefers the single model (BIC penalises extra params).
    res2 = fit_star(star, 0.01 * star, models, parallax_mas=10.0,
                    main_sequence=ms)
    assert res2["dbic_disk"] < 2 * np.log(res2["n_bins_used"])
    assert res2["dbic_binary"] < 2 * np.log(res2["n_bins_used"])


def test_blackbody_set_normalisation():
    from sx_phot.sedfit import blackbody_set, planck_fnu_jy

    lam = np.exp(np.linspace(np.log(0.05), np.log(500), 20000))
    f = planck_fnu_jy(lam, 1000.0)
    # integral of F_nu d nu = sigma T^4 (in W/m^2 after the Jy factor)
    nu = 299792458.0 / (lam * 1e-6)
    integral = -np.trapezoid(f * 1e-26, nu)
    assert integral == pytest.approx(5.670374e-8 * 1000.0 ** 4, rel=1e-3)
    temps, flux = blackbody_set(spherex_bin_edges(), np.array([500.0, 1500.0]))
    assert flux.shape == (2, spherex_bin_edges().size - 1)
    assert np.all(flux[1] > flux[0])
