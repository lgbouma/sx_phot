"""Tests for the empirical spectral sequence (sx_phot.specml)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sx_phot import specml

LAM = np.sqrt(specml.np.array([0.75 * np.exp(i / 40.0)
                               for i in range(77)])[:-1]
              * np.array([0.75 * np.exp(i / 40.0) for i in range(77)])[1:])


def _toy_sequence(n_bins: int = 76) -> tuple[np.ndarray, np.ndarray]:
    """A smooth one-parameter family: log flux = a(t) * x + b(t) * x^2."""
    x = np.linspace(-1, 1, n_bins)

    def templ(t):
        t = np.asarray(t, dtype=float)[:, None]
        y = (0.8 - 1.6 * t) * x[None] + 0.4 * np.sin(2.5 * t) * x[None] ** 2
        return y - y.mean(axis=1, keepdims=True)

    return x, templ


def test_extinction_curve_monotone_and_sane():
    lam = np.array([0.75, 1.0, 1.6, 2.2, 3.5, 4.8])
    k = specml.extinction_curve(lam)
    assert np.all(np.diff(k) < 0)
    assert 0.5 < k[0] < 0.8          # ~0.6 at 0.75 um
    assert 0.08 < k[3] < 0.14        # ~0.11 at K band
    fac = specml.deredden_factor(lam, np.array([0.0, 1.0, np.nan]))
    assert np.allclose(fac[0], 1.0)
    assert np.allclose(fac[2], 1.0)
    assert np.all(fac[1] > 1.0)


def test_prepare_and_normalize_rows():
    flux = np.array([[1.0, 2.0, 4.0], [1.0, np.nan, 4.0]], np.float32)
    err = np.full_like(flux, 0.1)
    nper = np.array([[3, 3, 3], [3, 0, 3]])
    lam = np.array([1.0, 2.0, 3.0])
    logf, sig, valid = specml.prepare_log_flux(flux, err, nper, lam,
                                               err_floor=0.0)
    assert valid.tolist() == [[True, True, True], [True, False, True]]
    assert np.isnan(logf[1, 1]) and sig[1, 1] == 0
    assert np.isclose(sig[0, 0], 0.1 / 1.0 / np.log(10))
    y, off = specml.normalize_rows(logf, valid)
    assert np.allclose(np.nanmean(y, axis=1), 0, atol=1e-6)
    assert np.isclose(off[0], np.mean(np.log10([1, 2, 4])))


def test_rank_uniform_and_interpolate_nodes():
    r = specml.rank_uniform(np.array([3.0, 1.0, 2.0]))
    assert np.allclose(r, [5 / 6, 1 / 6, 3 / 6])
    nodes = specml.node_positions(4)
    vals = np.arange(4, dtype=float)[:, None] * np.ones((1, 3))
    out = specml.interpolate_nodes(nodes, vals, np.array([nodes[0],
                                                          0.5, 2.0]))
    assert np.allclose(out[0], 0)
    assert np.allclose(out[1], 1.5)
    assert np.allclose(out[2], 3)      # clamped to the last node


def test_build_sequence_recovers_ordering():
    rng = np.random.default_rng(0)
    n, nb = 4000, 76
    x, templ = _toy_sequence(nb)
    t_true = rng.uniform(0, 1, n)
    y = templ(t_true) + rng.normal(0, 0.005, (n, nb))
    sig = np.full((n, nb), 0.005)
    # Initial ordering: a noisy, nonlinear proxy of t.
    init = t_true ** 2 + rng.normal(0, 0.1, n)
    fit = specml.build_sequence(y, sig, init, n_nodes=64, n_iter=4,
                                n_grid=512, verbose=False)
    rho = np.corrcoef(fit.t, t_true)[0, 1]
    assert rho > 0.99
    assert fit.templates.shape == (64, nb)
    assert fit.inlier.sum() > 0.95 * n
    assert fit.nodes[0] == 0 and fit.nodes[-1] == 1
    # Templates at the nodes match the truth to a few noise sigma (the
    # node windows average over a range of t).
    truth = templ(fit.nodes)
    assert np.nanmedian(np.abs(fit.templates - truth)) < 0.02
    # Outliers are clipped.
    y2 = y.copy()
    y2[:50] += rng.normal(0, 0.2, (50, nb))
    fit2 = specml.build_sequence(y2, sig, init, n_nodes=64, n_iter=3,
                                 n_grid=512, verbose=False)
    assert (~fit2.inlier[:50]).mean() > 0.9


def test_assign_log_handles_missing_bins():
    x, templ = _toy_sequence()
    nodes = specml.node_positions(16)
    tg, grid = specml.fine_grid(nodes, templ(nodes), 256)
    assert tg[0] == 0 and tg[-1] == 1
    y = templ(np.array([0.3, 0.7]))
    w = np.full(y.shape, 1e4)
    w[0, :30] = 0
    y[0, :30] = 0
    t, chi2, off = specml.assign_log(y, w, tg, grid)
    assert np.allclose(t, [0.3, 0.7], atol=0.01)
    # Linear interpolation between 16 nodes leaves ~1e-3 dex errors.
    assert np.all(chi2 < 5)


def test_library_roundtrip_and_mappings(tmp_path):
    rng = np.random.default_rng(1)
    n, nb = 3000, 76
    x, templ = _toy_sequence(nb)
    t_true = rng.uniform(0, 1, n)
    y = templ(t_true) + rng.normal(0, 0.005, (n, nb))
    sig = np.full((n, nb), 0.005)
    fit = specml.build_sequence(y, sig, t_true, n_nodes=24, n_iter=2,
                                n_grid=256, verbose=False)
    assert len(fit.node_lo) == 24
    cat = pd.DataFrame({
        "teff_tars": 7000 - 4000 * t_true + rng.normal(0, 50, n),
        "bprp0": 0.5 + 2.5 * t_true, "mg0": 3 + 8 * t_true,
        "a0": np.full(n, 0.05), "tmag": np.full(n, 12.0),
        "period": np.full(n, 3.0), "amplitude": np.full(n, 0.01),
        "plx": np.full(n, 10.0)})
    offset = 2.0 - 2 * t_true  # mean log flux
    lib = specml.assemble_library(fit, y, sig, offset, cat, LAM,
                                  meta={"note": "toy"})
    lib.save(tmp_path / "seq")
    lib2 = specml.SpectralSequence.load(tmp_path / "seq")
    assert np.allclose(lib2.templates, lib.templates)
    assert lib2.meta["note"] == "toy"
    # Teff mapping is monotone decreasing and invertible.
    assert np.all(np.diff(lib2.teff_mono) <= 0)
    t_back = lib2.t_from_teff(lib2.teff(np.array([0.2, 0.6])))
    assert np.allclose(t_back, [0.2, 0.6], atol=0.03)
    # Arc length is monotone from 0 to 1.
    assert lib2.arc[0] == 0 and np.isclose(lib2.arc[-1], 1)
    assert np.all(np.diff(lib2.arc) >= 0)
    # Absolute template reproduces the input offsets at 10 pc.
    # offset(0.5) = 1.0 and plx = 10 mas: + 4 - 2 log10(10) = 3.0.
    a = lib2.abs_template(np.array([0.5]))[0].mean()
    assert np.isclose(a, 3.0, atol=0.1)
    # Residual components are unit vectors.
    assert np.allclose(np.linalg.norm(lib2.resid_comps[5], axis=1), 1,
                       atol=1e-5)


def test_select_clean_and_ms_offset():
    rng = np.random.default_rng(2)
    n = 2000
    bprp = rng.uniform(0.5, 3.0, n)
    mg = 2 + 3 * bprp + rng.normal(0, 0.1, n)
    mg[:20] -= 2.0  # overluminous
    off = specml.main_sequence_offset(bprp, mg)
    assert np.nanmedian(np.abs(off[20:])) < 0.2
    assert np.all(off[:20] < -1.0)
    cat = pd.DataFrame({
        "med_snr": np.full(n, 100.0), "spline_ok": np.ones(n, bool),
        "n_used": np.full(n, 300), "a0": np.full(n, 0.05),
        "ruwe": np.ones(n), "nss": np.zeros(n), "plx": np.full(n, 5.0),
        "n_contams": np.zeros(n), "bprp0": bprp, "mg0": mg})
    cat.loc[100, "ruwe"] = 2.0
    cat.loc[101, "n_contams"] = 3
    valid = np.ones((n, 76), bool)
    valid[102, 5] = False
    ok = specml.select_clean(cat, valid)
    assert not ok[:20].any()
    assert not ok[100] and not ok[101] and not ok[102]
    assert ok[200:].mean() > 0.9


def test_monotone_decreasing():
    v = np.array([5.0, 4.0, 4.5, 3.0, 3.2, 1.0])
    m = specml.monotone_decreasing(v)
    assert np.all(np.diff(m) <= 0)
    assert np.isclose(m.sum(), v.sum(), atol=1e-9)


def test_node_windows():
    nodes, lo, hi = specml.uniform_windows(8)
    assert np.allclose(hi - lo, 0.25)
    x, templ = _toy_sequence()
    # Templates that change twice as fast for t > 0.5: arc-uniform nodes
    # are denser there.
    t = np.linspace(0, 1, 50)
    fast = np.where(t > 0.5, 0.5 + 2 * (t - 0.5), t)
    tn, lo, hi = specml.arc_windows(t, templ(np.clip(fast / 1.5, 0, 1)),
                                    16)
    assert tn[0] == 0 and np.isclose(tn[-1], 1)
    assert np.all(np.diff(tn) > 0)
    assert np.median(np.diff(tn)[-5:]) < np.median(np.diff(tn)[:5])
    rng = np.random.default_rng(0)
    y = templ(rng.uniform(0, 1, 500))
    tt = specml.rank_uniform(y[:, 0])
    med, sc, cnt = specml.node_medians(y, tt, (tn, lo, hi), min_members=20)
    assert med.shape == (16, 76) and cnt.min() >= 20
