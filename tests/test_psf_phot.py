"""Tests for sx_phot.psf_phot and sx_phot.gaia_field (synthetic data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from scipy import ndimage

from sx_phot.bulk_phot import (
    ROW_DTYPES,
    PhotConfig,
    init_run_dir,
    photometer_image,
    row_dtypes,
)
from sx_phot.gaia_field import load_neighbours
from sx_phot.psf_phot import (
    PSF_DTYPES,
    PsfParams,
    ZonePsf,
    fit_image,
    fit_star,
    psf_header_transposed,
)

OMEGA_SR = 9e-10


def _psf_hdul(sigma_samples: float = 8.0) -> fits.HDUList:
    """One-zone qr2-style PSF cube: Gaussian, 101 x 101 at 0.1 px."""
    yy, xx = np.mgrid[0:101, 0:101] - 50.0
    p = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma_samples ** 2)
    hdu = fits.ImageHDU(p[None].astype(np.float32), name="PSF")
    hdu.header["OVERSAMP"] = 10
    hdu.header["CRPIX1"] = 51
    hdu.header["XCTR_1"] = 1020.0
    hdu.header["YCTR_1"] = 1020.0
    return fits.HDUList([fits.PrimaryHDU(), hdu])


def _spline_stamp(p: np.ndarray, z: ZonePsf, fx: float,
                  fy: float) -> np.ndarray:
    """Stamp by a cubic-spline shift of the oversampled PSF."""
    o, r = z.o, z.reach
    n = ndimage.shift(p, (o * fy + z.ev, o * fx + z.ev), order=3,
                      mode="constant")
    m = 2 * r + 1
    return n[z.k0:z.k0 + o * m, z.k0:z.k0 + o * m].reshape(
        m, o, m, o).sum(axis=(1, 3))


def test_stamp_matches_spline_shift():
    """Cached integer-shift stamps match a cubic-spline shift."""
    hdul = _psf_hdul()
    z = ZonePsf(hdul)
    p = np.asarray(hdul["PSF"].data[0], float)
    rng = np.random.default_rng(1)
    for fx, fy in rng.uniform(-0.5, 0.5, size=(20, 2)):
        a = z.stamp(0, fx, fy)
        b = _spline_stamp(p, z, fx, fy)
        assert np.max(np.abs(a - b)) < 3e-4 * b.max()
    assert z.stamp(0, 0.0, 0.0).sum() == pytest.approx(z.psum(0),
                                                       rel=1e-3)


def _scene(sources, size: int = 64, bkg=(5.0, 0.1, -0.05),
           noise: float = 0.5, seed: int = 0):
    """Image with given (x, y, amplitude) sources plus a plane."""
    z = ZonePsf(_psf_hdul())
    img = np.zeros((size, size))
    for x, y, amp in sources:
        img += amp * z.render(0, x, y, size, size)
    yy, xx = np.mgrid[0:size, 0:size]
    img += bkg[0] + bkg[1] * (xx - 32) / 5 + bkg[2] * (yy - 32) / 5
    rng = np.random.default_rng(seed)
    img += rng.normal(0, noise, img.shape)
    var = np.full(img.shape, noise ** 2)
    flags = np.zeros(img.shape, dtype=np.int32)
    return img, flags, var, z


def test_fit_recovers_blended_pair():
    """Two sources 1.4 px apart are recovered within their errors."""
    src = [(32.1, 31.8, 400.0), (33.5, 31.8, 150.0)]
    img, flags, var, z = _scene(src)
    x = np.array([s[0] for s in src]) - z_shift()[0]
    y = np.array([s[1] for s in src]) - z_shift()[1]
    res = fit_star(img, flags, var, z, x, y, np.array([12.0, 13.0]), 0,
                   OMEGA_SR)
    assert res["status"] == 0
    assert res["n_comp"] == 2
    k = 1e6 * OMEGA_SR * z.psum(0)
    amp, err = res["flux_jy"] / k, res["flux_err_jy"] / k
    assert np.isfinite(err) and err > 0
    assert abs(amp - 400.0) < 4 * err
    assert abs(res["nb_flux_jy"] / k - 150.0) < 0.1 * 150.0
    assert res["min_corr"] < 0


def z_shift() -> tuple[float, float]:
    """Fixed qr2 offset applied by fit_star (sources are drawn at the
    true position, Gaia positions are offset by -SHIFT0)."""
    from sx_phot.psf_phot import SHIFT0

    return SHIFT0["cube"]


def test_nearly_unconstrained_neighbour_keeps_finite_error():
    """A neighbour whose stamp barely reaches the box leaves the target
    flux unchanged and its error finite (pseudo-inverse fallback)."""
    src = [(32.0, 32.0, 300.0)]
    img, flags, var, z = _scene(src, seed=3)
    # Neighbour centred 8.6 px right of the target: its 4 px stamp
    # reaches only the last box column, where the Gaussian wing is ~0.
    x = np.array([32.0, 40.6]) - z_shift()[0]
    y = np.array([32.0, 32.0]) - z_shift()[1]
    g = np.array([12.0, 11.0])
    kept = fit_star(img, flags, var, z, x, y, g, 0, OMEGA_SR)
    alone = fit_star(img, flags, var, z, x[:1], y[:1], g[:1], 0, OMEGA_SR)
    drop = fit_star(img, flags, var, z, x, y, g, 0, OMEGA_SR,
                    PsfParams(drop_frac=0.02))
    assert kept["n_comp"] == 2 and kept["n_drop"] == 0
    assert drop["n_comp"] == 1 and drop["n_drop"] == 1
    assert np.isfinite(kept["flux_err_jy"]) and kept["flux_err_jy"] > 0
    assert kept["flux_err_jy"] == pytest.approx(alone["flux_err_jy"],
                                                rel=5e-3)
    assert kept["flux_jy"] == pytest.approx(alone["flux_jy"], rel=1e-3)


def test_neighbour_without_light_is_dropped():
    """A neighbour whose stamp falls only on masked pixels is dropped."""
    img, flags, var, z = _scene([(32.0, 32.0, 300.0)])
    # Box spans image columns 27-37; the neighbour's stamp covers box
    # columns 9-10 only (image 36-37), which are masked.
    flags[:, 36:38] = 1
    x = np.array([32.0, 40.4]) - z_shift()[0]
    y = np.array([32.0, 32.0]) - z_shift()[1]
    res = fit_star(img, flags, var, z, x, y, np.array([12.0, 10.0]), 0,
                   OMEGA_SR)
    assert res["n_comp"] == 1 and res["n_drop"] == 1


def test_crowded_box_is_underdetermined():
    """More sources than the box can constrain gives status 5."""
    rng = np.random.default_rng(5)
    img, flags, var, z = _scene([(32.0, 32.0, 300.0)])
    n = 110
    x = np.r_[32.0, rng.uniform(27.0, 37.0, n)]
    y = np.r_[32.0, rng.uniform(27.0, 37.0, n)]
    g = np.r_[15.0, np.full(n, 16.0)]
    res = fit_star(img, flags, var, z, x, y, g, 0, OMEGA_SR)
    assert res["status"] == 5
    assert "flux_jy" in res


def test_fit_status_edge_and_masked():
    """Boxes off the detector are 'edge'; masked cores are 'masked'."""
    img, flags, var, z = _scene([(32.0, 32.0, 300.0)])
    g = np.array([12.0])
    res = fit_star(img, flags, var, z, np.array([3.0]), np.array([30.0]),
                   g, 0, OMEGA_SR)
    assert res["status"] == 1
    flags[30:35, 30:35] = 1
    res = fit_star(img, flags, var, z, np.array([32.0]), np.array([32.0]),
                   g, 0, OMEGA_SR)
    assert res["status"] == 2


def test_fit_image_no_gaia_match():
    """Stars without a Gaia match get status 3 and NaN fluxes."""
    img, flags, var, z = _scene([(32.0, 32.0, 300.0)])
    rows = fit_image(img, flags, var, z, OMEGA_SR, np.array([32.0]),
                     np.array([32.0]), np.array([12.0]),
                     np.array([0, 1, 1]), np.array([0, -1]))
    assert list(rows["psf_status"]) == [0, 3]
    assert np.isfinite(rows["psf_flux_jy"][0])
    assert np.isnan(rows["psf_flux_jy"][1])
    assert set(rows) == set(PSF_DTYPES)


def test_load_neighbours_targets_and_prefilter():
    """Targets are the nearest source within the match radius; faint
    distant sources that the selection rules cannot keep are dropped."""
    df = pd.DataFrame({
        "star_idx": [0, 0, 0, 0, 2, 2],
        "source_id": [1, 2, 3, 4, 5, 6],
        "sep_arcsec": [0.1, 5.0, 40.0, 70.0, 3.0, 20.0],
        "ra": np.zeros(6), "dec": np.zeros(6),
        "pmra": [1.0, np.nan, 0.0, 0.0, 0.0, 0.0],
        "pmdec": np.zeros(6),
        "phot_g_mean_mag": [12.0, 18.0, 15.0, 13.0, 14.0, 15.0],
    })
    nb = load_neighbours(Path("unused"), n_stars=3, df=df)
    # star 0: target kept; G 18 at 5" kept (near, < G_t + 6.5);
    # G 15 at 40" kept (< G_t + 4.25, inside the half-diagonal);
    # G 13 at 70" kept (< G_t + 1.5).  star 1: none.  star 2: no target
    # (nearest at 3"), sources kept unfiltered.
    assert list(nb.offsets) == [0, 4, 4, 6]
    assert list(nb.target) == [0, -1, -1]
    assert nb.pmra[1] == 0.0
    df.loc[3, "phot_g_mean_mag"] = 14.0      # now >= G_t + 1.5 at 70"
    df.loc[1, "phot_g_mean_mag"] = 18.6      # now >= G_t + 6.5
    nb = load_neighbours(Path("unused"), n_stars=3, df=df)
    assert list(nb.offsets) == [0, 2, 2, 4]


def test_row_dtypes():
    """PSF columns are added only for PSF runs."""
    assert row_dtypes(PhotConfig()) == ROW_DTYPES
    assert set(row_dtypes(PhotConfig(psf=True))) == (
        set(ROW_DTYPES) | set(PSF_DTYPES))


def test_init_run_dir_accepts_pre_psf_config(tmp_path: Path):
    """Run dirs created before the PSF options validate unchanged."""
    cat = pd.DataFrame({"star_idx": [0], "ticid": [5], "ra": [1.0],
                        "dec": [2.0], "pmra": [0.0], "pmdec": [0.0]})
    index = pd.DataFrame({"image_key": [1], "path": ["x"]})
    init_run_dir(tmp_path, cat, index, PhotConfig())
    cfg = json.loads((tmp_path / "config.json").read_text())
    for k in ("psf", "psf_drop_frac", "psf_bg_order"):
        del cfg["phot"][k]
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    init_run_dir(tmp_path, cat, index, PhotConfig())
    with pytest.raises(ValueError, match="config differs"):
        init_run_dir(tmp_path, cat, index, PhotConfig(psf=True))


def test_photometer_image_psf_needs_neighbours():
    """PSF mode without a neighbour table fails loudly."""
    with pytest.raises(ValueError, match="neighbour"):
        photometer_image("missing.fits", np.array([1.0]), np.array([2.0]),
                         config=PhotConfig(psf=True))


@pytest.mark.parametrize("version,expected", [
    ("6.4", True), ("6.5.5", True), ("6.5.3", True),
    ("6.4+psffix1", False), ("6.5.5+psffix1", False), ("6.5.6", False),
    ("7.0.5", False), ("6.3.9", False), (None, False), ("dev", False),
])
def test_psf_header_transposed(version, expected):
    """IRSA PSF header error: 6.4-6.5.5 without '+psffix' only."""
    assert psf_header_transposed(version) is expected


def test_zone_centres_swapped_for_affected_versions():
    """Affected files get x and y zone centres swapped back."""
    hdul = _psf_hdul()
    hdul["PSF"].header["XCTR_1"] = 93.0
    hdul["PSF"].header["YCTR_1"] = 279.0
    hdul[0].header["VERSION"] = "6.5.5+psffix1"
    z = ZonePsf(hdul)
    assert (z.zx[0], z.zy[0]) == (92.0, 278.0)
    hdul[0].header["VERSION"] = "6.4"
    z = ZonePsf(hdul)
    assert (z.zx[0], z.zy[0]) == (278.0, 92.0)


def test_query_split_halves_at_row_limit(monkeypatch, tmp_path: Path):
    """Chunks whose result reaches the archive row limit are split."""
    import sx_phot.gaia_field as gf

    calls = []

    def fake(cat, radius, xml, retries):
        calls.append(len(cat))
        n = 30 * len(cat)          # 30 sources per star
        return pd.DataFrame({"star_idx": np.repeat(
            cat["star_idx"].to_numpy(), 30)[:min(n, 100)]})

    monkeypatch.setattr(gf, "_query_chunk", fake)
    cat = pd.DataFrame({"star_idx": np.arange(8), "ra": np.zeros(8),
                        "dec": np.zeros(8)})
    res = gf._query_split(cat, 80.0, tmp_path / "u.xml", 1, max_rows=100)
    assert len(res) == 8 * 30
    assert calls[0] == 8 and min(calls) == 2


def _epsf_hdul(sigma_samples: float = 4.0) -> fits.HDUList:
    """One-zone qr3-style EPSF table: Gaussian, 33 x 33 at 0.2 px."""
    yy, xx = np.mgrid[0:33, 0:33] - 16.0
    p = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma_samples ** 2)
    p /= p.sum()
    cols = [
        fits.Column(name="XCENTER", format="E", array=np.array([1020.0])),
        fits.Column(name="YCENTER", format="E", array=np.array([1020.0])),
        fits.Column(name="EPSF", format=f"{33 * 33}E", dim="(33,33)",
                    array=p[None].astype(np.float32)),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="EPSF")
    hdu.header["OVSMPX"] = 5
    hdu.header["OVSMPY"] = 5
    return fits.HDUList([fits.PrimaryHDU(), hdu])


def test_epsf_stamp_is_point_sampled():
    """An effective PSF is sampled at pixel centres, not block-summed."""
    hdul = _epsf_hdul()
    z = ZonePsf(hdul)
    p = np.asarray(hdul["EPSF"].data["EPSF"][0], float)
    assert z.kind == "table" and z.o == 5 and z.reach == 2
    st = z.stamp(0, 0.0, 0.0)
    want = p[16 - 10:16 + 11:5, 16 - 10:16 + 11:5] * 25.0
    assert st.shape == (5, 5)
    assert np.allclose(st / st.sum(), want / want.sum(), rtol=0,
                       atol=1e-6)
    assert st.sum() == pytest.approx(z.psum(0), rel=2e-2)
    # a shift of +1 px moves the peak by one pixel
    st1 = z.stamp(0, 1.0, 0.0)
    assert np.allclose(st1[:, 1:] / st1[:, 1:].sum(),
                       st[:, :-1] / st[:, :-1].sum(), atol=1e-6)
    # the cube path still block-sums (its stamp at zero offset is an
    # interpolation between half-sample shifts, hence the tolerance)
    zc = ZonePsf(_psf_hdul())
    pc = np.asarray(_psf_hdul()["PSF"].data[0], float)
    assert zc.stamp(0, 0.0, 0.0)[4, 4] == pytest.approx(
        pc[45:55, 45:55].sum(), rel=1e-2)


def test_epsf_stamp_sum_independent_of_pixel_phase():
    """EPSF-table stamps conserve flux at every sub-pixel offset."""
    # a Gaussian of sigma 0.3 px is badly undersampled at pixel
    # centres, so its point-sampled sum depends strongly on phase
    hdul = _epsf_hdul(sigma_samples=1.5)
    offs = [(0.0, 0.0), (0.5, 0.5), (0.3, -0.2), (-0.5, 0.1)]
    raw = ZonePsf(hdul, renorm_table=False)
    sums = [raw.stamp(0, fx, fy).sum() for fx, fy in offs]
    assert np.ptp(sums) > 0.3 * np.mean(sums)
    z = ZonePsf(hdul)
    for fx, fy in offs:
        st = z.stamp(0, fx, fy)
        assert st.sum() == pytest.approx(z.stamp_sum(0), rel=1e-12)
        r = raw.stamp(0, fx, fy)
        assert np.allclose(st / st.sum(), r / r.sum(), atol=1e-12)
    # the cube path is not rescaled
    assert not ZonePsf(_psf_hdul()).renorm
