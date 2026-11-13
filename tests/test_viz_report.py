"""Tests for the sx-viz renderings: north-up cutouts, RGB stack, report."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sx_phot.sedmodels import spherex_bin_edges  # noqa: E402
from sx_phot.viz_products import star_products  # noqa: E402
from sx_phot.viz_report import (plot_star_report, preferred_model,  # noqa
                                render_rgb_cutout, render_spherex_cutout,
                                rgb_stack, sky_cutout)
from test_viz_products import CFG, _fake_rows  # noqa: E402

SCALE_DEG = 6.1 / 3600


def _write_image(path: Path, angle_deg: float, star: tuple[float, float],
                 neighbour_offset_arcsec: tuple[float, float]) -> None:
    """A 200x200 image with a TAN WCS rotated by ``angle_deg``.

    ``neighbour_offset_arcsec`` is (east, north) of the neighbour relative
    to the star; the neighbour pixel is placed via the WCS.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 20.0]
    w.wcs.crpix = [100.0, 100.0]
    th = np.radians(angle_deg)
    # Column 1 of the CD matrix is the sky step per x pixel; a positive
    # RA step with x makes the image east-right (flipped) at angle 0.
    w.wcs.cd = SCALE_DEG * np.array([[np.cos(th), -np.sin(th)],
                                     [np.sin(th), np.cos(th)]])
    img = np.zeros((200, 200), float)
    x, y = star
    img[int(round(y)), int(round(x))] = 100.0
    ra0, dec0 = w.all_pix2world(x, y, 0)
    de, dn = neighbour_offset_arcsec
    ra1 = ra0 + de / 3600 / np.cos(np.radians(dec0))
    dec1 = dec0 + dn / 3600
    xn, yn = w.all_world2pix(ra1, dec1, 0)
    img[int(round(float(yn))), int(round(float(xn)))] = 50.0
    flags = np.zeros((200, 200), np.int32)
    flags[int(round(float(yn))), int(round(float(xn)))] = 1
    hdul = fits.HDUList([fits.PrimaryHDU(),
                         fits.ImageHDU(img, header=w.to_header(),
                                       name="IMAGE"),
                         fits.ImageHDU(flags, name="FLAGS")])
    hdul.writeto(path, overwrite=True)


@pytest.mark.parametrize("angle", [0.0, 37.0, 200.0])
def test_sky_cutout_is_north_up_east_left(tmp_path, angle):
    path = tmp_path / f"img_{angle:.0f}.fits"
    # Neighbour 5 pixels east and 3 pixels north of the star.
    _write_image(path, angle, (90.3, 110.6), (5 * 6.1, 3 * 6.1))
    cut, flags, scale = sky_cutout(path, 90.3, 110.6, half_size=10)
    assert cut.shape == (21, 21)
    assert abs(scale - 6.1) < 0.01
    assert cut[10, 10] == 100.0
    iy, ix = np.unravel_index(np.argmax(np.where(cut == 50.0, 1, 0)),
                              cut.shape)
    # East is left (column decreases), north is up (row increases).
    assert (ix, iy) == (10 - 5, 10 + 3)
    assert flags[iy, ix] == 1 and flags[10, 10] == 0


def test_sky_cutout_off_detector_is_nan(tmp_path):
    path = tmp_path / "edge.fits"
    _write_image(path, 10.0, (3.0, 100.0), (6.1, 0.0))
    cut, _, _ = sky_cutout(path, 3.0, 100.0, half_size=10)
    assert cut[10, 10] == 100.0
    assert np.isnan(cut).any()


def test_rgb_stack_and_renders(tmp_path):
    path = tmp_path / "img.fits"
    _write_image(path, 45.0, (100.0, 100.0), (12.2, -6.1))
    cut, _, _ = sky_cutout(path, 100.0, 100.0, half_size=8)
    rgb = rgb_stack([cut, None, cut])
    assert rgb.shape == (17, 17, 3)
    assert rgb[8, 8, 0] == pytest.approx(1.0) and rgb[8, 8, 2] == 1.0
    assert rgb[..., 1].max() == 0.0
    png = render_rgb_cutout(rgb, 6.1, 2, 6, 8, title="t")
    assert png.startswith(b"\x89PNG")
    png = render_spherex_cutout(path, 100.0, 100.0, 2, 6, 8, half_size=8,
                                title="t")
    assert png.startswith(b"\x89PNG")


def test_preferred_model():
    assert preferred_model(None) is None
    assert preferred_model({"bic_single": 10.0}) == "full"
    assert preferred_model({"bic_single": 10.0, "bic_binary": 4.0,
                            "bic_disk": float("nan")}) == "binary"
    assert preferred_model({"bic_single": 10.0, "bic_binary": 4.0,
                            "bic_disk": 3.0}) == "disk"


def test_plot_star_report_with_cutouts():
    from PIL import Image

    rows = _fake_rows(n=300, seed=1)
    edges = spherex_bin_edges()
    spectrum, summary, binned = star_products(rows, CFG, edges)
    spectrum["tjd_avg"] = spectrum["mjd_avg"] - 57000.0  # as SpectrumStore
    summary.update({"ticid": 1, "Tmag": 10.0, "adopted_period": 1.5})
    binned["lam"] = np.sqrt(edges[:-1] * edges[1:])
    b = {k: [None if not np.isfinite(v) else float(v) for v in binned[k]]
         for k in ("lam", "flux", "err", "model")}
    buf = io.BytesIO()
    Image.fromarray(np.zeros((300, 300), np.uint8)).save(buf, format="JPEG")
    rgb = np.zeros((41, 41, 3))
    png = plot_star_report(spectrum, summary, b, None,
                           sky_jpeg=buf.getvalue(), spx_rgb=rgb,
                           spx_title="stack", phot=CFG)
    assert png.startswith(b"\x89PNG")
    png = plot_star_report(spectrum, summary, b, None)
    assert png.startswith(b"\x89PNG")


def test_propagate_position_and_decimal_year():
    from sx_phot.viz_report import decimal_year, propagate_position

    # 1000 mas/yr east over 10 years at dec 60: 10" / cos(60) in RA.
    ra, dec = propagate_position(10.0, 60.0, 1000.0, -500.0, 2006.0)
    assert ra == pytest.approx(10.0 - 10.0 / 3600 / 0.5, rel=1e-6)
    assert dec == pytest.approx(60.0 + 5.0 / 3600, rel=1e-9)
    assert propagate_position(1.0, 2.0, None, float("nan"), 1990) == (1, 2)
    assert decimal_year("1996-05-13T06:80:00") == pytest.approx(1996.364,
                                                                1e-3)
    assert decimal_year("1979-12-21T16:27") == pytest.approx(1979.97, 1e-3)
    assert decimal_year("") is None and decimal_year(None) is None


def test_resample_north_up_rescales(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS

    from sx_phot.viz_report import resample_north_up

    path = tmp_path / "img.fits"
    # Neighbour one pixel (6.1") east of the star in a 6.1"/px image
    # (angle 0 keeps it on an integer pixel).
    _write_image(path, 0.0, (100.0, 100.0), (6.1, 0.0))
    with fits.open(path) as hdul:
        data = np.array(hdul["IMAGE"].data, float)
        wcs = WCS(hdul["IMAGE"].header)
    img, scale = resample_north_up(data, wcs, 100.0, 100.0, 15,
                                   out_scale=0.61, order=0)
    assert scale == 0.61 and img.shape == (31, 31)
    # 6.1" east = 10 output pixels to the left of the centre; nearest
    # sampling spreads the input pixel over a 10 x 10 block.
    rows, cols = np.nonzero(img == 50.0)
    assert cols.mean() == pytest.approx(15 - 10, abs=0.5)
    assert rows.mean() == pytest.approx(15, abs=0.5)
    assert img[15, 15] == 100.0
