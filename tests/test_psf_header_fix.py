"""Tests for sx_phot.psf_header_fix (synthetic Level-2 files)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from sx_phot.psf_header_fix import fix_file, swap_zone_cards


def _l2_file(path: Path, version: str) -> np.ndarray:
    """Write a minimal Level-2 file with a two-zone PSF cube."""
    rng = np.random.default_rng(0)
    cube = rng.normal(size=(2, 7, 7)).astype(np.float32)
    prim = fits.PrimaryHDU()
    prim.header["VERSION"] = (version, "SSDC Pipelines software version")
    img = fits.ImageHDU(np.zeros((4, 4), np.float32), name="IMAGE")
    psf = fits.ImageHDU(cube, name="PSF")
    psf.header["XCTR_1"] = (93.0, "Center of x zone (0, 0)")
    psf.header["YCTR_1"] = (93.0, "Center of y zone (0, 0)")
    psf.header["XWID_1"] = (186.0, "Width of x zone (0, 0)")
    psf.header["YWID_1"] = (186.0, "Width of y zone (0, 0)")
    psf.header["XCTR_2"] = (93.0, "Center of x zone (0, 1)")
    psf.header["YCTR_2"] = (278.0, "Center of y zone (0, 1)")
    psf.header["XWID_2"] = (186.0, "Width of x zone (0, 1)")
    psf.header["YWID_2"] = (190.0, "Width of y zone (0, 1)")
    fits.HDUList([prim, img, psf]).writeto(path)
    return cube


def test_swap_zone_cards():
    """Values cross over; comments stay but flip their zone index."""
    h = fits.Header()
    h["NAXIS3"] = 1
    h["XCTR_1"] = (1.0, "Center of x zone (3, 7)")
    h["YCTR_1"] = (2.0, "Center of y zone (3, 7)")
    h["XWID_1"] = (3.0, "Width of x zone (3, 7)")
    h["YWID_1"] = (4.0, "Width of y zone (3, 7)")
    assert swap_zone_cards(h) == 1
    assert (h["XCTR_1"], h["YCTR_1"]) == (2.0, 1.0)
    assert (h["XWID_1"], h["YWID_1"]) == (4.0, 3.0)
    assert h.comments["XCTR_1"] == "Center of x zone (7, 3)"
    assert h.comments["YWID_1"] == "Width of y zone (7, 3)"


def test_fix_file_repairs_once(tmp_path: Path):
    """An affected file is repaired in place, once, with data intact."""
    p = tmp_path / "l2.fits"
    cube = _l2_file(p, "6.4")
    size = p.stat().st_size
    assert fix_file(p, dry_run=True) == "would-fix"
    assert fits.getheader(p, 0)["VERSION"] == "6.4"
    assert fix_file(p) == "fixed"
    assert p.stat().st_size == size
    with fits.open(p) as hdul:
        assert hdul[0].header["VERSION"] == "6.4+psffix1"
        h = hdul["PSF"].header
        assert (h["XCTR_2"], h["YCTR_2"]) == (278.0, 93.0)
        assert (h["XWID_2"], h["YWID_2"]) == (190.0, 186.0)
        assert h.comments["XCTR_2"] == "Center of x zone (1, 0)"
        assert h.comments["YCTR_1"] == "Center of y zone (0, 0)"
        assert np.array_equal(hdul["PSF"].data, cube)
        assert hdul["IMAGE"].data.shape == (4, 4)
    assert fix_file(p) == "skipped"
    assert fits.getheader(p, "PSF")["XCTR_2"] == 278.0


@pytest.mark.parametrize("version", ["6.5.6", "7.0.5", "6.5.5+psffix1"])
def test_fix_file_skips_unaffected(tmp_path: Path, version: str):
    """Unaffected and already-fixed versions are left alone."""
    p = tmp_path / "l2.fits"
    _l2_file(p, version)
    before = p.read_bytes()
    assert fix_file(p) == "skipped"
    assert p.read_bytes() == before
