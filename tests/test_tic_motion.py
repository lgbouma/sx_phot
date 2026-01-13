from __future__ import annotations

import numpy as np
from astropy.time import Time

from sx_phot.tic_motion import propagate_gaia_to_mjd


def test_propagate_gaia_no_motion() -> None:
    ra0 = 120.0
    dec0 = -30.0
    mjd = Time(2016.0, format="jyear").mjd

    ra, dec = propagate_gaia_to_mjd(
        ra_deg=ra0,
        dec_deg=dec0,
        pmra_masyr=0.0,
        pmdec_masyr=0.0,
        mjd=np.array([mjd]),
        ref_epoch_jyear=2015.5,
    )

    assert np.allclose(ra, [ra0])
    assert np.allclose(dec, [dec0])


def test_propagate_gaia_simple_ra_shift() -> None:
    ra0 = 10.0
    dec0 = 0.0
    pmra = 1000.0  # mas/yr = 1 arcsec/yr
    pmdec = 0.0

    ref_time = Time(2015.5, format="jyear")
    target_time = Time(2016.5, format="jyear")

    ra, dec = propagate_gaia_to_mjd(
        ra_deg=ra0,
        dec_deg=dec0,
        pmra_masyr=pmra,
        pmdec_masyr=pmdec,
        mjd=target_time.mjd,
        ref_epoch_jyear=ref_time.jyear,
    )

    expected_ra = ra0 + 1.0 / 3600.0
    assert np.isclose(ra[0], expected_ra, rtol=0, atol=1e-6)
    assert np.isclose(dec[0], dec0, rtol=0, atol=1e-8)
