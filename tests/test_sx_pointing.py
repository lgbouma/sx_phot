import astropy.units as u
from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord

from sx_phot.sx_pointing import in_spherex_deep_fields


def test_tic_220458356_in_deep_field() -> None:
    ra_deg = 74.60971924107795
    dec_deg = -56.734269675708575
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    ecl = coords.transform_to(GeocentricTrueEcliptic())
    elon = ecl.lon.to_value(u.deg)
    elat = ecl.lat.to_value(u.deg)

    assert in_spherex_deep_fields(elon, elat)
