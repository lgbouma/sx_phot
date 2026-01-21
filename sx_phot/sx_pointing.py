from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import astropy.units as u
from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord
from numpy.typing import ArrayLike


def in_spherex_deep_fields(
    elon: Union[ArrayLike, u.Quantity],
    elat: Union[ArrayLike, u.Quantity],
    radius_deg: float = 5.0,
    south_center_elon_deg: float = 44.8,
    south_center_elat_deg: float = -82.0,
    return_components: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return a boolean mask selecting sources in either SPHEREx deep field.
    Per Figure 9 of Bock+2025 https://arxiv.org/abs/2511.02985v2, the radius is
    strictly ~ 3.5 / 2, since the FoV of the combined focal plane is 3.5 x
    11 .5 degrees.  But setting radius_deg = 5 will include stars with extra
    coverage.

    Uses angular separation in the Geocentric True Ecliptic frame.

    Args:
        elon: Ecliptic longitude(s) in degrees (array-like) or Quantity.
        elat: Ecliptic latitude(s) in degrees (array-like) or Quantity.
        radius_deg: Selection radius (degrees) around each deep-field center.
        south_center_elon_deg: SDF center ecliptic longitude (deg).
        south_center_elat_deg: SDF center ecliptic latitude (deg).
        return_components: If True, return (in_north, in_south, in_either).

    Returns:
        Boolean array mask (in_either) or a 3-tuple of boolean arrays if
        return_components is True.
    """
    elon_q = u.Quantity(elon, u.deg)
    elat_q = u.Quantity(elat, u.deg)
    radius = u.Quantity(radius_deg, u.deg)

    frame = GeocentricTrueEcliptic()
    coords = SkyCoord(lon=elon_q, lat=elat_q, frame=frame)

    north_center = SkyCoord(lon=0.0 * u.deg, lat=90.0 * u.deg, frame=frame)
    south_center = SkyCoord(
        lon=south_center_elon_deg * u.deg,
        lat=south_center_elat_deg * u.deg,
        frame=frame,
    )

    in_north = coords.separation(north_center) <= radius
    in_south = coords.separation(south_center) <= radius
    in_either = in_north | in_south

    if return_components:
        return (
            np.asarray(in_north, dtype=bool),
            np.asarray(in_south, dtype=bool),
            np.asarray(in_either, dtype=bool),
        )

    return np.asarray(in_either, dtype=bool)
