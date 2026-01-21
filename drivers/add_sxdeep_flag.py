#!/Users/luke/local/miniconda3/envs/py311_sx/bin/python
"""Add ecliptic coordinates and SPHEREx deep-field flags to a target list."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord

from sx_phot.sx_pointing import in_spherex_deep_fields


DEFAULT_INPUT = Path(
    "/Users/luke/Dropbox/proj/sx_phot/sx_phot/data/targetlists/"
    "concat_R16_S17_S18_B20_S21_Z19_G22_P23_B24_qlp_0to100pc.csv"
)


def _default_output_path(input_path: Path) -> Path:
    """Return the default output CSV path for a given input file.

    Args:
        input_path: Path to the input CSV file.

    Returns:
        Output CSV path with the suffix ``_sxdeep`` appended to the stem.
    """
    return input_path.with_name(f"{input_path.stem}_sxdeep.csv")


def _compute_ecliptic_coords(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ecliptic longitude/latitude from RA/Dec.

    Args:
        ra_deg: Right ascension values in degrees.
        dec_deg: Declination values in degrees.

    Returns:
        Tuple of (elon_deg, elat_deg, valid_mask).
    """
    ra_deg = np.asarray(ra_deg, dtype=float)
    dec_deg = np.asarray(dec_deg, dtype=float)
    valid_mask = np.isfinite(ra_deg) & np.isfinite(dec_deg)
    elon = np.full(ra_deg.shape, np.nan, dtype=float)
    elat = np.full(dec_deg.shape, np.nan, dtype=float)

    if np.any(valid_mask):
        coords = SkyCoord(
            ra=ra_deg[valid_mask] * u.deg,
            dec=dec_deg[valid_mask] * u.deg,
            frame="icrs",
        )
        ecl = coords.transform_to(GeocentricTrueEcliptic())
        elon[valid_mask] = ecl.lon.to_value(u.deg)
        elat[valid_mask] = ecl.lat.to_value(u.deg)

    return elon, elat, valid_mask


def _deep_field_distances(
    elon_deg: np.ndarray,
    elat_deg: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Compute angular distance to the nearest SPHEREx deep field center.

    Args:
        elon_deg: Ecliptic longitude values in degrees.
        elat_deg: Ecliptic latitude values in degrees.
        valid_mask: Boolean mask for valid coordinates.

    Returns:
        Array of minimum angular separation (degrees) to the north/south centers.
    """
    dist = np.full(elon_deg.shape, np.nan, dtype=float)
    if not np.any(valid_mask):
        return dist

    frame = GeocentricTrueEcliptic()
    coords = SkyCoord(
        lon=elon_deg[valid_mask] * u.deg,
        lat=elat_deg[valid_mask] * u.deg,
        frame=frame,
    )
    north_center = SkyCoord(lon=0.0 * u.deg, lat=90.0 * u.deg, frame=frame)
    south_center = SkyCoord(lon=44.8 * u.deg, lat=-82.0 * u.deg, frame=frame)
    north_sep = coords.separation(north_center).deg
    south_sep = coords.separation(south_center).deg
    dist[valid_mask] = np.minimum(north_sep, south_sep)
    return dist


def add_sxdeep_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the input table with ecliptic and deep-field columns.

    Args:
        df: Input DataFrame containing ``tic8_ra`` and ``tic8_dec`` columns.

    Returns:
        New DataFrame with ``elon``, ``elat``, and ``in_sx_deep_field`` columns.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"tic8_ra", "tic8_dec"}
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    output = df.copy()
    elon, elat, valid_mask = _compute_ecliptic_coords(
        output["tic8_ra"].to_numpy(),
        output["tic8_dec"].to_numpy(),
    )
    output["elon"] = elon
    output["elat"] = elat
    output["sx_deep_center_sep_deg"] = _deep_field_distances(
        elon,
        elat,
        valid_mask,
    )

    in_deep = np.zeros(len(output), dtype=bool)
    if np.any(valid_mask):
        in_deep[valid_mask] = in_spherex_deep_fields(
            elon[valid_mask],
            elat[valid_mask],
        )
    output["in_sx_deep_field"] = in_deep
    return output


def main() -> None:
    """CLI entry point for annotating target lists."""
    parser = argparse.ArgumentParser(
        description=(
            "Annotate a target list with ecliptic coordinates and SPHEREx "
            "deep-field membership."
        ),
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input pipe-delimited CSV path.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (defaults to <input>_sxdeep.csv).",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.out) if args.out else _default_output_path(input_path)

    df = pd.read_csv(input_path, sep="|")
    updated = add_sxdeep_flag(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(output_path, sep="|", index=False)
    print(f"Wrote {len(updated)} rows to {output_path}")


if __name__ == "__main__":
    main()
