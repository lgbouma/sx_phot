#!/Users/luke/local/miniconda3/envs/py311_sx/bin/python
"""Annotate a CSV catalog with ra and dec columns with SPHEREx deep-field membership."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord

from sx_phot.sx_pointing import in_spherex_deep_fields


DEFAULT_INPUT = Path("/Users/luke/local/TARS/jan8_default_catalog_no_quality_flags.csv")
DEFAULT_OUTPUT = Path(
    "/Users/luke/local/TARS/jan8_default_catalog_no_quality_flags_spherex_obs.csv"
)


def _configure_logging(verbose: bool) -> None:
    """Configure logging for the script.

    Args:
        verbose: If True, use DEBUG level logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


def add_spherex_deep_field_flag(
    df: pd.DataFrame,
    ra_col: str,
    dec_col: str,
) -> pd.DataFrame:
    """Return a copy of the catalog with SPHEREx deep-field membership.

    Args:
        df: Input catalog with RA/Dec columns.
        ra_col: Column name for right ascension in degrees.
        dec_col: Column name for declination in degrees.

    Returns:
        New DataFrame with the ``in_spherex_deep_fields`` column added.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = [col for col in (ra_col, dec_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    output = df.copy()
    elon, elat, valid_mask = _compute_ecliptic_coords(
        output[ra_col].to_numpy(),
        output[dec_col].to_numpy(),
    )
    logging.debug("Computed ecliptic coords for %d valid rows.", int(valid_mask.sum()))

    in_deep = np.zeros(len(output), dtype=bool)
    if np.any(valid_mask):
        in_deep[valid_mask] = in_spherex_deep_fields(
            elon[valid_mask],
            elat[valid_mask],
        )
    output["in_spherex_deep_fields"] = in_deep
    logging.info(
        "Flagged %d/%d rows inside SPHEREx deep fields.",
        int(in_deep.sum()),
        len(output),
    )
    return output


def main() -> None:
    """CLI entry point for annotating the TARS catalog."""
    parser = argparse.ArgumentParser(
        description=(
            "Add an in_spherex_deep_fields flag to a catalog using RA/Dec "
            "coordinates."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--ra-col",
        default="ra",
        help="Column name for right ascension in degrees.",
    )
    parser.add_argument(
        "--dec-col",
        default="dec",
        help="Column name for declination in degrees.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    _configure_logging(args.verbose)

    input_path = Path(args.input)
    output_path = Path(args.output)
    deep_output_path = output_path.with_name(
        f"{output_path.stem}_deep_fields{output_path.suffix}"
    )

    logging.info("Reading catalog from %s", input_path)
    df = pd.read_csv(input_path)
    logging.info("Loaded %d rows with %d columns.", len(df), df.shape[1])

    updated = add_spherex_deep_field_flag(df, args.ra_col, args.dec_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(output_path, index=False)
    logging.info("Wrote updated catalog to %s", output_path)

    sdf = updated[updated["in_spherex_deep_fields"]].copy()
    sdf.to_csv(deep_output_path, index=False)
    logging.info(
        "Wrote %d deep-field rows to %s",
        len(sdf),
        deep_output_path,
    )


if __name__ == "__main__":
    main()
