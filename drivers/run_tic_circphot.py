"""Run SPHEREx aperture photometry for a TIC target using mean coordinates.

Usage:
    (py311_sx) python run_tic_circphot.py TIC_300651846
    (py311_sx) python run_tic_circphot.py 300651846

API:
    - normalize_tic_id(tic_id) coerces TIC identifiers to integers.
    - get_mean_spherex_coords(tic_id) returns mean RA/Dec propagated for SPHEREx.
    - get_sx_spectrum(...) runs the aperture photometry and writes outputs to
      test_results/ using IRSA cutouts by default.
"""

from __future__ import annotations

import argparse
import socket

from get_tic_spherex_coords import get_mean_spherex_coords
from sx_phot.circphot import get_supplemented_sx_spectrum
from sx_phot.tic_motion import normalize_tic_id


DEFAULT_TIC_ID = "TIC_300651846"
USE_CUTOUT = True if socket.gethostname() != 'wh3' else False


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run SPHEREx aperture photometry for a TIC ID using mean propagated "
            "SPHEREx coordinates."
        )
    )
    parser.add_argument(
        "tic_id",
        nargs="?",
        default=DEFAULT_TIC_ID,
        help="TIC identifier, e.g. TIC_300651846 or 300651846.",
    )
    return parser.parse_args()


def main() -> None:
    """Query mean SPHEREx coordinates for a TIC target and run photometry."""
    args = _parse_args()
    tic_id = normalize_tic_id(args.tic_id)
    ra_deg, dec_deg = get_mean_spherex_coords(tic_id)
    star_id = f"TIC_{tic_id}"

    print(
        f"Using mean coordinates for {star_id}: "
        f"RA={ra_deg:.6f}, Dec={dec_deg:.6f}"
    )

    get_supplemented_sx_spectrum(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        star_id=star_id,
        output_dir="test_results",
        use_cutout=USE_CUTOUT,
    )


if __name__ == "__main__":
    main()
