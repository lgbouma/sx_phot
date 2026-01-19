"""Generate a wavelength-ordered cutout animation for a TIC target.

Example:
    /Users/luke/local/miniconda3/envs/py311_sx/bin/python \
        drivers/animate_tic_cutouts.py TIC_300936690
"""

from __future__ import annotations

import argparse
from pathlib import Path

from get_tic_spherex_coords import get_mean_spherex_coords
from sx_phot.circphot import get_sx_spectrum
from sx_phot.cutout_animation import animate_cutouts_from_records
from sx_phot.tic_motion import normalize_tic_id


DEFAULT_TIC_ID = "TIC_300936690"
DEFAULT_OUTPUT_DIR = "test_results"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build an MP4 animation of wavelength-ordered cutouts for a TIC target."
        )
    )
    parser.add_argument(
        "tic_id",
        nargs="?",
        default=DEFAULT_TIC_ID,
        help="TIC identifier, e.g. TIC_300936690 or 300936690.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the animation and cached CSV.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output MP4 path. Defaults to <output-dir>/<star>_cutout_anim.mp4.",
    )
    parser.add_argument(
        "--size-pix",
        type=int,
        default=64,
        help="Cutout size in pixels (square).",
    )
    parser.add_argument(
        "--aperture-radius",
        type=float,
        default=2.0,
        help="Aperture radius in pixels.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for the output MP4.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Dots per inch for the output MP4.",
    )
    parser.add_argument(
        "--do-photometry",
        action="store_true",
        help="Force photometry recomputation even if cache CSV exists.",
    )
    parser.add_argument(
        "--use-cutout",
        dest="use_cutout",
        action="store_true",
        default=True,
        help="Use IRSA cutouts for downloads (default: True).",
    )
    parser.add_argument(
        "--no-use-cutout",
        dest="use_cutout",
        action="store_false",
        help="Disable cutouts; download full MEF from AWS mirror.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=99999,
        help="Maximum number of images to process.",
    )
    parser.add_argument(
        "--max-missing-fraction",
        type=float,
        default=0.05,
        help="Maximum missing cached fraction before recomputing.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root directory for cached TIC motion results.",
    )
    parser.add_argument(
        "--overwrite-coords",
        action="store_true",
        help="Re-query TIC/Gaia/SPHEREx coordinate caches.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the animation interactively.",
    )
    return parser.parse_args()


def main() -> None:
    """Resolve TIC coordinates, fetch photometry, and write the animation."""
    args = _parse_args()
    tic_id = normalize_tic_id(args.tic_id)
    star_id = f"TIC_{tic_id}"
    ra_deg, dec_deg = get_mean_spherex_coords(
        tic_id,
        results_dir=args.results_dir,
        overwrite=args.overwrite_coords,
    )
    print(
        f"Using mean coordinates for {star_id}: "
        f"RA={ra_deg:.6f}, Dec={dec_deg:.6f}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out)
        if args.out
        else output_dir / f"{star_id}_cutout_anim.mp4"
    )

    records = get_sx_spectrum(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        star_id=star_id,
        output_dir=output_dir,
        do_photometry=args.do_photometry,
        max_images=args.max_images,
        max_missing_fraction=args.max_missing_fraction,
        use_cutout=args.use_cutout,
        size_pix=args.size_pix,
        save_plot=False,
        show_cutout=False,
        save_csv=True,
    )

    if records is None or records.empty:
        print("No photometry records available for animation.")
        return

    title = f"{star_id} RA={ra_deg:.4f}, Dec={dec_deg:.4f}"
    animate_cutouts_from_records(
        records=records,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        out_path=out_path,
        size_pix=args.size_pix,
        aperture_radius_pix=args.aperture_radius,
        fps=args.fps,
        dpi=args.dpi,
        title=title,
        show=args.show,
    )


if __name__ == "__main__":
    main()
