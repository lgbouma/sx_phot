#!/Users/luke/local/miniconda3/envs/py311_sx/bin/python
"""Run SPHEREx photometry and diagnostics for a single TIC target with a period.

This script:
1) Resolves mean SPHEREx-era coordinates (via get_tic_spherex_coords).
2) Runs supplemented photometry to cache CSVs and plots.
3) Creates a phase residuals plot using the provided period (t0 defaults to 0).
4) Creates a spline-fit plot similar to tests/test_splinefit.py.

Outputs are written to <output_root>/TIC_<id>/.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import socket
import sys
from typing import Tuple

import numpy as np
import pandas as pd

from sx_phot.mp_circphot import get_supplemented_sx_spectrum
from sx_phot.splinefit import fit_spherex_spectrum_bspline
from sx_phot.tic_motion import normalize_tic_id
from sx_phot.visualization import plot_spectrum_with_spline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "ticid_with_period"
DEFAULT_YLIM = (-0.3, 0.2)
DEFAULT_BKGD_METHOD = "annulus"
DEFAULT_T0_TJD = 0.0
USE_CUTOUT = True if socket.gethostname() != "wh3" else False


def _log(message: str) -> None:
    """Log a timestamped message.

    Args:
        message: Message to print.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")


def _add_drivers_to_path() -> None:
    """Ensure the drivers directory is on sys.path for imports."""
    drivers_dir = ROOT / "drivers"
    if str(drivers_dir) not in sys.path:
        sys.path.insert(0, str(drivers_dir))


def _parse_ylim(text: str | None) -> Tuple[float, float]:
    """Parse y-limits from a comma-separated string.

    Args:
        text: Comma-separated limits like "-0.3,0.2".

    Returns:
        Tuple of (low, high) y-limits.

    Raises:
        ValueError: If the string is malformed.
    """
    if text is None:
        return DEFAULT_YLIM
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("ylim must be two comma-separated numbers.")
    return float(parts[0]), float(parts[1])


def _coerce_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a masked column to a boolean mask.

    Args:
        mask: Input mask array.

    Returns:
        Boolean mask array.
    """
    if mask.dtype == bool:
        return mask
    if np.issubdtype(mask.dtype, np.number):
        return mask != 0
    text = np.char.lower(mask.astype(str))
    return np.isin(text, ["1", "true", "t", "yes", "y"])


def _radec_str(ra_deg: float, dec_deg: float) -> str:
    """Format a RA/Dec pair to the cache filename string."""
    ra_str = str(ra_deg).replace(".", "p")
    dec_str = str(dec_deg).replace(".", "p")
    return f"ra{ra_str}_{dec_str}"


def _supplemented_csv_path(
    output_dir: Path,
    star_id: str,
    radecstr: str,
    bkgd_method: str,
) -> Path:
    """Build the supplemented cache path for a target."""
    star_prefix = f"{star_id}_" if star_id else ""
    return output_dir / f"sxphot_cache_{star_prefix}{radecstr}_{bkgd_method}_splsupp.csv"


def _make_spline_plot(
    df: pd.DataFrame,
    output_path: Path,
    *,
    star_id: str,
    ra_deg: float,
    dec_deg: float,
) -> None:
    """Generate a spline-fit diagnostic plot.

    Args:
        df: Supplemented photometry dataframe.
        output_path: Destination PNG path.
        star_id: Target identifier.
        ra_deg: Right ascension (degrees).
        dec_deg: Declination (degrees).
    """
    masked = None
    if "masked" in df.columns:
        masked = _coerce_mask(df["masked"].to_numpy())

    model_flux, fit_mask, dense_wave, dense_model, band_results = (
        fit_spherex_spectrum_bspline(
            wavelength_um=df["wavelength_um"].to_numpy(),
            bandwidth_um=df["bandwidth_um"].to_numpy(),
            flux_jy=df["flux_jy"].to_numpy(),
            flux_err_jy=df["flux_err_jy"].to_numpy(),
            masked=masked,
        )
    )

    knots = np.concatenate(
        [
            res.knot_vector_um
            for res in band_results.values()
            if res.knot_vector_um.size
        ]
    )

    fig, _ = plot_spectrum_with_spline(
        wavelength_um=df["wavelength_um"].to_numpy(),
        flux_jy=df["flux_jy"].to_numpy(),
        flux_err_jy=df["flux_err_jy"].to_numpy(),
        masked=masked,
        fit_mask=fit_mask,
        model_flux_jy=model_flux,
        dense_wavelength_um=dense_wave,
        dense_model_flux_jy=dense_model,
        knot_wavelength_um=knots if knots.size else None,
        star_id=star_id,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        show_residuals=True,
        output_path=output_path,
    )
    fig.clf()


def _write_done_marker(path: Path, message: str) -> None:
    """Write a done marker with a short message."""
    path.write_text(message)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run SPHEREx photometry and diagnostics for a single TIC target."
        )
    )
    parser.add_argument(
        "--tic-id",
        required=True,
        help="TIC identifier, e.g. TIC_300651846 or 300651846.",
    )
    parser.add_argument(
        "--period-days",
        required=True,
        type=float,
        help="Adopted period in days.",
    )
    parser.add_argument(
        "--t0-tjd",
        type=float,
        default=DEFAULT_T0_TJD,
        help="Reference epoch in TJD for phase folding (default: 0).",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root output directory for per-TIC products.",
    )
    parser.add_argument(
        "--ylim",
        default=None,
        help="Comma-separated y-limits for residuals (default: -0.3,0.2).",
    )
    parser.add_argument(
        "--do-photometry",
        action="store_true",
        help="Force recomputation of photometry even if caches exist.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if a done marker or cached result exists.",
    )
    parser.add_argument(
        "--bkgd-method",
        default=DEFAULT_BKGD_METHOD,
        choices=["zodi", "annulus"],
        help="Background subtraction method (default: annulus).",
    )
    parser.add_argument(
        "--photometry-workers",
        type=int,
        default=None,
        help=(
            "Number of processes for photometry (default: half the CPUs)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run the single-target pipeline.

    Returns:
        Exit code (0 on success, 1 on failure).
    """
    args = _parse_args()
    ylim = _parse_ylim(args.ylim)

    try:
        tic_id = normalize_tic_id(args.tic_id)
    except Exception as exc:
        _log(f"Invalid TIC ID: {exc}")
        return 1

    period_days = float(args.period_days)
    if not np.isfinite(period_days) or period_days <= 0:
        _log("Invalid period_days; must be finite and > 0.")
        return 1

    star_id = f"TIC_{tic_id}"
    output_root = Path(args.output_root)
    target_dir = output_root / star_id
    done_marker = target_dir / ".done"

    target_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing:
        if done_marker.exists():
            _log(f"{star_id}: done marker exists, skipping.")
            return 0
        cached = list(target_dir.glob("sxphot_cache_TIC_*_splsupp.csv"))
        if cached:
            _log(f"{star_id}: found cached supplemented CSV, skipping.")
            return 0

    _add_drivers_to_path()
    from get_tic_spherex_coords import get_mean_spherex_coords
    from plot_phase_residuals import plot_phase_residuals

    _log(f"{star_id}: resolving coordinates")
    try:
        ra_deg, dec_deg = get_mean_spherex_coords(
            tic_id,
            results_dir=str(output_root),
        )
    except Exception as exc:
        _log(f"{star_id}: coord lookup failed: {exc}")
        return 1

    radecstr = _radec_str(ra_deg, dec_deg)
    supp_csv = _supplemented_csv_path(
        target_dir, star_id, radecstr, args.bkgd_method
    )

    _log(f"{star_id}: running supplemented photometry")
    try:
        supp_df = get_supplemented_sx_spectrum(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            do_photometry=args.do_photometry,
            bkgd_method=args.bkgd_method,
            photometry_workers=args.photometry_workers,
            output_dir=target_dir,
            star_id=star_id,
            save_plot=True,
            save_csv=True,
            save_supp_csv=True,
            use_supp_cache=not args.do_photometry,
            use_cutout=USE_CUTOUT,
        )
    except Exception as exc:
        _log(f"{star_id}: photometry failed: {exc}")
        return 1

    if supp_df is None or len(supp_df) == 0:
        _log(f"{star_id}: no photometry results")
        return 1

    if not supp_csv.exists():
        try:
            supp_df.to_csv(supp_csv, index=False)
            _log(f"{star_id}: wrote {supp_csv}")
        except Exception as exc:
            _log(f"{star_id}: failed to write {supp_csv}: {exc}")
            return 1

    phase_plot = target_dir / f"{supp_csv.stem}_phase_residuals.png"
    _log(f"{star_id}: plotting phase residuals")
    try:
        plot_phase_residuals(
            csv_path=supp_csv,
            t0_tjd=float(args.t0_tjd),
            period_days=period_days,
            ylim=ylim,
            out_path=phase_plot,
        )
    except Exception as exc:
        _log(f"{star_id}: phase plot failed: {exc}")
        return 1

    spline_plot = target_dir / f"{supp_csv.stem}_spline.png"
    _log(f"{star_id}: plotting spline diagnostics")
    try:
        _make_spline_plot(
            supp_df,
            spline_plot,
            star_id=star_id,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
        )
    except Exception as exc:
        _log(f"{star_id}: spline plot failed: {exc}")
        return 1

    _write_done_marker(done_marker, f"Completed {star_id} at {datetime.now()}\n")
    _log(f"{star_id}: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
