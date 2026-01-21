#!/Users/luke/local/miniconda3/envs/py311_sx/bin/python
"""Plot phase-folded residuals per SPHEREx band with phase binning.

This script splits a supplemented cache CSV into the six SPHEREx bands and
creates one normalized residual plot per band. Each plot includes individual
measurements plus phase-binned medians with MAD-scaled error bars.

Options:
    --csv PATH         Input supplemented cache CSV.
    --t0_tjd FLOAT     Reference epoch in TJD.
    --period_days FLOAT
                       Period in days.
    --bins INT         Number of phase bins per period (default: 50).
    --ylim LOW,HIGH    Optional y-limits (use 'auto' or 'none' for defaults).
    --out PATH         Output directory or base PNG path; a band suffix is added.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from aesthetic.plot import set_style

from sx_phot.splinefit import spherex_band_definitions


DEFAULT_CSV = (
    "test_results/"
    "sxphot_cache_TIC_300651846_ra114p36670927895_-66p75737858669_annulus_splsupp.csv"
)
TJD_OFFSET = 56999.5


def _parse_ylim(text: Optional[str]) -> Optional[Tuple[float, float]]:
    """Parse y-limits from a CLI string."""
    if text is None:
        return None
    stripped = text.strip()
    if stripped.lower() in {"none", "auto"}:
        return None
    if (
        len(stripped) >= 2
        and stripped[0] in {"[", "("}
        and stripped[-1] in {"]", ")"}
    ):
        stripped = stripped[1:-1].strip()
    parts = [p for p in stripped.replace(",", " ").split() if p]
    if len(parts) != 2:
        raise ValueError(
            "ylim must be two comma-separated numbers, e.g. '1e-3,1e-1'"
        )
    return float(parts[0]), float(parts[1])


def _parse_title_from_filename(path: Path) -> Optional[str]:
    """Extract a plot title from the CSV filename stub."""
    name = path.stem
    if name.startswith("sxphot_cache_"):
        name = name[len("sxphot_cache_") :]
    if name.endswith("_splsupp"):
        name = name[: -len("_splsupp")]
    for tag in ("_annulus", "_zodi"):
        if tag in name:
            name = name.split(tag, 1)[0]

    if "_ra" not in name:
        return None

    star_id, ra_dec = name.split("_ra", 1)
    ra_str, dec_str = (ra_dec.split("_", 1) + [""])[:2]

    try:
        ra = float(ra_str.replace("p", "."))
        dec = float(dec_str.replace("p", "."))
    except ValueError:
        return star_id.replace("_", ":")

    return f"{star_id.replace('_',':')} RA={ra:.2f}, Dec={dec:.2f}"


def _default_out_path(csv_path: Path, band_name: str) -> Path:
    """Build a default band-specific output PNG path."""
    stem = csv_path.stem
    if stem.startswith("sxphot_cache_"):
        stem = stem[len("sxphot_cache_") :]
    if stem.endswith("_splsupp"):
        stem = stem[: -len("_splsupp")]
    return csv_path.parent / f"{stem}_{band_name}_phase_residuals.png"


def _resolve_out_path(
    out_arg: Optional[str],
    csv_path: Path,
    band_name: str,
) -> Path:
    """Resolve the band-specific output path from the CLI argument."""
    if not out_arg:
        return _default_out_path(csv_path, band_name)

    out_path = Path(out_arg)
    if out_path.suffix.lower() == ".png":
        return out_path.with_name(
            f"{out_path.stem}_{band_name}{out_path.suffix}"
        )

    out_path.mkdir(parents=True, exist_ok=True)
    return out_path / _default_out_path(csv_path, band_name).name


def _normalize_ylim_argv(argv: list[str]) -> list[str]:
    """Normalize --ylim to accept comma-separated values with negatives."""
    argv = list(argv)
    for idx, token in enumerate(argv):
        if token == "--ylim" and idx + 1 < len(argv):
            val = argv[idx + 1]
            if "," in val:
                argv[idx] = f"--ylim={val}"
                del argv[idx + 1]
            break
    return argv


def _apply_plot_style() -> None:
    """Apply the preferred matplotlib style with a fallback."""
    try:
        set_style("science")
    except OSError:
        set_style("default")


def _coerce_mask(mask: Optional[np.ndarray], n_points: int) -> np.ndarray:
    """Coerce a mask array to boolean values."""
    if mask is None:
        return np.zeros(n_points, dtype=bool)

    arr = np.asarray(mask)
    if arr.shape[0] != n_points:
        raise ValueError(
            f"mask length {arr.shape[0]} does not match expected {n_points}."
        )

    if arr.dtype == bool:
        return arr

    if np.issubdtype(arr.dtype, np.number):
        return arr != 0

    text = np.char.lower(arr.astype(str))
    return np.isin(text, ["1", "true", "t", "yes", "y"])


def _bin_phase_median(
    phase: np.ndarray,
    values: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute median and robust SEM per phase bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(n_bins, np.nan)
    errs = np.full(n_bins, np.nan)

    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (phase >= edges[i]) & (phase <= edges[i + 1])
        else:
            in_bin = (phase >= edges[i]) & (phase < edges[i + 1])
        if not np.any(in_bin):
            continue
        vals = values[in_bin]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        medians[i] = med
        errs[i] = 1.418 * mad / np.sqrt(vals.size)

    return centers, medians, errs


def phase_magseries(
    phase: np.ndarray,
    *series: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Wrap phase series around 0.0 by duplicating data at phase-1.

    Args:
        phase: Phase values in [0, 1).
        *series: Data arrays aligned with ``phase``.

    Returns:
        Tuple of wrapped (phase, series...) arrays.
    """
    phase = np.asarray(phase)
    wrapped_phase = np.concatenate((phase - 1.0, phase))
    wrapped_series = [np.concatenate((arr, arr)) for arr in series]
    return (wrapped_phase, *wrapped_series)


def plot_phase_bin_band_residuals(
    csv_path: Path,
    t0_tjd: float,
    period_days: float,
    n_bins: int = 50,
    ylim: Optional[Tuple[float, float]] = None,
    out_arg: Optional[str] = None,
) -> None:
    """Plot phase-folded residuals for each SPHEREx band.

    Args:
        csv_path: Path to a supplemented cache CSV with wavelength_um.
        t0_tjd: Reference epoch in TESS Julian Date.
        period_days: Orbital period in days.
        n_bins: Number of phase bins per period.
        ylim: Optional y-axis limits for normalized residuals.
        out_arg: Output directory or base PNG path.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(csv_path)

    required = {"flux_jy", "model_flux", "flux_err_jy", "wavelength_um"}
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if "tjd_avg" in df.columns:
        tjd = df["tjd_avg"].to_numpy()
    elif "mjd_avg" in df.columns:
        tjd = df["mjd_avg"].to_numpy() - TJD_OFFSET
    else:
        raise ValueError("Missing required time column: tjd_avg or mjd_avg.")

    lam = df["wavelength_um"].to_numpy()
    flux = df["flux_jy"].to_numpy()
    model_flux = df["model_flux"].to_numpy()
    err = df["flux_err_jy"].to_numpy()
    residuals = flux - model_flux

    phase = np.mod((tjd - t0_tjd) / period_days, 1.0)
    phase_finite = np.isfinite(phase)

    mask = _coerce_mask(df["masked"] if "masked" in df.columns else None, phase.size)
    fit_mask = (
        _coerce_mask(df["fit_mask"], phase.size)
        if "fit_mask" in df.columns
        else None
    )
    valid_model = np.isfinite(model_flux) & (model_flux != 0)
    norm = np.full_like(residuals, np.nan)
    norm[valid_model] = residuals[valid_model] / model_flux[valid_model]
    norm_err = np.full_like(err, np.nan)
    norm_err[valid_model] = err[valid_model] / model_flux[valid_model]

    finite_norm = np.isfinite(norm)
    mask = mask | (~finite_norm)
    use = ~mask
    fit_used = use if fit_mask is None else fit_mask & use
    clipped = use & (~fit_used)
    clipped &= np.isfinite(model_flux)

    res_valid = phase_finite & finite_norm & fit_used & np.isfinite(norm_err)
    res_masked = phase_finite & finite_norm & mask
    res_clipped = phase_finite & finite_norm & clipped

    base_title = _parse_title_from_filename(csv_path)
    bands = spherex_band_definitions()
    full_min = min(band["band_min_um"] for band in bands)
    full_max = max(band["band_max_um"] for band in bands)
    full_norm = plt.Normalize(vmin=full_min, vmax=full_max)

    for band in bands:
        band_name = str(band["name"])
        band_min = float(band["band_min_um"])
        band_max = float(band["band_max_um"])
        band_center = 0.5 * (band_min + band_max)
        band_color = plt.cm.magma(full_norm(band_center))
        band_mask = (
            np.isfinite(lam)
            & (lam >= band_min)
            & (lam <= band_max)
        )

        res_valid_band = res_valid & band_mask
        res_masked_band = res_masked & band_mask
        res_clipped_band = res_clipped & band_mask

        if base_title:
            title = f"{base_title} ({band_name})"
        else:
            title = band_name

        binned_centers = None
        binned_medians = None
        binned_errs = None
        offset = 0.0
        if np.any(res_valid_band):
            phase_used = phase[res_valid_band]
            norm_used = norm[res_valid_band]
            binned_centers, binned_medians, binned_errs = _bin_phase_median(
                phase_used,
                norm_used,
                n_bins,
            )
            finite_bins = np.isfinite(binned_medians)
            if np.any(finite_bins):
                offset = float(np.nanmedian(binned_medians[finite_bins]))

        _apply_plot_style()
        fig, ax = plt.subplots(figsize=(5, 5))

        fit_mask_band = (
            res_valid_band
            & np.isfinite(lam)
            & (lam > 1.0)
            & np.isfinite(norm_err)
            & (norm_err > 0)
        )
        if np.count_nonzero(fit_mask_band) >= 2:
            phase_fit = phase[fit_mask_band]
            norm_fit = norm[fit_mask_band]
            w = 1.0 / norm_err[fit_mask_band]
            design = np.column_stack(
                [
                    np.sin(2.0 * np.pi * phase_fit),
                    np.cos(2.0 * np.pi * phase_fit),
                ]
            )
            design_w = design * w[:, None]
            norm_w = norm_fit * w
            try:
                coeffs, _, _, _ = np.linalg.lstsq(design_w, norm_w, rcond=None)
            except np.linalg.LinAlgError:
                coeffs = None
            if coeffs is not None and coeffs.size == 2:
                amp = float(np.hypot(coeffs[0], coeffs[1]))
                phase0 = float(
                    (np.arctan2(-coeffs[1], coeffs[0]) / (2.0 * np.pi)) % 1.0
                )
                model_fit = (
                    coeffs[0] * np.sin(2.0 * np.pi * phase_fit)
                    + coeffs[1] * np.cos(2.0 * np.pi * phase_fit)
                )
                resid = norm_fit - model_fit
                dof = max(norm_fit.size - 2, 0)
                if dof > 0:
                    red_chi2 = float(np.sum((resid / norm_err[fit_mask_band]) ** 2) / dof)
                else:
                    red_chi2 = float("nan")
                print(
                    f"{band_name} sine fit: "
                    f"amp={amp:.6g}, "
                    f"phase0={phase0:.6f}, "
                    f"red_chi2={red_chi2:.3f}, "
                    f"n_fit={norm_fit.size}"
                )
                phase_grid = np.linspace(0.0, 1.0, 512)
                model = (
                    coeffs[0] * np.sin(2.0 * np.pi * phase_grid)
                    + coeffs[1] * np.cos(2.0 * np.pi * phase_grid)
                )
                phase_plot, model_plot = phase_magseries(
                    phase_grid,
                    model - offset,
                )
                ax.plot(
                    phase_plot,
                    model_plot,
                    color="0.4",
                    alpha=0.2,
                    linewidth=2.0,
                    zorder=0,
                    label="sine (>1um)",
                )

        if np.any(res_masked_band):
            phase_masked, norm_masked = phase_magseries(
                phase[res_masked_band],
                norm[res_masked_band] - offset,
            )
            ax.scatter(
                phase_masked,
                norm_masked,
                marker="x",
                color="black",
                s=36,
                linewidths=0.7,
                alpha=0.35,
                zorder=1,
            )
            ax.scatter(
                phase_masked,
                norm_masked,
                marker="x",
                color="r",
                s=25,
                linewidths=0.6,
                alpha=0.35,
                zorder=2,
                label="masked",
            )

        if np.any(res_clipped_band):
            phase_clipped, norm_clipped = phase_magseries(
                phase[res_clipped_band],
                norm[res_clipped_band] - offset,
            )
            ax.scatter(
                phase_clipped,
                norm_clipped,
                facecolors="none",
                edgecolors="black",
                marker="o",
                s=36,
                linewidths=0.6,
                alpha=0.35,
                zorder=2,
            )
            ax.scatter(
                phase_clipped,
                norm_clipped,
                facecolors="none",
                edgecolors="C3",
                marker="o",
                s=32,
                linewidths=0.6,
                alpha=0.35,
                zorder=3,
                label="clipped",
            )

        sc_used = None
        if np.any(res_valid_band):
            phase_valid, norm_valid, err_valid = phase_magseries(
                phase[res_valid_band],
                norm[res_valid_band] - offset,
                norm_err[res_valid_band],
            )
            ax.errorbar(
                phase_valid,
                norm_valid,
                yerr=err_valid,
                fmt="none",
                capsize=3,
                ecolor="0.6",
                elinewidth=1,
                alpha=0.2,
                zorder=2,
            )
            finite_lam = res_valid_band & np.isfinite(lam)
            if np.any(finite_lam):
                phase_used, norm_used, lam_used = phase_magseries(
                    phase[finite_lam],
                    norm[finite_lam] - offset,
                    lam[finite_lam],
                )
                sc_used = ax.scatter(
                    phase_used,
                    norm_used,
                    c=lam_used,
                    cmap=plt.cm.magma,
                    norm=full_norm,
                    s=20,
                    edgecolors="black",
                    linewidths=0.3,
                    alpha=0.35,
                    zorder=3,
                    label="used",
                )
            else:
                phase_used, norm_used = phase_magseries(
                    phase[res_valid_band],
                    norm[res_valid_band] - offset,
                )
                sc_used = ax.scatter(
                    phase_used,
                    norm_used,
                    color="k",
                    s=20,
                    edgecolors="black",
                    linewidths=0.3,
                    alpha=0.35,
                    zorder=3,
                    label="used",
                )

        if binned_centers is not None:
            good_bins = np.isfinite(binned_medians) & np.isfinite(binned_errs)
            if np.any(good_bins):
                centers_plot, medians_plot, errs_plot = phase_magseries(
                    binned_centers[good_bins],
                    binned_medians[good_bins] - offset,
                    binned_errs[good_bins],
                )
                ax.errorbar(
                    centers_plot,
                    medians_plot,
                    yerr=errs_plot,
                    fmt="none",
                    ecolor=band_color,
                    elinewidth=1.2,
                    capsize=3,
                    alpha=1.0,
                    zorder=4,
                )
                ax.scatter(
                    centers_plot,
                    medians_plot,
                    color=band_color,
                    s=45,
                    edgecolors="black",
                    linewidths=0.4,
                    alpha=1.0,
                    zorder=5,
                    label="binned median",
                )

        if not np.any(res_valid_band | res_masked_band | res_clipped_band):
            ax.text(
                0.5,
                0.5,
                "No data in band",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )

        ax.axhline(0.0, color="0.5", linewidth=0.8, zorder=0)
        ax.set_yscale("linear")
        ax.set_xlim(-0.7, 0.7)
        ax.set_xticks([-0.5, 0.0, 0.5])
        ax.set_xlabel("Phase")
        ax.set_ylabel("(Flux - Spline) / Spline minus median")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        ax.set_title(title)
        ax.text(
            0.96,
            0.96,
            f"T0={t0_tjd:.4f} TJD\nP={period_days * 24:.3f} hr",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

        if ylim is not None:
            ax.set_ylim(ylim)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="lower left", fontsize=8)

        if sc_used is not None and sc_used.get_array() is not None:
            cbar = fig.colorbar(sc_used, ax=ax, label="Wavelength (um)")
            cbar.formatter = FormatStrFormatter("%.1f")
            cbar.update_ticks()
            cbar.ax.hlines(
                band_center,
                0.0,
                1.0,
                colors="black",
                linewidth=0.8,
                alpha=0.6,
            )

        fig.tight_layout()

        out_path = _resolve_out_path(out_arg, csv_path, band_name)
        fig.savefig(out_path, dpi=300)
        print(f"Wrote band residuals plot to {out_path}")
        plt.close(fig)


def main() -> None:
    """CLI entry point for plotting per-band residuals."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-band phase-folded residuals from a supplemented "
            "SPHEREx cache CSV (T0 in TJD)."
        ),
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to cache CSV.")
    parser.add_argument(
        "--t0",
        type=float,
        required=True,
        help="Reference epoch T0 in TESS Julian Date (TJD).",
    )
    period_group = parser.add_mutually_exclusive_group(required=True)
    period_group.add_argument("--period_hr", type=float, help="Period in hours.")
    period_group.add_argument("--period_days", type=float, help="Period in days.")
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of phase bins per period.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output directory or base PNG path; band suffixes are added. "
            "Defaults to the CSV directory."
        ),
    )
    parser.add_argument(
        "--ylim",
        default=None,
        type=str,
        help="Comma-separated y-limits, e.g. '1e-3,1e-1' (use 'auto' for default).",
    )

    args = parser.parse_args(_normalize_ylim_argv(sys.argv[1:]))
    ylim = _parse_ylim(args.ylim)
    if args.period_days is not None:
        period_days = args.period_days
    else:
        period_days = args.period_hr / 24

    csv_path = Path(args.csv)

    plot_phase_bin_band_residuals(
        csv_path=csv_path,
        t0_tjd=float(args.t0),
        period_days=float(period_days),
        n_bins=int(args.bins),
        ylim=ylim,
        out_arg=args.out,
    )


if __name__ == "__main__":
    main()
