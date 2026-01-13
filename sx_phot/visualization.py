"""Plotting helpers for SPHEREx spectra and spline fits."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from aesthetic.plot import savefig, set_style


def _coerce_mask(mask: Optional[np.ndarray], n_points: int) -> np.ndarray:
    """Coerce a mask array to boolean values.

    Args:
        mask: Optional array-like input mask.
        n_points: Expected length of the mask.

    Returns:
        Boolean array mask with length n_points.
    """
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


def _apply_plot_style() -> None:
    """Apply the preferred matplotlib style with a safe fallback."""
    try:
        set_style("science")
    except OSError:
        set_style("default")


def plot_spectrum_with_spline(
    wavelength_um: np.ndarray,
    flux_jy: np.ndarray,
    flux_err_jy: np.ndarray,
    *,
    masked: Optional[np.ndarray] = None,
    fit_mask: Optional[np.ndarray] = None,
    model_flux_jy: Optional[np.ndarray] = None,
    dense_wavelength_um: Optional[np.ndarray] = None,
    dense_model_flux_jy: Optional[np.ndarray] = None,
    knot_wavelength_um: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    star_id: Optional[str] = None,
    ra_deg: Optional[float] = None,
    dec_deg: Optional[float] = None,
    show_residuals: bool = False,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a SPHEREx spectrum with an optional spline model overlay.

    Args:
        wavelength_um: Wavelengths (microns).
        flux_jy: Fluxes (Jy).
        flux_err_jy: Flux uncertainties (Jy).
        masked: Optional mask; True values are excluded from the main points.
        fit_mask: Optional mask; True values were retained in the spline fit.
        model_flux_jy: Optional model evaluated at input wavelengths.
        dense_wavelength_um: Optional dense wavelength grid for model overlay.
        dense_model_flux_jy: Optional model on the dense wavelength grid.
        knot_wavelength_um: Optional knot wavelengths to annotate.
        title: Optional plot title.
        star_id: Optional target name for title formatting.
        ra_deg: Optional RA in degrees for title formatting.
        dec_deg: Optional Dec in degrees for title formatting.
        show_residuals: If True, include residual and normalized residual panels.
        output_path: Optional path to save the figure.

    Returns:
        (figure, axis) handle for the plot.
    """
    wavelength_um = np.asarray(wavelength_um, dtype=float)
    flux_jy = np.asarray(flux_jy, dtype=float)
    flux_err_jy = np.asarray(flux_err_jy, dtype=float)

    n_points = wavelength_um.size
    if flux_jy.size != n_points or flux_err_jy.size != n_points:
        raise ValueError("Input arrays must have matching lengths.")

    mask = _coerce_mask(masked, n_points)
    finite = (
        np.isfinite(wavelength_um)
        & np.isfinite(flux_jy)
        & np.isfinite(flux_err_jy)
    )
    mask = mask | (~finite)
    use = ~mask
    if fit_mask is None:
        fit_used = use.copy()
    else:
        fit_used = _coerce_mask(fit_mask, n_points) & use

    model_flux = None
    if model_flux_jy is not None:
        model_flux = np.asarray(model_flux_jy, dtype=float)
        if model_flux.size != n_points:
            raise ValueError("model_flux_jy must match wavelength length.")

    if show_residuals and model_flux is None:
        raise ValueError("show_residuals requires model_flux_jy.")

    clipped = np.zeros(n_points, dtype=bool)
    if fit_mask is not None:
        clipped = use & (~fit_used)
        if model_flux is not None:
            clipped &= np.isfinite(model_flux)

    _apply_plot_style()
    if show_residuals:
        fig = plt.figure(figsize=(5, 7))
        grid = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)
        ax_spec = fig.add_subplot(grid[0])
        ax_res = fig.add_subplot(grid[1], sharex=ax_spec)
        ax_res_norm = fig.add_subplot(grid[2], sharex=ax_spec)
    else:
        fig, ax_spec = plt.subplots(figsize=(5, 5))
        ax_res = None
        ax_res_norm = None

    if np.any(mask):
        ax_spec.errorbar(
            wavelength_um[mask],
            flux_jy[mask],
            fmt="x",
            color="r",
            capsize=3,
            alpha=0.3,
            zorder=1,
            label="masked",
        )

    if np.any(clipped):
        ax_spec.scatter(
            wavelength_um[clipped],
            flux_jy[clipped],
            facecolors="none",
            edgecolors="C3",
            marker="o",
            s=32,
            alpha=0.8,
            zorder=2,
            label="clipped",
        )

    if np.any(use):
        ax_spec.errorbar(
            wavelength_um[use],
            flux_jy[use],
            yerr=flux_err_jy[use],
            fmt=".",
            capsize=3,
            c="k",
            zorder=3,
            label="used",
        )

    if dense_wavelength_um is not None and dense_model_flux_jy is not None:
        dense_wavelength_um = np.asarray(dense_wavelength_um, dtype=float)
        dense_model_flux_jy = np.asarray(dense_model_flux_jy, dtype=float)
        finite_dense = np.isfinite(dense_wavelength_um) & np.isfinite(
            dense_model_flux_jy
        )
        if np.any(finite_dense):
            order = np.argsort(dense_wavelength_um[finite_dense])
            ax_spec.plot(
                dense_wavelength_um[finite_dense][order],
                dense_model_flux_jy[finite_dense][order],
                color="C1",
                lw=1.5,
                alpha=0.8,
                zorder=4,
                label="spline",
            )
    elif model_flux is not None:
        finite_model = use & np.isfinite(model_flux)
        if np.any(finite_model):
            order = np.argsort(wavelength_um[finite_model])
            ax_spec.plot(
                wavelength_um[finite_model][order],
                model_flux[finite_model][order],
                color="C1",
                lw=1.5,
                alpha=0.8,
                zorder=4,
                label="spline",
            )

    ax_spec.set_yscale("log")
    finite_flux = use & np.isfinite(flux_jy)
    if np.any(finite_flux):
        y_min = float(np.nanmin(flux_jy[finite_flux]))
        y_max = float(np.nanmax(flux_jy[finite_flux]))
        if y_max > y_min and y_min > 0:
            y_range = y_max - y_min
            margin = 0.05 * y_range
            y_low = y_min - margin
            if y_low <= 0:
                y_low = y_min * 0.9
            y_high = y_max + margin
            ax_spec.set_ylim(y_low, y_high)

    if knot_wavelength_um is not None:
        knots = np.asarray(knot_wavelength_um, dtype=float)
        knots = knots[np.isfinite(knots)]
        if knots.size:
            finite_wave = np.isfinite(wavelength_um)
            if np.any(finite_wave):
                x_min = float(np.nanmin(wavelength_um[finite_wave]))
                x_max = float(np.nanmax(wavelength_um[finite_wave]))
                knots = knots[(knots >= x_min) & (knots <= x_max)]
            knots = np.unique(knots)
        if knots.size:
            y_low, _ = ax_spec.get_ylim()
            y_base = y_low * 1.02 if y_low > 0 else y_low
            ax_spec.plot(
                knots,
                np.full_like(knots, y_base),
                linestyle="None",
                marker="|",
                markersize=8,
                markeredgewidth=1.2,
                color="C2",
                alpha=0.5,
                zorder=2,
                label="knots",
            )
    if show_residuals:
        ax_spec.set_xlabel("")
    else:
        ax_spec.set_xlabel("Wavelength (um)")
    ax_spec.set_ylabel("Flux (Jy)")
    if title is None and star_id and ra_deg is not None and dec_deg is not None:
        safe_id = str(star_id).replace("_", ":")
        title = f"{safe_id} α={ra_deg:.4f}, δ={dec_deg:.4f}"
    if title:
        ax_spec.set_title(title)
    ax_spec.grid(True, which="both", linestyle="--", alpha=0.5)

    if show_residuals:
        handles, labels = ax_spec.get_legend_handles_labels()
        if handles:
            ax_spec.legend(
                handles,
                labels,
                loc="upper right",
                fontsize="xx-small",
                frameon=True,
            )

    if show_residuals and ax_res is not None:
        residuals = flux_jy - model_flux
        res_valid = np.isfinite(residuals) & fit_used & np.isfinite(flux_err_jy)
        res_masked = np.isfinite(residuals) & mask
        res_clipped = np.isfinite(residuals) & clipped

        if np.any(res_masked):
            ax_res.errorbar(
                wavelength_um[res_masked],
                residuals[res_masked],
                fmt="x",
                color="r",
                capsize=3,
                alpha=0.3,
                zorder=1,
            )

        if np.any(res_clipped):
            ax_res.scatter(
                wavelength_um[res_clipped],
                residuals[res_clipped],
                facecolors="none",
                edgecolors="C3",
                marker="o",
                s=32,
                alpha=0.8,
                zorder=2,
            )

        if np.any(res_valid):
            ax_res.errorbar(
                wavelength_um[res_valid],
                residuals[res_valid],
                yerr=flux_err_jy[res_valid],
                fmt=".",
                capsize=3,
                c="k",
                zorder=3,
            )

        if np.any(res_valid):
            amp = float(np.nanmax(np.abs(residuals[res_valid])))
            if amp > 0:
                ax_res.set_ylim(-1.1 * amp, 1.1 * amp)
        ax_res.axhline(0.0, color="0.5", linewidth=0.8, zorder=0)
        ax_res.set_ylabel("Flux - Spline (Jy)")
        ax_res.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.setp(ax_spec.get_xticklabels(), visible=False)

    if show_residuals and ax_res_norm is not None:
        residuals = flux_jy - model_flux
        valid_err = np.isfinite(flux_err_jy) & (flux_err_jy > 0)
        norm = np.full_like(residuals, np.nan)
        norm[valid_err] = residuals[valid_err] / flux_err_jy[valid_err]

        norm_valid = np.isfinite(norm) & fit_used
        norm_masked = np.isfinite(norm) & mask
        norm_clipped = np.isfinite(norm) & clipped

        if np.any(norm_masked):
            ax_res_norm.plot(
                wavelength_um[norm_masked],
                norm[norm_masked],
                linestyle="None",
                marker="x",
                color="r",
                alpha=0.3,
                zorder=1,
            )

        if np.any(norm_clipped):
            ax_res_norm.scatter(
                wavelength_um[norm_clipped],
                norm[norm_clipped],
                facecolors="none",
                edgecolors="C3",
                marker="o",
                s=32,
                alpha=0.8,
                zorder=2,
            )

        if np.any(norm_valid):
            ax_res_norm.plot(
                wavelength_um[norm_valid],
                norm[norm_valid],
                linestyle="None",
                marker=".",
                color="k",
                zorder=3,
            )

        if np.any(norm_valid):
            amp = float(np.nanmax(np.abs(norm[norm_valid])))
            if amp > 0:
                ax_res_norm.set_ylim(-1.1 * amp, 1.1 * amp)
        ax_res_norm.axhline(0.0, color="0.5", linewidth=0.8, zorder=0)
        ax_res_norm.set_ylabel("Residual / err (sigma)")
        ax_res_norm.set_xlabel("Wavelength (um)")
        ax_res_norm.grid(True, which="both", linestyle="--", alpha=0.4)
        if ax_res is not None:
            plt.setp(ax_res.get_xticklabels(), visible=False)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        savefig(fig, str(output_path), writepdf=0)

    return fig, ax_spec


__all__ = ["plot_spectrum_with_spline"]
