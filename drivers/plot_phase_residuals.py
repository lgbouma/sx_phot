#!/Users/luke/local/miniconda3/envs/py311_sx/bin/python
"""Plot phase-folded normalized residuals from a supplemented cache CSV.

The normalized residuals are defined as (flux_jy - model_flux) / model_flux.
Used points are colored by wavelength (magma), and a sinusoid is fit to points
with wavelength_um > 1.0. The ephemeris reference time (t0) is specified in
TESS Julian Date (TJD).

Options:
    --csv PATH         Input supplemented cache CSV (default: example in test_results).
    --t0_tjd FLOAT     Reference epoch in TJD.
    --period_days FLOAT
                       Period in days.
    --ylim LOW,HIGH    Optional y-limits (use 'auto' or 'none' for defaults).
    --out PATH         Output PNG path (default: alongside CSV).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aesthetic.plot import set_style


DEFAULT_CSV = (
    "test_results/"
    "sxphot_cache_TIC_300651846_ra114p36670927895_-66p75737858669_annulus_splsupp.csv"
)
TJD_OFFSET = 56999.5


def _parse_ylim(text: Optional[str]) -> Optional[Tuple[float, float]]:
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


def _default_out_path(csv_path: Path) -> Path:
    stem = csv_path.stem
    if stem.startswith("sxphot_cache_"):
        stem = stem[len("sxphot_cache_") :]
    if stem.endswith("_splsupp"):
        stem = stem[: -len("_splsupp")]
    return csv_path.parent / f"{stem}_phase_residuals.png"


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
    try:
        set_style("science")
    except OSError:
        set_style("default")


def _coerce_mask(mask: Optional[np.ndarray], n_points: int) -> np.ndarray:
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


def plot_phase_residuals(
    csv_path: Path,
    t0_tjd: float,
    period_days: float,
    ylim: Optional[Tuple[float, float]] = None,
    out_path: Optional[Path] = None,
) -> None:
    """Plot a phase-folded normalized residual light curve from a CSV.

    Args:
        csv_path: Path to a supplemented cache CSV with wavelength_um.
        t0_tjd: Reference epoch in TESS Julian Date.
        period_days: Orbital period in days.
        ylim: Optional y-axis limits for normalized residuals.
        out_path: Output PNG path; if None, no plot is saved.

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
    finite_res = np.isfinite(residuals)
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

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(5, 5))

    fit_mask = (
        res_valid
        & np.isfinite(lam)
        & (lam > 1.0)
        & np.isfinite(norm_err)
        & (norm_err > 0)
    )
    if np.count_nonzero(fit_mask) >= 2:
        phase_fit = phase[fit_mask]
        norm_fit = norm[fit_mask]
        w = 1.0 / norm_err[fit_mask]
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
                red_chi2 = float(np.sum((resid / norm_err[fit_mask]) ** 2) / dof)
            else:
                red_chi2 = float("nan")
            print(
                "Sine fit: "
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
            ax.plot(
                phase_grid,
                model,
                color="0.4",
                alpha=0.2,
                linewidth=2.0,
                zorder=0,
                label="sine (>1um)",
            )

    if np.any(res_masked):
        ax.errorbar(
            phase[res_masked],
            norm[res_masked],
            fmt="x",
            color="r",
            capsize=3,
            alpha=0.3,
            zorder=1,
            label="masked",
        )

    if np.any(res_clipped):
        ax.scatter(
            phase[res_clipped],
            norm[res_clipped],
            facecolors="none",
            edgecolors="C3",
            marker="o",
            s=32,
            alpha=0.8,
            zorder=2,
            label="clipped",
        )

    sc_used = None
    if np.any(res_valid):
        ax.errorbar(
            phase[res_valid],
            norm[res_valid],
            yerr=norm_err[res_valid],
            fmt="none",
            capsize=3,
            ecolor="0.6",
            elinewidth=1,
            alpha=0.7,
            zorder=2,
        )
        finite_lam = res_valid & np.isfinite(lam)
        if np.any(finite_lam):
            vmin = float(np.nanmin(lam[finite_lam]))
            vmax = float(np.nanmax(lam[finite_lam]))
            norm_lam = plt.Normalize(vmin=vmin, vmax=vmax)
            sc_used = ax.scatter(
                phase[finite_lam],
                norm[finite_lam],
                c=lam[finite_lam],
                cmap=plt.cm.magma,
                norm=norm_lam,
                s=20,
                edgecolors="none",
                zorder=3,
                label="used",
            )
        else:
            sc_used = ax.scatter(
                phase[res_valid],
                norm[res_valid],
                color="k",
                s=20,
                edgecolors="none",
                zorder=3,
                label="used",
            )

    ax.axhline(0.0, color="0.5", linewidth=0.8, zorder=0)
    ax.set_yscale("linear")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Phase")
    ax.set_ylabel("(Flux - Spline) / Spline")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    title = _parse_title_from_filename(csv_path)
    if title:
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
        ax.legend(loc="best", fontsize=8)

    if sc_used is not None and sc_used.get_array() is not None:
        fig.colorbar(sc_used, ax=ax, label="Wavelength (um)")

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300)
        print(f"Wrote phase residuals plot to {out_path}")

    plt.close(fig)


def main() -> None:
    """CLI entry point for plotting phase-folded residuals."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot a phase-folded residual light curve from a supplemented "
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
        "--out",
        default=None,
        help="Output plot path (PNG). Defaults to CSV directory with a matching stub.",
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
    out_path = Path(args.out) if args.out else _default_out_path(csv_path)

    plot_phase_residuals(
        csv_path=csv_path,
        t0_tjd=float(args.t0),
        period_days=float(period_days),
        ylim=ylim,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
