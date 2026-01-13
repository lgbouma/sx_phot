#!/Users/luke/local/miniconda3/envs/vbls/bin/python
"""Plot a SPHEREx spectrum labeled by in/out-of-transit windows.

This driver reads a cached CSV from `sx_phot.circphot`, computes orbital phase
using TESS Julian Date (TJD), and labels each point as "in" or "out" of transit
based on phase windows that are only applied over specific TJD spans.

Example (TIC 300651846):
    python drivers/plot_inout_spectrum.py \
        --csv test_results/sxphot_cache_TIC_300651846_ra114p36670927895_-66p75737858669_annulus.csv \
        --t0 1325.4461206 \
        --period_hr 8.254 \
        --windows "0.255,0.305,3810,3940;0.3875,0.4625,3880,3940"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aesthetic.plot import set_style

DEFAULT_CSV = (
    "test_results/"
    "sxphot_cache_TIC_300651846_ra114p36670927895_-66p75737858669_annulus.csv"
)
TJD_OFFSET = 56999.5


def _parse_ylim(text: str | None) -> tuple[float, float] | None:
    if text is None:
        return None
    if text.strip().lower() in {"none", "auto"}:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ValueError("ylim must be two comma-separated numbers, e.g. '1e-3,1e-1'")
    return float(parts[0]), float(parts[1])


def _parse_title_from_filename(path: Path) -> str | None:
    name = path.stem
    if name.startswith("sxphot_cache_"):
        name = name[len("sxphot_cache_") :]
    if "_annulus" in name:
        name = name.split("_annulus", 1)[0]

    if "_ra" not in name:
        return None

    star_id, ra_dec = name.split("_ra", 1)
    ra_str, dec_str = (ra_dec.split("_", 1) + [""])[:2]

    try:
        ra = float(ra_str.replace("p", "."))
        dec = float(dec_str.replace("p", "."))
    except ValueError:
        return star_id.replace("_", ":")

    return f"{star_id.replace('_',':')} α={ra:.2f}, δ={dec:.2f}"


def _default_out_path(csv_path: Path) -> Path:
    stem = csv_path.stem
    if stem.startswith("sxphot_cache_"):
        stem = stem[len("sxphot_cache_") :]
    return csv_path.parent / f"{stem}_inout.png"


def _default_out_csv_path(csv_path: Path) -> Path:
    stem = csv_path.stem
    if stem.startswith("sxphot_cache_"):
        stem = stem[len("sxphot_cache_") :]
    return csv_path.parent / f"{stem}_inout.csv"


def _parse_windows(spec: str) -> List[Tuple[float, float, float, float]]:
    """Parse phase/TJD windows from a single string.

    Args:
        spec: String formatted as
            "phase_lo,phase_hi,tjd_lo,tjd_hi;phase_lo,phase_hi,tjd_lo,tjd_hi".
            Phase values are in [0, 1). TJD spans are inclusive.
            If phase_lo > phase_hi, the phase window wraps around 1.0.

    Returns:
        List of (phase_lo, phase_hi, tjd_lo, tjd_hi) tuples.

    Raises:
        ValueError: If the spec is empty or any window is malformed.
    """
    if spec is None or not str(spec).strip():
        raise ValueError("windows spec must be a non-empty string")

    windows = []
    for chunk in str(spec).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",") if p.strip()]
        if len(parts) != 4:
            raise ValueError(
                "Each window must have 4 values: phase_lo,phase_hi,tjd_lo,tjd_hi"
            )
        try:
            phase_lo, phase_hi, tjd_lo, tjd_hi = [float(p) for p in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid numeric window: '{chunk}'") from exc
        if not (0.0 <= phase_lo <= 1.0 and 0.0 <= phase_hi <= 1.0):
            raise ValueError("Phase bounds must be between 0 and 1.")
        if tjd_hi < tjd_lo:
            raise ValueError("TJD window must satisfy tjd_lo <= tjd_hi.")
        windows.append((phase_lo, phase_hi, tjd_lo, tjd_hi))

    if not windows:
        raise ValueError("No valid windows found in spec.")
    return windows


def plot_inout_spectrum(
    csv_path: Path,
    t0_tjd: float,
    period_days: float,
    windows: List[Tuple[float, float, float, float]],
    ylim: tuple[float, float] | None = None,
    out_path: Path | None = None,
    out_csv: Path | None = None,
    show: bool = False,
) -> None:
    """Plot SPHEREx spectrum colored by in/out-of-transit windows.

    Args:
        csv_path: Path to a cached CSV from `sx_phot.circphot`.
        t0_tjd: Reference transit epoch in TESS Julian Date.
        period_days: Orbital period in days.
        windows: List of (phase_lo, phase_hi, tjd_lo, tjd_hi) windows.
        ylim: Optional y-axis limits.
        out_path: Output PNG path; if None, no plot is saved.
        out_csv: Output CSV path for labeled data; if None, no CSV is saved.
        show: If True, display the plot interactively.
    """
    df = pd.read_csv(csv_path)

    lam = df["wavelength_um"].to_numpy()
    flux = df["flux_jy"].to_numpy()
    err = df["flux_err_jy"].to_numpy() if "flux_err_jy" in df else None
    mjd = df["mjd_avg"].to_numpy() if "mjd_avg" in df else np.full_like(lam, np.nan)
    tjd = mjd - TJD_OFFSET
    phase = np.mod((tjd - t0_tjd) / period_days, 1.0)
    masked = df["masked"].to_numpy() if "masked" in df else np.zeros_like(lam, dtype=bool)

    if masked.dtype == object:
        masked = np.array([str(val).lower() == "true" for val in masked])

    in_transit = np.zeros_like(phase, dtype=bool)
    finite = np.isfinite(phase) & np.isfinite(tjd)
    for phase_lo, phase_hi, tjd_lo, tjd_hi in windows:
        tjd_sel = (tjd >= tjd_lo) & (tjd <= tjd_hi)
        if phase_lo <= phase_hi:
            phase_sel = (phase >= phase_lo) & (phase <= phase_hi)
        else:
            phase_sel = (phase >= phase_lo) | (phase <= phase_hi)
        in_transit |= finite & tjd_sel & phase_sel

    out_transit = ~in_transit

    set_style("science")
    fig, ax = plt.subplots(figsize=(5, 5))

    if err is not None:
        ax.errorbar(
            lam[~masked],
            flux[~masked],
            yerr=err[~masked],
            fmt="none",
            ecolor="0.6",
            elinewidth=1,
            capsize=2,
            alpha=0.7,
            zorder=-1,
        )

    in_color = "#e4572e"
    out_color = "#2a4f8f"

    in_unmasked = in_transit & ~masked
    out_unmasked = out_transit & ~masked
    ax.scatter(
        lam[out_unmasked],
        flux[out_unmasked],
        color=out_color,
        s=24,
        edgecolors="none",
        label="Out of transit",
    )
    ax.scatter(
        lam[in_unmasked],
        flux[in_unmasked],
        color=in_color,
        s=24,
        edgecolors="none",
        label="In transit",
    )

    in_masked = in_transit & masked
    out_masked = out_transit & masked
    if np.any(out_masked):
        ax.scatter(
            lam[out_masked],
            flux[out_masked],
            color=out_color,
            marker="x",
            s=36,
            label="Out of transit (masked)",
        )
    if np.any(in_masked):
        ax.scatter(
            lam[in_masked],
            flux[in_masked],
            color=in_color,
            marker="x",
            s=36,
            label="In transit (masked)",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Flux (Jy)")
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

    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300)
    if show:
        plt.show()

    plt.close(fig)

    if out_csv is not None:
        df_out = df.copy()
        df_out["tjd_avg"] = tjd
        df_out["phase"] = phase
        df_out["in_transit"] = in_transit
        df_out.to_csv(out_csv, index=False)


def main() -> None:
    """CLI entry point for plotting in/out-of-transit spectra."""
    parser = argparse.ArgumentParser(
        description="Plot SX spectrum labeled by in/out-of-transit windows (TJD).",
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to cache CSV.")
    parser.add_argument("--t0", type=float, required=True, help="Transit epoch T0 in TJD.")
    period_group = parser.add_mutually_exclusive_group(required=True)
    period_group.add_argument("--period_hr", type=float, help="Period in hours.")
    period_group.add_argument("--period_days", type=float, help="Period in days.")
    parser.add_argument(
        "--windows",
        required=True,
        help=(
            "Phase/TJD windows: "
            "'phase_lo,phase_hi,tjd_lo,tjd_hi;phase_lo,phase_hi,tjd_lo,tjd_hi'"
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output plot path (PNG). Defaults to CSV directory with a matching stub.",
    )
    parser.add_argument(
        "--outcsv",
        default=None,
        help="Output CSV path with in_transit labels (default: <stem>_inout.csv).",
    )
    parser.add_argument(
        "--ylim",
        default=None,
        help="Comma-separated y-limits, e.g. '1e-3,1e-1' (use 'auto' for default).",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")

    args = parser.parse_args()
    ylim = _parse_ylim(args.ylim)
    if args.period_days is not None:
        period_days = args.period_days
    else:
        period_days = args.period_hr / 24

    windows = _parse_windows(args.windows)
    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else _default_out_path(csv_path)
    out_csv = Path(args.outcsv) if args.outcsv else _default_out_csv_path(csv_path)

    plot_inout_spectrum(
        csv_path,
        t0_tjd=float(args.t0),
        period_days=float(period_days),
        windows=windows,
        ylim=ylim,
        out_path=out_path,
        out_csv=out_csv,
        show=args.show,
    )


if __name__ == "__main__":
    main()
