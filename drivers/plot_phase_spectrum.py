#!/Users/luke/local/miniconda3/envs/py311_sx/bin/python
"""Plot a SPHEREx spectrum colored by phase from a cached CSV.  Optionally, animate it."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aesthetic.plot import set_style


DEFAULT_CSV = (
    "test_results/"
    "sxphot_cache_TIC_300651846_ra114p36670927895_-66p75737858669_annulus.csv"
)


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
    if not stem.endswith("_mjdc"):
        stem = f"{stem}_mjdc"
    return csv_path.parent / f"{stem}_phase.png"


def _default_anim_path(out_path: Path) -> Path:
    return out_path.with_name(f"{out_path.stem}_anim.mp4")


def _phase_edges(width: float) -> np.ndarray:
    if width <= 0 or width > 1:
        raise ValueError("animatephasewidth must be > 0 and <= 1.")
    edges = np.arange(0.0, 1.0, width)
    if edges.size == 0 or edges[-1] < 1.0:
        edges = np.append(edges, 1.0)
    return edges


def plot_spectrum(
    csv_path: Path,
    ylim: tuple[float, float] | None = None,
    out_path: Path | None = None,
    show: bool = False,
    period: float | None = None,
) -> None:
    df = pd.read_csv(csv_path)

    lam = df["wavelength_um"].to_numpy()
    flux = df["flux_jy"].to_numpy()
    err = df["flux_err_jy"].to_numpy() if "flux_err_jy" in df else None
    mjd = df["mjd_avg"].to_numpy() if "mjd_avg" in df else np.full_like(lam, np.nan)
    mjd -= np.nanmin(mjd)
    phase = mjd / period - np.floor(mjd / period)
    masked = df["masked"].to_numpy() if "masked" in df else np.zeros_like(lam, dtype=bool)

    if masked.dtype == object:
        masked = np.array([str(val).lower() == "true" for val in masked])

    set_style("science")
    fig, ax = plt.subplots(figsize=(5, 5))

    finite_mjd = np.isfinite(mjd)
    if np.any(finite_mjd):
        vmin = float(np.nanmin(phase[finite_mjd]))
        vmax = float(np.nanmax(phase[finite_mjd]))
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.twilight

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

        sc = ax.scatter(
            lam[~masked],
            flux[~masked],
            c=phase[~masked],
            cmap=cmap,
            norm=norm,
            s=20,
            edgecolors="none",
        )
    else:
        sc = ax.scatter(lam[~masked], flux[~masked], color="0.5", s=20, edgecolors="none")

    if np.any(masked):
        ax.errorbar(lam[masked], flux[masked], fmt="x", color="r", capsize=3, zorder=-1)

    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Flux (Jy)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    title = _parse_title_from_filename(csv_path)
    if title:
        ax.set_title(title)
    if period is not None and np.isfinite(period):
        ax.text(
            0.02,
            0.88,
            f"Period: {period * 24:.2f} hr",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    if ylim is not None:
        ax.set_ylim(ylim)

    if np.any(finite_mjd):
        try:
            axpos = ax.get_position()
            cb_width = 0.25 * axpos.width
            cb_height = 0.03 * axpos.height
            cb_left = axpos.x0 + 0.07 * axpos.width
            cb_bottom = axpos.y0 + 0.12 * axpos.height
            cax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])
            cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
            if period is not None and np.isfinite(period):
                cb.set_label(f"Phase (P = {period * 24:.2f} hr)")
            else:
                cb.set_label("Phase")
        except Exception:
            pass

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300)
        print(f"Wrote phase spectrum plot to {out_path}")
    if show:
        plt.show()

    plt.close(fig)


def animate_phase_spectrum(
    csv_path: Path,
    period: float,
    ylim: tuple[float, float] | None = None,
    out_path: Path | None = None,
    show: bool = False,
    phase_width: float = 0.1,
) -> None:
    df = pd.read_csv(csv_path)

    lam = df["wavelength_um"].to_numpy()
    flux = df["flux_jy"].to_numpy()
    mjd = df["mjd_avg"].to_numpy() if "mjd_avg" in df else np.full_like(lam, np.nan)
    mjd -= np.nanmin(mjd)
    phase = mjd / period - np.floor(mjd / period)
    masked = df["masked"].to_numpy() if "masked" in df else np.zeros_like(lam, dtype=bool)

    if masked.dtype == object:
        masked = np.array([str(val).lower() == "true" for val in masked])

    finite_phase = np.isfinite(phase)
    if not np.any(finite_phase):
        return

    phase_edges = _phase_edges(phase_width)
    phase_bins = list(zip(phase_edges[:-1], phase_edges[1:]))
    valid = (~masked) & finite_phase

    set_style("science")
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Flux (Jy)")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    title = _parse_title_from_filename(csv_path)
    if title:
        ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        flux_pos = flux[valid & (flux > 0)]
        if flux_pos.size:
            ax.set_ylim(np.nanmin(flux_pos), np.nanmax(flux_pos))

    if np.any(valid):
        ax.set_xlim(np.nanmin(lam[valid]), np.nanmax(lam[valid]))

    initial_lo, initial_hi = phase_bins[0]
    initial_mask = valid & (phase >= initial_lo) & (phase < initial_hi)
    sc = ax.scatter(
        lam[initial_mask],
        flux[initial_mask],
        color="black",
        s=20,
        edgecolors="none",
    )

    phase_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    fig.tight_layout()

    def update(idx: int):
        lo, hi = phase_bins[idx]
        mask = valid & (phase >= lo) & (phase < hi)
        if np.any(mask):
            sc.set_offsets(np.column_stack((lam[mask], flux[mask])))
        else:
            sc.set_offsets(np.empty((0, 2)))
        phase_text.set_text(f"Phase {lo:.2f}-{hi:.2f}")
        return sc, phase_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(phase_bins),
        interval=500,
        blit=False,
        repeat=True,
    )

    if out_path is not None:
        anim.save(out_path, dpi=300, writer=animation.FFMpegWriter(fps=2))
        print(f"Wrote phase spectrum animation to {out_path}")
    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SX spectrum colored by time from a cache CSV.",
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to cache CSV.")
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
        help="Comma-separated y-limits, e.g. '1e-3,1e-1' (use 'auto' for default).",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    parser.add_argument(
        "--animatephase",
        action="store_true",
        help="Generate an animation stepping through phase bins.",
    )
    parser.add_argument(
        "--animatephasewidth",
        type=float,
        default=0.1,
        help="Phase bin width for animation (default: 0.1).",
    )

    args = parser.parse_args()
    ylim = _parse_ylim(args.ylim)
    if args.period_days is not None:
        period_days = args.period_days
    else:
        period_days = args.period_hr / 24

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else _default_out_path(csv_path)
    plot_spectrum(
        csv_path,
        ylim=ylim,
        out_path=out_path,
        show=args.show,
        period=period_days,
    )
    if args.animatephase:
        animate_phase_spectrum(
            csv_path,
            period=period_days,
            ylim=ylim,
            out_path=_default_anim_path(out_path),
            show=args.show,
            phase_width=args.animatephasewidth,
        )


if __name__ == "__main__":
    main()
