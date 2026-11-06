"""Matplotlib renderings for sx-viz: SPHEREx cutouts and star reports."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
from aesthetic.plot import set_style  # noqa: E402

set_style("science")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.ticker import (FixedLocator, NullFormatter,  # noqa: E402
                               ScalarFormatter)

LAM_TICKS = [0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]


def _log_lambda_axis(ax) -> None:
    """Log wavelength axis with plain tick labels."""
    ax.set_xscale("log")
    ax.set_xlim(0.72, 5.2)
    ax.xaxis.set_major_locator(FixedLocator(LAM_TICKS))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())

ARCSEC_PER_PIXEL = 6.1


def _figure_png(fig, dpi: int = 110) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_spherex_cutout(
    path: Path,
    x: float,
    y: float,
    aperture_radius: float,
    annulus_r_in: float,
    annulus_r_out: float,
    half_size: int = 20,
    title: Optional[str] = None,
) -> bytes:
    """Render a SPHEREx image cutout with the aperture and annulus.

    Args:
        path: Level-2 FITS file.
        x: Star x pixel (0-based) in the image.
        y: Star y pixel.
        aperture_radius: Aperture radius (pixels).
        annulus_r_in: Inner annulus radius (pixels).
        annulus_r_out: Outer annulus radius (pixels).
        half_size: Cutout half-width in pixels.
        title: Optional title.

    Returns:
        PNG bytes.
    """
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        img = hdul["IMAGE"].data
        flags = hdul["FLAGS"].data if "FLAGS" in hdul else None
        ny, nx = img.shape
        x0 = int(np.clip(round(x) - half_size, 0, nx))
        x1 = int(np.clip(round(x) + half_size + 1, 0, nx))
        y0 = int(np.clip(round(y) - half_size, 0, ny))
        y1 = int(np.clip(round(y) + half_size + 1, 0, ny))
        cut = np.array(img[y0:y1, x0:x1], dtype=float)
        fcut = (np.array(flags[y0:y1, x0:x1]) if flags is not None
                else None)
    finite = np.isfinite(cut)
    if finite.sum() > 10:
        lo, hi = np.nanpercentile(cut[finite], [5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(cut[finite]), np.nanmax(cut[finite])
    else:
        lo, hi = 0.0, 1.0
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    extent = (x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5)
    ax.imshow(cut, origin="lower", cmap="gray", vmin=lo, vmax=hi,
              extent=extent, interpolation="nearest")
    if fcut is not None:
        bad = (fcut & 0b1000011011000111) != 0
        if bad.any():
            overlay = np.ma.masked_where(~bad, bad.astype(float))
            ax.imshow(overlay, origin="lower", cmap="autumn", alpha=0.35,
                      extent=extent, interpolation="nearest", vmin=0, vmax=1)
    for r, c, ls in ((aperture_radius, "C0", "-"), (annulus_r_in, "C1", "--"),
                     (annulus_r_out, "C1", "--")):
        ax.add_patch(Circle((x, y), r, fill=False, color=c, lw=1.0, ls=ls))
    ax.plot([x], [y], "+", color="C0", ms=6, mew=0.8)
    # 1 arcmin scale bar.
    bar = 60.0 / ARCSEC_PER_PIXEL
    ax.plot([x0 + 1, x0 + 1 + bar], [y0 + 1.5, y0 + 1.5], color="w", lw=2)
    ax.text(x0 + 1 + bar / 2, y0 + 3, "1'", color="w", ha="center",
            va="bottom", fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)
    return _figure_png(fig)


def plot_star_report(
    spectrum,
    row: dict,
    binned: dict,
    sed: Optional[dict] = None,
) -> bytes:
    """Render a one-page report: spectrum, SED ratio, time series.

    Args:
        spectrum: Spectrum table (``SpectrumStore.read``).
        row: Table row (dict) with TARS and summary columns.
        binned: Binned spectrum dict (lam, flux, err, model).
        sed: Optional ``Data.sed_fit`` payload (fit + curves).

    Returns:
        PNG bytes.
    """
    lam = spectrum["wavelength_um"].to_numpy(float)
    fl = spectrum["flux_jy"].to_numpy(float) * 1e3
    er = spectrum["flux_err_jy"].to_numpy(float) * 1e3
    mk = spectrum["masked"].to_numpy(bool)
    fit = spectrum["fit_mask"].to_numpy(bool)
    model = spectrum["model_flux"].to_numpy(float) * 1e3
    tjd = spectrum["tjd_avg"].to_numpy(float)
    good = ~mk & np.isfinite(fl)
    clipped = good & ~fit

    fig = plt.figure(figsize=(10, 7.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1, 1], hspace=0.35,
                          wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    ax.errorbar(lam[fit], fl[fit], er[fit], fmt=".", ms=3, color="k",
                ecolor="0.6", elinewidth=0.5, lw=0, label="used", zorder=3)
    if clipped.any():
        ax.plot(lam[clipped], fl[clipped], "x", color="C3", ms=4,
                label="clipped", zorder=2)
    if mk.any():
        ax.plot(lam[mk], fl[mk], ".", color="0.8", ms=3, label="masked",
                zorder=1)
    o = np.argsort(lam)
    okm = np.isfinite(model[o])
    ax.plot(lam[o][okm], model[o][okm], "-", color="C0", lw=1.0,
            label="spline", zorder=4)
    if sed and "curves" in sed:
        styles = {"full": ("C1", "-", "single star"),
                  "blue": ("C2", "--", "blue-anchored"),
                  "binary": ("C4", "-.", "two stars"),
                  "disk": ("C0", (0, (5, 2)), "star + dust")}
        for key, (c, ls, lab) in styles.items():
            cur = sed["curves"].get(key)
            if cur:
                x = np.array(cur["lam"], float)
                y = np.array([v if v is not None else np.nan
                              for v in cur["flux"]], float) * 1e3
                ax.plot(x, y, ls=ls, color=c, lw=1.0, label=lab, alpha=0.9)
    _log_lambda_axis(ax)
    ax.set_yscale("log")
    pos = fl[fit & (fl > 0)]
    if pos.size:
        ax.set_ylim(np.nanpercentile(pos, 1) * 0.5,
                    np.nanpercentile(pos, 99) * 2.0)
    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel("Flux [mJy]")
    ax.legend(fontsize=7, loc="lower left", ncol=3)
    tic = row.get("ticid")
    bits = [f"TIC {tic}"]
    for key, fmt in (("Tmag", "T={:.1f}"), ("teff", "Teff(TARS)={:.0f}"),
                     ("adopted_period", "P={:.3f} d"),
                     ("median_amplitude", "amp={:.2f}%"),
                     ("ruwe", "RUWE={:.2f}")):
        v = row.get(key)
        if v is not None:
            bits.append(fmt.format(v))
    if sed and sed.get("fit"):
        f = sed["fit"]
        if f.get("teff_full") is not None:
            bits.append(f"Teff_fit={f['teff_full']:.0f}")
        if f.get("radius_full") is not None:
            bits.append(f"R={f['radius_full']:.2f}")
        if row.get("red_excess_rel") is not None:
            bits.append(f"red excess(rel)={100 * row['red_excess_rel']:.1f}%")
        if f.get("dbic_binary") is not None:
            bits.append(f"dBIC bin={f['dbic_binary']:.0f}")
        if f.get("dbic_disk") is not None:
            bits.append(f"dust={f['dbic_disk']:.0f}")
    ax.set_title("  ".join(bits), fontsize=9)

    # Ratio to blue-anchored model.
    ax2 = fig.add_subplot(gs[1, :])
    blam = np.array(binned["lam"], float)
    bfl = np.array([v if v is not None else np.nan for v in binned["flux"]],
                   float)
    ber = np.array([v if v is not None else np.nan for v in binned["err"]],
                   float)
    ref = None
    label = "binned / spline"
    if sed and sed.get("binned_models", {}).get("blue"):
        ref = np.array([v if v is not None else np.nan
                        for v in sed["binned_models"]["blue"]], float)
        label = "binned / blue-anchored model"
    elif binned.get("model"):
        ref = np.array([v if v is not None else np.nan
                        for v in binned["model"]], float)
    if ref is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            ax2.errorbar(blam, bfl / ref, ber / ref, fmt="o", ms=3,
                         color="k", ecolor="0.6", elinewidth=0.6, lw=0)
        ax2.axhline(1.0, color="C1", lw=0.8)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = bfl / ref
        r = r[np.isfinite(r)]
        if r.size:
            lo_r, hi_r = np.percentile(r, [1, 99])
            ax2.set_ylim(min(0.6, 0.9 * lo_r), max(1.6, 1.1 * hi_r))
        else:
            ax2.set_ylim(0.6, 1.6)
    _log_lambda_axis(ax2)
    ax2.set_ylabel(label, fontsize=8)
    ax2.set_xlabel("Wavelength [um]")

    # Time series of flux / spline.
    ratio = np.where(np.isfinite(model) & (model > 0), fl / model, np.nan)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(tjd[fit], ratio[fit], ".", ms=3, color="k")
    ax3.axhline(1.0, color="C1", lw=0.8)
    ax3.set_ylim(0.8, 1.2)
    ax3.set_xlabel("TJD")
    ax3.set_ylabel("flux / spline")
    ax4 = fig.add_subplot(gs[2, 1])
    period = row.get("adopted_period")
    if period and period > 0:
        phase = ((tjd - tjd[np.isfinite(tjd)].min()) / period) % 1.0
        ax4.plot(phase[fit], ratio[fit], ".", ms=3, color="k")
        ax4.set_xlabel(f"phase (P = {period:.4f} d, arbitrary epoch)")
    else:
        ax4.text(0.5, 0.5, "no period", ha="center", va="center",
                 transform=ax4.transAxes)
    ax4.axhline(1.0, color="C1", lw=0.8)
    ax4.set_ylim(0.8, 1.2)
    ax4.set_ylabel("flux / spline")
    return _figure_png(fig, dpi=120)


__all__ = ["plot_star_report", "render_spherex_cutout"]
