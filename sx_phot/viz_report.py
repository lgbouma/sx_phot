"""Matplotlib renderings for sx-viz: SPHEREx cutouts and star reports.

SPHEREx cutouts are resampled from detector pixels to a north-up,
east-left grid at the native pixel scale (``sky_cutout``), so that the
tiles of one star share an orientation whatever the roll angle of each
exposure.  ``rgb_stack`` combines three such cutouts into a colour
image.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Sequence

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
ARCSEC_PER_PIXEL = 6.1
# Level-2 FLAGS bits treated as bad pixels in the cutout overlays.
BAD_FLAG_BITS = 0b1000011011000111
# Model curves of the sed_fit payload: (colour, linestyle, label).  The
# keys that take part in the BIC comparison map to a bic_* column.
MODEL_STYLES = {
    "full": ("C1", "-", "single star"),
    "blue": ("C2", "--", "blue-anchored"),
    "binary": ("C4", "-.", "two stars"),
    "disk": ("C0", (0, (5, 2)), "star + dust"),
}
BIC_KEYS = {"full": "bic_single", "binary": "bic_binary", "disk": "bic_disk"}


def _log_lambda_axis(ax) -> None:
    """Log wavelength axis with plain tick labels."""
    ax.set_xscale("log")
    ax.set_xlim(0.72, 5.2)
    ax.xaxis.set_major_locator(FixedLocator(LAM_TICKS))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


def _figure_png(fig, dpi: int = 110) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def preferred_model(fit: Optional[dict]) -> Optional[str]:
    """Key of the lowest-BIC model among single / two stars / star+dust."""
    if not fit:
        return None
    best, best_bic = None, np.inf
    for key, col in BIC_KEYS.items():
        v = fit.get(col)
        if v is not None and np.isfinite(v) and v < best_bic:
            best, best_bic = key, v
    return best


# ----------------------------------------------------------------------
# Cutouts
# ----------------------------------------------------------------------

def sky_jacobian(wcs, x: float, y: float) -> np.ndarray:
    """Local pixel-to-sky Jacobian at (x, y), in arcsec per pixel.

    Returns the 2x2 matrix ``[[dxi/dx, dxi/dy], [deta/dx, deta/dy]]``
    with xi = dRA cos(Dec) (east) and eta = dDec (north).
    """
    r0, d0 = wcs.all_pix2world(x, y, 0)
    r1, d1 = wcs.all_pix2world(x + 1.0, y, 0)
    r2, d2 = wcs.all_pix2world(x, y + 1.0, 0)
    cosd = np.cos(np.radians(float(d0)))

    def dra(r):
        return ((float(r) - float(r0) + 180.0) % 360.0 - 180.0) * cosd * 3600

    return np.array([[dra(r1), dra(r2)],
                     [(float(d1) - float(d0)) * 3600,
                      (float(d2) - float(d0)) * 3600]])


def north_up_coords(
    wcs,
    x: float,
    y: float,
    half_size: int,
    out_scale: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Input pixel coordinates of a north-up, east-left output grid.

    The output is a (2 half_size + 1)-square whose central pixel is the
    sky position of input pixel (x, y).  Its pixel scale is
    ``out_scale`` (arcsec) or, by default, the input scale at (x, y).

    Returns:
        ``(yin, xin, scale)``: arrays of input pixel coordinates (0-based)
        for every output pixel, and the output pixel scale.
    """
    jac = sky_jacobian(wcs, x, y)
    scale = (float(out_scale) if out_scale
             else float(np.sqrt(abs(np.linalg.det(jac)))))
    n = 2 * half_size + 1
    rows, cols = np.mgrid[0:n, 0:n]
    xi = -(cols - half_size) * scale  # east to the left
    eta = (rows - half_size) * scale  # north up (origin="lower")
    inv = np.linalg.inv(jac)
    dx = inv[0, 0] * xi + inv[0, 1] * eta
    dy = inv[1, 0] * xi + inv[1, 1] * eta
    return y + dy, x + dx, scale


def resample_north_up(
    data: np.ndarray,
    wcs,
    x: float,
    y: float,
    half_size: int,
    out_scale: Optional[float] = None,
    order: int = 0,
    cval: float = np.nan,
) -> tuple[np.ndarray, float]:
    """Resample ``data`` onto a north-up grid centred on pixel (x, y).

    Args:
        data: 2-d image (float).
        wcs: Its astropy WCS.
        x: Centre x pixel (0-based).
        y: Centre y pixel (0-based).
        half_size: Output half-width in pixels.
        out_scale: Output pixel scale in arcsec (default: input scale).
        order: Spline order for ``scipy.ndimage.map_coordinates``.
        cval: Fill value outside the input array.

    Returns:
        ``(image, arcsec_per_pixel)``.
    """
    from scipy.ndimage import map_coordinates

    yin, xin, scale = north_up_coords(wcs, x, y, half_size, out_scale)
    if data.size == 0:
        return np.full(yin.shape, cval, float), scale
    img = map_coordinates(np.asarray(data, float), [yin, xin], order=order,
                          mode="constant", cval=cval)
    return img, scale


def sky_cutout(
    path: Path,
    x: float,
    y: float,
    half_size: int = 20,
) -> tuple[np.ndarray, Optional[np.ndarray], float]:
    """North-up, east-left cutout of a SPHEREx image around (x, y).

    The star lands on the central pixel of the (2 half_size + 1)-square
    output, which has the native pixel scale.  Pixels off the detector
    are NaN (image) or 0 (flags).

    Args:
        path: Level-2 FITS file.
        x: Star x pixel (0-based).
        y: Star y pixel (0-based).
        half_size: Output half-width in pixels.

    Returns:
        ``(image, flags, arcsec_per_pixel)``; flags is None if the file
        has no FLAGS extension.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(path, memmap=True) as hdul:
        hdu = hdul["IMAGE"]
        ny, nx = hdu.data.shape
        wcs = WCS(hdu.header)
        # Load a square big enough for any rotation of the output grid.
        pad = int(np.ceil(half_size * 1.5)) + 2
        x0 = int(np.clip(round(x) - pad, 0, nx))
        x1 = int(np.clip(round(x) + pad + 1, 0, nx))
        y0 = int(np.clip(round(y) - pad, 0, ny))
        y1 = int(np.clip(round(y) + pad + 1, 0, ny))
        sub = np.array(hdu.data[y0:y1, x0:x1], dtype=float)
        fsub = (np.array(hdul["FLAGS"].data[y0:y1, x0:x1])
                if "FLAGS" in hdul else None)
    # Shift the WCS reference pixel so (x - x0, y - y0) indexes ``sub``.
    wcs_sub = wcs.deepcopy()
    wcs_sub.wcs.crpix = wcs.wcs.crpix - np.array([x0, y0])
    img, scale = resample_north_up(sub, wcs_sub, x - x0, y - y0, half_size)
    flags = None
    if fsub is not None:
        f, _ = resample_north_up(fsub.astype(float), wcs_sub, x - x0,
                                 y - y0, half_size, cval=0.0)
        flags = f.astype(np.int64)
    return img, flags, scale


def propagate_position(
    ra: float,
    dec: float,
    pmra: Optional[float],
    pmdec: Optional[float],
    epoch: float,
    ref_epoch: float = 2016.0,
) -> tuple[float, float]:
    """Move (ra, dec) [deg] from ``ref_epoch`` to ``epoch`` with proper
    motion (mas/yr; pmra includes cos dec).  Missing proper motions count
    as zero.
    """
    dt = float(epoch) - float(ref_epoch)
    pmra = 0.0 if pmra is None or not np.isfinite(pmra) else float(pmra)
    pmdec = 0.0 if pmdec is None or not np.isfinite(pmdec) else float(pmdec)
    cosd = max(np.cos(np.radians(dec)), 1e-6)
    return (ra + pmra * dt / 3.6e6 / cosd, dec + pmdec * dt / 3.6e6)


def decimal_year(date_obs: str) -> Optional[float]:
    """Decimal year of a FITS DATE-OBS string (date part only).

    DSS plate headers carry malformed times (e.g. ``06:80:00``), so only
    ``YYYY-MM-DD`` is parsed.  Returns None if that fails.
    """
    from datetime import date

    try:
        y, m, d = (int(v) for v in str(date_obs)[:10].split("-"))
        day = date(y, m, d).timetuple().tm_yday
    except (TypeError, ValueError):
        return None
    days = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
    return y + (day - 0.5) / days


def _stretch(cut: np.ndarray) -> np.ndarray:
    """Percentile-clipped asinh stretch of one channel to [0, 1]."""
    finite = np.isfinite(cut)
    out = np.zeros(cut.shape, float)
    if finite.sum() < 10:
        return out
    lo, hi = np.nanpercentile(cut[finite], [5, 99.7])
    if not np.isfinite(hi) or hi - lo <= 1e-6 * max(1.0, abs(lo)):
        lo, hi = np.nanmin(cut[finite]), np.nanmax(cut[finite])
        if hi - lo <= 1e-6 * max(1.0, abs(lo)):
            return out
    v = np.clip((cut[finite] - lo) / (hi - lo), 0, 1)
    out[finite] = np.arcsinh(8.0 * v) / np.arcsinh(8.0)
    return out


def rgb_stack(cuts: Sequence[np.ndarray]) -> np.ndarray:
    """Combine up to three north-up cutouts (blue, green, red) into RGB.

    Each channel is stretched independently, so the brightest source in
    the field is near white and the colours show relative differences.
    Missing channels (``None``) are left black.
    """
    shape = next(c.shape for c in cuts if c is not None)
    rgb = np.zeros(shape + (3,), float)
    for ch, cut in zip((2, 1, 0), cuts):  # cuts ordered blue, green, red
        if cut is not None:
            rgb[..., ch] = _stretch(cut)
    return rgb


def _draw_apertures(ax, cx: float, cy: float, radii: Sequence[float]) -> None:
    """Aperture (solid) and annulus (dashed) circles around (cx, cy)."""
    for r, c, ls in zip(radii, ("C0", "C1", "C1"), ("-", "--", "--")):
        ax.add_patch(Circle((cx, cy), r, fill=False, color=c, lw=1.0, ls=ls))
    ax.plot([cx], [cy], "+", color="C0", ms=6, mew=0.8)


def _draw_compass_and_bar(ax, n: int, scale: float) -> None:
    """1 arcmin scale bar (bottom left) and N/E arrows (top right)."""
    bar = 60.0 / scale
    ax.plot([1, 1 + bar], [1.5, 1.5], color="w", lw=2)
    ax.text(1 + bar / 2, 3, "1'", color="w", ha="center", va="bottom",
            fontsize=7)
    cx, cy = n - 5.5, n - 5.5
    arm = 3.0
    ax.annotate("", xy=(cx, cy + arm), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="w", lw=0.8))
    ax.annotate("", xy=(cx - arm, cy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="w", lw=0.8))
    ax.text(cx, cy + arm + 0.4, "N", color="w", ha="center", va="bottom",
            fontsize=6)
    ax.text(cx - arm - 0.4, cy, "E", color="w", ha="right", va="center",
            fontsize=6)


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
    """Render a north-up SPHEREx cutout with the aperture and annulus.

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
    cut, fcut, scale = sky_cutout(path, x, y, half_size)
    finite = np.isfinite(cut)
    if finite.sum() > 10:
        lo, hi = np.nanpercentile(cut[finite], [5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = np.nanmin(cut[finite]), np.nanmax(cut[finite])
    else:
        lo, hi = 0.0, 1.0
    n = cut.shape[0]
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.set_facecolor("0.25")
    ax.imshow(cut, origin="lower", cmap="gray", vmin=lo, vmax=hi,
              interpolation="nearest")
    if fcut is not None:
        bad = (fcut & BAD_FLAG_BITS) != 0
        if bad.any():
            overlay = np.ma.masked_where(~bad, bad.astype(float))
            ax.imshow(overlay, origin="lower", cmap="autumn", alpha=0.35,
                      interpolation="nearest", vmin=0, vmax=1)
    _draw_apertures(ax, half_size, half_size,
                    (aperture_radius, annulus_r_in, annulus_r_out))
    _draw_compass_and_bar(ax, n, scale)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)
    return _figure_png(fig)


def render_rgb_cutout(
    rgb: np.ndarray,
    scale: float,
    aperture_radius: float,
    annulus_r_in: float,
    annulus_r_out: float,
    title: Optional[str] = None,
) -> bytes:
    """Render an ``rgb_stack`` image with the aperture and annulus."""
    n = rgb.shape[0]
    half = n // 2
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(rgb, origin="lower", interpolation="nearest")
    _draw_apertures(ax, half, half,
                    (aperture_radius, annulus_r_in, annulus_r_out))
    _draw_compass_and_bar(ax, n, scale)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)
    return _figure_png(fig)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def _sky_axis(ax, title: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7)


def plot_star_report(
    spectrum,
    row: dict,
    binned: dict,
    sed: Optional[dict] = None,
    sky_jpeg: Optional[bytes] = None,
    sky_title: str = "DSS2 red, 3'",
    spx_rgb: Optional[np.ndarray] = None,
    spx_title: str = "SPHEREx stack",
    phot: Optional[dict] = None,
) -> bytes:
    """Render a one-page report: spectrum, ratio, time series, cutouts.

    Args:
        spectrum: Spectrum table (``SpectrumStore.read``).
        row: Table row (dict) with TARS and summary columns.
        binned: Binned spectrum dict (lam, flux, err, model).
        sed: Optional ``Data.sed_fit`` payload (fit + curves).
        sky_jpeg: Optional SkyView JPEG (north up, east left).
        sky_title: Caption of the sky cutout.
        spx_rgb: Optional ``rgb_stack`` array (north up, east left).
        spx_title: Caption of the SPHEREx stack.
        phot: Photometry config (aperture_radius, annulus_r_in/out) for
            the circles on the SPHEREx stack.

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
    gs = fig.add_gridspec(3, 4, height_ratios=[2.2, 1, 1.15], hspace=0.4,
                          wspace=0.3)
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
    best = preferred_model(sed.get("fit") if sed else None)
    if sed and "curves" in sed:
        for key, (c, ls, lab) in MODEL_STYLES.items():
            cur = sed["curves"].get(key)
            if not cur:
                continue
            x = np.array(cur["lam"], float)
            y = np.array([v if v is not None else np.nan
                          for v in cur["flux"]], float) * 1e3
            if key == best:
                ax.plot(x, y, ls="-", color=c, lw=2.2, zorder=6,
                        label=lab + " (preferred)")
            else:
                ax.plot(x, y, ls=ls, color=c, lw=0.9, alpha=0.7, zorder=5,
                        label=lab)
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
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.plot(tjd[fit], ratio[fit], ".", ms=3, color="k")
    ax3.axhline(1.0, color="C1", lw=0.8)
    ax3.set_ylim(0.8, 1.2)
    ax3.set_xlabel("TJD")
    ax3.set_ylabel("flux / spline")

    # Cutouts, both north up and east left.
    ax4 = fig.add_subplot(gs[2, 2])
    if sky_jpeg:
        from PIL import Image

        im = np.asarray(Image.open(io.BytesIO(sky_jpeg)).convert("L"))
        ax4.imshow(im, cmap="gray", origin="upper", interpolation="nearest")
        if phot:
            ny, nx = im.shape
            # SkyView tile: CUTOUT_SIZE_DEG across nx pixels.
            cx, cy = (nx - 1) / 2, (ny - 1) / 2
            px_per_pix = nx / (0.05 * 3600) * ARCSEC_PER_PIXEL
            _draw_apertures(ax4, cx, cy, [px_per_pix * phot[k] for k in
                                          ("aperture_radius", "annulus_r_in",
                                           "annulus_r_out")])
    else:
        ax4.text(0.5, 0.5, "no sky cutout", ha="center", va="center",
                 fontsize=7, transform=ax4.transAxes)
    _sky_axis(ax4, sky_title)
    ax5 = fig.add_subplot(gs[2, 3])
    if spx_rgb is not None:
        ax5.imshow(spx_rgb, origin="lower", interpolation="nearest")
        n = spx_rgb.shape[0]
        if phot:
            _draw_apertures(ax5, n // 2, n // 2,
                            [phot[k] for k in ("aperture_radius",
                                               "annulus_r_in",
                                               "annulus_r_out")])
        _draw_compass_and_bar(ax5, n, ARCSEC_PER_PIXEL)
        ax5.set_xlim(-0.5, n - 0.5)
        ax5.set_ylim(-0.5, n - 0.5)
    else:
        ax5.text(0.5, 0.5, "no SPHEREx images", ha="center", va="center",
                 fontsize=7, transform=ax5.transAxes)
    _sky_axis(ax5, spx_title)
    return _figure_png(fig, dpi=120)


__all__ = ["decimal_year", "north_up_coords", "plot_star_report",
           "preferred_model", "propagate_position", "render_rgb_cutout",
           "render_spherex_cutout", "resample_north_up", "rgb_stack",
           "sky_cutout", "sky_jacobian"]
