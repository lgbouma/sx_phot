"""Utilities for wavelength-ordered cutout animations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

from aesthetic.plot import set_style


@dataclass(frozen=True)
class CutoutMetadata:
    """Lightweight metadata for one cutout frame.

    Args:
        wavelength_um: Wavelength in microns.
        file_path: Path to the source FITS file.
        mjd_avg: Optional mean MJD timestamp for the exposure.
    """

    wavelength_um: float
    file_path: Path
    mjd_avg: Optional[float] = None


@dataclass(frozen=True)
class CutoutFrame:
    """Prepared cutout frame with pixel coordinates for annotation.

    Args:
        wavelength_um: Wavelength in microns.
        file_path: Path to the source FITS file.
        image: 2D cutout image.
        xpix: Target x coordinate in cutout pixels.
        ypix: Target y coordinate in cutout pixels.
        dx_pix: Offset from cutout center in pixels along x.
        dy_pix: Offset from cutout center in pixels along y.
    """

    wavelength_um: float
    file_path: Path
    image: np.ndarray
    xpix: float
    ypix: float
    dx_pix: float
    dy_pix: float


def extract_cutout_metadata(records: pd.DataFrame) -> List[CutoutMetadata]:
    """Extract and sort cutout metadata from a photometry records table.

    Args:
        records: DataFrame with at least ``wavelength_um`` and ``file`` columns.

    Returns:
        List of CutoutMetadata sorted in wavelength order.

    Raises:
        ValueError: If required columns are missing.
    """
    if "wavelength_um" not in records.columns or "file" not in records.columns:
        raise ValueError("records must include 'wavelength_um' and 'file' columns.")

    mjd_col = "mjd_avg" if "mjd_avg" in records.columns else None
    metadata: List[CutoutMetadata] = []

    for _, row in records.iterrows():
        try:
            wave = float(row["wavelength_um"])
        except (TypeError, ValueError):
            wave = np.nan
        file_val = row["file"]
        file_str = str(file_val).strip() if file_val is not None else ""
        if not np.isfinite(wave) or not file_str:
            continue
        mjd_avg = None
        if mjd_col is not None:
            mjd_val = row[mjd_col]
            try:
                mjd_val = float(mjd_val)
            except (TypeError, ValueError):
                mjd_val = np.nan
            if np.isfinite(mjd_val):
                mjd_avg = float(mjd_val)
        metadata.append(
            CutoutMetadata(
                wavelength_um=wave,
                file_path=Path(file_str),
                mjd_avg=mjd_avg,
            )
        )

    metadata.sort(key=lambda item: item.wavelength_um)
    return metadata


def build_cutout_frames(
    metadata: Sequence[CutoutMetadata],
    ra_deg: float,
    dec_deg: float,
    *,
    size_pix: int = 64,
) -> List[CutoutFrame]:
    """Load FITS cutouts centered on the target and return sorted frames.

    Args:
        metadata: Sequence of CutoutMetadata sorted by wavelength.
        ra_deg: Target RA in degrees.
        dec_deg: Target Dec in degrees.
        size_pix: Cutout size in pixels (square).

    Returns:
        List of CutoutFrame objects in wavelength order.
    """
    skycoord = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg", frame="icrs")
    frames: List[CutoutFrame] = []

    for entry in metadata:
        try:
            image, xpix, ypix = _load_cutout(entry.file_path, skycoord, size_pix)
        except Exception as exc:
            _log(f"Skipping {entry.file_path}: {exc}")
            continue

        ny, nx = image.shape
        x_center = 0.5 * (nx - 1)
        y_center = 0.5 * (ny - 1)
        dx = float(xpix - x_center)
        dy = float(ypix - y_center)

        frames.append(
            CutoutFrame(
                wavelength_um=float(entry.wavelength_um),
                file_path=entry.file_path,
                image=image,
                xpix=float(xpix),
                ypix=float(ypix),
                dx_pix=dx,
                dy_pix=dy,
            )
        )

    return frames


def animate_cutout_frames(
    frames: Sequence[CutoutFrame],
    out_path: Path,
    *,
    aperture_radius_pix: float = 2.0,
    fps: int = 2,
    dpi: int = 200,
    title: Optional[str] = None,
    show: bool = False,
) -> None:
    """Write an MP4 animation of cutout frames.

    Args:
        frames: Sequence of CutoutFrame objects in display order.
        out_path: Destination MP4 path.
        aperture_radius_pix: Aperture radius in pixels.
        fps: Frames per second in the output MP4.
        dpi: Output dots per inch.
        title: Optional title for the animation.
        show: If True, display the animation interactively.

    Raises:
        ValueError: If no frames are provided.
    """
    if not frames:
        raise ValueError("No cutout frames supplied for animation.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _apply_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    first = frames[0]
    vmin, vmax = _cutout_display_limits(first.image)
    im = ax.imshow(
        first.image,
        origin="lower",
        cmap="magma",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

    aperture = Circle(
        (first.xpix, first.ypix),
        radius=float(aperture_radius_pix),
        fill=False,
        linewidth=0.6,
        edgecolor="gray",
    )
    ax.add_patch(aperture)
    center_marker = ax.plot(
        [first.xpix],
        [first.ypix],
        marker="+",
        color="white",
        markersize=6,
        markeredgewidth=0.8,
        linestyle="None",
    )[0]

    label_box = {"facecolor": "k", "alpha": 0.35, "pad": 1, "edgecolor": "none"}
    wave_text = ax.text(
        0.03,
        0.97,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="w",
        fontsize=8,
        bbox=label_box,
    )
    index_text = ax.text(
        0.97,
        0.97,
        "",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="w",
        fontsize=7,
        bbox=label_box,
    )
    offset_text = ax.text(
        0.03,
        0.03,
        "",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="w",
        fontsize=7,
        bbox=label_box,
    )

    n_frames = len(frames)

    def _update(idx: int):
        frame = frames[idx]
        vmin, vmax = _cutout_display_limits(frame.image)
        im.set_data(frame.image)
        im.set_clim(vmin, vmax)
        aperture.center = (frame.xpix, frame.ypix)
        center_marker.set_data([frame.xpix], [frame.ypix])
        wave_text.set_text(f"λ={frame.wavelength_um:.2f}μm")
        index_text.set_text(f"{idx + 1}/{n_frames}")
        offset_text.set_text(
            f"dx={frame.dx_pix:+.2f}, dy={frame.dy_pix:+.2f} pix"
        )
        return im, aperture, center_marker, wave_text, index_text, offset_text

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=int(1000 / max(fps, 1)),
        blit=False,
        repeat=True,
    )

    anim.save(out_path, dpi=dpi, writer=animation.FFMpegWriter(fps=fps))
    _log(f"Wrote cutout animation to {out_path}")

    if show:
        plt.show()

    plt.close(fig)


def animate_cutouts_from_records(
    records: pd.DataFrame,
    ra_deg: float,
    dec_deg: float,
    out_path: Path,
    *,
    size_pix: int = 64,
    aperture_radius_pix: float = 2.0,
    fps: int = 2,
    dpi: int = 200,
    title: Optional[str] = None,
    show: bool = False,
) -> List[CutoutFrame]:
    """Build cutouts from photometry records and save an MP4 animation.

    Args:
        records: Photometry records DataFrame from ``get_sx_spectrum``.
        ra_deg: Target RA in degrees.
        dec_deg: Target Dec in degrees.
        out_path: Destination MP4 path.
        size_pix: Cutout size in pixels (square).
        aperture_radius_pix: Aperture radius in pixels.
        fps: Frames per second in the output MP4.
        dpi: Output dots per inch.
        title: Optional title for the animation.
        show: If True, display the animation interactively.

    Returns:
        List of CutoutFrame objects used in the animation.
    """
    metadata = extract_cutout_metadata(records)
    frames = build_cutout_frames(
        metadata,
        ra_deg,
        dec_deg,
        size_pix=size_pix,
    )
    if not frames:
        raise ValueError("No valid cutout frames could be prepared.")

    animate_cutout_frames(
        frames,
        out_path,
        aperture_radius_pix=aperture_radius_pix,
        fps=fps,
        dpi=dpi,
        title=title,
        show=show,
    )
    return frames


def _apply_style() -> None:
    """Apply the preferred matplotlib style with a safe fallback."""
    try:
        set_style("science")
    except OSError:
        set_style("default")


def _cutout_display_limits(data: np.ndarray) -> tuple[float, float]:
    """Compute robust display limits for a cutout image."""
    finite = np.isfinite(data)
    if not np.any(finite):
        return 0.0, 1.0
    lo, hi = np.nanpercentile(data[finite], [5.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.nanmin(data[finite]))
        hi = float(np.nanmax(data[finite]))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def _load_cutout(
    file_path: Path,
    skycoord: SkyCoord,
    size_pix: int,
) -> tuple[np.ndarray, float, float]:
    """Load a cutout centered on the target from a FITS file."""
    with fits.open(file_path, memmap=False) as hdul:
        if "IMAGE" not in hdul:
            raise ValueError("FITS file missing IMAGE extension.")
        image = np.asarray(hdul["IMAGE"].data)
        if image.ndim != 2:
            raise ValueError("IMAGE extension is not 2D.")
        wcs = WCS(hdul["IMAGE"].header)
        xpix, ypix = wcs.world_to_pixel(skycoord)
        cutout = Cutout2D(
            image,
            (xpix, ypix),
            (size_pix, size_pix),
            wcs=wcs,
            mode="partial",
            fill_value=np.nan,
        )
        cutout_wcs = cutout.wcs
        x_cut, y_cut = cutout_wcs.world_to_pixel(skycoord)
        if not np.isfinite(x_cut) or not np.isfinite(y_cut):
            raise ValueError("Invalid cutout WCS for target position.")
        if not np.isfinite(cutout.data).any():
            raise ValueError("Cutout contains no finite pixels.")
        return np.asarray(cutout.data), float(x_cut), float(y_cut)


def _log(message: str) -> None:
    """Print a log message."""
    print(message)
