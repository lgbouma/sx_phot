"""
Circular aperture photometry for SPHEREx Level-2 MEF images.

This module exposes a callable API:

    get_sx_spectrum(ra_deg, dec_deg, **kwargs) -> pandas.DataFrame
    get_supplemented_sx_spectrum(ra_deg, dec_deg, **kwargs) -> pandas.DataFrame

Summary
- Queries the IRSA TAP service for SPHEREx observations overlapping a
  target (ra_deg, dec_deg).
- Downloads the primary spectral image MEF FITS (via the public S3
  mirror), or an IRSA cutout (default), caches them on disk, and
  performs aperture photometry at the target pixel.
- Saves a spectrum plot PNG and a CSV cache of per-image photometry.

Background methods
- bkgd_method = 'zodi': subtract the provided ZODI background image
  (default).
- bkgd_method = 'annulus': estimate a local background as the
  sigma‑clipped median within a circular annulus
  [annulus_r_in, annulus_r_out] in pixels, masking stars in the
  annulus via sigclip.

Outputs
- result_<radec>.png in the current directory.
- sxphot_cache_<radec>.csv in the current directory.
- sxphot_url_cache_<radec>.csv for cached datalink URL resolution.
- sxphot_cache_<radec>_splsupp.csv for supplemented spline diagnostics.

Notes
- Aperture area is converted from MJy/sr to Jy using the pixel solid
  angle.
"""

import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

import os
os.environ["MPLBACKEND"] = "Agg"
import re
import multiprocessing as mp
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import pandas as pd
import pyvo
from pyvo.dal.adhoc import DatalinkResults
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
from pathlib import Path
from datetime import datetime
from typing import Optional, Sequence, Union
from collections import Counter

from sx_phot.splinefit import fit_spherex_spectrum_bspline
from sx_phot.visualization import _add_cutout_inset

def log(message: str) -> None:
    """Log a timestamped message.

    Args:
        message: Message to print.
    """
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {message}")


def _missing_fraction(cache_len: int, expected_len: int) -> float:
    """Return the fraction of expected records missing from the cache.

    Args:
        cache_len: Number of cached records.
        expected_len: Expected number of records.

    Returns:
        Fraction of missing records in the cache.
    """
    if expected_len <= 0:
        return 0.0
    if cache_len >= expected_len:
        return 0.0
    return (expected_len - cache_len) / expected_len


def _is_cache_complete(
    cache_len: int,
    expected_len: int,
    max_missing_fraction: float,
) -> bool:
    """Return True if the cache is within the missing fraction threshold.

    Args:
        cache_len: Number of cached records.
        expected_len: Expected number of records.
        max_missing_fraction: Maximum fraction of missing records to allow.

    Returns:
        True if the cache is complete enough to reuse.
    """
    missing_frac = _missing_fraction(cache_len, expected_len)
    return missing_frac <= max_missing_fraction


def _pick_cutout_index(
    wavelength_um: np.ndarray,
    target_wavelength_um: float,
) -> Optional[int]:
    """Return the index of the wavelength closest to a target value.

    Args:
        wavelength_um: Wavelength array in microns.
        target_wavelength_um: Target wavelength in microns.

    Returns:
        Index of the closest wavelength, or None if no finite values exist.
    """
    wavelength_um = np.asarray(wavelength_um, dtype=float)
    if wavelength_um.size == 0:
        return None
    finite = np.isfinite(wavelength_um)
    if not np.any(finite):
        return None
    diffs = np.full_like(wavelength_um, np.inf, dtype=float)
    diffs[finite] = np.abs(wavelength_um[finite] - target_wavelength_um)
    return int(np.argmin(diffs))


def _center_cutout_for_display(
    image: np.ndarray,
    wcs: WCS,
    skycoord: SkyCoord,
    size_pix: int,
) -> tuple[np.ndarray, WCS]:
    """Return a cutout centered on the target position for display.

    Args:
        image: 2D image array.
        wcs: WCS for the input image.
        skycoord: Target sky coordinate.
        size_pix: Cutout size in pixels (square).

    Returns:
        Tuple of (cutout image, cutout WCS).
    """
    size_pix = int(size_pix)
    if size_pix <= 0:
        raise ValueError("size_pix must be positive.")
    xpix, ypix = wcs.world_to_pixel(skycoord)
    cutout = Cutout2D(
        image,
        (xpix, ypix),
        (size_pix, size_pix),
        wcs=wcs,
        mode="partial",
        fill_value=np.nan,
    )
    return np.asarray(cutout.data), cutout.wcs


def _coerce_mask(mask: Optional[np.ndarray], n_points: int) -> np.ndarray:
    """Coerce a mask array to boolean values.

    Args:
        mask: Optional array-like input mask.
        n_points: Expected length of the mask.

    Returns:
        Boolean array mask with length n_points.

    Raises:
        ValueError: If mask length does not match n_points.
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


def _format_radec_stub(ra_deg: float, dec_deg: float, decimals: int = 8) -> str:
    """Format RA/Dec into a filename-safe stub.

    Args:
        ra_deg: Right ascension in degrees.
        dec_deg: Declination in degrees.
        decimals: Decimal places for rounding.

    Returns:
        Filename-safe stub in the form ``ra<ra>_dec<dec>``.
    """
    ra_str = f"{ra_deg:.{decimals}f}".replace(".", "p")
    dec_str = f"{dec_deg:.{decimals}f}".replace(".", "p")
    return f"ra{ra_str}_dec{dec_str}"


def _coerce_cached_url(value: object) -> Optional[str]:
    """Coerce a cached URL value to a usable string.

    Args:
        value: Cached value from a CSV cell.

    Returns:
        URL string if valid; otherwise None.
    """
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>"}:
        return None
    return text


def _get_download_url(datalink_url: str) -> str:
    """Resolve a datalink URL to the primary spectral image URL.

    Args:
        datalink_url: Datalink URL from the TAP results.

    Returns:
        URL for the spectral image file.
    """
    datalink_content = DatalinkResults.from_result_url(datalink_url)
    return next(datalink_content.bysemantics("#this")).access_url


def _get_download_urls_multithreaded(
    urls: Sequence[str],
    max_workers_local: int = 8,
) -> list[Optional[str]]:
    """Resolve datalink URLs in parallel.

    Args:
        urls: Sequence of datalink URLs to resolve.
        max_workers_local: Number of threads to use.

    Returns:
        List of resolved URLs in the same order as ``urls``.
    """
    if not urls:
        return []
    max_workers_local = int(max(1, max_workers_local))
    resolved: list[Optional[str]] = [None] * len(urls)

    log(
        f"Resolving {len(urls)} datalink URLs with "
        f"{max_workers_local} threads..."
    )

    def _worker(idx_url: tuple[int, str]) -> tuple[int, Optional[str]]:
        idx, url = idx_url
        try:
            return idx, _get_download_url(url)
        except Exception as e:
            log(f"Failed to resolve datalink URL at index {idx}: {e}")
            return idx, None

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_workers_local) as ex:
        futures = {ex.submit(_worker, (i, u)): i for i, u in enumerate(urls)}
        for fut in as_completed(futures):
            idx, result = fut.result()
            resolved[idx] = result

    n_ok = sum(1 for r in resolved if r)
    n_bad = len(resolved) - n_ok
    log(f"Resolved {n_ok} OK, {n_bad} failed.")
    return resolved


def _parse_subdir_from_fname(fname: str, root: Path) -> Optional[Path]:
    """Parse the local subdirectory for a SPHEREx L2 filename.

    Args:
        fname: FITS filename to parse.
        root: Base cache root directory.

    Returns:
        Subdirectory path or None if parsing fails.
    """
    stem = Path(fname).stem
    parts = stem.split('_')
    # Expected: [level2, 2025W27, 1A, 0253, 1D3, spx, l2b-v13-YYYY-DDD]
    if len(parts) < 7 or parts[0].lower() != 'level2':
        return None
    week = parts[1]
    cam = parts[2]
    det_field = parts[4]  # e.g., 1D3
    l2b = parts[6]
    m = re.search(r'D(\d+)$', det_field)
    detnum = m.group(1) if m else det_field
    return root / f"{week}_{cam}" / l2b / detnum


from astropy import units as u  # noqa: F401
from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    aperture_photometry,
)
from astropy.stats import sigma_clipped_stats

# Imported for completeness; not used directly.

# --- Bitmask for SPHEREx bad flags ---
# Bit 0: TRANSIENT (e.g. cosmic ray) detected during SUR
# Bit 1: OVERFLOW reached during SUR.  (threshold = half of full well).
# Bit 2: SUR_ERROR chksum_error from instrument processing
# Bit 6: NONFUNC pre-flight assessment that pixel is dead
# Bit 7: DICHROIC pixels in dark corners of Bands 3 and 4
# Bit 9: MISSING_DATA corrupted packages on downlink
# Bit 10: HOT pixel, not usable for science
# Bit 11: COLD pixel, unresponsive to light
# Bit 15: NONLINEARity correction couldn't be determined
BITMASK = (
    (1 << 0)  | (1 << 1)  | (1 << 2)  |
    (1 << 6)  | (1 << 7)  | (1 << 9)  |
    (1 << 10) | (1 << 11) | (1 << 15)
)

ARCSEC_PER_PIXEL = 6.1

TJD_OFFSET = 56999.5


def detect_stars_mask(image, fwhm_pix, threshold_sigma):
    """Return a boolean mask of bright pixels via sigma clipping.

    Args:
        image: 2D image array.
        fwhm_pix: FWHM in pixels (currently unused).
        threshold_sigma: Sigma threshold above the median.

    Returns:
        Boolean mask of bright pixels.

    Notes:
        This simplified detector ignores the provided FWHM and
        identifies pixels brighter than median + threshold_sigma * std,
        where the statistics are computed with sigma clipping over
        finite pixels.
    """
    data = np.asarray(image)
    mask = np.zeros_like(data, dtype=bool)
    finite = np.isfinite(data)
    if not np.any(finite):
        return mask
    # Robust stats from finite pixels only
    _, median, std = sigma_clipped_stats(data[finite], sigma=3.0, maxiters=5)
    thresh = median + float(threshold_sigma) * std
    mask[finite] = data[finite] > thresh
    return mask


def _resolve_photometry_workers(n_workers: Optional[int]) -> int:
    """Return the number of photometry workers to use.

    Args:
        n_workers: Requested number of workers or None for default.

    Returns:
        Positive integer worker count.
    """
    if n_workers is None:
        n_cpu = os.cpu_count() or 1
        return max(1, n_cpu // 2)
    return max(1, int(n_workers))


def _photometry_worker(args: tuple) -> tuple[int, Optional[dict], list[str]]:
    """Run aperture photometry for one file path.

    Args:
        args: Tuple of arguments passed from the parent process.

    Returns:
        Tuple of (index, record dict or None, log messages).
    """
    (
        idx,
        n_total,
        fpath,
        ra_deg,
        dec_deg,
        star_id,
        aperture_radius,
        bkgd_method,
        annulus_r_in,
        annulus_r_out,
        star_fwhm,
        star_threshold_sigma,
    ) = args
    messages: list[str] = []

    try:
        messages.append(f"{idx}/{n_total} Doing photometry...")
        with fits.open(fpath, memmap=False) as hdul:
            wcs = WCS(hdul['IMAGE'].header)
            skycoord = SkyCoord(
                ra=ra_deg,
                dec=dec_deg,
                unit='degree',
                frame='icrs',
            )
            x, y = wcs.world_to_pixel(skycoord)

            if not (0 <= x < 2040 and 0 <= y < 2040):
                return idx, None, messages

            aperture = CircularAperture([(x, y)], r=aperture_radius)
            flux_img = hdul['IMAGE'].data

            extnames = {h.name.upper() for h in hdul}
            var_img = (
                hdul['VARIANCE'].data
                if 'VARIANCE' in extnames
                else np.full_like(flux_img, np.nan)
            )
            flags_img = (
                hdul['FLAGS'].data
                if 'FLAGS' in extnames
                else np.zeros_like(flux_img, dtype=np.uint32)
            )
            zodi_img = (
                hdul['ZODI'].data
                if 'ZODI' in extnames
                else np.zeros_like(flux_img)
            )

            if bkgd_method == 'zodi':
                img_for_phot = flux_img - zodi_img
                bkgd_level = None
            else:
                annulus = CircularAnnulus(
                    [(x, y)],
                    r_in=annulus_r_in,
                    r_out=annulus_r_out,
                )
                ann_mask_img = (
                    annulus.to_mask(method='center')[0]
                    .to_image(shape=flux_img.shape)
                    .astype(bool)
                )
                star_mask = detect_stars_mask(
                    flux_img,
                    fwhm_pix=star_fwhm,
                    threshold_sigma=star_threshold_sigma,
                )
                valid = ann_mask_img & (~star_mask) & np.isfinite(flux_img)
                if np.any(valid):
                    _, bkg_med, _ = sigma_clipped_stats(
                        flux_img[valid],
                        sigma=3.0,
                        maxiters=5,
                    )
                    bkgd_level = float(bkg_med)
                else:
                    bkgd_level = 0.0
                img_for_phot = flux_img

            flux_tbl = aperture_photometry(img_for_phot, aperture)
            var_tbl = aperture_photometry(var_img, aperture)

            flux_ap = flux_tbl['aperture_sum'][0]
            flux_err = np.sqrt(var_tbl['aperture_sum'][0])

            if not np.isfinite(flux_err):
                pos_flux_img = np.clip(flux_img, a_min=0, a_max=None)
                pos_tbl = aperture_photometry(pos_flux_img, aperture)
                flux_err = np.sqrt(pos_tbl['aperture_sum'][0])

            if bkgd_method == 'annulus' and bkgd_level is not None:
                ap_area_pix = np.pi * (aperture_radius ** 2)
                flux_ap = flux_ap - bkgd_level * ap_area_pix
            flux_val = flux_ap

            mask = aperture.to_mask(method="center")[0]
            aper_mask = mask.to_image(shape=flags_img.shape)
            aper_flags = flags_img[aper_mask.astype(bool)]
            is_flagged = np.any(aper_flags & BITMASK)
            aper_count = str(dict(Counter(aper_flags)))
            quality_flags = ",".join(np.unique(aper_flags).astype(str))

            wave_wcs = WCS(hdul[1].header, hdul, key='W')
            wave_wcs.sip = None
            lam, dlam = wave_wcs.wcs_pix2world(x, y, 0)

            mjd_avg = hdul['IMAGE'].header.get('MJD-AVG', np.nan)

            pix_area_sr = (ARCSEC_PER_PIXEL / 3600 * np.pi / 180) ** 2
            flux_jy = flux_val * pix_area_sr * 1e6
            flux_err_jy = flux_err * pix_area_sr * 1e6

            record = {
                "wavelength_um": lam,
                "bandwidth_um": dlam,
                "flux_jy": flux_jy,
                "flux_err_jy": flux_err_jy,
                "file": fpath,
                "masked": is_flagged,
                "quality": quality_flags,
                "aper_count": aper_count,
                "mjd_avg": mjd_avg,
                "tjd_avg": mjd_avg - TJD_OFFSET,
                "bkgd_method": bkgd_method,
                "aperture_radius_pix": aperture_radius,
                "annulus_r_in_pix": (
                    annulus_r_in
                    if bkgd_method == 'annulus'
                    else np.nan
                ),
                "annulus_r_out_pix": (
                    annulus_r_out
                    if bkgd_method == 'annulus'
                    else np.nan
                ),
            }

            messages.append(f"{fpath}")
            messages.append(
                f"Pixel: ({x:.1f}, {y:.1f}), λ = {lam:.4f} µm "
                f"± {dlam:.4f}, Flux = {flux_jy:.3e} Jy"
            )

            return idx, record, messages

    except Exception as e:
        messages.append(
            f" {star_id} {idx} {fpath} failed with {e}. "
            "skipping this image."
        )
        return idx, None, messages


def get_sx_spectrum(
    ra_deg: float,
    dec_deg: float,
    *,
    do_photometry: bool = False,
    bkgd_method: str = 'annulus',
    aperture_radius: float = 2.0,
    annulus_r_in: float = 6.0,
    annulus_r_out: float = 8.0,
    star_fwhm: float = 2.0,
    star_threshold_sigma: float = 5.0,
    min_images: int = 3,
    max_images: int = 99999,
    max_missing_fraction: float = 0.05,
    use_cutout: bool = True,
    max_workers: int = 40,
    photometry_workers: Optional[int] = None,
    size_pix: int = 64,
    save_plot: bool = True,
    show_cutout: bool = True,
    save_csv: bool = True,
    output_dir: Union[str, Path] = ".",
    star_id: Optional[str] = None,
):
    """Query SPHEREx L2 images overlapping (ra_deg, dec_deg) and perform
    aperture photometry.

    Args:
        ra_deg: Target ICRS coordinates in degrees.
        dec_deg: Target ICRS coordinates in degrees.
        do_photometry: Force recomputation even if a cache CSV exists.
        bkgd_method: Background method, 'zodi' or 'annulus'.
        aperture_radius: Circular aperture radius in pixels.
        annulus_r_in: Annulus inner radius in pixels.
        annulus_r_out: Annulus outer radius in pixels.
        star_fwhm: Star mask FWHM in pixels for annulus clipping.
        star_threshold_sigma: Sigma threshold for star masking.
        min_images: Minimum number of images to process.
        max_images: Maximum number of images to process.
        max_missing_fraction: Maximum fraction of missing cached records to
            allow before reprocessing.
        use_cutout: If True, download IRSA cutouts; else use full MEF.
        max_workers: Threads to resolve datalink URLs in parallel.
        photometry_workers: Processes to use for aperture photometry. Defaults
            to half the available CPUs if None.
        size_pix: Cutout size (square) in pixels when use_cutout is
            True.
        save_plot: Whether to save the result PNG.
        show_cutout: Whether to include a cutout in saved plots.
        save_csv: Whether to save the CSV cache.
        output_dir: Output directory for plots and CSVs.
        star_id: Optional star identifier used in output filenames.

    Returns:
        pandas.DataFrame: Per-image photometry records (may be empty).
    """
    # CSV cache of processed photometry records
    radecstr = _format_radec_stub(ra_deg, dec_deg)
    _a = f"_{bkgd_method}"
    staridstr = "" if star_id in (None, "") else f"{star_id}_"
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_csv = outdir / f"sxphot_cache_{staridstr}{radecstr}{_a}.csv"

    log(
        f"Starting get_sx_spectrum for {star_id} {radecstr} "
        f"{bkgd_method}"
    )

    skycoord = SkyCoord(ra=ra_deg, dec=dec_deg, unit='degree', frame='icrs')

    # Define the TAP service URL for IRSA
    tap_url = "https://irsa.ipac.caltech.edu/TAP"
    service = pyvo.dal.TAPService(tap_url)

    # ADQL query for all overlapping spectral images
    query = (
        "SELECT * FROM spherex.obscore WHERE CONTAINS(POINT('ICRS',"
        + str(ra_deg)
        + ","
        + str(dec_deg)
        + "), s_region)=1"
    )

    job = service.submit_job(query)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=300)
    results = job.fetch_result()

    if len(results) < min_images:
        return

    records = None
    if cache_csv.exists():
        try:
            df_cache = pd.read_csv(cache_csv)
            expected_count = (
                len(results)
                if max_images is None
                else min(len(results), max_images)
            )
            missing_count = max(expected_count - len(df_cache), 0)
            missing_frac = _missing_fraction(len(df_cache), expected_count)
            if _is_cache_complete(
                len(df_cache),
                expected_count,
                max_missing_fraction,
            ):
                log(
                    "Using cached CSV "
                    f"{cache_csv} with {len(df_cache)} records "
                    f"({missing_count} missing of {expected_count}, "
                    f"{missing_frac:.1%}). Skipping downloads."
                )
                records = df_cache.to_dict(orient='records')
            else:
                log(
                    "Cached CSV has "
                    f"{len(df_cache)} records; {missing_count} of "
                    f"{expected_count} ({missing_frac:.1%}) missing "
                    f"exceeds {max_missing_fraction:.1%}. "
                    "Will download/process all FITS."
                )
        except Exception as e:
            log(
                f"⚠️ Failed to read cache CSV {cache_csv}: {e}. "
                "Will download/process all FITS."
            )

    # Collect datalink URLs
    datalink_urls = [results['access_url'][i] for i in range(len(results))]
    url_cache_csv = outdir / f"sxphot_url_cache_{staridstr}{radecstr}.csv"
    url_cache_map: dict[str, Optional[str]] = {}

    if url_cache_csv.exists():
        try:
            url_cache_df = pd.read_csv(url_cache_csv)
            required = {"datalink_url", "download_url"}
            if required.issubset(url_cache_df.columns):
                for row in url_cache_df.itertuples(index=False):
                    datalink_url = _coerce_cached_url(
                        getattr(row, "datalink_url", None)
                    )
                    download_url = _coerce_cached_url(
                        getattr(row, "download_url", None)
                    )
                    if datalink_url:
                        url_cache_map[datalink_url] = download_url
                log(
                    f"Loaded {len(url_cache_map)} cached datalink URLs "
                    f"from {url_cache_csv}"
                )
            else:
                log(
                    f"⚠️ URL cache {url_cache_csv} missing columns "
                    f"{sorted(required)}; ignoring."
                )
        except Exception as e:
            log(f"⚠️ Failed to read URL cache {url_cache_csv}: {e}.")

    download_urls = [url_cache_map.get(url) for url in datalink_urls]
    missing_idx = [i for i, url in enumerate(download_urls) if not url]
    if missing_idx:
        missing_urls = [datalink_urls[i] for i in missing_idx]
        log(
            f"{len(missing_urls)} datalink URLs missing from cache; "
            "resolving."
        )
        resolved = _get_download_urls_multithreaded(
            missing_urls,
            max_workers_local=max_workers,
        )
        for idx, resolved_url in zip(missing_idx, resolved):
            download_urls[idx] = resolved_url
            if resolved_url:
                url_cache_map[datalink_urls[idx]] = resolved_url

        if save_csv:
            try:
                pd.DataFrame(
                    {
                        "datalink_url": datalink_urls,
                        "download_url": download_urls,
                    }
                ).to_csv(url_cache_csv, index=False)
                log(f"Saved URL cache CSV: {url_cache_csv}")
            except Exception as e:
                log(f"⚠️ Failed to save URL cache CSV {url_cache_csv}: {e}")
    irsa_spectral_image_urls = [u for u in download_urls if u]
    # Map FITS basename -> IRSA spectral image URL for later cutout retrieval
    basename_to_irsa_url = {Path(u).name: u for u in irsa_spectral_image_urls}
    log(
        f"Prepared {len(irsa_spectral_image_urls)} primary download "
        "URLs (e.g., AWS/Cutout derivable)."
    )

    assert len(download_urls) == len(results)

    # Build local file paths (and download if needed)
    file_paths = []
    if records is None:
        for i, irsa_spectral_image_url in zip(
            range(len(results)),
            irsa_spectral_image_urls,
        ):
            if i > max_images - 1:
                continue

            log(f"{i}/{len(results)}...")

            aws_spectral_image_url = irsa_spectral_image_url.replace(
                'https://irsa.ipac.caltech.edu/ibe/data/spherex',
                'https://nasa-irsa-spherex.s3.amazonaws.com'
            )
            irsa_cutout_image_url = (
                irsa_spectral_image_url
                + f"?center={ra_deg},{dec_deg}deg&size={int(size_pix)}pix"
            )

            local_fname = Path(irsa_spectral_image_url).name

            if use_cutout:
                cache_root = Path.home() / "local" / "SPHEREX" / "cutouts"
                subdir = cache_root / radecstr
                download_url = irsa_cutout_image_url
            else:
                cache_root = Path.home() / "local" / "SPHEREX" / "spherex_l2"

                subdir = (
                    _parse_subdir_from_fname(local_fname, cache_root)
                    or cache_root
                )
                download_url = aws_spectral_image_url

            cache_root.mkdir(parents=True, exist_ok=True)
            local_path = subdir / local_fname

            # Download file if not already present
            if not local_path.exists():
                # Fallback: consider flat layout directly under cache_root
                flat_path = cache_root / local_fname
                if flat_path.exists():
                    local_path = flat_path
                else:
                    try:
                        subdir.mkdir(parents=True, exist_ok=True)
                        log(f"Downloading to {local_path} ...")
                        urlretrieve(download_url, local_path)
                        log("Download complete.")
                    except (URLError, HTTPError) as e:
                        log(
                            f"⚠️ Download failed: {e}. Will try current "
                            "directory if available."
                        )

            # Fallback: if download failed but file exists in CWD, use it
            if not local_path.exists():
                cwd_fallback = Path(os.getcwd()) / local_fname
                if cwd_fallback.exists():
                    local_path = cwd_fallback

            file_paths.append(str(local_path))
            log(file_paths[i])

    # Decide whether to (re)compute photometry.
    do_phot = (records is None) or bool(do_photometry)

    # If we are recomputing and we had cached records, reuse their file paths.
    if (records is not None) and do_phot:
        file_paths = [str(r.get('file')) for r in records if 'file' in r]

    # Aperture photometry
    APERTURE_RADIUS = float(aperture_radius)

    if do_phot:
        records = []
        n_workers = _resolve_photometry_workers(photometry_workers)
        log(
            f"Running photometry on {len(file_paths)} files with "
            f"{n_workers} processes."
        )
        tasks = [
            (
                ix,
                len(file_paths),
                fpath,
                ra_deg,
                dec_deg,
                star_id,
                APERTURE_RADIUS,
                bkgd_method,
                annulus_r_in,
                annulus_r_out,
                star_fwhm,
                star_threshold_sigma,
            )
            for ix, fpath in enumerate(file_paths)
        ]
        if n_workers == 1:
            for args in tasks:
                _, record, messages = _photometry_worker(args)
                for message in messages:
                    log(message)
                if record is not None:
                    records.append(record)
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                for _, record, messages in pool.imap_unordered(
                    _photometry_worker,
                    tasks,
                ):
                    for message in messages:
                        log(message)
                    if record is not None:
                        records.append(record)


    # Plot + CSV
    if records:
        records = sorted(records, key=lambda r: r["wavelength_um"])
        lam = np.array([r["wavelength_um"] for r in records])
        flux = np.array([r["flux_jy"] for r in records])
        err = np.array([r["flux_err_jy"] for r in records])
        mjds = np.array([r["mjd_avg"] for r in records])
        mjds -= TJD_OFFSET  # MJD to TESS julian date
        masked_vals = [r.get("masked", False) for r in records]
        masked = np.array(
            [
                bool(v)
                if isinstance(v, (bool, np.bool_))
                else str(v).lower() in {"1", "true", "t", "yes"}
                for v in masked_vals
            ]
        )

        # Prepare a single cutout image near the band-1 center if requested.
        cutout_img = None
        cutout_wav = None
        cutout_wcs = None
        cutout_source_path: Optional[Path] = None
        if show_cutout:
            try:
                target_wav = 0.93
                idx_cutout = _pick_cutout_index(lam, target_wav)
                if idx_cutout is not None:
                    rec_cutout = records[idx_cutout]
                    cutout_wav = float(
                        rec_cutout.get("wavelength_um", np.nan)
                    )
                    rec_file = rec_cutout.get("file", "")
                    rec_fname = Path(str(rec_file)).name if rec_file else None

                    # Try to use existing file if it exists and is readable
                    candidate_path = Path(str(rec_file)) if rec_file else None
                    use_existing = (
                        candidate_path is not None
                        and candidate_path.exists()
                    )

                    if use_existing and use_cutout:
                        # Likely already a cutout; open directly
                        with fits.open(candidate_path, memmap=False) as hdul:
                            cutout_img = np.asarray(hdul['IMAGE'].data)
                            cutout_wcs = WCS(hdul['IMAGE'].header)
                        cutout_source_path = candidate_path
                    else:
                        # Build an IRSA cutout URL for this specific
                        # spectral image.
                        base_url = basename_to_irsa_url.get(rec_fname, None)
                        if base_url is None and use_existing:
                            # Attempt to recover by matching by name among
                            # all URLs.
                            for fname, url in basename_to_irsa_url.items():
                                if fname == rec_fname:
                                    base_url = url
                                    break
                        if base_url is not None:
                            irsa_cutout_url = (
                                f"{base_url}?center={ra_deg},{dec_deg}deg"
                                f"&size={int(size_pix)}pix"
                            )
                            # Cache path for the cutout
                            cache_root = (
                                Path.home() / "local" / "SPHEREX" / "cutouts"
                            )
                            subdir = cache_root / f"{radecstr}"
                            subdir.mkdir(parents=True, exist_ok=True)
                            cutout_path = subdir / (rec_fname or "cutout.fits")
                            if not cutout_path.exists():
                                try:
                                    log(
                                        "Downloading display cutout to "
                                        f"{cutout_path} ..."
                                    )
                                    urlretrieve(irsa_cutout_url, cutout_path)
                                except Exception as e:
                                    log(
                                        "⚠️ Failed to download display "
                                        f"cutout: {e}"
                                    )
                            if cutout_path.exists():
                                with fits.open(
                                    cutout_path,
                                    memmap=False,
                                ) as hdul:
                                    cutout_img = np.asarray(hdul['IMAGE'].data)
                                    cutout_wcs = WCS(hdul['IMAGE'].header)
                                cutout_source_path = cutout_path
                        elif use_existing:
                            # Fall back: read existing (full) image and show it
                            try:
                                with fits.open(
                                    candidate_path,
                                    memmap=False,
                                ) as hdul:
                                    cutout_img = np.asarray(hdul['IMAGE'].data)
                                    cutout_wcs = WCS(hdul['IMAGE'].header)
                                cutout_source_path = candidate_path
                            except Exception:
                                pass
                if cutout_img is not None and cutout_source_path is not None:
                    log(f"Cutout FITS source: {cutout_source_path}")
                if cutout_img is not None and cutout_wcs is not None:
                    try:
                        cutout_img, cutout_wcs = _center_cutout_for_display(
                            cutout_img,
                            cutout_wcs,
                            skycoord,
                            size_pix,
                        )
                    except Exception as e:
                        log(
                            "⚠️ Failed to center cutout for display: "
                            f"{e}"
                        )
                if cutout_wcs is not None:
                    try:
                        sc = SkyCoord(
                            ra=ra_deg,
                            dec=dec_deg,
                            unit="deg",
                            frame="icrs",
                        )
                        xpix, ypix = cutout_wcs.world_to_pixel(sc)
                        log(
                            "Cutout aperture center (x, y) = "
                            f"({xpix:.2f}, {ypix:.2f})"
                        )
                    except Exception as e:
                        log(
                            "⚠️ Failed to compute cutout aperture center: "
                            f"{e}"
                        )
            except Exception as e:
                log(f"⚠️ Failed preparing cutout image: {e}")

        if save_plot:

            from aesthetic.plot import set_style, savefig
            set_style('science')

            # Figure 1: just the spectrum
            plt.close('all')
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.errorbar(
                lam[~masked],
                flux[~masked],
                yerr=err[~masked],
                fmt='.',
                capsize=3,
                c='k',
            )
            ax.errorbar(
                lam[masked],
                flux[masked],
                fmt='x',
                color='r',
                capsize=3,
            )
            ax.set_yscale('log')
            ax.set_xlabel("Wavelength (µm)")
            ax.set_ylabel("Flux (Jy)")
            ax.set_title(
                f"{staridstr.replace('_',':')} α={ra_deg:.4f}, "
                f"δ={dec_deg:.4f}"
            )
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            # Optionally overlay a cutout inset in the upper-right
            if show_cutout:
                _add_cutout_inset(
                    fig,
                    ax,
                    cutout_img,
                    cutout_wav,
                    cutout_wcs,
                    ra_deg,
                    dec_deg,
                    APERTURE_RADIUS,
                    show_offsets=False,
                )
            fig.tight_layout()
            savpath = outdir / f"result_{staridstr}{radecstr}{_a}.png"
            savefig(fig, str(savpath), writepdf=0)
            log(f"Saved {savpath}")
            plt.close('all')

            # Figure 2: spectrum colored by MJD with a small colorbar
            plt.close('all')
            fig2, ax2 = plt.subplots(figsize=(5, 5))

            # Establish color normalization on available MJD values
            finite_mjd = np.isfinite(mjds)
            if np.any(finite_mjd):
                vmin = float(np.nanmin(mjds[finite_mjd]))
                vmax = float(np.nanmax(mjds[finite_mjd]))
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.viridis

                # Error bars in light gray beneath the colored points
                ax2.errorbar(
                    lam[~masked],
                    flux[~masked],
                    yerr=err[~masked],
                    fmt='none',
                    ecolor='0.6',
                    elinewidth=1,
                    capsize=2,
                    alpha=0.7,
                    zorder=-1,
                )

                sc = ax2.scatter(
                    lam[~masked],
                    flux[~masked],
                    c=mjds[~masked],
                    cmap=cmap,
                    norm=norm,
                    s=20,
                    edgecolors='none',
                )
            else:
                # Fallback if MJD is missing; plot in gray
                sc = ax2.scatter(
                    lam[~masked],
                    flux[~masked],
                    color='0.5',
                    s=20,
                    edgecolors='none',
                )

            # Mark masked points similarly to Figure 1
            if np.any(masked):
                ax2.errorbar(
                    lam[masked],
                    flux[masked],
                    fmt='x',
                    color='r',
                    capsize=3,
                )

            ax2.set_yscale('log')
            ax2.set_xlabel("Wavelength (µm)")
            ax2.set_ylabel("Flux (Jy)")
            ax2.set_title(
                f"{staridstr.replace('_',':')} α={ra_deg:.2f}, "
                f"δ={dec_deg:.2f}"
            )
            ax2.grid(True, which='both', linestyle='--', alpha=0.5)

            # Add a small horizontal colorbar at lower-left of the plot
            try:
                if np.any(finite_mjd):
                    # Place colorbar within the axes area, lower-left,
                    # ~25% axis width.
                    axpos = ax2.get_position()
                    cb_width = 0.25 * axpos.width
                    cb_height = 0.03 * axpos.height
                    cb_left = axpos.x0 + 0.07 * axpos.width
                    cb_bottom = axpos.y0 + 0.12 * axpos.height
                    cax = fig2.add_axes(
                        [cb_left, cb_bottom, cb_width, cb_height]
                    )
                    cb = fig2.colorbar(sc, cax=cax, orientation='horizontal')
                    cb.set_label('TESS JD')
            except Exception:
                # If anything goes wrong with manual placement, skip the
                # colorbar.
                pass

            # Optionally overlay the same cutout inset here too
            if show_cutout:
                _add_cutout_inset(
                    fig2,
                    ax2,
                    cutout_img,
                    cutout_wav,
                    cutout_wcs,
                    ra_deg,
                    dec_deg,
                    APERTURE_RADIUS,
                    show_offsets=False,
                )
            fig2.tight_layout()
            savpath2 = outdir / f"{staridstr}{radecstr}{_a}_mjdc.png"
            savefig(fig2, str(savpath2), writepdf=0)
            log(f"Saved {savpath2}")
            plt.close('all')

        if save_csv:
            try:
                pd.DataFrame(records).to_csv(cache_csv, index=False)
                log(f"Saved {len(records)} records to cache CSV: {cache_csv}")
            except Exception as e:
                log(f"⚠️ Failed to save cache CSV {cache_csv}: {e}")
    else:
        log("No valid measurements to plot.")

    # Return records as a DataFrame for programmatic use
    try:
        return pd.DataFrame(records) if records else pd.DataFrame()
    except Exception:
        # As a fallback, return the raw records list
        return records or []


def get_supplemented_sx_spectrum(
    ra_deg: float,
    dec_deg: float,
    *,
    do_photometry: bool = False,
    bkgd_method: str = 'annulus',
    aperture_radius: float = 2.0,
    annulus_r_in: float = 6.0,
    annulus_r_out: float = 8.0,
    star_fwhm: float = 2.0,
    star_threshold_sigma: float = 5.0,
    min_images: int = 3,
    max_images: int = 99999,
    max_missing_fraction: float = 0.05,
    use_cutout: bool = True,
    max_workers: int = 40,
    photometry_workers: Optional[int] = None,
    size_pix: int = 64,
    save_plot: bool = True,
    show_cutout: bool = True,
    save_csv: bool = True,
    output_dir: Union[str, Path] = ".",
    star_id: Optional[str] = None,
    n_res_el: float = 4.0,
    degree: int = 3,
    spacing_mode: str = "log",
    max_iter: int = 8,
    outlier_cut: float = 3.0,
    clip_on_normalized_residuals: bool = True,
    dense_n_per_band: int = 800,
    save_supp_csv: bool = True,
    use_supp_cache: bool = True,
) -> pd.DataFrame:
    """Return a SPHEREx spectrum augmented with spline-fit diagnostics.

    Args:
        ra_deg: Target ICRS coordinates in degrees.
        dec_deg: Target ICRS coordinates in degrees.
        do_photometry: Force recomputation even if a cache CSV exists.
        bkgd_method: Background method, 'zodi' or 'annulus'.
        aperture_radius: Circular aperture radius in pixels.
        annulus_r_in: Annulus inner radius in pixels.
        annulus_r_out: Annulus outer radius in pixels.
        star_fwhm: Star mask FWHM in pixels for annulus clipping.
        star_threshold_sigma: Sigma threshold for star masking.
        min_images: Minimum number of images to process.
        max_images: Maximum number of images to process.
        max_missing_fraction: Maximum fraction of missing cached records to
            allow before reprocessing.
        use_cutout: If True, download IRSA cutouts; else use full MEF.
        max_workers: Threads to resolve datalink URLs in parallel.
        photometry_workers: Processes to use for aperture photometry. Defaults
            to half the available CPUs if None.
        size_pix: Cutout size (square) in pixels when use_cutout is True.
        save_plot: Whether to save the result PNG.
        show_cutout: Whether to include a cutout in saved plots.
        save_csv: Whether to save the photometry CSV cache.
        output_dir: Output directory for plots and CSVs.
        star_id: Optional star identifier used in output filenames.
        n_res_el: Number of resolution elements per knot interval.
        degree: Spline degree.
        spacing_mode: 'log' for uniform in ln(lambda), 'linear' for uniform in
            lambda.
        max_iter: Maximum number of clip-refit iterations per band.
        outlier_cut: Robust sigma cut for clipping.
        clip_on_normalized_residuals: If True, clip on (resid/err).
        dense_n_per_band: Number of samples for dense grid per band.
        save_supp_csv: Whether to save the supplemented CSV cache.
        use_supp_cache: If True, reuse the supplemented cache when present.

    Returns:
        pandas.DataFrame: Photometry records with spline model columns.

    Raises:
        ValueError: If required columns are missing from the base spectrum.
    """
    radecstr = _format_radec_stub(ra_deg, dec_deg)
    _a = f"_{bkgd_method}"
    staridstr = "" if star_id in (None, "") else f"{star_id}_"
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    supp_cache_csv = (
        outdir / f"sxphot_cache_{staridstr}{radecstr}{_a}_splsupp.csv"
    )

    if use_supp_cache and supp_cache_csv.exists():
        try:
            df_cache = pd.read_csv(supp_cache_csv)
            log(
                f"Using supplemented cache {supp_cache_csv} with "
                f"{len(df_cache)} records."
            )
            return df_cache
        except Exception as e:
            log(
                f"⚠️ Failed to read supplemented cache {supp_cache_csv}: {e}. "
                "Will recompute."
            )

    base = get_sx_spectrum(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        do_photometry=do_photometry,
        bkgd_method=bkgd_method,
        aperture_radius=aperture_radius,
        annulus_r_in=annulus_r_in,
        annulus_r_out=annulus_r_out,
        star_fwhm=star_fwhm,
        star_threshold_sigma=star_threshold_sigma,
        min_images=min_images,
        max_images=max_images,
        max_missing_fraction=max_missing_fraction,
        use_cutout=use_cutout,
        max_workers=max_workers,
        photometry_workers=photometry_workers,
        size_pix=size_pix,
        save_plot=save_plot,
        show_cutout=show_cutout,
        save_csv=save_csv,
        output_dir=outdir,
        star_id=star_id,
    )

    if base is None:
        return pd.DataFrame()

    if not isinstance(base, pd.DataFrame):
        base = pd.DataFrame(base)

    if base.empty:
        return base

    required = ["wavelength_um", "bandwidth_um", "flux_jy", "flux_err_jy"]
    missing = [name for name in required if name not in base.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    wavelength_um = np.asarray(base["wavelength_um"], dtype=float)
    bandwidth_um = np.asarray(base["bandwidth_um"], dtype=float)
    flux_jy = np.asarray(base["flux_jy"], dtype=float)
    flux_err_jy = np.asarray(base["flux_err_jy"], dtype=float)
    masked_vals = base["masked"] if "masked" in base.columns else None
    mask = _coerce_mask(masked_vals, wavelength_um.size)

    (
        model_flux,
        fit_mask,
        _,
        _,
        _,
    ) = fit_spherex_spectrum_bspline(
        wavelength_um=wavelength_um,
        bandwidth_um=bandwidth_um,
        flux_jy=flux_jy,
        flux_err_jy=flux_err_jy,
        masked=mask,
        n_res_el=n_res_el,
        degree=degree,
        spacing_mode=spacing_mode,
        max_iter=max_iter,
        outlier_cut=outlier_cut,
        clip_on_normalized_residuals=clip_on_normalized_residuals,
        dense_n_per_band=dense_n_per_band,
    )

    finite = (
        np.isfinite(wavelength_um)
        & np.isfinite(flux_jy)
        & np.isfinite(flux_err_jy)
    )
    mask = mask | (~finite)
    use = ~mask
    fit_used = (
        use if fit_mask is None else np.asarray(fit_mask, dtype=bool) & use
    )
    clipped = use & (~fit_used)
    clipped &= np.isfinite(model_flux)

    residuals = flux_jy - model_flux
    res_valid = np.isfinite(residuals) & fit_used & np.isfinite(flux_err_jy)
    res_masked = np.isfinite(residuals) & mask
    res_clipped = np.isfinite(residuals) & clipped

    supplemented = base.copy()
    supplemented["model_flux"] = model_flux
    supplemented["fit_mask"] = fit_used
    supplemented["residuals"] = residuals
    supplemented["res_valid"] = res_valid
    supplemented["res_masked"] = res_masked
    supplemented["res_clipped"] = res_clipped

    if save_supp_csv:
        try:
            supplemented.to_csv(supp_cache_csv, index=False)
            log(
                f"Saved supplemented cache {supp_cache_csv} with "
                f"{len(supplemented)} records."
            )
        except Exception as e:
            log(f"⚠️ Failed to save supplemented cache {supp_cache_csv}: {e}")

    return supplemented


__all__ = ["get_sx_spectrum", "get_supplemented_sx_spectrum"]
