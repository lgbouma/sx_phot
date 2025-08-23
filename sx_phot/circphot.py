"""
Circular aperture photometry for SPHEREx Level-2 MEF images.

This module exposes a callable API:

    get_sx_spectrum(ra_deg, dec_deg, **kwargs) -> pandas.DataFrame

Summary
- Queries the IRSA TAP service for SPHEREx observations overlapping a target
  (ra_deg, dec_deg).
- Downloads the primary spectral image MEF FITS (via the public S3 mirror),
  or an IRSA cutout (default), caches them on disk, and performs aperture
  photometry at the target pixel.
- Saves a spectrum plot PNG and a CSV cache of per-image photometry.

Background methods
- bkgd_method = 'zodi': subtract the provided ZODI background image (default).
- bkgd_method = 'annulus': estimate a local background as the sigma‑clipped
  median within a circular annulus [annulus_r_in, annulus_r_out] in pixels,
  masking stars in the annulus via sigclip.

Outputs
- result_<radec>.png in the current directory.
- sxphot_cache_<radec>.csv in the current directory.

Notes
- Aperture area is converted from MJy/sr to Jy using the pixel solid angle.
"""

import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)

import os
os.environ["MPLBACKEND"] = "Agg"
import re
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import pandas as pd
import pyvo
from pyvo.dal.adhoc import DatalinkResults
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
from pathlib import Path
from datetime import datetime
from typing import Optional, Union


def log(message: str) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {message}")


from astropy import units as u  # noqa: F401 (imported for completeness; not used directly)
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.stats import sigma_clipped_stats

# --- Bitmask for SPHEREx bad flags ---
BITMASK = (
    (1 << 0)  | (1 << 1)  | (1 << 2)  |
    (1 << 6)  | (1 << 7)  | (1 << 9)  |
    (1 << 10) | (1 << 11) | (1 << 15)
)

ARCSEC_PER_PIXEL = 6.1


def detect_stars_mask(image, fwhm_pix, threshold_sigma):
    """Return a boolean mask of bright pixels via simple sigma clipping.

    This simplified detector ignores the provided FWHM and identifies
    pixels brighter than median + threshold_sigma * std, where the
    statistics are computed with sigma clipping over finite pixels.
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
    use_cutout: bool = True,
    max_workers: int = 40,
    size_pix: int = 64,
    save_plot: bool = True,
    save_csv: bool = True,
    output_dir: Union[str, Path] = ".",
    star_id: Optional[str] = None,
):
    """Query SPHEREx L2 images overlapping (ra_deg, dec_deg) and perform aperture photometry.

    Parameters
    - ra_deg, dec_deg: target ICRS coordinates in degrees.
    - do_photometry: force recomputation even if a cache CSV exists.
    - bkgd_method: 'zodi' or 'annulus'.
    - aperture_radius: circular aperture radius in pixels.
    - annulus_r_in, annulus_r_out: annulus radii in pixels (for bkgd_method='annulus').
    - star_fwhm, star_threshold_sigma: star masking parameters within the annulus.
    - min_images: minimum number of images to process.
    - max_images: maximum number of images to process.
    - use_cutout: if True, download IRSA cutouts centered at the target; else download full MEF from AWS.
    - max_workers: threads to resolve datalink URLs in parallel.
    - size_pix: cutout size (square) in pixels when use_cutout=True.
    - save_plot, save_csv: whether to save result PNG and CSV cache.

    Returns
    - pandas.DataFrame of per-image photometry records (may be empty).
    """
    # CSV cache of processed photometry records
    radecstr = f"ra{str(ra_deg).replace('.', 'p')}_{str(dec_deg).replace('.', 'p')}"
    _a = f"_{bkgd_method}"
    staridstr = "" if star_id in (None, "") else f"{star_id}_"
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_csv = outdir / f"sxphot_cache_{staridstr}{radecstr}{_a}.csv"

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
            if (len(df_cache) >= len(results)) or (len(df_cache) >= max_images):
                log(
                    f"Using cached CSV {cache_csv} with {len(df_cache)} records (>= {len(results)} results). Skipping downloads."
                )
                records = df_cache.to_dict(orient='records')
            else:
                log(
                    f"Cached CSV has {len(df_cache)} records (< {len(results)} results). Will download/process all FITS."
                )
        except Exception as e:
            log(f"⚠️ Failed to read cache CSV {cache_csv}: {e}. Will download/process all FITS.")

    # Collect datalink URLs
    datalink_urls = [results['access_url'][i] for i in range(len(results))]

    def get_download_url(datalink_url: str) -> str:
        datalink_content = DatalinkResults.from_result_url(datalink_url)
        irsa_spectral_image_url = next(datalink_content.bysemantics("#this")).access_url
        return irsa_spectral_image_url

    # Resolve datalink URLs in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def get_download_urls_multithreaded(urls, max_workers_local: int = 8):
        if not urls:
            return []
        max_workers_local = int(max(1, max_workers_local))
        resolved = [None] * len(urls)

        log(f"Resolving {len(urls)} datalink URLs with {max_workers_local} threads...")

        def _worker(idx_url):
            idx, url = idx_url
            try:
                return idx, get_download_url(url)
            except Exception as e:
                log(f"Failed to resolve datalink URL at index {idx}: {e}")
                return idx, None

        with ThreadPoolExecutor(max_workers=max_workers_local) as ex:
            futures = {ex.submit(_worker, (i, u)): i for i, u in enumerate(urls)}
            for fut in as_completed(futures):
                idx, result = fut.result()
                resolved[idx] = result

        n_ok = sum(1 for r in resolved if r)
        n_bad = len(resolved) - n_ok
        log(f"Resolved {n_ok} OK, {n_bad} failed.")
        return resolved

    download_urls = get_download_urls_multithreaded(datalink_urls, max_workers_local=max_workers)
    irsa_spectral_image_urls = [u for u in download_urls if u]
    log(
        f"Prepared {len(irsa_spectral_image_urls)} primary download URLs (e.g., AWS/Cutout derivable)."
    )

    assert len(download_urls) == len(results)

    # Build local file paths (and download if needed)
    file_paths = []
    if records is None:
        for i, irsa_spectral_image_url in zip(range(len(results)), irsa_spectral_image_urls):
            if i > max_images - 1:
                continue

            log(f"{i}/{len(results)}...")

            aws_spectral_image_url = irsa_spectral_image_url.replace(
                'https://irsa.ipac.caltech.edu/ibe/data/spherex',
                'https://nasa-irsa-spherex.s3.amazonaws.com'
            )
            irsa_cutout_image_url = (
                irsa_spectral_image_url + f"?center={ra_deg},{dec_deg}deg&size={int(size_pix)}pix"
            )

            local_fname = Path(irsa_spectral_image_url).name

            if use_cutout:
                cache_root = Path.home() / "local" / "SPHEREX" / "cutouts"
                subdir = cache_root / radecstr
                download_url = irsa_cutout_image_url
            else:
                cache_root = Path.home() / "local" / "SPHEREX" / "level2"

                def parse_subdir_from_fname(fname: str, root: Path):
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

                subdir = parse_subdir_from_fname(local_fname, cache_root) or cache_root
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
                        log(f"⚠️ Download failed: {e}. Will try current directory if available.")

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
        for ix, fpath in enumerate(file_paths):

            try:

                log(f"{ix}/{len(file_paths)} Doing photometry...")
                with fits.open(fpath, memmap=False) as hdul:
                    wcs = WCS(hdul['IMAGE'].header)
                    x, y = wcs.world_to_pixel(skycoord)

                    if not (0 <= x < 2040 and 0 <= y < 2040):
                        continue

                    # --- Aperture photometry ---
                    aperture = CircularAperture([(x, y)], r=APERTURE_RADIUS)
                    flux_img = hdul['IMAGE'].data

                    # Defaults if extensions are missing
                    extnames = {h.name.upper() for h in hdul}
                    var_img = hdul['VARIANCE'].data if 'VARIANCE' in extnames else np.full_like(flux_img, np.nan)
                    flags_img = hdul['FLAGS'].data if 'FLAGS' in extnames else np.zeros_like(flux_img, dtype=np.uint32)
                    zodi_img = hdul['ZODI'].data if 'ZODI' in extnames else np.zeros_like(flux_img)

                    # Background subtraction
                    if bkgd_method == 'zodi':
                        img_for_phot = flux_img - zodi_img
                        bkgd_level = None
                    else:
                        annulus = CircularAnnulus([(x, y)], r_in=annulus_r_in, r_out=annulus_r_out)
                        ann_mask_img = annulus.to_mask(method='center')[0].to_image(shape=flux_img.shape).astype(bool)
                        star_mask = detect_stars_mask(flux_img, fwhm_pix=star_fwhm, threshold_sigma=star_threshold_sigma)
                        valid = ann_mask_img & (~star_mask) & np.isfinite(flux_img)
                        if np.any(valid):
                            _, bkg_med, _ = sigma_clipped_stats(flux_img[valid], sigma=3.0, maxiters=5)
                            bkgd_level = float(bkg_med)
                        else:
                            bkgd_level = 0.0
                        img_for_phot = flux_img  # subtract constant after aperture sum

                    # Perform photometry
                    flux_tbl = aperture_photometry(img_for_phot, aperture)
                    var_tbl = aperture_photometry(var_img, aperture)

                    # Sum in aperture
                    flux_ap = flux_tbl['aperture_sum'][0]  # MJy/sr
                    flux_err = np.sqrt(var_tbl['aperture_sum'][0])  # MJy/sr

                    # If variance missing, Poisson fallback
                    if not np.isfinite(flux_err):
                        pos_flux_img = np.clip(flux_img, a_min=0, a_max=None)
                        pos_tbl = aperture_photometry(pos_flux_img, aperture)
                        flux_err = np.sqrt(pos_tbl['aperture_sum'][0])

                    # Subtract local background contribution if used
                    if bkgd_method == 'annulus' and bkgd_level is not None:
                        ap_area_pix = np.pi * (APERTURE_RADIUS ** 2)
                        flux_ap = flux_ap - bkgd_level * ap_area_pix
                    flux_val = flux_ap

                    # Evaluate mask across full image shape
                    mask = aperture.to_mask(method="center")[0]
                    aper_mask = mask.to_image(shape=flags_img.shape)
                    aper_flags = flags_img[aper_mask.astype(bool)]
                    is_flagged = np.any(aper_flags & BITMASK)

                    # Wavelength solution
                    wave_wcs = WCS(hdul[1].header, hdul, key='W')
                    wave_wcs.sip = None
                    lam, dlam = wave_wcs.wcs_pix2world(x, y, 0)

                    # Observation time (MJD-AVG)
                    mjd_avg = hdul['IMAGE'].header.get('MJD-AVG', np.nan)

                    # Convert MJy/sr to Jy using aperture area
                    pix_area_sr = (ARCSEC_PER_PIXEL / 3600 * np.pi / 180) ** 2
                    flux_jy = flux_val * pix_area_sr * 1e6
                    flux_err_jy = flux_err * pix_area_sr * 1e6

                    records.append({
                        "wavelength_um": lam,
                        "bandwidth_um": dlam,
                        "flux_jy": flux_jy,
                        "flux_err_jy": flux_err_jy,
                        "file": fpath,
                        "masked": is_flagged,
                        "mjd_avg": mjd_avg,
                        "bkgd_method": bkgd_method,
                        "aperture_radius_pix": APERTURE_RADIUS,
                        "annulus_r_in_pix": annulus_r_in if bkgd_method == 'annulus' else np.nan,
                        "annulus_r_out_pix": annulus_r_out if bkgd_method == 'annulus' else np.nan,
                    })

                    log(f"{fpath}")
                    log(
                        f"Pixel: ({x:.1f}, {y:.1f}), λ = {lam:.4f} µm ± {dlam:.4f}, Flux = {flux_jy:.3e} Jy"
                    )

            except Exception as e:
                print(f" {star_id} {ix} {fpath} failed with {e}.  skipping this image.")
                pass


    # Plot + CSV
    if records:
        records = sorted(records, key=lambda r: r["wavelength_um"])
        lam = np.array([r["wavelength_um"] for r in records])
        flux = np.array([r["flux_jy"] for r in records])
        err = np.array([r["flux_err_jy"] for r in records])
        masked_vals = [r.get("masked", False) for r in records]
        masked = np.array([
            bool(v) if isinstance(v, (bool, np.bool_)) else str(v).lower() in {"1", "true", "t", "yes"}
            for v in masked_vals
        ])

        if save_plot:
            plt.close('all')
            plt.figure(figsize=(8, 5))
            plt.errorbar(lam[~masked], flux[~masked], yerr=err[~masked], fmt='.', capsize=3)
            plt.errorbar(lam[masked], flux[masked], fmt='x', color='r', capsize=3)
            plt.yscale('log')
            plt.xlabel("Wavelength (µm)")
            plt.ylabel("Flux (Jy)")
            plt.title(f"SPHEREx Spectrum at RA={ra_deg:.4f}, Dec={dec_deg:.4f}")
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.tight_layout()
            savpath = outdir / f"result_{staridstr}{radecstr}{_a}.png"
            plt.savefig(str(savpath), dpi=300)
            log(f"Saved {savpath}")
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


__all__ = ["get_sx_spectrum"]
