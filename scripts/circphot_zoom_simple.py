"""
Circular aperture photometry for SPHEREx Level-2 MEF images.

Summary
- Queries the IRSA TAP service for SPHEREx observations overlapping a
  hard‑coded target (ra_deg, dec_deg).
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

Usage
  python circphot.py [options]

Options
- --do-photometry         Force re-computation of photometry. If cached CSV
                          exists, reuses its 'file' paths to avoid downloads.
- --bkgd-method {zodi,annulus}
                          Background subtraction method (default: zodi).
- --aperture-radius R     Aperture radius in pixels (default: 2).
- --annulus-r-in RIN      Annulus inner radius in pixels (default: 6).
- --annulus-r-out ROUT    Annulus outer radius in pixels (default: 8).
- --star-fwhm FWHM        Assumed FWHM in pixels for star masking (default: 2).
- --star-threshold-sigma S
                          Detection threshold in sigma for star masking (default: 5).
- --max-images N          Maximum number of images to process (default: 200).
- --use-cutout / --no-use-cutout
                          Use IRSA cutout service (default: use cutout).

Notes
- Aperture area is converted from MJy/sr to Jy using the pixel solid angle.
"""

import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)
import os
import re
import argparse
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import SkyCoord
import pandas as pd
import pyvo
from pyvo.dal.adhoc import DatalinkResults
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
from pathlib import Path
from datetime import datetime

def log(message: str) -> None:
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {message}")

parser = argparse.ArgumentParser(description="Circular aperture photometry with optional local-annulus background.")
parser.add_argument('--do-photometry', action='store_true', help='Force recomputation of photometry from FITS files in records or query results.')
parser.add_argument('--bkgd-method', default='zodi', choices=['zodi', 'annulus'], help='Background subtraction method: zodi image or local annulus.')
parser.add_argument('--aperture-radius', type=float, default=2.0, help='Circular aperture radius in pixels.')
parser.add_argument('--annulus-r-in', type=float, default=6.0, help='Background annulus inner radius in pixels (when using annulus method).')
parser.add_argument('--annulus-r-out', type=float, default=8.0, help='Background annulus outer radius in pixels (when using annulus method).')
parser.add_argument('--star-fwhm', type=float, default=2.0, help='Assumed FWHM (pixels) for star detection (annulus masking).')
parser.add_argument('--star-threshold-sigma', type=float, default=5.0, help='Detection threshold in sigma for star masking in annulus.')
parser.add_argument('--max-images', type=int, default=200, help='Maximum number of images to process from query results.')
parser.add_argument('--use-cutout', dest='use_cutout', action='store_true', default=True, help='Use IRSA cutout service for downloads (default: True).')
parser.add_argument('--no-use-cutout', dest='use_cutout', action='store_false', help='Disable cutouts; download full MEF from AWS mirror.')
args = parser.parse_args()

ra_deg, dec_deg = 0.8568686344700, -46.830975607410 # A star calib
ra_deg, dec_deg = 114.36670927895, -66.75737858669 # TIC 3006
ra_deg, dec_deg = 157.03727470138, -64.50520903903 # TOI-837
ra_deg, dec_deg = 207.52599561891, -40.83589917965 # HIP 67522

MAX_N_IMAGES = args.max_images

# CSV cache of processed photometry records
radecstr = f"ra{str(ra_deg).replace('.','p')}_{str(dec_deg).replace('.','p')}"
_a = f'_{args.bkgd_method}'
cache_csv = Path(f"sxphot_cache_{radecstr}{_a}.csv")

skycoord = SkyCoord(ra = ra_deg, dec = dec_deg, unit = 'degree', frame = 'icrs')

# Copy-paste from
# https://caltech-ipac.github.io/irsa-tutorials/tutorials/spherex/spherex_intro.html

# Define the TAP service URL for IRSA
tap_url = "https://irsa.ipac.caltech.edu/TAP"

# Connect to the TAP service
service = pyvo.dal.TAPService(tap_url)

# Define your ADQL query
query = "SELECT * FROM spherex.obscore WHERE CONTAINS(POINT('ICRS',"+str(ra_deg)+","+str(dec_deg)+"), s_region)=1"

# Submit the asynchronous query
job = service.submit_job(query)

# Run the job (starts the query execution on the server)
job.run()

# Wait for the job to complete (polling)
job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=300)

# Capture the results
# Each row of the results of your query represents a different spectral image.
# Because SPHEREx data will be released on a weekly basis, the number of rows
# returned will change depending on when you submit the query. Let’s see how many
# images are returned today.
results = job.fetch_result()

records = None  # Will hold list of dicts when populated
if cache_csv.exists():
    try:
        df_cache = pd.read_csv(cache_csv)
        if (len(df_cache) >= len(results)) or (len(df_cache) >= MAX_N_IMAGES):
            log(f"Using cached CSV {cache_csv} with {len(df_cache)} records (>= {len(results)} results). Skipping downloads.")
            records = df_cache.to_dict(orient='records')
        else:
            log(f"Cached CSV has {len(df_cache)} records (< {len(results)} results). Will download/process all FITS.")
    except Exception as e:
        log(f"⚠️ Failed to read cache CSV {cache_csv}: {e}. Will download/process all FITS.")


datalink_urls = []
for i in range(len(results)):
    datalink_urls.append(results['access_url'][i])

def get_download_url(datalink_url):
    datalink_content = DatalinkResults.from_result_url(datalink_url)
    irsa_spectral_image_url = next(datalink_content.bysemantics("#this")).access_url
    return irsa_spectral_image_url


# --- Multithreaded helpers to resolve datalink URLs ---
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_download_urls_multithreaded(datalink_urls, max_workers: int = 8):
    """Resolve a list of Datalink URLs to their primary download URLs in parallel.

    Returns a list of equal length in the same order as ``datalink_urls``.
    Any failures are logged and returned as None in the corresponding slot.
    """
    if not datalink_urls:
        return []

    max_workers = int(max(1, max_workers))
    resolved = [None] * len(datalink_urls)

    log(f"Resolving {len(datalink_urls)} datalink URLs with {max_workers} threads...")

    def _worker(idx_url):
        idx, url = idx_url
        try:
            return idx, get_download_url(url)
        except Exception as e:
            log(f"Failed to resolve datalink URL at index {idx}: {e}")
            return idx, None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_worker, (i, u)): i for i, u in enumerate(datalink_urls)}
        for fut in as_completed(futures):
            idx, result = fut.result()
            resolved[idx] = result

    n_ok = sum(1 for r in resolved if r)
    n_bad = len(resolved) - n_ok
    log(f"Resolved {n_ok} OK, {n_bad} failed.")
    return resolved


# Example call pattern: resolve to spectral image URLs before interactive work
_max_workers = 40
download_urls = get_download_urls_multithreaded(datalink_urls, max_workers=_max_workers)
# Optionally drop Nones
irsa_spectral_image_urls = [u for u in download_urls if u]
log(f"Prepared {len(irsa_spectral_image_urls )} primary download URLs (e.g., AWS/Cutout derivable).")

assert len(download_urls) == len(results)

# Link results from datalink url
file_paths = []
if records is None:
    for i, irsa_spectral_image_url in zip(
        range(len(results)), irsa_spectral_image_urls
    ):

        if i > MAX_N_IMAGES - 1:
            continue

        log(f"{i}/{len(results)}...")

        # The irsa_spectral_image_url url will look like
        # 'https://irsa.ipac.caltech.edu/ibe/data/spherex/qr/level2/2025W25_1B/l2b-v12-2025-178/3/level2_2025W25_1B_0325_1D3_spx_l2b-v12-2025-178.fits'
        aws_spectral_image_url = irsa_spectral_image_url.replace(
            'https://irsa.ipac.caltech.edu/ibe/data/spherex',
            'https://nasa-irsa-spherex.s3.amazonaws.com'
        )
        irsa_cutout_image_url = (
            irsa_spectral_image_url + f"?center={ra_deg},{dec_deg}deg&size=64pix"
        )

        # Determine cache directory based on use_cutout:
        # - Full MEF:  ~/local/SPHEREX/level2/<week_cam>/<l2b>/<detnum>/<filename>
        # - Cutouts:   ~/local/SPHEREX/cutouts/<radecstr>/<filename>
        local_fname = Path(irsa_spectral_image_url).name

        if args.use_cutout:
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

# Do photometry
from astropy import units as u
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.stats import sigma_clipped_stats
APERTURE_RADIUS = float(args.aperture_radius)
ARCSEC_PER_PIXEL = 6.1

# --- Bitmask for SPHEREx bad flags ---
BITMASK = (
    (1 << 0)  | (1 << 1)  | (1 << 2)  |
    (1 << 6)  | (1 << 7)  | (1 << 9)  |
    (1 << 10) | (1 << 11) | (1 << 15)
)

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


# Decide whether to (re)compute photometry.
do_photometry = (records is None) or bool(args.do_photometry)

# If we are recomputing and we had cached records, reuse their file paths.
if (records is not None) and do_photometry:
    # Pull file paths from cached records
    file_paths = [str(r.get('file')) for r in records if 'file' in r]

if do_photometry:
    records = []
    for ix, fpath in enumerate(file_paths):
        log(f"{ix}/{len(file_paths)} Doing photometry...")
        with fits.open(fpath, memmap=False) as hdul:
            wcs = WCS(hdul['IMAGE'].header)
            x, y = wcs.world_to_pixel(skycoord)

            if not (0 <= x < 2040 and 0 <= y < 2040):
                continue

            # --- Aperture photometry ---
            aperture = CircularAperture([(x, y)], r=APERTURE_RADIUS)
            flux_img = hdul['IMAGE'].data

            # Some cutouts may omit VARIANCE, FLAGS, or ZODI extensions.
            # Provide sensible defaults when missing.
            extnames = {h.name.upper() for h in hdul}
            if 'VARIANCE' in extnames:
                var_img = hdul['VARIANCE'].data
            else:
                var_img = np.full_like(flux_img, np.nan)
            if 'FLAGS' in extnames:
                flags_img = hdul['FLAGS'].data
            else:
                flags_img = np.zeros_like(flux_img, dtype=np.uint32)
            if 'ZODI' in extnames:
                zodi_img = hdul['ZODI'].data
            else:
                zodi_img = np.zeros_like(flux_img)

            # Background subtraction
            if args.bkgd_method == 'zodi':
                img_for_phot = flux_img - zodi_img
                bkgd_level = None
            else:
                # Local background: median within annulus excluding detected stars
                annulus = CircularAnnulus([(x, y)], r_in=args.annulus_r_in, r_out=args.annulus_r_out)
                ann_mask_img = annulus.to_mask(method='center')[0].to_image(shape=flux_img.shape).astype(bool)
                star_mask = detect_stars_mask(flux_img, fwhm_pix=args.star_fwhm, threshold_sigma=args.star_threshold_sigma)
                valid = ann_mask_img & (~star_mask) & np.isfinite(flux_img)
                if np.any(valid):
                    # Robust median of local annulus pixels
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

            # If variance was unavailable (NaNs), fall back to Poisson-limited error.
            if not np.isfinite(flux_err):
                # Poisson fallback: sqrt of summed positive signal within the aperture.
                pos_flux_img = np.clip(flux_img, a_min=0, a_max=None)
                pos_tbl = aperture_photometry(pos_flux_img, aperture)
                flux_err = np.sqrt(pos_tbl['aperture_sum'][0])

            # Subtract local background contribution if used
            if args.bkgd_method == 'annulus' and bkgd_level is not None:
                ap_area_pix = np.pi * (APERTURE_RADIUS ** 2)
                flux_ap = flux_ap - bkgd_level * ap_area_pix
            flux = flux_ap

            # Evaluate mask across full image shape
            mask = aperture.to_mask(method="center")[0]
            aper_mask = mask.to_image(shape=flags_img.shape)
            aper_flags = flags_img[aper_mask.astype(bool)]
            is_flagged = np.any(aper_flags & BITMASK)

            # NOTE these are the lines that yield a long info warning:
            # Inconsistent SIP distortion information is present in the FITS header and the WCS object:
            # SIP coefficients were detected, but CTYPE is missing a "-SIP" suffix.
            # astropy.wcs is using the SIP distortion coefficients,
            # therefore the coordinates calculated here might be incorrect.
            wave_wcs = WCS(hdul[1].header, hdul, key='W')
            wave_wcs.sip = None
            lam, dlam = wave_wcs.wcs_pix2world(x, y, 0)

            # --- Observation time (MJD-AVG) ---
            mjd_avg = hdul['IMAGE'].header.get('MJD-AVG', np.nan)

            # --- Convert from MJy/sr to Jy using aperture area ---
            pix_area_sr = (ARCSEC_PER_PIXEL / 3600 * np.pi / 180) ** 2
            flux_jy = flux * pix_area_sr * 1e6
            flux_err_jy = flux_err * pix_area_sr * 1e6

            records.append({
                "wavelength_um": lam,
                "bandwidth_um": dlam,
                "flux_jy": flux_jy,
                "flux_err_jy": flux_err_jy,
                "file": fpath,
                "masked": is_flagged,
                "mjd_avg": mjd_avg,
                "bkgd_method": args.bkgd_method,
                "aperture_radius_pix": APERTURE_RADIUS,
                "annulus_r_in_pix": args.annulus_r_in if args.bkgd_method == 'annulus' else np.nan,
                "annulus_r_out_pix": args.annulus_r_out if args.bkgd_method == 'annulus' else np.nan,
            })

            log(f"{fpath}")
            log(f"Pixel: ({x:.1f}, {y:.1f}), λ = {lam:.4f} µm ± {dlam:.4f}, Flux = {flux_jy:.3e} Jy")

# --- Plot the spectrum ---
if records:
    records = sorted(records, key=lambda r: r["wavelength_um"])
    lam = np.array([r["wavelength_um"] for r in records])
    flux = np.array([r["flux_jy"] for r in records])
    err = np.array([r["flux_err_jy"] for r in records])
    # Robustly coerce 'masked' to boolean for plotting
    masked_vals = [r.get("masked", False) for r in records]
    masked = np.array([
        bool(v) if isinstance(v, (bool, np.bool_)) else str(v).lower() in {"1", "true", "t", "yes"}
        for v in masked_vals
    ])

    plt.figure(figsize=(8, 5))
    plt.errorbar(lam[~masked], flux[~masked], yerr=err[~masked], fmt='.', capsize=3)
    plt.errorbar(lam[masked], flux[masked], fmt='x', color = 'r', capsize=3)
    plt.yscale('log')
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Flux (Jy)")
    plt.title(f"SPHEREx Spectrum at RA={ra_deg:.4f}, Dec={dec_deg:.4f}")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    #plt.ylim([2e-3, 2e-1])
    savpath = f'result_{radecstr}{_a}.png'
    plt.savefig(savpath, dpi=300)
    log(f"Saved {savpath}")
    # Save/update CSV cache after plotting
    try:
        pd.DataFrame(records).to_csv(cache_csv, index=False)
        log(f"Saved {len(records)} records to cache CSV: {cache_csv}")
    except Exception as e:
        log(f"⚠️ Failed to save cache CSV {cache_csv}: {e}")
else:
    log("No valid measurements to plot.")
