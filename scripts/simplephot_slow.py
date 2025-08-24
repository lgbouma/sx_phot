import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', category=FITSFixedWarning)
import os
import re
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

ra_deg, dec_deg = 0.8568686344700, -46.830975607410
MAX_N_IMAGES = 200

# CSV cache of processed photometry records
radecstr = f"ra{str(ra_deg).replace('.','p')}_{str(dec_deg).replace('.','p')}"
cache_csv = Path(f"sxphot_cache_{radecstr}.csv")

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

# Link results from datalink url
file_paths = []
if records is None:
    for i in range(len(results)):

        if i > MAX_N_IMAGES - 1:
            continue

        log(f"{i}/{len(results)}...")

        # The ‘access_url’ column is particularly important because it tells you how
        # to access the data. Let’s look at the ‘access_url’ value for the first row:
        # e.g.
        # 'https://irsa.ipac.caltech.edu/datalink/links/spherex?ID=ivo://irsa.ipac/spherex_qr?2025W25_1B_0325_1/D3'
        datalink_url = results['access_url'][i]

        # This url does not provide direct access to the SPHEREx spectral image.
        # Rather, it returns a file that lists all data products and services
        # associated with this image.  For the quick release products, this
        # includes a spectral image MEF and a cutout service.
        datalink_content = DatalinkResults.from_result_url(datalink_url)

        # Use the primary spectral image MEF product.
        # This url will look like
        # 'https://irsa.ipac.caltech.edu/ibe/data/spherex/qr/level2/2025W25_1B/l2b-v12-2025-178/3/level2_2025W25_1B_0325_1D3_spx_l2b-v12-2025-178.fits'
        irsa_spectral_image_url = next(datalink_content.bysemantics("#this")).access_url
        aws_spectral_image_url = irsa_spectral_image_url.replace(
            'https://irsa.ipac.caltech.edu/ibe/data/spherex',
            'https://nasa-irsa-spherex.s3.amazonaws.com'
        )

        # FIXME TODO FIXME TODO: convert the url to its AWS equivalent:
        # https://nasa-irsa-spherex.s3.amazonaws.com/qr/level2/2025W25_1B/l2b-v12-2025-178/3/level2_2025W25_1B_0325_1D3_spx_l2b-v12-2025-178.fits

        # Determine cache directory root for level2 files (structured layout):
        #   <root>/<week_cam>/<l2b>/<detnum>/<filename>
        # where <root> defaults to ~/local/SPHEREX/level2 or env SPHEREX_CACHE
        cache_root = Path(os.environ.get(
            "SPHEREX_CACHE",
            str(Path.home() / "local" / "SPHEREX" / "level2")
        ))
        cache_root.mkdir(parents=True, exist_ok=True)

        # Build structured subdir from filename, e.g.
        # level2_2025W27_1A_0253_1D3_spx_l2b-v13-2025-190.fits
        local_fname = Path(irsa_spectral_image_url).name

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
        local_path = subdir / local_fname

        # Download file if not already present
        if not local_path.exists():
            # Fallback: consider old flat layout under cache root
            flat_path = cache_root / local_fname
            if flat_path.exists():
                local_path = flat_path
            else:
                try:
                    subdir.mkdir(parents=True, exist_ok=True)
                    log(f"Downloading to {local_path} ...")
                    urlretrieve(aws_spectral_image_url, local_path)
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
from photutils.aperture import CircularAperture, aperture_photometry
APERTURE_RADIUS = 2
ARCSEC_PER_PIXEL = 6.1

# --- Bitmask for SPHEREx bad flags ---
BITMASK = (
    (1 << 0)  | (1 << 1)  | (1 << 2)  |
    (1 << 6)  | (1 << 7)  | (1 << 9)  |
    (1 << 10) | (1 << 11) | (1 << 15)
)

if records is None:
    records = []
    for fpath in file_paths:
        with fits.open(fpath, memmap=False) as hdul:
            wcs = WCS(hdul['IMAGE'].header)
            x, y = wcs.world_to_pixel(skycoord)

            if not (0 <= x < 2040 and 0 <= y < 2040):
                continue

            # --- Aperture photometry ---
            aperture = CircularAperture([(x, y)], r=APERTURE_RADIUS)
            flux_img = hdul['IMAGE'].data
            var_img = hdul['VARIANCE'].data
            flags_img = hdul['FLAGS'].data
            zodi_img = hdul['ZODI'].data
            flux_img -= zodi_img
            flux_tbl = aperture_photometry(flux_img, aperture)
            var_tbl = aperture_photometry(var_img, aperture)

            flux = flux_tbl['aperture_sum'][0]  # MJy/sr
            flux_err = np.sqrt(var_tbl['aperture_sum'][0])  # MJy/sr

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
    plt.errorbar(lam[~masked], flux[~masked], yerr=err[~masked], fmt='o', capsize=3)
    plt.errorbar(lam[masked], flux[masked], fmt='x', color = 'r', capsize=3)
    plt.yscale('log')
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Flux (Jy)")
    plt.title(f"SPHEREx Spectrum at RA={ra_deg:.4f}, Dec={dec_deg:.4f}")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    savpath = f'result_{radecstr}.png'
    plt.savefig(savpath)
    log(f"Saved {savpath}")
    # Save/update CSV cache after plotting
    try:
        pd.DataFrame(records).to_csv(cache_csv, index=False)
        log(f"Saved {len(records)} records to cache CSV: {cache_csv}")
    except Exception as e:
        log(f"⚠️ Failed to save cache CSV {cache_csv}: {e}")
else:
    log("No valid measurements to plot.")
