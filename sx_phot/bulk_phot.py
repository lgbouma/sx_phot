"""Image-centric bulk aperture photometry for SPHEREx Level-2 images.

Each local MEF is opened once and every catalog star that falls on it is
measured in one vectorized pass.  The per-star numbers reproduce
``sx_phot.circphot.get_sx_spectrum`` (exact-overlap circular aperture,
sigma-clipped annulus background with a global star mask, flag bitmask,
WAVE-TAB wavelength), with one documented difference: the global
star-mask statistics may be computed on a pixel subsample
(``PhotConfig.stats_stride``), which changes the mask threshold by <1%.

Work is sharded by detector directory (``<week>__<l2b version>__D<n>``);
each pass over a shard writes ``pass_<stamp>.parquet`` plus the list of
image keys it processed, so re-running a run directory only processes
files that arrived since the previous pass.

With ``PhotConfig.psf`` set, every star is also measured by Gaia-forced
PSF photometry (``sx_phot.psf_phot``), which adds the ``psf_*`` columns
and needs a Gaia neighbour table (``sx_phot.gaia_field``).
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.time import Time
from astropy.wcs import WCS
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)
from scipy.spatial import cKDTree

from sx_phot.circphot import ARCSEC_PER_PIXEL, BITMASK, TJD_OFFSET
from sx_phot.image_index import COARSE_RADIUS_DEG, NAXIS, unit_vectors
from sx_phot.psf_phot import (
    ARCSEC2_SR,
    PSF_DTYPES,
    Neighbours,
    PsfParams,
    ZonePsf,
    fit_image,
)

log = logging.getLogger(__name__)

PIX_AREA_SR = (ARCSEC_PER_PIXEL / 3600 * np.pi / 180) ** 2
MJY_SR_TO_JY = PIX_AREA_SR * 1e6
DEFAULT_EPOCH_JYEAR = 2016.0

ROW_DTYPES = {
    "star_idx": np.int32,
    "ticid": np.int64,
    "image_key": np.int64,
    "x": np.float32,
    "y": np.float32,
    "flux_jy": np.float32,
    "flux_err_jy": np.float32,
    "flux_raw_jy": np.float32,
    "bkg_jy": np.float32,
    "zodi_jy": np.float32,
    "flag_or": np.int32,
    "n_bad_pix": np.int16,
    "n_ap_pix": np.int16,
    "n_ann_pix": np.int16,
    "bkg_mode": np.int8,
    "wavelength_um": np.float32,
    "bandwidth_um": np.float32,
    "mjd_avg": np.float64,
}


@dataclass
class PhotConfig:
    """Photometry parameters (defaults match sx_phot.circphot).

    Attributes:
        aperture_radius: Circular aperture radius in pixels.
        annulus_r_in: Annulus inner radius in pixels.
        annulus_r_out: Annulus outer radius in pixels.
        star_threshold_sigma: Star-mask threshold above the sky median.
        clip_sigma: Sigma for all sigma-clipped statistics.
        clip_maxiters: Max iterations for sigma clipping.
        stats_stride: Pixel stride for the global sky statistics (1
            reproduces circphot exactly; 8 is about ten times faster).
        min_ann_pix: If fewer unmasked pixels than this remain in the
            annulus (e.g. at the bright band edges, where the global
            star mask flags the whole region), fall back to the
            sigma-clipped median of all finite annulus pixels.
        read_zodi: Also measure the ZODI model in the aperture.
        coarse_radius_deg: Catalog prefilter radius around image centers.
        min_file_age_sec: Skip files modified more recently than this
            (protects against reading files still being downloaded).
        epoch_jyear: Catalog coordinate epoch (Julian year).
        psf: Also run Gaia-forced PSF photometry (adds psf_* columns).
        psf_drop_frac: Drop PSF neighbours with at most this fraction of
            their light on usable box pixels (0: only those with none).
        psf_bg_order: PSF background polynomial (1 plane, 2 quadratic).
    """

    aperture_radius: float = 2.0
    annulus_r_in: float = 6.0
    annulus_r_out: float = 8.0
    star_threshold_sigma: float = 5.0
    clip_sigma: float = 3.0
    clip_maxiters: int = 5
    stats_stride: int = 8
    min_ann_pix: int = 20
    read_zodi: bool = True
    coarse_radius_deg: float = COARSE_RADIUS_DEG
    min_file_age_sec: float = 600.0
    epoch_jyear: float = DEFAULT_EPOCH_JYEAR
    psf: bool = False
    psf_drop_frac: float = 0.0
    psf_bg_order: int = 1


def row_dtypes(config: Optional[PhotConfig] = None) -> dict:
    """Return the output column dtypes for a configuration.

    Args:
        config: Photometry parameters (None: aperture only).

    Returns:
        ROW_DTYPES, plus PSF_DTYPES when ``config.psf`` is set.
    """
    if config is not None and config.psf:
        return {**ROW_DTYPES, **PSF_DTYPES}
    return dict(ROW_DTYPES)


def jyear_to_mjd(epoch_jyear: float) -> float:
    """Convert a Julian epoch to MJD."""
    return float(Time(float(epoch_jyear), format="jyear").mjd)


def propagate_linear(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    pmra_masyr: np.ndarray,
    pmdec_masyr: np.ndarray,
    epoch_mjd: float,
    mjd: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate positions linearly in proper motion to a target MJD.

    Args:
        ra_deg: RA at the catalog epoch (deg).
        dec_deg: Dec at the catalog epoch (deg).
        pmra_masyr: Proper motion in RA times cos(dec), mas/yr.
        pmdec_masyr: Proper motion in Dec, mas/yr.
        epoch_mjd: Catalog epoch as MJD.
        mjd: Target MJD.

    Returns:
        Tuple (ra_deg, dec_deg) at the target epoch.
    """
    dt_yr = (float(mjd) - float(epoch_mjd)) / 365.25
    dec = np.asarray(dec_deg, dtype=float) + (
        np.nan_to_num(pmdec_masyr) / 3.6e6 * dt_yr
    )
    cosd = np.cos(np.deg2rad(np.asarray(dec_deg, dtype=float)))
    ra = np.asarray(ra_deg, dtype=float) + (
        np.nan_to_num(pmra_masyr) / 3.6e6 * dt_yr / cosd
    )
    return np.mod(ra, 360.0), dec


def _empty_rows(config: Optional[PhotConfig] = None
                ) -> dict[str, np.ndarray]:
    """Return an empty row dict with the canonical dtypes."""
    return {k: np.zeros(0, dtype=v) for k, v in row_dtypes(config).items()}


def _psf_columns(
    hdul: fits.HDUList,
    img: np.ndarray,
    flags: np.ndarray,
    var: np.ndarray,
    wcs: WCS,
    mjd: float,
    sidx: np.ndarray,
    nb: Neighbours,
    cfg: PhotConfig,
) -> dict[str, np.ndarray]:
    """Forced PSF columns for the stars ``sidx`` on one image.

    Args:
        hdul: Open Level-2 HDUList.
        img: IMAGE array.
        flags: FLAGS array.
        var: VARIANCE array.
        wcs: Image WCS.
        mjd: Image MJD-AVG.
        sidx: Catalog star_idx of the stars on the image.
        nb: Gaia neighbour table.
        cfg: Photometry parameters.

    Returns:
        Dict of PSF_DTYPES columns aligned with ``sidx``.
    """
    sidx = np.asarray(sidx, dtype=np.int64)
    a, b = nb.offsets[sidx], nb.offsets[sidx + 1]
    counts = b - a
    starts = np.zeros(sidx.size + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    rows = np.repeat(a - starts[:-1], counts) + np.arange(starts[-1])
    targets = np.where(nb.target[sidx] >= 0, nb.target[sidx] - a, -1)
    ra_t, dec_t = propagate_linear(
        nb.ra[rows], nb.dec[rows], nb.pmra[rows], nb.pmdec[rows],
        jyear_to_mjd(2016.0), mjd,
    )
    nx, ny = wcs.all_world2pix(ra_t, dec_t, 0, quiet=True)
    params = PsfParams(drop_frac=float(cfg.psf_drop_frac),
                       bg_order=int(cfg.psf_bg_order))
    zpsf = ZonePsf(hdul, reach=params.reach)
    omega = float(hdul["IMAGE"].header["OMEGA_MEDIAN"]) * ARCSEC2_SR
    return fit_image(img, flags, var, zpsf, omega, nx, ny, nb.g[rows],
                     starts, targets, params)


def photometer_image(
    path: str,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    pmra_masyr: Optional[np.ndarray] = None,
    pmdec_masyr: Optional[np.ndarray] = None,
    star_idx: Optional[np.ndarray] = None,
    ticid: Optional[np.ndarray] = None,
    config: Optional[PhotConfig] = None,
    neighbours: Optional[Neighbours] = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Measure every given star that falls on one Level-2 image.

    Args:
        path: Local MEF path.
        ra_deg: Candidate star RA at the catalog epoch (deg).
        dec_deg: Candidate star Dec at the catalog epoch (deg).
        pmra_masyr: Proper motion in RA*cos(dec) (mas/yr), or None.
        pmdec_masyr: Proper motion in Dec (mas/yr), or None.
        star_idx: Catalog row index per star (default arange).
        ticid: Catalog identifier per star (default star_idx).
        config: Photometry parameters.
        neighbours: Gaia neighbour table indexed by star_idx (required
            when ``config.psf`` is set).

    Returns:
        Tuple (rows, meta).  ``rows`` maps column name to array (one entry
        per star on the image; ``image_key`` is left zero for the caller
        to fill).  ``meta`` holds per-image diagnostics.

    Raises:
        ValueError: If ``config.psf`` is set without ``neighbours``.
    """
    cfg = config or PhotConfig()
    if cfg.psf and neighbours is None:
        raise ValueError("PSF photometry needs a Gaia neighbour table")
    t0 = time.time()
    ra_deg = np.atleast_1d(np.asarray(ra_deg, dtype=float))
    dec_deg = np.atleast_1d(np.asarray(dec_deg, dtype=float))
    n = ra_deg.size
    if pmra_masyr is None:
        pmra_masyr = np.zeros(n)
    if pmdec_masyr is None:
        pmdec_masyr = np.zeros(n)
    if star_idx is None:
        star_idx = np.arange(n)
    if ticid is None:
        ticid = np.asarray(star_idx)
    meta = {"path": path, "n_cand": int(n), "n_in": 0}
    rows = _empty_rows(cfg)
    if n == 0:
        meta["elapsed_s"] = time.time() - t0
        return rows, meta

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with fits.open(path, memmap=False) as hdul:
            hdr = hdul["IMAGE"].header
            wcs = WCS(hdr)
            mjd_avg = float(hdr.get("MJD-AVG", np.nan))
            meta["mjd_avg"] = mjd_avg

            ra_t, dec_t = propagate_linear(
                ra_deg, dec_deg, pmra_masyr, pmdec_masyr,
                jyear_to_mjd(cfg.epoch_jyear), mjd_avg,
            )
            x, y = wcs.all_world2pix(ra_t, dec_t, 0, quiet=True)
            inside = (
                np.isfinite(x) & np.isfinite(y)
                & (x >= 0) & (x < NAXIS) & (y >= 0) & (y < NAXIS)
            )
            n_in = int(inside.sum())
            meta["n_in"] = n_in
            if n_in == 0:
                meta["elapsed_s"] = time.time() - t0
                return rows, meta

            sel = np.where(inside)[0]
            x = x[sel]
            y = y[sel]
            xy = np.column_stack([x, y])

            img = hdul["IMAGE"].data
            extnames = {h.name.upper() for h in hdul}
            var = (
                hdul["VARIANCE"].data if "VARIANCE" in extnames
                else np.full_like(img, np.nan)
            )
            flags = (
                hdul["FLAGS"].data if "FLAGS" in extnames
                else np.zeros(img.shape, dtype=np.int32)
            )
            zodi = None
            if cfg.read_zodi and "ZODI" in extnames:
                zodi = hdul["ZODI"].data

            # Global sky statistics and star mask (shared by all stars).
            finite = np.isfinite(img)
            vals = img[finite]
            stride = max(1, int(cfg.stats_stride))
            if vals.size:
                _, sky_med, sky_std = sigma_clipped_stats(
                    vals[::stride], sigma=cfg.clip_sigma,
                    maxiters=cfg.clip_maxiters,
                )
                star_mask = np.zeros(img.shape, dtype=bool)
                star_mask[finite] = vals > (
                    sky_med + cfg.star_threshold_sigma * sky_std
                )
            else:
                sky_med, sky_std = np.nan, np.nan
                star_mask = np.zeros(img.shape, dtype=bool)
            meta["sky_med"] = float(sky_med)
            meta["sky_std"] = float(sky_std)

            # Aperture sums (exact overlap, as in circphot).
            aperture = CircularAperture(xy, r=float(cfg.aperture_radius))
            flux_sum = np.asarray(
                aperture_photometry(img, aperture)["aperture_sum"], float
            )
            var_sum = np.asarray(
                aperture_photometry(var, aperture)["aperture_sum"], float
            )
            flux_err = np.sqrt(var_sum)
            bad_err = ~np.isfinite(flux_err)
            if bad_err.any():
                pos = np.clip(img, a_min=0, a_max=None)
                sub = CircularAperture(
                    xy[bad_err], r=float(cfg.aperture_radius)
                )
                pos_sum = np.asarray(
                    aperture_photometry(pos, sub)["aperture_sum"], float
                )
                flux_err[bad_err] = np.sqrt(pos_sum)
            zodi_sum = np.full(n_in, np.nan)
            if zodi is not None:
                zodi_sum = np.asarray(
                    aperture_photometry(zodi, aperture)["aperture_sum"], float
                )

            # Annulus background: sigma-clipped median of unmasked pixels
            # whose centers fall in the annulus (circphot 'annulus').
            annulus = CircularAnnulus(
                xy, r_in=float(cfg.annulus_r_in),
                r_out=float(cfg.annulus_r_out),
            )
            stats = ApertureStats(
                img, annulus, mask=(star_mask | ~finite),
                sigma_clip=SigmaClip(
                    sigma=cfg.clip_sigma, maxiters=cfg.clip_maxiters
                ),
            )
            bkg = np.asarray(stats.median, dtype=float)
            n_ann = np.nan_to_num(
                np.asarray(stats.center_aper_area, dtype=float)
            )
            # Fallback where the star mask removed (nearly) the whole
            # annulus: at the LVF band edges the sky is several times
            # brighter than the image median, so the global threshold
            # flags every pixel there and the background would be 0.
            bkg_mode = np.zeros(n_in, dtype=np.int8)
            fallback = ~np.isfinite(bkg) | (n_ann < int(cfg.min_ann_pix))
            if fallback.any():
                sub = CircularAnnulus(
                    xy[fallback], r_in=float(cfg.annulus_r_in),
                    r_out=float(cfg.annulus_r_out),
                )
                sub_stats = ApertureStats(
                    img, sub, mask=~finite,
                    sigma_clip=SigmaClip(
                        sigma=cfg.clip_sigma, maxiters=cfg.clip_maxiters
                    ),
                )
                sub_bkg = np.asarray(sub_stats.median, dtype=float)
                sub_n = np.nan_to_num(
                    np.asarray(sub_stats.center_aper_area, dtype=float)
                )
                bkg[fallback] = sub_bkg
                n_ann[fallback] = sub_n
                bkg_mode[fallback] = 1
            bkg_mode[~np.isfinite(bkg)] = 2
            bkg = np.where(np.isfinite(bkg), bkg, 0.0)
            ap_area = np.pi * float(cfg.aperture_radius) ** 2

            # Flags over pixels whose centers fall in the aperture.
            flag_or = np.zeros(n_in, dtype=np.int32)
            n_bad = np.zeros(n_in, dtype=np.int16)
            n_ap = np.zeros(n_in, dtype=np.int16)
            for i, m in enumerate(aperture.to_mask(method="center")):
                sl = m.get_overlap_slices(flags.shape)
                if sl[0] is None:
                    continue
                vals_f = flags[sl[0]][m.data[sl[1]] > 0]
                if vals_f.size == 0:
                    continue
                flag_or[i] = np.bitwise_or.reduce(vals_f.astype(np.int32))
                n_bad[i] = np.count_nonzero(vals_f & BITMASK)
                n_ap[i] = vals_f.size

            # Wavelength lookup (WAVE-TAB), no SIP, as in circphot.
            wave_wcs = WCS(hdul[1].header, hdul, key="W")
            wave_wcs.sip = None
            lam, dlam = wave_wcs.wcs_pix2world(x, y, 0)

            if cfg.psf:
                t_psf = time.time()
                psf_rows = _psf_columns(
                    hdul, img, flags, var, wcs, mjd_avg,
                    np.asarray(star_idx)[sel], neighbours, cfg,
                )
                meta["psf_s"] = time.time() - t_psf

    rows = {
        "star_idx": np.asarray(star_idx)[sel].astype(np.int32),
        "ticid": np.asarray(ticid)[sel].astype(np.int64),
        "image_key": np.zeros(n_in, dtype=np.int64),
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "flux_jy": ((flux_sum - bkg * ap_area) * MJY_SR_TO_JY).astype(
            np.float32
        ),
        "flux_err_jy": (flux_err * MJY_SR_TO_JY).astype(np.float32),
        "flux_raw_jy": (flux_sum * MJY_SR_TO_JY).astype(np.float32),
        "bkg_jy": (bkg * ap_area * MJY_SR_TO_JY).astype(np.float32),
        "zodi_jy": (zodi_sum * MJY_SR_TO_JY).astype(np.float32),
        "flag_or": flag_or,
        "n_bad_pix": n_bad,
        "n_ap_pix": n_ap,
        "n_ann_pix": np.nan_to_num(n_ann).astype(np.int16),
        "bkg_mode": bkg_mode,
        "wavelength_um": np.asarray(lam, float).astype(np.float32),
        "bandwidth_um": np.asarray(dlam, float).astype(np.float32),
        "mjd_avg": np.full(n_in, mjd_avg, dtype=np.float64),
    }
    if cfg.psf:
        rows.update(psf_rows)
    meta["elapsed_s"] = time.time() - t0
    return rows, meta


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
def load_catalog(
    path: Path,
    id_col: str = "TICID",
    ra_col: str = "ra",
    dec_col: str = "dec",
    pmra_col: Optional[str] = "pmra",
    pmdec_col: Optional[str] = "pmdec",
) -> pd.DataFrame:
    """Load a star catalog into the canonical bulk-photometry layout.

    Args:
        path: feather, parquet or CSV file.
        id_col: Integer identifier column.
        ra_col: RA column (deg).
        dec_col: Dec column (deg).
        pmra_col: Proper motion RA*cos(dec) column (mas/yr) or None.
        pmdec_col: Proper motion Dec column (mas/yr) or None.

    Returns:
        DataFrame with star_idx, ticid, ra, dec, pmra, pmdec (missing
        proper motions set to zero).

    Raises:
        ValueError: If the coordinates are not finite.
    """
    path = Path(path)
    cols = [c for c in (id_col, ra_col, dec_col, pmra_col, pmdec_col) if c]
    if path.suffix in (".feather", ".ftr"):
        df = pd.read_feather(path, columns=cols)
    elif path.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path, columns=cols)
    else:
        df = pd.read_csv(path, usecols=cols)
    out = pd.DataFrame({
        "star_idx": np.arange(len(df), dtype=np.int32),
        "ticid": df[id_col].to_numpy().astype(np.int64),
        "ra": df[ra_col].to_numpy(dtype=float),
        "dec": df[dec_col].to_numpy(dtype=float),
        "pmra": (
            np.nan_to_num(df[pmra_col].to_numpy(dtype=float))
            if pmra_col else np.zeros(len(df))
        ),
        "pmdec": (
            np.nan_to_num(df[pmdec_col].to_numpy(dtype=float))
            if pmdec_col else np.zeros(len(df))
        ),
    })
    if not (np.isfinite(out["ra"]).all() and np.isfinite(out["dec"]).all()):
        raise ValueError("Catalog contains non-finite coordinates.")
    return out


# ---------------------------------------------------------------------------
# Worker state (inherited by forked processes)
# ---------------------------------------------------------------------------
_CAT: dict[str, np.ndarray] = {}
_TREE: Optional[cKDTree] = None
_CFG: PhotConfig = PhotConfig()
_NB: Optional[Neighbours] = None


def _set_worker_state(
    catalog: pd.DataFrame,
    config: PhotConfig,
    neighbours: Optional[Neighbours] = None,
) -> None:
    """Store catalog arrays, the KD-tree and neighbours in globals."""
    global _CAT, _TREE, _CFG, _NB
    _CAT = {
        c: np.ascontiguousarray(catalog[c].to_numpy())
        for c in ("star_idx", "ticid", "ra", "dec", "pmra", "pmdec")
    }
    _TREE = cKDTree(unit_vectors(_CAT["ra"], _CAT["dec"]))
    _CFG = config
    _NB = neighbours


def _work(task: tuple) -> tuple:
    """Process one image: (image_key, path, s_ra, s_dec) -> results.

    Returns:
        Tuple (image_key, rows or None, meta, error string or None).
    """
    image_key, path, s_ra, s_dec = task
    try:
        age = time.time() - os.path.getmtime(path)
        if age < _CFG.min_file_age_sec:
            return image_key, None, {"path": path}, "too_recent"
        chord = 2.0 * np.sin(np.deg2rad(_CFG.coarse_radius_deg) / 2.0)
        cand = np.asarray(
            _TREE.query_ball_point(unit_vectors(s_ra, s_dec), r=chord),
            dtype=np.int64,
        )
        rows, meta = photometer_image(
            path,
            _CAT["ra"][cand], _CAT["dec"][cand],
            _CAT["pmra"][cand], _CAT["pmdec"][cand],
            star_idx=_CAT["star_idx"][cand], ticid=_CAT["ticid"][cand],
            config=_CFG, neighbours=_NB,
        )
        rows["image_key"][:] = image_key
        return image_key, rows, meta, None
    except Exception as exc:  # noqa: BLE001 - keep the pool alive
        return image_key, None, {"path": path}, repr(exc)


# ---------------------------------------------------------------------------
# Run directory and shards
# ---------------------------------------------------------------------------
def init_run_dir(
    run_dir: Path,
    catalog: pd.DataFrame,
    index: pd.DataFrame,
    config: PhotConfig,
    catalog_source: str = "",
    neighbours_source: str = "",
) -> Path:
    """Create or validate a run directory and snapshot its inputs.

    Args:
        run_dir: Run directory.
        catalog: Canonical catalog (see ``load_catalog``).
        index: Image index DataFrame (refreshed on every call).
        config: Photometry parameters (written once; later calls must
            match the stored values).
        catalog_source: Description of the catalog origin for config.json.
        neighbours_source: Gaia neighbour table path (PSF runs).

    Returns:
        The run directory path.

    Raises:
        ValueError: If an existing run has a different catalog or config.
    """
    run_dir = Path(run_dir)
    (run_dir / "phot").mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    cat_path = run_dir / "catalog.parquet"
    cfg = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "catalog_source": catalog_source,
        "n_stars": int(len(catalog)),
        "ticid_sum": int(catalog["ticid"].to_numpy().sum()),
        "phot": asdict(config),
    }
    if config.psf:
        cfg["neighbours_source"] = neighbours_source
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        same_cat = (
            old["n_stars"] == cfg["n_stars"]
            and old["ticid_sum"] == cfg["ticid_sum"]
        )
        if not same_cat:
            raise ValueError(f"{run_dir} was created with a different catalog")
        # Keys added after a run was created compare at their defaults,
        # so older run directories stay valid.
        defaults = asdict(PhotConfig())
        keys = [k for k in asdict(config) if k != "min_file_age_sec"]
        diff = {k: (old["phot"].get(k, defaults[k]), cfg["phot"][k])
                for k in keys
                if old["phot"].get(k, defaults[k]) != cfg["phot"][k]}
        if diff:
            raise ValueError(f"{run_dir} config differs: {diff}")
    else:
        cfg_path.write_text(json.dumps(cfg, indent=2))
        catalog.to_parquet(cat_path, index=False)
    tmp = run_dir / "index.parquet.part"
    index.to_parquet(tmp, index=False)
    os.replace(tmp, run_dir / "index.parquet")
    return run_dir


def processed_keys(shard_dir: Path) -> np.ndarray:
    """Return image keys already processed in a shard directory."""
    keys = [
        np.load(p)
        for p in sorted(Path(shard_dir).glob("pass_*.keys.npy"))
    ]
    if not keys:
        return np.zeros(0, dtype=np.int64)
    return np.unique(np.concatenate(keys)).astype(np.int64)


def plan_shards(
    run_dir: Path,
    index: pd.DataFrame,
    shard_glob: Optional[str] = None,
    max_shards: Optional[int] = None,
    max_files_per_shard: Optional[int] = None,
) -> list[tuple[str, pd.DataFrame]]:
    """List shards with files that still need processing.

    Args:
        run_dir: Run directory.
        index: Image index (rows without a local path are ignored).
        shard_glob: Optional fnmatch pattern on shard names.
        max_shards: Optional cap on the number of shards returned.
        max_files_per_shard: Optional cap on files per shard (testing).

    Returns:
        List of (shard name, DataFrame of index rows to process).
    """
    import fnmatch

    local = index[index["path"].notna()]
    plan = []
    for shard, grp in local.groupby("shard", sort=True):
        if shard_glob and not fnmatch.fnmatch(shard, shard_glob):
            continue
        done = processed_keys(Path(run_dir) / "phot" / shard)
        todo = grp[~grp["image_key"].isin(done)].sort_values("path")
        if max_files_per_shard:
            todo = todo.head(int(max_files_per_shard))
        if len(todo) == 0:
            continue
        plan.append((shard, todo))
        if max_shards and len(plan) >= max_shards:
            break
    return plan


def _write_pass(
    shard_dir: Path,
    stamp: str,
    chunks: list[dict[str, np.ndarray]],
    keys_done: list[int],
    summary: dict,
    dtypes: Optional[dict] = None,
) -> int:
    """Write one pass (parquet, keys, summary json) for a shard."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    shard_dir.mkdir(parents=True, exist_ok=True)
    cols = {
        k: np.concatenate([c[k] for c in chunks]) if chunks
        else np.zeros(0, dtype=v)
        for k, v in (dtypes or ROW_DTYPES).items()
    }
    table = pa.table({k: pa.array(v) for k, v in cols.items()})
    tmp = shard_dir / f"pass_{stamp}.parquet.part"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, shard_dir / f"pass_{stamp}.parquet")
    np.save(shard_dir / f"pass_{stamp}.keys.npy",
            np.asarray(keys_done, dtype=np.int64))
    summary["n_rows"] = int(table.num_rows)
    (shard_dir / f"pass_{stamp}.json").write_text(
        json.dumps(summary, indent=1)
    )
    return int(table.num_rows)


def run_bulk_photometry(
    run_dir: Path,
    catalog: pd.DataFrame,
    index: pd.DataFrame,
    config: Optional[PhotConfig] = None,
    nproc: int = 64,
    shard_glob: Optional[str] = None,
    max_shards: Optional[int] = None,
    max_files_per_shard: Optional[int] = None,
    chunksize: int = 2,
    neighbours: Optional[Neighbours] = None,
) -> dict:
    """Run image-centric photometry over all unprocessed local images.

    Args:
        run_dir: Run directory (created by ``init_run_dir`` or here).
        catalog: Canonical catalog.
        index: Image index.
        config: Photometry parameters.
        nproc: Worker processes.
        shard_glob: Optional shard-name pattern (testing / partial runs).
        max_shards: Optional cap on shards processed.
        max_files_per_shard: Optional cap on files per shard (testing).
        chunksize: Pool imap chunk size.
        neighbours: Gaia neighbour table (required for PSF runs).

    Returns:
        Dict with totals (files, rows, failed, elapsed_s).

    Raises:
        ValueError: If ``config.psf`` is set without ``neighbours``.
    """
    cfg = config or PhotConfig()
    if cfg.psf and neighbours is None:
        raise ValueError("PSF photometry needs a Gaia neighbour table")
    run_dir = Path(run_dir)
    plan = plan_shards(run_dir, index, shard_glob, max_shards,
                       max_files_per_shard)
    n_files_total = sum(len(g) for _, g in plan)
    log.info("Plan: %d shards, %d files to process, %d workers",
             len(plan), n_files_total, nproc)
    totals = {"files": 0, "rows": 0, "failed": 0, "too_recent": 0,
              "elapsed_s": 0.0}
    if not plan:
        return totals

    _set_worker_state(catalog, cfg, neighbours)
    ctx = get_context("fork")
    t_start = time.time()
    with ctx.Pool(int(nproc)) as pool:
        for ishard, (shard, grp) in enumerate(plan):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tasks = list(zip(
                grp["image_key"].to_numpy(), grp["path"].to_numpy(),
                grp["s_ra"].to_numpy(), grp["s_dec"].to_numpy(),
            ))
            t0 = time.time()
            chunks: list[dict[str, np.ndarray]] = []
            keys_done: list[int] = []
            failed: list[tuple[str, str]] = []
            n_recent = 0
            n_in_sum = 0
            psf_s = 0.0
            for key, rows, meta, err in pool.imap_unordered(
                _work, tasks, chunksize=chunksize
            ):
                if err == "too_recent":
                    n_recent += 1
                    continue
                if err is not None:
                    failed.append((meta.get("path", ""), err))
                    continue
                keys_done.append(int(key))
                n_in_sum += int(meta.get("n_in", 0))
                psf_s += float(meta.get("psf_s", 0.0))
                if rows["star_idx"].size:
                    chunks.append(rows)
            dt = time.time() - t0
            summary = {
                "shard": shard, "stamp": stamp, "n_files": len(tasks),
                "n_ok": len(keys_done), "n_failed": len(failed),
                "n_too_recent": n_recent, "elapsed_s": round(dt, 1),
                "files_per_s": round(len(keys_done) / max(dt, 1e-9), 2),
                "stats_stride": cfg.stats_stride, "failed": failed[:200],
            }
            if cfg.psf:
                summary["psf_cpu_s"] = round(psf_s, 1)
            n_rows = _write_pass(run_dir / "phot" / shard, stamp, chunks,
                                 keys_done, summary, row_dtypes(cfg))
            totals["files"] += len(keys_done)
            totals["rows"] += n_rows
            totals["failed"] += len(failed)
            totals["too_recent"] += n_recent
            elapsed = time.time() - t_start
            rate = totals["files"] / max(elapsed, 1e-9)
            remaining = n_files_total - totals["files"] - totals["failed"] \
                - totals["too_recent"]
            log.info(
                "[%d/%d] %s: %d files (%d failed, %d too recent) -> %d rows "
                "in %.0f s (%.1f files/s); total %d files, %.1f files/s, "
                "ETA %.1f h",
                ishard + 1, len(plan), shard, len(keys_done), len(failed),
                n_recent, n_rows, dt, summary["files_per_s"],
                totals["files"], rate, remaining / max(rate, 1e-9) / 3600,
            )
            for p, e in failed[:5]:
                log.warning("  failed %s: %s", p, e)
    totals["elapsed_s"] = time.time() - t_start
    log.info("Done: %d files, %d rows, %d failed, %d too recent, %.1f h",
             totals["files"], totals["rows"], totals["failed"],
             totals["too_recent"], totals["elapsed_s"] / 3600)
    return totals


__all__ = [
    "DEFAULT_EPOCH_JYEAR",
    "MJY_SR_TO_JY",
    "PhotConfig",
    "ROW_DTYPES",
    "TJD_OFFSET",
    "init_run_dir",
    "jyear_to_mjd",
    "load_catalog",
    "photometer_image",
    "plan_shards",
    "processed_keys",
    "propagate_linear",
    "row_dtypes",
    "run_bulk_photometry",
]
