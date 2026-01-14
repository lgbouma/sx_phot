"""Utilities for TIC/Gaia lookups and proper-motion propagation for SPHEREx."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
import pyvo
from astroquery.exceptions import ResolverError
from astroquery.mast import Catalogs, conf as mast_conf
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
SPHEREX_TAP_URL = "https://irsa.ipac.caltech.edu/TAP"
DEFAULT_GAIA_REF_EPOCH = 2015.5


def normalize_tic_id(tic_id: str | int) -> int:
    """Normalize a TIC identifier to its integer value.

    Args:
        tic_id: TIC identifier such as "TIC_300651846" or 300651846.

    Returns:
        Integer TIC identifier.

    Raises:
        ValueError: If the TIC identifier cannot be parsed.
    """
    if isinstance(tic_id, str):
        text = tic_id.strip().upper()
        if text.startswith("TIC"):
            text = text.replace("TIC", "", 1).lstrip("_").strip()
        if not text:
            raise ValueError("Empty TIC identifier.")
        return int(text)
    return int(tic_id)


def _log(message: str) -> None:
    """Log a timestamped message.

    Args:
        message: Message to print.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {col.lower(): col for col in columns}
    for candidate in candidates:
        key = lookup.get(candidate.lower())
        if key is not None:
            return key
    return None


def get_tic8_row(
    tic_id: int,
    cachedir: str | Path,
    timeout_sec: float = 60.0,
) -> pd.DataFrame:
    """Query TIC8 via astroquery and cache the first row.

    Args:
        tic_id: Integer TIC identifier.
        cachedir: Directory used to cache the TIC8 query results.
        timeout_sec: Per-request timeout (seconds) for MAST queries.

    Returns:
        DataFrame containing the cached TIC8 row with a `tic8_` prefix
        applied to columns.

    Raises:
        ResolverError: If the TIC entry cannot be resolved after retries.
    """
    cache_dir = Path(cachedir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"TIC_{tic_id}_mast_tic8_query.csv"

    if cache_path.exists():
        _log(f"Found {cache_path}, returning.")
        return pd.read_csv(cache_path)

    ticstr = f"TIC {tic_id}"
    max_iter = 3
    tic_data = None

    for ix in range(max_iter):
        try:
            with mast_conf.set_temp("timeout", int(timeout_sec)):
                tic_data = Catalogs.query_object(ticstr, catalog="TIC")
            if tic_data is None or len(tic_data) == 0:
                raise ResolverError(f"TIC {tic_id} returned no rows.")
            break
        except Exception as exc:
            time.sleep(3)
            tic_data = None
            _log(
                f"TIC {tic_id} failed MAST query with {exc}, "
                f"retrying {ix + 1}/{max_iter}..."
            )

    if tic_data is None:
        raise ResolverError(f"TIC {tic_id} failed to get MAST query.")

    t8_row = tic_data.to_pandas().iloc[[0]].copy()
    t8_row = t8_row.rename(
        {col: f"tic8_{col}" for col in t8_row.columns}, axis="columns"
    )

    t8_row.to_csv(cache_path, index=False)
    _log(f"Cached {cache_path}")
    return t8_row


def query_tic8_by_id(
    tic_id: int,
    timeout: float = 30.0,
    *,
    cachedir: str | Path | None = None,
) -> pd.DataFrame:
    """Query TIC8 for a single TIC ID via astroquery.

    Args:
        tic_id: Integer TIC identifier.
        timeout: Unused timeout placeholder for backward compatibility.
        cachedir: Optional cache directory; defaults to ./results/tic_motion.

    Returns:
        DataFrame containing the TIC8 row with `tic8_`-prefixed columns.

    Raises:
        ResolverError: If the TIC entry cannot be resolved after retries.
    """
    _ = timeout
    cache_root = (
        Path(cachedir)
        if cachedir is not None
        else Path.cwd() / "results" / "tic_motion"
    )
    return get_tic8_row(tic_id, cache_root)


def standardize_tic_row(tic_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize a TIC8 query row to canonical column names.

    Args:
        tic_df: DataFrame returned by `query_tic8_by_id`.

    Returns:
        DataFrame with columns: tic_id, tic_ra_deg, tic_dec_deg,
        gaia_dr2_source_id.

    Raises:
        ValueError: If required columns are missing.
    """
    if tic_df.empty:
        raise ValueError("TIC query returned no rows.")

    row = tic_df.iloc[0]
    columns = tic_df.columns

    id_key = _find_column(columns, ["id", "ticid", "tic_id", "tic8_id"])
    ra_key = _find_column(columns, ["ra", "ra_deg", "tic8_ra", "tic8_ra_deg"])
    dec_key = _find_column(
        columns, ["dec", "dec_deg", "tic8_dec", "tic8_dec_deg"]
    )
    gaia_key = _find_column(
        columns,
        [
            "gaia",
            "gaia_dr2",
            "gaia_dr2_id",
            "gaia_source_id",
            "gaia_id",
            "tic8_gaia",
            "tic8_gaia_dr2",
            "tic8_gaia_dr2_id",
            "tic8_gaia_source_id",
            "tic8_gaia_id",
        ],
    )

    missing = [
        name
        for name, key in (
            ("tic_id", id_key),
            ("ra", ra_key),
            ("dec", dec_key),
            ("gaia_dr2_source_id", gaia_key),
        )
        if key is None
    ]
    if missing:
        raise ValueError(f"Missing TIC8 columns: {', '.join(missing)}.")

    gaia_value = row[gaia_key]
    if pd.isna(gaia_value) or int(gaia_value) <= 0:
        raise ValueError("TIC row does not include a valid Gaia DR2 source ID.")

    return pd.DataFrame(
        [
            {
                "tic_id": int(row[id_key]),
                "tic_ra_deg": float(row[ra_key]),
                "tic_dec_deg": float(row[dec_key]),
                "gaia_dr2_source_id": int(gaia_value),
            }
        ]
    )


def query_gaia_dr2_astrometry(source_id: int) -> pd.DataFrame:
    """Query Gaia DR2 astrometry for a single source ID.

    Args:
        source_id: Gaia DR2 source identifier.

    Returns:
        DataFrame containing Gaia DR2 astrometry.

    Raises:
        ValueError: If the Gaia query returns no rows.
    """
    service = pyvo.dal.TAPService(GAIA_TAP_URL)
    query = (
        "SELECT source_id, ra, dec, pmra, pmdec, ref_epoch "
        "FROM gaiadr2.gaia_source "
        f"WHERE source_id = {source_id}"
    )
    results = service.search(query)
    df = results.to_table().to_pandas()
    if df.empty:
        raise ValueError(f"No Gaia DR2 data found for source_id={source_id}.")

    return df.rename(
        columns={
            "source_id": "gaia_source_id",
            "ra": "gaia_ra_deg",
            "dec": "gaia_dec_deg",
            "pmra": "pmra_masyr",
            "pmdec": "pmdec_masyr",
            "ref_epoch": "ref_epoch_jyear",
        }
    )


def query_spherex_obscore(ra_deg: float, dec_deg: float) -> pd.DataFrame:
    """Query the SPHEREx ObsCore table for observations covering a position.

    Args:
        ra_deg: Right ascension in degrees.
        dec_deg: Declination in degrees.

    Returns:
        DataFrame containing ObsCore metadata.
    """
    service = pyvo.dal.TAPService(SPHEREX_TAP_URL)
    query = (
        "SELECT obs_id, t_min, t_max, s_ra, s_dec "
        "FROM spherex.obscore "
        "WHERE CONTAINS(POINT('ICRS',"
        f"{ra_deg},{dec_deg}), s_region)=1"
    )

    job = service.submit_job(query)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=300)
    results = job.fetch_result()
    return results.to_table().to_pandas()


def add_mid_mjd(obscore_df: pd.DataFrame) -> pd.DataFrame:
    """Add a t_mid_mjd column to an ObsCore dataframe.

    Args:
        obscore_df: DataFrame with t_min and t_max columns.

    Returns:
        DataFrame with t_mid_mjd column populated.

    Raises:
        ValueError: If required columns are missing.
    """
    if not {"t_min", "t_max"}.issubset(obscore_df.columns):
        raise ValueError("ObsCore data missing t_min/t_max columns.")

    t_min = obscore_df["t_min"].to_numpy(dtype=float)
    t_max = obscore_df["t_max"].to_numpy(dtype=float)

    t_mid = np.where(
        np.isfinite(t_min) & np.isfinite(t_max),
        0.5 * (t_min + t_max),
        np.where(np.isfinite(t_min), t_min, t_max),
    )

    out = obscore_df.copy()
    out["t_mid_mjd"] = t_mid
    return out


def propagate_gaia_to_mjd(
    ra_deg: float,
    dec_deg: float,
    pmra_masyr: float,
    pmdec_masyr: float,
    mjd: np.ndarray | float,
    ref_epoch_jyear: float = DEFAULT_GAIA_REF_EPOCH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate Gaia astrometry to the requested MJD(s).

    Args:
        ra_deg: Gaia right ascension at the reference epoch (deg).
        dec_deg: Gaia declination at the reference epoch (deg).
        pmra_masyr: Proper motion in RA * cos(dec), mas/yr.
        pmdec_masyr: Proper motion in Dec, mas/yr.
        mjd: Target MJD or array of MJDs.
        ref_epoch_jyear: Reference epoch for the Gaia astrometry (Julian year).

    Returns:
        Tuple of (ra_deg, dec_deg) arrays at the requested MJD(s).
    """
    mjd_arr = np.atleast_1d(np.array(mjd, dtype=float))
    ref_time = Time(ref_epoch_jyear, format="jyear")
    target_time = Time(mjd_arr, format="mjd", scale="utc")

    coord = SkyCoord(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        pm_ra_cosdec=pmra_masyr * u.mas / u.yr,
        pm_dec=pmdec_masyr * u.mas / u.yr,
        distance=1 * u.pc,
        frame="icrs",
        obstime=ref_time,
    )

    new_coord = coord.apply_space_motion(new_obstime=target_time)
    return new_coord.ra.deg, new_coord.dec.deg


def load_or_query_csv(
    path: Path,
    query_func: Callable[[], pd.DataFrame],
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Load a cached CSV or query fresh data.

    Args:
        path: CSV path for cached data.
        query_func: Callable returning a pandas DataFrame.
        overwrite: If True, ignore the cache and query fresh data.

    Returns:
        DataFrame containing the cached or queried data.
    """
    if path.exists() and not overwrite:
        return pd.read_csv(path)

    df = query_func()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def propagate_obscore_positions(
    obscore_df: pd.DataFrame,
    gaia_row: Mapping[str, Any],
) -> pd.DataFrame:
    """Propagate Gaia astrometry to each ObsCore mid-time.

    Args:
        obscore_df: ObsCore DataFrame with t_mid_mjd column.
        gaia_row: Mapping with Gaia astrometry fields.

    Returns:
        DataFrame with propagated RA/Dec columns.
    """
    ra0 = float(gaia_row["gaia_ra_deg"])
    dec0 = float(gaia_row["gaia_dec_deg"])
    pmra = float(gaia_row["pmra_masyr"])
    pmdec = float(gaia_row["pmdec_masyr"])
    ref_epoch = float(gaia_row.get("ref_epoch_jyear", DEFAULT_GAIA_REF_EPOCH))

    if not np.isfinite([ra0, dec0, pmra, pmdec, ref_epoch]).all():
        raise ValueError("Gaia astrometry contains non-finite values.")

    mjd = obscore_df["t_mid_mjd"].to_numpy(dtype=float)
    ra_deg, dec_deg = propagate_gaia_to_mjd(
        ra_deg=ra0,
        dec_deg=dec0,
        pmra_masyr=pmra,
        pmdec_masyr=pmdec,
        mjd=mjd,
        ref_epoch_jyear=ref_epoch,
    )

    out = obscore_df.copy()
    out["ra_deg"] = ra_deg
    out["dec_deg"] = dec_deg
    return out


__all__ = [
    "normalize_tic_id",
    "get_tic8_row",
    "query_tic8_by_id",
    "standardize_tic_row",
    "query_gaia_dr2_astrometry",
    "query_spherex_obscore",
    "add_mid_mjd",
    "propagate_gaia_to_mjd",
    "load_or_query_csv",
    "propagate_obscore_positions",
]
