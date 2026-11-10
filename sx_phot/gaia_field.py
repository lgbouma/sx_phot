"""Gaia DR3 neighbour lists for forced PSF photometry.

For each catalog star, every Gaia DR3 source within ``radius_arcsec``
(default 80", enough for an 11 x 11 px box plus 4 px PSF stamps at
6.15"/px) is fetched with a TAP upload join against the Gaia archive.
Catalog positions are propagated to J2016.0 before the join, so the
separations are Gaia-epoch separations and the star itself appears at
sep ~ 0.

The query runs in chunks; each finished chunk is kept in
``<out>.parts/`` so an interrupted query resumes where it stopped.  The
archive truncates results at a row limit (50,000 for anonymous jobs as
of 2026-11), so a chunk that reaches ``max_rows`` is split in half and
queried again.
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sx_phot.bulk_phot import jyear_to_mjd, propagate_linear
from sx_phot.psf_phot import Neighbours, PsfParams

log = logging.getLogger(__name__)

GAIA_EPOCH_JYEAR = 2016.0
PIX_ARCSEC = 6.15
NEIGHBOUR_COLUMNS = ["star_idx", "source_id", "sep_arcsec", "ra", "dec",
                     "pmra", "pmdec", "phot_g_mean_mag"]

_SQL = """SELECT u.star_idx, n.source_id,
DISTANCE(POINT(u.ra, u.dec), POINT(n.ra, n.dec)) * 3600 AS sep_arcsec,
n.ra, n.dec, n.pmra, n.pmdec, n.phot_g_mean_mag
FROM tap_upload.cat AS u JOIN gaiadr3.gaia_source AS n
ON DISTANCE(POINT(u.ra, u.dec), POINT(n.ra, n.dec)) < {r} / 3600.0"""


def _query_chunk(cat: pd.DataFrame, radius_arcsec: float,
                 xml_path: Path, retries: int) -> pd.DataFrame:
    """Run one TAP upload join and return its rows."""
    from astropy.table import Table
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1
    Table({
        "star_idx": cat["star_idx"].to_numpy(np.int64),
        "ra": cat["ra"].to_numpy(float),
        "dec": cat["dec"].to_numpy(float),
    }).write(xml_path, format="votable", overwrite=True)
    sql = _SQL.format(r=float(radius_arcsec))
    for attempt in range(1, retries + 1):
        try:
            job = Gaia.launch_job_async(sql, upload_resource=str(xml_path),
                                        upload_table_name="cat")
            res = job.get_results().to_pandas()
            res.columns = [c.lower() for c in res.columns]
            return res[NEIGHBOUR_COLUMNS]
        except Exception as exc:  # noqa: BLE001 - archive hiccups
            log.warning("Gaia query attempt %d/%d failed: %r",
                        attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(30 * attempt)
    raise RuntimeError("unreachable")


def _query_split(cat: pd.DataFrame, radius_arcsec: float, xml_path: Path,
                 retries: int, max_rows: int) -> pd.DataFrame:
    """Query a chunk, halving it while the result hits the row limit."""
    res = _query_chunk(cat, radius_arcsec, xml_path, retries)
    if len(res) < max_rows:
        return res
    if len(cat) == 1:
        raise RuntimeError(
            f"star_idx {int(cat['star_idx'].iloc[0])}: >= {max_rows} "
            f"sources within {radius_arcsec} arcsec")
    log.info("  %d stars hit the %d-row limit; splitting", len(cat),
             max_rows)
    h = len(cat) // 2
    return pd.concat([
        _query_split(cat.iloc[:h], radius_arcsec, xml_path, retries,
                     max_rows),
        _query_split(cat.iloc[h:], radius_arcsec, xml_path, retries,
                     max_rows),
    ], ignore_index=True)


def query_gaia_neighbours(
    catalog: pd.DataFrame,
    out_path: Path,
    epoch_jyear: float = 2016.0,
    radius_arcsec: float = 80.0,
    chunk_size: int = 300,
    retries: int = 3,
    max_rows: int = 50000,
) -> pd.DataFrame:
    """Fetch Gaia DR3 sources around every catalog star.

    Args:
        catalog: Canonical catalog (``bulk_phot.load_catalog``).
        out_path: Output parquet path.
        epoch_jyear: Epoch of the catalog coordinates.
        radius_arcsec: Search radius around each star.
        chunk_size: Stars per TAP job.
        retries: Attempts per chunk.
        max_rows: Archive row limit; chunks reaching it are split.

    Returns:
        DataFrame with NEIGHBOUR_COLUMNS, sorted by star_idx and sep.
    """
    out_path = Path(out_path)
    parts = Path(str(out_path) + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    ra, dec = propagate_linear(
        catalog["ra"].to_numpy(), catalog["dec"].to_numpy(),
        catalog["pmra"].to_numpy(), catalog["pmdec"].to_numpy(),
        jyear_to_mjd(epoch_jyear), jyear_to_mjd(GAIA_EPOCH_JYEAR),
    )
    cat = pd.DataFrame({"star_idx": catalog["star_idx"].to_numpy(),
                        "ra": ra, "dec": dec})
    n_chunks = (len(cat) + chunk_size - 1) // chunk_size
    for k in range(n_chunks):
        part = parts / f"chunk_{k:05d}.parquet"
        if part.exists():
            continue
        t0 = time.time()
        res = _query_split(cat.iloc[k * chunk_size:(k + 1) * chunk_size],
                           radius_arcsec, parts / "upload.xml", retries,
                           max_rows)
        res.to_parquet(part, index=False)
        log.info("Gaia chunk %d/%d: %d sources in %.0f s", k + 1,
                 n_chunks, len(res), time.time() - t0)
    df = pd.concat([pd.read_parquet(p) for p in
                    sorted(parts.glob("chunk_*.parquet"))],
                   ignore_index=True)
    df = df.sort_values(["star_idx", "sep_arcsec"], kind="stable")
    df = df.reset_index(drop=True)
    tmp = Path(str(out_path) + ".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    shutil.rmtree(parts)
    log.info("Wrote %s: %d sources for %d of %d stars", out_path, len(df),
             df["star_idx"].nunique(), len(cat))
    return df


def load_neighbours(
    path: Path,
    n_stars: int,
    match_arcsec: float = 1.0,
    df: Optional[pd.DataFrame] = None,
    params: Optional[PsfParams] = None,
) -> Neighbours:
    """Load a neighbour table into the CSR layout used by the workers.

    Sources that the selection rules of ``psf_phot`` can never keep are
    dropped here (with a margin of 2 px on the separations): G >= G_t +
    dg_keep_near; G >= min(G_t + dg_keep, g_abs) beyond near_px; and
    G >= G_t + dg_out beyond the box half-diagonal.

    Args:
        path: Parquet written by ``query_gaia_neighbours``.
        n_stars: Number of catalog stars (star_idx runs 0..n_stars-1).
        match_arcsec: The nearest source within this separation is the
            star itself; stars without one get target = -1.
        df: Optional already-loaded table (``path`` is then ignored).
        params: PSF parameters used for the prefilter.

    Returns:
        Neighbours instance.
    """
    p = params or PsfParams()
    if df is None:
        df = pd.read_parquet(path, columns=NEIGHBOUR_COLUMNS)
    df = df.sort_values(["star_idx", "sep_arcsec"], kind="stable")
    df = df.reset_index(drop=True)
    sidx = df["star_idx"].to_numpy(np.int64)
    sep = df["sep_arcsec"].to_numpy(float)
    g = df["phot_g_mean_mag"].to_numpy(float)
    g = np.where(np.isfinite(g), g, 21.0)
    first = np.r_[True, sidx[1:] != sidx[:-1]]
    is_t = first & (sep < float(match_arcsec))
    gt = np.full(sidx.max() + 1 if sidx.size else 0, np.nan)
    gt[sidx[is_t]] = g[is_t]
    gt = gt[sidx] if sidx.size else gt
    near = (p.near_px + 2.0) * PIX_ARCSEC
    diag = (p.half * np.sqrt(2.0) + 2.0) * PIX_ARCSEC
    drop = (g >= gt + p.dg_keep_near)
    drop |= (g >= np.minimum(gt + p.dg_keep, p.g_abs)) & (sep > near)
    drop |= (g >= gt + p.dg_out) & (sep > diag)
    keep = is_t | ~drop       # NaN gt (no target): comparisons False
    df = df[keep].reset_index(drop=True)
    sidx = df["star_idx"].to_numpy(np.int64)
    counts = np.bincount(sidx, minlength=n_stars)[:n_stars]
    offsets = np.zeros(n_stars + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    sep = df["sep_arcsec"].to_numpy(float)
    first = offsets[:-1]
    has = counts > 0
    target = np.full(n_stars, -1, dtype=np.int64)
    ok = has.copy()
    ok[has] = sep[first[has]] < float(match_arcsec)
    target[ok] = first[ok]
    g = df["phot_g_mean_mag"].to_numpy(float)
    return Neighbours(
        offsets=offsets,
        ra=df["ra"].to_numpy(float),
        dec=df["dec"].to_numpy(float),
        pmra=np.nan_to_num(df["pmra"].to_numpy(float)),
        pmdec=np.nan_to_num(df["pmdec"].to_numpy(float)),
        g=np.where(np.isfinite(g), g, 21.0),
        target=target,
    )


__all__ = [
    "GAIA_EPOCH_JYEAR",
    "NEIGHBOUR_COLUMNS",
    "load_neighbours",
    "query_gaia_neighbours",
]
