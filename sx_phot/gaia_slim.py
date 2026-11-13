"""Local slim copy of Gaia DR3 (source_id, position, proper motion, G).

The ESA bulk CDN is IPv6-only and unreachable from wh3, so the copy is
pulled from the Heidelberg ARI mirror's TAP service (hard limit 10M
rows per job) by walking ``source_id`` in order: each job returns the
next ``max_rows`` sources above the last id seen, and the result is
written as one parquet part.  Parts are therefore sorted by
``source_id``; since the top bits of ``source_id`` are the HEALPix
level-12 (nside 4096, nested) pixel, the parts and their row groups are
spatially ordered and a cone search needs no extra index.

Layout of ``out_dir``::

    parts/part_00000.parquet ...   sorted rows, one per TAP job
    manifest.csv                   part, min_id, max_id, n_rows
    fetch.log                      progress (via the driver)

Resuming after an interruption continues from the manifest.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ARI_TAP = "https://gaia.ari.uni-heidelberg.de/tap"
COLUMNS = ["source_id", "ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"]
DTYPES = {"source_id": np.int64, "ra": np.float64, "dec": np.float64,
          "pmra": np.float32, "pmdec": np.float32,
          "phot_g_mean_mag": np.float32}
HEALPIX_SHIFT = 35          # source_id >> 35 is the level-12 pixel
DEFAULT_MAX_ROWS = 10_000_000

_SQL = ("SELECT TOP {n} source_id, ra, dec, pmra, pmdec, phot_g_mean_mag "
        "FROM gaiadr3.gaia_source WHERE source_id > {last} "
        "ORDER BY source_id")


def _to_frame(table) -> pd.DataFrame:
    """Convert an astropy table to the slim frame with fixed dtypes."""
    out = {}
    for c in COLUMNS:
        col = np.ma.getdata(table[c])
        mask = np.ma.getmaskarray(table[c])
        if np.issubdtype(DTYPES[c], np.integer):
            # Never route ids through float: source_id > 2**53 would be
            # rounded.  Gaia ids are never null.
            if mask.any():
                raise ValueError(f"masked values in integer column {c}")
            out[c] = np.asarray(col).astype(DTYPES[c])
        else:
            out[c] = np.where(mask, np.nan, col).astype(DTYPES[c])
    df = pd.DataFrame(out)
    return df.sort_values("source_id").reset_index(drop=True)


def fetch_chunk(service, last_id: int, max_rows: int,
                timeout_s: float = 4 * 3600) -> pd.DataFrame:
    """Run one async TAP job for the sources above ``last_id``.

    Args:
        service: ``pyvo.dal.TAPService``.
        last_id: Largest source_id already stored.
        max_rows: Rows requested (the mirror caps at 10M).
        timeout_s: Wait limit for the job.

    Returns:
        Frame sorted by source_id (fewer than ``max_rows`` rows at the
        end of the catalog).
    """
    job = service.submit_job(_SQL.format(n=int(max_rows), last=int(last_id)),
                             maxrec=int(max_rows))
    try:
        job.run()
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=timeout_s)
        if job.phase != "COMPLETED":
            raise RuntimeError(f"TAP job ended in phase {job.phase}")
        return _to_frame(job.fetch_result().to_table())
    finally:
        try:
            job.delete()
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass


def read_manifest(out_dir: Path) -> pd.DataFrame:
    """Return the manifest (empty frame if none yet)."""
    p = Path(out_dir) / "manifest.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame(columns=["part", "min_id", "max_id", "n_rows"])


def fetch_all(
    out_dir: Path,
    max_rows: int = DEFAULT_MAX_ROWS,
    tap_url: str = ARI_TAP,
    retries: int = 5,
    pause_s: float = 5.0,
    max_parts: Optional[int] = None,
) -> pd.DataFrame:
    """Walk the whole catalog in source_id order, resuming from parts.

    Args:
        out_dir: Output directory (created).
        max_rows: Rows per job.
        tap_url: TAP endpoint.
        retries: Attempts per chunk before giving up.
        pause_s: Pause between jobs.
        max_parts: Stop after this many new parts (testing).

    Returns:
        The manifest after the run.
    """
    import pyvo

    out_dir = Path(out_dir)
    parts = out_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    man = read_manifest(out_dir)
    last_id = int(man["max_id"].max()) if len(man) else -1
    k = int(man["part"].max()) + 1 if len(man) else 0
    if len(man) and int(man["n_rows"].iloc[-1]) < max_rows:
        log.info("Catalog already complete (%d parts)", len(man))
        return man
    service = pyvo.dal.TAPService(tap_url)
    n_new = 0
    while True:
        t0 = time.time()
        for attempt in range(1, retries + 1):
            try:
                df = fetch_chunk(service, last_id, max_rows)
                break
            except Exception as exc:  # noqa: BLE001 - archive hiccups
                log.warning("part %d attempt %d/%d failed: %r", k, attempt,
                            retries, exc)
                if attempt == retries:
                    raise
                time.sleep(60 * attempt)
        if not len(df):
            log.info("No rows above source_id %d; done", last_id)
            break
        if int(df["source_id"].iloc[0]) <= last_id:
            raise RuntimeError("TAP result is not above the cursor")
        path = parts / f"part_{k:05d}.parquet"
        tmp = path.with_suffix(".parquet.part")
        df.to_parquet(tmp, index=False, compression="zstd",
                      row_group_size=1 << 16)
        tmp.replace(path)
        row = {"part": k, "min_id": int(df["source_id"].iloc[0]),
               "max_id": int(df["source_id"].iloc[-1]), "n_rows": len(df)}
        man = pd.concat([man, pd.DataFrame([row])], ignore_index=True)
        man.to_csv(out_dir / "manifest.csv", index=False)
        log.info("part %d: %d rows, source_id %d..%d, %.0f s", k, len(df),
                 row["min_id"], row["max_id"], time.time() - t0)
        last_id = row["max_id"]
        k += 1
        n_new += 1
        if len(df) < max_rows:
            log.info("Last part was short; catalog complete")
            break
        if max_parts is not None and n_new >= max_parts:
            break
        time.sleep(pause_s)
    return man


def cone_id_ranges(ra_deg: float, dec_deg: float,
                   radius_arcsec: float) -> list[tuple[int, int]]:
    """Return the source_id ranges [lo, hi) that can hold a cone's sources.

    Args:
        ra_deg: Cone centre RA (deg).
        dec_deg: Cone centre Dec (deg).
        radius_arcsec: Cone radius.

    Returns:
        Sorted, merged half-open ranges built from the level-12 nested
        HEALPix pixels that overlap the cone.
    """
    import astropy.units as u
    from astropy_healpix import HEALPix

    hp = HEALPix(nside=1 << 12, order="nested")
    pix = np.sort(hp.cone_search_lonlat(ra_deg * u.deg, dec_deg * u.deg,
                                        radius_arcsec * u.arcsec))
    ranges: list[tuple[int, int]] = []
    for p in pix.astype(np.int64):
        lo, hi = int(p) << HEALPIX_SHIFT, int(p + 1) << HEALPIX_SHIFT
        if ranges and ranges[-1][1] == lo:
            ranges[-1] = (ranges[-1][0], hi)
        else:
            ranges.append((lo, hi))
    return ranges


class GaiaSlim:
    """Cone searches on the local slim copy.

    Args:
        out_dir: Directory written by ``fetch_all``.

    Raises:
        FileNotFoundError: If the manifest is missing.
    """

    def __init__(self, out_dir: Path) -> None:
        self.dir = Path(out_dir)
        man = read_manifest(self.dir)
        if not len(man):
            raise FileNotFoundError(f"no manifest in {self.dir}")
        self.parts = man["part"].to_numpy(np.int64)
        self.min_id = man["min_id"].to_numpy(np.int64)
        self.max_id = man["max_id"].to_numpy(np.int64)
        self._stats: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _part_path(self, k: int) -> Path:
        return self.dir / "parts" / f"part_{k:05d}.parquet"

    def _row_group_stats(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Min and max source_id of every row group of a part."""
        import pyarrow.parquet as pq

        if k not in self._stats:
            meta = pq.ParquetFile(self._part_path(k)).metadata
            col = COLUMNS.index("source_id")
            lo = np.array([meta.row_group(i).column(col).statistics.min
                           for i in range(meta.num_row_groups)], np.int64)
            hi = np.array([meta.row_group(i).column(col).statistics.max
                           for i in range(meta.num_row_groups)], np.int64)
            self._stats[k] = (lo, hi)
        return self._stats[k]

    def ids_between(self, lo: int, hi: int) -> pd.DataFrame:
        """Rows with lo <= source_id < hi, reading only needed row groups."""
        import pyarrow.parquet as pq

        frames = []
        sel = np.flatnonzero((self.max_id >= lo) & (self.min_id < hi))
        for k in self.parts[sel]:
            rg_lo, rg_hi = self._row_group_stats(int(k))
            groups = np.flatnonzero((rg_hi >= lo) & (rg_lo < hi))
            if not groups.size:
                continue
            table = pq.ParquetFile(self._part_path(int(k))).read_row_groups(
                groups.tolist(), columns=COLUMNS)
            df = table.to_pandas()
            frames.append(df[(df["source_id"] >= lo) & (df["source_id"] < hi)])
        if not frames:
            return pd.DataFrame({c: np.zeros(0, DTYPES[c]) for c in COLUMNS})
        return pd.concat(frames, ignore_index=True)

    def cone(self, ra_deg: float, dec_deg: float,
             radius_arcsec: float) -> pd.DataFrame:
        """Sources within ``radius_arcsec`` of a position (J2016.0).

        Args:
            ra_deg: Centre RA (deg).
            dec_deg: Centre Dec (deg).
            radius_arcsec: Radius.

        Returns:
            Frame with COLUMNS plus ``sep_arcsec``, sorted by separation.
        """
        parts = [self.ids_between(lo, hi)
                 for lo, hi in cone_id_ranges(ra_deg, dec_deg, radius_arcsec)]
        df = pd.concat(parts, ignore_index=True) if parts else \
            pd.DataFrame({c: np.zeros(0, DTYPES[c]) for c in COLUMNS})
        ra = np.radians(df["ra"].to_numpy())
        dec = np.radians(df["dec"].to_numpy())
        ra0, dec0 = np.radians(ra_deg), np.radians(dec_deg)
        cosd = (np.sin(dec) * np.sin(dec0)
                + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0))
        sep = np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0))) * 3600.0
        df["sep_arcsec"] = sep
        df = df[sep <= radius_arcsec].sort_values("sep_arcsec")
        return df.reset_index(drop=True)


__all__ = [
    "ARI_TAP",
    "COLUMNS",
    "DEFAULT_MAX_ROWS",
    "GaiaSlim",
    "HEALPIX_SHIFT",
    "cone_id_ranges",
    "fetch_all",
    "fetch_chunk",
    "read_manifest",
]
