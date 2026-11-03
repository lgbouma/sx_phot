"""Local SPHEREx image index: one row per (obs_id, detector) image.

The index replaces the per-target IRSA TAP query and the per-image
datalink resolution with a local table that joins the IRSA ObsCore
footprints to the Level-2 files mirrored on disk.

Sources
- ObsCore snapshot (primary): one bulk TAP query (about 3.5 min for
  1.6M rows) gives obs_id, detector, release tag, image center,
  footprint polygon and observation times.
- FITS headers (fallback): local files absent from the snapshot are
  indexed from their IMAGE headers (CRVAL, TAN-SIP corners, MJD range).

Key
- ``image_key`` (int64) encodes release tag, week, camera, observation,
  dither and detector; see ``encode_image_key`` / ``decode_image_key``.
  It is stable across index rebuilds, so photometry rows can be joined
  to any later index.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

IRSA_TAP_SYNC = "https://irsa.ipac.caltech.edu/TAP/sync"
OBSCORE_COLUMNS = (
    "obs_id,obs_collection,energy_bandpassname,s_ra,s_dec,t_min,t_max,"
    "s_region"
)
DEFAULT_STORE = Path("/ar0/SPHEREX/spherex_l2")
DEFAULT_INDEX_DIR = Path("/ar0/SPHEREX/index")
DEFAULT_STATE_DIR = (
    Path(__file__).resolve().parents[1] / "scripts" / "LOGS" / "state"
)
NAXIS = 2040
# Half diagonal of a 2040 px x 6.15 arcsec detector is 2.46 deg.
COARSE_RADIUS_DEG = 2.6

OBS_ID_RE = re.compile(r"^(\d{4})W(\d{2})_(\d)([A-D])_(\d{4})_(\d)$")
FNAME_RE = re.compile(
    r"^level2_(\d{4}W\d{2}_\d[A-D]_\d{4}_\d)D(\d)_spx_(.+)\.fits$"
)
CAM_CODES = {"A": 0, "B": 1, "C": 2, "D": 3}
CAM_LETTERS = {v: k for k, v in CAM_CODES.items()}

INDEX_COLUMNS = [
    "image_key", "obs_id", "det", "qr", "week_cam", "ver", "path",
    "shard", "s_ra", "s_dec", "t_min", "t_max", "ra1", "dec1", "ra2",
    "dec2", "ra3", "dec3", "ra4", "dec4", "collection", "source",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keys and filenames
# ---------------------------------------------------------------------------
def encode_image_key(obs_id: str, det: int, qr: int) -> int:
    """Encode (obs_id, detector, release) into a decodable int64 key.

    Args:
        obs_id: ObsCore observation id, e.g. ``2025W19_2B_0422_1``.
        det: Detector number 1-6.
        qr: Quick-release number, e.g. 2 for ``qr2``.

    Returns:
        Integer key with digit fields qr|yy|ww|half|cam|obs(4)|dither|det.

    Raises:
        ValueError: If obs_id does not match the expected pattern.
    """
    m = OBS_ID_RE.match(obs_id)
    if m is None:
        raise ValueError(f"Unexpected obs_id format: {obs_id!r}")
    year, week, half, cam, obsnum, dither = m.groups()
    key = int(qr)
    key = key * 100 + (int(year) - 2000)
    key = key * 100 + int(week)
    key = key * 10 + int(half)
    key = key * 10 + CAM_CODES[cam]
    key = key * 10000 + int(obsnum)
    key = key * 10 + int(dither)
    key = key * 10 + int(det)
    return key


def decode_image_key(key: int) -> dict:
    """Decode an image key produced by ``encode_image_key``.

    Args:
        key: Integer image key.

    Returns:
        Dict with obs_id, det, qr, week_cam.
    """
    key = int(key)
    det = key % 10
    key //= 10
    dither = key % 10
    key //= 10
    obsnum = key % 10000
    key //= 10000
    cam = CAM_LETTERS[key % 10]
    key //= 10
    half = key % 10
    key //= 10
    week = key % 100
    key //= 100
    year = 2000 + key % 100
    key //= 100
    qr = key
    week_cam = f"{year:04d}W{week:02d}_{half}{cam}"
    obs_id = f"{week_cam}_{obsnum:04d}_{dither}"
    return {"obs_id": obs_id, "det": det, "qr": qr, "week_cam": week_cam}


def parse_filename(name: str) -> Optional[tuple[str, int, str]]:
    """Parse a Level-2 filename into (obs_id, det, l2b version).

    Args:
        name: Basename such as
            ``level2_2025W19_2B_0422_1D1_spx_l2b-v20-2025-247.fits``.

    Returns:
        Tuple (obs_id, det, ver) or None if the name does not parse.
    """
    m = FNAME_RE.match(name)
    if m is None:
        return None
    return m.group(1), int(m.group(2)), m.group(3)


def shard_name(week_cam: str, ver: str, det: int) -> str:
    """Return the work-shard name for one detector directory."""
    return f"{week_cam}__{ver}__D{int(det)}"


# ---------------------------------------------------------------------------
# ObsCore snapshot
# ---------------------------------------------------------------------------
def fetch_obscore_snapshot(
    out_csv: Path,
    maxrec: int = 3_000_000,
    timeout_sec: float = 3600.0,
) -> Path:
    """Download the full spherex.obscore footprint table to a CSV file.

    Args:
        out_csv: Destination CSV path.
        maxrec: TAP MAXREC row cap.
        timeout_sec: HTTP read timeout in seconds.

    Returns:
        Path to the written CSV.

    Raises:
        RuntimeError: If the response is not a CSV with the expected header.
    """
    import requests

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": "csv",
        "MAXREC": str(int(maxrec)),
        "QUERY": f"SELECT {OBSCORE_COLUMNS} FROM spherex.obscore",
    }
    log.info("Fetching ObsCore snapshot from %s ...", IRSA_TAP_SYNC)
    t0 = datetime.now()
    tmp = out_csv.with_suffix(".csv.part")
    with requests.get(
        IRSA_TAP_SYNC, params=params, stream=True, timeout=timeout_sec
    ) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    with open(tmp, "r", encoding="utf-8") as fh:
        header = fh.readline().strip()
    if not header.startswith("obs_id,"):
        raise RuntimeError(
            f"Unexpected ObsCore response header: {header[:200]!r}"
        )
    os.replace(tmp, out_csv)
    nrows = sum(1 for _ in open(out_csv, "rb")) - 1
    dt = (datetime.now() - t0).total_seconds()
    log.info("ObsCore snapshot: %d rows in %.0f s -> %s", nrows, dt, out_csv)
    return out_csv


def parse_obscore_snapshot(csv_path: Path) -> pd.DataFrame:
    """Parse an ObsCore snapshot CSV into index-style rows.

    Rows without a per-detector bandpass (calibration products) or with
    an unparseable obs_id or polygon are dropped with a log message.

    Args:
        csv_path: CSV written by ``fetch_obscore_snapshot``.

    Returns:
        DataFrame with obs_id, det, qr, collection, s_ra, s_dec, t_min,
        t_max, ra1..ra4, dec1..dec4, image_key, week_cam, source.
    """
    df = pd.read_csv(
        csv_path,
        dtype={"obs_id": str, "obs_collection": str,
               "energy_bandpassname": str, "s_region": str},
    )
    n0 = len(df)
    det = df["energy_bandpassname"].astype(str).str.extract(r"D(\d)$")[0]
    qr = df["obs_collection"].astype(str).str.extract(r"qr(\d+)")[0]
    ok_id = df["obs_id"].astype(str).str.match(OBS_ID_RE)
    keep = det.notna() & qr.notna() & ok_id
    df = df[keep].copy()
    df["det"] = det[keep].astype(int)
    df["qr"] = qr[keep].astype(int)
    log.info(
        "ObsCore: kept %d of %d rows (%d without detector/release/obs_id)",
        len(df), n0, n0 - len(df),
    )

    tokens = df["s_region"].astype(str).str.split()
    ntok = tokens.str.len()
    good = ntok == 12  # POLYGON ICRS + 5 pairs (closed)
    good |= ntok == 10  # POLYGON ICRS + 4 pairs (open)
    if (~good).any():
        log.warning("Dropping %d rows with non-4-corner polygons",
                    int((~good).sum()))
        df = df[good].copy()
        tokens = tokens[good]
    arr = np.array(
        [t[2:10] for t in tokens], dtype=float
    ).reshape(len(df), 8)
    for i in range(4):
        df[f"ra{i + 1}"] = arr[:, 2 * i]
        df[f"dec{i + 1}"] = arr[:, 2 * i + 1]
    df = df.rename(columns={"obs_collection": "collection"})
    df["week_cam"] = df["obs_id"].str.slice(0, 10)
    df["image_key"] = [
        encode_image_key(o, d, q)
        for o, d, q in zip(df["obs_id"], df["det"], df["qr"])
    ]
    df["source"] = "obscore"
    cols = ["image_key", "obs_id", "det", "qr", "week_cam", "s_ra", "s_dec",
            "t_min", "t_max", "ra1", "dec1", "ra2", "dec2", "ra3", "dec3",
            "ra4", "dec4", "collection", "source"]
    return df[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Local store
# ---------------------------------------------------------------------------
def scan_local_store(root: Path = DEFAULT_STORE) -> pd.DataFrame:
    """Scan the local store for Level-2 files.

    Expects ``<root>/<week_cam>/<l2b version>/<det>/level2_*.fits``.

    Args:
        root: Store root directory.

    Returns:
        DataFrame with obs_id, det, ver, week_cam, path.
    """
    root = Path(root)
    rows = []
    n_unparsed = 0
    for week in sorted(os.scandir(root), key=lambda e: e.name):
        if not week.is_dir():
            continue
        for ver in os.scandir(week.path):
            if not ver.is_dir():
                continue
            for det in os.scandir(ver.path):
                if not det.is_dir():
                    continue
                for f in os.scandir(det.path):
                    if not f.name.endswith(".fits"):
                        continue
                    parsed = parse_filename(f.name)
                    if parsed is None:
                        n_unparsed += 1
                        continue
                    obs_id, detnum, fver = parsed
                    rows.append((obs_id, detnum, ver.name, week.name, f.path))
    if n_unparsed:
        log.warning("scan_local_store: %d unparsed .fits names", n_unparsed)
    df = pd.DataFrame(
        rows, columns=["obs_id", "det", "ver", "week_cam", "path"]
    )
    log.info("Local store %s: %d Level-2 files in %d week dirs",
             root, len(df), df["week_cam"].nunique() if len(df) else 0)
    return df


def load_week_tags(state_dir: Path = DEFAULT_STATE_DIR) -> dict[str, int]:
    """Map week_cam -> lowest release number from get_spherex.sh state.

    Args:
        state_dir: Directory containing ``weeks_qr<N>.txt`` files.

    Returns:
        Dict week_cam -> qr number (empty if no state files exist).
    """
    out: dict[str, int] = {}
    state_dir = Path(state_dir)
    if not state_dir.exists():
        return out
    for p in sorted(state_dir.glob("weeks_qr*.txt")):
        m = re.search(r"weeks_qr(\d+)\.txt$", p.name)
        if m is None:
            continue
        qr = int(m.group(1))
        for line in p.read_text().split():
            out.setdefault(line.strip(), qr)
    return out


def _header_row(path: str) -> dict:
    """Read index fields from one Level-2 IMAGE header.

    Args:
        path: FITS file path.

    Returns:
        Dict with obs_id, det, s_ra, s_dec, corners, t_min, t_max, path,
        or a dict with an ``err`` entry on failure.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(path, memmap=True) as hdul:
                hdr = hdul["IMAGE"].header
                wcs = WCS(hdr)
                xs = np.array([-0.5, NAXIS - 0.5, NAXIS - 0.5, -0.5])
                ys = np.array([-0.5, -0.5, NAXIS - 0.5, NAXIS - 0.5])
                ra, dec = wcs.all_pix2world(xs, ys, 0)
                row = {
                    "obs_id": str(hdr["OBSID"]),
                    "det": int(hdr["DETECTOR"]),
                    "s_ra": float(hdr["CRVAL1"]),
                    "s_dec": float(hdr["CRVAL2"]),
                    "t_min": float(hdr.get("MJD-BEG", np.nan)),
                    "t_max": float(hdr.get("MJD-END", np.nan)),
                    "path": path,
                }
                for i in range(4):
                    row[f"ra{i + 1}"] = float(ra[i])
                    row[f"dec{i + 1}"] = float(dec[i])
                return row
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        return {"path": path, "err": repr(exc)}


def index_from_headers(
    paths: Sequence[str],
    nproc: int = 32,
) -> pd.DataFrame:
    """Build index rows from FITS headers for the given local files.

    Args:
        paths: Local Level-2 file paths.
        nproc: Worker processes.

    Returns:
        DataFrame with the same footprint columns as the ObsCore parser
        (no qr/collection), plus ``source='header'``.
    """
    if len(paths) == 0:
        return pd.DataFrame()
    ctx = get_context("fork")
    with ctx.Pool(int(max(1, nproc))) as pool:
        rows = pool.map(_header_row, list(paths), chunksize=50)
    df = pd.DataFrame(rows)
    if "err" in df.columns:
        bad = df["err"].notna()
        if bad.any():
            log.warning("index_from_headers: %d files failed, e.g. %s",
                        int(bad.sum()), df.loc[bad, "err"].iloc[0])
        df = df[~bad].drop(columns=["err"])
    df["source"] = "header"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_image_index(
    store_root: Path = DEFAULT_STORE,
    obscore_csv: Optional[Path] = None,
    out_parquet: Optional[Path] = None,
    state_dir: Path = DEFAULT_STATE_DIR,
    nproc: int = 32,
    fetch: bool = True,
) -> pd.DataFrame:
    """Build the image index from ObsCore plus the local store.

    Args:
        store_root: Local Level-2 store root.
        obscore_csv: Existing snapshot CSV to reuse; if None and ``fetch``
            is True a new snapshot is downloaded into the index directory.
        out_parquet: Output parquet path (default
            ``/ar0/SPHEREX/index/image_index.parquet``).
        state_dir: get_spherex.sh state directory (release tags per week).
        nproc: Processes for header reads of files missing from ObsCore.
        fetch: Whether to download a snapshot when ``obscore_csv`` is None.

    Returns:
        The index DataFrame (also written to parquet).
    """
    out_parquet = Path(
        out_parquet or DEFAULT_INDEX_DIR / "image_index.parquet"
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    if obscore_csv is None and fetch:
        stamp = datetime.now().strftime("%Y%m%d")
        obscore_csv = fetch_obscore_snapshot(
            out_parquet.parent / f"obscore_{stamp}.csv"
        )
    obs = (
        parse_obscore_snapshot(obscore_csv)
        if obscore_csv is not None
        else pd.DataFrame()
    )

    local = scan_local_store(store_root)
    week_tags = load_week_tags(state_dir)

    if len(obs):
        # Attach local paths.  A local (obs_id, det) joins to the lowest
        # release row when several releases list the same image.
        obs = obs.sort_values(["obs_id", "det", "qr"]).reset_index(drop=True)
        first = ~obs.duplicated(["obs_id", "det"], keep="first")
        n_multi = int((~first).sum())
        if n_multi:
            log.warning(
                "%d (obs_id, det) pairs appear under several releases; "
                "local files are attached to the lowest release", n_multi,
            )
        joined = obs[first].merge(
            local[["obs_id", "det", "ver", "path"]],
            on=["obs_id", "det"], how="left",
        )
        rest = obs[~first].copy()
        rest["ver"] = None
        rest["path"] = None
        obs = pd.concat([joined, rest], ignore_index=True)
        have = set(zip(obs["obs_id"], obs["det"]))
    else:
        have = set()

    missing = local[
        [(o, d) not in have for o, d in zip(local["obs_id"], local["det"])]
    ]
    if len(missing):
        log.info("%d local files are not in ObsCore; indexing from headers",
                 len(missing))
        hdr = index_from_headers(list(missing["path"]), nproc=nproc)
        if len(hdr):
            hdr = hdr.merge(
                local[["path", "ver", "week_cam"]], on="path", how="left"
            )
            hdr["qr"] = [
                week_tags.get(w, max(week_tags.values()) if week_tags else 0)
                for w in hdr["week_cam"]
            ]
            hdr["collection"] = [f"local_qr{q}" for q in hdr["qr"]]
            hdr["image_key"] = [
                encode_image_key(o, d, q)
                for o, d, q in zip(hdr["obs_id"], hdr["det"], hdr["qr"])
            ]
            obs = pd.concat([obs, hdr], ignore_index=True)

    if obs.empty:
        raise RuntimeError("Image index is empty: no ObsCore rows or files.")

    obs["shard"] = [
        shard_name(w, v, d) if isinstance(v, str) else None
        for w, v, d in zip(obs["week_cam"], obs["ver"], obs["det"])
    ]
    obs = obs.drop_duplicates("image_key").sort_values(
        ["week_cam", "ver", "det", "obs_id"], na_position="last"
    ).reset_index(drop=True)
    obs = obs[INDEX_COLUMNS]
    obs["det"] = obs["det"].astype("int8")
    obs["qr"] = obs["qr"].astype("int8")
    obs["image_key"] = obs["image_key"].astype("int64")

    tmp = out_parquet.with_suffix(".parquet.part")
    obs.to_parquet(tmp, index=False)
    os.replace(tmp, out_parquet)
    n_local = int(obs["path"].notna().sum())
    log.info(
        "Image index: %d images (%d local, %d not yet local, %d from "
        "headers) -> %s",
        len(obs), n_local, len(obs) - n_local,
        int((obs["source"] == "header").sum()), out_parquet,
    )
    return obs


def load_image_index(
    path: Optional[Path] = None,
    columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load an image index parquet file.

    Args:
        path: Parquet path (default
            ``/ar0/SPHEREX/index/image_index.parquet``).
        columns: Optional subset of columns.

    Returns:
        Index DataFrame.
    """
    path = Path(path or DEFAULT_INDEX_DIR / "image_index.parquet")
    return pd.read_parquet(path, columns=list(columns) if columns else None)


# ---------------------------------------------------------------------------
# Footprint queries
# ---------------------------------------------------------------------------
def unit_vectors(ra_deg, dec_deg) -> np.ndarray:
    """Return (N, 3) unit vectors for RA/Dec in degrees."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    return np.stack(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)],
        axis=-1,
    )


def _gnomonic(ra, dec, ra0, dec0):
    """Project (ra, dec) onto the tangent plane at (ra0, dec0); radians."""
    dra = ra - ra0
    cosc = (np.sin(dec0) * np.sin(dec)
            + np.cos(dec0) * np.cos(dec) * np.cos(dra))
    xi = np.cos(dec) * np.sin(dra) / cosc
    eta = (np.cos(dec0) * np.sin(dec)
           - np.sin(dec0) * np.cos(dec) * np.cos(dra)) / cosc
    return xi, eta


def _points_in_quads(px, py, qx, qy) -> np.ndarray:
    """Ray-casting test of points (N,) against quads (N, 4)."""
    inside = np.zeros(px.shape, dtype=bool)
    j = 3
    for i in range(4):
        xi, yi, xj, yj = qx[:, i], qy[:, i], qx[:, j], qy[:, j]
        cross = (yi > py) != (yj > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (py - yi) / (yj - yi) + xi
        inside ^= cross & (px < xint)
        j = i
    return inside


def images_containing(
    index: pd.DataFrame,
    ra_deg: float,
    dec_deg: float,
    coarse_radius_deg: float = COARSE_RADIUS_DEG,
) -> np.ndarray:
    """Return index row positions whose ObsCore footprint contains a point.

    The polygon test agrees with the exact TAN-SIP header WCS for 99.8%
    of star-image pairs; disagreements lie within about 6 pixels of an
    image edge.  Use the photometry pass for the exact per-pixel test.

    Args:
        index: Image index DataFrame.
        ra_deg: Point RA in degrees.
        dec_deg: Point Dec in degrees.
        coarse_radius_deg: Center-separation prefilter radius.

    Returns:
        Integer positions (``iloc``) of matching rows.
    """
    p = unit_vectors(ra_deg, dec_deg)
    centers = unit_vectors(index["s_ra"].to_numpy(), index["s_dec"].to_numpy())
    cosr = np.cos(np.deg2rad(coarse_radius_deg))
    cand = np.where(centers @ p >= cosr)[0]
    if cand.size == 0:
        return cand
    sub = index.iloc[cand]
    ra0 = np.deg2rad(sub["s_ra"].to_numpy())
    dec0 = np.deg2rad(sub["s_dec"].to_numpy())
    qx = np.empty((cand.size, 4))
    qy = np.empty((cand.size, 4))
    for i in range(4):
        qx[:, i], qy[:, i] = _gnomonic(
            np.deg2rad(sub[f"ra{i + 1}"].to_numpy()),
            np.deg2rad(sub[f"dec{i + 1}"].to_numpy()), ra0, dec0,
        )
    px, py = _gnomonic(np.deg2rad(ra_deg), np.deg2rad(dec_deg), ra0, dec0)
    return cand[_points_in_quads(px, py, qx, qy)]


__all__ = [
    "COARSE_RADIUS_DEG",
    "DEFAULT_INDEX_DIR",
    "DEFAULT_STORE",
    "INDEX_COLUMNS",
    "build_image_index",
    "decode_image_key",
    "encode_image_key",
    "fetch_obscore_snapshot",
    "images_containing",
    "index_from_headers",
    "load_image_index",
    "parse_filename",
    "parse_obscore_snapshot",
    "scan_local_store",
    "shard_name",
    "unit_vectors",
]
