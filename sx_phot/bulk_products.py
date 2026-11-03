"""Products from a bulk photometry run: per-star gather and legacy CSVs.

Layout of a run directory (see ``sx_phot.bulk_phot``)::

    <run>/config.json, catalog.parquet, index.parquet
    <run>/phot/<shard>/pass_<stamp>.parquet      per-image rows
    <run>/bystar/bucket=<N>/part-*.parquet       rows grouped by star

Buckets hold ``stars_per_bucket`` consecutive ``star_idx`` values, so a
star's bucket is ``star_idx // stars_per_bucket``.
"""

from __future__ import annotations

import json
import logging
import os
import resource
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sx_phot.bulk_phot import ROW_DTYPES, TJD_OFFSET
from sx_phot.circphot import BITMASK, _format_radec_stub
from sx_phot.splinefit import fit_spherex_spectrum_bspline

log = logging.getLogger(__name__)

DEFAULT_STARS_PER_BUCKET = 1024
LEGACY_COLUMNS = [
    "wavelength_um", "bandwidth_um", "flux_jy", "flux_err_jy", "file",
    "masked", "quality", "aper_count", "mjd_avg", "tjd_avg", "bkgd_method",
    "aperture_radius_pix", "annulus_r_in_pix", "annulus_r_out_pix",
]


def _raise_open_files_limit(n: int) -> None:
    """Raise the soft open-files limit to at least ``n`` if allowed."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = min(max(soft, n), hard)
    if want > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))


def gather_by_star(
    run_dir: Path,
    stars_per_bucket: int = DEFAULT_STARS_PER_BUCKET,
) -> dict:
    """Regroup all per-image rows into per-star buckets.

    Streams every ``pass_*.parquet`` through a partitioned dataset write;
    the previous ``bystar`` tree is replaced.

    Args:
        run_dir: Run directory.
        stars_per_bucket: Consecutive star_idx values per bucket (power
            of two).

    Returns:
        Dict with n_rows, n_buckets, stars_per_bucket.

    Raises:
        ValueError: If stars_per_bucket is not a power of two.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    spb = int(stars_per_bucket)
    if spb <= 0 or (spb & (spb - 1)):
        raise ValueError("stars_per_bucket must be a power of two")
    shift = int(np.log2(spb))
    run_dir = Path(run_dir)
    files = sorted((run_dir / "phot").glob("*/pass_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No pass parquet files under {run_dir}/phot")
    n_stars = len(pd.read_parquet(run_dir / "catalog.parquet",
                                  columns=["star_idx"]))
    n_buckets = (n_stars + spb - 1) // spb
    _raise_open_files_limit(n_buckets + 256)

    out_dir = run_dir / "bystar"
    tmp_dir = run_dir / "bystar.part"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    schema = pa.schema(
        [(k, pa.from_numpy_dtype(v)) for k, v in ROW_DTYPES.items()]
    )
    out_schema = schema.append(pa.field("bucket", pa.int32()))
    dataset = ds.dataset([str(f) for f in files], format="parquet",
                         schema=schema)

    def _batches():
        for batch in dataset.to_batches(batch_size=1 << 18):
            bucket = pc.cast(
                pc.shift_right(batch.column("star_idx"), shift), pa.int32()
            )
            yield pa.RecordBatch.from_arrays(
                list(batch.columns) + [bucket], schema=out_schema
            )

    log.info("Gathering %d pass files into %d buckets (%d stars each) ...",
             len(files), n_buckets, spb)
    ds.write_dataset(
        _batches(), schema=out_schema, base_dir=str(tmp_dir),
        format="parquet", partitioning=["bucket"],
        partitioning_flavor="hive",
        existing_data_behavior="overwrite_or_ignore",
        max_open_files=n_buckets + 64, max_partitions=n_buckets + 64,
        min_rows_per_group=1 << 16, max_rows_per_group=1 << 20,
        file_options=ds.ParquetFileFormat().make_write_options(
            compression="zstd"
        ),
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp_dir, out_dir)
    n_rows = sum(
        pq_rows(p) for p in out_dir.glob("bucket=*/*.parquet")
    )
    meta = {"n_rows": int(n_rows), "n_buckets": int(n_buckets),
            "stars_per_bucket": spb, "n_pass_files": len(files)}
    (run_dir / "bystar.json").write_text(json.dumps(meta, indent=2))
    log.info("Gathered %d rows -> %s", n_rows, out_dir)
    return meta


def pq_rows(path: Path) -> int:
    """Return the row count of a parquet file from its metadata."""
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _bucket_params(run_dir: Path) -> int:
    meta = json.loads((Path(run_dir) / "bystar.json").read_text())
    return int(meta["stars_per_bucket"])


def star_index_for_ticid(run_dir: Path, ticid: int) -> int:
    """Return the catalog star_idx for a TIC id in a run.

    Raises:
        KeyError: If the id is not in the run catalog.
    """
    cat = pd.read_parquet(Path(run_dir) / "catalog.parquet",
                          columns=["star_idx", "ticid"])
    hit = cat.index[cat["ticid"].to_numpy() == int(ticid)]
    if len(hit) == 0:
        raise KeyError(f"TIC {ticid} not in {run_dir}/catalog.parquet")
    return int(cat.loc[hit[0], "star_idx"])


_INDEX_CACHE: dict[tuple, pd.DataFrame] = {}


def _run_index(run_dir: Path) -> pd.DataFrame:
    """Return (cached) join columns of a run's index.parquet."""
    path = Path(run_dir) / "index.parquet"
    key = (str(path), os.path.getmtime(path))
    if key not in _INDEX_CACHE:
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = pd.read_parquet(
            path, columns=["image_key", "path", "qr", "ver", "week_cam",
                           "obs_id", "det"],
        )
    return _INDEX_CACHE[key]


def load_star_rows(run_dir: Path, ticid: int) -> pd.DataFrame:
    """Load all photometry rows for one star from the gathered buckets.

    Args:
        run_dir: Run directory (must have been gathered).
        ticid: Catalog identifier.

    Returns:
        DataFrame of per-image rows sorted by wavelength, joined with the
        image index columns path, qr, ver, week_cam, obs_id, det.
    """
    import pyarrow.dataset as ds

    run_dir = Path(run_dir)
    spb = _bucket_params(run_dir)
    sidx = star_index_for_ticid(run_dir, ticid)
    bucket_dir = run_dir / "bystar" / f"bucket={sidx // spb}"
    if not bucket_dir.exists():
        return pd.DataFrame({k: np.zeros(0, dtype=v)
                             for k, v in ROW_DTYPES.items()})
    dataset = ds.dataset(str(bucket_dir), format="parquet")
    table = dataset.to_table(filter=ds.field("ticid") == int(ticid))
    rows = table.to_pandas()
    rows = rows.merge(_run_index(run_dir), on="image_key", how="left")
    return rows.sort_values("wavelength_um").reset_index(drop=True)


def to_legacy_frame(
    rows: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Convert bulk rows into the ``sxphot_cache_*`` column layout.

    Args:
        rows: Output of ``load_star_rows``.
        config: The ``phot`` section of the run's config.json.

    Returns:
        DataFrame with the legacy columns (``quality`` holds the bitwise
        OR of aperture flags; ``aper_count`` holds the flagged pixel count).
    """
    masked = (rows["flag_or"].to_numpy().astype(np.int64) & BITMASK) != 0
    mjd = rows["mjd_avg"].to_numpy(dtype=float)
    out = pd.DataFrame({
        "wavelength_um": rows["wavelength_um"].astype(float),
        "bandwidth_um": rows["bandwidth_um"].astype(float),
        "flux_jy": rows["flux_jy"].astype(float),
        "flux_err_jy": rows["flux_err_jy"].astype(float),
        "file": rows["path"].astype(object),
        "masked": masked,
        "quality": rows["flag_or"].astype(str),
        "aper_count": rows["n_bad_pix"].astype(int),
        "mjd_avg": mjd,
        "tjd_avg": mjd - TJD_OFFSET,
        "bkgd_method": "annulus",
        "aperture_radius_pix": float(config["aperture_radius"]),
        "annulus_r_in_pix": float(config["annulus_r_in"]),
        "annulus_r_out_pix": float(config["annulus_r_out"]),
    })
    for extra in ("flux_raw_jy", "bkg_jy", "zodi_jy", "n_ann_pix",
                  "bkg_mode", "x", "y", "qr", "image_key"):
        if extra in rows.columns:
            out[extra] = rows[extra].to_numpy()
    return out


def supplement_with_spline(
    base: pd.DataFrame,
    n_res_el: float = 4.0,
    degree: int = 3,
    spacing_mode: str = "log",
    max_iter: int = 8,
    outlier_cut: float = 3.0,
    clip_on_normalized_residuals: bool = True,
    dense_n_per_band: int = 800,
) -> pd.DataFrame:
    """Add the spline-model columns of ``get_supplemented_sx_spectrum``.

    Args:
        base: Legacy-layout spectrum (wavelength_um, bandwidth_um,
            flux_jy, flux_err_jy, masked).
        n_res_el: Resolution elements per knot interval.
        degree: Spline degree.
        spacing_mode: 'log' or 'linear' knot spacing.
        max_iter: Clip-refit iterations per band.
        outlier_cut: Robust sigma cut.
        clip_on_normalized_residuals: Clip on residual / error.
        dense_n_per_band: Dense grid samples per band.

    Returns:
        Copy of ``base`` with model_flux, fit_mask, residuals, res_valid,
        res_masked, res_clipped.
    """
    wavelength = np.asarray(base["wavelength_um"], dtype=float)
    bandwidth = np.asarray(base["bandwidth_um"], dtype=float)
    flux = np.asarray(base["flux_jy"], dtype=float)
    err = np.asarray(base["flux_err_jy"], dtype=float)
    mask = np.asarray(base["masked"], dtype=bool)
    model_flux, fit_mask, _, _, _ = fit_spherex_spectrum_bspline(
        wavelength_um=wavelength, bandwidth_um=bandwidth, flux_jy=flux,
        flux_err_jy=err, masked=mask, n_res_el=n_res_el, degree=degree,
        spacing_mode=spacing_mode, max_iter=max_iter,
        outlier_cut=outlier_cut,
        clip_on_normalized_residuals=clip_on_normalized_residuals,
        dense_n_per_band=dense_n_per_band,
    )
    finite = np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(err)
    mask = mask | ~finite
    use = ~mask
    fit_used = use if fit_mask is None else np.asarray(fit_mask, bool) & use
    clipped = use & ~fit_used & np.isfinite(model_flux)
    residuals = flux - model_flux
    out = base.copy()
    out["model_flux"] = model_flux
    out["fit_mask"] = fit_used
    out["residuals"] = residuals
    out["res_valid"] = np.isfinite(residuals) & fit_used & np.isfinite(err)
    out["res_masked"] = np.isfinite(residuals) & mask
    out["res_clipped"] = np.isfinite(residuals) & clipped
    return out


def materialize_star(
    run_dir: Path,
    ticid: int,
    out_dir: Path,
    supplement: bool = True,
    star_id: Optional[str] = None,
) -> dict[str, Path]:
    """Write the legacy per-star CSV products for one star.

    Args:
        run_dir: Gathered run directory.
        ticid: Catalog identifier.
        out_dir: Output directory.
        supplement: Also write the ``_splsupp.csv`` spline product.
        star_id: Label used in filenames (default ``TIC_<ticid>``).

    Returns:
        Dict with the written paths (``cache`` and optionally ``supp``).
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run_dir / "config.json").read_text())["phot"]
    cat = pd.read_parquet(run_dir / "catalog.parquet")
    row = cat[cat["ticid"] == int(ticid)]
    if row.empty:
        raise KeyError(f"TIC {ticid} not in run catalog")
    radec = _format_radec_stub(float(row["ra"].iloc[0]),
                               float(row["dec"].iloc[0]))
    star_id = star_id or f"TIC_{int(ticid)}"
    rows = load_star_rows(run_dir, ticid)
    base = to_legacy_frame(rows, cfg)
    stem = f"sxphot_cache_{star_id}_{radec}_annulus"
    paths = {"cache": out_dir / f"{stem}.csv"}
    base.to_csv(paths["cache"], index=False)
    if supplement and len(base) >= 4:
        try:
            supp = supplement_with_spline(base)
        except ValueError as exc:
            log.warning("TIC %d: spline supplement skipped (%s)", ticid, exc)
        else:
            paths["supp"] = out_dir / f"{stem}_splsupp.csv"
            supp.to_csv(paths["supp"], index=False)
    return paths


__all__ = [
    "DEFAULT_STARS_PER_BUCKET",
    "LEGACY_COLUMNS",
    "gather_by_star",
    "load_star_rows",
    "materialize_star",
    "star_index_for_ticid",
    "supplement_with_spline",
    "to_legacy_frame",
]
