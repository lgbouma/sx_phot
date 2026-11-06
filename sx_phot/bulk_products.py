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
import shutil
from pathlib import Path
from typing import Optional, Sequence

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


def _split_pass(args: tuple) -> int:
    """Split one pass file into per-group parquet files (stage 1).

    Args:
        args: (pass_path, group_dir, gshift, columns).

    Returns:
        Number of rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pass_path, group_dir, gshift, columns = args
    pass_path = Path(pass_path)
    table = pq.read_table(pass_path, columns=columns)
    group = np.right_shift(table.column("star_idx").to_numpy(), gshift)
    order = np.argsort(group, kind="stable")
    table = table.take(pa.array(order))
    group = group[order]
    stem = f"{pass_path.parent.name}__{pass_path.stem}"
    for start, stop in _runs(group):
        gdir = Path(group_dir) / f"group={int(group[start])}"
        gdir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            table.slice(start, stop - start), gdir / f"{stem}.parquet",
            compression="zstd", row_group_size=1 << 20,
        )
    return int(len(group))


def _runs(keys: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, stop) slices of equal consecutive values."""
    if len(keys) == 0:
        return []
    edges = np.flatnonzero(np.diff(keys)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [len(keys)]])
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def _write_group(args: tuple) -> int:
    """Sort one group of rows by star and write its bucket files.

    Args:
        args: (group_dir, out_dir, shift, columns).

    Returns:
        Number of rows written.
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    group_dir, out_dir, shift, columns = args
    files = sorted(Path(group_dir).glob("*.parquet"))
    if not files:
        return 0
    table = ds.dataset([str(f) for f in files],
                       format="parquet").to_table(columns=columns)
    order = pc.sort_indices(table, sort_keys=[("star_idx", "ascending")])
    table = table.take(order)
    del order
    bucket = np.right_shift(table.column("star_idx").to_numpy(), shift)
    for start, stop in _runs(bucket):
        bdir = Path(out_dir) / f"bucket={int(bucket[start])}"
        bdir.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            table.slice(start, stop - start), bdir / "part-0.parquet",
            compression="zstd", row_group_size=1 << 20,
        )
    return int(len(bucket))


def gather_by_star(
    run_dir: Path,
    stars_per_bucket: int = DEFAULT_STARS_PER_BUCKET,
    buckets_per_group: int = 64,
    nproc: int = 4,
) -> dict:
    """Regroup all per-image rows into per-star buckets.

    Two stages keep memory bounded regardless of the number of buckets.
    Stage 1 reads one ``pass_*.parquet`` at a time and writes its rows
    into ``bystar.groups/group=G/`` (one group per ``buckets_per_group``
    buckets).  Stage 2 reads one group at a time, sorts it by star_idx
    and splits it into bucket files.  The previous ``bystar`` tree is
    replaced.  Peak memory is about ``nproc`` times one group of rows
    (about 10 GB per worker for a 7.5M-star, 3.6G-row run with the
    defaults).

    Args:
        run_dir: Run directory.
        stars_per_bucket: Consecutive star_idx values per bucket (power
            of two).
        buckets_per_group: Buckets per stage-1 group (power of two).
        nproc: Worker processes.

    Returns:
        Dict with n_rows, n_buckets, stars_per_bucket.

    Raises:
        ValueError: If stars_per_bucket or buckets_per_group is not a
            power of two.
    """
    from multiprocessing import Pool

    spb = int(stars_per_bucket)
    bpg = int(buckets_per_group)
    for name, val in (("stars_per_bucket", spb),
                      ("buckets_per_group", bpg)):
        if val <= 0 or (val & (val - 1)):
            raise ValueError(f"{name} must be a power of two")
    shift = int(np.log2(spb))
    gshift = shift + int(np.log2(bpg))
    run_dir = Path(run_dir)
    files = sorted((run_dir / "phot").glob("*/pass_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No pass parquet files under {run_dir}/phot")
    n_stars = len(pd.read_parquet(run_dir / "catalog.parquet",
                                  columns=["star_idx"]))
    n_buckets = (n_stars + spb - 1) // spb
    n_groups = (n_buckets + bpg - 1) // bpg

    out_dir = run_dir / "bystar"
    tmp_dir = run_dir / "bystar.part"
    group_dir = run_dir / "bystar.groups"
    for d in (tmp_dir, group_dir):
        if d.exists():
            shutil.rmtree(d)
    group_dir.mkdir()
    columns = list(ROW_DTYPES)
    nproc = max(1, int(nproc))

    log.info("Gathering %d pass files into %d buckets (%d stars each) "
             "via %d groups, %d workers ...", len(files), n_buckets, spb,
             n_groups, nproc)
    jobs = [(f, group_dir, gshift, columns) for f in files]
    n_in = 0
    with Pool(processes=nproc) as pool:
        for i, n in enumerate(pool.imap_unordered(_split_pass, jobs), 1):
            n_in += n
            if i % 100 == 0 or i == len(jobs):
                log.info("  stage 1: %d/%d pass files, %d rows",
                         i, len(jobs), n_in)
    jobs = [(group_dir / f"group={g}", tmp_dir, shift, columns)
            for g in range(n_groups) if (group_dir / f"group={g}").exists()]
    n_rows = 0
    with Pool(processes=nproc) as pool:
        for i, n in enumerate(pool.imap_unordered(_write_group, jobs), 1):
            n_rows += n
            if i % 10 == 0 or i == len(jobs):
                log.info("  stage 2: %d/%d groups, %d rows",
                         i, len(jobs), n_rows)
    if n_rows != n_in:
        raise RuntimeError(
            f"gather row mismatch: {n_in} read, {n_rows} written")
    shutil.rmtree(group_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp_dir, out_dir)
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
    skip_failed_bands: bool = False,
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
        skip_failed_bands: Leave out bands with too few points instead
            of raising (their model_flux is NaN, fit_mask False).

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
        skip_failed_bands=skip_failed_bands,
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


def _materialize_group(args: tuple) -> dict:
    """Worker: write the CSV products for one star's rows."""
    ticid, star_id, radec, rows, cfg, out_dir, supplement = args
    out_dir = Path(out_dir)
    base = to_legacy_frame(rows.sort_values("wavelength_um")
                           .reset_index(drop=True), cfg)
    stem = f"sxphot_cache_{star_id}_{radec}_annulus"
    base.to_csv(out_dir / f"{stem}.csv", index=False)
    info = {"ticid": int(ticid), "star_id": star_id,
            "cache": str(out_dir / f"{stem}.csv"), "supp": "",
            "n": int(len(base)),
            "n_masked": int(base["masked"].sum()),
            "med_snr": float("nan"), "spline_ok": False}
    fl = base["flux_jy"].to_numpy()
    er = base["flux_err_jy"].to_numpy()
    good = ~base["masked"].to_numpy() & np.isfinite(fl) & (fl > 0)
    if good.any():
        info["med_snr"] = float(np.nanmedian(fl[good] / er[good]))
    if supplement and len(base) >= 4:
        try:
            supp = supplement_with_spline(base, skip_failed_bands=True)
        except ValueError as exc:
            log.warning("%s: spline supplement skipped (%s)", star_id, exc)
        else:
            supp.to_csv(out_dir / f"{stem}_splsupp.csv", index=False)
            info["supp"] = str(out_dir / f"{stem}_splsupp.csv")
            info["spline_ok"] = True
            info["n_fit"] = int(supp["fit_mask"].sum())
    return info


def materialize_run(
    run_dir: Path,
    out_dir: Path,
    id_prefix: str = "TIC",
    supplement: bool = True,
    nproc: int = 8,
    ticids: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Write the legacy per-star CSV products for every star in a run.

    Buckets are read once each and split by star, so this is much faster
    than calling ``materialize_star`` per star.  Stars with no rows get
    no files but are listed in the summary with ``n == 0``.  The spline
    supplement skips bands with too few unmasked points (saturated bright
    stars) instead of failing, as in ``viz_products``; ``n_fit`` in the
    summary counts the points used in the fit.

    Args:
        run_dir: Gathered run directory.
        out_dir: Output directory for the CSV files.
        id_prefix: Filename label prefix (``TIC`` or e.g. ``GaiaDR3``).
        supplement: Also write the ``_splsupp.csv`` spline products.
        nproc: Worker processes for the spline fits.
        ticids: Restrict to these catalog ids (e.g. a QA sample of a
            large run); only their buckets are read.  Ids missing from
            the catalog are logged and skipped.

    Returns:
        Summary DataFrame, one row per catalog star (or per requested
        star), also written to ``out_dir/summary.csv``.
    """
    import pyarrow.dataset as ds
    from multiprocessing import Pool

    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run_dir / "config.json").read_text())["phot"]
    cat = pd.read_parquet(run_dir / "catalog.parquet")
    if ticids is not None:
        want = np.unique(np.asarray(list(ticids), dtype=np.int64))
        cat = cat[cat["ticid"].isin(want)]
        if len(cat) < len(want):
            log.warning("%d of %d requested ids are not in the catalog",
                        len(want) - len(cat), len(want))
    spb = _bucket_params(run_dir)
    index = _run_index(run_dir)
    cat["bucket"] = cat["star_idx"].to_numpy() // spb
    infos: dict[int, dict] = {}
    with Pool(int(nproc)) as pool:
        for bucket, grp in cat.groupby("bucket", sort=True):
            bucket_dir = run_dir / "bystar" / f"bucket={int(bucket)}"
            if not bucket_dir.exists():
                continue
            dataset = ds.dataset(str(bucket_dir), format="parquet")
            if ticids is None:
                rows = dataset.to_table()
            else:
                rows = dataset.to_table(filter=ds.field("ticid").isin(
                    [int(t) for t in grp["ticid"]]))
            rows = rows.to_pandas().merge(index, on="image_key", how="left")
            by_tic = {int(t): r for t, r in rows.groupby("ticid")}
            tasks = []
            for crow in grp.itertuples(index=False):
                tic = int(crow.ticid)
                if tic not in by_tic:
                    continue
                radec = _format_radec_stub(float(crow.ra), float(crow.dec))
                tasks.append((tic, f"{id_prefix}_{tic}", radec, by_tic[tic],
                              cfg, str(out_dir), supplement))
            for info in pool.imap_unordered(_materialize_group, tasks,
                                            chunksize=4):
                infos[info["ticid"]] = info
            log.info("bucket %d: %d stars written", int(bucket), len(tasks))
    summary = cat[["star_idx", "ticid", "ra", "dec"]].copy()
    for col, default in (("star_id", ""), ("n", 0), ("n_masked", 0),
                         ("med_snr", np.nan), ("spline_ok", False),
                         ("n_fit", 0), ("cache", ""), ("supp", "")):
        summary[col] = [infos.get(int(t), {}).get(col, default)
                        for t in summary["ticid"]]
    summary.loc[summary["star_id"] == "", "star_id"] = [
        f"{id_prefix}_{int(t)}" for t in
        summary.loc[summary["star_id"] == "", "ticid"]]
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


__all__ = [
    "DEFAULT_STARS_PER_BUCKET",
    "LEGACY_COLUMNS",
    "gather_by_star",
    "load_star_rows",
    "materialize_run",
    "materialize_star",
    "star_index_for_ticid",
    "supplement_with_spline",
    "to_legacy_frame",
]
