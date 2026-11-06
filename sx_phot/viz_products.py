"""Per-star visualization products from a gathered bulk run.

Layout written under ``<run>/viz``::

    spectra/bucket=<N>.parquet   one row group per star, sorted by ticid
    summary/bucket=<N>.parquet   one row per star
    binned/bucket=<N>.npz        binned spectra (ticid, flux, err, std, n,
                                 model), rows aligned with the summary
    summary.parquet, binned_*.npy, bin_edges.npy
                                 concatenations written by ``finalize``

The spectra tables carry the legacy columns plus the spline model, so a
star page needs one row-group read (``read_star_spectrum``) and no
spline fit.  ``masked`` here is the viewer's quality mask (hard flag
bits, detector edge, gross outliers; see ``quality_masks``), which is
wider than the legacy ``BITMASK`` on edges and narrower on the soft
TRANSIENT/OVERFLOW bits; ``mask_bits`` says why.  The binned spectra
live on ``sedmodels.spherex_bin_edges`` and feed the SED fits.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sx_phot.bulk_phot import TJD_OFFSET
from sx_phot.bulk_products import (_run_index, supplement_with_spline,
                                   to_legacy_frame)
from sx_phot.sedmodels import spherex_bin_edges

log = logging.getLogger(__name__)

SPECTRUM_COLUMNS = {
    "wavelength_um": np.float32,
    "bandwidth_um": np.float32,
    "flux_jy": np.float32,
    "flux_err_jy": np.float32,
    "masked": bool,
    "mask_bits": np.int8,
    "fit_mask": bool,
    "model_flux": np.float32,
    "mjd_avg": np.float64,
    "image_key": np.int64,
    "x": np.float32,
    "y": np.float32,
    "flag_or": np.int32,
    "n_bad_pix": np.int16,
    "bkg_jy": np.float32,
    "bkg_mode": np.int8,
    "flux_raw_jy": np.float32,
    "zodi_jy": np.float32,
    "n_ann_pix": np.int16,
}

BAND_WINDOWS_UM = [(0.75, 1.1075), (1.1075, 1.6435), (1.6435, 2.421),
                   (2.423, 3.8155), (3.8155, 4.416), (4.416, 5.002)]

# Quality masking.  SPHEREx flag bits (see sx_phot.circphot.BITMASK):
# TRANSIENT (0) and OVERFLOW (1) mark pixels whose ramp was fit around a
# jump or with fewer samples.  Over 1024 TARS stars their fluxes sit on
# the clean-point spline to 1% (MAD 3%), but their quoted errors are too
# small (median |resid/err| of 2 and 10) and for individual stars the
# flag can come from a neighbour's pixel inside the aperture.  They are
# therefore "soft": the spline is fit to clean points only, and a soft
# point is used (in fit_mask and the binned spectrum) only if it agrees
# with that spline within SOFT_NSIGMA (with SOFT_ERR_FLOOR added in
# quadrature) and a factor SOFT_MAX_RATIO.  The other bits are "hard".
SOFT_FLAG_BITS = (1 << 0) | (1 << 1)
HARD_FLAG_BITS = ((1 << 2) | (1 << 6) | (1 << 7) | (1 << 9) | (1 << 10)
                  | (1 << 11) | (1 << 15))
INCLUDE_SOFT_FLAGS = True
SOFT_ERR_FLOOR = 0.03  # fractional, added in quadrature to soft points
SOFT_NSIGMA = 3.0
SOFT_MAX_RATIO = 1.25
# Stars whose centre is within this many pixels of the array edge have
# part of the 2-pixel aperture off the detector (median flux ratio 0.86
# at 0-1 px, 0.98 at 1-2 px, 0.99 beyond).
DETECTOR_NPIX = 2040
EDGE_MIN_PIX = 2.0
# Gross-outlier prefilter: ratio to a running median of the neighbouring
# points outside [1/OUTLIER_FACTOR, OUTLIER_FACTOR] and more than
# OUTLIER_NSIGMA away; stellar features at R~40 never reach a factor 1.8.
OUTLIER_WINDOW = 21
OUTLIER_FACTOR = 1.8
OUTLIER_NSIGMA = 5.0

# Bits of the ``mask_bits`` column.
MASK_HARD = 1
MASK_SOFT = 2
MASK_EDGE = 4
MASK_OUTLIER = 8


def running_median(values: np.ndarray, window: int = OUTLIER_WINDOW
                   ) -> np.ndarray:
    """Centred running median (NaN-aware) of a 1-D array."""
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rolling(window, center=True, min_periods=1).median(
    ).to_numpy()


def gross_outliers(
    wavelength_um: np.ndarray,
    flux_jy: np.ndarray,
    flux_err_jy: np.ndarray,
    candidate: np.ndarray,
    window: int = OUTLIER_WINDOW,
    factor: float = OUTLIER_FACTOR,
    nsigma: float = OUTLIER_NSIGMA,
) -> np.ndarray:
    """Flag points far from the running median of their neighbours.

    Args:
        wavelength_um: Wavelengths.
        flux_jy: Fluxes.
        flux_err_jy: Errors.
        candidate: Points eligible for the running median and for
            flagging (others are ignored).
        window: Running-median window in points (sorted by wavelength).
        factor: Flag if flux / median is outside [1/factor, factor] ...
        nsigma: ... and |flux - median| exceeds nsigma * error.

    Returns:
        Boolean array, True for gross outliers.
    """
    out = np.zeros(flux_jy.size, dtype=bool)
    idx = np.flatnonzero(candidate & np.isfinite(flux_jy)
                         & np.isfinite(wavelength_um))
    if idx.size < 5:
        return out
    order = idx[np.argsort(wavelength_um[idx])]
    med = running_median(flux_jy[order], window)
    f = flux_jy[order]
    e = flux_err_jy[order]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = f / med
        dev = np.abs(f - med) / np.where(e > 0, e, np.inf)
    bad = np.isfinite(ratio) & ((ratio > factor) | (ratio < 1.0 / factor))
    bad &= dev > nsigma
    out[order[bad]] = True
    return out


def quality_masks(
    rows: pd.DataFrame,
    include_soft: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask_bits, masked) for per-image rows.

    Args:
        rows: Rows with flag_or, x, y, wavelength_um, flux_jy,
            flux_err_jy.
        include_soft: Treat TRANSIENT/OVERFLOW-only points as unmasked
            (``star_products`` instead admits them individually).

    Returns:
        mask_bits (int8; MASK_HARD, MASK_SOFT, MASK_EDGE, MASK_OUTLIER)
        and the boolean ``masked`` (hard, edge or outlier; plus soft if
        ``include_soft`` is False).
    """
    flags = rows["flag_or"].to_numpy().astype(np.int64)
    hard = (flags & HARD_FLAG_BITS) != 0
    soft = ((flags & SOFT_FLAG_BITS) != 0) & ~hard
    x = rows["x"].to_numpy(float)
    y = rows["y"].to_numpy(float)
    last = DETECTOR_NPIX - 1
    edge_dist = np.minimum(np.minimum(x, y), np.minimum(last - x, last - y))
    edge = ~(edge_dist >= EDGE_MIN_PIX)  # NaN positions count as edge
    masked = hard | edge | (soft & ~include_soft)
    outlier = gross_outliers(
        rows["wavelength_um"].to_numpy(float),
        rows["flux_jy"].to_numpy(float),
        rows["flux_err_jy"].to_numpy(float),
        ~masked,
    )
    bits = (hard * MASK_HARD + soft * MASK_SOFT + edge * MASK_EDGE
            + outlier * MASK_OUTLIER).astype(np.int8)
    return bits, masked | outlier


def bin_spectrum(
    wavelength_um: np.ndarray,
    flux_jy: np.ndarray,
    flux_err_jy: np.ndarray,
    good: np.ndarray,
    edges: np.ndarray,
    model_flux: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Inverse-variance bin a spectrum onto fixed wavelength bins.

    Args:
        wavelength_um: Wavelengths.
        flux_jy: Fluxes.
        flux_err_jy: Errors.
        good: Points to include.
        edges: Bin edges (um).
        model_flux: Optional per-point model to bin the same way.

    Returns:
        Dict with flux, err (1/sqrt(sum w)), std (weighted scatter), n,
        and model (NaN where empty), each of length n_bins.
    """
    n_bins = len(edges) - 1
    out = {k: np.full(n_bins, np.nan, dtype=np.float32)
           for k in ("flux", "err", "std", "model")}
    out["n"] = np.zeros(n_bins, dtype=np.int16)
    sel = good & np.isfinite(wavelength_um) & np.isfinite(flux_jy)
    sel &= np.isfinite(flux_err_jy) & (flux_err_jy > 0)
    if not sel.any():
        return out
    lam = wavelength_um[sel]
    f = flux_jy[sel]
    e = flux_err_jy[sel]
    idx = np.searchsorted(edges, lam, side="right") - 1
    inside = (idx >= 0) & (idx < n_bins)
    idx, f, e = idx[inside], f[inside], e[inside]
    w = 1.0 / e ** 2
    sw = np.bincount(idx, weights=w, minlength=n_bins)
    swf = np.bincount(idx, weights=w * f, minlength=n_bins)
    swf2 = np.bincount(idx, weights=w * f * f, minlength=n_bins)
    n = np.bincount(idx, minlength=n_bins)
    has = sw > 0
    mean = np.where(has, swf / np.where(has, sw, 1.0), np.nan)
    var = np.where(has, swf2 / np.where(has, sw, 1.0) - mean ** 2, np.nan)
    out["flux"][:] = mean
    out["err"][:] = np.where(has, 1.0 / np.sqrt(np.where(has, sw, 1.0)),
                             np.nan)
    out["std"][:] = np.sqrt(np.clip(var, 0, None))
    out["n"][:] = n
    if model_flux is not None:
        m = model_flux[sel][inside]
        ok = np.isfinite(m)
        swm = np.bincount(idx[ok], weights=(w * m)[ok], minlength=n_bins)
        swo = np.bincount(idx[ok], weights=w[ok], minlength=n_bins)
        hasm = swo > 0
        out["model"][:] = np.where(hasm, swm / np.where(hasm, swo, 1.0),
                                   np.nan)
    return out


def star_products(
    rows: pd.DataFrame,
    cfg: dict,
    edges: np.ndarray,
) -> tuple[pd.DataFrame, dict, dict]:
    """Build the spectrum table, summary row and binned spectrum.

    Args:
        rows: Per-image rows of one star joined with the image index.
        cfg: ``phot`` section of the run config.
        edges: Bin edges for the binned spectrum.

    Returns:
        (spectrum, summary, binned).
    """
    rows = rows.sort_values("wavelength_um").reset_index(drop=True)
    base = to_legacy_frame(rows, cfg)
    n = len(base)
    # The spline sees clean points only (soft points masked here).
    mask_bits, masked = quality_masks(rows, include_soft=False)
    base["masked"] = masked
    soft = (mask_bits & MASK_SOFT) != 0
    # Soft-flagged points carry an error floor (their quoted errors are
    # too small); it also enters the binned spectrum.
    err_eff = base["flux_err_jy"].to_numpy(float).copy()
    err_eff[soft] = np.sqrt(
        err_eff[soft] ** 2
        + (SOFT_ERR_FLOOR * np.abs(base["flux_jy"].to_numpy(float)[soft]))
        ** 2)
    base["flux_err_jy"] = err_eff
    spline_ok = False
    if n >= 4:
        try:
            supp = supplement_with_spline(base, skip_failed_bands=True)
            spline_ok = True
        except (ValueError, IndexError) as exc:
            log.debug("spline skipped: %s", exc)
    if not spline_ok:
        supp = base.copy()
        supp["model_flux"] = np.nan
        supp["fit_mask"] = ~base["masked"].to_numpy()
        supp["residuals"] = np.nan

    lam = base["wavelength_um"].to_numpy(float)
    fl = base["flux_jy"].to_numpy(float)
    er = base["flux_err_jy"].to_numpy(float)  # with the soft floor
    mk = base["masked"].to_numpy(bool)
    fit = supp["fit_mask"].to_numpy(bool)
    model = supp["model_flux"].to_numpy(float)
    finite = np.isfinite(fl) & np.isfinite(er) & (er > 0)
    # Soft points are admitted if they agree with the clean spline.
    other = (mask_bits & (MASK_HARD | MASK_EDGE | MASK_OUTLIER)) != 0
    soft_cand = soft & ~other & finite
    if INCLUDE_SOFT_FLAGS and spline_ok:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = fl / model
            dev = np.abs(fl - model) / er
        soft_ok = (soft_cand & np.isfinite(ratio)
                   & (ratio < SOFT_MAX_RATIO) & (ratio > 1.0 / SOFT_MAX_RATIO)
                   & (dev < SOFT_NSIGMA))
    else:
        soft_ok = np.zeros(n, dtype=bool)
    mk = other | (soft & ~soft_ok)
    good = ~mk & finite
    # Points in bands the spline could not fit (model NaN) are kept.
    used = good & (fit | soft_ok | ~np.isfinite(model))
    mjd = base["mjd_avg"].to_numpy(float)

    spectrum = pd.DataFrame({
        "wavelength_um": lam, "bandwidth_um": base["bandwidth_um"],
        "flux_jy": fl, "flux_err_jy": er, "masked": mk,
        "mask_bits": mask_bits, "fit_mask": used,
        "model_flux": model, "mjd_avg": mjd,
        "image_key": rows["image_key"].to_numpy(),
        "x": rows["x"].to_numpy(), "y": rows["y"].to_numpy(),
        "flag_or": rows["flag_or"].to_numpy(),
        "n_bad_pix": rows["n_bad_pix"].to_numpy(),
        "bkg_jy": rows["bkg_jy"].to_numpy(),
        "bkg_mode": rows["bkg_mode"].to_numpy(),
        "flux_raw_jy": rows["flux_raw_jy"].to_numpy(),
        "zodi_jy": rows["zodi_jy"].to_numpy(),
        "n_ann_pix": rows["n_ann_pix"].to_numpy(),
    })
    for col, dt in SPECTRUM_COLUMNS.items():
        spectrum[col] = spectrum[col].astype(dt)

    binned = bin_spectrum(lam, fl, er, used, edges, model)
    snr = fl[used] / er[used] if used.any() else np.array([np.nan])
    med_snr = float(np.nanmedian(snr)) if used.any() else np.nan
    resid = (fl - model) / er
    summary = {
        "n_rows": n, "n_good": int(good.sum()), "n_used": int(used.sum()),
        "n_masked": int(mk.sum()), "n_clipped": int((good & ~used).sum()),
        "n_hard": int(((mask_bits & MASK_HARD) != 0).sum()),
        "n_soft": int(soft.sum()),
        "n_soft_used": int(soft_ok.sum()),
        "n_edge": int(((mask_bits & MASK_EDGE) != 0).sum()),
        "n_outlier": int(((mask_bits & MASK_OUTLIER) != 0).sum()),
        "n_neg": int((good & (fl <= 0)).sum()),
        "n_bkg_fallback": int((rows["bkg_mode"].to_numpy() != 0).sum()),
        "med_snr": med_snr,
        "spline_ok": bool(spline_ok),
        "resid_mad": float(np.nanmedian(np.abs(resid[used])))
        if used.any() and spline_ok else np.nan,
        "mjd_min": float(np.nanmin(mjd)) if n else np.nan,
        "mjd_max": float(np.nanmax(mjd)) if n else np.nan,
        "n_epochs": int(np.unique(np.floor(mjd[np.isfinite(mjd)])).size),
    }
    for k, (lo, hi) in enumerate(BAND_WINDOWS_UM, start=1):
        sel = used & (lam >= lo) & (lam < hi)
        summary[f"n_b{k}"] = int(sel.sum())
        summary[f"flux_b{k}"] = (float(np.nanmedian(fl[sel])) if sel.any()
                                 else np.nan)
    # Simple empirical colours from the binned spectrum (log10 flux ratio).
    bf = binned["flux"].astype(float)
    centers = np.sqrt(edges[:-1] * edges[1:])

    def _mean_in(lo, hi):
        s = (centers >= lo) & (centers < hi) & np.isfinite(bf) & (bf > 0)
        return float(np.nanmean(bf[s])) if s.any() else np.nan

    f1 = _mean_in(0.85, 1.0)
    f2 = _mean_in(1.5, 1.7)
    f4 = _mean_in(3.4, 3.8)
    f6 = _mean_in(4.5, 5.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        summary["c_1p6_0p9"] = float(np.log10(f2 / f1))
        summary["c_3p6_1p6"] = float(np.log10(f4 / f2))
        summary["c_4p7_3p6"] = float(np.log10(f6 / f4))
    return spectrum, summary, binned


def build_bucket(
    run_dir: Path,
    bucket: int,
    out_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> dict:
    """Write the viz products of one bucket.

    Args:
        run_dir: Gathered run directory.
        bucket: Bucket number.
        out_dir: Output root (default ``<run>/viz``).
        overwrite: Rebuild even if the outputs exist.

    Returns:
        Dict with bucket, n_stars, n_rows, seconds.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "viz"
    spec_path = out_dir / "spectra" / f"bucket={bucket}.parquet"
    summ_path = out_dir / "summary" / f"bucket={bucket}.parquet"
    binned_path = out_dir / "binned" / f"bucket={bucket}.npz"
    if (not overwrite and spec_path.exists() and summ_path.exists()
            and binned_path.exists()):
        return {"bucket": bucket, "skipped": True}
    for p in (spec_path, summ_path, binned_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cfg = json.loads((run_dir / "config.json").read_text())["phot"]
    edges = spherex_bin_edges()
    bucket_dir = run_dir / "bystar" / f"bucket={bucket}"
    if not bucket_dir.exists():
        return {"bucket": bucket, "n_stars": 0, "n_rows": 0,
                "seconds": 0.0, "missing": True}
    table = ds.dataset(str(bucket_dir), format="parquet").to_table()
    rows_all = table.to_pandas()
    idx = _run_index(run_dir)[["image_key", "path"]]
    # Only the path is needed by to_legacy_frame; keep the merge small.
    rows_all = rows_all.merge(idx, on="image_key", how="left")
    cat = pd.read_parquet(run_dir / "catalog.parquet")

    specs, summaries, binned_rows = [], [], []
    row_group = 0
    for tic, rows in rows_all.groupby("ticid", sort=True):
        spectrum, summary, binned = star_products(rows, cfg, edges)
        sidx = int(rows["star_idx"].iloc[0])
        crow = cat.iloc[sidx]
        summary.update({
            "ticid": int(tic), "star_idx": sidx, "bucket": int(bucket),
            "row_group": row_group, "ra": float(crow["ra"]),
            "dec": float(crow["dec"]),
        })
        spectrum.insert(0, "ticid", np.int64(tic))
        specs.append(spectrum)
        summaries.append(summary)
        binned_rows.append(binned)
        row_group += 1

    if specs:
        tmp = spec_path.with_suffix(".parquet.tmp")
        schema = pa.Schema.from_pandas(specs[0], preserve_index=False)
        with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
            for spectrum in specs:
                writer.write_table(
                    pa.Table.from_pandas(spectrum, schema=schema,
                                         preserve_index=False),
                    row_group_size=max(len(spectrum), 1),
                )
        os.replace(tmp, spec_path)
        summ = pd.DataFrame(summaries)
        summ.to_parquet(summ_path, index=False)
        np.savez_compressed(
            binned_path,
            ticid=summ["ticid"].to_numpy(np.int64),
            flux=np.stack([b["flux"] for b in binned_rows]),
            err=np.stack([b["err"] for b in binned_rows]),
            std=np.stack([b["std"] for b in binned_rows]),
            n=np.stack([b["n"] for b in binned_rows]),
            model=np.stack([b["model"] for b in binned_rows]),
            edges=edges,
        )
    return {"bucket": bucket, "n_stars": len(specs), "n_rows": len(rows_all),
            "seconds": round(time.time() - t0, 1)}


def finalize(run_dir: Path, out_dir: Optional[Path] = None) -> dict:
    """Concatenate the per-bucket summaries and binned spectra.

    Writes ``summary.parquet`` and ``binned_{ticid,flux,err,std,n,model}
    .npy`` plus ``bin_edges.npy`` (rows aligned) under the viz directory.

    Returns:
        Dict with n_stars and n_buckets.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "viz"
    files = sorted((out_dir / "summary").glob("bucket=*.parquet"),
                   key=lambda p: int(p.stem.split("=")[1]))
    summ = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    flux, err, std, n, model, tics = [], [], [], [], [], []
    edges = None
    for f in files:
        b = int(f.stem.split("=")[1])
        with np.load(out_dir / "binned" / f"bucket={b}.npz") as z:
            tics.append(z["ticid"])
            flux.append(z["flux"])
            err.append(z["err"])
            std.append(z["std"])
            n.append(z["n"])
            model.append(z["model"])
            edges = z["edges"]
    tics = np.concatenate(tics)
    if not np.array_equal(tics, summ["ticid"].to_numpy()):
        raise RuntimeError("binned/summary ticid order mismatch")
    # Write to temporary names and rename, so a running server keeps its
    # memory maps of the previous files (never truncate in place).
    tmp = out_dir / "summary.parquet.tmp"
    summ.to_parquet(tmp, index=False)
    os.replace(tmp, out_dir / "summary.parquet")
    arrays = {
        "binned_ticid": tics, "binned_flux": np.concatenate(flux),
        "binned_err": np.concatenate(err), "binned_std": np.concatenate(std),
        "binned_n": np.concatenate(n), "binned_model": np.concatenate(model),
        "bin_edges": edges,
    }
    for name, arr in arrays.items():
        tmp = out_dir / f"{name}.npy.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.replace(tmp, out_dir / f"{name}.npy")
    meta = {"n_stars": int(len(summ)), "n_buckets": len(files),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / "viz.json").write_text(json.dumps(meta, indent=2))
    return meta


class SpectrumStore:
    """Fast per-star reads from the viz spectra tables.

    Args:
        viz_dir: ``<run>/viz``.
        summary: The summary DataFrame (indexed lookups of bucket and
            row_group); loaded from ``summary.parquet`` if omitted.
    """

    def __init__(self, viz_dir: Path,
                 summary: Optional[pd.DataFrame] = None,
                 max_open: int = 64) -> None:
        self.viz_dir = Path(viz_dir)
        # Each open ParquetFile keeps its parsed footer (1024 row groups,
        # ~17 MB) on the heap, so only the most recently used files stay
        # open; a cold open costs ~50 ms.
        self.max_open = int(max_open)
        if summary is None:
            summary = pd.read_parquet(
                self.viz_dir / "summary.parquet",
                columns=["ticid", "bucket", "row_group"])
        self._loc = {
            int(t): (int(b), int(r)) for t, b, r in zip(
                summary["ticid"].to_numpy(), summary["bucket"].to_numpy(),
                summary["row_group"].to_numpy())
        }
        self._files: dict[int, object] = {}

    def __contains__(self, ticid: int) -> bool:
        return int(ticid) in self._loc

    def _file(self, bucket: int):
        import pyarrow.parquet as pq

        f = self._files.pop(bucket, None)
        if f is None:
            f = pq.ParquetFile(
                self.viz_dir / "spectra" / f"bucket={bucket}.parquet")
            while len(self._files) >= self.max_open:
                self._files.pop(next(iter(self._files)))
        self._files[bucket] = f  # most recently used last
        return f

    def prewarm(self) -> int:
        """Open up to ``max_open`` spectra files ahead of the first reads.

        Returns the number of files opened.
        """
        buckets = sorted({b for b, _ in self._loc.values()})[:self.max_open]
        n = 0
        for b in buckets:
            try:
                self._file(b)
                n += 1
            except OSError:
                break
        return n

    def read(self, ticid: int) -> pd.DataFrame:
        """Return the spectrum table of one star.

        Raises:
            KeyError: If the star has no products.
        """
        bucket, rg = self._loc[int(ticid)]
        table = self._file(bucket).read_row_group(rg)
        df = table.to_pandas()
        if len(df) and int(df["ticid"].iloc[0]) != int(ticid):
            raise RuntimeError(f"row group mismatch for TIC {ticid}")
        df["tjd_avg"] = df["mjd_avg"] - TJD_OFFSET
        return df


def read_star_spectrum(viz_dir: Path, ticid: int) -> pd.DataFrame:
    """One-off read of a star's spectrum table (builds a store)."""
    return SpectrumStore(viz_dir).read(ticid)


__all__ = [
    "BAND_WINDOWS_UM",
    "MASK_EDGE",
    "MASK_HARD",
    "MASK_OUTLIER",
    "MASK_SOFT",
    "SPECTRUM_COLUMNS",
    "SpectrumStore",
    "bin_spectrum",
    "build_bucket",
    "finalize",
    "gross_outliers",
    "quality_masks",
    "read_star_spectrum",
    "star_products",
]
