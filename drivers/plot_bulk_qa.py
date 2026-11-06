#!/home/luke/miniconda3/envs/py311_sx/bin/python
"""QA plots for a materialized bulk run (output of materialize_bulk_run).

Usage:
    python drivers/plot_bulk_qa.py \
        --csv-dir /ar0/SPHEREX/bulkphot/spyglass_viia_psp_v0/csv \
        --out-dir /ar0/SPHEREX/bulkphot/spyglass_viia_psp_v0/qa \
        --nproc 8 [--max-stars 100] [--grid-n 24]

Per star: <star_id>_spline.png (spectrum, spline model, residuals) when a
_splsupp.csv exists, else <star_id>_spectrum.png.  Each carries the
circphot-style cutout inset (image nearest --cutout-wavelength, default
0.93 um, aperture drawn) to show crowding.  Also overview_grid.png
(random sample plus the highest-S/N stars) and qa_summary.csv.
"""

from __future__ import annotations

import argparse
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

from aesthetic.plot import set_style
set_style("science")
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sx_phot.visualization import (  # noqa: E402
    _add_cutout_inset,
    plot_spectrum_with_spline,
)

log = logging.getLogger(__name__)

CUTOUT_SIZE_PIX = 64


def _cutout_for_row(
    base: pd.DataFrame, target_um: float, size_pix: int = CUTOUT_SIZE_PIX,
) -> Optional[tuple]:
    """Cut out the image nearest ``target_um`` around the star.

    Unmasked rows are preferred.  Returns (data, cutout WCS, wavelength,
    ra_deg, dec_deg) or None when no image can be read.
    """
    import warnings

    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS

    if "x" not in base.columns or "file" not in base.columns:
        return None
    lam = base["wavelength_um"].to_numpy(dtype=float)
    ok = np.isfinite(lam)
    unmasked = ok & ~base["masked"].to_numpy(dtype=bool)
    pool = unmasked if unmasked.any() else ok
    if not pool.any():
        return None
    d = np.where(pool, np.abs(lam - target_um), np.inf)
    i = int(np.argmin(d))
    path = str(base["file"].iloc[i])
    x = float(base["x"].iloc[i])
    y = float(base["y"].iloc[i])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(path, memmap=True) as hdul:
                wcs = WCS(hdul["IMAGE"].header)
                cut = Cutout2D(hdul["IMAGE"].data, (x, y),
                               (size_pix, size_pix), wcs=wcs,
                               mode="partial", fill_value=np.nan)
                sky = wcs.pixel_to_world(x, y)
    except Exception as exc:  # noqa: BLE001 - cutout is optional
        log.warning("cutout skipped for %s (%s)", path, exc)
        return None
    return (np.asarray(cut.data, dtype=float), cut.wcs, float(lam[i]),
            float(sky.ra.deg), float(sky.dec.deg))


def _plot_one(args: tuple) -> dict:
    """Worker: plot one star from its cache (and optional supp) CSV."""
    star_id, cache, supp, ra, dec, out_dir, cutout_um = args
    out_dir = Path(out_dir)
    base = pd.read_csv(cache)
    cut = _cutout_for_row(base, cutout_um) if cutout_um else None
    ap_r = float(base["aperture_radius_pix"].iloc[0]) \
        if "aperture_radius_pix" in base.columns and len(base) else 2.0
    lam = base["wavelength_um"].to_numpy(dtype=float)
    fl = base["flux_jy"].to_numpy(dtype=float)
    er = base["flux_err_jy"].to_numpy(dtype=float)
    mk = base["masked"].to_numpy(dtype=bool)
    tjd = base["tjd_avg"].to_numpy(dtype=float)
    good = ~mk & np.isfinite(fl) & (fl > 0)
    info = {"star_id": star_id, "n": int(len(base)),
            "n_masked": int(mk.sum()),
            "n_good": int(good.sum()),
            "med_snr": float(np.nanmedian(fl[good] / er[good]))
            if good.any() else np.nan,
            "tjd_min": float(np.nanmin(tjd)) if len(tjd) else np.nan,
            "tjd_max": float(np.nanmax(tjd)) if len(tjd) else np.nan,
            "med_abs_normres": np.nan, "png": ""}
    title = (f"{star_id.replace('_', ' ')}  n={len(base)} "
             f"masked={mk.sum()}\n a={ra:.4f} d={dec:.4f}  "
             f"TJD {info['tjd_min']:.0f}-{info['tjd_max']:.0f}")
    if supp and Path(supp).exists():
        s = pd.read_csv(supp)
        png = out_dir / f"{star_id}_spline.png"
        fig, ax = plot_spectrum_with_spline(
            lam, fl, er, masked=mk, fit_mask=s["fit_mask"].to_numpy(bool),
            model_flux_jy=s["model_flux"].to_numpy(float), title=title,
            show_residuals=True)
        if cut is not None:
            # The inset takes the upper-right corner; move the legend
            # outside the axes so the two never overlap.
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                      fontsize=7, frameon=False)
            _add_cutout_inset(fig, ax, cut[0], cut[2], cut[1], cut[3],
                              cut[4], ap_r)
        fig.savefig(png, dpi=250, bbox_inches="tight")
        valid = s["res_valid"].to_numpy(bool)
        if valid.any():
            r = s["residuals"].to_numpy(float)[valid] / er[valid]
            info["med_abs_normres"] = float(np.nanmedian(np.abs(r)))
    else:
        png = out_dir / f"{star_id}_spectrum.png"
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.errorbar(lam[good], fl[good], yerr=er[good], fmt=".", c="k",
                    ms=3, elinewidth=0.5, capsize=0)
        if mk.any():
            ax.plot(lam[mk], fl[mk], "x", c="r", ms=3, alpha=0.4)
        ax.set_yscale("log")
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Flux (Jy)")
        ax.set_title(title, fontsize=8)
        if cut is not None:
            _add_cutout_inset(fig, ax, cut[0], cut[2], cut[1], cut[3],
                              cut[4], ap_r)
        fig.tight_layout()
        fig.savefig(png, dpi=200)
    plt.close("all")
    info["png"] = str(png)
    return info


def overview_grid(
    summary: pd.DataFrame, out_path: Path, grid_n: int = 24, seed: int = 42,
) -> None:
    """Plot a grid of spectra: top-S/N quarter plus a random sample."""
    has = summary[summary["n"] > 0]
    if has.empty:
        return
    n_top = max(1, grid_n // 4)
    top = has.sort_values("med_snr", ascending=False).head(n_top)
    rest = has.drop(top.index)
    rng = np.random.default_rng(seed)
    k = min(grid_n - len(top), len(rest))
    pick = rest.iloc[rng.choice(len(rest), k, replace=False)] if k else rest
    items = pd.concat([top, pick])
    ncol = 4
    nrow = int(np.ceil(len(items) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow),
                            squeeze=False)
    for ax, row in zip(axs.ravel(), items.itertuples(index=False)):
        base = pd.read_csv(row.cache)
        lam = base["wavelength_um"].to_numpy(float)
        fl = base["flux_jy"].to_numpy(float)
        er = base["flux_err_jy"].to_numpy(float)
        mk = base["masked"].to_numpy(bool)
        ok = ~mk & np.isfinite(fl) & (fl > 0)
        ax.errorbar(lam[ok], fl[ok], yerr=er[ok], fmt=".", c="k", ms=3,
                    elinewidth=0.5, capsize=0)
        if mk.any():
            ax.plot(lam[mk], fl[mk], "x", c="r", ms=3, alpha=0.4)
        ax.set_yscale("log")
        ax.set_title(f"{row.star_id.replace('_', ' ')}  n={row.n}  "
                     f"S/N={row.med_snr:.0f}", fontsize=8)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
    for ax in axs.ravel()[len(items):]:
        ax.axis("off")
    fig.supxlabel("Wavelength (µm)")
    fig.supylabel("Flux (Jy)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close("all")


def main() -> None:
    """Make the QA plots."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--max-stars", type=int, default=None,
                        help="only plot the first N stars with rows")
    parser.add_argument("--grid-n", type=int, default=24)
    parser.add_argument("--cutout-wavelength", type=float, default=0.93,
                        help="inset cutout from the image nearest this "
                             "wavelength (um); 0 disables the inset")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(csv_dir / "summary.csv",
                          dtype={"cache": str, "supp": str},
                          keep_default_na=False)
    summary["n"] = summary["n"].astype(int)
    summary["med_snr"] = pd.to_numeric(summary["med_snr"], errors="coerce")
    has = summary[summary["n"] > 0]
    if args.max_stars:
        has = has.head(args.max_stars)
    tasks = [(r.star_id, r.cache, r.supp, float(r.ra), float(r.dec),
              str(out_dir), float(args.cutout_wavelength))
             for r in has.itertuples(index=False)]
    with Pool(int(args.nproc)) as pool:
        infos = list(pool.imap_unordered(_plot_one, tasks, chunksize=4))
    qa = pd.DataFrame(infos).merge(
        summary[["star_id", "ticid", "ra", "dec"]], on="star_id",
        how="left")
    qa.to_csv(out_dir / "qa_summary.csv", index=False)
    overview_grid(summary, out_dir / "overview_grid.png",
                  grid_n=args.grid_n)
    log.info("%d stars plotted -> %s", len(qa), out_dir)


if __name__ == "__main__":
    main()
