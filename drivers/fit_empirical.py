#!/usr/bin/env python
"""Fit the empirical spectral sequence to every star of a viz build.

Reads ``<ml-dir>/sequence.npz`` (``drivers/build_spectral_sequence.py``)
and the binned spectra of ``<viz-dir>``, runs ``sx_phot.empfit.fit_chunk``
over chunks of stars in parallel and writes ``<ml-dir>/empfit.parquet``
(one row per star, aligned with ``binned_ticid.npy``).

Columns (NaN where undefined):

| group | columns |
|---|---|
| single | t_single, teff_emp, scale_single, chi2_single, dof_single, lum_ratio, radius_ratio_emp, dmag_ms, radius_emp |
| residuals | excess_b1..b6 (fractional), excess_b1..b6_sig, red_excess(_sig), blue_excess(_sig), max_abs_z, lam_max_abs_z, n_bins_4sig, resid_a1..a3 (dex) |
| main sequence at the Gaia distance | t_ms, chi2_ms |
| binary (shape) | t_bin1, t_bin2, teff_bin1, teff_bin2, scale_binary, chi2_binary, dchi2_binary, flux_ratio_binary, lum_ratio_binary |
| binary (Gaia distance) | t_bin1_ms, t_bin2_ms, teff_bin1_ms, teff_bin2_ms, chi2_binary_ms, dchi2_binary_ms |
| disk | t_disk, teff_disk_star, scale_disk, teff_disk, omega_disk, chi2_disk, dchi2_disk, disk_frac_b6, f_disk |
| BIC | bic_single, bic_ms, bic_binary, dbic_binary, bic_binary_ms, dbic_binary_ms, bic_disk, dbic_disk, dbic_disk_vs_binary |
| population | <col>_pct percentile ranks at fixed t_single |

Usage:
    python drivers/fit_empirical.py --viz-dir /ar0/SPHEREX/bulkphot/tars_v0/viz \
        --workers 40
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sx_phot import empfit, specml  # noqa: E402

log = logging.getLogger("fit_empirical")
DEFAULT_TARS = Path("/ar0/local/TARS/tars_table_2.feather")

_G: dict = {}


def _init(viz_dir: str, lib_path: str, a0: np.ndarray, plx: np.ndarray,
          opts: dict) -> None:
    viz = Path(viz_dir)
    _G["flux"] = np.load(viz / "binned_flux.npy", mmap_mode="r")
    _G["err"] = np.load(viz / "binned_err.npy", mmap_mode="r")
    _G["n"] = np.load(viz / "binned_n.npy", mmap_mode="r")
    _G["a0"] = a0
    _G["plx"] = plx
    _G["em"] = empfit.EmpiricalModels(specml.SpectralSequence.load(lib_path))
    _G["opts"] = opts


def _work(bounds: tuple) -> pd.DataFrame:
    lo, hi = bounds
    o = _G["opts"]
    df = empfit.fit_chunk(
        _G["flux"][lo:hi], _G["err"][lo:hi], _G["n"][lo:hi],
        _G["plx"][lo:hi], _G["a0"][lo:hi] if o["deredden"] else None,
        _G["em"], err_floor=o["err_floor"], do_binary=o["binary"],
        do_disk=o["disk"])
    df.insert(0, "row", np.arange(lo, hi))
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--viz-dir", type=Path, required=True)
    p.add_argument("--ml-dir", type=Path, default=None,
                   help="default <viz-dir>/../ml")
    p.add_argument("--tars", type=Path, default=DEFAULT_TARS)
    p.add_argument("--out", type=Path, default=None,
                   help="default <ml-dir>/empfit.parquet")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk", type=int, default=500)
    p.add_argument("--err-floor", type=float, default=empfit.DEFAULT_ERR_FLOOR)
    p.add_argument("--no-binary", action="store_true")
    p.add_argument("--no-disk", action="store_true")
    p.add_argument("--no-deredden", action="store_true")
    p.add_argument("--max-stars", type=int, default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    viz = args.viz_dir
    ml = args.ml_dir or (viz.parent / "ml")
    out = args.out or (ml / "empfit.parquet")
    t0 = time.time()
    cat = specml.load_catalog(viz, args.tars)
    n = len(cat) if not args.max_stars else min(args.max_stars, len(cat))
    opts = {"err_floor": args.err_floor, "binary": not args.no_binary,
            "disk": not args.no_disk, "deredden": not args.no_deredden}
    bounds = [(lo, min(lo + args.chunk, n)) for lo in range(0, n, args.chunk)]
    log.info("fitting %d stars in %d chunks with %d workers", n,
             len(bounds), args.workers)
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    init_args = (str(viz), str(ml / "sequence"),
                 cat["a0"].to_numpy(dtype=float),
                 cat["plx"].to_numpy(dtype=float), opts)
    parts = []
    done = 0
    with ctx.Pool(args.workers, initializer=_init, initargs=init_args) as pool:
        for df in pool.imap_unordered(_work, bounds, chunksize=1):
            parts.append(df)
            done += 1
            if done % 100 == 0 or done == len(bounds):
                log.info("%d / %d chunks (%.0f s)", done, len(bounds),
                         time.time() - t0)
    table = pd.concat(parts).sort_values("row").reset_index(drop=True)
    table.insert(0, "ticid", cat["ticid"].to_numpy()[table["row"].to_numpy()])
    table = empfit.add_population_columns(table)
    table.to_parquet(out, index=False)
    ok = np.isfinite(table["chi2_single"])
    log.info("wrote %s: %d rows, %d fitted; median chi2/dof %.2f (%.0f s)",
             out, len(table), ok.sum(),
             np.nanmedian(table["chi2_single"] / table["dof_single"]),
             time.time() - t0)


if __name__ == "__main__":
    main()
