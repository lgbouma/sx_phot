#!/usr/bin/env python
"""Build the empirical one-parameter spectral sequence of a viz build.

Reads the binned spectra and summary of ``<viz-dir>``, joins the TARS
table, selects a clean main-sequence training set, runs the
principal-curve iteration (``sx_phot.specml.build_sequence``) and writes
the library to ``<out-dir>/sequence.npz`` (+ ``sequence_nodes.csv``,
``sequence_train.parquet`` with the training stars' parameters, and
``sequence.json`` with the selection and iteration history).

Usage:
    OMP_NUM_THREADS=32 python drivers/build_spectral_sequence.py \
        --viz-dir /ar0/SPHEREX/bulkphot/tars_v0/viz \
        --out-dir /ar0/SPHEREX/bulkphot/tars_v0/ml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sx_phot import specml  # noqa: E402

log = logging.getLogger("build_spectral_sequence")

DEFAULT_TARS = Path("/ar0/local/TARS/tars_table_2.feather")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--viz-dir", type=Path, required=True)
    p.add_argument("--tars", type=Path, default=DEFAULT_TARS)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default <viz-dir>/../ml")
    p.add_argument("--n-nodes", type=int, default=192)
    p.add_argument("--n-iter", type=int, default=6)
    p.add_argument("--n-grid", type=int, default=2048)
    p.add_argument("--min-snr", type=float, default=50.0)
    p.add_argument("--max-a0", type=float, default=0.2)
    p.add_argument("--max-ruwe", type=float, default=1.4)
    p.add_argument("--ms-window", type=float, nargs=2, default=(-0.4, 0.5),
                   help="M_G0 offset from the main sequence (bright, faint)")
    p.add_argument("--err-floor", type=float, default=0.02)
    p.add_argument("--max-contams", type=int, default=0,
                   help="max TESS contaminants (-1: no cut)")
    p.add_argument("--clip-factor", type=float, default=4.0,
                   help="chi2 clip relative to the median (0: none)")
    p.add_argument("--no-deredden", action="store_true")
    p.add_argument("--max-stars", type=int, default=None,
                   help="random subsample of the clean set (tests)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    viz = args.viz_dir
    out = args.out_dir or (viz.parent / "ml")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cat = specml.load_catalog(viz, args.tars)
    lam = specml.bin_centers(viz)
    flux = np.load(viz / "binned_flux.npy", mmap_mode="r")
    err = np.load(viz / "binned_err.npy", mmap_mode="r")
    nper = np.load(viz / "binned_n.npy", mmap_mode="r")
    a0 = None if args.no_deredden else cat["a0"].to_numpy(dtype=float)
    logf, sig, valid = specml.prepare_log_flux(
        flux, err, nper, lam, a0=a0, err_floor=args.err_floor)
    y, offset = specml.normalize_rows(logf, valid)
    _, sig0, _ = specml.prepare_log_flux(flux, err, nper, lam, a0=a0,
                                         err_floor=0.0)
    log.info("loaded %d spectra in %.0f s", len(y), time.time() - t0)

    clean = specml.select_clean(
        cat, valid, min_snr=args.min_snr, max_a0=args.max_a0,
        max_ruwe=args.max_ruwe, ms_window=tuple(args.ms_window),
        max_contams=None if args.max_contams < 0 else args.max_contams)
    idx = np.flatnonzero(clean)
    if args.max_stars and len(idx) > args.max_stars:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(idx, args.max_stars, replace=False))
    log.info("clean training set: %d of %d stars", len(idx), len(y))

    fit = specml.build_sequence(
        y[idx], sig[idx], cat["bprp0"].to_numpy(dtype=float)[idx],
        n_nodes=args.n_nodes, n_iter=args.n_iter, n_grid=args.n_grid,
        clip_factor=args.clip_factor or None)
    meta = {
        "viz_dir": str(viz), "tars": str(args.tars),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_train": int(len(idx)), "n_inlier": int(fit.inlier.sum()),
        "n_total": int(len(y)), "max_contams": args.max_contams,
        "clip_factor": args.clip_factor,
        "n_nodes": args.n_nodes, "n_iter": args.n_iter,
        "min_snr": args.min_snr, "max_a0": args.max_a0,
        "max_ruwe": args.max_ruwe, "ms_window": list(args.ms_window),
        "err_floor": args.err_floor, "deredden": not args.no_deredden,
        "init": "bprp0", "history": fit.history,
    }
    lib = specml.assemble_library(fit, y[idx], sig[idx], offset[idx],
                                  cat.iloc[idx].reset_index(drop=True), lam,
                                  meta=meta, sigma_meas=sig0[idx])
    lib.save(out / "sequence")
    train = pd.DataFrame({"ticid": cat["ticid"].to_numpy()[idx],
                          "row": idx, "t": fit.t, "chi2": fit.chi2,
                          "inlier": fit.inlier})
    train.to_parquet(out / "sequence_train.parquet", index=False)
    (out / "sequence.json").write_text(json.dumps(meta, indent=1))
    s = lib.summary
    log.info("nodes: t=0 Teff %.0f K, t=1 Teff %.0f K; counts %d-%d",
             s["teff_tars_med"].iloc[0], s["teff_tars_med"].iloc[-1],
             s["counts"].min(), s["counts"].max())
    log.info("wrote %s (%.0f s total)", out / "sequence.npz",
             time.time() - t0)


if __name__ == "__main__":
    main()
