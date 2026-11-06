#!/home/luke/miniconda3/envs/py311_sx/bin/python
"""Fit SEDs (SPHINX + ck04) to every binned spectrum of a viz build.

Reads ``<viz>/summary.parquet`` and ``<viz>/binned_*.npy`` (written by
``sx_phot.viz_products.finalize``), joins Gaia parallaxes from the TARS
table, runs ``sx_phot.sedfit.fit_star`` per star in a process pool and
writes ``<viz>/sed.parquet`` (rows aligned with the summary) including
the Teff-relative excess columns of ``sedfit.add_population_columns``.

Usage:
    python drivers/fit_viz_seds.py \
        --viz-dir /ar0/SPHEREX/bulkphot/tars_v0/viz \
        --tars /ar0/local/TARS/tars_table_2.feather --workers 8
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger("fit_viz_seds")

_STATE: dict = {}


def _init(viz_dir: str, blue_max_um: float, err_floor: float) -> None:
    os.nice(19)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    from sx_phot.sedfit import MainSequence, build_model_set

    ms = MainSequence()
    _STATE["ms"] = ms
    _STATE["models"] = build_model_set(main_sequence=ms)
    _STATE["flux"] = np.load(Path(viz_dir) / "binned_flux.npy", mmap_mode="r")
    _STATE["err"] = np.load(Path(viz_dir) / "binned_err.npy", mmap_mode="r")
    _STATE["blue_max_um"] = blue_max_um
    _STATE["err_floor"] = err_floor


def _chunk(args: tuple) -> list[dict]:
    from sx_phot.sedfit import fit_star, strip_private

    start, stop, plx = args
    flux = np.asarray(_STATE["flux"][start:stop], dtype=float)
    err = np.asarray(_STATE["err"][start:stop], dtype=float)
    out = []
    for k in range(stop - start):
        try:
            res = fit_star(
                flux[k], err[k], _STATE["models"], parallax_mas=plx[k],
                blue_max_um=_STATE["blue_max_um"],
                err_floor=_STATE["err_floor"],
                main_sequence=_STATE["ms"],
            )
        except Exception as exc:  # noqa: BLE001
            res = {"n_bins_used": 0, "error": repr(exc)}
        res["row"] = start + k
        out.append(strip_private(res))
    return out


def _postprocess(df: pd.DataFrame, viz: Path) -> None:
    """Add the Teff-relative columns (needs n_used from the summary)."""
    from sx_phot.sedfit import add_population_columns

    summ = pd.read_parquet(viz / "summary.parquet",
                           columns=["ticid", "n_used"])
    n_used = df[["ticid"]].merge(summ, on="ticid", how="left")["n_used"]
    df["n_used"] = n_used.to_numpy()
    add_population_columns(df)
    df.drop(columns=["n_used"], inplace=True)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--viz-dir", required=True)
    parser.add_argument("--tars",
                        default="/ar0/local/TARS/tars_table_2.feather")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=2000)
    parser.add_argument("--blue-max-um", type=float, default=1.7)
    parser.add_argument("--err-floor", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=None,
                        help="fit only the first N stars (testing)")
    parser.add_argument("--postprocess-only", action="store_true",
                        help="only (re)compute the population-relative "
                             "columns of an existing sed.parquet")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    viz = Path(args.viz_dir)
    out = viz / "sed.parquet"
    if args.postprocess_only:
        df = pd.read_parquet(out)
        df = df.drop(columns=[c for c in df.columns if c.endswith("_rel")
                              or c.endswith("_rel_sig")])
        _postprocess(df, viz)
        df.to_parquet(out, index=False)
        log.info("rewrote %s with population columns", out)
        return
    summ = pd.read_parquet(viz / "summary.parquet", columns=["ticid"])
    tars = pd.read_feather(args.tars, columns=["TICID", "parallax"])
    plx_map = dict(zip(tars["TICID"].astype(np.int64), tars["parallax"]))
    tic = summ["ticid"].to_numpy(np.int64)
    plx = np.array([plx_map.get(int(t), np.nan) for t in tic])
    n = len(tic) if args.limit is None else min(args.limit, len(tic))
    chunks = [(s, min(s + args.chunk, n), plx[s:min(s + args.chunk, n)])
              for s in range(0, n, args.chunk)]
    log.info("%d stars in %d chunks, %d workers", n, len(chunks),
             args.workers)
    t0 = time.time()
    results: list[dict] = []
    with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init,
            initargs=(str(viz), args.blue_max_um, args.err_floor)) as pool:
        for k, part in enumerate(pool.map(_chunk, chunks)):
            results.extend(part)
            if (k + 1) % 20 == 0 or k + 1 == len(chunks):
                log.info("%d / %d chunks, %.0f s", k + 1, len(chunks),
                         time.time() - t0)
    df = pd.DataFrame(results).sort_values("row").reset_index(drop=True)
    df.insert(0, "ticid", tic[df["row"].to_numpy()])
    df = df.drop(columns=["row"])
    _postprocess(df, viz)
    df.to_parquet(out, index=False)
    log.info("wrote %s (%d rows, %d cols) in %.0f s", out, len(df),
             df.shape[1], time.time() - t0)


if __name__ == "__main__":
    main()
