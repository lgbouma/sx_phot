#!/home/luke/miniconda3/envs/py311_sx/bin/python
"""Build the per-star viz products (spectra, summaries, binned) of a run.

Runs ``sx_phot.viz_products.build_bucket`` over every bucket with a
process pool, then ``finalize`` concatenates the summaries.  Buckets
whose outputs already exist are skipped unless ``--overwrite`` is given,
so the script can be re-run after a run is extended.

Usage:
    python drivers/build_viz_products.py \
        --run-dir /ar0/SPHEREX/bulkphot/tars_v0 \
        --workers 12 [--buckets 0 1 2] [--overwrite] [--no-finalize]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sx_phot.viz_products import build_bucket, finalize  # noqa: E402

log = logging.getLogger("build_viz_products")


def _worker(run_dir: str, bucket: int, out_dir: str, overwrite: bool) -> dict:
    os.nice(19)
    try:
        return build_bucket(Path(run_dir), bucket, Path(out_dir), overwrite)
    except Exception as exc:  # noqa: BLE001 - report and continue
        return {"bucket": bucket, "error": repr(exc)}


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default=None,
                        help="default <run-dir>/viz")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--buckets", type=int, nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-finalize", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = args.buckets
    if buckets is None:
        buckets = sorted(
            int(p.name.split("=")[1])
            for p in (run_dir / "bystar").glob("bucket=*"))
    log.info("%d buckets, %d workers -> %s", len(buckets), args.workers,
             out_dir)
    t0 = time.time()
    n_done = n_err = n_skip = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_worker, str(run_dir), b, str(out_dir),
                        args.overwrite): b for b in buckets
        }
        for fut in as_completed(futures):
            res = fut.result()
            if "error" in res:
                n_err += 1
                log.error("bucket %d failed: %s", res["bucket"],
                          res["error"])
            elif res.get("skipped"):
                n_skip += 1
            else:
                n_done += 1
                log.info("bucket %d: %d stars, %d rows, %.1f s  "
                         "[%d done, %d skipped, %d failed, %.0f s]",
                         res["bucket"], res["n_stars"], res["n_rows"],
                         res["seconds"], n_done, n_skip, n_err,
                         time.time() - t0)
    log.info("done: %d built, %d skipped, %d failed in %.0f s", n_done,
             n_skip, n_err, time.time() - t0)
    if not args.no_finalize and n_err == 0:
        meta = finalize(run_dir, out_dir)
        log.info("finalized: %s", meta)
    elif n_err:
        log.warning("not finalizing because of failures; re-run to retry")


if __name__ == "__main__":
    main()
