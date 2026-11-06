#!/home/luke/miniconda3/envs/py311_sx/bin/python
"""Warm the sx-viz on-demand caches (reports, cutouts) for a star list.

Renders the report PNG, the six default SPHEREx cutouts and the DSS2
sky cutouts for every star selected by a table query, so that a vetting
session over e.g. the top red-excess candidates never waits.  Uses the
server's ``Data`` class directly (no running server needed; a running
server shares the same disk cache).

Usage:
    python drivers/precompute_viz_cache.py \
        --run-dir /ar0/SPHEREX/bulkphot/tars_v0 \
        --query "min_red_excess_sig=5&min_n_used=100&sort=red_excess_sig" \
        --limit 500 [--no-sky] [--workers 4]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "apps" / "sx-viz"))

log = logging.getLogger("precompute_viz_cache")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", default="/ar0/SPHEREX/bulkphot/tars_v0")
    parser.add_argument("--viz-dir", default=None)
    parser.add_argument("--tars",
                        default="/ar0/local/TARS/tars_table_2.feather")
    parser.add_argument("--query", default="sort=red_excess_sig&desc=1",
                        help="/api/table query string selecting the stars")
    parser.add_argument("--ticid", type=int, nargs="*", default=None,
                        help="explicit TIC ids instead of --query")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-sky", action="store_true",
                        help="skip the SkyView cutouts")
    parser.add_argument("--surveys", default="dss2r,dss2ir")
    parser.add_argument("--refresh", action="store_true",
                        help="re-render reports even if cached")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    os.nice(10)
    import server  # noqa: E402  (apps/sx-viz/server.py)

    data = server.Data(Path(args.run_dir),
                       Path(args.viz_dir) if args.viz_dir else None,
                       Path(args.tars))
    if args.ticid:
        tics = [t for t in args.ticid if t in data.row_of]
    else:
        params = parse_qs(args.query)
        params["ids_only"] = ["1"]
        params["limit"] = [str(args.limit)]
        tics = data.query(params)["ids"]
    log.info("%d stars", len(tics))
    surveys = [s for s in args.surveys.split(",") if s]

    def _one(tic: int) -> str:
        t0 = time.time()
        parts = []
        try:
            data.report(tic, refresh=args.refresh)
            parts.append("report")
        except Exception as exc:  # noqa: BLE001
            parts.append(f"report FAILED ({exc!r})")
        n_cut = 0
        for c in data.default_cutout_keys(tic):
            try:
                if data.spherex_cutout(tic, int(c["image_key"])) is not None:
                    n_cut += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("TIC %d cutout %s: %r", tic, c["image_key"], exc)
        parts.append(f"{n_cut} spx")
        if not args.no_sky:
            n_sky = sum(1 for s in surveys if data.sky_cutout(tic, s))
            parts.append(f"{n_sky} sky")
        return f"TIC {tic}: {', '.join(parts)} ({time.time() - t0:.1f} s)"

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for k, msg in enumerate(pool.map(_one, tics), start=1):
            log.info("[%d/%d] %s", k, len(tics), msg)
    log.info("done in %.0f s", time.time() - t0)


if __name__ == "__main__":
    main()
