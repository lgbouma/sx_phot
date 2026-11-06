#!/home/luke/miniconda3/envs/py311_sx/bin/python
"""Write per-star CSV products for every star in a gathered bulk run.

Usage:
    python drivers/materialize_bulk_run.py \
        --run-dir /ar0/SPHEREX/bulkphot/spyglass_viia_psp_v0 \
        --out-dir /ar0/SPHEREX/bulkphot/spyglass_viia_psp_v0/csv \
        --id-prefix GaiaDR3 --nproc 8

Writes sxphot_cache_<prefix>_<id>_<radec>_annulus.csv (and _splsupp.csv)
per star plus summary.csv (one row per catalog star).
For a QA sample of a large run, --sample 200 (or --ticid-file) limits
the output to those stars and reads only their buckets.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from sx_phot.bulk_products import materialize_run


def main() -> None:
    """Materialize all stars of a run."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--id-prefix", default="TIC",
                        help="filename label prefix (TIC, GaiaDR3, ...)")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--no-supplement", action="store_true")
    parser.add_argument("--ticid-file",
                        help="only these catalog ids (one per line)")
    parser.add_argument("--sample", type=int, default=0,
                        help="only a random sample of this many catalog "
                             "stars (for QA of large runs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ticids = None
    if args.ticid_file:
        ticids = [int(x) for x in Path(args.ticid_file).read_text().split()]
    if args.sample > 0:
        cat = pd.read_parquet(Path(args.run_dir) / "catalog.parquet",
                              columns=["ticid"])
        if ticids is not None:
            cat = cat[cat["ticid"].isin(ticids)]
        ticids = cat["ticid"].sample(
            n=min(args.sample, len(cat)), random_state=args.seed,
        ).tolist()
        logging.info("sampled %d stars", len(ticids))
    summary = materialize_run(
        args.run_dir, args.out_dir, id_prefix=args.id_prefix,
        supplement=not args.no_supplement, nproc=args.nproc,
        ticids=ticids,
    )
    logging.info("%d stars in catalog, %d with rows, %d spline products",
                 len(summary), int((summary["n"] > 0).sum()),
                 int(summary["spline_ok"].sum()))


if __name__ == "__main__":
    main()
