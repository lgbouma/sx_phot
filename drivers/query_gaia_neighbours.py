#!/home/luke/miniconda3/envs/py311_sx/bin/python
"""Fetch Gaia DR3 neighbours of every catalog star for PSF photometry.

Usage:
    python drivers/query_gaia_neighbours.py \
        --catalog /ar0/local/sx-targets/list.csv --id-col source_id \
        --out /ar0/PROJ/<user>/list_neighbours.parquet

The query runs in chunks against the Gaia archive (TAP upload join) and
resumes from ``<out>.parts/`` if interrupted.  Use the same catalog
options as drivers/run_bulk_phot.py so star_idx values agree.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sx_phot.bulk_phot import load_catalog
from sx_phot.gaia_field import query_gaia_neighbours


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--id-col", default="TICID")
    parser.add_argument("--ra-col", default="ra")
    parser.add_argument("--dec-col", default="dec")
    parser.add_argument("--pmra-col", default="pmra")
    parser.add_argument("--pmdec-col", default="pmdec")
    parser.add_argument("--no-pm", action="store_true")
    parser.add_argument("--epoch", type=float, default=2016.0)
    parser.add_argument("--radius-arcsec", type=float, default=80.0)
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--max-rows", type=int, default=50000,
                        help="archive row limit (chunks hitting it are "
                             "split)")
    return parser.parse_args()


def main() -> None:
    """Load the catalog and run the chunked Gaia query."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    catalog = load_catalog(
        Path(args.catalog), id_col=args.id_col, ra_col=args.ra_col,
        dec_col=args.dec_col,
        pmra_col=None if args.no_pm else args.pmra_col,
        pmdec_col=None if args.no_pm else args.pmdec_col,
    )
    query_gaia_neighbours(catalog, Path(args.out), epoch_jyear=args.epoch,
                          radius_arcsec=args.radius_arcsec,
                          chunk_size=args.chunk_size,
                          max_rows=args.max_rows)


if __name__ == "__main__":
    main()
