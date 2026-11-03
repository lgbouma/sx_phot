#!/home/luke/miniconda3/envs/py311_sx/bin/python
"""Run image-centric bulk photometry for a star catalog.

Usage (TARS):
    python drivers/run_bulk_phot.py \
        --catalog /ar0/local/TARS/tars_table_2.feather \
        --run-dir /ar0/SPHEREX/bulkphot/tars_v0 --nproc 64

Re-running with the same --run-dir processes only files that were not
processed before (new downloads), so it is safe to call daily.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sx_phot.bulk_phot import (
    PhotConfig,
    init_run_dir,
    load_catalog,
    run_bulk_photometry,
)
from sx_phot.image_index import DEFAULT_INDEX_DIR, load_image_index


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--id-col", default="TICID")
    parser.add_argument("--ra-col", default="ra")
    parser.add_argument("--dec-col", default="dec")
    parser.add_argument("--pmra-col", default="pmra")
    parser.add_argument("--pmdec-col", default="pmdec")
    parser.add_argument("--no-pm", action="store_true",
                        help="Ignore proper motion columns.")
    parser.add_argument("--epoch", type=float, default=2016.0,
                        help="Catalog epoch (Julian year).")
    parser.add_argument(
        "--index", default=str(DEFAULT_INDEX_DIR / "image_index.parquet"),
    )
    parser.add_argument("--nproc", type=int, default=64)
    parser.add_argument("--stats-stride", type=int, default=8)
    parser.add_argument("--min-ann-pix", type=int, default=20,
                        help="below this many unmasked annulus pixels, "
                             "use the unmasked annulus for the background")
    parser.add_argument("--no-zodi", action="store_true")
    parser.add_argument("--min-file-age-sec", type=float, default=600.0)
    parser.add_argument("--shard-glob", default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-files-per-shard", type=int, default=None)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> None:
    """Load inputs, initialise the run directory, and run photometry."""
    args = _parse_args()
    handlers = [logging.StreamHandler()]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or str(run_dir / "run_bulk_phot.log")
    handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logging.getLogger("astropy").setLevel(logging.ERROR)
    log = logging.getLogger("run_bulk_phot")

    config = PhotConfig(
        stats_stride=args.stats_stride,
        min_ann_pix=args.min_ann_pix,
        read_zodi=not args.no_zodi,
        min_file_age_sec=args.min_file_age_sec,
        epoch_jyear=args.epoch,
    )
    catalog = load_catalog(
        Path(args.catalog), id_col=args.id_col, ra_col=args.ra_col,
        dec_col=args.dec_col,
        pmra_col=None if args.no_pm else args.pmra_col,
        pmdec_col=None if args.no_pm else args.pmdec_col,
    )
    log.info("Catalog %s: %d stars", args.catalog, len(catalog))
    index = load_image_index(Path(args.index))
    log.info("Index %s: %d images, %d local", args.index, len(index),
             int(index["path"].notna().sum()))
    init_run_dir(run_dir, catalog, index, config,
                 catalog_source=str(args.catalog))
    run_bulk_photometry(
        run_dir, catalog, index, config=config, nproc=args.nproc,
        shard_glob=args.shard_glob, max_shards=args.max_shards,
        max_files_per_shard=args.max_files_per_shard,
    )


if __name__ == "__main__":
    main()
