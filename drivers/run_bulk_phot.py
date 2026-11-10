#!/home/luke/miniconda3/envs/py311_sx/bin/python
"""Run image-centric bulk photometry for a star catalog.

Usage (TARS):
    python drivers/run_bulk_phot.py \
        --catalog /ar0/local/TARS/tars_table_2.feather \
        --run-dir /ar0/SPHEREX/bulkphot/tars_v0 --nproc 64

Re-running with the same --run-dir processes only files that were not
processed before (new downloads), so it is safe to call daily.

Gaia-forced PSF photometry (adds psf_* columns; neighbour table from
drivers/query_gaia_neighbours.py):
    python drivers/run_bulk_phot.py --catalog CAT --run-dir RUN \
        --psf --neighbours RUN_neighbours.parquet --nproc 64
"""

from __future__ import annotations

import os

# One BLAS/OpenMP thread per worker: the PSF solves are small, and with
# multithreaded BLAS every worker would start one thread per core.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
from pathlib import Path  # noqa: E402

from sx_phot.bulk_phot import (  # noqa: E402
    PhotConfig,
    init_run_dir,
    load_catalog,
    run_bulk_photometry,
)
from sx_phot.gaia_field import load_neighbours  # noqa: E402
from sx_phot.image_index import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    load_image_index,
)


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
    parser.add_argument("--psf", action="store_true",
                        help="also run Gaia-forced PSF photometry")
    parser.add_argument("--neighbours", default=None,
                        help="Gaia neighbour parquet (required with --psf)")
    parser.add_argument("--psf-drop-frac", type=float, default=0.0)
    parser.add_argument("--psf-bg-order", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Load inputs, initialise the run directory, and run photometry."""
    args = _parse_args()
    if args.psf and not args.neighbours:
        raise SystemExit("--psf needs --neighbours "
                         "(drivers/query_gaia_neighbours.py)")
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
        psf=args.psf,
        psf_drop_frac=args.psf_drop_frac,
        psf_bg_order=args.psf_bg_order,
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
    neighbours = None
    if args.psf:
        neighbours = load_neighbours(Path(args.neighbours), len(catalog))
        log.info("Neighbours %s: %d sources, %d of %d stars matched",
                 args.neighbours, len(neighbours.ra),
                 int((neighbours.target >= 0).sum()), len(catalog))
    init_run_dir(run_dir, catalog, index, config,
                 catalog_source=str(args.catalog),
                 neighbours_source=str(args.neighbours or ""))
    run_bulk_photometry(
        run_dir, catalog, index, config=config, nproc=args.nproc,
        shard_glob=args.shard_glob, max_shards=args.max_shards,
        max_files_per_shard=args.max_files_per_shard,
        neighbours=neighbours,
    )


if __name__ == "__main__":
    main()
