#!/home/luke/miniconda3/envs/py311_sx/bin/python
"""Regroup bulk photometry rows by star (run after run_bulk_phot.py).

Usage:
    python drivers/gather_bulk_phot.py --run-dir /ar0/SPHEREX/bulkphot/tars_v0
"""

from __future__ import annotations

import argparse
import logging

from sx_phot.bulk_products import DEFAULT_STARS_PER_BUCKET, gather_by_star


def main() -> None:
    """Gather per-image rows into per-star buckets."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stars-per-bucket", type=int,
                        default=DEFAULT_STARS_PER_BUCKET)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    gather_by_star(args.run_dir, stars_per_bucket=args.stars_per_bucket)


if __name__ == "__main__":
    main()
