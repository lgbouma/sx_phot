#!/home/luke/miniconda3/envs/py311_sx/bin/python
"""Write legacy per-star CSV products from a gathered bulk run.

Usage:
    python drivers/materialize_bulk_star.py \
        --run-dir /ar0/SPHEREX/bulkphot/tars_v0 --ticid 300651846 \
        --out-dir results/bulk_tars/TIC_300651846
"""

from __future__ import annotations

import argparse
import logging

from sx_phot.bulk_products import materialize_star
from sx_phot.tic_motion import normalize_tic_id


def main() -> None:
    """Materialize one or more stars."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ticid", required=True, nargs="+")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--no-supplement", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for tic in args.ticid:
        ticid = normalize_tic_id(tic)
        paths = materialize_star(
            args.run_dir, ticid, args.out_dir,
            supplement=not args.no_supplement,
        )
        for name, path in paths.items():
            logging.info("TIC %d %s -> %s", ticid, name, path)


if __name__ == "__main__":
    main()
