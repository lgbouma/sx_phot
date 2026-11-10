#!/home/luke/miniconda3/envs/py311_sx/bin/python
"""Build the local slim Gaia DR3 copy (sx_phot.gaia_slim).

Usage:
    python drivers/fetch_gaia_slim.py --out-dir /ar0/GAIA/gaia_dr3_slim \
        --log-file /ar0/GAIA/gaia_dr3_slim/fetch.log

Walks gaiadr3.gaia_source at the Heidelberg ARI TAP mirror in
source_id order, 10M rows per job, one job at a time; resumable.
Run it under the systemd-run memory cap (see scripts/README.txt).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sx_phot.gaia_slim import ARI_TAP, DEFAULT_MAX_ROWS, fetch_all


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default="/ar0/GAIA/gaia_dr3_slim")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--max-parts", type=int, default=None,
                        help="stop after this many new parts (testing)")
    parser.add_argument("--tap-url", default=ARI_TAP)
    parser.add_argument("--pause-s", type=float, default=5.0)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> None:
    """Run the resumable fetch."""
    args = _parse_args()
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="[%(asctime)s] %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    man = fetch_all(Path(args.out_dir), max_rows=args.max_rows,
                    tap_url=args.tap_url, pause_s=args.pause_s,
                    max_parts=args.max_parts)
    logging.getLogger(__name__).info(
        "%d parts, %d rows so far", len(man), int(man["n_rows"].sum()))


if __name__ == "__main__":
    main()
