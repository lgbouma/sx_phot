#!/home/luke/miniconda3/envs/py311_sx/bin/python
"""Build the local SPHEREx image index (ObsCore snapshot + local store).

Usage:
    python drivers/build_image_index.py                  # fetch + build
    python drivers/build_image_index.py --obscore-csv X  # reuse snapshot
    python drivers/build_image_index.py --no-fetch       # headers only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sx_phot.image_index import (
    DEFAULT_INDEX_DIR,
    DEFAULT_STATE_DIR,
    DEFAULT_STORE,
    build_image_index,
)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--store-root", default=str(DEFAULT_STORE))
    parser.add_argument("--obscore-csv", default=None,
                        help="Reuse an existing ObsCore snapshot CSV.")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Do not download a snapshot; headers only.")
    parser.add_argument(
        "--out", default=str(DEFAULT_INDEX_DIR / "image_index.parquet"),
    )
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--nproc", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Build and write the image index."""
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    build_image_index(
        store_root=Path(args.store_root),
        obscore_csv=Path(args.obscore_csv) if args.obscore_csv else None,
        out_parquet=Path(args.out),
        state_dir=Path(args.state_dir),
        nproc=args.nproc,
        fetch=not args.no_fetch,
    )


if __name__ == "__main__":
    main()
