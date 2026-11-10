#!/home/luke/miniconda3/envs/py311_sx/bin/python
"""Repair the IRSA PSF zone-header error across the local L2 mirror.

Usage:
    python drivers/fix_psf_headers.py --dry-run --limit 1000
    python drivers/fix_psf_headers.py --nproc 4 \
        --log-file /ar0/SPHEREX/spherex_l2/fix_psf_headers.log

Candidates are the local files of the image index whose version
directory matches --ver-regex (default: l2b-v20..v23, pipeline 6.4 to
6.5.5); each file's VERSION keyword is checked again before writing,
so the run is idempotent (see sx_phot.psf_header_fix).  Do not run
while photometry jobs read the mirror.
"""

from __future__ import annotations

import argparse
import collections
import logging
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from sx_phot.image_index import DEFAULT_INDEX_DIR
from sx_phot.psf_header_fix import fix_file

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--index",
                        default=str(DEFAULT_INDEX_DIR / "image_index.parquet"))
    parser.add_argument("--ver-regex", default=r"^l2b-v2[0-3]-")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def _one(args: tuple) -> tuple:
    path, dry_run = args
    try:
        return path, fix_file(Path(path), dry_run=dry_run), None
    except Exception as exc:  # noqa: BLE001 - report, keep going
        return path, "error", repr(exc)


def main() -> None:
    """Select candidate files and repair them in parallel."""
    args = _parse_args()
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="[%(asctime)s] %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    index = pd.read_parquet(args.index, columns=["path", "ver"])
    sel = index["path"].notna() & index["ver"].str.match(args.ver_regex)
    paths = sorted(index.loc[sel, "path"].tolist())
    if args.limit:
        paths = paths[:args.limit]
    log.info("%d candidate files (%s)%s", len(paths), args.ver_regex,
             " [dry run]" if args.dry_run else "")
    counts: collections.Counter = collections.Counter()
    t0 = time.time()
    with Pool(max(1, args.nproc)) as pool:
        jobs = ((p, args.dry_run) for p in paths)
        for i, (path, status, err) in enumerate(
                pool.imap_unordered(_one, jobs, chunksize=16), 1):
            counts[status] += 1
            if err:
                log.error("%s: %s", path, err)
            if i % 5000 == 0 or i == len(paths):
                log.info("%d/%d files, %.0f s: %s", i, len(paths),
                         time.time() - t0, dict(counts))
    log.info("Done: %s", dict(counts))


if __name__ == "__main__":
    main()
