#!/usr/bin/env python
"""Write the one-row-per-star TARS table 4 catalog.

Collapses the per-sector table 4 feather (Gaia columns repeated on every
row) to one row per TICID with ``sx_phot.tars_catalog.star_table``,
joins the rotation columns of table 2, checks the star set against the
de-duplicated list used for the bulk photometry and writes a parquet
that ``sx_phot.specml.load_catalog`` reads like table 2.

Usage:
    python drivers/make_tars4_catalog.py \
        --out /ar0/local/TARS/zenodo_19917941/tars_table_4_stars.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.ipc as ipc

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sx_phot.tars_catalog import TABLE4_COLUMNS, star_table  # noqa: E402

log = logging.getLogger("make_tars4_catalog")
TARS_DIR = Path("/ar0/local/TARS")
DEFAULT_T4 = TARS_DIR / "zenodo_19917941" / "tars_table_4.feather"
DEFAULT_T2 = TARS_DIR / "tars_table_2.feather"
DEFAULT_UNIQUE = (TARS_DIR / "zenodo_19917941"
                  / "tars_table_4_unique_stars.parquet")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--table4", type=Path, default=DEFAULT_T4)
    p.add_argument("--table2", type=Path, default=DEFAULT_T2)
    p.add_argument("--unique", type=Path, default=DEFAULT_UNIQUE,
                   help="de-duplicated star list to check against")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    with pa.memory_map(str(args.table4)) as src:
        avail = ipc.open_file(src).schema.names
    cols = [c for c in TABLE4_COLUMNS if c in avail]
    t4 = pf.read_table(args.table4, columns=cols, memory_map=True
                       ).to_pandas(self_destruct=True)
    log.info("table 4: %d sector rows, %d columns (%.0f s)", len(t4),
             len(cols), time.time() - t0)
    t2 = None
    if args.table2 and args.table2.exists():
        t2 = pd.read_feather(args.table2)
        log.info("table 2: %d stars", len(t2))
    out = star_table(t4, t2)
    del t4
    log.info("%d stars (%.0f s)", len(out), time.time() - t0)
    if args.unique and args.unique.exists():
        uniq = pd.read_parquet(args.unique, columns=["TICID"])["TICID"]
        same = np.array_equal(np.sort(uniq.to_numpy()),
                              out["TICID"].to_numpy())
        log.info("star set matches %s: %s (%d vs %d)", args.unique.name,
                 same, len(uniq), len(out))
        if not same:
            log.warning("star sets differ; the bulk run catalog is the "
                        "reference for row order, this table is joined "
                        "by TICID")
    for c in ("parallax", "BpmRp0", "phot_g_mean_mag_0", "teff",
              "extinction_a0", "ruwe"):
        if c in out:
            log.info("  %-18s finite %.4f", c,
                     np.isfinite(out[c].to_numpy(dtype=float)).mean())
    log.info("  in table 2: %d", int(out["in_table2"].sum()))
    if t2 is not None and "final_n_contams" in t2:
        shared = out["in_table2"].to_numpy()
        t2c = t2.drop_duplicates("TICID").set_index("TICID")[
            "final_n_contams"].reindex(out.loc[shared, "TICID"]).to_numpy(
            dtype=float)
        t4c = out.loc[shared, "final_n_contams"].to_numpy(dtype=float)
        ok = np.isfinite(t2c) & np.isfinite(t4c)
        log.info("  table 2 final_n_contams == sector min num_contams "
                 "for %.4f of %d shared stars", (t2c[ok] == t4c[ok]).mean(),
                 int(ok.sum()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".parquet.tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(args.out)
    log.info("wrote %s: %d rows, %d columns (%.0f s)", args.out, len(out),
             out.shape[1], time.time() - t0)


if __name__ == "__main__":
    main()
