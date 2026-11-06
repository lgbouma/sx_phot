#!/usr/bin/env python
"""Overview figures of the UMAP embeddings.

For each embedding in ``<ml-dir>/umap_*.npy`` writes
``<out-dir>/umap_<name>.png``: a density panel and panels coloured by
the median of a few columns per pixel (Teff, CMD offset, binary and disk
evidence, chi2, contaminants, S/N).

Usage:
    python drivers/plot_umap_qa.py --run-dir /ar0/SPHEREX/bulkphot/tars_v0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from aesthetic.plot import set_style

set_style("science")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sx_phot.specml import load_catalog, main_sequence_offset  # noqa: E402

DEFAULT_TARS = Path("/ar0/local/TARS/tars_table_2.feather")
PANELS = [
    ("teff_emp", "Teff_emp [K]", "turbo_r", (3000, 7500)),
    ("dms", "M_G0 - main sequence [mag]", "RdBu", (-1.5, 1.5)),
    ("dmag_ms", "SPHEREx overluminosity [mag]", "RdBu", (-1.5, 1.5)),
    ("dbic_binary_ms_pct", "binary (Gaia dist.) percentile", "viridis",
     (0, 1)),
    ("dbic_disk_pct", "disk percentile", "viridis", (0, 1)),
    ("red_excess", "red excess", "RdBu_r", (-0.1, 0.1)),
    ("chi2_red", "chi2 / dof", "magma", (0.5, 30)),
    ("n_contams", "TESS contaminants", "magma", (0, 10)),
    ("med_snr", "median S/N", "magma", (10, 300)),
    ("log_period", "log10 P_rot [d]", "viridis", (-0.7, 1.5)),
    ("ruwe", "RUWE", "magma", (0.8, 3)),
]


def binned_median(x, y, v, bins, extent):
    """Median of ``v`` per pixel (NaN where empty)."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    xi = np.clip(((x[ok] - extent[0]) / (extent[1] - extent[0]) * bins
                  ).astype(int), 0, bins - 1)
    yi = np.clip(((y[ok] - extent[2]) / (extent[3] - extent[2]) * bins
                  ).astype(int), 0, bins - 1)
    key = yi * bins + xi
    order = np.argsort(key, kind="stable")
    key = key[order]
    vals = v[ok][order]
    img = np.full(bins * bins, np.nan)
    edges = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [len(key)]])
    for s, e in zip(starts, stops):
        img[key[s]] = np.median(vals[s:e])
    return img.reshape(bins, bins)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", type=Path,
                   default=Path("/ar0/SPHEREX/bulkphot/tars_v0"))
    p.add_argument("--tars", type=Path, default=DEFAULT_TARS)
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "results" / "sx_ml_analysis")
    p.add_argument("--bins", type=int, default=400)
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    viz = args.run_dir / "viz"
    ml = args.run_dir / "ml"
    cat = load_catalog(viz, args.tars)
    emp = pd.read_parquet(ml / "empfit.parquet",
                          columns=["teff_emp", "dmag_ms", "dbic_binary_ms_pct",
                                   "dbic_disk_pct", "red_excess", "chi2_single",
                                   "dof_single"])
    df = pd.concat([cat, emp], axis=1)
    df["dms"] = main_sequence_offset(df["bprp0"].to_numpy(float),
                                     df["mg0"].to_numpy(float))
    with np.errstate(all="ignore"):
        df["chi2_red"] = df["chi2_single"] / df["dof_single"]
        df["log_period"] = np.log10(df["period"].to_numpy(float))
    for path in sorted(ml.glob("umap_*.npy")):
        name = path.stem[len("umap_"):]
        xy = np.load(path)
        x, y = xy[:, 0], xy[:, 1]
        ok = np.isfinite(x)
        lo = np.percentile(xy[ok], 0.2, axis=0)
        hi = np.percentile(xy[ok], 99.8, axis=0)
        extent = (lo[0], hi[0], lo[1], hi[1])
        panels = [(c, l, cm, r) for c, l, cm, r in PANELS if c in df]
        ncol = 4
        nrow = int(np.ceil((len(panels) + 1) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow))
        axs = axs.ravel()
        ax = axs[0]
        h = ax.hist2d(x[ok], y[ok], bins=args.bins,
                      range=[extent[:2], extent[2:]], norm=LogNorm(),
                      cmap="Greys")
        fig.colorbar(h[3], ax=ax, label="stars per pixel")
        ax.set_title(f"{name}: {ok.sum():,} stars", fontsize=9)
        for ax, (col, label, cmap, rng) in zip(axs[1:], panels):
            v = df[col].to_numpy(dtype=float)
            img = binned_median(x, y, v, args.bins, extent)
            kw = {}
            if col in ("chi2_red", "med_snr", "ruwe"):
                kw["norm"] = LogNorm(vmin=rng[0], vmax=rng[1])
            else:
                kw["vmin"], kw["vmax"] = rng
            im = ax.imshow(img, origin="lower", extent=extent, cmap=cmap,
                           aspect="auto", interpolation="nearest", **kw)
            fig.colorbar(im, ax=ax, label=label)
            ax.set_title(f"median {col} per pixel", fontsize=9)
        for ax in axs[len(panels) + 1:]:
            ax.axis("off")
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out / f"umap_{name}.png", dpi=150)
        plt.close(fig)
        print(f"wrote {out / f'umap_{name}.png'}")


if __name__ == "__main__":
    main()
