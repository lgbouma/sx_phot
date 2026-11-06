#!/usr/bin/env python
"""QA figures for an empirical spectral sequence.

Writes to ``--out-dir``:

* ``seq_param_vs_catalog.png``: training-set ``t`` against BP-RP0 and
  TARS Teff (2-D histograms) with the node medians.
* ``seq_templates.png``: the templates as an image (t vs wavelength) and
  a few template curves.
* ``seq_nodes.png``: per-node Teff, M_G0, scatter and arc length.
* ``seq_vs_models.png``: ratio of the templates to SPHINX / ck04 models
  at the mapped Teff (the calibration systematics the templates absorb).
* ``seq_resid_comps.png``: the first residual eigenvector at a few nodes.

Usage:
    python drivers/plot_sequence_qa.py --ml-dir /ar0/SPHEREX/bulkphot/tars_v0/ml
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

from sx_phot.specml import SpectralSequence, load_catalog  # noqa: E402

DEFAULT_TARS = Path("/ar0/local/TARS/tars_table_2.feather")


def param_vs_catalog(lib, cat, tr, out):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    t = tr["t"].to_numpy()
    for ax, col, lab, rng in ((axs[0], "bprp0", "BP-RP0", (0.2, 3.6)),
                              (axs[1], "teff_tars", "TARS Teff [K]",
                               (2800, 8000)),
                              (axs[2], "mg0", "M_G0", (1, 13))):
        v = cat[col].to_numpy(dtype=float)
        ok = np.isfinite(v)
        ax.hist2d(t[ok], v[ok], bins=[96, 120], range=[(0, 1), rng],
                  norm=LogNorm(), cmap="Greys")
        ax.plot(lib.nodes, lib.summary[f"{col}_med"], "r-", lw=1)
        ax.plot(lib.nodes, lib.summary[f"{col}_p16"], "r:", lw=0.7)
        ax.plot(lib.nodes, lib.summary[f"{col}_p84"], "r:", lw=0.7)
        ax.set_xlabel("t")
        ax.set_ylabel(lab)
    axs[1].plot(lib.nodes, lib.teff_mono, "b--", lw=0.8, label="monotone")
    axs[1].legend(fontsize=7)
    fig.suptitle(f"training set: {len(tr)} stars, {lib.n_nodes} nodes")
    fig.tight_layout()
    fig.savefig(out / "seq_param_vs_catalog.png", dpi=180)
    plt.close(fig)


def templates_figure(lib, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4),
                            gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    im = ax.pcolormesh(lib.lam_um, lib.nodes, lib.templates, cmap="RdBu_r",
                       vmin=-0.6, vmax=0.6, shading="nearest")
    ax.set_xscale("log")
    ax.set_xlabel("wavelength [um]")
    ax.set_ylabel("t")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("normalized log10 F_nu")
    ax = axs[1]
    picks = np.linspace(0, lib.n_nodes - 1, 9).astype(int)
    cmap = plt.get_cmap("coolwarm_r")
    for i, k in enumerate(picks):
        teff = lib.summary["teff_tars_med"].iloc[k]
        ax.plot(lib.lam_um, lib.templates[k] - 0.15 * i,
                color=cmap(i / (len(picks) - 1)), lw=1,
                label=f"t={lib.nodes[k]:.2f}, {teff:.0f} K")
    ax.set_xscale("log")
    ax.set_xlabel("wavelength [um]")
    ax.set_ylabel("normalized log10 F_nu (offset)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "seq_templates.png", dpi=180)
    plt.close(fig)


def nodes_figure(lib, out):
    s = lib.summary
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    ax = axs[0, 0]
    ax.plot(lib.nodes, s["teff_tars_med"], "k-", label="TARS Teff median")
    ax.fill_between(lib.nodes, s["teff_tars_p16"], s["teff_tars_p84"],
                    color="0.8", label="16-84%")
    ax.plot(lib.nodes, lib.teff_mono, "r--", lw=0.8, label="monotone")
    ax.set_ylabel("Teff [K]")
    ax.legend(fontsize=7)
    ax = axs[0, 1]
    ax.plot(lib.nodes, s["mg0_med"], "k-")
    ax.fill_between(lib.nodes, s["mg0_p16"], s["mg0_p84"], color="0.8")
    ax.invert_yaxis()
    ax.set_ylabel("M_G0")
    ax = axs[1, 0]
    for j, lam in enumerate((0.9, 1.6, 2.5, 3.5, 4.7)):
        i = int(np.argmin(np.abs(lib.lam_um - lam)))
        ax.plot(lib.nodes, 100 * np.log(10) * lib.scatter_int[:, i],
                label=f"{lam} um")
    ax.set_ylabel("intrinsic scatter [%]")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax = axs[1, 1]
    ax.plot(lib.nodes, lib.arc, "k-", label="arc length")
    ax.plot(lib.nodes, s["counts"] / s["counts"].max(), "b:",
            label="window count (scaled)")
    ax.set_ylabel("scaled")
    ax.legend(fontsize=7)
    for ax in axs.ravel():
        ax.set_xlabel("t")
    fig.tight_layout()
    fig.savefig(out / "seq_nodes.png", dpi=180)
    plt.close(fig)


def vs_models_figure(lib, out):
    try:
        from sx_phot.sedfit import build_model_set
        models = build_model_set()
    except Exception as exc:  # noqa: BLE001
        print(f"model grids unavailable ({exc}); skipping seq_vs_models")
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4),
                            gridspec_kw={"width_ratios": [1.2, 1]})
    ratio = np.full_like(lib.templates, np.nan, dtype=float)
    mteff = np.asarray(models.teff, dtype=float)
    for k in range(lib.n_nodes):
        teff = lib.teff_mono[k]
        j = int(np.argmin(np.abs(mteff - teff)))
        m = np.log10(models.flux[j])
        m = m - np.nanmean(m)
        ratio[k] = lib.templates[k] - m
    ax = axs[0]
    im = ax.pcolormesh(lib.lam_um, lib.nodes, ratio, cmap="PuOr_r",
                       vmin=-0.1, vmax=0.1, shading="nearest")
    ax.set_xscale("log")
    ax.set_xlabel("wavelength [um]")
    ax.set_ylabel("t")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("log10 (template / model at mapped Teff)")
    ax = axs[1]
    picks = np.linspace(0, lib.n_nodes - 1, 7).astype(int)
    cmap = plt.get_cmap("coolwarm_r")
    for i, k in enumerate(picks):
        ax.plot(lib.lam_um, 100 * (10 ** ratio[k] - 1),
                color=cmap(i / (len(picks) - 1)), lw=1,
                label=f"{lib.teff_mono[k]:.0f} K")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_ylim(-25, 25)
    ax.set_xlabel("wavelength [um]")
    ax.set_ylabel("template / model - 1 [%]")
    ax.legend(fontsize=6, ncol=2)
    fig.suptitle("SPHINX (<= 4000 K) and ck04 (>= 4250 K), dwarf logg, "
                 "[M/H] = 0; both normalized to mean log flux 0")
    fig.tight_layout()
    fig.savefig(out / "seq_vs_models.png", dpi=180)
    plt.close(fig)


def resid_comps_figure(lib, out):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    picks = np.linspace(0, lib.n_nodes - 1, 6).astype(int)
    cmap = plt.get_cmap("coolwarm_r")
    for i, k in enumerate(picks):
        col = cmap(i / (len(picks) - 1))
        axs[0].plot(lib.lam_um, lib.resid_comps[k, 0] * lib.resid_amp[k, 0],
                    color=col, lw=1, label=f"{lib.teff_mono[k]:.0f} K")
        axs[1].plot(lib.lam_um, lib.resid_comps[k, 1] * lib.resid_amp[k, 1],
                    color=col, lw=1)
    for ax, j in zip(axs, (1, 2)):
        ax.set_xscale("log")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("wavelength [um]")
        ax.set_ylabel(f"residual component {j} x rms amplitude [dex]")
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "seq_resid_comps.png", dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ml-dir", type=Path, required=True)
    p.add_argument("--viz-dir", type=Path, default=None)
    p.add_argument("--tars", type=Path, default=DEFAULT_TARS)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default <ml-dir>/qa")
    args = p.parse_args()
    ml = args.ml_dir
    out = args.out_dir or (ml / "qa")
    out.mkdir(parents=True, exist_ok=True)
    lib = SpectralSequence.load(ml / "sequence")
    viz = args.viz_dir or Path(lib.meta.get("viz_dir", ml.parent / "viz"))
    tr = pd.read_parquet(ml / "sequence_train.parquet")
    cat = load_catalog(viz, args.tars).iloc[tr["row"].to_numpy()]
    cat = cat.reset_index(drop=True)
    param_vs_catalog(lib, cat, tr, out)
    templates_figure(lib, out)
    nodes_figure(lib, out)
    vs_models_figure(lib, out)
    resid_comps_figure(lib, out)
    print(f"wrote QA figures to {out}")


if __name__ == "__main__":
    main()
