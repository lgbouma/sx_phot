#!/usr/bin/env python
"""UMAP embeddings of the binned SPHEREx spectra.

Two embeddings are written to ``<ml-dir>``:

* ``umap_shape.npy``: the dereddened, normalized log spectra (missing
  bins imputed from the fitted empirical template), reduced to
  ``--n-pc`` principal components, then UMAP.  Stars line up along the
  temperature sequence; binaries, disks and giants sit off it.
* ``umap_resid.npy``: the noise-whitened residuals from the single
  empirical template (``(F - s T(t)) / sigma``), same reduction.  Single
  stars collapse into one cloud and departures spread out.

Both are (n_stars, 2) float32 arrays aligned with ``binned_ticid.npy``
(NaN for stars that were not embedded), plus ``pca_<name>.npz`` with
the PCA basis and scores and ``umap.json`` with the settings.

Usage:
    NUMBA_NUM_THREADS=32 python drivers/run_umap.py \
        --viz-dir /ar0/SPHEREX/bulkphot/tars_v0/viz
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sx_phot import empfit, specml  # noqa: E402

log = logging.getLogger("run_umap")
DEFAULT_TARS = Path("/ar0/local/TARS/tars_table_2.feather")


def pca_reduce(x: np.ndarray, n_pc: int, n_fit: int, seed: int
               ) -> tuple[np.ndarray, dict]:
    """Centre, fit a PCA on a random subset and project everything."""
    rng = np.random.default_rng(seed)
    fit_idx = rng.choice(len(x), min(n_fit, len(x)), replace=False)
    mean = x[fit_idx].mean(axis=0)
    xs = x[fit_idx] - mean
    _, sv, vt = np.linalg.svd(xs, full_matrices=False)
    var = sv ** 2 / (sv ** 2).sum()
    comps = vt[:n_pc]
    scores = np.empty((len(x), n_pc), np.float32)
    for lo in range(0, len(x), 100000):
        scores[lo:lo + 100000] = (x[lo:lo + 100000] - mean) @ comps.T
    return scores, {"mean": mean, "components": comps,
                    "explained_variance_ratio": var[:max(n_pc, 20)]}


def run_umap(scores: np.ndarray, n_neighbors: int, min_dist: float,
             seed: int, n_epochs: int | None) -> np.ndarray:
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, metric="euclidean",
                        random_state=None, n_epochs=n_epochs,
                        low_memory=False, verbose=True)
    del seed
    return reducer.fit_transform(scores).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--viz-dir", type=Path, required=True)
    p.add_argument("--ml-dir", type=Path, default=None)
    p.add_argument("--tars", type=Path, default=DEFAULT_TARS)
    p.add_argument("--which", nargs="+", default=["shape", "resid"],
                   choices=["shape", "resid"])
    p.add_argument("--n-pc", type=int, default=12)
    p.add_argument("--n-neighbors", type=int, default=30)
    p.add_argument("--min-dist", type=float, default=0.05)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--min-bins", type=int, default=40)
    p.add_argument("--min-snr", type=float, default=10.0)
    p.add_argument("--max-stars", type=int, default=None,
                   help="random subsample (tests)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    viz = args.viz_dir
    ml = args.ml_dir or (viz.parent / "ml")
    t0 = time.time()
    cat = specml.load_catalog(viz, args.tars)
    lam = specml.bin_centers(viz)
    lib = specml.SpectralSequence.load(ml / "sequence")
    em = empfit.EmpiricalModels(lib)
    fits = pd.read_parquet(ml / "empfit.parquet",
                           columns=["ticid", "t_single", "scale_single",
                                    "n_bins_used"])
    assert np.array_equal(fits["ticid"].to_numpy(), cat["ticid"].to_numpy())
    flux = np.load(viz / "binned_flux.npy", mmap_mode="r")
    err = np.load(viz / "binned_err.npy", mmap_mode="r")
    nper = np.load(viz / "binned_n.npy", mmap_mode="r")
    a0 = cat["a0"].to_numpy(dtype=float)
    logf, _, valid = specml.prepare_log_flux(flux, err, nper, lam, a0=a0,
                                             err_floor=0.0)
    t = fits["t_single"].to_numpy(dtype=float)
    scale = fits["scale_single"].to_numpy(dtype=float)
    use = (np.isfinite(t) & (fits["n_bins_used"].to_numpy() >= args.min_bins)
           & (cat["med_snr"].to_numpy(dtype=float) >= args.min_snr))
    idx = np.flatnonzero(use)
    if args.max_stars and len(idx) > args.max_stars:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(idx, args.max_stars, replace=False))
    log.info("embedding %d of %d stars (%.0f s)", len(idx), len(cat),
             time.time() - t0)
    n = len(cat)
    settings = vars(args).copy()
    settings = {k: (str(v) if isinstance(v, Path) else v)
                for k, v in settings.items()}
    settings["n_embedded"] = int(len(idx))
    meta = {"settings": settings, "embeddings": {}}

    # Template model for the embedded stars, in chunks.
    model = np.empty((len(idx), len(lam)), np.float32)
    sig = np.empty((len(idx), len(lam)), np.float32)
    for lo in range(0, len(idx), 50000):
        sl = idx[lo:lo + 50000]
        m = scale[sl][:, None] * em.norm_flux_at(t[sl])
        model[lo:lo + 50000] = m
        f = np.asarray(flux[sl], dtype=np.float64) * specml.deredden_factor(
            lam, a0[sl])
        e = np.asarray(err[sl], dtype=np.float64) * specml.deredden_factor(
            lam, a0[sl])
        sig[lo:lo + 50000] = empfit.effective_sigma(
            f, e, em.scatter_lin_at(t[sl]))
    if "shape" in args.which:
        y = logf[idx].astype(np.float32)
        v = valid[idx]
        with np.errstate(all="ignore"):
            y = np.where(v, y, np.log10(np.maximum(model, 1e-30)))
        y = y - y.mean(axis=1, keepdims=True)
        scores, pca = pca_reduce(y, args.n_pc, 200000, args.seed)
        log.info("shape PCA: explained %s", np.round(
            pca["explained_variance_ratio"][:args.n_pc], 4).tolist())
        np.savez(ml / "pca_shape.npz", rows=idx, scores=scores, **pca)
        emb = run_umap(scores, args.n_neighbors, args.min_dist, args.seed,
                       args.n_epochs)
        full = np.full((n, 2), np.nan, np.float32)
        full[idx] = emb
        np.save(ml / "umap_shape.npy", full)
        meta["embeddings"]["shape"] = {"file": "umap_shape.npy",
                                       "seconds": time.time() - t0}
        log.info("shape embedding done (%.0f s)", time.time() - t0)
    if "resid" in args.which:
        f = np.asarray(flux[idx], dtype=np.float32) * specml.deredden_factor(
            lam, a0[idx]).astype(np.float32)
        z = np.where(valid[idx], (f - model) / sig, 0.0)
        z = np.clip(z, -10, 10).astype(np.float32)
        scores, pca = pca_reduce(z, args.n_pc, 200000, args.seed)
        log.info("resid PCA: explained %s", np.round(
            pca["explained_variance_ratio"][:args.n_pc], 4).tolist())
        np.savez(ml / "pca_resid.npz", rows=idx, scores=scores, **pca)
        emb = run_umap(scores, args.n_neighbors, args.min_dist, args.seed,
                       args.n_epochs)
        full = np.full((n, 2), np.nan, np.float32)
        full[idx] = emb
        np.save(ml / "umap_resid.npy", full)
        meta["embeddings"]["resid"] = {"file": "umap_resid.npy",
                                       "seconds": time.time() - t0}
        log.info("resid embedding done (%.0f s)", time.time() - t0)
    prev = {}
    if (ml / "umap.json").exists():
        prev = json.loads((ml / "umap.json").read_text())
    prev.setdefault("embeddings", {}).update(meta["embeddings"])
    prev["settings"] = settings
    prev["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (ml / "umap.json").write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
    main()
