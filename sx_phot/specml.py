"""Empirical spectral sequence from the binned SPHEREx spectra.

The R = 40 binned spectra of a bulk run (``viz/binned_*.npy``, see
``sx_phot.viz_products``) are dereddened with the catalog A0, turned
into normalized log fluxes (the mean over the 76 bins removed, so the
distance and radius drop out) and ordered along one parameter ``t`` in
[0, 1] by a principal-curve iteration:

1. start from the rank of a colour (Gaia BP-RP0);
2. node templates are the per-bin medians of the stars near each of
   ``n_nodes`` equally spaced values of ``t``;
3. every star is reassigned to the ``t`` whose (linearly interpolated)
   template fits it best with a free scale;
4. ``t`` is rank-transformed so that the nodes stay equally populated;
   repeat.

Medians make the templates robust to the minority of binaries, disks
and bad spectra at every ``t``.  The final library
(:class:`SpectralSequence`) carries, per node, the normalized template,
its population scatter, the absolute flux offset at 10 pc (from the
Gaia parallaxes), the first residual eigenvectors (a second, empirical
parameter), and catalog summaries (median TARS Teff, BP-RP0, M_G0, ...)
so that ``t`` can be read as a temperature or mass.  ``t`` = 0 is the
blue end.

``sx_phot.empfit`` fits the library to every star.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

LN10 = np.log(10.0)
N_BINS = 76
DEFAULT_ERR_FLOOR = 0.02  # fractional, added in quadrature


# ---------------------------------------------------------------------------
# Extinction
# ---------------------------------------------------------------------------

def extinction_curve(lam_um: np.ndarray, rv: float = 3.1) -> np.ndarray:
    """A_lambda / A_V for the Cardelli, Clayton & Mathis (1989) law.

    The infrared power law (0.3 <= 1/lambda <= 1.1 um^-1) is extended
    beyond 3.3 um; at A0 < 1 the correction there is below 1 %.

    Args:
        lam_um: Wavelengths in um.
        rv: Ratio of total to selective extinction.

    Returns:
        A_lambda / A_V at each wavelength.
    """
    x = 1.0 / np.asarray(lam_um, dtype=float)
    a = np.empty_like(x)
    b = np.empty_like(x)
    ir = x <= 1.1
    a[ir] = 0.574 * x[ir] ** 1.61
    b[ir] = -0.527 * x[ir] ** 1.61
    opt = ~ir
    y = x[opt] - 1.82
    a[opt] = (1 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3
              + 0.72085 * y ** 4 + 0.01979 * y ** 5 - 0.77530 * y ** 6
              + 0.32999 * y ** 7)
    b[opt] = (1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3
              - 5.38434 * y ** 4 - 0.62251 * y ** 5 + 5.30260 * y ** 6
              - 2.09002 * y ** 7)
    return a + b / rv


def deredden_factor(lam_um: np.ndarray, a0: np.ndarray) -> np.ndarray:
    """Multiplicative flux correction 10^(0.4 A_lambda) for A0 = A_V.

    Args:
        lam_um: (n_bins,) wavelengths.
        a0: (n_stars,) extinction parameter; NaN or negative -> 0.

    Returns:
        (n_stars, n_bins) factors.
    """
    a0 = np.nan_to_num(np.asarray(a0, dtype=float), nan=0.0)
    a0 = np.clip(a0, 0.0, None)
    k = extinction_curve(lam_um)
    return 10.0 ** (0.4 * a0[:, None] * k[None, :])


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

CATALOG_COLUMNS = {
    "TICID": "ticid", "teff": "teff_tars", "BpmRp0": "bprp0",
    "phot_g_mean_mag_0": "mg0", "parallax": "plx", "extinction_a0": "a0",
    "ruwe": "ruwe", "non_single_star": "nss", "Tmag": "tmag",
    "adopted_period": "period", "median_amplitude": "amplitude",
    "final_n_contams": "n_contams", "phot_g_mean_mag": "gmag",
}


def load_catalog(viz_dir: Path, tars_path: Path) -> pd.DataFrame:
    """Catalog and summary columns aligned with ``binned_ticid.npy``.

    Args:
        viz_dir: Viz build directory.
        tars_path: TARS table (feather or parquet) with ``TICID``.

    Returns:
        DataFrame with one row per binned spectrum (same order).
    """
    viz_dir = Path(viz_dir)
    tic = np.load(viz_dir / "binned_ticid.npy")
    tars_path = Path(tars_path)
    cols = list(CATALOG_COLUMNS)
    if tars_path.suffix == ".feather":
        tars = pd.read_feather(tars_path)
    else:
        tars = pd.read_parquet(tars_path)
    tars = tars[[c for c in cols if c in tars.columns]].rename(
        columns=CATALOG_COLUMNS)
    tars = tars.drop_duplicates("ticid").set_index("ticid")
    cat = tars.reindex(tic).reset_index()
    cat["ticid"] = tic
    summ = pd.read_parquet(
        viz_dir / "summary.parquet",
        columns=["ticid", "med_snr", "spline_ok", "n_used", "resid_mad"])
    summ = summ.set_index("ticid").reindex(tic).reset_index(drop=True)
    for c in summ.columns:
        cat[c] = summ[c].to_numpy()
    cat["spline_ok"] = cat["spline_ok"].fillna(False).astype(bool)
    return cat


def bin_centers(viz_dir: Path) -> np.ndarray:
    """Geometric bin centres (um) of a viz build."""
    edges = np.load(Path(viz_dir) / "bin_edges.npy")
    return np.sqrt(edges[:-1] * edges[1:])


def prepare_log_flux(
    flux: np.ndarray,
    err: np.ndarray,
    nper: np.ndarray,
    lam_um: np.ndarray,
    a0: Optional[np.ndarray] = None,
    err_floor: float = DEFAULT_ERR_FLOOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dereddened log10 flux, its error in dex and the validity mask.

    Args:
        flux: (n, n_bins) binned flux (Jy).
        err: (n, n_bins) binned errors (Jy).
        nper: (n, n_bins) number of points per bin.
        lam_um: (n_bins,) bin centres.
        a0: (n,) extinction to remove, or None.
        err_floor: Fractional floor added in quadrature.

    Returns:
        (log_flux, sigma_log, valid); invalid entries are NaN / 0 /
        False.
    """
    flux = np.asarray(flux, dtype=np.float32)
    err = np.asarray(err, dtype=np.float32)
    valid = (np.asarray(nper) > 0) & np.isfinite(flux) & (flux > 0)
    valid &= np.isfinite(err)
    if a0 is not None:
        fac = deredden_factor(lam_um, a0).astype(np.float32)
    else:
        fac = np.ones((1, flux.shape[1]), np.float32)
    with np.errstate(all="ignore"):
        logf = np.where(valid, np.log10(flux * fac), np.nan)
        frac = np.where(valid, err / flux, np.inf)
        sig = np.sqrt(frac ** 2 + err_floor ** 2) / LN10
    sig = np.where(valid, sig, 0.0).astype(np.float32)
    return logf.astype(np.float32), sig, valid


def normalize_rows(logf: np.ndarray, valid: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Subtract each row's mean log flux over its valid bins.

    Returns:
        (y, offset) with ``y`` NaN where invalid and ``offset`` the
        removed mean (NaN for rows with no valid bin).
    """
    with np.errstate(all="ignore"):
        s = np.where(valid, logf, 0.0).sum(axis=1)
        n = valid.sum(axis=1)
        offset = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    y = np.where(valid, logf - offset[:, None], np.nan)
    return y.astype(np.float32), offset.astype(np.float32)


# ---------------------------------------------------------------------------
# Principal-curve iteration
# ---------------------------------------------------------------------------

def rank_uniform(values: np.ndarray) -> np.ndarray:
    """Map values to their rank in (0, 1); ties broken by order."""
    order = np.argsort(values, kind="stable")
    t = np.empty(len(values), dtype=np.float64)
    t[order] = (np.arange(len(values)) + 0.5) / len(values)
    return t


def node_positions(n_nodes: int) -> np.ndarray:
    """Node parameters, equally spaced in (0, 1)."""
    return (np.arange(n_nodes) + 0.5) / n_nodes


def uniform_windows(n_nodes: int, half_width: Optional[float] = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes equally spaced in rank with windows of ``+-half_width``.

    Returns:
        (nodes, lo, hi) arrays of length ``n_nodes``.
    """
    if half_width is None:
        half_width = 1.0 / n_nodes
    nodes = node_positions(n_nodes)
    return nodes, nodes - half_width, nodes + half_width


def arc_windows(
    nodes: np.ndarray,
    templates: np.ndarray,
    n_nodes: int,
    n_grid: int = 2048,
    overlap: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes equally spaced in arc length along the current templates.

    The first and last nodes sit at ``t`` = 0 and 1 (one-sided windows)
    so that the ends of the sequence, where the spectra change fastest
    per star, are resolved rather than averaged into the interior.

    Args:
        nodes: Current node parameters.
        templates: (n_nodes_old, n_bins) current templates.
        n_nodes: Number of new nodes.
        n_grid: Fine grid used to measure the arc length.
        overlap: Window half-width in units of the node spacing.

    Returns:
        (nodes, lo, hi): new node parameters and window bounds in ``t``.
    """
    # Arc length of the templates smoothed along t on a uniform grid:
    # chords between noisy node templates would otherwise count noise
    # as arc and make dense node regions denser.
    tg = np.linspace(0.0, 1.0, 257)
    templ = interpolate_nodes(nodes, templates, tg)
    templ = np.where(np.isfinite(templ), templ, 0.0)
    kernel = np.ones(5) / 5.0
    pad = np.pad(templ, ((2, 2), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(pad[:, j], kernel, mode="valid")
                       for j in range(templ.shape[1])], axis=1)
    step = np.sqrt((np.diff(smooth, axis=0) ** 2).sum(axis=1))
    arc = np.concatenate([[0.0], np.cumsum(step)])
    arc = arc / arc[-1] if arc[-1] > 0 else tg.copy()
    arc = arc + 1e-6 * tg  # strictly increasing through flat ends
    arc = arc / arc[-1]
    del n_grid
    s_nodes = np.linspace(0.0, 1.0, n_nodes)
    hw = overlap / (n_nodes - 1)
    t_nodes = np.interp(s_nodes, arc, tg)
    t_lo = np.interp(np.clip(s_nodes - hw, 0, 1), arc, tg)
    t_hi = np.interp(np.clip(s_nodes + hw, 0, 1), arc, tg)
    return t_nodes, t_lo, t_hi


def node_medians(
    y: np.ndarray,
    t: np.ndarray,
    windows: tuple,
    min_members: int = 100,
    linear: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin median and MAD scatter of the stars in each node window.

    With ``linear`` the median is taken of ``y - slope (t - t_node)``,
    the slope being a per-bin least-squares fit inside the window, so
    that one-sided or lopsided windows (the ends of the sequence) do not
    bias the template toward the window centre.

    Args:
        y: (n, n_bins) normalized log flux (NaN allowed).
        t: (n,) parameter in [0, 1].
        windows: (nodes, lo, hi) from :func:`uniform_windows` or
            :func:`arc_windows`.
        min_members: Windows with fewer stars are widened symmetrically
            (in ``t``) until they hold this many.
        linear: Apply the local slope correction.

    Returns:
        (templates, scatter, counts): (n_nodes, n_bins) medians, 1.4826
        MAD per bin, and the number of stars per window.
    """
    nodes, lo_t, hi_t = windows
    n_nodes = len(nodes)
    order = np.argsort(t, kind="stable")
    ts = t[order]
    ys = y[order]
    templ = np.full((n_nodes, y.shape[1]), np.nan, np.float32)
    scat = np.full((n_nodes, y.shape[1]), np.nan, np.float32)
    counts = np.zeros(n_nodes, np.int64)
    n = len(ts)
    for k in range(n_nodes):
        a, b = float(lo_t[k]), float(hi_t[k])
        lo = int(np.searchsorted(ts, a))
        hi = int(np.searchsorted(ts, b))
        while hi - lo < min(min_members, n) and (lo > 0 or hi < n):
            pad = 0.25 * max(b - a, 1e-4)
            a -= pad
            b += pad
            lo = int(np.searchsorted(ts, a))
            hi = int(np.searchsorted(ts, b))
        block = ys[lo:hi]
        if block.size == 0:
            continue
        if linear and hi - lo >= 10:
            dt = (ts[lo:hi] - nodes[k])[:, None]
            var = float((dt ** 2).mean() - dt.mean() ** 2)
            if var > 0:
                fin = np.isfinite(block)
                b0 = np.where(fin, block, 0.0)
                cnt = np.maximum(fin.sum(axis=0), 1)
                ym = b0.sum(axis=0) / cnt
                tm = (dt * fin).sum(axis=0) / cnt
                cov = ((dt - tm) * (b0 - ym) * fin).sum(axis=0) / cnt
                tvar = (((dt - tm) ** 2) * fin).sum(axis=0) / cnt
                slope = np.where(tvar > 0, cov / np.where(tvar > 0, tvar, 1),
                                 0.0)
                block = block - slope[None, :] * dt
        if np.isnan(block).any():
            med = np.nanmedian(block, axis=0)
            mad = np.nanmedian(np.abs(block - med), axis=0)
        else:
            med = np.median(block, axis=0)
            mad = np.median(np.abs(block - med), axis=0)
        templ[k] = med
        scat[k] = 1.4826 * mad
        counts[k] = hi - lo
    return templ, scat, counts


def interpolate_nodes(nodes: np.ndarray, values: np.ndarray,
                      t: np.ndarray) -> np.ndarray:
    """Linear interpolation of per-node arrays at parameters ``t``.

    Args:
        nodes: (n_nodes,) increasing node parameters.
        values: (n_nodes, ...) values at the nodes.
        t: (m,) query parameters (clamped to the node range).

    Returns:
        (m, ...) interpolated values.
    """
    t = np.clip(np.asarray(t, dtype=float), nodes[0], nodes[-1])
    idx = np.clip(np.searchsorted(nodes, t) - 1, 0, len(nodes) - 2)
    frac = (t - nodes[idx]) / (nodes[idx + 1] - nodes[idx])
    shape = (len(t),) + (1,) * (values.ndim - 1)
    frac = frac.reshape(shape)
    return (1 - frac) * values[idx] + frac * values[idx + 1]


def fine_grid(nodes: np.ndarray, templates: np.ndarray,
              n_grid: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Templates on a fine grid over [0, 1] (constant beyond the end nodes)."""
    tg = np.linspace(0.0, 1.0, n_grid)
    return tg, interpolate_nodes(nodes, templates, tg)


def assign_log(
    y: np.ndarray,
    w: np.ndarray,
    t_grid: np.ndarray,
    templates_grid: np.ndarray,
    chunk: int = 4000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best grid parameter per star for a log-space fit with free offset.

    Minimizes ``sum_i w_i (y_i - c - T_i(t))^2`` over the offset ``c``
    (closed form) and the grid of ``t``.  Invalid bins must have
    ``w = 0`` (and finite ``y``).  The minimum is refined by a parabola
    through the three grid points around it.

    Args:
        y: (n, n_bins) normalized log flux, NaN replaced by 0.
        w: (n, n_bins) weights (1 / sigma^2 in dex, 0 where invalid).
        t_grid: (n_grid,) parameters.
        templates_grid: (n_grid, n_bins) templates.
        chunk: Stars per matrix product.

    Returns:
        (t_best, chi2_min, offset) per star.
    """
    n = y.shape[0]
    tg = np.asarray(templates_grid, dtype=np.float64)
    tg_t = tg.T
    tg2_t = (tg ** 2).T
    t_best = np.empty(n)
    chi2_min = np.empty(n)
    offset = np.empty(n)
    for s in range(0, n, chunk):
        yy = np.asarray(y[s:s + chunk], dtype=np.float64)
        ww = np.asarray(w[s:s + chunk], dtype=np.float64)
        yy = np.where(ww > 0, yy, 0.0)
        wy = ww * yy
        sw = ww.sum(axis=1)
        sy = wy.sum(axis=1)
        syy = (wy * yy).sum(axis=1)
        a = wy @ tg_t
        b = ww @ tg2_t
        c = ww @ tg_t
        sw_safe = np.where(sw > 0, sw, 1.0)
        chi2 = syy[:, None] - 2 * a + b - (sy[:, None] - c) ** 2 / sw_safe[:, None]
        k = np.argmin(chi2, axis=1)
        rows = np.arange(len(k))
        # Parabolic refinement.
        km = np.clip(k - 1, 0, len(t_grid) - 1)
        kp = np.clip(k + 1, 0, len(t_grid) - 1)
        c0 = chi2[rows, k]
        cm = chi2[rows, km]
        cp = chi2[rows, kp]
        denom = cm - 2 * c0 + cp
        shift = np.where((denom > 0) & (k > 0) & (k < len(t_grid) - 1),
                         0.5 * (cm - cp) / np.where(denom > 0, denom, 1.0),
                         0.0)
        shift = np.clip(shift, -1.0, 1.0)
        dt = t_grid[1] - t_grid[0]
        t_best[s:s + chunk] = t_grid[k] + shift * dt
        chi2_min[s:s + chunk] = np.where(sw > 0, c0, np.nan)
        offset[s:s + chunk] = (sy - c[rows, k]) / sw_safe
    return t_best, chi2_min, offset


@dataclass
class SequenceFit:
    """Result of :func:`build_sequence`."""

    nodes: np.ndarray
    templates: np.ndarray
    scatter: np.ndarray
    counts: np.ndarray
    t: np.ndarray
    chi2: np.ndarray
    inlier: np.ndarray
    node_lo: np.ndarray
    node_hi: np.ndarray
    history: list = field(default_factory=list)

    @property
    def windows(self) -> tuple:
        return self.nodes, self.node_lo, self.node_hi


def build_sequence(
    y: np.ndarray,
    sigma: np.ndarray,
    t_init: np.ndarray,
    n_nodes: int = 192,
    n_iter: int = 6,
    n_grid: int = 2048,
    half_width: Optional[float] = None,
    clip_factor: float = 4.0,
    arc_nodes: bool = True,
    min_members: int = 100,
    verbose: bool = True,
) -> SequenceFit:
    """Principal-curve iteration on normalized log spectra.

    Iteration 0 uses nodes equally spaced in rank; afterwards the nodes
    are respaced uniformly in arc length along the current templates
    (``arc_nodes``), with the end nodes at ``t`` = 0 and 1.  From the
    second iteration on, stars whose chi2 exceeds ``clip_factor`` times
    the running median chi2 at their ``t`` are dropped from the template
    medians and from the rank transform (blends, binaries, bad spectra);
    they still receive a ``t`` from the final assignment.

    Args:
        y: (n, n_bins) normalized log flux (NaN where invalid).
        sigma: (n, n_bins) errors in dex (0 where invalid).
        t_init: (n,) initial ordering variable (e.g. BP-RP0).
        n_nodes: Number of template nodes.
        n_iter: Iterations (each: templates, reassignment, re-ranking).
        n_grid: Fine grid points for the reassignment.
        half_width: Window half-width in ``t`` for iteration 0.
        clip_factor: chi2 clipping threshold relative to the local
            median; None disables clipping.
        arc_nodes: Respace the nodes in arc length after iteration 0.
        min_members: Minimum stars per node window.
        verbose: Log per-iteration diagnostics.

    Returns:
        :class:`SequenceFit` with the final nodes and windows, templates,
        scatter, per-star ``t`` (rank-uniform among inliers; outliers are
        placed by interpolation of the inlier ranks), chi2 and the
        inlier mask.
    """
    valid = np.isfinite(y) & (sigma > 0)
    w = np.where(valid, 1.0 / np.maximum(sigma, 1e-6) ** 2, 0.0)
    y0 = np.where(valid, y, 0.0)
    n = len(y)
    inlier = np.ones(n, dtype=bool)
    t = rank_uniform(np.asarray(t_init, dtype=float))
    history = []
    windows = uniform_windows(n_nodes, half_width)

    def _rank_with_outliers(t_raw):
        # Inliers get uniform ranks; outliers the rank of the nearest
        # inlier value (so they never shape the node spacing).
        r = np.empty(n)
        r[inlier] = rank_uniform(t_raw[inlier])
        order = np.argsort(t_raw[inlier])
        r[~inlier] = np.interp(t_raw[~inlier], t_raw[inlier][order],
                               r[inlier][order])
        return r

    def _local_clip(chi2, t_cur):
        ref, _, _ = node_medians(np.where(np.isfinite(chi2), chi2, np.nan
                                          )[inlier][:, None], t_cur[inlier],
                                 windows, min_members)
        ref_star = np.interp(t_cur, windows[0], ref[:, 0])
        return np.isfinite(chi2) & (chi2 <= clip_factor * ref_star)

    for it in range(n_iter):
        t0 = time.time()
        templates, scatter, counts = node_medians(
            y[inlier], t[inlier], windows, min_members)
        nodes = windows[0]
        t_grid, tg = fine_grid(nodes, templates, n_grid)
        t_new, chi2, _ = assign_log(y0, w, t_grid, tg)
        if clip_factor is not None and it >= 1:
            inlier = _local_clip(chi2, t)
        rank_new = _rank_with_outliers(t_new)
        moved = np.abs(rank_new - t)[inlier]
        rho = np.corrcoef(rank_new[inlier], t[inlier])[0, 1]
        rec = {"iter": it, "spearman": float(rho),
               "median_move": float(np.median(moved)),
               "p90_move": float(np.percentile(moved, 90)),
               "median_chi2": float(np.nanmedian(chi2[inlier])),
               "n_inlier": int(inlier.sum()),
               "min_count": int(counts.min()),
               "seconds": time.time() - t0}
        history.append(rec)
        if verbose:
            log.info("iter %d: rank corr %.5f, median |dt| %.4f, p90 %.4f,"
                     " median chi2 %.1f, inliers %d, min window %d"
                     " (%.1f s)", it, rho, rec["median_move"],
                     rec["p90_move"], rec["median_chi2"], rec["n_inlier"],
                     rec["min_count"], rec["seconds"])
        t = rank_new
        if arc_nodes:
            windows = arc_windows(nodes, templates, n_nodes, n_grid)
    # Final templates at the converged ordering.
    templates, scatter, counts = node_medians(y[inlier], t[inlier],
                                              windows, min_members)
    nodes = windows[0]
    t_grid, tg = fine_grid(nodes, templates, n_grid)
    t_fit, chi2, _ = assign_log(y0, w, t_grid, tg)
    if clip_factor is not None:
        inlier = _local_clip(chi2, t)
    return SequenceFit(nodes=nodes, templates=templates, scatter=scatter,
                       counts=counts, t=_rank_with_outliers(t_fit),
                       chi2=chi2, inlier=inlier, node_lo=windows[1],
                       node_hi=windows[2], history=history)


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

def residual_components(
    resid: np.ndarray,
    t: np.ndarray,
    windows: tuple,
    n_comp: int = 3,
    clip_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First residual eigenvectors per node (the 'second parameter').

    Args:
        resid: (n, n_bins) residuals from the interpolated template,
            NaN where invalid (rows with NaN are skipped).
        t: (n,) parameters.
        windows: (nodes, lo, hi) node windows.
        n_comp: Components to keep.
        clip_sigma: Rows whose rms residual exceeds this many times the
            median rms are dropped before the SVD.

    Returns:
        (components, amp_std, frac_var): (n_nodes, n_comp, n_bins) unit
        vectors (sign: positive at the reddest bin), the standard
        deviation of the projections, and the explained variance
        fractions.
    """
    nodes, lo_t, hi_t = windows
    n_nodes = len(nodes)
    nb = resid.shape[1]
    comps = np.zeros((n_nodes, n_comp, nb), np.float32)
    amp = np.zeros((n_nodes, n_comp), np.float32)
    fvar = np.zeros((n_nodes, n_comp), np.float32)
    ok = np.isfinite(resid).all(axis=1)
    order = np.argsort(t[ok], kind="stable")
    ts = t[ok][order]
    rs = resid[ok][order]
    for k in range(n_nodes):
        lo = int(np.searchsorted(ts, lo_t[k]))
        hi = int(np.searchsorted(ts, hi_t[k]))
        block = np.asarray(rs[lo:hi], dtype=np.float64)
        if len(block) < 3 * n_comp:
            continue
        rms = np.sqrt((block ** 2).mean(axis=1))
        keep = rms <= clip_sigma * np.median(rms)
        block = block[keep]
        block = block - block.mean(axis=0)
        try:
            _, sv, vt = np.linalg.svd(block, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        for j in range(min(n_comp, len(sv))):
            v = vt[j]
            if v[-1] < 0:
                v = -v
            comps[k, j] = v
            amp[k, j] = sv[j] / np.sqrt(len(block))
            fvar[k, j] = sv[j] ** 2 / (sv ** 2).sum()
    return comps, amp, fvar


def node_summary(
    cat: pd.DataFrame,
    t: np.ndarray,
    windows: tuple,
    columns: tuple = ("teff_tars", "bprp0", "mg0", "a0", "tmag", "period",
                      "amplitude", "plx"),
) -> pd.DataFrame:
    """Median and 16/84 percentiles of catalog columns per node."""
    nodes, lo_t, hi_t = windows
    order = np.argsort(t, kind="stable")
    ts = t[order]
    rows = []
    for k, tk in enumerate(nodes):
        lo = int(np.searchsorted(ts, lo_t[k]))
        hi = int(np.searchsorted(ts, hi_t[k]))
        sel = order[lo:hi]
        rec = {"node": k, "t": tk, "n": hi - lo}
        for c in columns:
            if c not in cat:
                continue
            v = cat[c].to_numpy(dtype=float)[sel]
            v = v[np.isfinite(v)]
            if len(v) == 0:
                rec[f"{c}_med"] = np.nan
                rec[f"{c}_p16"] = np.nan
                rec[f"{c}_p84"] = np.nan
                continue
            p16, med, p84 = np.percentile(v, [16, 50, 84])
            rec[f"{c}_med"] = med
            rec[f"{c}_p16"] = p16
            rec[f"{c}_p84"] = p84
        rows.append(rec)
    return pd.DataFrame(rows)


def monotone_decreasing(values: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators fit of a non-increasing sequence."""
    v = np.asarray(values, dtype=float).copy()
    n = len(v)
    # Isotonic regression (non-increasing) via PAV on -v.
    x = -v
    w = np.ones(n)
    blocks = [[i, i, x[i], 1.0] for i in range(n)]  # lo, hi, mean, weight
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][2] > out[-1][2]:
            b2 = out.pop()
            b1 = out.pop()
            wt = b1[3] + b2[3]
            out.append([b1[0], b2[1], (b1[2] * b1[3] + b2[2] * b2[3]) / wt,
                        wt])
    res = np.empty(n)
    for lo, hi, m, _ in out:
        res[lo:hi + 1] = -m
    del w
    return res


def arc_length(templates: np.ndarray, scatter: Optional[np.ndarray] = None
               ) -> np.ndarray:
    """Cumulative chord length along the node templates, scaled to [0, 1].

    Args:
        templates: (n_nodes, n_bins).
        scatter: Optional (n_nodes, n_bins) scatter; if given, steps are
            measured in units of the local scatter (a chi-like metric).
    """
    d = np.diff(templates, axis=0)
    if scatter is not None:
        s = 0.5 * (scatter[1:] + scatter[:-1])
        d = d / np.maximum(s, 1e-3)
    step = np.sqrt(np.nansum(d ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(step)])
    return s / s[-1] if s[-1] > 0 else s


class SpectralSequence:
    """Empirical one-parameter template library.

    Arrays (``n_nodes`` = K, ``n_bins`` = 76):

    * ``nodes`` (K): parameter ``t`` of each node.
    * ``lam_um`` (76): bin centres.
    * ``templates`` (K, 76): median normalized log10 flux (mean 0).
    * ``scatter`` (K, 76): 1.4826 MAD of the normalized log flux (dex).
    * ``scatter_int`` (K, 76): scatter with the median measurement
      error removed in quadrature (intrinsic + systematic).
    * ``abs_offset`` (K): median of the per-star mean log10 flux (Jy) at
      10 pc; ``templates + abs_offset[:, None]`` is the absolute
      template.  ``abs_scatter`` (K) is its MAD (dex).
    * ``resid_comps`` (K, n_comp, 76), ``resid_amp`` (K, n_comp): first
      residual eigenvectors and the rms of their amplitudes.
    * ``arc`` (K): arc length along the sequence, scaled to [0, 1].
    * ``summary``: DataFrame of catalog medians per node (``teff_med``
      etc.); ``teff_mono`` is a monotone version of the TARS Teff.
    """

    def __init__(self, arrays: dict, summary: pd.DataFrame,
                 meta: Optional[dict] = None) -> None:
        self.nodes = np.asarray(arrays["nodes"], dtype=float)
        self.lam_um = np.asarray(arrays["lam_um"], dtype=float)
        self.templates = np.asarray(arrays["templates"], dtype=np.float32)
        self.scatter = np.asarray(arrays["scatter"], dtype=np.float32)
        self.scatter_int = np.asarray(arrays.get("scatter_int",
                                                 self.scatter),
                                      dtype=np.float32)
        self.abs_offset = np.asarray(arrays["abs_offset"], dtype=float)
        self.abs_scatter = np.asarray(arrays["abs_scatter"], dtype=float)
        self.resid_comps = np.asarray(arrays["resid_comps"],
                                      dtype=np.float32)
        self.resid_amp = np.asarray(arrays["resid_amp"], dtype=np.float32)
        self.arc = np.asarray(arrays["arc"], dtype=float)
        self.summary = summary.reset_index(drop=True)
        self.meta = dict(meta or {})
        teff = self.summary["teff_tars_med"].to_numpy(dtype=float)
        self.teff_mono = monotone_decreasing(np.where(np.isfinite(teff),
                                                      teff, np.nan)
                                             if np.isfinite(teff).all()
                                             else pd.Series(teff).interpolate(
                                                 limit_direction="both"
                                             ).to_numpy())

    # -- persistence ------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write ``<path>.npz`` and ``<path>_nodes.csv``."""
        path = Path(path)
        np.savez_compressed(
            path.with_suffix(".npz"), nodes=self.nodes, lam_um=self.lam_um,
            templates=self.templates, scatter=self.scatter,
            scatter_int=self.scatter_int, abs_offset=self.abs_offset,
            abs_scatter=self.abs_scatter, resid_comps=self.resid_comps,
            resid_amp=self.resid_amp, arc=self.arc,
            meta=json.dumps(self.meta))
        self.summary.to_csv(path.with_name(path.stem + "_nodes.csv"),
                            index=False)

    @classmethod
    def load(cls, path: Path) -> "SpectralSequence":
        """Read a library written by :meth:`save`."""
        path = Path(path)
        npz = np.load(path.with_suffix(".npz"), allow_pickle=False)
        arrays = {k: npz[k] for k in npz.files if k != "meta"}
        meta = json.loads(str(npz["meta"])) if "meta" in npz.files else {}
        summary = pd.read_csv(path.with_name(path.stem + "_nodes.csv"))
        return cls(arrays, summary, meta)

    # -- interpolation ----------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def template(self, t) -> np.ndarray:
        """Normalized log10 template at ``t`` (scalar or array)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return interpolate_nodes(self.nodes, self.templates, t)

    def scatter_at(self, t, intrinsic: bool = True) -> np.ndarray:
        """Population scatter (dex) at ``t``."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        src = self.scatter_int if intrinsic else self.scatter
        return interpolate_nodes(self.nodes, src, t)

    def abs_template(self, t) -> np.ndarray:
        """log10 flux (Jy) at 10 pc of the median star at ``t``."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        off = interpolate_nodes(self.nodes, self.abs_offset, t)
        return self.template(t) + off[:, None]

    def teff(self, t) -> np.ndarray:
        """Monotone TARS Teff mapped from ``t``."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return interpolate_nodes(self.nodes, self.teff_mono, t)

    def t_from_teff(self, teff) -> np.ndarray:
        """Inverse of :meth:`teff` (Teff decreases with ``t``)."""
        teff = np.atleast_1d(np.asarray(teff, dtype=float))
        return np.interp(-teff, -self.teff_mono, self.nodes)

    def t_from_arc(self, s) -> np.ndarray:
        """Parameter at a scaled arc length ``s`` in [0, 1]."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        return np.interp(s, self.arc, self.nodes)

    def arc_at(self, t) -> np.ndarray:
        """Scaled arc length at ``t``."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return np.interp(t, self.nodes, self.arc)

    def column(self, name: str, t) -> np.ndarray:
        """Interpolate a node-summary column (e.g. ``mg0_med``) at ``t``."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        v = self.summary[name].to_numpy(dtype=float)
        v = pd.Series(v).interpolate(limit_direction="both").to_numpy()
        return interpolate_nodes(self.nodes, v, t)

    def grid(self, n_grid: int = 2048) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray]:
        """Fine grid of (t, templates, intrinsic scatter)."""
        tg, templ = fine_grid(self.nodes, self.templates, n_grid)
        scat = interpolate_nodes(self.nodes, self.scatter_int, tg)
        return tg, templ, scat


def assemble_library(
    fit: SequenceFit,
    y: np.ndarray,
    sigma: np.ndarray,
    log_flux_offset: np.ndarray,
    cat: pd.DataFrame,
    lam_um: np.ndarray,
    n_resid_comp: int = 3,
    meta: Optional[dict] = None,
    sigma_meas: Optional[np.ndarray] = None,
) -> SpectralSequence:
    """Build the library from a converged sequence fit of the clean set.

    Args:
        fit: Output of :func:`build_sequence`.
        y: (n, n_bins) normalized log flux of the same stars.
        sigma: (n, n_bins) errors in dex (with the floor used in the fit).
        log_flux_offset: (n,) per-star mean log10 flux (Jy) removed by
            :func:`normalize_rows`.
        cat: Catalog rows of the same stars (needs ``plx`` in mas).
        lam_um: (n_bins,) bin centres.
        n_resid_comp: Residual eigenvectors per node.
        meta: Provenance dict stored with the library.
        sigma_meas: (n, n_bins) floor-free measurement errors in dex,
            removed in quadrature from the scatter to give
            ``scatter_int`` (default: ``sigma``).
    """
    windows = fit.windows
    keep = fit.inlier
    t = fit.t[keep]
    y = y[keep]
    sigma = sigma[keep]
    log_flux_offset = log_flux_offset[keep]
    cat = cat.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    # Intrinsic scatter: remove the median measurement error per node.
    if sigma_meas is None:
        sigma_meas = sigma
    else:
        sigma_meas = sigma_meas[keep]
    med_sig, _, _ = node_medians(np.where(sigma_meas > 0, sigma_meas,
                                          np.nan), t, windows)
    scatter_int = np.sqrt(np.clip(fit.scatter.astype(float) ** 2
                                  - med_sig.astype(float) ** 2, 0, None))
    scatter_int = np.maximum(scatter_int, 1e-3).astype(np.float32)
    # Absolute offsets at 10 pc.
    plx = cat["plx"].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        abs_mean = log_flux_offset + 4.0 - 2.0 * np.log10(plx)
    abs_mean = np.where(np.isfinite(abs_mean) & (plx > 0), abs_mean, np.nan)
    abs_off, abs_sc, _ = node_medians(abs_mean[:, None], t, windows)
    # Residuals from the interpolated template (free offset in log space).
    templ_star = interpolate_nodes(fit.nodes, fit.templates, t)
    resid = y - templ_star
    valid = np.isfinite(resid)
    off = np.where(valid, resid, 0).sum(1) / np.maximum(valid.sum(1), 1)
    resid = resid - off[:, None]
    comps, amp, fvar = residual_components(resid, t, windows, n_resid_comp)
    summary = node_summary(cat, t, windows)
    summary["counts"] = fit.counts
    for j in range(n_resid_comp):
        summary[f"resid_var_frac_{j + 1}"] = fvar[:, j]
    summary["t_lo"] = fit.node_lo
    summary["t_hi"] = fit.node_hi
    arrays = {
        "nodes": fit.nodes, "lam_um": lam_um, "templates": fit.templates,
        "scatter": fit.scatter, "scatter_int": scatter_int,
        "abs_offset": abs_off[:, 0], "abs_scatter": abs_sc[:, 0],
        "resid_comps": comps, "resid_amp": amp,
        "arc": arc_length(fit.templates),
    }
    lib = SpectralSequence(arrays, summary, meta)
    lib.summary["teff_mono"] = lib.teff_mono
    lib.summary["arc"] = lib.arc
    return lib


# ---------------------------------------------------------------------------
# Clean-sample selection
# ---------------------------------------------------------------------------

def main_sequence_offset(bprp0: np.ndarray, mg0: np.ndarray,
                         n_bins: int = 60) -> np.ndarray:
    """M_G0 minus the running median main sequence at each BP-RP0.

    Positive values are fainter than the sequence.  NaN where the
    inputs are missing or outside the populated colour range.
    """
    bprp0 = np.asarray(bprp0, dtype=float)
    mg0 = np.asarray(mg0, dtype=float)
    ok = np.isfinite(bprp0) & np.isfinite(mg0)
    if ok.sum() < 100:
        return np.full(len(bprp0), np.nan)
    edges = np.quantile(bprp0[ok], np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    idx = np.clip(np.searchsorted(edges, bprp0[ok]) - 1, 0, n_bins - 1)
    med = np.array([np.median(mg0[ok][idx == k]) if (idx == k).any()
                    else np.nan for k in range(n_bins)])
    centers = 0.5 * (edges[1:] + edges[:-1])
    good = np.isfinite(med)
    # Two passes: refit the median after clipping to +-1 mag.
    ms = np.interp(bprp0[ok], centers[good], med[good])
    close = np.abs(mg0[ok] - ms) < 1.0
    med2 = np.array([np.median(mg0[ok][(idx == k) & close])
                     if ((idx == k) & close).any() else np.nan
                     for k in range(n_bins)])
    good = np.isfinite(med2)
    ms = np.interp(bprp0[ok], centers[good], med2[good])
    out = np.full(len(bprp0), np.nan)
    out[ok] = mg0[ok] - ms
    return out


def select_clean(
    cat: pd.DataFrame,
    valid: np.ndarray,
    min_snr: float = 50.0,
    max_a0: float = 0.2,
    max_ruwe: float = 1.4,
    ms_window: tuple = (-0.4, 0.5),
    min_n_used: int = 150,
    max_contams: Optional[int] = 0,
) -> np.ndarray:
    """Boolean mask of stars used to build the sequence.

    Requirements: all bins valid, median S/N, spline ok, low extinction,
    RUWE below the limit, no Gaia non-single-star flag, parallax, at
    most ``max_contams`` TESS contaminants (None: no cut; blended
    neighbours dominate the SPHEREx aperture for faint stars), and
    within ``ms_window`` magnitudes of the dereddened main sequence.
    """
    n_valid = valid.sum(axis=1)
    ok = n_valid == valid.shape[1]
    ok &= cat["med_snr"].to_numpy(dtype=float) >= min_snr
    ok &= cat["spline_ok"].to_numpy(dtype=bool)
    ok &= cat["n_used"].to_numpy(dtype=float) >= min_n_used
    a0 = cat["a0"].to_numpy(dtype=float)
    ok &= np.isfinite(a0) & (a0 <= max_a0)
    ruwe = cat["ruwe"].to_numpy(dtype=float)
    ok &= ~(ruwe > max_ruwe)
    nss = cat["nss"].to_numpy(dtype=float)
    ok &= ~(nss > 0)
    plx = cat["plx"].to_numpy(dtype=float)
    ok &= np.isfinite(plx) & (plx > 0)
    if max_contams is not None and "n_contams" in cat:
        nc = cat["n_contams"].to_numpy(dtype=float)
        ok &= ~(nc > max_contams)
    ok &= np.isfinite(cat["bprp0"].to_numpy(dtype=float))
    dms = main_sequence_offset(cat["bprp0"].to_numpy(dtype=float),
                               cat["mg0"].to_numpy(dtype=float))
    ok &= (dms >= ms_window[0]) & (dms <= ms_window[1])
    return ok
