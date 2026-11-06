"""Fit the empirical spectral sequence to binned SPHEREx spectra.

All models are evaluated in linear flux (Jy) on the R = 40 bins, after
dereddening with the catalog A0.  With ``T(t)`` the normalized log10
template of :class:`sx_phot.specml.SpectralSequence`, ``A(t)`` the
absolute template (log10 Jy at 10 pc) and ``c = (plx / 100 mas)^2`` the
Gaia distance factor:

| key | model | free parameters |
|---|---|---|
| ``single`` | ``s 10^T(t)`` | t, s |
| ``ms`` | ``c 10^A(t)`` | t |
| ``binary`` | ``s [10^A(t1) + 10^A(t2)]``, t2 >= t1 | t1, t2, s |
| ``binary_ms`` | ``c [10^A(t1) + 10^A(t2)]`` | t1, t2 |
| ``disk`` | ``s 10^T(t) + f B_nu(T_bb)`` | t, s, T_bb, f |

``binary`` tests the spectral shape alone (the flux ratio of the two
stars follows from the main-sequence luminosities); ``binary_ms`` adds
the Gaia parallax, so an overluminous star counts as evidence too.

Errors are the measurement errors plus the population scatter of the
sequence at the star's ``t`` (``scatter_int``), so chi2 compares a star
with the single-star population rather than with a model atmosphere:
shared calibration features (band edges, the 4.4-5 um band) are part of
the templates and do not count as excess.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from sx_phot.sedfit import MainSequence
from sx_phot.specml import (LN10, SpectralSequence, deredden_factor,
                            interpolate_nodes)

log = logging.getLogger(__name__)

H_PLANCK = 6.62607015e-34
K_BOLTZ = 1.380649e-23
C_LIGHT = 299792458.0
RSUN_M = 6.957e8
PC_M = 3.0857e16
DISK_TEMPS_K = np.exp(np.linspace(np.log(200.0), np.log(2000.0), 16))
N_PARAMS = {"single": 2, "ms": 1, "binary": 3, "binary_ms": 2, "disk": 4}
DEFAULT_ERR_FLOOR = 0.005
BAND_EDGES_UM = {
    "b1": (0.75, 1.1075), "b2": (1.1075, 1.6435), "b3": (1.6435, 2.421),
    "b4": (2.423, 3.8155), "b5": (3.8155, 4.416), "b6": (4.416, 5.002),
}
RED_MIN_UM = 2.4
BLUE_MAX_UM = 1.0


def planck_jy_sr(lam_um: np.ndarray, teff: float) -> np.ndarray:
    """B_nu(T) in Jy / sr at wavelengths in um."""
    nu = C_LIGHT / (np.asarray(lam_um, dtype=float) * 1e-6)
    x = H_PLANCK * nu / (K_BOLTZ * float(teff))
    with np.errstate(over="ignore"):
        b = 2 * H_PLANCK * nu ** 3 / C_LIGHT ** 2 / np.expm1(x)
    return b / 1e-26


class EmpiricalModels:
    """Precomputed template grids for the fits.

    Args:
        lib: The spectral sequence.
        n_grid: Fine parameter grid for the single-star searches.
        disk_temps: Blackbody temperatures (K).
        main_sequence: Teff-radius relation for radii and ``f_disk``.
    """

    def __init__(self, lib: SpectralSequence, n_grid: int = 1536,
                 disk_temps: np.ndarray = DISK_TEMPS_K,
                 main_sequence: Optional[MainSequence] = None) -> None:
        self.lib = lib
        self.lam = lib.lam_um
        self.t_grid, templ, scat = lib.grid(n_grid)
        self.lin_grid = 10.0 ** templ.astype(np.float64)     # (G, 76)
        self.scat_grid = scat.astype(np.float64)
        self.abs_off_grid = interpolate_nodes(lib.nodes, lib.abs_offset,
                                              self.t_grid)
        self.abs_grid = self.lin_grid * 10.0 ** self.abs_off_grid[:, None]
        # Pair grid on the nodes: index pairs (i1, i2) with i2 >= i1.
        nodes = lib.nodes
        self.node_abs = 10.0 ** (lib.templates.astype(np.float64)
                                 + lib.abs_offset[:, None])
        i1, i2 = np.triu_indices(len(nodes))
        self.pair_i1 = i1
        self.pair_i2 = i2
        self.pair_flux = self.node_abs[i1] + self.node_abs[i2]  # (P, 76)
        self.disk_temps = np.asarray(disk_temps, dtype=float)
        self.bb = np.array([planck_jy_sr(self.lam, t)
                            for t in self.disk_temps])  # (B, 76)
        self.ms = main_sequence or MainSequence()
        self.band_masks = {b: (self.lam >= lo) & (self.lam < hi)
                           for b, (lo, hi) in BAND_EDGES_UM.items()}
        self.red_mask = self.lam >= RED_MIN_UM
        self.blue_mask = self.lam < BLUE_MAX_UM

    def node_spacing(self, t: np.ndarray) -> np.ndarray:
        """Local node spacing in ``t`` (nodes may be non-uniform)."""
        nodes = self.lib.nodes
        d = np.diff(nodes)
        mid = 0.5 * (nodes[1:] + nodes[:-1])
        return np.interp(np.asarray(t, dtype=float), mid, d)

    # -- template evaluation ---------------------------------------------

    def abs_flux_at(self, t: np.ndarray) -> np.ndarray:
        """Absolute template flux (Jy at 10 pc) at arbitrary ``t``."""
        a = self.lib.abs_template(np.ravel(t))
        return (10.0 ** a).reshape(np.shape(t) + (len(self.lam),))

    def norm_flux_at(self, t: np.ndarray) -> np.ndarray:
        """Normalized template flux (mean log 0) at arbitrary ``t``."""
        a = self.lib.template(np.ravel(t))
        return (10.0 ** a).reshape(np.shape(t) + (len(self.lam),))

    def scatter_lin_at(self, t: np.ndarray) -> np.ndarray:
        """Fractional population scatter at ``t``."""
        return LN10 * self.lib.scatter_at(np.ravel(t)).reshape(
            np.shape(t) + (len(self.lam),))

    def radius_ms_rsun(self, t: np.ndarray) -> np.ndarray:
        """Dwarf radius at the Teff mapped from ``t``."""
        return self.ms.radius_at(self.lib.teff(np.ravel(t))).reshape(
            np.shape(t))


# ---------------------------------------------------------------------------
# Vectorized searches
# ---------------------------------------------------------------------------

def _free_scale_search(flux: np.ndarray, w: np.ndarray, models: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Best model index with a free non-negative scale.

    Args:
        flux: (n, 76) fluxes (0 where invalid).
        w: (n, 76) weights (0 where invalid).
        models: (M, 76) model fluxes.

    Returns:
        (index, scale, chi2) per star, plus the full chi2 matrix is not
        kept; ``chi2`` is the minimum.
    """
    wf = w * flux
    num = wf @ models.T                        # (n, M)
    den = w @ (models ** 2).T                  # (n, M)
    den = np.where(den > 0, den, 1.0)
    scale = np.clip(num / den, 0.0, None)
    sff = (wf * flux).sum(axis=1)
    chi2 = sff[:, None] - 2 * scale * num + scale ** 2 * den
    k = np.argmin(chi2, axis=1)
    rows = np.arange(len(k))
    return k, scale[rows, k], chi2[rows, k], chi2


def _fixed_scale_chi2(flux: np.ndarray, w: np.ndarray, models: np.ndarray,
                      c: np.ndarray) -> np.ndarray:
    """chi2 matrix (n, M) for models scaled by a per-star constant ``c``."""
    wf = w * flux
    num = wf @ models.T
    den = w @ (models ** 2).T
    sff = (wf * flux).sum(axis=1)
    return sff[:, None] - 2 * c[:, None] * num + c[:, None] ** 2 * den


def _parabolic_refine(chi2: np.ndarray, k: np.ndarray, grid: np.ndarray
                      ) -> np.ndarray:
    """Sub-grid minimum from a parabola through three grid points."""
    rows = np.arange(len(k))
    km = np.clip(k - 1, 0, len(grid) - 1)
    kp = np.clip(k + 1, 0, len(grid) - 1)
    c0, cm, cp = chi2[rows, k], chi2[rows, km], chi2[rows, kp]
    denom = cm - 2 * c0 + cp
    inner = (k > 0) & (k < len(grid) - 1) & (denom > 0)
    shift = np.where(inner, 0.5 * (cm - cp) / np.where(denom > 0, denom, 1),
                     0.0)
    shift = np.clip(shift, -1, 1)
    return grid[k] + shift * (grid[1] - grid[0])


def _per_star_chi2(flux: np.ndarray, w: np.ndarray, models: np.ndarray,
                   c: Optional[np.ndarray] = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """chi2 for per-star model sets (n, M, 76); free scale unless ``c``.

    Returns:
        (chi2, scale) each (n, M).
    """
    wf = (w * flux)[:, None, :]
    num = (wf * models).sum(axis=2)
    den = (w[:, None, :] * models ** 2).sum(axis=2)
    sff = (w * flux ** 2).sum(axis=1)[:, None]
    if c is None:
        den_safe = np.where(den > 0, den, 1.0)
        scale = np.clip(num / den_safe, 0.0, None)
    else:
        scale = np.broadcast_to(c[:, None], num.shape)
    chi2 = sff - 2 * scale * num + scale ** 2 * den
    return chi2, scale


def _refine_pair(
    em: EmpiricalModels,
    flux: np.ndarray,
    w: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    c: Optional[np.ndarray],
    n_sub: int = 17,
    passes: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refine a pair (t1, t2) by alternating 1-D fine searches.

    Each pass searches ``t1`` within one node spacing of its current
    value with ``t2`` fixed, then ``t2`` with ``t1`` fixed (keeping
    ``t2 >= t1``).

    Returns:
        (t1, t2, scale, chi2) per star.
    """
    lo, hi = em.lib.nodes[0], em.lib.nodes[-1]
    frac = np.linspace(-1.0, 1.0, n_sub)
    scale = np.ones(len(t1))
    chi2 = np.full(len(t1), np.inf)
    for _ in range(passes):
        for which in (0, 1):
            base = t1 if which == 0 else t2
            dnode = em.node_spacing(base)
            cand = np.clip(base[:, None] + dnode[:, None] * frac[None, :],
                           lo, hi)
            if which == 0:
                cand = np.minimum(cand, t2[:, None])
                other = em.abs_flux_at(t2)[:, None, :]
                models = em.abs_flux_at(cand) + other
            else:
                cand = np.maximum(cand, t1[:, None])
                other = em.abs_flux_at(t1)[:, None, :]
                models = other + em.abs_flux_at(cand)
            c2, sc = _per_star_chi2(flux, w, models, c)
            k = np.argmin(c2, axis=1)
            rows = np.arange(len(k))
            if which == 0:
                t1 = cand[rows, k]
            else:
                t2 = cand[rows, k]
            scale = sc[rows, k]
            chi2 = c2[rows, k]
    return t1, t2, scale, chi2


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def effective_sigma(flux: np.ndarray, err: np.ndarray, frac_scatter: np.ndarray,
                    err_floor: float = DEFAULT_ERR_FLOOR) -> np.ndarray:
    """Measurement error plus fractional population scatter and floor."""
    f = np.abs(flux)
    return np.sqrt(err ** 2 + (frac_scatter * f) ** 2 + (err_floor * f) ** 2)


def fit_chunk(
    flux: np.ndarray,
    err: np.ndarray,
    nper: np.ndarray,
    plx_mas: np.ndarray,
    a0: Optional[np.ndarray],
    em: EmpiricalModels,
    err_floor: float = DEFAULT_ERR_FLOOR,
    do_binary: bool = True,
    do_disk: bool = True,
    min_bins: int = 20,
) -> pd.DataFrame:
    """Fit all models to a block of binned spectra.

    Args:
        flux: (n, 76) binned flux (Jy).
        err: (n, 76) binned errors (Jy).
        nper: (n, 76) points per bin (0 = missing).
        plx_mas: (n,) parallaxes (NaN allowed; the ``*_ms`` models and
            radii are then NaN).
        a0: (n,) extinction to remove, or None.
        em: Precomputed models.
        err_floor: Fractional floor added in quadrature.
        do_binary: Run the pair searches.
        do_disk: Run the star + blackbody search.
        min_bins: Stars with fewer valid bins get NaN results.

    Returns:
        One row per star (see the module docstring for the models);
        columns are documented in ``drivers/fit_empirical.py``.
    """
    n, nb = flux.shape
    lam = em.lam
    flux = np.asarray(flux, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)
    valid = (np.asarray(nper) > 0) & np.isfinite(flux) & (flux > 0)
    valid &= np.isfinite(err) & (err > 0)
    if a0 is not None:
        fac = deredden_factor(lam, a0)
        flux = flux * fac
        err = err * fac
    flux = np.where(valid, flux, 0.0)
    err = np.where(valid, err, 1.0)
    n_used = valid.sum(axis=1)
    usable = n_used >= min_bins
    plx = np.asarray(plx_mas, dtype=float)
    has_plx = np.isfinite(plx) & (plx > 0)
    c_dist = np.where(has_plx, (plx / 100.0) ** 2, np.nan)

    out: dict[str, np.ndarray] = {"n_bins_used": n_used}

    # Pass 1: single template with floor-only errors -> t0 for the scatter.
    w1 = np.where(valid, 1.0 / (err ** 2 + (0.02 * flux) ** 2), 0.0)
    k, _, _, _ = _free_scale_search(flux, w1, em.lin_grid)
    t0 = em.t_grid[k]
    frac_scat = em.scatter_lin_at(t0)
    sig = effective_sigma(flux, err, frac_scat, err_floor)
    w = np.where(valid, 1.0 / sig ** 2, 0.0)

    # Single (free scale).
    k, scale, chi2, chi2_all = _free_scale_search(flux, w, em.lin_grid)
    t_single = _parabolic_refine(chi2_all, k, em.t_grid)
    model_single = scale[:, None] * em.norm_flux_at(t_single)
    # Recompute scale/chi2 at the refined t.
    num = (w * flux * model_single).sum(1)
    den = (w * model_single ** 2).sum(1)
    adj = np.where(den > 0, num / np.where(den > 0, den, 1.0), 1.0)
    model_single = model_single * adj[:, None]
    scale = scale * adj
    chi2 = (w * (flux - model_single) ** 2).sum(1)
    del chi2_all
    out["t_single"] = t_single
    out["teff_emp"] = em.lib.teff(t_single)
    out["scale_single"] = scale
    out["chi2_single"] = chi2
    out["dof_single"] = n_used - N_PARAMS["single"]
    # Luminosity relative to the main sequence at the Gaia distance.
    abs_off = interpolate_nodes(em.lib.nodes, em.lib.abs_offset, t_single)
    with np.errstate(all="ignore"):
        lum_ratio = scale * 10.0 ** (-abs_off) / c_dist
    out["lum_ratio"] = lum_ratio
    with np.errstate(all="ignore"):
        out["radius_ratio_emp"] = np.sqrt(lum_ratio)
        out["dmag_ms"] = np.where(lum_ratio > 0, -2.5 * np.log10(
            np.where(lum_ratio > 0, lum_ratio, 1.0)), np.nan)
    r_ms = em.radius_ms_rsun(t_single)
    out["radius_emp"] = r_ms * np.sqrt(lum_ratio)

    # Residuals of the single fit.
    resid = np.where(valid, flux - model_single, np.nan)
    with np.errstate(all="ignore"):
        frac = resid / model_single
        z = resid / sig
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    for b, m in em.band_masks.items():
        nbm = (valid & m).sum(1)
        with np.errstate(all="ignore"):
            out[f"excess_{b}"] = np.nanmean(np.where(m, frac, np.nan), 1)
            out[f"excess_{b}_sig"] = (np.nansum(np.where(m, z, np.nan), 1)
                                      / np.sqrt(np.maximum(nbm, 1)))
        out[f"excess_{b}_sig"][nbm == 0] = np.nan
    for name, m in (("red_excess", em.red_mask), ("blue_excess",
                                                   em.blue_mask)):
        nbm = (valid & m).sum(1)
        with np.errstate(all="ignore"):
            out[name] = np.nanmean(np.where(m, frac, np.nan), 1)
            out[f"{name}_sig"] = (np.nansum(np.where(m, z, np.nan), 1)
                                  / np.sqrt(np.maximum(nbm, 1)))
        out[f"{name}_sig"][nbm == 0] = np.nan
    with np.errstate(all="ignore"):
        absz = np.abs(np.where(valid, z, 0))
        out["max_abs_z"] = absz.max(1)
        out["lam_max_abs_z"] = lam[np.argmax(absz, 1)]
        out["n_bins_4sig"] = (absz > 4).sum(1)
    # Residual eigenvector amplitudes (second parameter) in dex.
    comps = interpolate_nodes(em.lib.nodes, em.lib.resid_comps, t_single)
    with np.errstate(all="ignore"):
        rlog = np.where(valid, np.log10(np.maximum(flux, 1e-30))
                        - np.log10(np.maximum(model_single, 1e-30)), 0.0)
    rlog = rlog - rlog.sum(1, keepdims=True) / np.maximum(n_used, 1)[:, None]
    rlog = np.where(valid, rlog, 0.0)
    for j in range(comps.shape[1]):
        out[f"resid_a{j + 1}"] = (rlog * comps[:, j, :]).sum(1)

    # Main-sequence single at the Gaia distance (no free scale).
    chi2_ms = _fixed_scale_chi2(flux, w, em.abs_grid,
                                np.where(has_plx, c_dist, 0.0))
    k_ms = np.argmin(chi2_ms, axis=1)
    t_ms = _parabolic_refine(chi2_ms, k_ms, em.t_grid)
    out["t_ms"] = np.where(has_plx, t_ms, np.nan)
    out["chi2_ms"] = np.where(has_plx, chi2_ms[np.arange(n), k_ms], np.nan)
    del chi2_ms

    if do_binary:
        # Shape-only binary: two main-sequence stars, free overall scale.
        kp, sc_p, chi2_p, _ = _free_scale_search(flux, w, em.pair_flux)
        t1 = em.lib.nodes[em.pair_i1[kp]]
        t2 = em.lib.nodes[em.pair_i2[kp]]
        t1, t2, sc_p, chi2_p = _refine_pair(em, flux, w, t1, t2, None)
        out["t_bin1"] = t1
        out["t_bin2"] = t2
        out["teff_bin1"] = em.lib.teff(t1)
        out["teff_bin2"] = em.lib.teff(t2)
        out["scale_binary"] = sc_p
        out["chi2_binary"] = chi2_p
        out["dchi2_binary"] = chi2 - chi2_p
        f1 = em.abs_flux_at(t1)
        f2 = em.abs_flux_at(t2)
        out["flux_ratio_binary"] = f2.sum(1) / f1.sum(1)
        with np.errstate(all="ignore"):
            lr = sc_p * 10.0 ** 0.0 / c_dist
        out["lum_ratio_binary"] = lr  # system luminosity / MS pair
        # Binary at the Gaia distance (no free scale).
        chi2_pm = _fixed_scale_chi2(flux, w, em.pair_flux,
                                    np.where(has_plx, c_dist, 0.0))
        kpm = np.argmin(chi2_pm, axis=1)
        del chi2_pm
        t1m = em.lib.nodes[em.pair_i1[kpm]]
        t2m = em.lib.nodes[em.pair_i2[kpm]]
        t1m, t2m, _, chi2_pm_best = _refine_pair(
            em, flux, w, t1m, t2m, np.where(has_plx, c_dist, 0.0))
        out["t_bin1_ms"] = np.where(has_plx, t1m, np.nan)
        out["t_bin2_ms"] = np.where(has_plx, t2m, np.nan)
        out["teff_bin1_ms"] = np.where(has_plx, em.lib.teff(t1m), np.nan)
        out["teff_bin2_ms"] = np.where(has_plx, em.lib.teff(t2m), np.nan)
        out["chi2_binary_ms"] = np.where(has_plx, chi2_pm_best, np.nan)
        out["dchi2_binary_ms"] = out["chi2_ms"] - out["chi2_binary_ms"]

    if do_disk:
        # Star + blackbody: t near the single solution, grid in T_bb,
        # linear solve for (s, f >= 0).
        dnode = em.node_spacing(t_single)
        offs = np.linspace(-3.0, 3.0, 13)
        tc = np.clip(t_single[:, None] + dnode[:, None] * offs[None, :],
                     em.lib.nodes[0], em.lib.nodes[-1])       # (n, T)
        star = em.norm_flux_at(tc)                           # (n, T, 76)
        bb = em.bb                                           # (B, 76)
        wf = w * flux
        # Normal-equation pieces.
        mm = (w[:, None, :] * star ** 2).sum(2)              # (n, T)
        mf = (wf[:, None, :] * star).sum(2)                  # (n, T)
        bbb = w @ (bb ** 2).T                                # (n, B)
        bf = wf @ bb.T                                       # (n, B)
        mb = np.einsum("ntk,bk,nk->ntb", star, bb, w)        # (n, T, B)
        det = mm[:, :, None] * bbb[:, None, :] - mb ** 2
        det = np.where(det > 0, det, np.nan)
        s_fit = (mf[:, :, None] * bbb[:, None, :] - mb * bf[:, None, :]) / det
        f_fit = (mm[:, :, None] * bf[:, None, :] - mb * mf[:, :, None]) / det
        neg = ~(f_fit > 0) | ~(s_fit > 0)
        # Where f < 0 or s < 0 fall back to the star-only solution.
        s_only = np.where(mm > 0, mf / np.where(mm > 0, mm, 1), 0.0)
        s_fit = np.where(neg, s_only[:, :, None], s_fit)
        f_fit = np.where(neg, 0.0, f_fit)
        sff = (wf * flux).sum(1)[:, None, None]
        chi2_d = (sff - 2 * s_fit * mf[:, :, None] - 2 * f_fit * bf[:, None, :]
                  + s_fit ** 2 * mm[:, :, None] + f_fit ** 2 * bbb[:, None, :]
                  + 2 * s_fit * f_fit * mb)
        chi2_d = np.where(np.isfinite(chi2_d), chi2_d, np.inf)
        flat = chi2_d.reshape(n, -1)
        kd = np.argmin(flat, axis=1)
        it, ib = np.unravel_index(kd, chi2_d.shape[1:])
        rows = np.arange(n)
        t_d = tc[rows, it]
        s_d = s_fit[rows, it, ib]
        f_d = f_fit[rows, it, ib]
        tbb = em.disk_temps[ib]
        chi2_disk = flat[rows, kd]
        out["t_disk"] = t_d
        out["teff_disk_star"] = em.lib.teff(t_d)
        out["scale_disk"] = s_d
        out["teff_disk"] = tbb
        out["omega_disk"] = f_d
        out["chi2_disk"] = chi2_disk
        out["dchi2_disk"] = chi2 - chi2_disk
        # Fractional excess of the blackbody at the reddest band.
        star_d = s_d[:, None] * star[rows, it]
        bb_d = f_d[:, None] * bb[ib]
        m6 = em.band_masks["b6"]
        with np.errstate(all="ignore"):
            out["disk_frac_b6"] = (bb_d[:, m6].sum(1) / star_d[:, m6].sum(1))
            # L_dust / L_star with the star on the main sequence.
            abs_off_d = interpolate_nodes(em.lib.nodes, em.lib.abs_offset,
                                          t_d)
            r_over_d2 = (s_d * 10.0 ** (-abs_off_d)
                         * (em.radius_ms_rsun(t_d) * RSUN_M / (10 * PC_M)) ** 2)
            out["f_disk"] = ((f_d / np.pi) * tbb ** 4
                             / (r_over_d2 * em.lib.teff(t_d) ** 4))

    # Information criteria.
    ln_n = np.log(np.maximum(n_used, 1))
    out["bic_single"] = chi2 + N_PARAMS["single"] * ln_n
    out["bic_ms"] = out["chi2_ms"] + N_PARAMS["ms"] * ln_n
    if do_binary:
        out["bic_binary"] = out["chi2_binary"] + N_PARAMS["binary"] * ln_n
        out["dbic_binary"] = out["bic_single"] - out["bic_binary"]
        out["bic_binary_ms"] = (out["chi2_binary_ms"]
                                + N_PARAMS["binary_ms"] * ln_n)
        out["dbic_binary_ms"] = out["bic_ms"] - out["bic_binary_ms"]
    if do_disk:
        out["bic_disk"] = out["chi2_disk"] + N_PARAMS["disk"] * ln_n
        out["dbic_disk"] = out["bic_single"] - out["bic_disk"]
        if do_binary:
            out["dbic_disk_vs_binary"] = out["bic_binary"] - out["bic_disk"]

    df = pd.DataFrame(out)
    df.loc[~usable, [c for c in df.columns if c != "n_bins_used"]] = np.nan
    return df


def model_curves(em: EmpiricalModels, row: dict) -> dict:
    """Model fluxes on the bins for one fitted star (for plotting).

    Args:
        em: Models.
        row: A row of the fit table as a dict.

    Returns:
        Dict of name -> list of fluxes (Jy, dereddened frame) for
        ``single``, ``ms`` (if parallax), ``bin1``, ``bin2``, ``binary``,
        ``disk_star``, ``disk_bb``, ``disk``; plus ``lam``.
    """
    curves = {"lam": em.lam.tolist()}

    def _ok(key):
        v = row.get(key)
        return v is not None and np.isfinite(v)

    if _ok("t_single"):
        curves["single"] = (row["scale_single"]
                            * em.norm_flux_at(np.array([row["t_single"]]))[0])
    if _ok("t_bin1"):
        f1 = row["scale_binary"] * em.abs_flux_at(np.array([row["t_bin1"]]))[0]
        f2 = row["scale_binary"] * em.abs_flux_at(np.array([row["t_bin2"]]))[0]
        curves["bin1"], curves["bin2"], curves["binary"] = f1, f2, f1 + f2
    if _ok("t_disk"):
        st = row["scale_disk"] * em.norm_flux_at(np.array([row["t_disk"]]))[0]
        bb = row["omega_disk"] * planck_jy_sr(em.lam, row["teff_disk"])
        curves["disk_star"], curves["disk_bb"], curves["disk"] = st, bb, st + bb
    return {k: (np.asarray(v, dtype=float).tolist() if k != "lam" else v)
            for k, v in curves.items()}


def add_population_columns(df: pd.DataFrame, n_bins: int = 100,
                           columns: tuple = ("dbic_binary", "dbic_binary_ms",
                                             "dbic_disk", "chi2_single",
                                             "red_excess", "lum_ratio",
                                             "resid_a1")
                           ) -> pd.DataFrame:
    """Percentile rank of columns among stars of similar ``t_single``.

    Adds ``<col>_pct`` in [0, 1] (fraction of stars at that ``t`` with a
    smaller value), computed in ``n_bins`` quantile bins of ``t``.
    """
    t = df["t_single"].to_numpy(dtype=float)
    ok = np.isfinite(t)
    edges = np.quantile(t[ok], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    idx = np.full(len(t), -1)
    idx[ok] = np.clip(np.searchsorted(edges, t[ok], side="right") - 1, 0,
                      n_bins - 1)
    for col in columns:
        if col not in df:
            continue
        v = df[col].to_numpy(dtype=float)
        pct = np.full(len(v), np.nan)
        for k in range(n_bins):
            sel = (idx == k) & np.isfinite(v)
            if sel.sum() < 2:
                continue
            order = np.argsort(v[sel], kind="stable")
            ranks = np.empty(sel.sum())
            ranks[order] = (np.arange(sel.sum()) + 0.5) / sel.sum()
            pct[sel] = ranks
        df[f"{col}_pct"] = pct
    return df
