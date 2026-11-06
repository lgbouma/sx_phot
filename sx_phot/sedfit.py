"""SED fits of binned SPHEREx spectra against stellar model grids.

The data are a spectrum binned onto ``sedmodels.spherex_bin_edges()``
(R = 40, 0.75-5.0 um).  A model set is a stack of fluxes on the same
bins at ``(R/d)^2 = 1``; the fit scale is therefore ``(R/d)^2`` and,
with a parallax, yields the stellar radius.

Three products are computed per star:

* ``full``: single model fit to every usable bin (Teff, scale, radius).
* ``blue``: single model fit to bins below ``blue_max_um`` (default
  1.7 um, SPHEREx bands 1-2) where the primary dominates; the model is
  then extrapolated redward and the fractional excess of the data over
  the model is measured per band (``excess_b3`` .. ``excess_b6``) and
  overall for lambda > 2.4 um (``red_excess``).  A positive red excess
  with a good blue fit may indicate an unresolved cooler companion or
  circumstellar dust.
* ``binary``: two-component fit (primary + cooler secondary, both scales
  non-negative), compared with the single fit through delta chi^2.
* ``disk``: star + blackbody (dust) component with free temperature
  (``DISK_TEMPS_K``) and non-negative scale, for infrared excesses that
  a stellar companion cannot produce.

The three models are compared with the Bayesian information criterion,
``BIC = chi^2 + k ln(n)`` with k = 2 (single), 4 (binary), 4 (disk)
free parameters and n used bins; ``dbic_binary`` and ``dbic_disk`` are
``BIC_single - BIC_model`` (positive favours the model), and
``dbic_disk_vs_binary = BIC_binary - BIC_disk``.

``add_population_columns`` adds Teff-relative versions of the excess
columns (the population median at each Teff subtracted, scatter folded
into the significance) for selecting outliers.

Errors used in chi^2 are the binned errors with a fractional floor
(``err_floor``, default 2 percent) added in quadrature so that a few
very high S/N bins do not dominate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from sx_phot.sedmodels import ModelGrid, load_grid, spherex_bin_edges

PC_M = 3.0856775814913674e16
RSUN_M = 6.957e8

MAMAJEK_PATH = Path(os.environ.get(
    "SX_MAMAJEK_TABLE",
    "/ar0/local/models/mamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt"))

SPHINX_TEFF_MAX = 4000.0
CK04_TEFF_MIN = 4250.0
DEFAULT_TEFF_MAX = 12000.0

H_PLANCK = 6.62607015e-34
K_BOLTZ = 1.380649e-23
C_LIGHT = 299792458.0
DISK_TEMPS_K = np.exp(np.linspace(np.log(200.0), np.log(2000.0), 25))
N_PARAMS = {"single": 2, "binary": 4, "disk": 4}

BAND_EDGES_UM = {
    "b1": (0.75, 1.1075), "b2": (1.1075, 1.6435), "b3": (1.6435, 2.421),
    "b4": (2.423, 3.8155), "b5": (3.8155, 4.416), "b6": (4.416, 5.002),
}


class MainSequence:
    """Dwarf Teff-radius-logg relation from the Pecaut & Mamajek table.

    Args:
        path: ``EEM_dwarf_UBVIJHK_colors_Teff.txt``.  If missing, a short
            built-in table is used.
    """

    _FALLBACK = np.array([
        # Teff, R/Rsun, Msun (approximate dwarf values)
        [2500, 0.10, 0.08], [2800, 0.13, 0.10], [3000, 0.19, 0.15],
        [3200, 0.28, 0.25], [3400, 0.39, 0.37], [3600, 0.49, 0.50],
        [3800, 0.55, 0.58], [4000, 0.62, 0.65], [4400, 0.68, 0.70],
        [4800, 0.75, 0.78], [5200, 0.85, 0.87], [5600, 0.95, 0.97],
        [6000, 1.10, 1.08], [6400, 1.30, 1.25], [7000, 1.55, 1.50],
        [8000, 1.90, 1.90], [10000, 2.40, 2.50], [12000, 2.80, 3.20],
    ])

    def __init__(self, path: Path = MAMAJEK_PATH) -> None:
        table = self._read(Path(path)) if Path(path).exists() else None
        if table is None:
            table = self._FALLBACK
        order = np.argsort(table[:, 0])
        self.teff = table[order, 0]
        self.radius = table[order, 1]
        self.mass = table[order, 2]

    @staticmethod
    def _read(path: Path) -> Optional[np.ndarray]:
        rows = []
        header = None
        for line in path.read_text().splitlines():
            if line.startswith("#SpT"):
                header = line[1:].split()
                continue
            if header is None or not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < len(header) - 1:
                continue
            try:
                rec = dict(zip(header, parts))
                t = float(rec["Teff"])
                r = float(rec["R_Rsun"])
                m = float(rec["Msun"])
            except (ValueError, KeyError):
                continue
            if not rec["SpT"].endswith("V"):
                continue
            rows.append([t, r, m])
        if not rows:
            return None
        arr = np.array(rows)
        # Keep unique Teff values (the table repeats some).
        _, idx = np.unique(arr[:, 0], return_index=True)
        return arr[idx]

    def radius_at(self, teff) -> np.ndarray:
        """Dwarf radius (Rsun) at Teff (K), log-interpolated."""
        teff = np.asarray(teff, dtype=float)
        return np.exp(np.interp(np.log(teff), np.log(self.teff),
                                np.log(self.radius)))

    def mass_at(self, teff) -> np.ndarray:
        """Dwarf mass (Msun) at Teff (K), log-interpolated."""
        teff = np.asarray(teff, dtype=float)
        return np.exp(np.interp(np.log(teff), np.log(self.teff),
                                np.log(self.mass)))

    def logg_at(self, teff) -> np.ndarray:
        """Dwarf surface gravity (cgs) at Teff (K)."""
        m = self.mass_at(teff)
        r = self.radius_at(teff)
        return np.log10(m / r ** 2) + 4.438


@dataclass
class ModelSet:
    """A stack of binned models used for fitting.

    Attributes:
        teff: (n_models,) Teff values.
        grid: (n_models,) grid names ('sphinx' or 'ck04').
        logg: (n_models,) logg used.
        mh: (n_models,) [M/H] used.
        flux: (n_models, n_bins) F_nu in Jy for (R/d)^2 = 1.
        bin_edges: (n_bins + 1,) bin edges in um.
    """

    teff: np.ndarray
    grid: np.ndarray
    logg: np.ndarray
    mh: np.ndarray
    flux: np.ndarray
    bin_edges: np.ndarray = field(default_factory=spherex_bin_edges)

    @property
    def bin_centers(self) -> np.ndarray:
        """Geometric bin centers (um)."""
        return np.sqrt(self.bin_edges[:-1] * self.bin_edges[1:])

    def blackbodies(self) -> tuple[np.ndarray, np.ndarray]:
        """(temps, flux) of the dust blackbody components (cached)."""
        cache = getattr(self, "_bb", None)
        if cache is None:
            cache = blackbody_set(self.bin_edges)
            object.__setattr__(self, "_bb", cache)
        return cache


def planck_fnu_jy(lam_um, teff: float) -> np.ndarray:
    """Emergent blackbody flux pi B_nu(T) in Jy at wavelengths in um.

    Normalised like the stellar grids (its integral is sigma T^4), so a
    scale of ``(R/d)^2`` turns it into an observed flux.
    """
    lam_m = np.asarray(lam_um, dtype=float) * 1e-6
    nu = C_LIGHT / lam_m
    x = H_PLANCK * nu / (K_BOLTZ * float(teff))
    with np.errstate(over="ignore"):
        b = 2 * H_PLANCK * nu ** 3 / C_LIGHT ** 2 / np.expm1(x)
    return np.pi * b / 1e-26


def blackbody_set(
    bin_edges: np.ndarray,
    temps: np.ndarray = DISK_TEMPS_K,
    oversample: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin-averaged blackbodies on the SPHEREx bins.

    Returns:
        (temps, flux) with flux shaped (n_temps, n_bins), in Jy for
        ``(R/d)^2 = 1``.
    """
    frac = (np.arange(oversample) + 0.5) / oversample
    sub = (bin_edges[:-1, None]
           * (bin_edges[1:] / bin_edges[:-1])[:, None] ** frac[None])
    flux = np.array([planck_fnu_jy(sub.ravel(), t).reshape(sub.shape).mean(
        axis=1) for t in temps])
    return np.asarray(temps, float), flux


def build_model_set(
    logg: Optional[float] = None,
    mh: float = 0.0,
    teff_step: float = 25.0,
    teff_min: float = 2000.0,
    teff_max: float = DEFAULT_TEFF_MAX,
    grids: tuple[str, ...] = ("sphinx", "ck04"),
    main_sequence: Optional[MainSequence] = None,
) -> ModelSet:
    """Concatenate Teff sequences from the SPHINX and ck04 grids.

    SPHINX covers Teff <= 4000 K and ck04 Teff >= 4250 K; no
    interpolation crosses the seam.  If ``logg`` is None, each grid uses
    the dwarf logg at the midpoint of its Teff range (snapped to the
    nearest node); otherwise ``logg`` is snapped per grid.

    Args:
        logg: Fixed logg or None for a dwarf value.
        mh: Metallicity (snapped per grid).
        teff_step: Teff spacing of the interpolated sequence (K).
        teff_min: Lower Teff limit.
        teff_max: Upper Teff limit.
        grids: Which grids to include.
        main_sequence: Relation used for the default logg.

    Returns:
        ModelSet sorted by Teff.
    """
    ms = main_sequence or MainSequence()
    parts = []
    for name in grids:
        g = load_grid(name)
        if name == "sphinx":
            lo, hi = teff_min, min(teff_max, SPHINX_TEFF_MAX)
        else:
            lo, hi = max(teff_min, CK04_TEFF_MIN), teff_max
        if lo > hi:
            continue
        # Build piecewise so logg can follow the main sequence.
        if logg is None:
            segments = _teff_segments(lo, hi, 500.0)
        else:
            segments = [(lo, hi)]
        for seg_lo, seg_hi in segments:
            use_logg = (logg if logg is not None
                        else float(ms.logg_at(0.5 * (seg_lo + seg_hi))))
            use_logg = g.nearest(use_logg, "logg")
            use_mh = g.nearest(mh, "mh")
            try:
                t, f = g.teff_sequence(use_logg, use_mh, teff_step=teff_step,
                                       teff_min=seg_lo, teff_max=seg_hi)
            except ValueError:
                continue
            parts.append((t, np.full(t.size, name, dtype=object),
                          np.full(t.size, use_logg), np.full(t.size, use_mh),
                          f))
    if not parts:
        raise ValueError("No models in the requested Teff range")
    teff = np.concatenate([p[0] for p in parts])
    grid = np.concatenate([p[1] for p in parts])
    lg = np.concatenate([p[2] for p in parts])
    mhs = np.concatenate([p[3] for p in parts])
    flux = np.concatenate([p[4] for p in parts])
    # Drop duplicate Teff at segment joins, keep sorted.
    _, idx = np.unique(teff + 1e-3 * (grid == "ck04"), return_index=True)
    idx = np.sort(idx)
    return ModelSet(teff=teff[idx], grid=grid[idx].astype(str), logg=lg[idx],
                    mh=mhs[idx], flux=flux[idx],
                    bin_edges=load_grid(grids[0]).bin_edges)


def _teff_segments(lo: float, hi: float, width: float) -> list:
    edges = np.arange(lo, hi, width).tolist() + [hi]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)
            if edges[i + 1] > edges[i]]


def effective_errors(flux: np.ndarray, err: np.ndarray,
                     err_floor: float = 0.02) -> np.ndarray:
    """Add a fractional error floor in quadrature."""
    return np.sqrt(np.asarray(err, float) ** 2
                   + (err_floor * np.abs(np.asarray(flux, float))) ** 2)


def usable_bins(flux: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Bins with finite positive flux and finite positive error."""
    flux = np.asarray(flux, float)
    err = np.asarray(err, float)
    return np.isfinite(flux) & np.isfinite(err) & (flux > 0) & (err > 0)


def chi2_single(
    flux: np.ndarray,
    err: np.ndarray,
    models: np.ndarray,
    use: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """chi^2 and best scale of each model against one spectrum.

    Args:
        flux: (n_bins,) data.
        err: (n_bins,) errors.
        models: (n_models, n_bins) model fluxes.
        use: (n_bins,) boolean mask of bins to fit.

    Returns:
        (chi2, scale), each (n_models,).  Models with no finite flux in
        the used bins get chi2 = inf.
    """
    w = np.where(use, 1.0 / err ** 2, 0.0)
    y = np.where(use, flux, 0.0)
    m = np.where(use[None, :], models, 0.0)
    m = np.where(np.isfinite(m), m, 0.0)
    mm = (w[None, :] * m * m).sum(axis=1)
    my = (w[None, :] * m * y).sum(axis=1)
    yy = (w * y * y).sum()
    scale = np.where(mm > 0, my / np.where(mm > 0, mm, 1.0), 0.0)
    chi2 = yy - scale * my
    chi2 = np.where(mm > 0, chi2, np.inf)
    return chi2, scale


def chi2_pairs(
    flux: np.ndarray,
    err: np.ndarray,
    models: np.ndarray,
    use: np.ndarray,
    primary: Optional[np.ndarray] = None,
    secondary: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """chi^2 of non-negative two-model combinations.

    Args:
        flux: (n_bins,) data.
        err: (n_bins,) errors.
        models: (n_models, n_bins) model fluxes.
        use: (n_bins,) mask of bins to fit.
        primary: Indices of candidate primaries (default all).
        secondary: Indices of candidate secondaries (default all).

    Returns:
        (chi2, s1, s2), each (n_primary, n_secondary); element [i, j] is
        the fit with primary[i] scaled by s1 and secondary[j] by s2.
        Pairs where a scale would be negative, or singular pairs, get
        chi2 = inf.
    """
    n = models.shape[0]
    primary = np.arange(n) if primary is None else np.asarray(primary)
    secondary = np.arange(n) if secondary is None else np.asarray(secondary)
    w = np.where(use, 1.0 / err ** 2, 0.0)
    y = np.where(use, flux, 0.0)
    m = np.where(use[None, :], models, 0.0)
    m = np.where(np.isfinite(m), m, 0.0)
    mw = m * w[None, :]
    b = mw @ y  # (n,)
    yy = (w * y * y).sum()
    m1 = mw[primary]
    m2 = m[secondary]
    a11 = (m1 * m[primary]).sum(axis=1)[:, None]
    a22 = (mw[secondary] * m2).sum(axis=1)[None, :]
    a12 = m1 @ m2.T  # (n_primary, n_secondary)
    b = (b[primary][:, None], b[secondary][None, :])
    det = a11 * a22 - a12 ** 2
    good = det > 1e-12 * np.maximum(a11 * a22, 1e-300)
    det_safe = np.where(good, det, 1.0)
    s1 = (a22 * b[0] - a12 * b[1]) / det_safe
    s2 = (a11 * b[1] - a12 * b[0]) / det_safe
    chi2 = yy - (s1 * b[0] + s2 * b[1])
    bad = (~good) | (s1 < 0) | (s2 < 0)
    chi2 = np.where(bad, np.inf, chi2)
    return chi2, s1, s2


def radius_rsun(scale, parallax_mas) -> np.ndarray:
    """Radius (Rsun) from ``scale = (R/d)^2`` and parallax (mas)."""
    scale = np.asarray(scale, float)
    plx = np.asarray(parallax_mas, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_m = 1000.0 / plx * PC_M
        return np.sqrt(np.clip(scale, 0, None)) * d_m / RSUN_M


def band_excess(
    flux: np.ndarray,
    err: np.ndarray,
    model: np.ndarray,
    centers: np.ndarray,
    lam_lo: float,
    lam_hi: float,
    use: np.ndarray,
) -> tuple[float, float, int]:
    """Weighted fractional excess of data over model in a window.

    Returns:
        (excess, excess_err, n_bins): excess is the error-weighted mean
        of (flux / model - 1) over usable bins in [lam_lo, lam_hi).
    """
    sel = use & (centers >= lam_lo) & (centers < lam_hi) & (model > 0)
    sel &= np.isfinite(model)
    if sel.sum() == 0:
        return np.nan, np.nan, 0
    ratio = flux[sel] / model[sel] - 1.0
    sig = err[sel] / model[sel]
    w = 1.0 / sig ** 2
    return (float((w * ratio).sum() / w.sum()),
            float(1.0 / np.sqrt(w.sum())), int(sel.sum()))


def _teff_interval(teff: np.ndarray, chi2: np.ndarray,
                   delta: float = 1.0) -> tuple[float, float]:
    """Teff range where chi2 <= min + delta (grid-resolution limited)."""
    ok = np.isfinite(chi2)
    if not ok.any():
        return np.nan, np.nan
    within = ok & (chi2 <= np.nanmin(chi2[ok]) + delta)
    return float(teff[within].min()), float(teff[within].max())


def fit_star(
    flux: np.ndarray,
    err: np.ndarray,
    models: ModelSet,
    parallax_mas: Optional[float] = None,
    blue_max_um: float = 1.7,
    err_floor: float = 0.02,
    do_binary: bool = True,
    binary_secondary_max_teff: float = SPHINX_TEFF_MAX,
    binary_primary_window: float = 600.0,
    main_sequence: Optional[MainSequence] = None,
    lam_min_um: float = 0.0,
    lam_max_um: float = 10.0,
    do_disk: bool = True,
) -> dict:
    """Fit one binned spectrum: full, blue-anchored, binary, disk models.

    Args:
        flux: (n_bins,) binned flux in Jy.
        err: (n_bins,) binned errors in Jy.
        models: Model set on the same bins.
        parallax_mas: Parallax for radii (optional).
        blue_max_um: Upper wavelength of the blue-anchored fit window.
        err_floor: Fractional error floor added in quadrature.
        do_binary: Run the two-component fit.
        binary_secondary_max_teff: Secondaries are drawn from models at
            or below this Teff (and never hotter than the primary).
        binary_primary_window: Primaries are searched within this many
            K of the full-range single-fit Teff.
        main_sequence: Dwarf relation for the radius ratio.
        lam_min_um: Ignore bins below this wavelength.
        lam_max_um: Ignore bins above this wavelength.
        do_disk: Run the star + blackbody fit.

    Returns:
        Flat dict of fit results (NaN where undefined).  Keys include
        n_bins_used, teff_full, teff_full_lo, teff_full_hi, grid_full,
        scale_full, chi2_full, dof_full, radius_full, radius_ratio_full,
        teff_blue, chi2_blue, dof_blue, scale_blue, radius_blue,
        excess_b3..excess_b6 (+ _err), red_excess, red_excess_err,
        red_excess_sig; for the binary fit teff_bin1, teff_bin2,
        scale_bin1, scale_bin2, chi2_binary, dchi2_binary, radius_bin1,
        radius_bin2; for the disk fit teff_disk_star, scale_disk_star,
        teff_disk, scale_disk, chi2_disk, dchi2_disk, radius_disk_star,
        radius_disk (sqrt(scale) * d, in Rsun), f_disk (L_dust / L_star);
        and bic_single, bic_binary, bic_disk, dbic_binary, dbic_disk,
        dbic_disk_vs_binary.
    """
    ms = main_sequence or MainSequence()
    flux = np.asarray(flux, float)
    err_eff = effective_errors(flux, np.asarray(err, float), err_floor)
    centers = models.bin_centers
    use = usable_bins(flux, err_eff) & (centers >= lam_min_um)
    use &= centers <= lam_max_um
    out: dict = {"n_bins_used": int(use.sum())}
    nan = float("nan")
    plx = parallax_mas if parallax_mas is not None else nan
    if use.sum() < 3:
        return out

    # Full-range single fit.
    chi2, scale = chi2_single(flux, err_eff, models.flux, use)
    i = int(np.argmin(chi2))
    lo, hi = _teff_interval(models.teff, chi2)
    model_full = scale[i] * models.flux[i]
    out.update({
        "teff_full": float(models.teff[i]), "teff_full_lo": lo,
        "teff_full_hi": hi, "grid_full": str(models.grid[i]),
        "logg_full": float(models.logg[i]), "mh_full": float(models.mh[i]),
        "scale_full": float(scale[i]), "chi2_full": float(chi2[i]),
        "dof_full": int(use.sum() - 2),
        "radius_full": float(radius_rsun(scale[i], plx)),
    })
    out["radius_ratio_full"] = float(
        out["radius_full"] / ms.radius_at(models.teff[i]))
    out["_model_full"] = model_full

    # Blue-anchored fit and red excess.
    use_blue = use & (centers < blue_max_um)
    if use_blue.sum() >= 3:
        chi2b, scaleb = chi2_single(flux, err_eff, models.flux, use_blue)
        j = int(np.argmin(chi2b))
        model_blue = scaleb[j] * models.flux[j]
        out.update({
            "teff_blue": float(models.teff[j]),
            "grid_blue": str(models.grid[j]),
            "scale_blue": float(scaleb[j]), "chi2_blue": float(chi2b[j]),
            "dof_blue": int(use_blue.sum() - 2),
            "radius_blue": float(radius_rsun(scaleb[j], plx)),
        })
        out["radius_ratio_blue"] = float(
            out["radius_blue"] / ms.radius_at(models.teff[j]))
        for band, (blo, bhi) in BAND_EDGES_UM.items():
            e, ee, n = band_excess(flux, err_eff, model_blue, centers, blo,
                                   bhi, use)
            out[f"excess_{band}"] = e
            out[f"excess_{band}_err"] = ee
        e, ee, n = band_excess(flux, err_eff, model_blue, centers, 2.4, 5.1,
                               use)
        out["red_excess"] = e
        out["red_excess_err"] = ee
        out["red_excess_nbins"] = n
        out["red_excess_sig"] = e / ee if (n and ee > 0) else nan
        out["_model_blue"] = model_blue
    else:
        out["red_excess"] = nan

    # Two-component fit (primary + cooler secondary).
    if do_binary:
        t = models.teff
        prim = np.flatnonzero(np.abs(t - t[i]) <= binary_primary_window)
        sec = np.flatnonzero(t <= min(binary_secondary_max_teff, t[i]))
        if prim.size and sec.size:
            chi2p, s1, s2 = chi2_pairs(flux, err_eff, models.flux, use,
                                       prim, sec)
            allowed = t[sec][None, :] <= t[prim][:, None]
            chi2p = np.where(allowed, chi2p, np.inf)
            k = int(np.argmin(chi2p))
            ia, ib = np.unravel_index(k, chi2p.shape)
            a, b = int(prim[ia]), int(sec[ib])
        else:
            chi2p = np.full((1, 1), np.inf)
            ia = ib = a = b = 0
        if np.isfinite(chi2p[ia, ib]):
            out.update({
                "teff_bin1": float(t[a]), "teff_bin2": float(t[b]),
                "scale_bin1": float(s1[ia, ib]),
                "scale_bin2": float(s2[ia, ib]),
                "chi2_binary": float(chi2p[ia, ib]),
                "dof_binary": int(use.sum() - 4),
                "dchi2_binary": float(chi2[i] - chi2p[ia, ib]),
                "radius_bin1": float(radius_rsun(s1[ia, ib], plx)),
                "radius_bin2": float(radius_rsun(s2[ia, ib], plx)),
            })
            out["_model_bin1"] = s1[ia, ib] * models.flux[a]
            out["_model_bin2"] = s2[ia, ib] * models.flux[b]

    # Star + dust blackbody.
    if do_disk:
        t = models.teff
        bb_t, bb_f = models.blackbodies()
        prim = np.flatnonzero(np.abs(t - t[i]) <= binary_primary_window)
        stack = np.vstack([models.flux, bb_f])
        sec = models.flux.shape[0] + np.arange(bb_t.size)
        chi2d, sd1, sd2 = chi2_pairs(flux, err_eff, stack, use, prim, sec)
        k = int(np.argmin(chi2d))
        ia, ib = np.unravel_index(k, chi2d.shape)
        if np.isfinite(chi2d[ia, ib]):
            a = int(prim[ia])
            s_star, s_dust = float(sd1[ia, ib]), float(sd2[ia, ib])
            t_dust = float(bb_t[ib])
            with np.errstate(invalid="ignore", divide="ignore"):
                f_disk = (s_dust * t_dust ** 4) / (s_star * t[a] ** 4)
            out.update({
                "teff_disk_star": float(t[a]), "scale_disk_star": s_star,
                "teff_disk": t_dust, "scale_disk": s_dust,
                "chi2_disk": float(chi2d[ia, ib]),
                "dof_disk": int(use.sum() - 4),
                "dchi2_disk": float(chi2[i] - chi2d[ia, ib]),
                "radius_disk_star": float(radius_rsun(s_star, plx)),
                "radius_disk": float(radius_rsun(s_dust, plx)),
                "f_disk": float(f_disk),
            })
            out["_model_disk_star"] = s_star * models.flux[a]
            out["_model_disk_bb"] = s_dust * bb_f[ib]

    # Model comparison.
    n_use = int(use.sum())
    logn = np.log(n_use)
    out["bic_single"] = float(chi2[i] + N_PARAMS["single"] * logn)
    if "chi2_binary" in out:
        out["bic_binary"] = out["chi2_binary"] + N_PARAMS["binary"] * logn
        out["dbic_binary"] = out["bic_single"] - out["bic_binary"]
    if "chi2_disk" in out:
        out["bic_disk"] = out["chi2_disk"] + N_PARAMS["disk"] * logn
        out["dbic_disk"] = out["bic_single"] - out["bic_disk"]
        if "bic_binary" in out:
            out["dbic_disk_vs_binary"] = out["bic_binary"] - out["bic_disk"]
    return out


def add_population_columns(
    df,
    teff_key: str = "teff_blue",
    bin_width: float = 100.0,
    min_per_bin: int = 50,
    min_n_used: int = 100,
    n_used_key: str = "n_used",
):
    """Add Teff-relative versions of the excess and radius columns.

    Model systematics (grid calibration, band-boundary flux steps) give
    the raw ``red_excess`` a Teff-dependent offset of several percent.
    For each ``bin_width`` K bin of ``teff_key`` this computes the
    population median and robust scatter (1.4826 MAD) of each column over
    stars with at least ``min_n_used`` points, and adds::

        <col>_rel      = col - median(bin)
        <col>_rel_sig  = <col>_rel / sqrt(err^2 + scatter(bin)^2)

    for ``red_excess`` and ``excess_b3`` .. ``excess_b6`` (``_rel_sig``
    only where an ``<col>_err`` column exists),
    ``radius_ratio_full_rel = radius_ratio_full / median(bin)`` binned on
    ``teff_full``, and ``dbic_*_pct``, the percentile rank (0-1) of the
    delta-BIC statistics within the ``teff_full`` bins.  Bins with fewer
    than ``min_per_bin`` stars borrow the nearest populated bin.

    Args:
        df: DataFrame from ``fit_star`` results (modified in place).
        teff_key: Teff column used to bin the excess columns.
        bin_width: Bin width in K.
        min_per_bin: Minimum stars per bin for its own statistics.
        min_n_used: Only stars with at least this many spectrum points
            define the population statistics (all stars get values).
        n_used_key: Column holding the number of points.

    Returns:
        The same DataFrame.
    """
    import pandas as pd

    def _binned_stats(teff, values, good):
        teff = np.asarray(teff, float)
        values = np.asarray(values, float)
        ok = good & np.isfinite(teff) & np.isfinite(values)
        if not ok.any():
            return np.full(teff.size, np.nan), np.full(teff.size, np.nan)
        centers = np.round(teff / bin_width) * bin_width
        frame = pd.DataFrame({"c": centers[ok], "v": values[ok]})
        grp = frame.groupby("c")["v"]
        med = grp.median()
        mad = grp.apply(lambda v: 1.4826 * np.median(np.abs(v - v.median())))
        count = grp.size()
        keep = count >= min_per_bin
        if not keep.any():
            keep[:] = True
        xs = med.index.to_numpy(float)[keep.to_numpy()]
        med_v = med.to_numpy()[keep.to_numpy()]
        mad_v = mad.to_numpy()[keep.to_numpy()]
        has_t = np.isfinite(teff)
        out_med = np.full(teff.size, np.nan)
        out_mad = np.full(teff.size, np.nan)
        near = np.abs(teff[has_t][:, None] - xs[None, :]).argmin(axis=1)
        out_med[has_t] = med_v[near]
        out_mad[has_t] = mad_v[near]
        return out_med, out_mad

    n_used = (df[n_used_key].to_numpy(float) if n_used_key in df
              else np.full(len(df), np.inf))
    good = n_used >= min_n_used
    for col in ("red_excess", "excess_b3", "excess_b4", "excess_b5",
                "excess_b6"):
        if col not in df or teff_key not in df:
            continue
        med, mad = _binned_stats(df[teff_key], df[col], good)
        rel = df[col].to_numpy(float) - med
        df[f"{col}_rel"] = rel
        if f"{col}_err" in df:
            err = df[f"{col}_err"].to_numpy(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                df[f"{col}_rel_sig"] = rel / np.sqrt(err ** 2 + mad ** 2)
    if "radius_ratio_full" in df and "teff_full" in df:
        med, _ = _binned_stats(df["teff_full"], df["radius_ratio_full"],
                               good)
        with np.errstate(invalid="ignore", divide="ignore"):
            df["radius_ratio_full_rel"] = (
                df["radius_ratio_full"].to_numpy(float) / med)
    # Percentile rank of the model-comparison statistics within Teff
    # bins (0-1), so a cut at e.g. 0.95 selects the same fraction of
    # stars at every Teff regardless of the Teff-dependent model
    # systematics.
    for col in ("dbic_binary", "dbic_disk", "dbic_disk_vs_binary"):
        if col in df and "teff_full" in df:
            df[f"{col}_pct"] = _teff_percentile(
                df["teff_full"].to_numpy(float), df[col].to_numpy(float),
                good, bin_width)
    return df


def _teff_percentile(teff: np.ndarray, values: np.ndarray, good: np.ndarray,
                     bin_width: float) -> np.ndarray:
    """Percentile rank (0-1) of ``values`` within ``bin_width`` Teff bins.

    The reference distribution in each bin is the ``good`` stars; every
    star with finite inputs gets a rank.  Bins with fewer than 50
    reference stars are merged with the nearest populated bin.
    """
    out = np.full(teff.size, np.nan)
    ok = np.isfinite(teff) & np.isfinite(values)
    ref = ok & good
    if not ref.any():
        return out
    centers = np.round(teff / bin_width) * bin_width
    ref_bins = np.unique(centers[ref])
    counts = {c: int(np.sum(ref & (centers == c))) for c in ref_bins}
    populated = np.array([c for c in ref_bins if counts[c] >= 50])
    if populated.size == 0:
        populated = ref_bins
    for c in np.unique(centers[ok]):
        use_c = populated[np.argmin(np.abs(populated - c))]
        sample = np.sort(values[ref & (centers == use_c)])
        sel = ok & (centers == c)
        out[sel] = np.searchsorted(sample, values[sel], side="right") / max(
            sample.size, 1)
    return out


def strip_private(result: dict) -> dict:
    """Drop the ``_model_*`` arrays from a ``fit_star`` result."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


__all__ = [
    "BAND_EDGES_UM",
    "MainSequence",
    "DISK_TEMPS_K",
    "N_PARAMS",
    "add_population_columns",
    "blackbody_set",
    "ModelSet",
    "band_excess",
    "build_model_set",
    "chi2_pairs",
    "chi2_single",
    "effective_errors",
    "fit_star",
    "planck_fnu_jy",
    "radius_rsun",
    "strip_private",
    "usable_bins",
]
