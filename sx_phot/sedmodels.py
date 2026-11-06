"""Stellar model grids resampled for SPHEREx SED fitting.

Two grids are supported:

* SPHINX (Iyer et al. 2023, Zenodo 11392341): M dwarfs, Teff 2000-4000 K,
  logg 4.0-5.5, [M/H] -1 to +1, C/O 0.3-0.9.  Only C/O = 0.5 is loaded.
  Files are ``Teff_<T>_logg_<g>_logZ_<z>_CtoO_<c>.txt`` with columns
  wavelength [um] and emergent flux F_lambda [W/m^2/m].
* Castelli & Kurucz 2004 ATLAS9 (``ck04models`` from STScI): Teff
  3500-50000 K, logg 0-5, [M/H] -2.5 to +0.5.  FITS tables with
  WAVELENGTH [A] and columns g00..g50 of F_lambda [erg/s/cm^2/A].

Both grids tabulate the emergent surface flux (its integral equals
sigma Teff^4), so an observed flux is ``F_model * (R / d)^2``.

Every model is stored on two wavelength grids: the SPHEREx binning grid
(``bin_edges``, R = 40 from 0.75 to 5.0 um, bin-averaged) used for
fitting, and a fine grid (R = 300) used for plotting.  Fluxes are in Jy
for ``(R/d)^2 = 1``.  A resampled grid is cached as an ``.npz`` next to
the model directory so start-up costs a few ms.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

C_M_PER_S = 299792458.0
JY = 1e-26  # W / m^2 / Hz

DEFAULT_MODEL_ROOT = Path(os.environ.get("SX_MODEL_ROOT",
                                         "/ar0/local/models"))
SPHINX_DIR = DEFAULT_MODEL_ROOT / "sphinx" / "NEWCORRECTED_CLOUDFREE"
CK04_DIR = DEFAULT_MODEL_ROOT / "ck04_dl"
CACHE_DIR = DEFAULT_MODEL_ROOT / "cache"

BIN_LAMBDA_MIN_UM = 0.75
BIN_LAMBDA_MAX_UM = 5.0
BIN_RESOLUTION = 40.0
FINE_RESOLUTION = 300.0
FINE_LAMBDA_MIN_UM = 0.70
FINE_LAMBDA_MAX_UM = 5.10

_SPHINX_RE = re.compile(
    r"Teff_([0-9.]+)_logg_([0-9.]+)_logZ_([-+0-9.]+)_CtoO_([0-9.]+)\.txt$"
)
_CK04_RE = re.compile(r"ck([mp])(\d\d)_(\d+)\.fits$")


def spherex_bin_edges(
    lam_min: float = BIN_LAMBDA_MIN_UM,
    lam_max: float = BIN_LAMBDA_MAX_UM,
    resolution: float = BIN_RESOLUTION,
) -> np.ndarray:
    """Return log-spaced bin edges (um) at the given resolving power.

    Args:
        lam_min: First edge.
        lam_max: Approximate last edge (the last bin may overshoot it by
            less than one bin width).
        resolution: lambda / delta-lambda.

    Returns:
        Array of n_bins + 1 edges.
    """
    n = int(np.ceil(np.log(lam_max / lam_min) * resolution))
    return lam_min * np.exp(np.arange(n + 1) / resolution)


def fine_wavelength_grid(
    lam_min: float = FINE_LAMBDA_MIN_UM,
    lam_max: float = FINE_LAMBDA_MAX_UM,
    resolution: float = FINE_RESOLUTION,
) -> np.ndarray:
    """Return a log-spaced wavelength grid (um) for plotting models."""
    n = int(np.ceil(np.log(lam_max / lam_min) * resolution))
    return lam_min * np.exp(np.arange(n + 1) / resolution)


def flam_si_to_jy(lam_um: np.ndarray, flam_w_m2_m: np.ndarray) -> np.ndarray:
    """Convert F_lambda [W/m^2/m] to F_nu [Jy] at wavelengths in um."""
    lam_m = np.asarray(lam_um, dtype=float) * 1e-6
    return np.asarray(flam_w_m2_m, dtype=float) * lam_m ** 2 / C_M_PER_S / JY


def bin_average(
    lam_um: np.ndarray,
    flux: np.ndarray,
    edges: np.ndarray,
    oversample: int = 16,
) -> np.ndarray:
    """Average a spectrum over bins by interpolating on a fine sub-grid.

    Args:
        lam_um: Model wavelengths (monotonic).
        flux: Model fluxes.
        edges: Bin edges.
        oversample: Sub-samples per bin.

    Returns:
        Bin-averaged flux, one value per bin.
    """
    n_bins = len(edges) - 1
    frac = (np.arange(oversample) + 0.5) / oversample
    sub = edges[:-1, None] * (edges[1:] / edges[:-1])[:, None] ** frac[None]
    vals = np.interp(sub.ravel(), lam_um, flux, left=np.nan, right=np.nan)
    return vals.reshape(n_bins, oversample).mean(axis=1)


@dataclass
class ModelGrid:
    """A resampled model grid.

    Attributes:
        name: Grid name ('sphinx' or 'ck04').
        teff: Node Teff values (n_models).
        logg: Node logg values (n_models).
        mh: Node [M/H] values (n_models).
        bin_edges: SPHEREx bin edges (n_bins + 1).
        bin_flux_jy: Bin-averaged F_nu for (R/d)^2 = 1 (n_models, n_bins).
        fine_lam_um: Fine wavelength grid for plotting.
        fine_flux_jy: F_nu on the fine grid (n_models, n_fine).
    """

    name: str
    teff: np.ndarray
    logg: np.ndarray
    mh: np.ndarray
    bin_edges: np.ndarray
    bin_flux_jy: np.ndarray
    fine_lam_um: np.ndarray
    fine_flux_jy: np.ndarray

    @property
    def bin_centers(self) -> np.ndarray:
        """Geometric bin centers (um)."""
        return np.sqrt(self.bin_edges[:-1] * self.bin_edges[1:])

    @property
    def teff_values(self) -> np.ndarray:
        """Sorted unique Teff nodes."""
        return np.unique(self.teff)

    @property
    def logg_values(self) -> np.ndarray:
        """Sorted unique logg nodes."""
        return np.unique(self.logg)

    @property
    def mh_values(self) -> np.ndarray:
        """Sorted unique [M/H] nodes."""
        return np.unique(self.mh)

    def nearest(self, value: float, axis: str) -> float:
        """Snap a value to the nearest node of ``axis`` ('logg' or 'mh')."""
        nodes = self.logg_values if axis == "logg" else self.mh_values
        return float(nodes[np.argmin(np.abs(nodes - value))])

    def teff_sequence(
        self,
        logg: float,
        mh: float,
        teff_step: float = 25.0,
        teff_min: Optional[float] = None,
        teff_max: Optional[float] = None,
        fine: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return models along Teff at fixed (logg, [M/H]).

        Models are linearly interpolated in log flux between the nearest
        Teff nodes.  Missing nodes are skipped.

        Args:
            logg: Requested logg (snapped to the nearest node).
            mh: Requested [M/H] (snapped to the nearest node).
            teff_step: Spacing of the returned Teff sequence (K).
            teff_min: Lower limit (default: grid minimum).
            teff_max: Upper limit (default: grid maximum).
            fine: Return the fine (plotting) grid instead of the bins.

        Returns:
            (teff, flux) with flux shaped (n_teff, n_wavelength).

        Raises:
            ValueError: If fewer than two nodes exist at (logg, [M/H]).
        """
        logg = self.nearest(logg, "logg")
        mh = self.nearest(mh, "mh")
        sel = np.isclose(self.logg, logg) & np.isclose(self.mh, mh)
        flux_all = self.fine_flux_jy if fine else self.bin_flux_jy
        nodes = self.teff[sel]
        order = np.argsort(nodes)
        nodes = nodes[order]
        node_flux = flux_all[sel][order]
        if len(nodes) < 2:
            raise ValueError(
                f"{self.name}: need >=2 Teff nodes at logg={logg} mh={mh}"
            )
        lo = nodes[0] if teff_min is None else max(teff_min, nodes[0])
        hi = nodes[-1] if teff_max is None else min(teff_max, nodes[-1])
        teff = np.arange(np.ceil(lo / teff_step) * teff_step, hi + 1e-6,
                         teff_step)
        if teff.size == 0 or teff[0] > hi:
            teff = np.array([lo])
        logf = np.log(np.clip(node_flux, 1e-300, None))
        out = np.empty((teff.size, logf.shape[1]))
        for k in range(logf.shape[1]):
            out[:, k] = np.interp(teff, nodes, logf[:, k])
        return teff, np.exp(out)

    def model_at(self, teff: float, logg: float, mh: float,
                 fine: bool = False) -> np.ndarray:
        """Return one interpolated model (bins or fine grid)."""
        t, f = self.teff_sequence(logg, mh, teff_step=1.0, teff_min=teff,
                                  teff_max=teff, fine=fine)
        return f[0]

    def save(self, path: Path) -> None:
        """Save to an ``.npz`` cache."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, name=self.name, teff=self.teff, logg=self.logg, mh=self.mh,
            bin_edges=self.bin_edges, bin_flux_jy=self.bin_flux_jy,
            fine_lam_um=self.fine_lam_um, fine_flux_jy=self.fine_flux_jy,
        )

    @classmethod
    def load(cls, path: Path) -> "ModelGrid":
        """Load from an ``.npz`` cache."""
        with np.load(path) as z:
            return cls(
                name=str(z["name"]), teff=z["teff"], logg=z["logg"],
                mh=z["mh"], bin_edges=z["bin_edges"],
                bin_flux_jy=z["bin_flux_jy"], fine_lam_um=z["fine_lam_um"],
                fine_flux_jy=z["fine_flux_jy"],
            )


def _resample(lam_um: np.ndarray, fnu_jy: np.ndarray, edges: np.ndarray,
              fine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    binned = bin_average(lam_um, fnu_jy, edges)
    fine_flux = np.interp(fine, lam_um, fnu_jy)
    return binned, fine_flux


def build_sphinx_grid(
    model_dir: Path = SPHINX_DIR,
    cto: float = 0.5,
    bin_edges: Optional[np.ndarray] = None,
    fine_lam: Optional[np.ndarray] = None,
) -> ModelGrid:
    """Resample the SPHINX grid (one C/O value) onto the SPHEREx grids.

    Args:
        model_dir: Directory of SPHINX ``.txt`` spectra.
        cto: C/O ratio to load.
        bin_edges: Bin edges (default ``spherex_bin_edges()``).
        fine_lam: Fine grid (default ``fine_wavelength_grid()``).

    Returns:
        ModelGrid named 'sphinx'.

    Raises:
        FileNotFoundError: If no matching files exist.
    """
    edges = spherex_bin_edges() if bin_edges is None else bin_edges
    fine = fine_wavelength_grid() if fine_lam is None else fine_lam
    files = sorted(Path(model_dir).glob("Teff_*.txt"))
    teff, logg, mh, binned, fines = [], [], [], [], []
    for f in files:
        m = _SPHINX_RE.search(f.name)
        if not m or not np.isclose(float(m.group(4)), cto):
            continue
        data = np.loadtxt(f)
        lam = data[:, 0]
        keep = (lam > 0.5) & (lam < 6.0)
        fnu = flam_si_to_jy(lam[keep], data[keep, 1])
        b, fi = _resample(lam[keep], fnu, edges, fine)
        teff.append(float(m.group(1)))
        logg.append(float(m.group(2)))
        mh.append(float(m.group(3)))
        binned.append(b)
        fines.append(fi)
    if not teff:
        raise FileNotFoundError(f"No SPHINX C/O={cto} files in {model_dir}")
    log.info("SPHINX: %d models (C/O=%.1f)", len(teff), cto)
    return ModelGrid(
        name="sphinx", teff=np.array(teff), logg=np.array(logg),
        mh=np.array(mh), bin_edges=edges, bin_flux_jy=np.array(binned),
        fine_lam_um=fine, fine_flux_jy=np.array(fines),
    )


def build_ck04_grid(
    model_dir: Path = CK04_DIR,
    teff_max: float = 15000.0,
    bin_edges: Optional[np.ndarray] = None,
    fine_lam: Optional[np.ndarray] = None,
) -> ModelGrid:
    """Resample the Castelli-Kurucz 2004 grid onto the SPHEREx grids.

    Args:
        model_dir: Directory holding ``ck{m,p}NN/`` subdirectories.
        teff_max: Skip models hotter than this.
        bin_edges: Bin edges (default ``spherex_bin_edges()``).
        fine_lam: Fine grid (default ``fine_wavelength_grid()``).

    Returns:
        ModelGrid named 'ck04'.  Columns of zeros (logg values without a
        converged model) are skipped.

    Raises:
        FileNotFoundError: If no files are found.
    """
    from astropy.io import fits

    edges = spherex_bin_edges() if bin_edges is None else bin_edges
    fine = fine_wavelength_grid() if fine_lam is None else fine_lam
    files = sorted(Path(model_dir).glob("ck*/ck*_*.fits"))
    teff, logg, mh, binned, fines = [], [], [], [], []
    for f in files:
        m = _CK04_RE.search(f.name)
        if not m:
            continue
        t = float(m.group(3))
        if t > teff_max:
            continue
        z = float(m.group(2)) / 10.0 * (-1 if m.group(1) == "m" else 1)
        with fits.open(f) as hdul:
            tab = hdul[1].data
            lam_um = np.asarray(tab["WAVELENGTH"], dtype=float) * 1e-4
            keep = (lam_um > 0.5) & (lam_um < 6.0)
            for col in tab.columns.names:
                if not col.startswith("g"):
                    continue
                flam = np.asarray(tab[col], dtype=float)[keep]
                if not np.any(flam > 0):
                    continue
                # erg/s/cm^2/A -> W/m^2/m: 1e-7 * 1e4 * 1e10
                fnu = flam_si_to_jy(lam_um[keep], flam * 1e7)
                b, fi = _resample(lam_um[keep], fnu, edges, fine)
                teff.append(t)
                logg.append(float(col[1:]) / 10.0)
                mh.append(z)
                binned.append(b)
                fines.append(fi)
    if not teff:
        raise FileNotFoundError(f"No ck04 files in {model_dir}")
    log.info("ck04: %d models", len(teff))
    return ModelGrid(
        name="ck04", teff=np.array(teff), logg=np.array(logg),
        mh=np.array(mh), bin_edges=edges, bin_flux_jy=np.array(binned),
        fine_lam_um=fine, fine_flux_jy=np.array(fines),
    )


_GRID_CACHE: dict[str, ModelGrid] = {}


def load_grid(name: str, cache_dir: Path = CACHE_DIR,
              rebuild: bool = False) -> ModelGrid:
    """Load a resampled grid, building and caching it on first use.

    Args:
        name: 'sphinx' or 'ck04'.
        cache_dir: Directory for the ``.npz`` cache.
        rebuild: Force a rebuild from the raw model files.

    Returns:
        The ModelGrid (memoized per process).
    """
    if name in _GRID_CACHE and not rebuild:
        return _GRID_CACHE[name]
    path = Path(cache_dir) / f"sedgrid_{name}_R{int(BIN_RESOLUTION)}.npz"
    if path.exists() and not rebuild:
        grid = ModelGrid.load(path)
    else:
        if name == "sphinx":
            grid = build_sphinx_grid()
        elif name == "ck04":
            grid = build_ck04_grid()
        else:
            raise ValueError(f"Unknown grid {name!r}")
        grid.save(path)
        log.info("Cached %s grid -> %s", name, path)
    _GRID_CACHE[name] = grid
    return grid


__all__ = [
    "BIN_RESOLUTION",
    "ModelGrid",
    "bin_average",
    "build_ck04_grid",
    "build_sphinx_grid",
    "fine_wavelength_grid",
    "flam_si_to_jy",
    "load_grid",
    "spherex_bin_edges",
]
