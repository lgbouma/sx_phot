"""Serve sx-ml: UMAP explorer and empirical spectral sequence viewer.

Static files are served from this directory.  Data come from a viz
build (``<run>/viz``) and the ML products (``<run>/ml``: ``sequence.npz``
from ``drivers/build_spectral_sequence.py``, ``empfit.parquet`` from
``drivers/fit_empirical.py``, ``umap_<name>.npy`` from
``drivers/run_umap.py``), joined with the TARS table.  Endpoints:

* ``GET /api/ping``                    liveness
* ``GET /api/meta``                    counts, columns, embeddings, build
* ``GET /api/points?emb=<name>``       float32 LE (2 N): x0 y0 x1 y1 ...
* ``GET /api/col/<name>``              float32 LE (N) column values
* ``GET /api/ticids``                  uint32 LE (N)
* ``GET /api/star/<row>``              everything the side panel needs
  (``?by=ticid`` to look up by TIC id)
* ``POST /api/selection``              ``{"rows": [...]}`` -> group stats
* ``GET /api/sequence``                the template library
* ``GET /api/sequence/at?t=&n=&a1=``   template at ``t``, model
  comparison, example stars

Usage:
    python server.py --run-dir /ar0/SPHEREX/bulkphot/tars_v0 --port 8052
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import threading
import time
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from sx_phot import empfit, specml  # noqa: E402
from sx_phot.sedfit import MainSequence  # noqa: E402

LOGGER = logging.getLogger("sx-ml")
MAX_SELECTION = 200000
MAX_BODY_BYTES = 4_000_000
DEFAULT_SXVIZ = os.environ.get("SX_VIZ_URL", "http://wh3:8051")

SUMMARY_COLUMNS = ["ticid", "med_snr", "n_used", "n_good", "n_masked",
                   "spline_ok", "resid_mad", "n_epochs", "ra", "dec"]
SED_COLUMNS = ["ticid", "teff_full", "chi2_full", "radius_ratio_full",
               "teff_blue", "red_excess", "red_excess_sig", "red_excess_rel",
               "red_excess_rel_sig", "excess_b6", "dchi2_binary",
               "dbic_binary", "dbic_binary_pct", "dbic_disk", "dbic_disk_pct",
               "teff_bin2", "f_disk", "teff_disk"]
TARS_EXTRA = ["TICID", "flag_possible_binary", "flag_multiple_periods",
              "n_secs", "dr3_source_id"]

COLUMN_NOTES = {
    "teff_emp": "Teff mapped from the empirical sequence parameter",
    "t_single": "empirical sequence parameter (0 = blue end)",
    "teff_tars": "TARS photometric Teff",
    "bprp0": "Gaia BP-RP, dereddened (TARS)",
    "mg0": "absolute G, dereddened (TARS)",
    "dms": "M_G0 minus the running-median main sequence at BP-RP0",
    "dmag_ms": "-2.5 log10(lum_ratio): SPHEREx overluminosity vs the "
               "main sequence at the Gaia distance (negative = bright)",
    "lum_ratio": "SPHEREx flux / main-sequence template at the Gaia "
                 "distance",
    "chi2_single": "chi2 of the single empirical template",
    "chi2_red": "chi2_single / dof",
    "red_excess": "fractional excess over the single template at "
                  "> 2.4 um",
    "red_excess_sig": "its significance (population scatter included)",
    "blue_excess": "fractional excess at < 1.0 um",
    "dbic_binary": "BIC(single) - BIC(two main-sequence templates, free "
                   "scale)",
    "dbic_binary_ms": "BIC(single at Gaia distance) - BIC(two stars at "
                      "Gaia distance)",
    "dbic_disk": "BIC(single) - BIC(single + blackbody)",
    "dbic_disk_vs_binary": "BIC(binary) - BIC(disk)",
    "teff_bin2": "secondary Teff of the shape-only binary fit",
    "flux_ratio_binary": "secondary / primary flux over 0.75-5 um",
    "teff_disk": "blackbody temperature of the disk fit",
    "disk_frac_b6": "blackbody / star flux in band 6 (4.4-5 um)",
    "resid_a1": "amplitude of the first residual eigenvector (dex)",
    "resid_a2": "amplitude of the second residual eigenvector (dex)",
    "max_abs_z": "largest |residual| / sigma of the single fit",
    "n_bins_4sig": "bins deviating by more than 4 sigma",
    "log_period": "log10 TARS rotation period [d]",
    "log_amplitude": "log10 TARS median amplitude",
    "ruwe": "Gaia RUWE",
    "nss": "Gaia non_single_star flag",
    "n_contams": "TESS contaminants (TARS)",
    "a0": "Gaia A0 used for dereddening",
    "plx": "Gaia parallax [mas]",
    "tmag": "TESS magnitude",
    "med_snr": "median per-point S/N of the spectrum",
    "sed_dbic_binary_pct": "model-atmosphere binary evidence, percentile "
                           "at fixed Teff (sx-viz)",
    "sed_red_excess_rel": "model-atmosphere red excess relative to the "
                          "Teff median (sx-viz)",
    "sequence_train": "1 if the star was used to build the sequence",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _round(arr, nd=5) -> list:
    out = []
    for v in np.asarray(arr, dtype=float).ravel().tolist():
        out.append(round(v, nd) if math.isfinite(v) else None)
    return out


class Data:
    """Table, embeddings, spectra and the template library.

    Args:
        run_dir: Bulk run directory.
        viz_dir: Viz build (default ``<run>/viz``).
        ml_dir: ML products (default ``<run>/ml``).
        tars_path: TARS table.
        sxviz_url: Base URL of the sx-viz app for star-page links.
    """

    def __init__(self, run_dir: Path, viz_dir: Optional[Path],
                 ml_dir: Optional[Path], tars_path: Path,
                 sxviz_url: str = DEFAULT_SXVIZ) -> None:
        t0 = time.time()
        self.run_dir = Path(run_dir)
        self.viz_dir = Path(viz_dir) if viz_dir else self.run_dir / "viz"
        self.ml_dir = Path(ml_dir) if ml_dir else self.run_dir / "ml"
        self.sxviz_url = sxviz_url.rstrip("/")
        cat = specml.load_catalog(self.viz_dir, tars_path)
        n = len(cat)
        tic = cat["ticid"].to_numpy()
        # Extra TARS columns.
        if Path(tars_path).suffix == ".feather":
            tars = pd.read_feather(tars_path, columns=TARS_EXTRA)
        else:
            tars = pd.read_parquet(tars_path, columns=TARS_EXTRA)
        tars = tars.rename(columns={"TICID": "ticid"}).drop_duplicates(
            "ticid").set_index("ticid").reindex(tic)
        for c in tars.columns:
            v = tars[c]
            if str(v.dtype) in ("bool", "boolean"):
                v = v.astype("boolean").fillna(False).astype(float)
            cat[c] = v.to_numpy()
        summ = pd.read_parquet(self.viz_dir / "summary.parquet",
                               columns=[c for c in SUMMARY_COLUMNS])
        summ = summ.set_index("ticid").reindex(tic)
        for c in summ.columns:
            if c not in cat:
                cat[c] = summ[c].to_numpy()
        sed_path = self.viz_dir / "sed.parquet"
        if sed_path.exists():
            sed = pd.read_parquet(sed_path, columns=SED_COLUMNS)
            sed = sed.set_index("ticid").reindex(tic)
            for c in sed.columns:
                cat[f"sed_{c}"] = sed[c].to_numpy()
        emp_path = self.ml_dir / "empfit.parquet"
        self.has_fits = emp_path.exists()
        if self.has_fits:
            emp = pd.read_parquet(emp_path)
            assert np.array_equal(emp["ticid"].to_numpy(), tic)
            for c in emp.columns:
                if c in ("ticid", "row"):
                    continue
                cat[c] = emp[c].to_numpy()
            with np.errstate(all="ignore"):
                cat["chi2_red"] = cat["chi2_single"] / cat["dof_single"]
        train_path = self.ml_dir / "sequence_train.parquet"
        self.train_rows = np.array([], dtype=int)
        self.train_t = np.array([])
        if train_path.exists():
            tr = pd.read_parquet(train_path)
            tr = tr[tr["inlier"]] if "inlier" in tr else tr
            flag = np.zeros(n, dtype=float)
            flag[tr["row"].to_numpy()] = 1.0
            cat["sequence_train"] = flag
            order = np.argsort(tr["t"].to_numpy())
            self.train_rows = tr["row"].to_numpy()[order]
            self.train_t = tr["t"].to_numpy()[order]
        with np.errstate(all="ignore"):
            cat["dms"] = specml.main_sequence_offset(
                cat["bprp0"].to_numpy(dtype=float),
                cat["mg0"].to_numpy(dtype=float))
            cat["log_period"] = np.log10(cat["period"].to_numpy(dtype=float))
            cat["log_amplitude"] = np.log10(
                cat["amplitude"].to_numpy(dtype=float))
            cat["dist_pc"] = 1000.0 / cat["plx"].to_numpy(dtype=float)
        self.table = cat
        self.n = n
        self.ticid = tic
        self.row_of = {int(t): i for i, t in enumerate(tic)}
        self._cols: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        # Embeddings.
        self.embeddings: dict[str, np.ndarray] = {}
        for path in sorted(self.ml_dir.glob("umap_*.npy")):
            name = path.stem[len("umap_"):]
            arr = np.load(path, mmap_mode="r")
            if arr.shape == (n, 2):
                self.embeddings[name] = np.asarray(arr, dtype=np.float32)
        self.umap_meta = {}
        if (self.ml_dir / "umap.json").exists():
            self.umap_meta = json.loads((self.ml_dir / "umap.json").read_text())
        # Spectra.
        self.flux = np.load(self.viz_dir / "binned_flux.npy", mmap_mode="r")
        self.err = np.load(self.viz_dir / "binned_err.npy", mmap_mode="r")
        self.nper = np.load(self.viz_dir / "binned_n.npy", mmap_mode="r")
        self.lam = specml.bin_centers(self.viz_dir)
        # Library.
        self.lib = specml.SpectralSequence.load(self.ml_dir / "sequence")
        self.em = empfit.EmpiricalModels(self.lib)
        self.ms = MainSequence()
        self._sequence_json: Optional[dict] = None
        self._models = None
        self._models_tried = False
        self.columns = self._column_meta()
        LOGGER.info("loaded %d stars, %d columns, embeddings %s in %.1f s",
                    n, len(self.columns), list(self.embeddings),
                    time.time() - t0)

    # -- columns -----------------------------------------------------------

    def _column_meta(self) -> list[dict]:
        out = []
        for name in self.table.columns:
            s = self.table[name]
            kind = s.dtype.kind
            if kind not in "iufb" and str(s.dtype) != "boolean":
                continue
            v = self.col(name)
            ok = np.isfinite(v)
            if not ok.any():
                continue
            lo, p1, p99, hi = np.percentile(v[ok], [0, 1, 99, 100])
            out.append({"name": name, "min": float(lo), "max": float(hi),
                        "p1": float(p1), "p99": float(p99),
                        "n": int(ok.sum()),
                        "note": COLUMN_NOTES.get(name, "")})
        return out

    def col(self, name: str) -> np.ndarray:
        """float32 view of a column (NaN where missing)."""
        with self._lock:
            if name not in self._cols:
                if name not in self.table:
                    raise KeyError(name)
                s = self.table[name]
                if str(s.dtype) == "boolean":
                    arr = s.fillna(False).to_numpy(dtype=float)
                else:
                    arr = s.to_numpy(dtype=float)
                self._cols[name] = arr.astype(np.float32)
            return self._cols[name]

    def meta(self) -> dict:
        return {
            "n": self.n, "columns": self.columns,
            "embeddings": [{"name": k,
                            "n": int(np.isfinite(v[:, 0]).sum())}
                           for k, v in self.embeddings.items()],
            "umap": self.umap_meta.get("settings", {}),
            "has_fits": self.has_fits,
            "sequence": {"n_nodes": self.lib.n_nodes,
                         "meta": self.lib.meta},
            "sxviz_url": self.sxviz_url,
            "run_dir": str(self.run_dir),
        }

    # -- stars -------------------------------------------------------------

    def row_dict(self, row: int) -> dict:
        rec = self.table.iloc[row].to_dict()
        return _json_safe({k: (v if not isinstance(v, (bytes,)) else None)
                           for k, v in rec.items()})

    def star(self, row: int) -> dict:
        """Everything the side panel needs for one star."""
        if row < 0 or row >= self.n:
            raise KeyError(row)
        rec = self.row_dict(row)
        flux = np.asarray(self.flux[row], dtype=float)
        err = np.asarray(self.err[row], dtype=float)
        nper = np.asarray(self.nper[row])
        valid = (nper > 0) & np.isfinite(flux) & (flux > 0)
        a0 = rec.get("a0")
        fac = specml.deredden_factor(self.lam, np.array([a0 or 0.0]))[0]
        payload = {
            "row": row, "ticid": int(self.ticid[row]), "info": rec,
            "lam": _round(self.lam, 5),
            "flux": _round(np.where(valid, flux, np.nan), 7),
            "err": _round(np.where(valid, err, np.nan), 7),
            "flux_dered": _round(np.where(valid, flux * fac, np.nan), 7),
            "err_dered": _round(np.where(valid, err * fac, np.nan), 7),
            "n": nper.astype(int).tolist(),
            "embedding": {k: _round(v[row], 4)
                          for k, v in self.embeddings.items()},
            "links": {"sxviz": f"{self.sxviz_url}/star.html?ticid="
                               f"{int(self.ticid[row])}"},
        }
        if self.has_fits and rec.get("t_single") is not None:
            curves = empfit.model_curves(self.em, rec)
            payload["curves"] = {k: (_round(v, 7) if k != "lam" else None)
                                 for k, v in curves.items() if k != "lam"}
            t = rec["t_single"]
            sig = empfit.effective_sigma(
                flux * fac, err * fac,
                self.em.scatter_lin_at(np.array([t]))[0])
            payload["sigma_eff"] = _round(np.where(valid, sig, np.nan), 7)
            payload["template_norm"] = _round(
                self.lib.template(np.array([t]))[0], 5)
        return payload

    # -- selections --------------------------------------------------------

    def selection(self, rows: np.ndarray, columns: tuple = (
            "teff_emp", "bprp0", "mg0", "dms", "dmag_ms", "lum_ratio",
            "log_period", "log_amplitude", "ruwe", "a0", "tmag", "med_snr",
            "n_contams", "chi2_red", "red_excess", "dbic_binary",
            "dbic_disk", "resid_a1", "sed_dbic_binary_pct")) -> dict:
        """Median residual spectrum and column medians of a group."""
        rows = np.unique(rows[(rows >= 0) & (rows < self.n)])
        if len(rows) == 0:
            return {"n": 0}
        if len(rows) > MAX_SELECTION:
            rng = np.random.default_rng(0)
            rows = np.sort(rng.choice(rows, MAX_SELECTION, replace=False))
        out: dict[str, Any] = {"n": int(len(rows))}
        medians = {}
        for c in columns:
            if c in self.table:
                v = self.col(c)[rows]
                ok = np.isfinite(v)
                if ok.any():
                    p16, med, p84 = np.percentile(v[ok], [16, 50, 84])
                    medians[c] = {"med": float(med), "p16": float(p16),
                                  "p84": float(p84), "n": int(ok.sum())}
        out["columns"] = medians
        if self.has_fits:
            t = self.col("t_single")[rows].astype(float)
            s = self.col("scale_single")[rows].astype(float)
            ok = np.isfinite(t) & np.isfinite(s) & (s > 0)
            if ok.sum() >= 1:
                sub = rows[ok]
                flux = np.asarray(self.flux[sub], dtype=float)
                nper = np.asarray(self.nper[sub])
                valid = (nper > 0) & np.isfinite(flux) & (flux > 0)
                a0 = np.nan_to_num(self.col("a0")[sub].astype(float))
                flux = flux * specml.deredden_factor(self.lam, a0)
                model = s[ok][:, None] * self.em.norm_flux_at(t[ok])
                with np.errstate(all="ignore"):
                    rlog = np.where(valid, np.log10(flux / model), np.nan)
                    ynorm = np.where(valid, np.log10(flux / s[ok][:, None]),
                                     np.nan)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out["resid_med"] = _round(np.nanmedian(rlog, 0), 5)
                    out["resid_p16"] = _round(np.nanpercentile(rlog, 16, 0), 5)
                    out["resid_p84"] = _round(np.nanpercentile(rlog, 84, 0), 5)
                    out["norm_med"] = _round(np.nanmedian(ynorm, 0), 5)
                    out["n_valid"] = valid.sum(0).astype(int).tolist()
                t_med = float(np.median(t[ok]))
                out["t_med"] = t_med
                out["teff_med"] = float(self.lib.teff(np.array([t_med]))[0])
                out["template_med"] = _round(
                    self.lib.template(np.array([t_med]))[0], 5)
                h, e = np.histogram(t[ok], bins=40, range=(0, 1))
                out["t_hist"] = {"counts": h.tolist(), "edges": _round(e, 4)}
        out["lam"] = _round(self.lam, 5)
        out["rows"] = rows[:5000].astype(int).tolist()
        out["ticids"] = self.ticid[rows[:5000]].astype(int).tolist()
        return out

    # -- sequence ----------------------------------------------------------

    def sequence(self) -> dict:
        """The library as JSON (cached)."""
        if self._sequence_json is None:
            lib = self.lib
            s = lib.summary
            teff = lib.teff_mono
            cols = {}
            for c in s.columns:
                if c in ("node",):
                    continue
                if s[c].dtype.kind in "iuf":
                    cols[c] = _round(s[c].to_numpy(dtype=float), 4)
            self._sequence_json = {
                "nodes": _round(lib.nodes, 5), "lam": _round(lib.lam_um, 5),
                "templates": [_round(r, 5) for r in lib.templates],
                "scatter_int": [_round(r, 5) for r in lib.scatter_int],
                "abs_offset": _round(lib.abs_offset, 4),
                "abs_scatter": _round(lib.abs_scatter, 4),
                "arc": _round(lib.arc, 5), "teff_mono": _round(teff, 1),
                "mass_ms": _round(self.ms.mass_at(teff), 3),
                "radius_ms": _round(self.ms.radius_at(teff), 3),
                "resid_comps": [[_round(c, 5) for c in node[:2]]
                                for node in lib.resid_comps],
                "resid_amp": [_round(a[:2], 5) for a in lib.resid_amp],
                "summary": cols, "meta": lib.meta,
                "n_train_rows": int(len(self.train_rows)),
            }
        return self._sequence_json

    def _model_set(self):
        if self._models is None and not self._models_tried:
            self._models_tried = True
            try:
                from sx_phot.sedfit import build_model_set
                ms = build_model_set()
                if ms.flux.shape[1] == len(self.lam):
                    self._models = ms
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("model grids unavailable: %s", exc)
        return self._models

    def sequence_at(self, t: float, n_examples: int = 12,
                    source: str = "train", seed: Optional[int] = None
                    ) -> dict:
        """Template, model comparison and example stars at ``t``."""
        t = float(np.clip(t, self.lib.nodes[0], self.lib.nodes[-1]))
        lib = self.lib
        tt = np.array([t])
        teff = float(lib.teff(tt)[0])
        out = {
            "t": t, "teff": teff, "arc": float(lib.arc_at(tt)[0]),
            "template": _round(lib.template(tt)[0], 5),
            "scatter_int": _round(lib.scatter_at(tt, intrinsic=True)[0], 5),
            "abs_template": _round(lib.abs_template(tt)[0], 5),
            "mass_ms": float(self.ms.mass_at(teff)),
            "radius_ms": float(self.ms.radius_at(teff)),
        }
        for c in ("teff_tars_med", "teff_tars_p16", "teff_tars_p84",
                  "bprp0_med", "mg0_med", "mg0_p16", "mg0_p84", "a0_med",
                  "tmag_med", "period_med", "plx_med"):
            if c in lib.summary:
                out[c] = float(lib.column(c, tt)[0])
        comps = specml.interpolate_nodes(lib.nodes, lib.resid_comps, tt)[0]
        amps = specml.interpolate_nodes(lib.nodes, lib.resid_amp, tt)[0]
        out["resid_comps"] = [_round(c, 5) for c in comps[:2]]
        out["resid_amp"] = _round(amps[:2], 5)
        models = self._model_set()
        if models is not None:
            mteff = np.asarray(models.teff, dtype=float)
            j = int(np.argmin(np.abs(mteff - teff)))
            m = np.log10(np.asarray(models.flux[j], dtype=float))
            out["model"] = {"teff": float(mteff[j]),
                            "grid": str(models.grid[j]),
                            "log_norm": _round(m - np.nanmean(m), 5)}
        # Example stars near t.
        rows = np.array([], dtype=int)
        if source == "train" and len(self.train_rows):
            lo = int(np.searchsorted(self.train_t, t - 0.004))
            hi = int(np.searchsorted(self.train_t, t + 0.004))
            rows = self.train_rows[lo:hi]
        elif self.has_fits:
            ts = self.col("t_single")
            rows = np.flatnonzero(np.abs(ts - t) < 0.004)
        if len(rows) > n_examples:
            rng = np.random.default_rng(seed)
            rows = np.sort(rng.choice(rows, n_examples, replace=False))
        examples = []
        if len(rows) and self.has_fits:
            s = self.col("scale_single")[rows].astype(float)
            a0 = np.nan_to_num(self.col("a0")[rows].astype(float))
            flux = np.asarray(self.flux[rows], dtype=float)
            nper = np.asarray(self.nper[rows])
            valid = (nper > 0) & np.isfinite(flux) & (flux > 0)
            flux = flux * specml.deredden_factor(self.lam, a0)
            with np.errstate(all="ignore"):
                y = np.where(valid & (s[:, None] > 0),
                             np.log10(flux / s[:, None]), np.nan)
            for i, r in enumerate(rows):
                examples.append({
                    "row": int(r), "ticid": int(self.ticid[r]),
                    "teff_tars": _json_safe(self.col("teff_tars")[r]),
                    "tmag": _json_safe(self.col("tmag")[r]),
                    "y": _round(y[i], 5)})
        out["examples"] = examples
        out["n_near"] = int(len(rows))
        return out


class Handler(SimpleHTTPRequestHandler):
    """Static files plus the JSON / binary API."""

    def __init__(self, *args, data: Data, **kwargs) -> None:
        self.data = data
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102
        LOGGER.debug(fmt, *args)

    def end_headers(self) -> None:  # noqa: D102
        path = urlsplit(self.path).path
        if not path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(_json_safe(payload), separators=(",", ":")
                          ).encode()
        self._send(body, "application/json", status)

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def do_GET(self) -> None:  # noqa: N802
        url = urlsplit(self.path)
        path = url.path
        if not path.startswith("/api/"):
            if path == "/":
                self.path = "/index.html"
            return super().do_GET()
        params = parse_qs(url.query)
        try:
            self._api_get(path, params)
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, f"not found: {exc}")
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("GET %s failed", self.path)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _api_get(self, path: str, params: dict[str, list[str]]) -> None:
        d = self.data
        seg = path[len("/api/"):].strip("/").split("/")
        one = {k: v[0] for k, v in params.items()}
        if seg[0] == "ping":
            self._json({"ok": True, "time": time.time()})
        elif seg[0] == "meta":
            self._json(d.meta())
        elif seg[0] == "points":
            name = one.get("emb", next(iter(d.embeddings), ""))
            if name not in d.embeddings:
                raise KeyError(name)
            arr = np.ascontiguousarray(d.embeddings[name], dtype="<f4")
            self._send(arr.tobytes(), "application/octet-stream")
        elif seg[0] == "col" and len(seg) >= 2:
            arr = np.ascontiguousarray(d.col(seg[1]), dtype="<f4")
            self._send(arr.tobytes(), "application/octet-stream")
        elif seg[0] == "ticids":
            arr = np.ascontiguousarray(d.ticid, dtype="<u4")
            self._send(arr.tobytes(), "application/octet-stream")
        elif seg[0] == "star" and len(seg) >= 2:
            key = int(seg[1])
            if one.get("by") == "ticid":
                key = d.row_of[key]
            self._json(d.star(key))
        elif seg[0] == "sequence":
            if len(seg) >= 2 and seg[1] == "at":
                t = float(one.get("t", 0.5))
                n = min(int(one.get("n", 12)), 100)
                source = one.get("source", "train")
                seed = int(one["seed"]) if "seed" in one else None
                self._json(d.sequence_at(t, n, source, seed))
            else:
                self._json(d.sequence())
        else:
            raise KeyError(path)

    def do_POST(self) -> None:  # noqa: N802
        url = urlsplit(self.path)
        if url.path != "/api/selection":
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY_BYTES:
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                               "selection too large")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            rows = np.asarray(body.get("rows", []), dtype=np.int64)
            self._json(self.data.selection(rows))
        except (ValueError, TypeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("POST selection failed")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", type=Path,
                   default=Path("/ar0/SPHEREX/bulkphot/tars_v0"))
    p.add_argument("--viz-dir", type=Path, default=None)
    p.add_argument("--ml-dir", type=Path, default=None)
    p.add_argument("--tars", type=Path,
                   default=Path("/ar0/local/TARS/tars_table_2.feather"))
    p.add_argument("--sxviz-url", default=DEFAULT_SXVIZ)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8052)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    data = Data(args.run_dir, args.viz_dir, args.ml_dir, args.tars,
                args.sxviz_url)
    server = ThreadingHTTPServer((args.host, args.port),
                                 partial(Handler, data=data))
    server.daemon_threads = True
    LOGGER.info("serving sx-ml on http://%s:%d/", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
