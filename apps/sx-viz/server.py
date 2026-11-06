"""Serve sx-viz: SPHEREx spectra of TARS rotators.

Static files are served from this directory.  All data come from a viz
build (``drivers/build_viz_products.py`` + ``drivers/fit_viz_seds.py``)
of a bulk photometry run, joined with the TARS table.  Endpoints:

* ``GET /api/ping?delay_ms=``       liveness (optional delay, for tests)
* ``GET /api/meta``                 counts, columns, build info
* ``GET /api/table?...``            server-side filtered, sorted page
* ``GET /api/star/<tic>``           everything one star page needs
* ``GET /api/sed/<tic>?...``        on-demand SED fit with options
* ``GET /api/cutout/<tic>?survey=`` sky cutout (SkyView, disk-cached)
* ``GET /api/spxcut/<tic>/<key>``   SPHEREx cutout PNG (disk-cached)
* ``GET /api/report/<tic>.png``     matplotlib report (disk-cached)
* ``GET /api/tarsvet/<tic>``        TARS vetting JPEG URLs, proxied from
                                    the lcviz app on wh2 (``LCVIZ_URL``)
* ``GET/POST /api/verdicts``        per-star review records

Table filters are query parameters ``min_<col>=`` and ``max_<col>=`` on
any numeric column, ``flag_<col>=0|1`` on boolean columns, ``q=`` (TIC
prefix or exact), ``sort=<col>``, ``desc=1``, ``offset=``, ``limit=``.

Verdict POSTs need the token in ``~/.config/sx-viz/token`` (header
``X-SX-Token``) if that file exists; reads are open.

Usage:
    python server.py --run-dir /ar0/SPHEREX/bulkphot/tars_v0 --port 8051
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import io
import json
import logging
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from sx_phot.sedfit import (MainSequence, build_model_set, fit_star,  # noqa
                            planck_fnu_jy, strip_private)
from sx_phot.sedmodels import load_grid  # noqa: E402
from sx_phot.viz_products import SpectrumStore  # noqa: E402

LOGGER = logging.getLogger("sx-viz")
DEFAULT_TOKEN_FILE = Path.home() / ".config" / "sx-viz" / "token"
VALID_VERDICTS = {"binary", "single", "unsure", "junk", ""}
MAX_NOTES_CHARS = 5000
MAX_BODY_BYTES = 20000
MAX_PAGE = 2000

EMPFIT_COLUMNS = [
    "ticid", "t_single", "teff_emp", "chi2_single", "dof_single",
    "lum_ratio", "dmag_ms", "red_excess", "red_excess_sig", "blue_excess",
    "dbic_binary", "dbic_binary_pct", "dbic_binary_ms", "dbic_binary_ms_pct",
    "teff_bin2", "flux_ratio_binary", "dbic_disk", "dbic_disk_pct",
    "dbic_disk_vs_binary", "teff_disk", "disk_frac_b6", "resid_a1",
    "max_abs_z", "lum_ratio_pct", "red_excess_pct",
]

TARS_COLUMNS = [
    "TICID", "dr3_source_id", "ra", "dec", "parallax", "pmra", "pmdec",
    "teff", "Tmag", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag",
    "BpmRp0", "phot_g_mean_mag_0", "extinction_a0", "ruwe", "non_single_star",
    "adopted_period", "adopted_period_unc", "flag_multiple_periods",
    "flag_possible_binary", "final_n_contams", "flag_one_cvz_sector",
    "flag_doubled_period", "n_secs", "n_sec_ratio", "median_amplitude",
    "sectors", "sector_periods", "sector_amplitudes",
]
TARS_DETAIL_COLUMNS = [
    "sector_sys_scores", "sector_sig_scores", "sector_match_scores",
    "sector_harmonic_scores", "sector_match_probs", "sector_harmonic_probs",
]

LCVIZ_URL = os.environ.get("SX_LCVIZ_URL", "http://lcviz.lgbouma.com:8056")
LCVIZ_TTL_SEC = 6 * 3600

SKYVIEW_SURVEYS = {
    "dss2r": "DSS2 Red", "dss2ir": "DSS2 IR", "dss2b": "DSS2 Blue",
    "wise1": "WISE 3.4", "wise2": "WISE 4.6", "2massj": "2MASS-J",
    "2massk": "2MASS-K",
}
SKYVIEW_MIN_BYTES = 3000  # smaller JPEGs are SkyView's blank "no data" image
SKYVIEW_URL = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
CUTOUT_SIZE_DEG = 0.05
CUTOUT_PIXELS = 300


# ----------------------------------------------------------------------
# Review records (verdicts), ported from cpv-viz
# ----------------------------------------------------------------------

def parse_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a verdict POST body."""
    verdict = str(payload.get("verdict", ""))
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Invalid verdict: {verdict!r}")
    notes = str(payload.get("notes", ""))
    if len(notes) > MAX_NOTES_CHARS:
        raise ValueError("notes too long")
    return {"verdict": verdict, "notes": notes}


class RecordStore:
    """Thread-safe JSON store of ``{ticid: {verdict, notes, updated_utc}}``.

    Mirrors to CSV and appends every change to a history CSV.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.csv_path = path.with_suffix(".csv")
        self.history_path = path.with_name(f"{path.stem}_history.csv")
        self._lock = threading.Lock()

    def load(self) -> dict[str, Any]:
        """Return all records."""
        if not self.path.exists():
            return {}
        with self._lock:
            return json.loads(self.path.read_text() or "{}")

    def save(self, ticid: int, record: dict[str, Any]) -> dict[str, Any]:
        """Insert or clear one record and persist."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._lock:
            data = (json.loads(self.path.read_text() or "{}")
                    if self.path.exists() else {})
            key = str(int(ticid))
            if not record["verdict"] and not record["notes"]:
                data.pop(key, None)
                saved = {"verdict": "", "notes": "", "updated_utc": now}
            else:
                saved = dict(record, updated_utc=now)
                data[key] = saved
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
            os.replace(tmp, self.path)
            with self.csv_path.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["ticid", "verdict", "notes", "updated_utc"])
                for k in sorted(data, key=int):
                    r = data[k]
                    w.writerow([k, r["verdict"], r["notes"], r["updated_utc"]])
            new = not self.history_path.exists()
            with self.history_path.open("a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["ticid", "verdict", "notes", "updated_utc"])
                w.writerow([key, saved["verdict"], saved["notes"], now])
        return saved


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    """Convert numpy / NaN values to JSON-serializable Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is pd.NA or value is None:
        return None
    return value


def _array_json(arr: np.ndarray, ndigits: Optional[int] = None) -> list:
    """Array to list with NaN -> None (and optional rounding)."""
    a = np.asarray(arr)
    if a.dtype.kind == "b":
        return a.tolist()
    if a.dtype.kind in "iu":
        return a.tolist()
    a = a.astype(float)
    if ndigits is not None:
        a = np.round(a, ndigits)
    out = a.tolist()
    bad = ~np.isfinite(a)
    if bad.any():
        for i in np.flatnonzero(bad):
            out[i] = None
    return out


class Data:
    """In-memory table plus per-star accessors.

    Args:
        run_dir: Bulk run directory.
        viz_dir: Viz build directory (default ``<run>/viz``).
        tars_path: TARS table (feather).
    """

    def __init__(self, run_dir: Path, viz_dir: Optional[Path],
                 tars_path: Path) -> None:
        self.run_dir = Path(run_dir)
        self.viz_dir = Path(viz_dir) if viz_dir else self.run_dir / "viz"
        t0 = time.time()
        summ = pd.read_parquet(self.viz_dir / "summary.parquet")
        sed_path = self.viz_dir / "sed.parquet"
        if sed_path.exists():
            sed = pd.read_parquet(sed_path)
            sed = sed.drop(columns=[c for c in sed.columns
                                    if c in ("error",)], errors="ignore")
            summ = summ.merge(sed, on="ticid", how="left")
            self.has_sed = True
        else:
            self.has_sed = False
        # Empirical-template fits (apps/sx-ml), prefixed to avoid clashes
        # with the model-atmosphere columns.
        emp_path = self.run_dir / "ml" / "empfit.parquet"
        if emp_path.exists():
            emp = pd.read_parquet(emp_path, columns=[
                c for c in EMPFIT_COLUMNS
                if c in pq.ParquetFile(emp_path).schema_arrow.names])
            emp = emp.rename(columns={c: f"emp_{c}" for c in emp.columns
                                      if c != "ticid"})
            if "emp_chi2_single" in emp and "emp_dof_single" in emp:
                emp["emp_chi2_red"] = (emp["emp_chi2_single"]
                                       / emp["emp_dof_single"])
                emp = emp.drop(columns=["emp_dof_single"])
            summ = summ.merge(emp, on="ticid", how="left")
        tars = pd.read_feather(tars_path)
        tars = tars.rename(columns={"TICID": "ticid"})
        tars = tars.drop(columns=[c for c in ("ra", "dec") if c in tars])
        self.tars_detail = tars.set_index("ticid")[
            [c for c in TARS_DETAIL_COLUMNS if c in tars.columns]]
        keep = [c for c in TARS_COLUMNS if c in tars.columns
                and c not in ("TICID", "ra", "dec")] + ["ticid"]
        table = summ.merge(tars[keep], on="ticid", how="left")
        for col in ("flag_possible_binary",):
            if col in table:
                table[col] = table[col].astype("boolean").fillna(False)
        table = table.reset_index(drop=True)
        first = [c for c in ("ticid", "ra", "dec", "Tmag", "teff")
                 if c in table]
        table = table[first + [c for c in table.columns if c not in first]]
        self.table = table
        self.row_of = {int(t): i for i, t in
                       enumerate(table["ticid"].to_numpy())}
        self.tic_str = table["ticid"].astype(str).to_numpy().astype(str)
        self.numeric = [c for c in table.columns
                        if table[c].dtype.kind in "iuf"]
        self.boolean = [c for c in table.columns
                        if str(table[c].dtype) in ("bool", "boolean")]
        self._cols: dict[str, np.ndarray] = {}
        self._order: dict[str, np.ndarray] = {}
        self.spectra = SpectrumStore(self.viz_dir, table)
        self.binned_flux = np.load(self.viz_dir / "binned_flux.npy",
                                   mmap_mode="r")
        self.binned_err = np.load(self.viz_dir / "binned_err.npy",
                                  mmap_mode="r")
        self.binned_n = np.load(self.viz_dir / "binned_n.npy", mmap_mode="r")
        self.binned_model = np.load(self.viz_dir / "binned_model.npy",
                                    mmap_mode="r")
        self.bin_edges = np.load(self.viz_dir / "bin_edges.npy")
        self.bin_centers = np.sqrt(self.bin_edges[:-1] * self.bin_edges[1:])
        binned_tic = np.load(self.viz_dir / "binned_ticid.npy")
        self.binned_row = {int(t): i for i, t in enumerate(binned_tic)}
        self.config = json.loads((self.run_dir / "config.json").read_text())
        meta_path = self.viz_dir / "viz.json"
        self.viz_meta = (json.loads(meta_path.read_text())
                         if meta_path.exists() else {})
        self.index = pd.read_parquet(
            self.run_dir / "index.parquet",
            columns=["image_key", "path", "obs_id", "det"]).set_index(
                "image_key")
        self.ms = MainSequence()
        self._models_lock = threading.Lock()
        self._models_cache: dict[tuple, Any] = {}
        # Warm the default model set so the first star page is fast.
        self.models(None, 0.0, ("sphinx", "ck04"))
        # Open every spectra file in the background (first opens cost
        # ~1 s each cold; the handles are kept for the server lifetime).
        threading.Thread(target=self.spectra.prewarm, daemon=True).start()
        LOGGER.info("loaded %d stars (%d columns) in %.1f s", len(table),
                    table.shape[1], time.time() - t0)

    # -- table queries ---------------------------------------------------

    def col(self, name: str) -> np.ndarray:
        """Float view of a numeric or boolean column (cached)."""
        if name not in self._cols:
            s = self.table[name]
            if str(s.dtype) == "boolean":
                arr = s.fillna(False).to_numpy(dtype=float)
            else:
                arr = s.to_numpy(dtype=float)
            self._cols[name] = arr
        return self._cols[name]

    def order(self, name: str, desc: bool) -> np.ndarray:
        """Argsort of a column (NaN last; cached)."""
        key = (name, desc)
        if key not in self._order:
            arr = self.col(name)
            fill = -np.inf if desc else np.inf
            vals = np.where(np.isfinite(arr), arr, fill)
            if desc:
                vals = -vals
            self._order[key] = np.argsort(vals, kind="stable")
        return self._order[key]

    def query(self, params: dict[str, list[str]]) -> dict:
        """Filter, sort and page the table."""
        n = len(self.table)
        mask = np.ones(n, dtype=bool)
        active = {}
        for key, vals in params.items():
            val = vals[-1] if vals else ""
            if val == "":
                continue
            if key.startswith("min_") and key[4:] in self.numeric:
                col = key[4:]
                x = self.col(col)
                mask &= np.isfinite(x) & (x >= float(val))
                active[key] = float(val)
            elif key.startswith("max_") and key[4:] in self.numeric:
                col = key[4:]
                x = self.col(col)
                mask &= np.isfinite(x) & (x <= float(val))
                active[key] = float(val)
            elif key.startswith("flag_") and key[5:] in self.boolean:
                col = key[5:]
                x = self.col(col)
                mask &= (x > 0.5) if val in ("1", "true") else (x < 0.5)
                active[key] = val
            elif key.startswith("has_") and key[4:] in self.numeric:
                x = self.col(key[4:])
                has = np.isfinite(x)
                mask &= has if val in ("1", "true") else ~has
                active[key] = val
        q = (params.get("q", [""])[-1] or "").strip()
        if q:
            digits = "".join(ch for ch in q if ch.isdigit())
            if digits:
                mask &= np.char.startswith(self.tic_str, digits)
            else:
                mask &= False
        sort = params.get("sort", ["ticid"])[-1]
        if sort not in self.numeric:
            sort = "ticid"
        desc = params.get("desc", ["0"])[-1] in ("1", "true")
        order = self.order(sort, desc)
        sel = order[mask[order]]
        total = int(sel.size)
        offset = max(0, int(params.get("offset", ["0"])[-1] or 0))
        limit = min(MAX_PAGE, max(1, int(params.get("limit", ["200"])[-1]
                                         or 200)))
        cols = params.get("cols", [""])[-1]
        if params.get("ids_only", ["0"])[-1] in ("1", "true"):
            ids = self.table["ticid"].to_numpy()[sel[offset:offset + limit]]
            return {"total": total, "ids": ids.tolist(), "sort": sort,
                    "desc": desc}
        want = [c for c in cols.split(",") if c in self.table.columns] \
            if cols else list(self.table.columns)
        if "ticid" not in want:
            want.insert(0, "ticid")
        page = self.table.iloc[sel[offset:offset + limit]][want]
        rows = [_json_safe(dict(zip(want, r)))
                for r in page.itertuples(index=False, name=None)]
        return {"total": total, "offset": offset, "limit": limit,
                "sort": sort, "desc": desc, "rows": rows, "columns": want,
                "active": active}

    # -- star accessors ----------------------------------------------------

    def star_row(self, ticid: int) -> dict:
        """Table row of one star as a JSON-safe dict."""
        i = self.row_of[int(ticid)]
        row = self.table.iloc[i].to_dict()
        if int(ticid) in self.tars_detail.index:
            row.update(self.tars_detail.loc[int(ticid)].to_dict())
        return _json_safe(row)

    def spectrum_json(self, ticid: int) -> dict:
        """Spectrum arrays of one star."""
        df = self.spectra.read(ticid)
        out = {}
        for col in ("wavelength_um", "bandwidth_um", "flux_jy",
                    "flux_err_jy", "model_flux", "bkg_jy", "flux_raw_jy",
                    "zodi_jy"):
            out[col] = _array_json(df[col].to_numpy(), 7)
        out["masked"] = df["masked"].tolist()
        out["mask_bits"] = (df["mask_bits"].tolist() if "mask_bits" in df
                            else [0] * len(df))
        out["fit_mask"] = df["fit_mask"].tolist()
        out["tjd_avg"] = _array_json(df["tjd_avg"].to_numpy(), 5)
        out["image_key"] = [str(k) for k in df["image_key"].tolist()]
        out["flag_or"] = df["flag_or"].tolist()
        out["n_bad_pix"] = df["n_bad_pix"].tolist()
        out["bkg_mode"] = df["bkg_mode"].tolist()
        out["x"] = _array_json(df["x"].to_numpy(), 1)
        out["y"] = _array_json(df["y"].to_numpy(), 1)
        return out

    def binned_json(self, ticid: int) -> dict:
        """Binned spectrum of one star."""
        i = self.binned_row[int(ticid)]
        return {
            "lam": _array_json(self.bin_centers, 4),
            "flux": _array_json(self.binned_flux[i], 7),
            "err": _array_json(self.binned_err[i], 7),
            "n": self.binned_n[i].tolist(),
            "model": _array_json(self.binned_model[i], 7),
        }

    def models(self, logg: Optional[float], mh: float, grids: tuple):
        """Cached model set for the given options."""
        key = (logg, mh, grids)
        with self._models_lock:
            if key not in self._models_cache:
                if len(self._models_cache) > 16:
                    self._models_cache.clear()
                self._models_cache[key] = build_model_set(
                    logg=logg, mh=mh, grids=grids, main_sequence=self.ms)
            return self._models_cache[key]

    def fine_model(self, grid: str, teff: float, logg: float, mh: float,
                   scale: float) -> dict:
        """Model curve on the fine grid, scaled."""
        g = load_grid(grid)
        f = g.model_at(teff, logg, mh, fine=True) * scale
        return {"lam": _array_json(g.fine_lam_um, 4),
                "flux": _array_json(f, 8)}

    def sed_fit(self, ticid: int, params: dict[str, list[str]]) -> dict:
        """On-demand SED fit with query options."""
        def _f(name, default):
            v = params.get(name, [""])[-1]
            return float(v) if v not in ("", None) else default

        logg_raw = params.get("logg", [""])[-1]
        logg = float(logg_raw) if logg_raw not in ("", "auto") else None
        mh = _f("mh", 0.0)
        grids_raw = params.get("grids", ["sphinx,ck04"])[-1]
        grids = tuple(g for g in grids_raw.split(",") if g in
                      ("sphinx", "ck04")) or ("sphinx", "ck04")
        blue_max = _f("blue_max_um", 1.7)
        err_floor = _f("err_floor", 0.02)
        lam_min = _f("lam_min_um", 0.0)
        lam_max = _f("lam_max_um", 10.0)
        binary = params.get("binary", ["1"])[-1] not in ("0", "false")
        models = self.models(logg, mh, grids)
        i = self.binned_row[int(ticid)]
        row = self.table.iloc[self.row_of[int(ticid)]]
        plx = row.get("parallax", np.nan)
        res = fit_star(
            np.asarray(self.binned_flux[i], float),
            np.asarray(self.binned_err[i], float), models,
            parallax_mas=float(plx) if np.isfinite(plx) else None,
            blue_max_um=blue_max, err_floor=err_floor, do_binary=binary,
            main_sequence=self.ms, lam_min_um=lam_min, lam_max_um=lam_max,
        )
        curves = {}
        if "teff_full" in res:
            curves["full"] = self.fine_model(
                res["grid_full"], res["teff_full"], res["logg_full"],
                res["mh_full"], res["scale_full"])
        if "teff_blue" in res:
            k = int(np.argmin(np.abs(models.teff - res["teff_blue"])))
            curves["blue"] = self.fine_model(
                res["grid_blue"], res["teff_blue"], float(models.logg[k]),
                float(models.mh[k]), res["scale_blue"])
        if "teff_bin1" in res:
            k1 = int(np.argmin(np.abs(models.teff - res["teff_bin1"])))
            k2 = int(np.argmin(np.abs(models.teff - res["teff_bin2"])))
            c1 = self.fine_model(str(models.grid[k1]), res["teff_bin1"],
                                 float(models.logg[k1]), float(models.mh[k1]),
                                 res["scale_bin1"])
            c2 = self.fine_model(str(models.grid[k2]), res["teff_bin2"],
                                 float(models.logg[k2]), float(models.mh[k2]),
                                 res["scale_bin2"])
            curves["bin1"] = c1
            curves["bin2"] = c2
            tot = np.array(c1["flux"], float) + np.array(c2["flux"], float)
            curves["binary"] = {"lam": c1["lam"], "flux": _array_json(tot, 8)}
        if "teff_disk" in res:
            k1 = int(np.argmin(np.abs(models.teff - res["teff_disk_star"])))
            c1 = self.fine_model(str(models.grid[k1]), res["teff_disk_star"],
                                 float(models.logg[k1]), float(models.mh[k1]),
                                 res["scale_disk_star"])
            lam_fine = np.array(c1["lam"], float)
            bb = planck_fnu_jy(lam_fine, res["teff_disk"]) * res["scale_disk"]
            star_f = np.array([v if v is not None else np.nan
                               for v in c1["flux"]], float)
            curves["disk_star"] = c1
            curves["disk_bb"] = {"lam": c1["lam"], "flux": _array_json(bb, 8)}
            curves["disk"] = {"lam": c1["lam"],
                              "flux": _array_json(star_f + bb, 8)}
        binned_models = {k: _array_json(res[f"_model_{k}"], 8)
                         for k in ("full", "blue", "bin1", "bin2",
                                   "disk_star", "disk_bb")
                         if f"_model_{k}" in res}
        return {"fit": _json_safe(strip_private(res)), "curves": curves,
                "binned_models": binned_models,
                "options": {"logg": logg_raw or "auto", "mh": mh,
                            "grids": list(grids), "blue_max_um": blue_max,
                            "err_floor": err_floor, "lam_min_um": lam_min,
                            "lam_max_um": lam_max, "binary": binary}}

    # -- cutouts -----------------------------------------------------------

    def sky_cutout(self, ticid: int, survey: str) -> Optional[bytes]:
        """Fetch (and cache) a SkyView JPEG cutout."""
        import requests

        if survey not in SKYVIEW_SURVEYS:
            raise ValueError("unknown survey")
        cache = self.viz_dir / "cutouts" / survey / f"{int(ticid)}.jpg"
        if cache.exists():
            return cache.read_bytes()
        row = self.table.iloc[self.row_of[int(ticid)]]
        params = {
            "Position": f"{row['ra']:.6f},{row['dec']:.6f}",
            "Survey": SKYVIEW_SURVEYS[survey],
            "Size": f"{CUTOUT_SIZE_DEG}", "Pixels": f"{CUTOUT_PIXELS}",
            "Return": "JPEG", "Projection": "Tan", "Sampler": "LI",
        }
        try:
            resp = requests.get(SKYVIEW_URL, params=params, timeout=45)
        except requests.RequestException as exc:
            LOGGER.warning("skyview %s TIC %d: %s", survey, ticid, exc)
            return None
        if (resp.status_code != 200
                or not resp.content.startswith(b"\xff\xd8")
                or len(resp.content) < SKYVIEW_MIN_BYTES):
            LOGGER.warning("skyview %s TIC %d: HTTP %d (%d bytes)", survey,
                           ticid, resp.status_code, len(resp.content))
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(resp.content)
        os.replace(tmp, cache)
        return resp.content

    def spherex_cutout(self, ticid: int, image_key: int) -> Optional[bytes]:
        """Render (and cache) a SPHEREx cutout PNG around the star."""
        from sx_phot.viz_report import render_spherex_cutout

        cache = (self.viz_dir / "cutouts" / "spherex"
                 / f"{int(ticid) % 1000:03d}"
                 / f"{int(ticid)}_{int(image_key)}.png")
        if cache.exists():
            return cache.read_bytes()
        if int(image_key) not in self.index.index:
            return None
        path = Path(self.index.loc[int(image_key), "path"])
        if not path.exists():
            return None
        df = self.spectra.read(ticid)
        hit = df[df["image_key"] == int(image_key)]
        if hit.empty:
            return None
        r = hit.iloc[0]
        phot = self.config["phot"]
        png = render_spherex_cutout(
            path, float(r["x"]), float(r["y"]), phot["aperture_radius"],
            phot["annulus_r_in"], phot["annulus_r_out"],
            title=(f"{r['wavelength_um']:.3f} um  TJD {r['tjd_avg']:.2f}\n"
                   f"{r['flux_jy'] * 1e3:.3f} mJy  "
                   f"{'masked' if r['masked'] else 'ok'}"),
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(png)
        os.replace(tmp, cache)
        return png

    def report(self, ticid: int, refresh: bool = False) -> bytes:
        """Render (and cache) the matplotlib star report PNG."""
        from sx_phot.viz_report import plot_star_report

        cache = (self.viz_dir / "reports" / f"{int(ticid) % 1000:03d}"
                 / f"{int(ticid)}.png")
        if cache.exists() and not refresh:
            return cache.read_bytes()
        df = self.spectra.read(ticid)
        row = self.star_row(ticid)
        sed = None
        if self.has_sed and row.get("teff_full") is not None:
            sed = self.sed_fit(ticid, {})
        png = plot_star_report(df, row, self.binned_json(ticid), sed)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(png)
        os.replace(tmp, cache)
        return png

    def tars_vetting(self, ticid: int) -> dict:
        """TARS vetting JPEG URLs for a star, proxied from lcviz (cached).

        Returns a dict with ``images`` (absolute URLs), ``star_url`` (the
        lcviz star page) and ``error`` if lcviz could not be reached.
        """
        import requests

        cache = getattr(self, "_tarsvet_cache", None)
        if cache is None:
            cache = self._tarsvet_cache = {}
        hit = cache.get(int(ticid))
        if hit and time.time() - hit[0] < LCVIZ_TTL_SEC:
            return hit[1]
        out = {"ticid": int(ticid), "images": [],
               "star_url": f"{LCVIZ_URL}/star.html?ticid={int(ticid)}"}
        try:
            resp = requests.get(f"{LCVIZ_URL}/api/star",
                                params={"ticid": int(ticid)}, timeout=10)
            if resp.status_code == 200:
                payload = resp.json()
                for item in payload.get("img_files", []):
                    rel = item.get("path", "")
                    sector = None
                    if "_s" in rel:
                        try:
                            sector = int(rel.split("_s")[1][:4])
                        except ValueError:
                            sector = None
                    out["images"].append({
                        "url": LCVIZ_URL + item["url"], "sector": sector,
                        "name": Path(rel).name,
                    })
                out["images"].sort(key=lambda d: (d["sector"] is None,
                                                  d["sector"]))
            elif resp.status_code == 404:
                out["error"] = "not in the lcviz catalog"
            else:
                out["error"] = f"lcviz HTTP {resp.status_code}"
        except (requests.RequestException, ValueError) as exc:
            out["error"] = f"lcviz unreachable: {exc.__class__.__name__}"
        if "error" not in out or len(cache) < 100000:
            cache[int(ticid)] = (time.time(), out)
        return out

    def default_cutout_keys(self, ticid: int, n: int = 6) -> list[dict]:
        """Pick unmasked images spread over wavelength for cutouts."""
        df = self.spectra.read(ticid)
        good = df[~df["masked"] & np.isfinite(df["flux_jy"])]
        if good.empty:
            good = df
        targets = np.array([0.9, 1.35, 2.0, 3.0, 4.1, 4.75])[:n]
        picks = []
        used = set()
        for t in targets:
            k = int(np.argmin(np.abs(good["wavelength_um"].to_numpy() - t)))
            r = good.iloc[k]
            key = int(r["image_key"])
            if key in used:
                continue
            used.add(key)
            picks.append({"image_key": str(key),
                          "wavelength_um": round(float(r["wavelength_um"]), 3),
                          "tjd": round(float(r["tjd_avg"]), 2)})
        return picks


# ----------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------

class Handler(SimpleHTTPRequestHandler):
    """Static files plus the JSON API."""

    data: Data
    verdicts: RecordStore
    token: Optional[str]

    def __init__(self, *args, data: Data, verdicts: RecordStore,
                 token: Optional[str], **kwargs) -> None:
        self.data = data
        self.verdicts = verdicts
        self.token = token
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102
        LOGGER.info("%s " + fmt, self.address_string(), *args)

    def end_headers(self) -> None:  # noqa: D102
        # Static files (html/js/css) must revalidate on every load so that
        # a browser never pairs a cached star.html with a newer star.js.
        # The vendored plotly bundle is large and immutable.
        path = urlsplit(self.path).path
        if not path.startswith("/api/"):
            if path.startswith("/vendor/"):
                self.send_header("Cache-Control", "public, max-age=604800")
            else:
                self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    # -- helpers -------------------------------------------------------------

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, ctype: str,
                    cache_seconds: int = 86400) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", f"public, max-age={cache_seconds}")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    # -- routing -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parts = urlsplit(self.path)
        path = parts.path
        params = parse_qs(parts.query)
        if not path.startswith("/api/"):
            if path == "/":
                self.path = "/index.html"
            return super().do_GET()
        try:
            self._api_get(path, params)
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("client went away during %s", self.path)
        except KeyError as exc:
            self._error(HTTPStatus.NOT_FOUND, f"not found: {exc}")
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("GET %s failed", self.path)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))

    def _api_get(self, path: str, params: dict[str, list[str]]) -> None:
        d = self.data
        seg = path.split("/")[2:]
        if seg[0] == "ping":
            delay = min(10000, int(params.get("delay_ms", ["0"])[-1] or 0))
            time.sleep(delay / 1000.0)
            self._send_json({"ok": True, "delay_ms": delay})
        elif seg[0] == "meta":
            self._send_json({
                "n_stars": len(d.table), "numeric": d.numeric,
                "boolean": d.boolean, "columns": list(d.table.columns),
                "has_sed": d.has_sed, "viz": d.viz_meta,
                "run_dir": str(d.run_dir), "surveys": SKYVIEW_SURVEYS,
                "phot": d.config.get("phot", {}),
                "lcviz_url": LCVIZ_URL,
            })
        elif seg[0] == "table":
            self._send_json(d.query(params))
        elif seg[0] == "star" and len(seg) >= 2:
            tic = int(seg[1])
            if tic not in d.row_of:
                raise KeyError(f"TIC {tic}")
            payload = {
                "ticid": tic, "row": d.star_row(tic),
                "spectrum": d.spectrum_json(tic), "binned": d.binned_json(tic),
                "cutouts": d.default_cutout_keys(tic),
                "verdict": self.verdicts.load().get(str(tic)),
            }
            if d.has_sed and payload["row"].get("teff_full") is not None:
                payload["sed"] = d.sed_fit(tic, {})
            self._send_json(payload)
        elif seg[0] == "sed" and len(seg) >= 2:
            tic = int(seg[1])
            if tic not in d.binned_row:
                raise KeyError(f"TIC {tic}")
            self._send_json(d.sed_fit(tic, params))
        elif seg[0] == "cutout" and len(seg) >= 2:
            tic = int(seg[1])
            survey = params.get("survey", ["dss2r"])[-1]
            body = d.sky_cutout(tic, survey)
            if body is None:
                self._error(HTTPStatus.BAD_GATEWAY, "cutout unavailable")
            else:
                self._send_bytes(body, "image/jpeg")
        elif seg[0] == "spxcut" and len(seg) >= 3:
            tic = int(seg[1])
            key = int(seg[2].split(".")[0])
            body = d.spherex_cutout(tic, key)
            if body is None:
                self._error(HTTPStatus.NOT_FOUND, "image not available")
            else:
                self._send_bytes(body, "image/png")
        elif seg[0] == "report" and len(seg) >= 2:
            tic = int(seg[1].split(".")[0])
            refresh = params.get("refresh", ["0"])[-1] == "1"
            self._send_bytes(d.report(tic, refresh), "image/png",
                             cache_seconds=0)
        elif seg[0] == "tarsvet" and len(seg) >= 2:
            tic = int(seg[1])
            if tic not in d.row_of:
                raise KeyError(f"TIC {tic}")
            self._send_json(d.tars_vetting(tic))
        elif seg[0] == "verdicts":
            self._send_json(self.verdicts.load())
        else:
            raise KeyError(path)

    def do_POST(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path != "/api/verdicts":
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        if self.token is not None:
            sent = self.headers.get("X-SX-Token", "")
            if not hmac.compare_digest(sent, self.token):
                self._error(HTTPStatus.UNAUTHORIZED, "bad token")
                return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._error(HTTPStatus.BAD_REQUEST, "bad body length")
            return
        try:
            payload = json.loads(self.rfile.read(length).decode())
            tic = int(payload["ticid"])
            record = parse_verdict(payload)
        except (ValueError, KeyError, TypeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, f"invalid payload: {exc}")
            return
        saved = self.verdicts.save(tic, record)
        self._send_json({"ticid": tic, **saved})


def load_token(path: Path) -> Optional[str]:
    """Read the write token, or None if absent."""
    if path.exists():
        tok = path.read_text().strip()
        return tok or None
    return None


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", default="/ar0/SPHEREX/bulkphot/tars_v0")
    parser.add_argument("--viz-dir", default=None)
    parser.add_argument("--tars",
                        default="/ar0/local/TARS/tars_table_2.feather")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--verdicts", default=str(APP_DIR / "data"
                                                  / "verdicts.json"))
    parser.add_argument("--token-file", default=str(DEFAULT_TOKEN_FILE))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    data = Data(Path(args.run_dir), Path(args.viz_dir) if args.viz_dir
                else None, Path(args.tars))
    verdicts = RecordStore(Path(args.verdicts))
    token = load_token(Path(args.token_file))
    if token is None:
        LOGGER.warning("no token file at %s: verdict writes are open",
                       args.token_file)
    handler = partial(Handler, data=data, verdicts=verdicts, token=token)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    LOGGER.info("serving on http://%s:%d/", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
