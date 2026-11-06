#!/usr/bin/env python
"""Compare the empirical-template fits with the model-atmosphere fits.

Reads ``<ml>/empfit.parquet`` (``drivers/fit_empirical.py``),
``<viz>/sed.parquet`` (``drivers/fit_viz_seds.py``) and the TARS table,
and writes figures, CSV candidate lists and ``summary.txt`` to
``--out-dir`` (default ``results/sx_ml_analysis``):

* ``chi2_calibration.png``: reduced chi2 of the single fits, and the
  fraction of stars above delta-BIC thresholds as a function of Teff.
* ``binary_enrichment.png``: fraction of Gaia RUWE > 1.4 and
  non-single-star flagged stars among the top-x % of each binary
  statistic (does the statistic find astrometric binaries?).
* ``red_excess_vs_teff.png``: median red excess against Teff for both
  methods (the band systematics the templates absorb).
* ``binary_vs_period.png``: top-decile fraction of the binary evidence
  at fixed temperature against rotation period.
* ``binary_fraction_vs_period.png``: fraction of stars more than
  0.375 mag above the main sequence (SPHEREx flux at the Gaia distance,
  Gaia CMD), RUWE > 1.4 and NSS against rotation period, by Teff.
* ``excess_vs_tmag.png``: median band-6 residual against Tmag at fixed t.
* ``binary_candidates.csv``, ``disk_candidates.csv``: ranked lists.

Usage:
    python scripts/analyze_empfit.py --run-dir /ar0/SPHEREX/bulkphot/tars_v0
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

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sx_phot.specml import load_catalog, main_sequence_offset  # noqa: E402

DEFAULT_TARS = Path("/ar0/local/TARS/tars_table_2.feather")


def _frac_above(x, thresholds):
    x = x[np.isfinite(x)]
    return [(x > th).mean() for th in thresholds]


def chi2_calibration(df, out, lines):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axs[0]
    bins = np.logspace(-0.7, 2.5, 60)
    for col, dof, lab in (("chi2_single", "dof_single", "empirical templates"),
                          ("chi2_full", "dof_full", "model atmospheres")):
        v = (df[col] / df[dof]).to_numpy(dtype=float)
        v = v[np.isfinite(v) & (v > 0)]
        ax.hist(v, bins=bins, histtype="step", label=f"{lab} (median {np.median(v):.2f})")
        lines.append(f"{lab}: median chi2/dof {np.median(v):.2f}, "
                     f"90% {np.percentile(v, 90):.1f}")
    ax.set_xscale("log")
    ax.set_xlabel("chi2 / dof (single star)")
    ax.set_ylabel("stars")
    ax.legend(fontsize=7)
    thresholds = [10, 100, 1000]
    for ax, col, lab in ((axs[1], "dbic_binary", "empirical: dBIC binary"),
                         (axs[2], "sed_dbic_binary", "model: dBIC binary")):
        teff = df["teff_emp"].to_numpy(dtype=float)
        v = df[col].to_numpy(dtype=float)
        edges = np.arange(3000, 7600, 200)
        for th in thresholds:
            fr = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (teff >= lo) & (teff < hi) & np.isfinite(v)
                fr.append((v[m] > th).mean() if m.sum() > 50 else np.nan)
            ax.plot(0.5 * (edges[1:] + edges[:-1]), fr, label=f"dBIC > {th}")
            lines.append(f"{lab} > {th}: {np.nanmean(v > th):.3f} of stars")
        ax.set_xlabel("Teff_emp [K]")
        ax.set_ylabel("fraction selected")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1)
        ax.set_title(lab, fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "chi2_calibration.png", dpi=180)
    plt.close(fig)


def binary_enrichment(df, out, lines):
    stats = {
        "emp dBIC binary (shape)": "dbic_binary_pct",
        "emp dBIC binary (Gaia dist.)": "dbic_binary_ms_pct",
        "emp overluminosity": "lum_ratio_pct",
        "emp red excess": "red_excess_pct",
        "emp chi2": "chi2_single_pct",
        "model dBIC binary": "sed_dbic_binary_pct",
        "CMD offset (bright)": "dms_pct",
    }
    ruwe = df["ruwe"].to_numpy(dtype=float) > 1.4
    nss = df["nss"].to_numpy(dtype=float) > 0
    base_r, base_n = ruwe.mean(), nss.mean()
    tops = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for lab, col in stats.items():
        if col not in df:
            continue
        p = df[col].to_numpy(dtype=float)
        fr_r, fr_n = [], []
        for top in tops:
            m = np.isfinite(p) & (p >= 1 - top)
            fr_r.append(ruwe[m].mean() if m.sum() else np.nan)
            fr_n.append(nss[m].mean() if m.sum() else np.nan)
        axs[0].plot(tops, fr_r, "o-", ms=3, label=lab)
        axs[1].plot(tops, fr_n, "o-", ms=3, label=lab)
        lines.append(f"{lab}: RUWE>1.4 fraction in top 10% / 1%: "
                     f"{fr_r[2]:.3f} / {fr_r[5]:.3f} (all stars {base_r:.3f});"
                     f" NSS>0: {fr_n[2]:.3f} / {fr_n[5]:.3f} (all {base_n:.3f})")
    for ax, base, lab in ((axs[0], base_r, "fraction with RUWE > 1.4"),
                          (axs[1], base_n, "fraction with Gaia NSS flag")):
        ax.axhline(base, color="k", ls=":", lw=0.8, label="all stars")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("top fraction of the statistic (percentile at fixed t)")
        ax.set_ylabel(lab)
    axs[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "binary_enrichment.png", dpi=180)
    plt.close(fig)


def red_excess_vs_teff(df, out, lines):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    teff = df["teff_emp"].to_numpy(dtype=float)
    edges = np.arange(3000, 7600, 100)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for ax, cols, title in (
            (axs[0], [("sed_red_excess", "model: red excess (> 2.4 um)"),
                      ("sed_excess_b6", "model: band 6 excess")],
             "model atmospheres"),
            (axs[1], [("red_excess", "empirical: red excess"),
                      ("excess_b6", "empirical: band 6"),
                      ("excess_b1", "empirical: band 1")],
             "empirical templates")):
        for col, lab in cols:
            if col not in df:
                continue
            v = df[col].to_numpy(dtype=float)
            med, p16, p84 = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (teff >= lo) & (teff < hi) & np.isfinite(v)
                if m.sum() < 50:
                    med.append(np.nan); p16.append(np.nan); p84.append(np.nan)
                    continue
                q = np.percentile(v[m], [16, 50, 84])
                p16.append(q[0]); med.append(q[1]); p84.append(q[2])
            med, p16, p84 = map(np.array, (med, p16, p84))
            line, = ax.plot(mid, 100 * med, label=lab)
            ax.fill_between(mid, 100 * p16, 100 * p84, color=line.get_color(),
                            alpha=0.15)
            ok = np.isfinite(med)
            lines.append(f"{lab}: median over Teff bins of |median excess| "
                         f"{100 * np.nanmedian(np.abs(med)):.2f}%, 16-84% width "
                         f"{100 * np.nanmedian(p84 - p16):.2f}%")
            del ok
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("Teff_emp [K]")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("excess over single-star model [%] (median, 16-84%)")
    axs[0].set_ylim(-15, 15)
    fig.tight_layout()
    fig.savefig(out / "red_excess_vs_teff.png", dpi=180)
    plt.close(fig)


def binary_vs_period(df, out, lines):
    per = df["period"].to_numpy(dtype=float)
    ok = np.isfinite(per) & (per > 0)
    edges = np.logspace(np.log10(0.2), np.log10(40), 12)
    mid = np.sqrt(edges[1:] * edges[:-1])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for col, lab in (("dbic_binary_pct", "emp dBIC binary (shape)"),
                     ("dbic_binary_ms_pct", "emp dBIC binary (Gaia dist.)"),
                     ("lum_ratio_pct", "emp overluminosity"),
                     ("sed_dbic_binary_pct", "model dBIC binary")):
        if col not in df:
            continue
        p = df[col].to_numpy(dtype=float)
        fr, err = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = ok & (per >= lo) & (per < hi) & np.isfinite(p)
            n = m.sum()
            f = (p[m] >= 0.9).mean() if n > 100 else np.nan
            fr.append(f)
            err.append(np.sqrt(f * (1 - f) / n) if n > 100 else np.nan)
        ax.errorbar(mid, fr, yerr=err, fmt="o-", ms=3, label=lab)
        lines.append(f"{lab}: top-decile fraction at P<1 d {np.nanmean(np.array(fr)[mid < 1]):.3f}, "
                     f"P>10 d {np.nanmean(np.array(fr)[mid > 10]):.3f}")
    ax.axhline(0.1, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("TARS rotation period [d]")
    ax.set_ylabel("fraction in the top decile at fixed t")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "binary_vs_period.png", dpi=180)
    plt.close(fig)


def binary_fraction_vs_period(df, out, lines):
    """Overluminous fraction against rotation period.

    A star is called overluminous when its SPHEREx flux at the Gaia
    distance exceeds the main-sequence template by more than 0.375 mag
    (flux ratio 1.41; an equal-mass pair gives 0.753 mag).  The same
    threshold on the Gaia CMD offset, RUWE > 1.4 and the NSS flag are
    shown for comparison.  Clean stars only: no TESS contaminants, at
    least 60 bins, median S/N >= 30, A0 <= 0.5, not at the sequence ends.
    """
    thr = -0.375
    per = df["period"].to_numpy(dtype=float)
    t = df["t_single"].to_numpy(dtype=float)
    clean = (np.isfinite(per) & (per > 0)
             & (df["n_contams"].fillna(0).to_numpy() == 0)
             & (df["n_bins_used"].to_numpy(dtype=float) >= 60)
             & (df["med_snr"].to_numpy(dtype=float) >= 30)
             & (df["a0"].fillna(9).to_numpy(dtype=float) <= 0.5)
             & (df["plx"].to_numpy(dtype=float) > 0)
             & (t > 0.003) & (t < 0.997))
    teff = df["teff_emp"].to_numpy(dtype=float)
    dmag = df["dmag_ms"].to_numpy(dtype=float)
    dms = df["dms"].to_numpy(dtype=float)
    ruwe = df["ruwe"].to_numpy(dtype=float)
    nss = df["nss"].fillna(0).to_numpy(dtype=float)
    edges = np.logspace(np.log10(0.2), np.log10(40), 12)
    mid = np.sqrt(edges[1:] * edges[:-1])

    def fraction(flag, sel):
        fr, err = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = sel & (per >= lo) & (per < hi) & np.isfinite(flag)
            n = m.sum()
            if n < 200:
                fr.append(np.nan)
                err.append(np.nan)
                continue
            f = flag[m].mean()
            fr.append(f)
            err.append(np.sqrt(f * (1 - f) / n))
        return np.array(fr), np.array(err)

    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axs[0]
    sel = clean & (teff >= 3500) & (teff < 6500)
    for flag, lab, ls in (
            ((dmag < thr).astype(float), "SPHEREx > 0.375 mag above MS", "o-"),
            ((dms < thr).astype(float), "Gaia CMD > 0.375 mag above MS", "s-"),
            ((ruwe > 1.4).astype(float), "RUWE > 1.4", "^--"),
            ((nss > 0).astype(float), "Gaia NSS", "v:")):
        fr, err = fraction(flag, sel)
        ax.errorbar(mid, fr, yerr=err, fmt=ls, ms=3, label=lab)
        lines.append(f"{lab} (3500-6500 K, N={sel.sum()}): fraction at "
                     f"P<1 d {np.nanmean(fr[mid < 1]):.3f}, "
                     f"P>10 d {np.nanmean(fr[mid > 10]):.3f}")
    ax.set_title(f"clean stars, 3500-6500 K (N={sel.sum():,})", fontsize=9)
    ax.set_ylabel("fraction")
    ax.legend(fontsize=7)
    ax = axs[1]
    tbins = [(3500, 4250), (4250, 5000), (5000, 5750), (5750, 6500)]
    for lo, hi in tbins:
        sel = clean & (teff >= lo) & (teff < hi)
        fr, err = fraction((dmag < thr).astype(float), sel)
        ax.errorbar(mid, fr, yerr=err, fmt="o-", ms=3,
                    label=f"{lo}-{hi} K (N={sel.sum():,})")
        lines.append(f"SPHEREx overluminous fraction {lo}-{hi} K: "
                     f"P<1 d {np.nanmean(fr[mid < 1]):.3f}, "
                     f"P>10 d {np.nanmean(fr[mid > 10]):.3f}")
    ax.set_title("SPHEREx > 0.375 mag above MS, by Teff", fontsize=9)
    ax.legend(fontsize=7)
    for ax in axs[:2]:
        ax.set_xscale("log")
        ax.set_xlabel("TARS rotation period [d]")
        ax.set_ylim(0, None)
    ax = axs[2]
    sel = clean & (teff >= 4500) & (teff < 5500)
    bins = np.linspace(-2, 1, 61)
    for plo, phi, c in ((0.2, 2, "C3"), (10, 40, "C0")):
        m = sel & (per >= plo) & (per < phi)
        ax.hist(dmag[m], bins=bins, density=True, histtype="step", color=c,
                label=f"P = {plo}-{phi} d (N={m.sum():,})")
        ax.hist(dms[m], bins=bins, density=True, histtype="step", color=c,
                ls="--", alpha=0.6)
    ax.axvline(thr, color="k", ls=":", lw=0.8)
    ax.axvline(-0.753, color="k", ls="--", lw=0.8)
    ax.set_xlabel("mag above main sequence at Gaia distance")
    ax.set_ylabel("density")
    ax.set_title("4500-5500 K; solid SPHEREx, dashed Gaia CMD", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "binary_fraction_vs_period.png", dpi=180)
    plt.close(fig)


def excess_vs_tmag(df, out, lines):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    t = df["t_single"].to_numpy(dtype=float)
    tmag = df["tmag"].to_numpy(dtype=float)
    edges = np.arange(8, 16.5, 0.5)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for lo_t, hi_t, lab in ((0.0, 0.3, "t < 0.3 (hot)"), (0.3, 0.7, "0.3-0.7"),
                            (0.7, 1.01, "t > 0.7 (M dwarfs)")):
        for col, ls in (("excess_b6", "-"), ("excess_b1", "--")):
            v = df[col].to_numpy(dtype=float)
            med = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (tmag >= lo) & (tmag < hi) & (t >= lo_t) & (t < hi_t) & np.isfinite(v)
                med.append(np.median(v[m]) if m.sum() > 100 else np.nan)
            ax.plot(mid, 100 * np.array(med), ls, label=f"{col} {lab}")
            lines.append(f"{col} {lab}: median excess at Tmag 10 / 15: "
                         f"{100 * np.interp(10, mid, np.nan_to_num(med)):.2f}% / "
                         f"{100 * np.interp(15, mid, np.nan_to_num(med)):.2f}%")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Tmag")
    ax.set_ylabel("median excess over single template [%]")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "excess_vs_tmag.png", dpi=180)
    plt.close(fig)


def candidate_lists(df, out, lines):
    cols = ["ticid", "teff_emp", "teff_tars", "bprp0", "mg0", "dms", "dmag_ms",
            "tmag", "plx", "ruwe", "nss", "n_contams", "period", "med_snr",
            "chi2_red", "red_excess", "red_excess_sig", "dbic_binary",
            "dbic_binary_pct", "dbic_binary_ms", "dbic_binary_ms_pct",
            "teff_bin1", "teff_bin2", "flux_ratio_binary", "dbic_disk",
            "dbic_disk_vs_binary", "teff_disk", "disk_frac_b6",
            "sed_dbic_binary_pct", "sed_red_excess_rel"]
    cols = [c for c in cols if c in df]
    # Stars pinned at either end of the sequence (hotter or cooler than
    # the templates reach) are fit by a second component instead; keep
    # them out of the lists.
    t = df["t_single"]
    edge = (t <= 0.003) | (t >= 0.997)
    lines.append(f"stars at the sequence ends (t <= 0.003 or >= 0.997): "
                 f"{int(edge.sum())} ({edge.mean():.3%}); of the stars with "
                 f"dbic_disk > 100 and disk over binary, "
                 f"{((df['dbic_disk'] > 100) & (df['dbic_disk_vs_binary'] > 10) & edge).sum()} "
                 f"are at the ends")
    good = ((df["n_bins_used"] >= 60) & (df["n_contams"].fillna(0) == 0)
            & ~edge)
    # Binaries: both the shape and the luminosity prefer two stars.
    b = df[good & (df["dbic_binary_pct"] > 0.98) & (df["dbic_binary_ms_pct"] > 0.98)
           & (df["dbic_disk_vs_binary"] < 0)]
    b = b.sort_values("dbic_binary_ms", ascending=False)
    b[cols].to_csv(out / "binary_candidates.csv", index=False)
    lines.append(f"binary candidates (shape and Gaia-distance dBIC in the top 2% "
                 f"at fixed t, binary preferred over disk, no contaminants): "
                 f"{len(b)}; RUWE>1.4 fraction {(b['ruwe'] > 1.4).mean():.3f}, "
                 f"NSS>0 fraction {(b['nss'] > 0).mean():.3f}")
    # Disks: blackbody preferred over a second star, strong band-6 excess.
    d = df[good & (df["dbic_disk"] > 100) & (df["dbic_disk_vs_binary"] > 10)
           & (df["disk_frac_b6"] > 0.2) & (df["excess_b6_sig"] > 5)]
    d = d.sort_values("dbic_disk_vs_binary", ascending=False)
    d[cols].to_csv(out / "disk_candidates.csv", index=False)
    lines.append(f"disk candidates (dBIC disk > 100, disk over binary by > 10, "
                 f"blackbody > 20% of the star in band 6, no contaminants): "
                 f"{len(d)}; median Teff {d['teff_emp'].median():.0f} K, "
                 f"median T_bb {d['teff_disk'].median():.0f} K")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-dir", type=Path,
                   default=Path("/ar0/SPHEREX/bulkphot/tars_v0"))
    p.add_argument("--tars", type=Path, default=DEFAULT_TARS)
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "results" / "sx_ml_analysis")
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    viz = args.run_dir / "viz"
    ml = args.run_dir / "ml"
    cat = load_catalog(viz, args.tars)
    emp = pd.read_parquet(ml / "empfit.parquet")
    assert np.array_equal(emp["ticid"].to_numpy(), cat["ticid"].to_numpy())
    sed = pd.read_parquet(viz / "sed.parquet",
                          columns=["ticid", "chi2_full", "dof_full",
                                   "dbic_binary", "dbic_binary_pct",
                                   "red_excess", "red_excess_rel", "excess_b6",
                                   "teff_full"])
    sed = sed.set_index("ticid").reindex(cat["ticid"]).reset_index(drop=True)
    sed = sed.rename(columns={c: f"sed_{c}" for c in sed.columns})
    df = pd.concat([cat.reset_index(drop=True),
                    emp.drop(columns=["ticid"]).reset_index(drop=True), sed],
                   axis=1)
    df["chi2_full"] = df["sed_chi2_full"]
    df["dof_full"] = df["sed_dof_full"]
    df["dms"] = main_sequence_offset(df["bprp0"].to_numpy(dtype=float),
                                     df["mg0"].to_numpy(dtype=float))
    # Percentile of the CMD offset (bright = high) at fixed t.
    from sx_phot.empfit import add_population_columns
    df["neg_dms"] = -df["dms"]
    df = add_population_columns(df, columns=("neg_dms",))
    df["dms_pct"] = df["neg_dms_pct"]
    lines = [f"stars: {len(df)}, fitted: {int(np.isfinite(df['chi2_single']).sum())}"]
    chi2_calibration(df, out, lines)
    binary_enrichment(df, out, lines)
    red_excess_vs_teff(df, out, lines)
    binary_vs_period(df, out, lines)
    binary_fraction_vs_period(df, out, lines)
    excess_vs_tmag(df, out, lines)
    candidate_lists(df, out, lines)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
