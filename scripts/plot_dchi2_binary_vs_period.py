#!/home/luke/miniconda3/envs/py311_sx/bin/python
"""Binary-fit evidence versus TARS rotation period.

Reads the sx-viz SED fits (``<viz>/sed.parquet``), the per-star summary
and the TARS table, then writes

* ``<metric>_hist.png``: histogram of log10(metric) over stars with at
  least ``--min-n-used`` spectrum points, with the thresholds marked;
* ``<metric>_vs_period.png``: fraction of stars with metric above each
  threshold in log-uniform rotation-period bins (Wilson 68 percent
  intervals), plus the number of stars per bin;
* ``<metric>_vs_period.csv``: the binned numbers.

The metric is ``dchi2_binary`` by default (chi^2 improvement of the
two-star fit over the single-star fit); ``dbic_binary``, ``dbic_disk``
or ``dbic_disk_vs_binary`` work the same way.

Usage:
    python scripts/plot_dchi2_binary_vs_period.py \\
        [--viz-dir /ar0/SPHEREX/bulkphot/tars_v0/viz] \\
        [--metric dchi2_binary] [--thresholds 10 100] \\
        [--teff-min 0 --teff-max 1e5] [--tmag-min -10 --tmag-max 30] \\
        [--out-dir results/sx_viz_analysis]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aesthetic.plot import set_style

set_style("science")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

METRIC_LABELS = {
    "dchi2_binary": r"$\Delta\chi^2$ (single $-$ binary)",
    "dbic_binary": r"$\Delta$BIC (single $-$ binary)",
    "dbic_disk": r"$\Delta$BIC (single $-$ star+dust)",
    "dbic_disk_vs_binary": r"$\Delta$BIC (binary $-$ star+dust)",
    "dbic_binary_pct": r"$\Delta$BIC binary, percentile at Teff",
    "dbic_disk_pct": r"$\Delta$BIC star+dust, percentile at Teff",
}


def wilson_interval(k: np.ndarray, n: np.ndarray, z: float = 1.0):
    """Wilson score interval for k successes in n trials."""
    n = np.asarray(n, float)
    k = np.asarray(k, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def load(viz_dir: Path, tars_path: Path, metric: str,
         min_n_used: int) -> pd.DataFrame:
    """Join the SED fits with n_used and the TARS period and Teff."""
    sed = pd.read_parquet(viz_dir / "sed.parquet", columns=["ticid", metric])
    summ = pd.read_parquet(viz_dir / "summary.parquet",
                           columns=["ticid", "n_used"])
    tars = pd.read_feather(tars_path, columns=["TICID", "adopted_period",
                                               "teff", "Tmag"])
    tars = tars.rename(columns={"TICID": "ticid"})
    df = sed.merge(summ, on="ticid").merge(tars, on="ticid")
    df = df[(df["n_used"] >= min_n_used) & np.isfinite(df[metric])
            & (df["adopted_period"] > 0)]
    return df.reset_index(drop=True)


def plot_hist(df: pd.DataFrame, metric: str, thresholds: list[float],
              out: Path) -> None:
    """Histogram of log10(metric)."""
    vals = df[metric].to_numpy(float)
    pos = vals[vals > 0]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    linear = metric.endswith("_pct")
    if linear:
        bins = np.linspace(0, 1, 101)
        ax.hist(vals, bins=bins, color="0.3", histtype="stepfilled")
    else:
        bins = np.linspace(-1, 5, 121)
        ax.hist(np.clip(np.log10(pos), bins[0], bins[-1]), bins=bins,
                color="0.3", histtype="stepfilled")
    for t, c in zip(thresholds, ("C1", "C3", "C2", "C4")):
        ax.axvline(t if linear else np.log10(t), color=c, lw=1, ls="--",
                   label=f"{t:g} ({np.mean(vals > t) * 100:.1f}%)")
    ax.set_xlabel(("" if linear else r"$\log_{10}$ ")
                  + METRIC_LABELS.get(metric, metric))
    ax.set_ylabel("stars")
    ax.set_yscale("log")
    ax.legend(fontsize=7, title="threshold (fraction above)",
              title_fontsize=7)
    ax.set_title(f"{len(vals):,} stars; {np.mean(vals <= 0) * 100:.1f}% "
                 f"at or below 0", fontsize=8)
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)


def binned_fractions(df: pd.DataFrame, metric: str, thresholds: list[float],
                     n_bins: int, p_min: float, p_max: float) -> pd.DataFrame:
    """Fraction above each threshold in log-uniform period bins."""
    edges = np.logspace(np.log10(p_min), np.log10(p_max), n_bins + 1)
    period = df["adopted_period"].to_numpy(float)
    vals = df[metric].to_numpy(float)
    idx = np.digitize(period, edges) - 1
    rows = []
    for b in range(n_bins):
        sel = idx == b
        row = {"p_lo": edges[b], "p_hi": edges[b + 1],
               "p_mid": np.sqrt(edges[b] * edges[b + 1]), "n": int(sel.sum())}
        for t in thresholds:
            k = int((vals[sel] > t).sum())
            lo, hi = wilson_interval(k, max(sel.sum(), 1))
            row[f"k_gt_{t:g}"] = k
            row[f"frac_gt_{t:g}"] = k / sel.sum() if sel.sum() else np.nan
            row[f"frac_gt_{t:g}_lo"] = float(lo)
            row[f"frac_gt_{t:g}_hi"] = float(hi)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_fraction(table: pd.DataFrame, metric: str, thresholds: list[float],
                  out: Path, title: str) -> None:
    """Fraction above threshold versus period, with counts below."""
    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(4.2, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    ok = table["n"] > 0
    for t, c in zip(thresholds, ("C1", "C3", "C2", "C4")):
        f = table[f"frac_gt_{t:g}"]
        lo = table[f"frac_gt_{t:g}_lo"]
        hi = table[f"frac_gt_{t:g}_hi"]
        ax.errorbar(table["p_mid"][ok], f[ok],
                    yerr=[(f - lo)[ok], (hi - f)[ok]], fmt="o-", ms=3,
                    lw=1, color=c, capsize=2,
                    label=f"{METRIC_LABELS.get(metric, metric)} > {t:g}")
    ax.set_xscale("log")
    ax.set_ylabel("fraction of stars")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=8)
    axn.bar(table["p_mid"], table["n"],
            width=table["p_hi"] - table["p_lo"], color="0.5", align="center")
    axn.set_yscale("log")
    axn.set_xlabel("TARS rotation period [d]")
    axn.set_ylabel("stars")
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--viz-dir",
                        default="/ar0/SPHEREX/bulkphot/tars_v0/viz")
    parser.add_argument("--tars",
                        default="/ar0/local/TARS/tars_table_2.feather")
    parser.add_argument("--metric", default="dchi2_binary")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[10.0, 100.0])
    parser.add_argument("--min-n-used", type=int, default=100)
    parser.add_argument("--teff-min", type=float, default=0.0)
    parser.add_argument("--teff-max", type=float, default=1e5)
    parser.add_argument("--tmag-min", type=float, default=-10.0)
    parser.add_argument("--tmag-max", type=float, default=30.0)
    parser.add_argument("--n-bins", type=int, default=14)
    parser.add_argument("--p-min", type=float, default=None,
                        help="default: 0.5 percentile of the periods")
    parser.add_argument("--p-max", type=float, default=None,
                        help="default: 99.5 percentile of the periods")
    parser.add_argument("--out-dir",
                        default=str(REPO / "results" / "sx_viz_analysis"))
    parser.add_argument("--tag", default="",
                        help="suffix for the output file names")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(Path(args.viz_dir), Path(args.tars), args.metric,
              args.min_n_used)
    df = df[(df["teff"] >= args.teff_min) & (df["teff"] < args.teff_max)
            & (df["Tmag"] >= args.tmag_min) & (df["Tmag"] < args.tmag_max)]
    p_min = args.p_min or float(np.percentile(df["adopted_period"], 0.5))
    p_max = args.p_max or float(np.percentile(df["adopted_period"], 99.5))
    stem = args.metric + (f"_{args.tag}" if args.tag else "")
    plot_hist(df, args.metric, args.thresholds, out_dir / f"{stem}_hist.png")
    table = binned_fractions(df, args.metric, args.thresholds, args.n_bins,
                             p_min, p_max)
    table.to_csv(out_dir / f"{stem}_vs_period.csv", index=False)
    title = (f"{len(df):,} TARS stars, n_used >= {args.min_n_used}"
             + (f", {args.teff_min:.0f} <= Teff < {args.teff_max:.0f} K"
                if args.teff_min > 0 or args.teff_max < 1e5 else "")
             + (f", {args.tmag_min:.0f} <= T < {args.tmag_max:.0f}"
                if args.tmag_min > -10 or args.tmag_max < 30 else ""))
    plot_fraction(table, args.metric, args.thresholds,
                  out_dir / f"{stem}_vs_period.png", title)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"wrote {out_dir}/{stem}_hist.png, {stem}_vs_period.png, "
          f"{stem}_vs_period.csv")


if __name__ == "__main__":
    main()
