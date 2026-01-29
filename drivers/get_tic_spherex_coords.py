#!/Users/luke/local/miniconda3/envs/py311_sx/bin/python
"""Resolve a TIC ID to SPHEREx-era coordinates using Gaia DR2 proper motions.

Example:
    python drivers/get_tic_spherex_coords.py --tic-id TIC_300651846
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
import socket

import pandas as pd

from sx_phot.tic_motion import (
    add_mid_mjd,
    get_tic8_row,
    load_or_query_csv,
    normalize_tic_id,
    propagate_obscore_positions,
    query_gaia_dr2_astrometry,
    query_spherex_obscore,
    standardize_tic_row,
)

def _default_tic8_lookup_path() -> Path:
    """Return the local TIC8 lookup catalog path for this host."""
    host = socket.gethostname()
    if host in {"wh1", "wh2", "wh3"}:
        return Path("/ar0/local/TARS/tic8_plxGT2_TmagLT17_lukebouma.csv")
    return Path("/Users/luke/local/TARS/tic8_plxGT2_TmagLT17_lukebouma.csv")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Get RA/Dec for a TIC target near SPHEREx observation times "
            "by propagating Gaia DR2 proper motions."
        )
    )
    parser.add_argument(
        "--tic-id",
        required=True,
        help="TIC identifier, e.g. TIC_300651846 or 300651846.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root directory for cached query results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-query TIC/Gaia/SPHEREx even if cached files exist.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of ObsCore rows to keep.",
    )
    return parser.parse_args()


def _save_summary(df: pd.DataFrame, path: Path) -> None:
    """Write a small text summary of the propagated positions.

    Args:
        df: Output dataframe with t_mid_mjd, ra_deg, and dec_deg.
        path: Destination path for the summary text.
    """
    summary = {
        "n_rows": len(df),
        "t_mid_min": float(df["t_mid_mjd"].min()),
        "t_mid_max": float(df["t_mid_mjd"].max()),
        "ra_min": float(df["ra_deg"].min()),
        "ra_max": float(df["ra_deg"].max()),
        "ra_mean": float(df["ra_deg"].mean()),
        "dec_min": float(df["dec_deg"].min()),
        "dec_max": float(df["dec_deg"].max()),
        "dec_mean": float(df["dec_deg"].mean()),
    }
    path.write_text("\n".join(f"{k}: {v}" for k, v in summary.items()))


def _lookup_gaia_source_id(
    tic_id: int,
    local_path: Optional[Path] = None,
) -> Optional[int]:
    """Look up a Gaia DR2 source id for a TIC ID via a local catalog.

    Args:
        tic_id: TIC identifier as an integer.
        local_path: Path to the local TIC8 lookup CSV.

    Returns:
        Gaia DR2 source id if found, otherwise None.
    """
    local_path = local_path or _default_tic8_lookup_path()
    if not local_path.exists():
        return None

    usecols = ["ID", "GAIA"]
    for chunk in pd.read_csv(local_path, usecols=usecols, chunksize=200_000):
        ids = pd.to_numeric(chunk["ID"], errors="coerce")
        matches = ids == int(tic_id)
        if not matches.any():
            continue
        gaia_val = chunk.loc[matches, "GAIA"].iloc[0]
        try:
            return int(gaia_val)
        except (TypeError, ValueError):
            return None
    return None


def _load_spherex_coords_dataframe(
    tic_id: str | int,
    results_dir: str,
    overwrite: bool,
    max_rows: int | None,
) -> tuple[pd.DataFrame | None, Path, Path, str]:
    """Load or query propagated SPHEREx coordinates for a TIC target.

    Args:
        tic_id: TIC identifier, e.g. "TIC_300651846" or 300651846.
        results_dir: Root directory for cached query results.
        overwrite: If True, ignore cached results and re-query.
        max_rows: Optional maximum number of ObsCore rows to keep.

    Returns:
        Tuple of (dataframe, output CSV path, summary path, tic label). The
        dataframe is None when no ObsCore rows are found.
    """
    tic_id = normalize_tic_id(tic_id)
    tic_label = f"TIC_{tic_id}"

    cache_dir = Path(results_dir) / "tic_motion"
    tic_cache = cache_dir / f"tic8_{tic_label}.csv"
    gaia_cache = cache_dir / f"gaia_dr2_{tic_label}.csv"
    obscore_cache = cache_dir / f"spherex_obscore_{tic_label}.csv"
    out_cache = cache_dir / f"spherex_coords_{tic_label}.csv"
    summary_path = cache_dir / f"spherex_coords_{tic_label}_summary.txt"

    gaia_source_id = _lookup_gaia_source_id(tic_id)
    if gaia_source_id is None:
        tic_df = load_or_query_csv(
            tic_cache,
            lambda: standardize_tic_row(get_tic8_row(tic_id, cache_dir)),
            overwrite=overwrite,
        )
        gaia_source_id = int(tic_df.loc[0, "gaia_dr2_source_id"])
    gaia_df = load_or_query_csv(
        gaia_cache,
        lambda: query_gaia_dr2_astrometry(gaia_source_id),
        overwrite=overwrite,
    )

    gaia_row = gaia_df.iloc[0]
    search_ra = float(gaia_row["gaia_ra_deg"])
    search_dec = float(gaia_row["gaia_dec_deg"])

    obscore_df = load_or_query_csv(
        obscore_cache,
        lambda: query_spherex_obscore(search_ra, search_dec),
        overwrite=overwrite,
    )

    if obscore_df.empty:
        return None, out_cache, summary_path, tic_label

    obscore_df = add_mid_mjd(obscore_df)
    if max_rows is not None and len(obscore_df) > max_rows:
        obscore_df = obscore_df.sort_values("t_mid_mjd").head(max_rows)

    out_df = propagate_obscore_positions(obscore_df, gaia_row)
    out_df.insert(0, "tic_id", tic_id)
    out_df.insert(1, "gaia_source_id", gaia_source_id)

    out_df = out_df.sort_values("t_mid_mjd")
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_cache, index=False)
    _save_summary(out_df, summary_path)

    return out_df, out_cache, summary_path, tic_label


def get_mean_spherex_coords(
    tic_id: str | int,
    results_dir: str = "results",
    overwrite: bool = False,
    max_rows: int | None = None,
) -> tuple[float, float]:
    """Return mean propagated coordinates for a TIC target.

    Args:
        tic_id: TIC identifier, e.g. "TIC_300651846" or 300651846.
        results_dir: Root directory for cached query results.
        overwrite: If True, ignore cached results and re-query.
        max_rows: Optional maximum number of ObsCore rows to keep.

    Returns:
        Tuple of (ra_deg, dec_deg) mean coordinates.

    Raises:
        ValueError: If no SPHEREx ObsCore rows are found.
    """
    out_df, _, _, tic_label = _load_spherex_coords_dataframe(
        tic_id=tic_id,
        results_dir=results_dir,
        overwrite=overwrite,
        max_rows=max_rows,
    )
    if out_df is None or out_df.empty:
        raise ValueError(f"No SPHEREx ObsCore rows found for {tic_label}.")

    ra_mean = float(out_df["ra_deg"].mean())
    dec_mean = float(out_df["dec_deg"].mean())
    return ra_mean, dec_mean


def _log(message: str) -> None:
    """Print a timestamped log message."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}")


def main() -> None:
    """Run TIC/Gaia/SPHEREx queries and write propagated coordinates."""
    args = _parse_args()
    out_df, out_cache, summary_path, tic_label = _load_spherex_coords_dataframe(
        tic_id=args.tic_id,
        results_dir=args.results_dir,
        overwrite=args.overwrite,
        max_rows=args.max_rows,
    )

    if out_df is None or out_df.empty:
        print(f"No SPHEREx ObsCore rows found for {tic_label}.")
        return

    _log(f"Wrote {out_cache}")
    _log(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
