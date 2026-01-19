#!/Users/luke/local/miniconda3/envs/py311_sx/bin/python
"""Select random eclipsing binaries from the Kostov 2025 MRT catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import astropy.units as u
import numpy as np
from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord
from astropy.table import Table

CATALOG_REL_PATHS = (
    Path("data/literature/Kostov_2025_tess10k_apjsade2d8t3_mrt.txt"),
    Path(
        "sx_phot/data/literature/"
        "Kostov_2025_tess10k_apjsade2d8t3_mrt.txt"
    ),
)
ECL_LAT_MAX_DEG = -72.0
TMAG_MAX = 14.5
PER_MAX_DAYS = 2.0
RANDOM_SEED = 42
OUTPUT_SUBDIR = Path("results/random_ebs")
OUTPUT_NAME = (
    "kostov_2025_ebs_ecl_lat_lt_-72_tmag_lt_14p5_per_lt_2.csv"
)
OUTPUT_TXT_NAME = (
    "kostov_2025_ebs_ecl_lat_lt_-72_tmag_lt_14p5_per_lt_2.txt"
)


def repo_root() -> Path:
    """Return the repository root path inferred from the script location."""
    return Path(__file__).resolve().parents[1]


def load_catalog(path: Path) -> Table:
    """Load the Kostov 2025 MRT catalog from disk.

    Args:
        path: Path to the MRT catalog.

    Returns:
        Astropy Table with catalog contents.

    Raises:
        FileNotFoundError: If the catalog is missing.
        OSError: If the catalog cannot be read.
    """
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")
    return Table.read(path, format="ascii.mrt")


def resolve_catalog_path(root: Path) -> Path:
    """Resolve the first existing catalog path under the repo root.

    Args:
        root: Repository root path.

    Returns:
        Path to the catalog file.

    Raises:
        FileNotFoundError: If no catalog path exists.
    """
    for rel_path in CATALOG_REL_PATHS:
        candidate = root / rel_path
        if candidate.exists():
            return candidate
    searched = ", ".join(str(root / rel) for rel in CATALOG_REL_PATHS)
    raise FileNotFoundError(
        f"Catalog not found. Checked: {searched}"
    )


def require_columns(table: Table, columns: Iterable[str]) -> None:
    """Ensure the table has all required columns.

    Args:
        table: Catalog table to inspect.
        columns: Column names required for processing.

    Raises:
        ValueError: If any required columns are missing.
    """
    missing = [name for name in columns if name not in table.colnames]
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )


def add_ecliptic_columns(table: Table) -> None:
    """Compute and append ecliptic longitude/latitude columns.

    Args:
        table: Catalog table with RAdeg and DEdeg columns.
    """
    ra_deg = np.asarray(table["RAdeg"], dtype=float)
    dec_deg = np.asarray(table["DEdeg"], dtype=float)
    coords = SkyCoord(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        frame="icrs",
    )
    ecl = coords.transform_to(BarycentricTrueEcliptic())
    table["ecl_lon_deg"] = ecl.lon.deg
    table["ecl_lat_deg"] = ecl.lat.deg


def filter_catalog(table: Table) -> Table:
    """Filter the catalog by ecliptic latitude, TESS magnitude, and period.

    Args:
        table: Catalog table with ecliptic latitude, period, and Teff.

    Returns:
        Filtered table sorted by period.
    """
    lat = np.asarray(table["ecl_lat_deg"], dtype=float)
    tmag = np.asarray(table["Tmag"], dtype=float)
    per = np.asarray(table["Per"], dtype=float)
    mask = (
        (lat < ECL_LAT_MAX_DEG)
        & (tmag < TMAG_MAX)
        & (per < PER_MAX_DAYS)
    )
    filtered = table[mask]
    filtered.sort("Per")
    return filtered


def _format_id(value: object) -> str:
    """Format a TIC-like identifier for printing."""
    if value is np.ma.masked or value is None:
        return "nan"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _format_float(value: object, fmt: str) -> str:
    """Format a float value with a fallback for masked/invalid entries."""
    if value is np.ma.masked or value is None:
        return "nan"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(val):
        return "nan"
    return f"{val:{fmt}}"


def format_row(row: object) -> str:
    """Format a row for CSV-like printing.

    Args:
        row: Table row with required fields.
    """
    tic = _format_id(row["TIC"])
    ra = _format_float(row["RAdeg"], ".6f")
    dec = _format_float(row["DEdeg"], ".6f")
    tmag = _format_float(row["Tmag"], ".3f")
    per = _format_float(row["Per"], ".6f")
    t0 = _format_float(row["T0-pri"], ".6f")
    return f"{tic}, {ra}, {dec}, {tmag}, {per}, {t0}"


def write_text_list(table: Table, output_path: Path) -> None:
    """Write the filtered catalog to a plain text list.

    Args:
        table: Filtered catalog table.
        output_path: Destination text file path.
    """
    if len(table) == 0:
        print("No rows remain after filtering; skipping text output.")
        return

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("TIC, RAdeg, DEdeg, Tmag, Per, T0-pri\n")
        for row in table:
            handle.write(f"{format_row(row)}\n")


def main() -> None:
    """Run the filtered selection and cache results."""
    root = repo_root()
    catalog_path = resolve_catalog_path(root)
    output_path = root / OUTPUT_SUBDIR / OUTPUT_NAME

    table = load_catalog(catalog_path)
    require_columns(
        table, ["RAdeg", "DEdeg", "Per", "TIC", "Tmag", "T0-pri"]
    )
    add_ecliptic_columns(table)
    filtered = filter_catalog(table)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.write(output_path, format="ascii.csv", overwrite=True)
    print(f"Wrote {len(filtered)} rows to {output_path}")
    text_path = output_path.with_name(OUTPUT_TXT_NAME)
    write_text_list(filtered, text_path)
    if text_path.exists():
        print(f"Wrote text list to {text_path}")


if __name__ == "__main__":
    main()
