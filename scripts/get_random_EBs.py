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
ECL_LAT_RANGE_DEG = (-76.0, -72.0)
TEFF_MAX_K = 6500.0
RANDOM_SEED = 42
SAMPLE_SIZE = 50
OUTPUT_SUBDIR = Path("results/random_ebs")
OUTPUT_NAME = (
    "kostov_2025_ebs_ecl_lat_-76_-72_teff_lt_6500.csv"
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
    """Filter the catalog by ecliptic latitude and temperature.

    Args:
        table: Catalog table with ecliptic latitude, period, and Teff.

    Returns:
        Filtered table sorted by period.
    """
    lat_min, lat_max = sorted(ECL_LAT_RANGE_DEG)
    lat = np.asarray(table["ecl_lat_deg"], dtype=float)
    teff = np.asarray(table["Teff"], dtype=float)
    mask = (
        (lat >= lat_min)
        & (lat <= lat_max)
        & (teff < TEFF_MAX_K)
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


def print_random_draws(table: Table) -> None:
    """Print a random subset of the filtered table.

    Args:
        table: Filtered catalog table.
    """
    if len(table) == 0:
        print("No rows remain after filtering.")
        return

    rng = np.random.default_rng(RANDOM_SEED)
    n_draws = min(SAMPLE_SIZE, len(table))
    indices = rng.choice(len(table), size=n_draws, replace=False)
    sample = table[indices]

    print(f"Random draws (seed={RANDOM_SEED}, n={n_draws}):")
    print("TIC, RAdeg, DEdeg, Tmag, Per")
    for row in sample:
        tic = _format_id(row["TIC"])
        ra = _format_float(row["RAdeg"], ".6f")
        dec = _format_float(row["DEdeg"], ".6f")
        tmag = _format_float(row["Tmag"], ".3f")
        per = _format_float(row["Per"], ".6f")
        print(f"{tic}, {ra}, {dec}, {tmag}, {per}")


def main() -> None:
    """Run the filtered selection and cache results."""
    root = repo_root()
    catalog_path = resolve_catalog_path(root)
    output_path = root / OUTPUT_SUBDIR / OUTPUT_NAME

    table = load_catalog(catalog_path)
    require_columns(
        table,
        ["RAdeg", "DEdeg", "Per", "Teff", "TIC", "Tmag"],
    )
    add_ecliptic_columns(table)
    filtered = filter_catalog(table)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.write(output_path, format="ascii.csv", overwrite=True)
    print(f"Wrote {len(filtered)} rows to {output_path}")

    print_random_draws(filtered)


if __name__ == "__main__":
    main()
