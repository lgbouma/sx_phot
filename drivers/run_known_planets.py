"""
Driver: Query NASA Exoplanet Archive for known planets within 50 pc and
run SPHEREx circular-aperture photometry for each host.

Outputs are written to the directory: 'known_planets_50pc'.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any

import requests

from sx_phot.circphot import get_sx_spectrum


EXA_TAP_SYNC = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def sanitize_star_id(name: str) -> str:
    """Make a safe star_id string for filenames: alnum, dash, underscore.
    Collapses whitespace to single underscore and strips other punctuation.
    """
    if not name:
        return "unknown"
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s or "unknown"


def query_known_planets_within_distance(
    min_pc: float = 30.0,
    max_pc: float = 60.0,
    limit: int | None = None
    ) -> List[Dict[str, Any]]:

    """Query pscomppars for host RA, Dec, name, and system distance.

    Returns a list of rows (dicts) with keys: hostname, ra, dec, sy_dist.
    """
    base_query = (
        "SELECT DISTINCT hostname, ra, dec, sy_dist "
        "FROM pscomppars "
        f"WHERE sy_dist < {float(max_pc)} "
        f"AND sy_dist > {float(min_pc)} "
        "AND ra IS NOT NULL AND dec IS NOT NULL AND sy_dist IS NOT NULL "
        "ORDER BY sy_dist ASC"
    )
    if limit and limit > 0:
        query = f"{base_query} OFFSET 0 ROWS FETCH NEXT {int(limit)} ROWS ONLY"
    else:
        query = base_query

    params = {"query": query, "format": "json"}
    resp = requests.get(EXA_TAP_SYNC, params=params, timeout=60)
    resp.raise_for_status()
    rows = resp.json()
    # Ensure types are correct
    out = []
    for r in rows:
        try:
            ra = float(r.get("ra"))
            dec = float(r.get("dec"))
            dist = float(r.get("sy_dist"))
            host = str(r.get("hostname") or "")
        except Exception:
            continue
        out.append({"hostname": host, "ra": ra, "dec": dec, "sy_dist": dist})
    return out


def main() -> None:
    min_pc = 30.
    max_pc = 60.

    output_dir = Path("known_planets_{min_pc}-{max_pc}pc")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch nearby known planet hosts
    rows = query_known_planets_within_distance(min_pc=min_pc, max_pc=max_pc, limit=None)
    print(f"Found {len(rows)} hosts from {min_pc}-{max_pc} pc. Running photometry...")

    for i, r in enumerate(rows, start=1):
        host = sanitize_star_id(r["hostname"]) if r.get("hostname") else None
        print(f"[{i}/{len(rows)}] {r['hostname']} @ {r['ra']:.5f}, {r['dec']:.5f} (d={r['sy_dist']:.1f} pc)")
        try:
            get_sx_spectrum(
                ra_deg=r["ra"],
                dec_deg=r["dec"],
                star_id=host,
                output_dir=str(output_dir),
                # You can tweak these if desired:
                # use_cutout=True,
                # max_images=200,
                # do_photometry=False,
            )
        except Exception as e:
            print(f"  Skipped {r['hostname']}: {e}")


if __name__ == "__main__":
    main()
