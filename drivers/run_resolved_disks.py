"""
Driver: Read resolved disks from data/resolved_disks_extracted.csv and
run SPHEREx circular-aperture photometry for each target using RA/Dec (J2000).

Outputs are written to the directory: 'resolved_disks'.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Optional

from sx_phot.circphot import get_sx_spectrum


CSV_PATH = Path("data/resolved_disks_extracted.csv")


def sanitize_star_id(name: str) -> str:
    """Make a safe star_id string for filenames: alnum, dash, underscore.
    Collapses whitespace to single underscore and strips other punctuation.
    """
    if not name:
        return "unknown"
    s = re.sub(r"\s+", "_", name.strip())
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s or "unknown"


def _clean_coord_text(s: str) -> str:
    """Normalize a coordinate string: unify minus, strip extraneous chars, compress spaces."""
    if s is None:
        return ""
    # Replace various unicode dashes with ASCII minus
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    # Keep digits, sign, spaces, and decimal point
    s = re.sub(r"[^0-9+\-\.\s]", " ", s)
    # Compress whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_ra_to_deg(text: str) -> Optional[float]:
    """Parse RA in either sexagesimal hours (HH MM SS.S) or degrees (DD MM SS.S) to degrees.

    Heuristics:
    - If first field > 24, treat as degrees; else treat as hours.
    - Accept 1-3 numeric fields; missing fields are treated as 0.
    - Normalize minutes/seconds that exceed 60.
    """
    s = _clean_coord_text(text)
    if not s:
        return None
    parts = s.split(" ")
    try:
        nums = [float(p) for p in parts if p]
    except ValueError:
        return None
    if not nums:
        return None
    # Pad to [a, b, c]
    while len(nums) < 3:
        nums.append(0.0)
    a, b, c = nums[:3]

    # If clearly degrees mode
    deg_mode = a > 24.0

    total_seconds = abs(a) * 3600.0 + b * 60.0 + c
    if deg_mode:
        deg = total_seconds / 3600.0
    else:
        # hours -> degrees
        deg = total_seconds / 240.0  # (sec / 3600) * 15 = sec / 240
    return deg


def parse_dec_to_deg(text: str) -> Optional[float]:
    """Parse Dec in sexagesimal degrees (sign DD MM SS.S) to degrees.

    Accepts 1-3 numeric fields; missing fields are treated as 0.
    Handles unicode minus and sign on the first component or leading.
    """
    raw = text or ""
    s = _clean_coord_text(raw)
    if not s:
        return None

    sign = -1.0 if s.startswith("-") else 1.0
    # Remove leading sign for splitting into numbers
    if s[0] in "+-":
        s_num = s[1:]
    else:
        s_num = s

    parts = s_num.split(" ")
    try:
        nums = [float(p) for p in parts if p]
    except ValueError:
        return None
    if not nums:
        return None

    while len(nums) < 3:
        nums.append(0.0)
    d, m, ssec = nums[:3]

    total_seconds = abs(d) * 3600.0 + m * 60.0 + ssec
    deg = sign * (total_seconds / 3600.0)
    return deg


def main() -> None:
    output_dir = Path("resolved_disks")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} resolved-disk rows. Running photometry...")

    for i, row in enumerate(rows, start=1):
        name = (row.get("Object") or "").strip()
        ra_txt = (row.get("RA (J2000)") or "").strip()
        dec_txt = (row.get("DEC (J2000)") or "").strip()

        ra_deg = parse_ra_to_deg(ra_txt)
        dec_deg = parse_dec_to_deg(dec_txt)

        if ra_deg is None or dec_deg is None or math.isnan(ra_deg) or math.isnan(dec_deg):
            print(f"[{i}/{len(rows)}] Skipping '{name}': invalid coords RA='{ra_txt}' Dec='{dec_txt}'")
            continue

        star_id = sanitize_star_id(name)
        print(f"[{i}/{len(rows)}] {name} @ {ra_deg:.6f}, {dec_deg:.6f}")

        try:
            get_sx_spectrum(
                ra_deg=ra_deg,
                dec_deg=dec_deg,
                star_id=star_id,
                output_dir=str(output_dir),
                # You can tweak these if desired:
                # use_cutout=True,
                # max_images=200,
                # do_photometry=False,
            )
        except Exception as e:
            print(f"  Skipped {name}: {e}")


if __name__ == "__main__":
    main()
