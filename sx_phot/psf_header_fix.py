"""Repair the IRSA PSF zone-header error in Level-2 files in place.

Pipeline versions 6.4 to 6.5.5 wrote the PSF cube zone cards with x
and y exchanged (IRSA notice of 2026-03-13).  IRSA's corrected files
differ from the originals only in two header blocks: ``VERSION`` gains
the suffix ``+psffix1`` and, for every zone ``i``, ``XCTR_i`` takes the
value of ``YCTR_i`` and vice versa (likewise ``XWID_i``/``YWID_i``),
while each card keeps its own comment with the zone index written
``(j, i)`` instead of ``(i, j)``.  Verified card for card against the
S3 copy of one file on 2026-11-10 (data arrays identical).

Both header blocks keep their byte size, so the repair overwrites the
two blocks in place with ``os.pwrite`` and leaves the data untouched.
Do not run it while another process reads the same files.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from astropy.io import fits

from sx_phot.psf_phot import psf_header_transposed

log = logging.getLogger(__name__)

PSFFIX_SUFFIX = "+psffix1"
_ZONE_RE = re.compile(r"\((\d+), (\d+)\)")
_PAIRS = (("XCTR", "YCTR"), ("XWID", "YWID"))


def swap_zone_cards(header: fits.Header) -> int:
    """Exchange the x/y zone cards of a PSF extension header in place.

    Args:
        header: Header of the ``PSF`` image cube extension.

    Returns:
        Number of zones whose cards were exchanged.

    Raises:
        KeyError: If a zone card is missing.
    """
    n = int(header.get("NAXIS3", 0))
    for i in range(1, n + 1):
        for a, b in _PAIRS:
            ka, kb = f"{a}_{i}", f"{b}_{i}"
            va, ca = header[ka], header.comments[ka]
            vb, cb = header[kb], header.comments[kb]
            header[ka] = (vb, _ZONE_RE.sub(r"(\2, \1)", ca))
            header[kb] = (va, _ZONE_RE.sub(r"(\2, \1)", cb))
    return n


def fix_file(path: Path, dry_run: bool = False) -> str:
    """Repair one Level-2 file if its PSF header has the error.

    Args:
        path: Level-2 FITS file.
        dry_run: Report what would happen without writing.

    Returns:
        ``"fixed"`` (or ``"would-fix"`` with ``dry_run``), ``"skipped"``
        when the version is unaffected or already carries the suffix,
        or ``"no-psf"`` when the file has no PSF extension.

    Raises:
        RuntimeError: If a repaired header block changes size, or the
            file does not read back as expected after writing.
    """
    path = Path(path)
    with fits.open(path, memmap=True) as hdul:
        version = hdul[0].header.get("VERSION")
        if not psf_header_transposed(version):
            return "skipped"
        if "PSF" not in hdul:
            return "no-psf"
        if dry_run:
            return "would-fix"
        prim, psf = hdul[0], hdul["PSF"]
        blocks = []
        for hdu, edit in ((prim, None), (psf, swap_zone_cards)):
            old = hdu.header.tostring()
            hdr = hdu.header.copy()
            if edit is None:
                hdr["VERSION"] = str(version) + PSFFIX_SUFFIX
            else:
                edit(hdr)
            new = hdr.tostring()
            if len(new) != len(old):
                raise RuntimeError(f"{path}: header size would change")
            blocks.append((int(hdu._header_offset), new.encode("ascii")))
    fd = os.open(path, os.O_RDWR)
    try:
        for offset, data in blocks:
            if os.pwrite(fd, data, offset) != len(data):
                raise RuntimeError(f"{path}: short write")
        os.fsync(fd)
    finally:
        os.close(fd)
    check = fits.getheader(path, 0).get("VERSION")
    if check != str(version) + PSFFIX_SUFFIX:
        raise RuntimeError(f"{path}: VERSION reads back as {check!r}")
    return "fixed"


__all__ = ["PSFFIX_SUFFIX", "fix_file", "swap_zone_cards"]
