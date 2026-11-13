"""Gaia-forced PSF photometry for SPHEREx Level-2 images.

Derived from SPIFF (Gagne et al. 2026, github.com/jgagneastro/SPIFF),
which fits one PSF per cutout with a free flux and shift.  Here every
Gaia DR3 source whose PSF reaches an 11 x 11 px box around the target
is modelled at its own proper-motion-propagated position with its own
free flux, together with a plane background, in one weighted linear
least-squares solve.  Neighbours are therefore fitted rather than
absorbed into the target or the background.

Model choices (validated on SPYGLASS VII-A and Cep-Her/CMaN stars):

* Positions are fixed by Gaia.  A common offset per PSF product is
  applied (qr2 'PSF' cube: +0.055, +0.045 px; qr3 'EPSF' table: 0),
  the median of free fits at S/N > 10 (scatter 0.04 px).
* The zone PSF nearest to the target is used for every source in the
  box.  It is rendered by shifting the oversampled PSF on its own grid
  so that the samples fall symmetrically inside detector pixels, then
  block-summing (qr2 'PSF' cube, an oversampled instrumental PSF) or
  taking the sample at each pixel centre (qr3 'EPSF' table, an
  effective PSF that already includes the pixel response).  Treating
  the EPSF like the cube broadened the model: chi2/nu 5.4 instead of
  2.5 and qr3 fluxes 6% above qr2 for the same stars (2026-11-10,
  5,444 star-rows); point sampling gives 1.002, matching the aperture
  ratio 0.992.  Stamps at integer sample shifts are cached per zone
  and interpolated with 4-point (cubic Lagrange) weights in the
  fractional shift, which matches a cubic-spline shift of the
  oversampled PSF to about 1e-4 of the stamp peak.
* The pixel sum of the qr3 'EPSF' table stamps depends on the
  sub-pixel position of the star (2.1% rms on detector 3), which an
  effective PSF should not do, and aperture fluxes do not follow it
  (slope -0.005, 2026-11-11).  EPSF stamps are therefore rescaled to
  the mean stamp sum of their zone, so the model conserves flux at
  every pixel phase.  The stamp is 5 x 5 px, so the true phase
  dependence of the truncated sum (about 0.2-0.3% rms) is removed
  as well.
* Files from pipeline versions 6.4-6.5.5 without the '+psffix1'
  VERSION suffix describe the PSF cube zones with x and y swapped
  (IRSA notice of 2026-03-13, SPHEREx/docs/psfhdrerr.html; up to 5%
  photometric error).  Their zone centres are swapped back here.
* Sources are kept if G < min(G_t + 4.25, 20.5), or G < G_t + 6.5
  within 1.5 px of the target; sources centred outside the box only
  if G < G_t + 1.5.  Sources closer than 0.25 px are merged into one
  flux-weighted component.
* Neighbour components with no PSF light on usable box pixels are
  dropped.  Components with very little light remain nearly
  unconstrained (condition number ~1e17); the target flux is still
  well determined, so the covariance falls back to the pseudo-inverse
  when the normal matrix is singular.  Without that fallback the
  flux error is NaN for ~12% of qr2 rows.  Dropping components with
  < 2% of their light on usable pixels instead biased crowded fields
  (bright neighbours put real light in the box).
* Boxes with fewer than ``min_dof`` degrees of freedom left (hundreds
  of Gaia sources toward the bulge) get status 5 'underdetermined'.

Flux = amplitude (MJy/sr) x OMEGA_MEDIAN (sr) x sum of the oversampled
PSF, as in SPIFF.  Note that the aperture columns of ``bulk_phot`` use
the nominal pixel area instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from sx_phot.circphot import BITMASK

log = logging.getLogger(__name__)

# BITMASK plus PERSIST (17) and OUTLIER (19).
PSF_BITMASK = int(BITMASK | (1 << 17) | (1 << 19))
ARCSEC2_SR = (np.pi / 180.0 / 3600.0) ** 2
SHIFT0 = {"cube": (0.055, 0.045), "table": (0.0, 0.0)}
STATUS = {0: "ok", 1: "edge", 2: "masked", 3: "no_gaia", 4: "error",
          5: "underdetermined"}

PSF_DTYPES = {
    "psf_status": np.int8,
    "psf_flux_jy": np.float32,
    "psf_flux_err_jy": np.float32,
    "psf_chi2nu": np.float32,
    "psf_n_comp": np.int16,
    "psf_n_drop": np.int16,
    "psf_nb_flux_jy": np.float32,
    "psf_bkg_mjysr": np.float32,
    "psf_blend_frac_g": np.float32,
    "psf_min_corr": np.float32,
    "psf_apsub_jy": np.float32,
    "psf_apsub_err_jy": np.float32,
    "psf_ap_ee": np.float32,
    "psf_zone": np.int16,
    "psf_n_bad_c3": np.int16,
}


@dataclass
class PsfParams:
    """Forced-PSF fit parameters.

    Attributes:
        half: Fit box half-size (box is 2 * half + 1 px).
        reach: Rendered PSF stamp half-size in px.
        merge_px: Sources closer than this are one component.
        dg_keep: Keep sources with G < G_t + dg_keep ...
        g_abs: ... and G < g_abs.
        dg_keep_near: Within near_px, keep G < G_t + dg_keep_near.
        near_px: Radius for dg_keep_near (px).
        dg_out: Sources centred outside the box: keep G < G_t + dg_out.
        ap_r: Radius of the neighbour-subtracted aperture (px).
        bg_order: Background polynomial order (1 plane, 2 quadratic).
        drop_frac: Drop neighbours with at most this fraction of their
            PSF light on usable box pixels (0: only those with none).
        max_bad_c3: Status 'masked' if more central 3 x 3 pixels are bad.
        min_good: Status 'masked' if fewer usable box pixels remain.
        min_dof: Status 'underdetermined' below this many degrees of
            freedom (fluxes are still reported).
    """

    half: int = 5
    reach: int = 4
    merge_px: float = 0.25
    dg_keep: float = 4.25
    g_abs: float = 20.5
    dg_keep_near: float = 6.5
    near_px: float = 1.5
    dg_out: float = 1.5
    ap_r: float = 2.0
    bg_order: int = 1
    drop_frac: float = 0.0
    max_bad_c3: int = 4
    min_good: int = 50
    min_dof: int = 20


@dataclass
class Neighbours:
    """Gaia sources around each catalog star in CSR layout.

    Rows ``offsets[i]:offsets[i + 1]`` belong to catalog star_idx ``i``.
    ``target[i]`` is the global row of the star itself, or -1 if no
    Gaia source lies within the match radius.

    Attributes:
        offsets: (n_stars + 1,) row offsets.
        ra: Gaia RA at J2016.0 (deg).
        dec: Gaia Dec at J2016.0 (deg).
        pmra: Proper motion RA*cos(dec) (mas/yr), NaN filled with 0.
        pmdec: Proper motion Dec (mas/yr), NaN filled with 0.
        g: Gaia G (mag), NaN filled with 21.
        target: (n_stars,) global target row or -1.
    """

    offsets: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    pmra: np.ndarray
    pmdec: np.ndarray
    g: np.ndarray
    target: np.ndarray


def psf_header_transposed(version: Optional[str]) -> bool:
    """Whether a file's PSF zone centres have x and y swapped.

    Args:
        version: Primary-header VERSION keyword (e.g. '6.4',
            '6.5.5+psffix1', '7.0.5').

    Returns:
        True for pipeline versions 6.4 to 6.5.5 without '+psffix'.
    """
    if not version or "psffix" in str(version):
        return False
    try:
        v = tuple(int(t) for t in str(version).split("+")[0].split("."))
    except ValueError:
        return False
    v = (v + (0, 0, 0))[:3]
    return (6, 4, 0) <= v <= (6, 5, 5)


def empty_psf_rows(n: int) -> dict[str, np.ndarray]:
    """Return PSF columns for n rows, filled with NaN / zero.

    Args:
        n: Number of rows.

    Returns:
        Dict of column name to array.
    """
    out = {}
    for k, v in PSF_DTYPES.items():
        fill = np.nan if np.issubdtype(v, np.floating) else 0
        out[k] = np.full(n, fill, dtype=v)
    out["psf_zone"][:] = -1
    return out


class ZonePsf:
    """Zone PSFs of one Level-2 image with cached pixel stamps.

    Handles the qr2 'PSF' image cube (121 zones of 101 x 101 at 0.1 px,
    zone centres 1-based in the header; pixels are block sums of the
    samples) and the qr3 'EPSF' table (441 zones of 33 x 33 at 0.2 px,
    zone centres 0-based; an effective PSF, so pixels take the sample
    at their centre).

    Args:
        hdul: Open Level-2 HDUList.
        reach: Stamp half-size in px (reduced if the PSF array is
            smaller).
        renorm_table: Rescale EPSF-table stamps to the mean stamp sum
            of their zone (see the module docstring).

    Raises:
        KeyError: If the file has neither a PSF nor an EPSF extension.
    """

    def __init__(self, hdul, reach: int = 4,
                 renorm_table: bool = True) -> None:
        names = {h.name.upper() for h in hdul}
        if "PSF" in names:
            hp = hdul["PSF"].header
            self.kind = "cube"
            self.o = int(hp["OVERSAMP"])
            self.c = int(hp["CRPIX1"]) - 1
            n = int(hp["NAXIS3"])
            self.zx = np.array([hp[f"XCTR_{i}"] for i in range(1, n + 1)],
                               float) - 1.0
            self.zy = np.array([hp[f"YCTR_{i}"] for i in range(1, n + 1)],
                               float) - 1.0
            self._data = hdul["PSF"].data
            self.transposed = psf_header_transposed(
                hdul[0].header.get("VERSION"))
            if self.transposed:
                self.zx, self.zy = self.zy, self.zx
        elif "EPSF" in names:
            e = hdul["EPSF"]
            self.kind = "table"
            self.o = int(e.header["OVSMPX"])
            self._data = np.asarray(e.data["EPSF"], float)
            self.c = (self._data.shape[-1] - 1) // 2
            self.zx = np.asarray(e.data["XCENTER"], float)
            self.zy = np.asarray(e.data["YCENTER"], float)
            self.transposed = False
        else:
            raise KeyError("no PSF or EPSF extension")
        o = self.o
        self.ev = 0.5 if o % 2 == 0 else 0.0
        self.reach = min(int(reach),
                         int((self.c + self.ev - (o - 1) / 2) // o))
        self.k0 = int(round(self.c + self.ev - (o - 1) / 2 - o * self.reach))
        self.pad = o + 3
        self._padded: dict[int, np.ndarray] = {}
        self._psum: dict[int, float] = {}
        self.renorm = bool(renorm_table) and self.kind == "table"
        self._ssum: dict[int, float] = {}
        self._stamps: dict[tuple[int, int, int], np.ndarray] = {}

    def zone(self, x: float, y: float) -> int:
        """Return the zone index nearest to 0-based pixel (x, y)."""
        return int(np.argmin(np.hypot(self.zx - x, self.zy - y)))

    def _load(self, zone: int) -> np.ndarray:
        if zone not in self._padded:
            p = np.asarray(self._data[zone], float)
            self._psum[zone] = float(p.sum())
            self._padded[zone] = np.pad(p, self.pad)
        return self._padded[zone]

    def psum(self, zone: int) -> float:
        """Return the sum of the oversampled PSF of a zone."""
        self._load(zone)
        return self._psum[zone]

    def stamp_sum(self, zone: int) -> float:
        """Return the stamp sum of a zone averaged over pixel phase.

        The average runs over the o x o integer sample shifts, which
        tile one pixel.
        """
        if zone not in self._ssum:
            o = self.o
            self._ssum[zone] = float(np.mean(
                [self._int_stamp(zone, sy, sx).sum()
                 for sy in range(o) for sx in range(o)]))
        return self._ssum[zone]

    def _int_stamp(self, zone: int, sy: int, sx: int) -> np.ndarray:
        """Pixel stamp for an integer shift (sy, sx) in samples.

        The qr2 cube is an oversampled instrumental PSF, so each pixel
        is the sum of its o x o samples.  The qr3 EPSF is an effective
        PSF (already integrated over the pixel), so each pixel takes the
        sample at its centre, scaled by o^2 to keep the stamp sum equal
        to the sample sum.
        """
        key = (zone, sy, sx)
        if key not in self._stamps:
            pp = self._load(zone)
            o, n = self.o, 2 * self.reach + 1
            a = self.k0 - sy + self.pad
            b = self.k0 - sx + self.pad
            blk = pp[a:a + o * n, b:b + o * n].reshape(n, o, n, o)
            if self.kind == "cube":
                stamp = blk.sum(axis=(1, 3))
            elif o % 2:
                stamp = blk[:, o // 2, :, o // 2] * (o * o)
            else:
                c = slice(o // 2 - 1, o // 2 + 1)
                stamp = blk[:, c, :, c].mean(axis=(1, 3)) * (o * o)
            self._stamps[key] = stamp
        return self._stamps[key]

    def stamp(self, zone: int, fx: float, fy: float) -> np.ndarray:
        """Unit-amplitude pixel stamp for a sub-pixel offset.

        Args:
            zone: Zone index.
            fx: Source x minus the nearest pixel centre (-0.5..0.5).
            fy: Source y minus the nearest pixel centre (-0.5..0.5).

        Returns:
            (2 * reach + 1)^2 stamp centred on the nearest pixel.
        """
        ty = self.o * fy + self.ev
        tx = self.o * fx + self.ev
        y0, x0 = int(np.floor(ty)), int(np.floor(tx))
        wy, wx = _lagrange4(ty - y0), _lagrange4(tx - x0)
        out = 0.0
        for i in range(4):
            row = 0.0
            for j in range(4):
                row = row + wx[j] * self._int_stamp(zone, y0 + i - 1,
                                                    x0 + j - 1)
            out = out + wy[i] * row
        if self.renorm:
            tot = float(out.sum())
            if tot > 0:
                out = out * (self.stamp_sum(zone) / tot)
        return out

    def render(self, zone: int, xl: float, yl: float, h: int,
               w: int) -> np.ndarray:
        """Render a unit-amplitude source at local (xl, yl) on an h x w box.

        Args:
            zone: Zone index.
            xl: Source x in box coordinates (0-based pixel centres).
            yl: Source y in box coordinates.
            h: Box height.
            w: Box width.

        Returns:
            (h, w) array.
        """
        x0, y0 = int(np.round(xl)), int(np.round(yl))
        st = self.stamp(zone, xl - x0, yl - y0)
        r = self.reach
        out = np.zeros((h, w))
        ya, xa = y0 - r, x0 - r
        cy0, cy1 = max(ya, 0), min(y0 + r + 1, h)
        cx0, cx1 = max(xa, 0), min(x0 + r + 1, w)
        if cy1 > cy0 and cx1 > cx0:
            out[cy0:cy1, cx0:cx1] = st[cy0 - ya:cy1 - ya, cx0 - xa:cx1 - xa]
        return out


def _lagrange4(t: float) -> tuple[float, float, float, float]:
    """Cubic Lagrange weights at nodes -1, 0, 1, 2 for 0 <= t < 1."""
    return (-t * (t - 1) * (t - 2) / 6.0,
            (t + 1) * (t - 1) * (t - 2) / 2.0,
            -(t + 1) * t * (t - 2) / 2.0,
            (t + 1) * t * (t - 1) / 6.0)


def solve(cols: list[np.ndarray], d: np.ndarray,
          iv: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted linear least squares.

    Args:
        cols: Design-matrix columns.
        d: Data.
        iv: Inverse variances.

    Returns:
        Tuple (coefficients, covariance, chi2).  The covariance falls
        back to the pseudo-inverse if the normal matrix is singular.
    """
    a = np.column_stack(cols)
    sw = np.sqrt(iv)
    aw, dw = a * sw[:, None], d * sw
    c, *_ = np.linalg.lstsq(aw, dw, rcond=None)
    ata = aw.T @ aw
    try:
        cov = np.linalg.inv(ata)
        if not np.all(np.isfinite(np.diag(cov))) or np.any(np.diag(cov) < 0):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(ata)
    return c, cov, float(np.sum((dw - aw @ c) ** 2))


def _components(
    xl: np.ndarray, yl: np.ndarray, g: np.ndarray, it: int, w: int,
    h: int, reach: int, p: PsfParams,
) -> list[list]:
    """Select and merge Gaia sources for one box.

    Returns:
        List of [x, y, flux weight] in box coordinates, target first.
    """
    gt = g[it]
    r = np.hypot(xl - xl[it], yl - yl[it])
    keep = ((xl > -reach - 0.5) & (xl < w + reach - 0.5)
            & (yl > -reach - 0.5) & (yl < h + reach - 0.5))
    inbox = (xl > -0.5) & (xl < w - 0.5) & (yl > -0.5) & (yl < h - 0.5)
    keep &= ((g < min(gt + p.dg_keep, p.g_abs))
             | ((r < p.near_px) & (g < gt + p.dg_keep_near)))
    keep &= inbox | (g < gt + p.dg_out)
    keep[it] = True
    order = [it] + [i for i in np.argsort(g, kind="stable")
                    if keep[i] and i != it]
    fg = 10 ** (-0.4 * g)
    comps: list[list] = []
    for i in order:
        for c in comps:
            if np.hypot(c[0] - xl[i], c[1] - yl[i]) < p.merge_px:
                wsum = c[2] + fg[i]
                c[0] = (c[0] * c[2] + xl[i] * fg[i]) / wsum
                c[1] = (c[1] * c[2] + yl[i] * fg[i]) / wsum
                c[2] = wsum
                break
        else:
            comps.append([xl[i], yl[i], fg[i]])
    return comps


def fit_star(
    img: np.ndarray,
    flags: np.ndarray,
    var: np.ndarray,
    zpsf: ZonePsf,
    x: np.ndarray,
    y: np.ndarray,
    g: np.ndarray,
    it: int,
    omega_sr: float,
    params: Optional[PsfParams] = None,
) -> dict:
    """Forced PSF fit of one star on one image.

    Args:
        img: Full IMAGE array (MJy/sr).
        flags: Full FLAGS array.
        var: Full VARIANCE array.
        zpsf: Zone PSFs of the image.
        x: 0-based pixel x of the Gaia sources around the star.
        y: 0-based pixel y of the same sources.
        g: Gaia G of the same sources (NaN replaced by 21).
        it: Index of the star itself in x, y, g.
        omega_sr: Pixel solid angle (OMEGA_MEDIAN) in sr.
        params: Fit parameters.

    Returns:
        Dict with the PSF_DTYPES columns (without the ``psf_`` prefix
        for brevity: status, flux_jy, ...).
    """
    p = params or PsfParams()
    out: dict = {"status": 0}
    hb = p.half
    ny, nx = img.shape
    xt, yt = float(x[it]), float(y[it])
    x0, y0 = int(np.round(xt)) - hb, int(np.round(yt)) - hb
    nbox = 2 * hb + 1
    if x0 < 0 or y0 < 0 or x0 + nbox > nx or y0 + nbox > ny:
        out["status"] = 1
        return out
    sl = (slice(y0, y0 + nbox), slice(x0, x0 + nbox))
    im = np.asarray(img[sl], float)
    fl = np.asarray(flags[sl]).astype(np.int64)
    va = np.asarray(var[sl], float)
    good = ((fl & PSF_BITMASK) == 0) & np.isfinite(im) & np.isfinite(va)
    good &= va > 0
    c3 = (slice(hb - 1, hb + 2), slice(hb - 1, hb + 2))
    out["n_bad_c3"] = int((~good[c3]).sum())
    if out["n_bad_c3"] > p.max_bad_c3 or good.sum() < p.min_good:
        out["status"] = 2
        return out

    xl, yl = x - x0, y - y0
    comps = _components(xl, yl, g, it, nbox, nbox, zpsf.reach, p)
    fg_t = 10 ** (-0.4 * g[it])
    out["blend_frac_g"] = float((comps[0][2] - fg_t) / fg_t)
    zone = zpsf.zone(xt, yt)
    psum = zpsf.psum(zone)
    sh = SHIFT0[zpsf.kind]
    mods = [zpsf.render(zone, c[0] + sh[0], c[1] + sh[1], nbox, nbox)
            for c in comps]
    ok = [0] + [j for j in range(1, len(mods))
                if mods[j][good].sum() > p.drop_frac * psum]
    out["n_drop"] = len(mods) - len(ok)
    comps = [comps[j] for j in ok]
    mods = [mods[j] for j in ok]
    nc = len(comps)

    yy, xx = np.mgrid[0:nbox, 0:nbox]
    u, v = (xx - hb) / hb, (yy - hb) / hb
    bgt = [np.ones_like(u), u, v]
    if p.bg_order >= 2:
        bgt += [u ** 2, v ** 2, u * v]
    cols = [m[good] for m in mods] + [b[good] for b in bgt]
    coef, cov, chi2 = solve(cols, im[good], 1.0 / va[good])
    dof = int(good.sum()) - nc - len(bgt)
    k_jy = 1e6 * omega_sr * psum
    sig = np.sqrt(np.clip(np.diag(cov)[:nc], 0, None))
    rho = [cov[0, i] / (sig[0] * sig[i]) for i in range(1, nc)
           if sig[0] > 0 and sig[i] > 0]

    # Neighbour-subtracted aperture: data minus fitted neighbours and
    # plane within ap_r of the target; bad pixels take the target model.
    plane = sum(coef[nc + i] * b for i, b in enumerate(bgt))
    nbm = (sum(coef[i] * mods[i] for i in range(1, nc)) if nc > 1
           else np.zeros_like(im))
    clean = np.where(good, im - nbm - plane, coef[0] * mods[0])
    ap = np.hypot(xx - (comps[0][0] + sh[0]),
                  yy - (comps[0][1] + sh[1])) <= p.ap_r
    if dof < p.min_dof:
        out["status"] = 5
    out.update(
        flux_jy=coef[0] * k_jy,
        flux_err_jy=sig[0] * k_jy,
        chi2nu=chi2 / max(dof, 1),
        n_comp=nc,
        nb_flux_jy=float(np.sum(coef[1:nc])) * k_jy,
        bkg_mjysr=coef[nc],
        min_corr=float(np.min(rho)) if rho else 0.0,
        apsub_jy=float(clean[ap].sum()) * 1e6 * omega_sr,
        apsub_err_jy=float(np.sqrt(np.sum(np.where(good, va, 0)[ap])))
        * 1e6 * omega_sr,
        ap_ee=float(mods[0][ap].sum() / psum),
        zone=zone,
    )
    return out


def fit_image(
    img: np.ndarray,
    flags: np.ndarray,
    var: np.ndarray,
    zpsf: ZonePsf,
    omega_sr: float,
    nb_x: np.ndarray,
    nb_y: np.ndarray,
    nb_g: np.ndarray,
    starts: np.ndarray,
    targets: np.ndarray,
    params: Optional[PsfParams] = None,
) -> dict[str, np.ndarray]:
    """Fit every star of one image.

    Args:
        img: IMAGE array.
        flags: FLAGS array.
        var: VARIANCE array.
        zpsf: Zone PSFs of the image.
        omega_sr: Pixel solid angle in sr.
        nb_x: Concatenated pixel x of the Gaia sources of all stars.
        nb_y: Concatenated pixel y.
        nb_g: Concatenated G (NaN replaced by 21).
        starts: (n + 1,) offsets of each star's sources in nb_*.
        targets: (n,) local index of each star within its slice, or -1.
        params: Fit parameters.

    Returns:
        Dict of PSF_DTYPES columns, one entry per star.
    """
    n = len(targets)
    rows = empty_psf_rows(n)
    for i in range(n):
        a, b = int(starts[i]), int(starts[i + 1])
        if targets[i] < 0:
            rows["psf_status"][i] = 3
            continue
        try:
            res = fit_star(img, flags, var, zpsf, nb_x[a:b], nb_y[a:b],
                           nb_g[a:b], int(targets[i]), omega_sr, params)
        except Exception as exc:  # noqa: BLE001 - keep the image
            log.debug("PSF fit failed for star %d: %r", i, exc)
            rows["psf_status"][i] = 4
            continue
        for k, val in res.items():
            rows["psf_" + k][i] = val
    return rows


__all__ = [
    "Neighbours",
    "PSF_BITMASK",
    "PSF_DTYPES",
    "PsfParams",
    "SHIFT0",
    "STATUS",
    "ZonePsf",
    "empty_psf_rows",
    "fit_image",
    "fit_star",
    "psf_header_transposed",
    "solve",
]
