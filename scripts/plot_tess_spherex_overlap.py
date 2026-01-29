from inspect import signature

import astropy.units as u
from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord
from matplotlib import patheffects
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import tesswcs
from tesswcs.locate import get_observability_mask

from sx_phot.sx_pointing import in_spherex_deep_fields


def main():
    pointings = tesswcs.pointings[["RA", "Dec", "Roll", "Sector"]].to_pandas()
    selectsectors = pointings[(pointings["Sector"] >= 92) & (pointings["Sector"] <= 99)]
    if selectsectors.empty:
        raise RuntimeError()

    # Grid of RA and Dec to check; increase resolution to trade speed for detail.
    RA, Dec = np.mgrid[:360:200j, -90:90:121j]
    sky_coords = SkyCoord(RA, Dec, unit="deg")

    nobs = np.zeros(RA.shape, dtype=float)
    for ra, dec, roll, sector in tqdm(
        selectsectors[["RA", "Dec", "Roll", "Sector"]].values,
        desc="Pointing",
        leave=True,
        position=0,
    ):
        for camera in range(1, 5):
            for ccd in range(1, 5):
                wcs = tesswcs.WCS.predict(ra, dec, roll, camera, ccd)
                mask = get_observability_mask(wcs, sky_coords).astype(int)
                nobs += mask

    fig, ax = plt.subplots(dpi=250)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_extremes(under="white")
    im = ax.pcolormesh(RA, Dec, nobs, cmap=cmap, vmin=1, vmax=6, shading="nearest")
    _add_spherex_deep_field_circles(ax)

    cbar = plt.colorbar(im, ax=ax)
    ax.set(
        xlabel="RA [deg]",
        ylabel="Dec [deg]",
        title="TESS (S92-97) SPHERExDF simult. (06.2025 - 01.2026)",
    )
    cbar.set_label("Number of Observations")

    outfile = "../results/tess_x_spherex_coverage.png"
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


def _add_spherex_deep_field_circles(ax: plt.Axes) -> None:
    radius_deg, south_elon_deg, south_elat_deg = _deep_field_params()

    frame = GeocentricTrueEcliptic()
    north_center = SkyCoord(lon=0.0 * u.deg, lat=90.0 * u.deg, frame=frame).icrs
    south_center = SkyCoord(
        lon=south_elon_deg * u.deg,
        lat=south_elat_deg * u.deg,
        frame=frame,
    ).icrs

    _plot_deep_field_circle(
        ax,
        center=north_center,
        radius_deg=radius_deg,
        color="deepskyblue",
        label="SPHEREx NDF",
    )
    _plot_deep_field_circle(
        ax,
        center=south_center,
        radius_deg=radius_deg,
        color="gold",
        label="SPHEREx SDF",
    )

    ax.legend(loc="lower right", frameon=True, fontsize=8)


def _deep_field_params() -> tuple[float, float, float]:
    params = signature(in_spherex_deep_fields).parameters
    radius_deg = float(params["radius_deg"].default)
    south_elon_deg = float(params["south_center_elon_deg"].default)
    south_elat_deg = float(params["south_center_elat_deg"].default)
    return radius_deg, south_elon_deg, south_elat_deg


def _plot_deep_field_circle(
    ax: plt.Axes,
    center: SkyCoord,
    radius_deg: float,
    color: str,
    label: str,
) -> None:
    angles = np.linspace(0.0, 360.0, 361) * u.deg
    circle = center.directional_offset_by(angles, radius_deg * u.deg)
    coords = np.column_stack([circle.ra.deg, circle.dec.deg])
    fill = Polygon(
        coords,
        closed=True,
        facecolor=color,
        edgecolor="none",
        alpha=0.2,
        zorder=4,
    )
    ax.add_patch(fill)
    line = ax.plot(
        circle.ra.deg,
        circle.dec.deg,
        color=color,
        linewidth=2.5,
        label=label,
        zorder=5,
    )[0]
    line.set_path_effects(
        [
            patheffects.Stroke(linewidth=4.0, foreground="black"),
            patheffects.Normal(),
        ]
    )


if __name__ == "__main__":
    main()
