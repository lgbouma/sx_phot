from pathlib import Path
import re

import numpy as np
import pandas as pd

from sx_phot.splinefit import (
    fit_band_bspline,
    fit_spherex_spectrum_bspline,
    spherex_band_definitions,
)
from sx_phot.visualization import plot_spectrum_with_spline


ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "drivers" / "test_results" / (
    "sxphot_cache_TIC_300651846_ra114p36670927895_-66p75737858669_annulus.csv"
)


def _coerce_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == bool:
        return mask
    if np.issubdtype(mask.dtype, np.number):
        return mask != 0
    text = np.char.lower(mask.astype(str))
    return np.isin(text, ["1", "true", "t", "yes", "y"])


def _parse_star_id_and_coords(stem: str) -> tuple[str, float, float]:
    pattern = r"_ra(?P<ra>[-0-9p]+)_(?P<dec>-?[0-9p]+)"
    match = re.search(pattern, stem)
    if not match:
        raise ValueError(f"Unable to parse coords from {stem}.")

    star_id = stem[: match.start()]
    if star_id.startswith("sxphot_cache_"):
        star_id = star_id[len("sxphot_cache_") :]
    ra_deg = float(match.group("ra").replace("p", "."))
    dec_deg = float(match.group("dec").replace("p", "."))
    return star_id, ra_deg, dec_deg


def test_band_assignment_is_unique() -> None:
    bands = spherex_band_definitions()
    for i, band in enumerate(bands):
        assert band["assign_min_um"] >= band["band_min_um"]
        assert band["assign_max_um"] <= band["band_max_um"]
        if i > 0:
            prev = bands[i - 1]
            assert prev["assign_max_um"] <= band["assign_min_um"]


def test_fit_real_spectrum_smoke() -> None:
    assert TEST_CSV.exists(), f"Missing test spectrum: {TEST_CSV}"
    df = pd.read_csv(TEST_CSV)

    masked = None
    if "masked" in df.columns:
        masked = _coerce_mask(df["masked"].to_numpy())

    model_flux, fit_mask, dense_wave, dense_model, band_results = (
        fit_spherex_spectrum_bspline(
            wavelength_um=df["wavelength_um"].to_numpy(),
            bandwidth_um=df["bandwidth_um"].to_numpy(),
            flux_jy=df["flux_jy"].to_numpy(),
            flux_err_jy=df["flux_err_jy"].to_numpy(),
            masked=masked,
        )
    )

    n = len(df)
    assert model_flux.shape == (n,)
    assert fit_mask.shape == (n,)
    assert dense_wave.size == dense_model.size
    assert dense_wave.size > 0
    assert np.any(fit_mask)
    assert np.all(np.isfinite(model_flux[fit_mask]))

    expected_bands = {f"band{i}" for i in range(1, 7)}
    assert set(band_results.keys()) == expected_bands

    if masked is not None:
        assert not np.any(fit_mask & masked)

    output_dir = Path(__file__).resolve().parents[1] / "results" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{TEST_CSV.stem}_spline.png"
    star_id, ra_deg, dec_deg = _parse_star_id_and_coords(TEST_CSV.stem)
    knots = np.concatenate(
        [res.knot_vector_um for res in band_results.values() if res.knot_vector_um.size]
    )
    fig, _ = plot_spectrum_with_spline(
        wavelength_um=df["wavelength_um"].to_numpy(),
        flux_jy=df["flux_jy"].to_numpy(),
        flux_err_jy=df["flux_err_jy"].to_numpy(),
        masked=masked,
        fit_mask=fit_mask,
        model_flux_jy=model_flux,
        dense_wavelength_um=dense_wave,
        dense_model_flux_jy=dense_model,
        knot_wavelength_um=knots if knots.size else None,
        star_id=star_id,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        show_residuals=True,
        output_path=plot_path,
    )
    fig.clf()


def test_fit_band_clips_outlier() -> None:
    band = spherex_band_definitions()[0]
    rng = np.random.default_rng(123)
    n_points = 40

    wavelength = np.linspace(band["assign_min_um"], band["assign_max_um"], n_points)
    bandwidth = np.full(n_points, 0.01)
    flux = 1.0 + 0.02 * rng.standard_normal(n_points)
    flux_err = np.full(n_points, 0.02)

    outlier_idx = n_points // 2
    flux[outlier_idx] += 1.0

    res = fit_band_bspline(
        wavelength_um=wavelength,
        bandwidth_um=bandwidth,
        flux_jy=flux,
        flux_err_jy=flux_err,
        input_mask=np.ones(n_points, dtype=bool),
        band_name=str(band["name"]),
        band_min_um=float(band["band_min_um"]),
        band_max_um=float(band["band_max_um"]),
        assign_min_um=float(band["assign_min_um"]),
        assign_max_um=float(band["assign_max_um"]),
        max_iter=6,
    )

    outlier_pos = np.where(res.input_indices == outlier_idx)[0][0]
    assert not res.fit_mask[outlier_pos]
