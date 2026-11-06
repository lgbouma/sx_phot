from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sx_phot.visualization import plot_simple_spectrum, plot_spectrum_with_spline


def test_plot_spectrum_with_spline_writes() -> None:
    wavelength = np.linspace(1.0, 2.0, 30)
    flux = 1.0 + 0.1 * np.sin(wavelength)
    flux_err = np.full_like(wavelength, 0.05)
    masked = np.zeros_like(wavelength, dtype=bool)
    masked[0] = True

    dense_wave = np.linspace(1.0, 2.0, 300)
    dense_model = 1.0 + 0.1 * np.sin(dense_wave)

    output_dir = (
        Path(__file__).resolve().parents[1] / "results" / "test_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / "spline_plot.png"
    fig, _ = plot_spectrum_with_spline(
        wavelength_um=wavelength,
        flux_jy=flux,
        flux_err_jy=flux_err,
        masked=masked,
        dense_wavelength_um=dense_wave,
        dense_model_flux_jy=dense_model,
        title="Spline fit test",
        output_path=outpath,
    )

    plt.close(fig)
    assert outpath.exists()


def test_plot_simple_spectrum_writes() -> None:
    wavelength = np.linspace(1.0, 2.0, 30)
    flux = 1.0 + 0.1 * np.sin(wavelength)
    flux_err = np.full_like(wavelength, 0.05)
    masked = np.zeros_like(wavelength, dtype=bool)
    masked[0] = True

    output_dir = (
        Path(__file__).resolve().parents[1] / "results" / "test_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outpath = output_dir / "simple_spectrum.png"
    fig, _ = plot_simple_spectrum(
        wavelength_um=wavelength,
        flux_jy=flux,
        flux_err_jy=flux_err,
        masked=masked,
        ra_deg=120.0,
        dec_deg=-30.0,
        output_path=outpath,
    )

    plt.close(fig)
    assert outpath.exists()


def test_cutout_inset_has_asinh_stretch_and_colorbar():
    """The inset uses an asinh stretch and adds a labelled colorbar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.visualization import AsinhStretch

    from sx_phot.visualization import _add_cutout_inset

    rng = np.random.default_rng(0)
    img = rng.normal(1.0, 0.1, (64, 64))
    img[30, 30] = 500.0
    fig, ax = plt.subplots()
    _add_cutout_inset(fig, ax, img, 0.93, None, 10.0, 20.0, 2.0)
    assert len(fig.axes) == 3  # host, inset image, colorbar
    im = fig.axes[1].images[0]
    assert isinstance(im.norm.stretch, AsinhStretch)
    assert im.norm.vmax > 100.0
    assert fig.axes[2].get_ylabel() == "MJy/sr"
    plt.close(fig)
