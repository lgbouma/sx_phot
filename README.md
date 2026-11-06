# sx_phot

Minimal SPHEREx circular-aperture photometry utilities and driver.

## Install
- Install: `python -m pip install .`
- Editable + dev: `python -m pip install -e .[dev]`

## Thank you
Zafar Rustamkulov and Kishalay De for offering the starting point.
SPHEREx team for creating a wonderful instrument and data product.

## Viewer
`apps/sx-viz` serves the bulk TARS spectra (table, star pages, SED fits
against SPHINX and Castelli-Kurucz grids, cutouts, verdicts); see
`apps/sx-viz/README.md`.

## Machine learning
`sx_phot/specml.py` builds an empirical one-parameter spectral sequence
(template library) from the clean main-sequence stars of a bulk run,
`sx_phot/empfit.py` fits it to every star (single, two-star, disk), and
`drivers/run_umap.py` embeds the spectra and the residuals with UMAP.
`apps/sx-ml` serves the UMAP explorer and the sequence slider; see
`apps/sx-ml/README.md`.
