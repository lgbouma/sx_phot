# sx-ml

Machine-learning views of the bulk SPHEREx spectra of the TARS rotators:
an empirical one-parameter spectral sequence (template library), fits of
that library to every star (single, binary, disk), UMAP embeddings of the
spectral shapes and of the residuals, and two web pages to explore them.
Same design as `apps/sx-viz`: static HTML/JS, a stdlib
`ThreadingHTTPServer`, JSON and binary endpoints, no framework.  Star pages
link to sx-viz (port 8051).

## Products (`<run>/ml`)

```
<run>/viz/binned_*.npy, summary.parquet      (viz build; inputs)
        |  drivers/build_spectral_sequence.py      (~90 s, 32 threads)
        v
sequence.npz, sequence_nodes.csv             template library (sx_phot.specml.SpectralSequence)
sequence_train.parquet, sequence.json        training stars (t, chi2, inlier) and provenance
        |  drivers/fit_empirical.py                (~9 min, 40 workers)
        v
empfit.parquet                               per-star fits (sx_phot.empfit)
        |  drivers/run_umap.py                     (~1 h for both embeddings, 32 threads)
        v
umap_shape.npy, umap_resid.npy, pca_*.npz, umap.json
        |  drivers/plot_sequence_qa.py  ->  qa/seq_*.png
        |  drivers/plot_umap_qa.py      ->  results/sx_ml_analysis/umap_*.png
        |  scripts/analyze_empfit.py    ->  results/sx_ml_analysis/
        |  server.py
        v
http://wh3:8052/
```

### The sequence (`sx_phot/specml.py`)

Inputs are the R = 40 binned spectra (76 bins, 0.75-5 um), dereddened
with the catalog (GSP-Phot) A0 using a CCM89 law and turned into
normalized log fluxes (mean over the bins removed).  The training set is
the clean main sequence: all 76 bins present, median S/N >= 50, spline
ok, A0 <= 0.2, RUWE < 1.4, no Gaia NSS flag, no TESS contaminants, and
within (-0.4, +0.5) mag of the running-median main sequence in
(BP-RP0, M_G0).  Stars are ordered along one parameter `t` in [0, 1] by a
principal-curve iteration: node templates are per-bin medians (with a
local slope correction) of the stars in each node window; each star is
reassigned to the `t` whose interpolated template fits it best (free
scale, log space); `t` is rank-transformed; nodes are respaced uniformly
in arc length along the templates (end nodes at `t` = 0 and 1); stars
with chi2 above four times the local median are clipped from the medians
and the ranks.  Six iterations converge (rank correlation 0.9999).

The library stores per node: the normalized template, the population
scatter per bin (MAD; `scatter_int` has the median measurement error
removed), the absolute flux offset at 10 pc from the Gaia parallaxes,
the first three residual eigenvectors (an empirical second parameter),
the arc length, and catalog medians (TARS Teff, BP-RP0, M_G0, A0, Tmag,
period).  `t` = 0 is the blue end; `teff_mono` is a monotone version of
the median TARS Teff used to label `t`.

### The fits (`sx_phot/empfit.py`)

| key | model | free parameters |
|---|---|---|
| `single` | `s 10^T(t)` | t, s |
| `ms` | `c 10^A(t)` (star on the main sequence at the Gaia distance, `c = (plx/100 mas)^2`) | t |
| `binary` | `s [10^A(t1) + 10^A(t2)]`, t2 >= t1 (flux ratio from the main-sequence luminosities) | t1, t2, s |
| `binary_ms` | `c [10^A(t1) + 10^A(t2)]` | t1, t2 |
| `disk` | `s 10^T(t) + f B_nu(T_bb)`, T_bb 200-2000 K | t, s, T_bb, f |

Errors are the measurement errors plus the population scatter at the
star's `t` plus a 0.5 % floor, so chi2 compares a star with the single
star population: shared features (band edges, the 4.4-5 um band) are in
the templates and do not count as excess.  The median chi2/dof is 1.3
(the model-atmosphere fits in sx-viz sit at 3).  Outputs per star
(`empfit.parquet`, documented in `drivers/fit_empirical.py`): `t_single`,
`teff_emp`, `lum_ratio` / `dmag_ms` (SPHEREx overluminosity relative to
the main sequence at the Gaia distance), per-band fractional excesses and
significances, `red_excess`, `blue_excess`, residual eigenvector
amplitudes `resid_a1..3`, the binary and disk parameters, chi2 and BIC
differences (`dbic_binary`, `dbic_binary_ms`, `dbic_disk`,
`dbic_disk_vs_binary`) and percentile ranks at fixed `t` (`*_pct`).

### The embeddings (`drivers/run_umap.py`)

`shape`: normalized log spectra (missing bins imputed from the fitted
template) -> 12 PCs -> UMAP (n_neighbors 30, min_dist 0.05).  `resid`:
noise-whitened residuals from the single template -> 12 PCs -> UMAP.
Stars with fewer than 40 bins or median S/N < 10 are left out (NaN).

## Build / refresh

```bash
PY=$HOME/miniconda3/envs/py311_sx/bin/python
RUN=/ar0/SPHEREX/bulkphot/tars_v0
OMP_NUM_THREADS=32 nice -n 10 $PY drivers/build_spectral_sequence.py --viz-dir $RUN/viz
nice -n 10 $PY drivers/fit_empirical.py --viz-dir $RUN/viz --workers 40
$PY drivers/plot_sequence_qa.py --ml-dir $RUN/ml
NUMBA_NUM_THREADS=32 nice -n 10 $PY drivers/run_umap.py --viz-dir $RUN/viz
apps/sx-ml/scripts/ensure_server.sh
```

Re-run after each viz rebuild (the fits and embeddings depend on the
binned spectra).  The server loads everything at start-up (~40 s) and
must be restarted to see new products.

## Server

```bash
$PY apps/sx-ml/server.py --run-dir $RUN --port 8052
```

| Endpoint | Returns |
| --- | --- |
| `/api/meta` | counts, column list with ranges and notes, embeddings, UMAP settings |
| `/api/points?emb=shape` | float32 LE, 2 N: x0 y0 x1 y1 ... (NaN where not embedded) |
| `/api/col/<name>` | float32 LE, N values of a table column |
| `/api/ticids` | uint32 LE, N |
| `/api/star/<row>` (`?by=ticid`) | table row, binned spectrum (raw and dereddened), template curves, effective errors, embedding coordinates, sx-viz link |
| `POST /api/selection {rows}` | median residual spectrum (and 16-84 %), median normalized spectrum, `t` histogram, column medians, TIC ids |
| `/api/sequence` | the library (nodes, templates, scatter, absolute offsets, arc, Teff map, residual eigenvectors, node summaries) |
| `/api/sequence/at?t=&n=&source=&seed=` | template at `t`, model atmosphere at the mapped Teff, example stars (training set, or `source=all`) |

Log: `~/local/sx-ml/server.log`.  `SX_VIZ_URL` (default `http://wh3:8051`)
sets the sx-viz link base.

## Pages

**umap.html**: WebGL2 scatter of all embedded stars (custom renderer,
`scatter.js`; no library; without WebGL2 it falls back to splatting the
points into a 2D canvas, ~150 ms per redraw for 1e6 points, and says so
in the HUD).  Script errors are shown in the status bar.  Sidebar: embedding, colour column (any
numeric column; searchable), colour range and map, point size and
opacity, a range filter on any column, TIC lookup.  Drag pans, wheel
zooms, click loads the star (binned dereddened spectrum with the single /
binary / disk template curves, residuals with the population band, the
fit and catalog columns), shift+drag lassos a group: the selection panel
shows the median residual from each star's own single template with the
16-84 % band, the median normalized spectrum against the template at the
median `t`, the `t` histogram and column medians; CSV export of the
selected rows.  View state is in the URL hash.

**sequence.html**: slider along the sequence in arc length (equal
spectral change per step), `t` (equal numbers of stars) or Teff; shows
the template (normalized or absolute at 10 pc) with the population band,
neighbouring templates, example training stars and the SPHINX / ck04
model at the mapped Teff; a second slider adds the first residual
eigenvector.  Below: percent panel (template / model, eigenvectors,
scatter), the whole library as an image (click to jump), the Teff and
M_G0 mapping, and the scatter and star counts along the sequence.

## Browser check without a display

wh3 has no X server; headless Firefox (`/usr/bin/firefox`, no WebGL) can
still screenshot a page.  It needs a writable HOME and profile, and
`--screenshot` fires at the load event, before the page's fetches finish,
so delay the load event with an `<img>` served slowly (a 10-line Python
HTTP server that sleeps) from a test copy of the page:

```bash
T=/ar0/local/tmp/claude-1000/ff; mkdir -p $T/home $T/prof
MOZ_HEADLESS=1 HOME=$T/home firefox --headless --no-remote --profile $T/prof \
  --window-size=1600,1000 --screenshot $T/shot.png "http://wh3:8052/umap.html"
```

## Tests

```bash
$PY -m pytest tests/test_specml.py tests/test_empfit.py tests/test_sxml_server.py -q
```
