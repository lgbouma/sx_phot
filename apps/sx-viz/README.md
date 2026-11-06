# sx-viz

Viewer for the bulk SPHEREx spectra of the TARS (Boyle et al. 2026) rotators:
a server-side filtered table of all stars in a bulk run, and a star page with
the spectrum (interactive, plotly), SED fits against the SPHINX and
Castelli-Kurucz model grids, sky and SPHEREx cutouts, TARS/Gaia information
and a verdict store for vetting unresolved-binary ("red excess") candidates.

Everything is precomputed except the on-demand pieces (SED refits with
custom options, cutouts, report PNGs), which are cached on disk on first
view.  Design follows `cpv/apps/cpv-viz`: static HTML/JS, a stdlib
`ThreadingHTTPServer`, JSON endpoints, no framework.

## Data flow

```
<run>/bystar/bucket=N/            gathered photometry rows (bulk pipeline)
        |  drivers/build_viz_products.py   (~35 s per 1024-star bucket per worker)
        v
<run>/viz/spectra/bucket=N.parquet   one row group per star (legacy columns
                                     + spline model + fit mask)
<run>/viz/summary.parquet            one row per star (counts, S/N, band
                                     fluxes, colours, bucket/row_group)
<run>/viz/binned_*.npy               R=40 binned spectra, 0.75-5 um (76 bins)
        |  drivers/fit_viz_seds.py        (~4 ms per star per worker)
        v
<run>/viz/sed.parquet                single / blue-anchored / binary fits
        |  server.py (+ TARS table)
        v
http://wh3:8051/
```

Model grids live under `/ar0/local/models` (`sphinx/`, `ck04_dl/`,
`mamajek/`); `sx_phot.sedmodels.load_grid` resamples them once and caches
the result in `/ar0/local/models/cache/`.

## Build / refresh

```bash
PY=$HOME/miniconda3/envs/py311_sx/bin/python
RUN=/ar0/SPHEREX/bulkphot/tars_v0
# 1. per-star products (skips buckets already built; --overwrite to redo)
nice -n 19 $PY drivers/build_viz_products.py --run-dir $RUN --workers 12
# 2. SED fits for every star
nice -n 19 $PY drivers/fit_viz_seds.py --viz-dir $RUN/viz --workers 8
# 3. (re)start the server
apps/sx-viz/scripts/ensure_server.sh
```

After a run is extended by the daily download, re-run the bulk gather, then
`apps/sx-viz/scripts/refresh_products.sh`, which runs steps 1-3 (step 1
with `--overwrite`, since every bucket changes) and logs to
`<run>/viz/refresh.log`; about 45 min for the TARS run.  The server loads
the table at start-up (~30 s) and must be restarted to see new products.

To warm the on-demand caches for a vetting list (reports, SPHEREx and sky
cutouts), e.g. the top 500 red-excess stars:

```bash
$PY drivers/precompute_viz_cache.py --run-dir $RUN --limit 500 \
    --query "min_red_excess_sig=5&min_n_used=100&sort=red_excess_sig&desc=1"
```

## Server

```bash
$PY apps/sx-viz/server.py --run-dir $RUN --port 8051
```

Suggested cron entries (same pattern as cpv-viz):

```
@reboot      /ar0/PROJ/luke/proj/sx_phot/apps/sx-viz/scripts/ensure_server.sh
*/5 * * * *  /ar0/PROJ/luke/proj/sx_phot/apps/sx-viz/scripts/ensure_server.sh
```

Log: `~/local/sx-viz/server.log`.  Verdicts: `apps/sx-viz/data/verdicts.json`
(+ `.csv`, `_history.csv`).  If `~/.config/sx-viz/token` exists, POSTs must
carry it (the star page asks once and stores it in the browser).

### Endpoints

| Endpoint | Returns |
| --- | --- |
| `/api/meta` | counts, column lists, build date, survey names |
| `/api/table?min_<col>=&max_<col>=&flag_<col>=0/1&q=&sort=&desc=&offset=&limit=&cols=&ids_only=1` | filtered, sorted page (`total`, `rows`) |
| `/api/star/<tic>` | table row, spectrum arrays, binned spectrum, default SED fit, cutout picks, verdict |
| `/api/sed/<tic>?grids=&logg=&mh=&blue_max_um=&err_floor=&lam_min_um=&lam_max_um=&binary=` | on-demand fit + model curves |
| `/api/cutout/<tic>?survey=dss2r|dss2b|2massj|2massk|wise1|wise2` | SkyView JPEG, 3', cached |
| `/api/spxcut/<tic>/<image_key>.png` | SPHEREx cutout (4', aperture + annulus, flagged pixels in orange), cached |
| `/api/report/<tic>.png` | matplotlib summary figure, cached (`?refresh=1` to redo) |
| `/api/tarsvet/<tic>` | TARS vetting JPEG URLs proxied from lcviz on wh2 (`SX_LCVIZ_URL`, default `http://lcviz.lgbouma.com:8056`); the browser hotlinks the images |
| `/api/verdicts` | GET all / POST `{ticid, verdict, notes}` |

Timing on wh3 (warm): table query 25 ms, star payload 20-150 ms, SED refit
15 ms, SPHEREx cutout 0.3 s, SkyView cutout ~1 s (network).

## Pages

**index.html**: sidebar presets and min/max filters on any numeric column,
flag chips (three-state), column picker (remembered per browser), sortable
headers, paging, `r` random star, CSV of the current page.  Filters live in
the URL.  Clicking a row stores the current filtered id list (up to 2000)
so the star page can page through it.

**star.html?ticid=**: left column verdict (keys `1` binary candidate, `2`
single, `3` unsure, `4` junk, `0` clear; notes; `n` focuses notes; "next
after verdict" auto-advances), TARS/Gaia and spectrum summary tables,
external links.  Centre: spectrum (log-log; used points, hollow = used but
soft-flagged, clipped, masked with the reason on hover; spline; binned
points; SED model curves; clicking a point loads that image's SPHEREx
cutout), ratio of the binned spectrum to the blue-anchored model, then
one TARS vetting JPEG from lcviz (`[`/`]` or the buttons page through the
sectors; neighbours are preloaded), then the cutouts (two SkyView fields
with the aperture and annulus drawn, the selected SPHEREx image, six
SPHEREx cutouts spread over wavelength).  Right: the SED model comparison table (single star / two
stars / star + dust with chi2, BIC and delta-BIC; lowest BIC
highlighted), fit details, and the static report PNG.  `j`/`k`
next/previous star.  The on-demand refit endpoint `/api/sed/<tic>?...`
still exists (link "fit JSON") but has no controls on the page.

## Quality masks (sx_phot.viz_products.quality_masks)

| mask_bits | reason | rule |
| --- | --- | --- |
| 1 | hard flag | SPHEREx bits 2, 6, 7, 9, 10, 11, 15 in the aperture |
| 2 | soft flag | bits 0 (TRANSIENT) or 1 (OVERFLOW) only; the point is used if it agrees with the clean-point spline within 3 sigma (3% error floor) and a factor 1.25, else masked |
| 4 | detector edge | star centre within 2 px of the array edge (part of the aperture is off the detector; median flux ratio 0.86 at 0-1 px) |
| 8 | gross outlier | flux / running median (21 points) outside [1/1.8, 1.8] and more than 5 sigma away |

Spline clipping (3 sigma, iterative) then runs on the remaining points;
`fit_mask` marks the points that survive and feed the binned spectrum.

## SED fitting (sx_phot.sedfit)

Data: the R=40 binned spectrum (inverse-variance means of the unmasked,
unclipped points per bin), errors with a 2 % floor added in quadrature.
Models: SPHINX (Teff 2000-4000 K, C/O = 0.5) and Castelli-Kurucz 2004
(4250 K and up), interpolated in log flux to 25 K steps at fixed log g
(dwarf value from the Pecaut & Mamajek table unless given) and [M/H].  The
scale is `(R/d)^2`, so with the Gaia parallax it gives a radius;
`radius_ratio_*` compares that with the dwarf radius at the fitted Teff.

| Product | Fit window | Reported |
| --- | --- | --- |
| `full` | all bins | `teff_full` (+ delta-chi2 = 1 range), `chi2_full`, `radius_full`, `radius_ratio_full` |
| `blue` | lambda < 1.7 um | `teff_blue`, `excess_b3..b6` (fractional data/model excess per SPHEREx band), `red_excess` (lambda > 2.4 um), `red_excess_sig` |
| `binary` | all bins, primary + cooler secondary | `teff_bin1`, `teff_bin2`, `radius_bin1/2`, `chi2_binary`, `dchi2_binary` |
| `disk` | all bins, star + blackbody (200-2000 K) | `teff_disk_star`, `teff_disk`, `f_disk` (L_dust / L_star), `radius_disk`, `chi2_disk`, `dchi2_disk` |
| BIC | chi2 + k ln n, k = 2 / 4 / 4 | `bic_single`, `bic_binary`, `bic_disk`, `dbic_binary`, `dbic_disk` (single minus model), `dbic_disk_vs_binary` (binary minus disk) |
| population | per 100 K bin of the fitted Teff | `red_excess_rel`, `*_rel_sig`, `radius_ratio_full_rel`, `dbic_*_pct` (percentile rank of the delta-BIC at that Teff) |

The single-star fit leaves chi2/dof ~ 3 (median) because of model and
calibration systematics, so delta-BIC thresholds of 10 or 100 select
57% and 29% of all stars and track the Teff-dependent systematics rather
than binarity; use the `_pct` columns (same selected fraction at every
Teff) or thresholds in the tail (delta-BIC > 1000: 1.3%).
`scripts/plot_dchi2_binary_vs_period.py` makes the histogram and the
fraction-above-threshold versus rotation period plots in
`results/sx_viz_analysis/`.

Caveats: a positive red excess also appears for blended neighbours (check
the cutouts), circumstellar dust, bad backgrounds and giants; SPHINX Teff
run ~200-300 K hotter than the TARS photometric Teff near 3500-4000 K; the
few-percent flux steps at the band boundaries (TODO.txt) enter the band
excesses directly, so compare stars at similar Teff rather than trusting
absolute values.

## Tests

```bash
$PY -m pytest tests/test_sedfit.py tests/test_viz_products.py -q
```
