# Data products on wh3 (read this before answering "where are the spectra")

Notes for agents and humans.  Absolute paths; everything below lives on
the wh3 server (`/ar0`, spinning MegaRAID).  Keep this file current when
runs are added or completed (see TODO.txt / DONE.txt for history).

## Bulk SPHEREx spectra (image-centric pipeline)

One pass over every local Level-2 image measures every catalog star on
it (`sx_phot/bulk_phot.py`); rows are then regrouped by star
(`sx_phot/bulk_products.py`).  Cost is set by disk reads, not by catalog
size: ~14 files/s with 64 workers, ~30 h for the full 1.6M-image mirror.
Small target lists open only the images with a candidate star.
`run_bulk_phot.py --psf --neighbours <parquet>` adds Gaia-forced PSF
photometry (`sx_phot/psf_phot.py`, merged 2026-11-10 from PR #1):
every Gaia DR3 source in an 11 x 11 px box is fitted with the zone
PSF, giving `psf_flux_jy`, `psf_status` (0 ok) and the other `psf_*`
columns listed in `drivers/README.txt`.  The neighbour table comes from
`drivers/query_gaia_neighbours.py` (Gaia TAP, ~8 s per 300 stars).
qr3 files carry an effective PSF (`EPSF`), which `psf_phot` point-samples
(fixed 2026-11-10; PSF rows written before that for qr3 images are ~6%
high, including the SPYGLASS-IV run below).
The gather needs ~1 h and ~10 GB per worker for table 4; run it (and
anything else large) under `systemd-run --user -p MemoryMax=64G
-p MemorySwapMax=0` so a runaway cannot push wh3 into swap.

| Run dir | Catalog | Stars | Status (2026-11-05) |
|---|---|---|---|
| `/ar0/SPHEREX/bulkphot/tars_v0` | TARS table 2 (`/ar0/local/TARS/tars_table_2.feather`) | 1,046,317 | complete through the 2026-11-03 download; 550.3M rows, 1022 buckets |
| `/ar0/SPHEREX/bulkphot/tars4_v0` | TARS table 4, de-duplicated (`/ar0/local/TARS/zenodo_19917941/tars_table_4_unique_stars.parquet`) | 7,481,030 | complete 2026-11-05 (images through the 2026-11-03 download, 2026W40): 3.57B rows, 7306 buckets, 164 GB bystar/; identical rows to tars_v0 for shared stars; 200-star QA sample in `qa_sample/` (csv + plots); supersedes tars_v0 (table 2 is a subset).  viz/ and ml/ products built 2026-11-06 (see ML products); sx-viz still serves tars_v0 |
| `/ar0/SPHEREX/bulkphot/spyglass_viia_psp_v0` | collaborator list (`/ar0/local/sx-targets/spyglass_viia_psp_members_minimal.csv`, Gaia DR3 ids) | 1,161 | complete; `csv/` is the hand-off |
| `/ar0/PROJ/fperezpa/spyglass_full_psf/run_psf` | SPYGLASS-IV members with Psp > 0.5 (`../spyglass4_psp_astrometry.csv`, Gaia DR3 ids, epoch 2016.0) | 183,458 | complete 2026-11-07 (Facundo Perez Paolino): aperture plus Gaia-forced PSF photometry (`--psf`, `psf_*` columns), 74.6M rows, 1434 buckets of 128 stars, 15 GB; neighbour table `../neighbours.parquet` (1.4 GB); SED fits in `../sed/` |

Run-dir layout:

```
catalog.parquet        star_idx, ticid (the catalog id; Gaia DR3 id for
                       non-TIC lists), ra, dec, pmra, pmdec (epoch 2016.0)
index.parquet          copy of the image index used by the run
config.json            PhotConfig (aperture 2 px, annulus 6-8 px, ...)
phot/<shard>/pass_*.parquet   rows per image, shard = (week_cam, ver, det)
bystar/bucket=N/       rows regrouped by star (1024 stars per bucket,
                       bucket = star_idx // 1024); bystar.json has counts
csv/                   per-star legacy CSVs (only for hand-off runs)
qa/                    QA PNGs (spline fit + cutout inset) and summaries
all_rows.parquet       every row in one table (small runs only)
README.txt             hand-off description (small runs only)
run_bulk_phot.log, pipeline_<stamp>.log   progress; one line per shard
```

Row columns (phot/ and bystar/): star_idx, ticid, image_key, x, y,
flux_jy (annulus-subtracted), flux_err_jy, flux_raw_jy, bkg_jy, zodi_jy,
flag_or, n_bad_pix, n_ap_pix, n_ann_pix, bkg_mode, wavelength_um,
bandwidth_um, mjd_avg.  Full definitions: `drivers/README.txt`.

How to get a spectrum:

```
# one star (TIC id) from a gathered run -> legacy sxphot_cache CSVs
python drivers/materialize_bulk_star.py --run-dir /ar0/SPHEREX/bulkphot/tars_v0 \
    --ticid 300651846 --out-dir /some/dir
# every star of a run (small runs), with --id-prefix for non-TIC ids
python drivers/materialize_bulk_run.py --run-dir <run> --out-dir <run>/csv
# a QA sample of a big run (reads only the needed buckets)
python drivers/materialize_bulk_run.py --run-dir <run> --out-dir <run>/qa_sample/csv --sample 200
# QA plots from a materialized run
python drivers/plot_bulk_qa.py --csv-dir <run>/csv --out-dir <run>/qa
# in Python
from sx_phot.bulk_products import load_star_rows, to_legacy_frame
```

New lists: `scripts/run_bulk_phot_tars.sh` with `CATALOG`, `RUN_DIR`,
`NPROC`, `EXTRA="--id-col <col>"`; it is incremental (re-run after the
nightly image download to add new weeks).  See `scripts/README.txt`.

## Images and index

- Level-2 MEF mirror: `/ar0/SPHEREX/spherex_l2/<week_cam>/<ver>/<det>/`
  (66 MB each; qr2 through 2026W29, qr3 from 2026W30; downloaded nightly
  at 03:00 by `scripts/run_get_spherex_daily.sh`).
- Image index: `/ar0/SPHEREX/index/image_index.parquet` (one row per
  obs_id and detector: path, footprint, times, release) built by
  `drivers/build_image_index.py` from an IRSA ObsCore dump; snapshots
  `/ar0/SPHEREX/index/obscore_*.csv` are reused if IRSA is down.

## TARS catalogs

- `/ar0/local/TARS/tars_table_2.feather`: table 2 (rotators), one row
  per star with the Gaia columns (`BpmRp0`, `phot_g_mean_mag_0`,
  `parallax`, `extinction_a0`, `ruwe`, `non_single_star`, `teff`, ...).
- `/ar0/local/TARS/zenodo_19917941/tars_table_4.feather`: table 4 per
  (TICID, sector), 39M rows, Gaia columns repeated per row.
  `tars_table_4_unique_stars.parquet` next to it is the de-duplicated
  star list used for the bulk run (TICID, ra, dec, pm, Tmag,
  dr3_source_id only).
- `/ar0/local/TARS/zenodo_19917941/tars_table_4_stars.parquet`: one row
  per table 4 star (7,481,030; built 2026-11-06 by
  `drivers/make_tars4_catalog.py`, `sx_phot/tars_catalog.py`): the Gaia
  columns, `final_n_contams` (sector minimum of `num_contams`, which is
  table 2's definition), `num_contams_max`, `contamratio`,
  `n_secs_table4`, and the table 2 rotation columns (`adopted_period`,
  `median_amplitude`, flags, `n_secs`) with `in_table2`.  Readable by
  `sx_phot.specml.load_catalog` like table 2.

## Gaia DR3 slim copy

`/ar0/local/gaia_dr3_slim/` (fetch started 2026-11-10 09:00, ~14 h; check
`manifest.csv` and `fetch.log`): source_id, ra, dec, pmra, pmdec,
phot_g_mean_mag for all 1.81e9 Gaia DR3 sources as parquet parts
sorted by source_id (HEALPix order), built by
`drivers/fetch_gaia_slim.py` from the Heidelberg ARI TAP mirror.
Cone search: `sx_phot.gaia_slim.GaiaSlim(dir).cone(ra, dec, r_arcsec)`.
Complete when the last manifest row has fewer than 10,000,000 rows.

## Viewer

`apps/sx-viz` (port 8051) serves the table 2 spectra, SED fits and
star pages from `/ar0/SPHEREX/bulkphot/tars_v0/viz`; see
`apps/sx-viz/README.md`.

## ML products (empirical sequence, template fits, UMAP)

`/ar0/SPHEREX/bulkphot/tars_v0/ml/` (built 2026-11-04 from the viz
build of the same date; rebuild after every viz refresh):

| File | Content | Maker |
|---|---|---|
| `sequence.npz`, `sequence_nodes.csv` | empirical one-parameter template library (192 nodes, 76 bins): normalized log templates, population scatter, absolute offsets at 10 pc, residual eigenvectors, node summaries (TARS Teff, BP-RP0, M_G0, ...) | `drivers/build_spectral_sequence.py` |
| `sequence_train.parquet`, `sequence.json` | the ~146k clean training stars (row, t, chi2, inlier) and the selection / iteration history | same |
| `empfit.parquet` | per-star fits of the library (single, main-sequence-at-Gaia-distance, two-star, disk; chi2, BIC differences, band excesses, overluminosity, percentile ranks at fixed t), aligned with `viz/binned_ticid.npy` | `drivers/fit_empirical.py` |
| `umap_shape.npy`, `umap_resid.npy`, `pca_*.npz`, `umap.json` | 2-D UMAP of the normalized spectral shapes and of the template residuals (NaN for stars left out) | `drivers/run_umap.py` |
| `qa/seq_*.png` | library QA figures | `drivers/plot_sequence_qa.py` |

Python: `sx_phot.specml.SpectralSequence.load(<ml>/sequence)` gives
`template(t)`, `abs_template(t)`, `teff(t)`, `t_from_teff(...)`;
`sx_phot.empfit.EmpiricalModels` + `fit_chunk` fit new spectra.
`apps/sx-ml` (port 8052) serves the UMAP explorer and the sequence
slider; see `apps/sx-ml/README.md`.  Analysis figures and candidate
lists: `results/sx_ml_analysis/` (`scripts/analyze_empfit.py`).

`/ar0/SPHEREX/bulkphot/tars4_v0/{viz,ml}` (started 2026-11-06): the same
viz build for all 7.48M table 4 stars (`drivers/build_viz_products.py`,
48 workers) and the same fits and embeddings (`scripts/run_tars4_ml.sh`)
using the table 2 library copied into `tars4_v0/ml`
(`sequence_provenance.txt`), so `t_single`, `teff_emp`, `dmag_ms` are on
one scale in both runs.  sx-ml serves both runs (`apps/sx-ml/datasets.json`;
`?ds=tars2|tars4`), loading tars4 automatically once its products are
complete.  No SED fits (sx-viz) for tars4 yet.

## Caveats worth knowing

- `masked` rows (any flagged aperture pixel) are unreliable; bright stars
  saturate below ~4 um.  Single unflagged outliers remain; the spline
  supplement clips them.
- qr2 and qr3 use different calibrations; never mix them in one time
  series (the `qr` column separates them).
- qr2 files on the mirror through 2026W06 (VERSION 6.4-6.5.5 without
  `+psffix1`) have the IRSA PSF zone-header error (XCTR/YCTR swapped);
  the data arrays are identical to IRSA's fixed copies.  `psf_phot`
  corrects it on read; anything else reading the PSF extension must
  too, until the headers are patched in place (TODO.txt).
- Rows at x or y near 0 or 2039 are gross outliers (detector edge).
- The old per-star pipeline (`sx_phot/circphot.py`, IRSA based) is
  1.4-5 stars/h; use it only for stars outside every bulk run, or
  for cutout animations.
- Old caches in `drivers/test_results` predate the band-edge background
  fix (2026-11-02) and are wrong at 1.0-1.1 um.
