run_circphot.py:  Given RA/DEC, calls sx_phot.circphot to get the SPHEREx
  spectrum.  This is like scripts/circphot_zoom_simple below but enables wrapping.
get_tic_spherex_coords.py: Given a TIC ID, fetches Gaia DR2 proper motion and
  propagates RA/DEC to SPHEREx observation times, caching results under results/.
animate_tic_cutouts.py: Given a TIC ID, builds a wavelength-ordered cutout MP4
  with the aperture centered on the target coordinates.

Bulk (image-centric) photometry, no IRSA queries per star:
build_image_index.py: Build /ar0/SPHEREX/index/image_index.parquet, one row
  per (obs_id, detector): local path, release, footprint polygon, times.
  Primary source is one bulk ObsCore TAP dump (~3.5 min); local files
  missing from ObsCore are indexed from their headers.
run_bulk_phot.py: Given a star catalog (feather/parquet/CSV with id, ra,
  dec, pmra, pmdec at a given epoch), open every local L2 image once and
  measure all catalog stars on it (sx_phot.bulk_phot).  Output rows go to
  <run-dir>/phot/<shard>/pass_<stamp>.parquet; re-running processes only
  files not seen before.  Row columns: star_idx, ticid, image_key, x, y,
  flux_jy (annulus-subtracted), flux_err_jy, flux_raw_jy, bkg_jy, zodi_jy,
  flag_or, n_bad_pix, n_ap_pix, n_ann_pix, bkg_mode, wavelength_um,
  bandwidth_um, mjd_avg.  bkg_mode: 0 = sigma-clipped median of the
  star-masked annulus; 1 = fewer than --min-ann-pix (20) unmasked pixels
  were left, so the unmasked annulus was used instead (happens at the
  bright LVF band edges, e.g. 1.0-1.1 um, where the global star mask
  flags the whole region); 2 = no finite annulus pixels, background 0.
  --psf adds Gaia-forced PSF photometry (sx_phot.psf_phot) with the
  neighbour table given by --neighbours: every Gaia DR3 source whose
  PSF reaches an 11 x 11 px box is fitted at its propagated position
  with a free flux, plus a plane background.  Extra columns: psf_status
  (0 ok, 1 edge, 2 masked core, 3 no Gaia match, 4 fit error, 5 fewer
  than 20 degrees of freedom left, e.g. >100 sources toward the bulge),
  psf_flux_jy, psf_flux_err_jy, psf_chi2nu, psf_n_comp, psf_n_drop
  (neighbours with no light on usable pixels), psf_nb_flux_jy, psf_bkg_mjysr,
  psf_blend_frac_g (G-band flux of sources merged with the star),
  psf_min_corr (most negative target-neighbour flux correlation),
  psf_apsub_jy / psf_apsub_err_jy (2 px aperture after subtracting the
  fitted neighbours and plane), psf_ap_ee (model fraction of the star's
  light in that aperture), psf_zone, psf_n_bad_c3.  Old qr2 files
  (pipeline 6.4-6.5.5 without '+psffix1') have the IRSA PSF zone header
  error; the zone centres are swapped back automatically.
query_gaia_neighbours.py: Gaia DR3 sources within 80" of every catalog
  star (TAP upload join, chunked and resumable; chunks that hit the
  archive's 50,000-row limit are split) -> the --neighbours parquet for
  run_bulk_phot.py --psf.  Use the same catalog options as the run.
fetch_gaia_slim.py: Build the local slim Gaia DR3 copy
  (/ar0/local/gaia_dr3_slim: source_id, ra, dec, pmra, pmdec, G for all
  1.81e9 sources) from the Heidelberg ARI TAP mirror, 10M rows per job
  in source_id order, resumable from manifest.csv (sx_phot/gaia_slim.py).
  Parts are HEALPix-ordered through source_id, so
  sx_phot.gaia_slim.GaiaSlim(dir).cone(ra, dec, r_arcsec) reads only
  the row groups that can hold the cone.  Run under the systemd-run cap.
fix_psf_headers.py: Repair the IRSA PSF zone-header error (pipeline
  6.4-6.5.5, VERSION without '+psffix1') in the L2 mirror in place:
  XCTR_i/YCTR_i and XWID_i/YWID_i exchanged, VERSION suffixed, nothing
  else touched (sx_phot/psf_header_fix.py; byte-identical to IRSA's
  fixed S3 copy).  --dry-run and --limit first; idempotent; do not run
  while photometry jobs read the mirror.
gather_bulk_phot.py: Regroup the rows by star into <run-dir>/bystar/
  bucket=<N>/ (1024 stars per bucket; rows sorted by star_idx).  Two
  stages so memory stays bounded for any bucket count: --nproc workers
  (4) split one pass file each into bystar.groups/group=G/
  (--buckets-per-group buckets per group, default 64), then read one
  group each, sort it and write its bucket files.  Peak memory is about
  10 GB per worker for the 7.5M-star run; run it under
  `systemd-run --user -p MemoryMax=64G -p MemorySwapMax=0` on wh3.
materialize_bulk_star.py: Write the legacy sxphot_cache_*_annulus.csv and
  _splsupp.csv for a star from a gathered run, for the existing plot
  drivers.
materialize_bulk_run.py: Same for every star of a gathered run in one
  pass (buckets read once, spline fits in parallel); --id-prefix sets the
  filename label (TIC, GaiaDR3, ...).  Writes summary.csv (one row per
  catalog star; n == 0 means no coverage).
plot_bulk_qa.py: QA plots from a materialized run: <star>_spline.png per
  star (or _spectrum.png without a spline product), each with the
  circphot-style 64 px cutout inset from the image nearest
  --cutout-wavelength (0.93 um; 0 disables) with the aperture drawn, to
  show crowding; overview_grid.png (top-S/N quarter plus a random
  sample) and qa_summary.csv.

Viewer products (apps/sx-viz; see apps/sx-viz/README.md):
build_viz_products.py: From a gathered run, write <run-dir>/viz/: per-star
  spectrum tables (spectra/bucket=<N>.parquet, one row group per star,
  legacy columns + spline model + fit mask), per-star summaries
  (summary.parquet) and R=40 binned spectra (binned_*.npy).  Skips
  buckets already built unless --overwrite.  ~35 s per bucket per worker.
fit_viz_seds.py: Fit every binned spectrum with the SPHINX (<= 4000 K) and
  Castelli-Kurucz (>= 4250 K) grids (sx_phot.sedfit): full-range single
  fit, blue-anchored fit with per-band red excess, two-component fit;
  radii from Gaia parallax; Teff-relative excess columns.  Writes
  <run-dir>/viz/sed.parquet (--postprocess-only recomputes only the
  relative columns).  ~4 ms per star per worker.
precompute_viz_cache.py: Warm the server's on-demand caches (report PNG,
  SPHEREx cutouts, DSS2 cutouts) for the stars selected by a table query.
