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
