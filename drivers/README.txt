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
  bucket=<N>/ (1024 stars per bucket).
materialize_bulk_star.py: Write the legacy sxphot_cache_*_annulus.csv and
  _splsupp.csv for a star from a gathered run, for the existing plot
  drivers.
