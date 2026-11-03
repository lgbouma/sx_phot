------------------------------------------
Shell script to download all SPHEREx images

get_spherex.sh:  Use s5cmd to pull all LVF images from AWS.  These are the
  higher quality data product vs cutouts.  Mission started data release 2025W17
  (2025/04/20).  Produces ~1.6Tb/wk in images, or ~100Tb/yr +change.  Scale by
  your time of download.

  Release prefixes (updated 2026-11-01): IRSA ships each quick release under its
  own S3 prefix (qr2/level2/ ended at 2026W29; qr3/level2/ began at 2026W30 on
  2026-09-16 with new calibrations).  The script discovers every qr*/ prefix
  in the bucket and mirrors all of them into the same DEST tree, so a future
  qr4/ is picked up automatically.  Config knobs (env or edit the script):
    - SKIP_TAGS="qr2"      frozen releases: checked for new week dirs (cheap)
                           but not re-listed / re-compared / re-downloaded.
    - DOWNLOAD_BACKFILL=0  weeks that reappear in a later prefix (i.e. a
                           reprocessing of weeks already on disk, ~100 TB for
                           the full mission) are reported but NOT downloaded
                           unless this is 1.  Note QR2 and QR3 calibrations
                           differ; do not mix them in one time series.
    - DRY_RUN=1            everything except the s5cmd download.
  Email alerts go to ALERT_TO (via sendmail, like ~/bin/check_ar0_disk.sh) when
  a new qr*/ prefix appears, when backfilled weeks are detected (once per week
  dir), or when a frozen prefix unexpectedly gains weeks.
  Persistent state lives in LOGS/state/ (known_tags.txt, weeks_<tag>.txt,
  backfill_alerted_<tag>.txt).  Artifacts below are now per tag:
  s3_list_<tag>_YYYYMMDD.txt etc.

  - Artifacts: The script writes three date-stamped files into `./LOGS` (created if missing):
    - `s3_list_YYYYMMDD.txt`: Raw listing from S3
    - `sizes_keys_YYYYMMDD.txt`: Parsed `size<TAB>key` lines
    - `manifest_YYYYMMDD.txt`: `s5cmd run`-compatible copy commands
  - Idempotent: If any artifact already exists and is non-empty for the same date, the step is skipped to save time.
  - Daily wrapper: `run_get_spherex_daily.sh` runs `get_spherex.sh` and appends output to `LOGS/cron_spherex_imagegetter_YYYYMMDD.log`.

  Files
  - `get_spherex.sh`: Main pipeline (list → parse → compare → manifest → `s5cmd run`).
  - `run_get_spherex_daily.sh`: Cron-friendly wrapper that sets `PATH`, ensures `LOGS/`, and invokes the main script.
  - `LOGS/`: Persistent logs and artifacts, date-stamped by `YYYYMMDD`.

  Requirements
  - `aws` CLI available in `PATH` (used with `--no-sign-request`).
  - `s5cmd` available in `PATH`.
  - Destination directory configured inside `get_spherex.sh` (`DEST=`) must be writable.

  Manual Run
  - From this directory: `./get_spherex.sh`
  - Or via wrapper (recommended to mirror cron PATH and logging): `./run_get_spherex_daily.sh`

  Setup Cron (run once per day)
  1) Make sure the wrapper is executable:
     `chmod +x /PATH/TO/sx_phot/scripts/run_get_spherex_daily.sh`
  2) Add a crontab entry (examples run at 03:00 local time):
     `crontab -e`
     Then add:
     `0 3 * * * /bin/bash /PATH/TO/sx_phot/scripts/run_get_spherex_daily.sh`

  Notes
  - PATH: The wrapper exports a PATH that includes common locations for `aws` and `s5cmd` so cron can find them.
  - Logs: Daily output is appended to `LOGS/cron_spherex_imagegetter_YYYYMMDD.log`.
  - Reuse: On the same day, the script reuses existing artifacts and proceeds directly to later steps.
  - Locking: the wrapper takes `flock` on `LOGS/.get_spherex.lock`; an overlapping cron invocation logs one line and exits.

------------------------------------------
Photometry scripts to get a SPHEREx spectrum

(based on the work by Kishalay and Zafar)

  simplephot_slowest.py:
    (Also, the simplest) Given RA/DEC, query IRSA to get all available images,
    then do simple aperture photometry and make a plot.

  simplephot_slow.py:
    Given RA/DEC, query a mix of IRSA and AWS to get all available images, and do
    simple aperture photometry.  Makes a plot and a CSV table.  Better cacheing
    than simplephot_slowest.py (e.g. including timestamps).

  circphot_slow.py:
    Given RA/DEC, query a mix of IRSA and AWS to get all available images, and do
    simple aperture photometry with either zodi or annulus subtraction.  Makes a
    plot and a CSV table.  Good cacheing both of fits images and of table (e.g.
    including timestamps).  Annulus seems preferable.

  circphot_zoom_simple.py:
    Like circphot_slow.py, but way faster because it i) downloads IRSA cutouts by
    default rather than full images, and ii) multithreads over some other
    network-dependent steps.

------------------------------------------
Bulk photometry wrapper

run_bulk_phot_tars.sh: index -> run_bulk_phot.py -> gather for the TARS
  catalog.  Env knobs: CATALOG, RUN_DIR (default
  /ar0/SPHEREX/bulkphot/tars_v0), NPROC (64), OBSCORE_CSV (reuse a
  snapshot instead of fetching), EXTRA (extra run_bulk_phot.py args).
  Takes a flock on $RUN_DIR/.lock; safe to re-run daily after the image
  download: only new files are photometered and the buckets are rebuilt.
  Logs: $RUN_DIR/pipeline_<stamp>.log and $RUN_DIR/run_bulk_phot.log.
