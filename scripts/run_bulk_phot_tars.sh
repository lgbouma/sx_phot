#!/usr/bin/env bash
# Full TARS bulk photometry: index -> photometry -> gather.
# Safe to re-run: the index is refreshed, only new files are photometered,
# and the per-star buckets are rebuilt.
set -euo pipefail

PY=/home/luke/miniconda3/envs/py311_sx/bin/python
REPO=/ar0/PROJ/luke/proj/sx_phot
CATALOG=${CATALOG:-/ar0/local/TARS/tars_table_2.feather}
RUN_DIR=${RUN_DIR:-/ar0/SPHEREX/bulkphot/tars_v0}
NPROC=${NPROC:-64}
OBSCORE_CSV=${OBSCORE_CSV:-}   # set to reuse a snapshot instead of fetching
EXTRA=${EXTRA:-}               # extra args for run_bulk_phot.py

mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec 9>"$RUN_DIR/.lock"
if ! flock -n 9; then
  echo "another run holds $RUN_DIR/.lock; exiting" | tee -a "$LOG"
  exit 0
fi
{
  echo "[$(date '+%F %T')] 1) image index"
  if [ -n "$OBSCORE_CSV" ]; then
    "$PY" "$REPO/drivers/build_image_index.py" --obscore-csv "$OBSCORE_CSV"
  else
    "$PY" "$REPO/drivers/build_image_index.py"
  fi
  echo "[$(date '+%F %T')] 2) photometry"
  # shellcheck disable=SC2086
  "$PY" "$REPO/drivers/run_bulk_phot.py" --catalog "$CATALOG" \
    --run-dir "$RUN_DIR" --nproc "$NPROC" $EXTRA
  echo "[$(date '+%F %T')] 3) gather by star"
  "$PY" "$REPO/drivers/gather_bulk_phot.py" --run-dir "$RUN_DIR"
  echo "[$(date '+%F %T')] done"
} >> "$LOG" 2>&1
