#!/usr/bin/env bash
# Rebuild the sx-viz products of a run end to end and restart the server:
#   1. drivers/build_viz_products.py  (per-star spectra, summaries, binned)
#   2. drivers/fit_viz_seds.py        (SED fits -> sed.parquet)
#   3. restart server.py via ensure_server.sh
#
# Usage: refresh_products.sh [--no-overwrite] [--no-restart]
# Environment: SX_PYTHON, SX_RUN_DIR, SX_VIZ_PORT, SX_BUILD_WORKERS (12),
# SX_FIT_WORKERS (10).  Logs go to <run>/viz/refresh.log.

set -euo pipefail

app_dir="$(cd "$(dirname "$0")/.." && pwd)"
repo_dir="$(cd "$app_dir/../.." && pwd)"
sx_python="${SX_PYTHON:-$HOME/miniconda3/envs/py311_sx/bin/python}"
run_dir="${SX_RUN_DIR:-/ar0/SPHEREX/bulkphot/tars_v0}"
port="${SX_VIZ_PORT:-8051}"
build_workers="${SX_BUILD_WORKERS:-12}"
fit_workers="${SX_FIT_WORKERS:-10}"
overwrite="--overwrite"
restart=1
for arg in "$@"; do
  case "$arg" in
    --no-overwrite) overwrite="" ;;
    --no-restart) restart=0 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

viz_dir="$run_dir/viz"
mkdir -p "$viz_dir"
log="$viz_dir/refresh.log"
exec >> "$log" 2>&1
echo "$(date -Is) refresh_products: start ($run_dir)"

cd "$repo_dir"
nice -n 19 "$sx_python" drivers/build_viz_products.py \
  --run-dir "$run_dir" --workers "$build_workers" $overwrite
nice -n 19 "$sx_python" drivers/fit_viz_seds.py \
  --viz-dir "$viz_dir" --workers "$fit_workers" --chunk 4000

if [ "$restart" = 1 ]; then
  pids="$(ss -ltnp "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)"
  for pid in $pids; do
    echo "$(date -Is) stopping server pid $pid"
    kill "$pid" || true
  done
  # Wait for the port to be released (ensure_server.sh is a no-op while
  # something still listens).
  for _ in $(seq 1 60); do
    if ! ss -ltn "sport = :$port" | grep -q LISTEN; then break; fi
    sleep 1
  done
  if ss -ltn "sport = :$port" | grep -q LISTEN; then
    echo "$(date -Is) port $port still busy; forcing"
    for pid in $pids; do kill -9 "$pid" || true; done
    sleep 2
  fi
  SX_RUN_DIR="$run_dir" SX_VIZ_PORT="$port" "$app_dir/scripts/ensure_server.sh"
fi
echo "$(date -Is) refresh_products: done"
