#!/usr/bin/env bash
# Start the sx-viz server if nothing is listening on its port.
#
# Idempotent; meant to be called from cron, e.g.
#   @reboot      /ar0/PROJ/luke/proj/sx_phot/apps/sx-viz/scripts/ensure_server.sh
#   */5 * * * *  /ar0/PROJ/luke/proj/sx_phot/apps/sx-viz/scripts/ensure_server.sh
#
# Environment overrides: SX_PYTHON, SX_VIZ_HOST, SX_VIZ_PORT, SX_VIZ_LOG,
# SX_RUN_DIR, SX_VIZ_DIR.

set -euo pipefail

app_dir="$(cd "$(dirname "$0")/.." && pwd)"
sx_python="${SX_PYTHON:-$HOME/miniconda3/envs/py311_sx/bin/python}"
host="${SX_VIZ_HOST:-0.0.0.0}"
port="${SX_VIZ_PORT:-8051}"
log="${SX_VIZ_LOG:-$HOME/local/sx-viz/server.log}"
run_dir="${SX_RUN_DIR:-/ar0/SPHEREX/bulkphot/tars_v0}"
viz_dir="${SX_VIZ_DIR:-$run_dir/viz}"
mkdir -p "$(dirname "$log")"

if ss -ltn "sport = :$port" | grep -q LISTEN; then
  exit 0
fi

echo "$(date -Is) ensure_server: starting sx-viz on ${host}:${port}" >> "$log"
cd "$app_dir"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
nohup nice -n 5 "$sx_python" -u server.py --host "$host" --port "$port" \
  --run-dir "$run_dir" --viz-dir "$viz_dir" >> "$log" 2>&1 < /dev/null &
