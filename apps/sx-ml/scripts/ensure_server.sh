#!/usr/bin/env bash
# Start the sx-ml server if nothing is listening on its port.
#
# Idempotent; meant to be called from cron, e.g.
#   @reboot      /ar0/PROJ/luke/proj/sx_phot/apps/sx-ml/scripts/ensure_server.sh
#   */5 * * * *  /ar0/PROJ/luke/proj/sx_phot/apps/sx-ml/scripts/ensure_server.sh
#
# Environment overrides: SX_PYTHON, SX_ML_HOST, SX_ML_PORT, SX_ML_LOG,
# SX_RUN_DIR, SX_VIZ_URL.

set -euo pipefail

app_dir="$(cd "$(dirname "$0")/.." && pwd)"
sx_python="${SX_PYTHON:-$HOME/miniconda3/envs/py311_sx/bin/python}"
host="${SX_ML_HOST:-0.0.0.0}"
port="${SX_ML_PORT:-8052}"
log="${SX_ML_LOG:-$HOME/local/sx-ml/server.log}"
run_dir="${SX_RUN_DIR:-/ar0/SPHEREX/bulkphot/tars_v0}"
mkdir -p "$(dirname "$log")"

if ss -ltn "sport = :$port" | grep -q LISTEN; then
  exit 0
fi

echo "$(date -Is) ensure_server: starting sx-ml on ${host}:${port}" >> "$log"
cd "$app_dir"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
nohup nice -n 5 "$sx_python" -u server.py --host "$host" --port "$port" \
  --run-dir "$run_dir" >> "$log" 2>&1 < /dev/null &
