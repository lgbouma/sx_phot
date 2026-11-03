#!/usr/bin/env bash
set -euo pipefail

# Resolve to this script's directory (works in cron)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure LOGS directory exists before redirecting logs
LOGDIR="$SCRIPT_DIR/LOGS"
mkdir -p "$LOGDIR"

# Ensure common PATH locations for aws/s5cmd in cron
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

# Date-stamped log (YYYYMMDD)
ts="$(date +%Y%m%d)"
LOG="$LOGDIR/cron_spherex_imagegetter_${ts}.log"

# Never run two copies at once (a multi-TB backlog pull can outlast the
# 24 h cron interval).  Lock is released automatically when this shell exits.
exec 9>"$LOGDIR/.get_spherex.lock"
if ! flock -n 9; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] another get_spherex run holds the lock; skipping this invocation" >> "$LOG"
  exit 0
fi

cd "$SCRIPT_DIR"
/bin/bash ./get_spherex.sh >> "$LOG" 2>&1
