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

cd "$SCRIPT_DIR"
/bin/bash ./get_spherex.sh >> "$LOGDIR/cron_spherex_imagegetter_${ts}.log" 2>&1
