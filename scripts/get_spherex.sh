#!/usr/bin/env bash
#
# Build a manifest of S3 objects (with sizes) and run s5cmd to download
# only missing / size-mismatched files into /ar0/SPHEREX/spherex_l2.
# Safe for million-file scale: one LIST pass, then local comparisons.
set -euo pipefail

# === CONFIG ===
BUCKET="nasa-irsa-spherex"
PREFIX="qr/level2/"                   # trailing slash
DEST="/ar0/SPHEREX/spherex_l2"
NUMWORKERS=96
AWS_CLI="${AWS_CLI:-aws}"             # override via env if desired
S5CMD="${S5CMD:-s5cmd}"               # override via env if desired
AWS_REGION="us-east-1"                # only needed for aws cli in some envs
# Logs directory in current working directory
LOGDIR="./LOGS"
# Date stamp for log/artifact filenames
DATESTAMP="$(date +%Y%m%d)"
# ==============

if ! command -v "$S5CMD" >/dev/null 2>&1; then
  echo "error: s5cmd not found in PATH. Install it and retry." >&2
  exit 1
fi

# Ensure logs directory exists
mkdir -p "$LOGDIR"

# Log/artifact files (non-temporary, date-stamped)
LIST_F="$LOGDIR/s3_list_${DATESTAMP}.txt"
SIZES_KEYS_F="$LOGDIR/sizes_keys_${DATESTAMP}.txt"
MANIFEST_F="$LOGDIR/manifest_${DATESTAMP}.txt"

mkdir -p "$DEST"

if [ -s "$LIST_F" ]; then
  echo "1) Found existing s3 list: $LIST_F (skipping)"
else
  echo "1) Listing S3 objects (single pass) into $LIST_F ..."
  # Use aws CLI recursive listing (includes size + key). Works without creds.
  # If you prefer s5cmd ls, swap the command below.
  "$AWS_CLI" s3 ls "s3://$BUCKET/$PREFIX" --recursive \
    --no-sign-request --region "$AWS_REGION" > "$LIST_F"
fi

if [ -s "$SIZES_KEYS_F" ]; then
  echo "2) Found existing sizes+keys: $SIZES_KEYS_F (skipping)"
else
  echo "2) Parsing list -> size + key -> $SIZES_KEYS_F"
  # awk: size is $3, key is fields 4..NF (handles keys with spaces)
  awk '{
    size=$3;
    key=$4;
    for(i=5;i<=NF;i++){ key = key " " $i }
    printf("%s\t%s\n", size, key)
  }' "$LIST_F" > "$SIZES_KEYS_F"
fi

# determine stat command (linux vs mac fallback)
if stat -c%s /dev/null >/dev/null 2>&1; then
  STAT_CMD='stat -c%s'
else
  STAT_CMD='stat -f%z'
fi

if [ -s "$MANIFEST_F" ]; then
  echo "3) Found existing manifest: $MANIFEST_F (skipping)"
else
  : > "$MANIFEST_F"
  echo "3) Comparing to local files and building manifest at $MANIFEST_F ..."
  count=0
  while IFS=$'\t' read -r size key; do
    # compute relative path after PREFIX
    rel="${key#${PREFIX}}"
    localpath="$DEST/$rel"
    localdir=$(dirname "$localpath")
    mkdir -p "$localdir"

    local_size=0
    if [ -f "$localpath" ]; then
      # guard against stat failures
      if local_size=$($STAT_CMD "$localpath" 2>/dev/null); then
        :
      else
        local_size=0
      fi
    fi

    if [ "${local_size:-0}" -ne "$size" ]; then
      # produce a cp command that preserves the path
      # (we don't add -s/-n because manifest already filtered by size)
      printf 'cp s3://%s/%s %s\n' "$BUCKET" "$key" "$localpath" \
        >> "$MANIFEST_F"
    fi
    # progress log every 1000 lines
    count=$((count + 1))
    if (( count % 1000 == 0 )); then
      ts=$(date '+%Y-%m-%d %H:%M:%S')
      echo "[$ts] ... compared $count lines"
    fi
  done < "$SIZES_KEYS_F"
fi

nlines=$(wc -l < "$MANIFEST_F" || echo 0)
echo "Manifest created with $nlines entries."

if [ "$nlines" -eq 0 ]; then
  echo "Nothing to download; exiting."
  exit 0
fi

echo "4) Running s5cmd manifest with $NUMWORKERS workers ..."
# run manifest; s5cmd will execute the cp commands in parallel
"$S5CMD" --no-sign-request --numworkers "$NUMWORKERS" run "$MANIFEST_F"

echo "Done."
