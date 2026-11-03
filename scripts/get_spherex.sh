#!/usr/bin/env bash
#
# Mirror SPHEREx Level-2 spectral images from the IRSA S3 bucket into DEST.
#
# For every quick-release prefix in the bucket (qr2/, qr3/, qr4/, ...):
#   1. Cheap non-recursive listing of <tag>/level2/ -> set of week dirs
#      (e.g. 2026W30_1B), diffed against persistent LOGS/state/weeks_<tag>.txt.
#   2. BACKFILL detection: weeks in qrN that were already released under some
#      earlier qrM.  A full reprocessing of the mission is ~100 TB and mixes
#      calibrations with data already on disk, so backfilled weeks are
#      reported (log + email) but NOT downloaded unless DOWNLOAD_BACKFILL=1.
#   3. Unless the tag is in SKIP_TAGS (frozen releases): one recursive LIST
#      pass, local size comparison, and an s5cmd manifest download of only
#      the missing / size-mismatched files.
#
# Alerts (email via sendmail, same mechanism as ~/bin/check_ar0_disk.sh):
#   - a never-before-seen qr*/ prefix appears
#   - backfilled weeks appear in a prefix (once per week dir)
#   - a frozen (SKIP_TAGS) prefix unexpectedly gains new week dirs
#
# Safe for million-file scale: one LIST pass per prefix, then local comparisons.
set -euo pipefail

# === CONFIG ===
BUCKET="nasa-irsa-spherex"
DEST="/ar0/SPHEREX/spherex_l2"
NUMWORKERS=96
AWS_CLI="${AWS_CLI:-aws}"             # override via env if desired
S5CMD="${S5CMD:-s5cmd}"               # override via env if desired
AWS_REGION="us-east-1"
# Frozen releases: still checked for new week dirs (cheap, seconds), but not
# re-listed recursively, re-compared, or re-downloaded.  Space separated.
SKIP_TAGS="${SKIP_TAGS:-qr2}"
# Set DOWNLOAD_BACKFILL=1 to also download weeks that were already released
# under an earlier prefix (e.g. a qr3 reprocessing of the qr2 weeks).
DOWNLOAD_BACKFILL="${DOWNLOAD_BACKFILL:-0}"
# Set DRY_RUN=1 to do everything except the s5cmd download.
DRY_RUN="${DRY_RUN:-0}"
# Email alerts
ALERT_TO="${ALERT_TO:-bouma.luke@gmail.com}"
ALERT_FROM="$(hostname) spherex getter <luke@astro.caltech.edu>"
# Logs/artifacts directory in current working directory
LOGDIR="./LOGS"
STATEDIR="$LOGDIR/state"
# Date stamp for log/artifact filenames
DATESTAMP="$(date +%Y%m%d)"
# ==============

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

send_alert() {  # SUBJECT, body on stdin
  local subject="$1"
  if [ -x /usr/sbin/sendmail ]; then
    { printf 'From: %s\nTo: %s\nSubject: %s\n\n' "$ALERT_FROM" "$ALERT_TO" "$subject"; cat; } \
      | /usr/sbin/sendmail -t -oi || log "warning: sendmail failed for: $subject"
    log "   alert emailed to $ALERT_TO: $subject"
  else
    cat >/dev/null
    log "warning: sendmail not available; alert NOT emailed: $subject"
  fi
}

in_list() {  # needle, haystack (space separated)
  local needle="$1" x
  shift
  for x in $*; do [ "$x" = "$needle" ] && return 0; done
  return 1
}

if ! command -v "$S5CMD" >/dev/null 2>&1; then
  echo "error: s5cmd not found in PATH. Install it and retry." >&2
  exit 1
fi
if ! command -v "$AWS_CLI" >/dev/null 2>&1; then
  echo "error: aws CLI not found in PATH. Install it and retry." >&2
  exit 1
fi

mkdir -p "$LOGDIR" "$STATEDIR" "$DEST"

# determine stat command (linux vs mac fallback)
if stat -c%s /dev/null >/dev/null 2>&1; then
  STAT_CMD='stat -c%s'
else
  STAT_CMD='stat -f%z'
fi

# ---------------------------------------------------------------------------
# 0) Discover quick-release prefixes
# ---------------------------------------------------------------------------
log "0) Discovering qr*/ prefixes in s3://$BUCKET/ ..."
TAGS=$("$AWS_CLI" s3 ls "s3://$BUCKET/" --no-sign-request --region "$AWS_REGION" \
       | awk '$1=="PRE" && $2 ~ /^qr[0-9]+\/$/ {sub("/","",$2); print $2}' \
       | sort -V)
if [ -z "$TAGS" ]; then
  echo "error: no qr*/ prefixes found in s3://$BUCKET/" >&2
  exit 1
fi
log "   found: $(echo $TAGS)   (frozen: ${SKIP_TAGS:-none})"

KNOWN_TAGS_F="$STATEDIR/known_tags.txt"
touch "$KNOWN_TAGS_F"
first_run=0
[ -s "$KNOWN_TAGS_F" ] || first_run=1
for tag in $TAGS; do
  if ! grep -qx "$tag" "$KNOWN_TAGS_F"; then
    echo "$tag" >> "$KNOWN_TAGS_F"
    if [ "$first_run" -eq 0 ]; then
      log "   NEW PREFIX: $tag has appeared in the bucket"
      { echo "A new SPHEREx release prefix s3://$BUCKET/$tag/ appeared on $(date)."
        echo "get_spherex.sh will start mirroring $tag/level2/ automatically."
        echo "If it is a full reprocessing, expect BACKFILL alerts next; those"
        echo "weeks are not downloaded unless DOWNLOAD_BACKFILL=1."
      } | send_alert "[SPHEREx] new release prefix $tag in $BUCKET"
    fi
  fi
done

summary=""

# ---------------------------------------------------------------------------
# Per-prefix loop
# ---------------------------------------------------------------------------
for tag in $TAGS; do
  PREFIX="$tag/level2/"                 # trailing slash
  tagnum="${tag#qr}"
  WEEKS_F="$STATEDIR/weeks_${tag}.txt"  # persistent union of week dirs seen
  touch "$WEEKS_F"
  log "== $tag : s3://$BUCKET/$PREFIX =="

  # 1) cheap week-level listing, diffed against persistent state
  CUR_WEEKS_F="$LOGDIR/weeks_${tag}_${DATESTAMP}.txt"
  "$AWS_CLI" s3 ls "s3://$BUCKET/$PREFIX" --no-sign-request --region "$AWS_REGION" \
    | awk '$1=="PRE" {sub("/","",$2); print $2}' | sort > "$CUR_WEEKS_F"
  nweeks=$(wc -l < "$CUR_WEEKS_F")
  new_weeks=$(comm -23 "$CUR_WEEKS_F" <(sort "$WEEKS_F") || true)
  if [ -n "$new_weeks" ]; then
    log "   $nweeks week dirs; NEW since last run: $(echo $new_weeks)"
    echo "$new_weeks" >> "$WEEKS_F"
    sort -u -o "$WEEKS_F" "$WEEKS_F"
  else
    log "   $nweeks week dirs; no new weeks since last run"
  fi

  # 2) backfill detection: weeks in this tag already released by an earlier tag
  BACKFILL_F="$LOGDIR/backfill_${tag}_${DATESTAMP}.txt"
  : > "$BACKFILL_F"
  for prev in "$STATEDIR"/weeks_qr*.txt; do
    [ -e "$prev" ] || continue
    pt=$(basename "$prev" .txt); pt=${pt#weeks_}; pn=${pt#qr}
    [ "$pn" -lt "$tagnum" ] || continue
    comm -12 "$CUR_WEEKS_F" <(sort "$prev") >> "$BACKFILL_F" || true
  done
  sort -u -o "$BACKFILL_F" "$BACKFILL_F"
  nback=$(wc -l < "$BACKFILL_F")
  if [ "$nback" -gt 0 ]; then
    log "   BACKFILL DETECTED: $nback week dirs in $tag were already released under an earlier prefix:"
    log "   $(tr '\n' ' ' < "$BACKFILL_F")"
    ALERTED_F="$STATEDIR/backfill_alerted_${tag}.txt"
    touch "$ALERTED_F"
    unalerted=$(comm -23 "$BACKFILL_F" <(sort "$ALERTED_F") || true)
    if [ -n "$unalerted" ]; then
      { echo "s3://$BUCKET/$PREFIX now contains week dirs that were already"
        echo "released under an earlier prefix (reprocessing / backfill)."
        echo
        echo "Newly detected backfilled weeks ($(echo "$unalerted" | wc -l)):"
        echo "$unalerted"
        echo
        echo "Total backfilled weeks in $tag so far: $nback"
        if [ "$DOWNLOAD_BACKFILL" = "1" ]; then
          echo "DOWNLOAD_BACKFILL=1 is set: these ARE being downloaded."
        else
          echo "These weeks are NOT being downloaded (DOWNLOAD_BACKFILL=0)."
          echo "A full reprocessing of the mission is ~100 TB.  To pull it:"
          echo "  cd $(pwd) && DOWNLOAD_BACKFILL=1 ./get_spherex.sh"
          echo "or set DOWNLOAD_BACKFILL=1 in get_spherex.sh for the cron job."
        fi
        echo; df -h "$DEST"
      } | send_alert "[SPHEREx] BACKFILL detected in $tag ($nback weeks)"
      echo "$unalerted" >> "$ALERTED_F"
      sort -u -o "$ALERTED_F" "$ALERTED_F"
    fi
  fi

  # 3) frozen releases: nothing more to do (but alert if they changed)
  if in_list "$tag" "$SKIP_TAGS"; then
    if [ -n "$new_weeks" ] && [ "$first_run" -eq 0 ]; then
      log "   WARNING: frozen prefix $tag gained new week dirs; consider removing it from SKIP_TAGS"
      { echo "Prefix s3://$BUCKET/$PREFIX is in SKIP_TAGS (frozen) but gained new week dirs on $(date):"
        echo "$new_weeks"
        echo; echo "Remove $tag from SKIP_TAGS in get_spherex.sh if you want them."
      } | send_alert "[SPHEREx] frozen prefix $tag gained new weeks"
    fi
    log "   $tag is frozen (SKIP_TAGS); not re-listing or downloading."
    summary+="   $tag: $nweeks weeks, frozen (skipped)"$'\n'
    continue
  fi

  # 4) full recursive listing (one pass)
  LIST_F="$LOGDIR/s3_list_${tag}_${DATESTAMP}.txt"
  SIZES_KEYS_F="$LOGDIR/sizes_keys_${tag}_${DATESTAMP}.txt"
  MANIFEST_F="$LOGDIR/manifest_${tag}_${DATESTAMP}.txt"

  if [ -s "$LIST_F" ]; then
    log "   4) Found existing s3 list: $LIST_F (skipping)"
  else
    log "   4) Listing S3 objects (single pass) into $LIST_F ..."
    "$AWS_CLI" s3 ls "s3://$BUCKET/$PREFIX" --recursive \
      --no-sign-request --region "$AWS_REGION" > "$LIST_F"
  fi
  nobj=$(wc -l < "$LIST_F")
  log "      $nobj objects, $(awk '{s+=$3} END{printf "%.2f", s/1e12}' "$LIST_F") TB in $PREFIX"

  # 5) parse list -> size<TAB>key, optionally excluding backfilled weeks
  if [ -s "$SIZES_KEYS_F" ]; then
    log "   5) Found existing sizes+keys: $SIZES_KEYS_F (skipping)"
  else
    log "   5) Parsing list -> size + key -> $SIZES_KEYS_F"
    # awk: size is $3, key is fields 4..NF (handles keys with spaces)
    awk '{
      size=$3;
      key=$4;
      for(i=5;i<=NF;i++){ key = key " " $i }
      printf("%s\t%s\n", size, key)
    }' "$LIST_F" > "$SIZES_KEYS_F"
    if [ "$nback" -gt 0 ] && [ "$DOWNLOAD_BACKFILL" != "1" ]; then
      nbefore=$(wc -l < "$SIZES_KEYS_F")
      awk -F'\t' -v pfx="$PREFIX" '
        NR==FNR { skip[$1]=1; next }
        { rel=substr($2, length(pfx)+1); split(rel, a, "/");
          if (!(a[1] in skip)) print }
      ' "$BACKFILL_F" "$SIZES_KEYS_F" > "$SIZES_KEYS_F.tmp"
      mv "$SIZES_KEYS_F.tmp" "$SIZES_KEYS_F"
      log "      excluded $((nbefore - $(wc -l < "$SIZES_KEYS_F"))) backfilled objects (DOWNLOAD_BACKFILL=0)"
    fi
  fi

  # 6) compare to local files and build manifest
  if [ -s "$MANIFEST_F" ]; then
    log "   6) Found existing manifest: $MANIFEST_F (skipping)"
  else
    : > "$MANIFEST_F"
    log "   6) Comparing to local files and building manifest at $MANIFEST_F ..."
    count=0
    while IFS=$'\t' read -r size key; do
      rel="${key#${PREFIX}}"
      localpath="$DEST/$rel"
      localdir="${localpath%/*}"
      [ -d "$localdir" ] || mkdir -p "$localdir"

      local_size=0
      if [ -f "$localpath" ]; then
        if local_size=$($STAT_CMD "$localpath" 2>/dev/null); then
          :
        else
          local_size=0
        fi
      fi

      if [ "${local_size:-0}" -ne "$size" ]; then
        printf 'cp s3://%s/%s %s\n' "$BUCKET" "$key" "$localpath" >> "$MANIFEST_F"
      fi
      count=$((count + 1))
      if (( count % 10000 == 0 )); then
        log "      ... compared $count lines"
      fi
    done < "$SIZES_KEYS_F"
  fi

  nlines=$(wc -l < "$MANIFEST_F" || echo 0)
  log "   Manifest for $tag has $nlines entries."
  summary+="   $tag: $nweeks weeks, $nobj objects, $nback backfilled weeks, $nlines to download"$'\n'

  if [ "$nlines" -eq 0 ]; then
    log "   Nothing to download for $tag."
    continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    log "   DRY_RUN=1: skipping s5cmd download for $tag."
    continue
  fi

  # 7) download
  log "   7) Running s5cmd manifest for $tag with $NUMWORKERS workers ..."
  "$S5CMD" --no-sign-request --numworkers "$NUMWORKERS" run "$MANIFEST_F"
  log "   7) s5cmd finished for $tag."
done

log "Done.  Summary:"
printf '%s' "$summary"
