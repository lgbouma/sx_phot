#!/usr/bin/env bash
# ML products for the TARS table 4 run (all T<16, d<500 pc targets):
# empirical-sequence fits and UMAP embeddings, using the table 2 library
# (copied into <run>/ml, see sequence_provenance.txt there).
#
# Waits for the viz build (<run>/viz/viz.json from
# drivers/build_viz_products.py), then runs drivers/fit_empirical.py and
# drivers/run_umap.py (shape and resid in parallel).  Meant to run under a
# memory cap, e.g.
#   systemd-run --user -p MemoryMax=120G -p MemorySwapMax=0 \
#       scripts/run_tars4_ml.sh
# Environment overrides: SX_PYTHON, SX_RUN_DIR, SX_TARS, SX_WORKERS,
# SX_UMAP_THREADS.

set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
py="${SX_PYTHON:-$HOME/miniconda3/envs/py311_sx/bin/python}"
run="${SX_RUN_DIR:-/ar0/SPHEREX/bulkphot/tars4_v0}"
tars="${SX_TARS:-/ar0/local/TARS/zenodo_19917941/tars_table_4_stars.parquet}"
workers="${SX_WORKERS:-48}"
threads="${SX_UMAP_THREADS:-24}"
ml="$run/ml"
log="$ml/run_tars4_ml.log"
mkdir -p "$ml"

stamp() { date -Is; }
echo "$(stamp) waiting for $run/viz/viz.json" >> "$log"
until [ -f "$run/viz/viz.json" ]; do sleep 60; done
echo "$(stamp) viz build present: $(cat "$run/viz/viz.json" | tr -d '\n ')" >> "$log"

for f in sequence.npz sequence_nodes.csv; do
  [ -f "$ml/$f" ] || { echo "$(stamp) missing $ml/$f" >> "$log"; exit 1; }
done
cd "$repo"

if [ ! -f "$ml/empfit.parquet" ]; then
  echo "$(stamp) fit_empirical: $workers workers" >> "$log"
  nice -n 10 "$py" drivers/fit_empirical.py --viz-dir "$run/viz" \
    --tars "$tars" --workers "$workers" > "$ml/fit_empirical.log" 2>&1
  echo "$(stamp) fit_empirical done" >> "$log"
fi

echo "$(stamp) run_umap: shape and resid in parallel, $threads threads each" >> "$log"
pids=()
for which in shape resid; do
  if [ -f "$ml/umap_$which.npy" ]; then continue; fi
  NUMBA_NUM_THREADS="$threads" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    nice -n 10 "$py" drivers/run_umap.py --viz-dir "$run/viz" --tars "$tars" \
    --which "$which" > "$ml/umap_$which.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]:-}"; do
  [ -n "$p" ] && { wait "$p" || status=$?; }
done
echo "$(stamp) run_umap done (status $status); $(ls "$ml"/umap_*.npy 2>/dev/null | tr '\n' ' ')" >> "$log"
echo "$(stamp) restart the sx-ml server to serve the new dataset: apps/sx-ml/scripts/ensure_server.sh" >> "$log"
exit "$status"
