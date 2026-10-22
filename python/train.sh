#!/usr/bin/env bash
# Train the affinity network on the 0.5 mm cache, on a machine with a GPU.
#
#   ./train.sh sanity   /path/to/pseg3d/data/casx/cache   # a few minutes: 300 steps of the affinity stage, one validation
#   ./train.sh affinity /path/to/pseg3d/data/casx/cache   # stage 1: the edge affinities, 30 000 steps on 96^3 patches; resumable
#   ./train.sh walker   /path/to/pseg3d/data/casx/cache   # stage 2: + the walker's loss, 20 000 steps on 160^3 patches, from stage 1's best; resumable
#   ./train.sh eval     /path/to/pseg3d/data/casx/cache   # the simulated user on 20 whole test scans, Learned against Basic
#
# Runs in pieces when the machine is shared: PIECE_MINUTES=120 ./train.sh affinity ... ends after two
# hours, and the same command carries on from where it stopped (--resume) until it prints "done".
# Short of memory in the walker stage: EXTRA="--patch 128" or "--accumulate 2".
#
# Defaults are for a 24-32 GB GPU (bf16; batch 4 of 96^3 in stage 1, batch 1 of 160^3 in stage 2).
# The checkpoints and the logs go to runs/affinity and runs/walker (RUNS); point
# MRFSEG_LEARNED_WEIGHTS at runs/walker/best.pt for the editor (or copy it to ~/.cache/mrfseg/learned.pt).

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:?sanity | affinity | walker | eval}"
CACHE="${2:?absolute path of the cache (pseg3d/data/casx/cache)}"
RUNS="${RUNS:-runs}"
PYTHON="${PYTHON:-.venv/bin/python}"
EXTRA="${EXTRA:-}"

piece() {  # --resume when there is something to resume, --time-limit when asked for pieces
  local out="$1"
  if [ -f "$out/last.pt" ]; then echo --resume; fi
  if [ -n "${PIECE_MINUTES:-}" ]; then echo --time-limit "$PIECE_MINUTES"; fi
}

case "$MODE" in
  sanity)
    # The real schedule (30 000 steps, a short warm-up), stopped after 300: the learning rate is
    # then the full run's, so what the losses do is what the full run would do in its first minutes.
    rm -rf "$RUNS/sanity"
    "$PYTHON" -m mrfseg.learned.train --cache "$CACHE" --out "$RUNS/sanity" --steps 30000 --stop-after 300 --warmup 100 \
      --log-every 25 --val-every 300 --val-cases 2 --val-clicks 3 --workers 8 $EXTRA
    ;;
  affinity)
    "$PYTHON" -m mrfseg.learned.train --cache "$CACHE" --out "$RUNS/affinity" --steps 30000 --batch 4 --patch 96 --workers 8 \
      --val-every 2000 $(piece "$RUNS/affinity") $EXTRA
    ;;
  walker)
    INIT=()
    if [ ! -f "$RUNS/walker/last.pt" ]; then INIT=(--init "$RUNS/affinity/best.pt"); fi
    "$PYTHON" -m mrfseg.learned.train --cache "$CACHE" --out "$RUNS/walker" --steps 20000 --batch 1 --patch 160 --workers 8 \
      --walker-weight 1 --lr 1e-4 --warmup 500 --val-every 1000 "${INIT[@]}" $(piece "$RUNS/walker") $EXTRA
    ;;
  eval)
    "$PYTHON" -m mrfseg.learned.evaluate --cache "$CACHE" --model "$RUNS/walker/best.pt" --basic --cases 20 --out "$RUNS/eval"
    ;;
  *)
    echo "unknown mode $MODE: sanity | affinity | walker | eval" >&2
    exit 2
    ;;
esac
