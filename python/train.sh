#!/usr/bin/env bash
# Train the potentials network on the 0.5 mm cache, on a machine with a GPU.
#
#   ./train.sh sanity /path/to/pseg3d/data/casx/cache     # ~10 minutes: 300 steps, one validation; checks memory and that the loss falls
#   ./train.sh full   /path/to/pseg3d/data/casx/cache     # the real run: 40 000 steps, validating every 2000; resumable
#   ./train.sh eval   /path/to/pseg3d/data/casx/cache     # the simulated user on 20 test scans, Learned against Basic
#
# Runs in pieces when the machine is shared: FULL_PIECE_MINUTES=120 ./train.sh full ... ends after two
# hours, and the same command carries on from where it stopped (--resume) until it prints "done".
# Short of memory: add --checkpoint-sweeps, or --batch 2 --accumulate 2, to EXTRA.
#
# Defaults are for a 24 GB GPU (bf16, batch 4, 96^3 patches, 20 sweeps). The checkpoints and the
# log go to runs/first (OUT); point MRFSEG_LEARNED_WEIGHTS at runs/first/best.pt for the editor.

set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:?sanity | full | eval}"
CACHE="${2:?absolute path of the cache (pseg3d/data/casx/cache)}"
OUT="${OUT:-runs/first}"
PYTHON="${PYTHON:-.venv/bin/python}"
EXTRA="${EXTRA:-}"

case "$MODE" in
  sanity)
    "$PYTHON" -m mrfseg.learned.train --cache "$CACHE" --out "${OUT}-sanity" --steps 300 --warmup 50 \
      --log-every 25 --val-every 300 --val-cases 4 --val-clicks 3 --workers 8 $EXTRA
    ;;
  full)
    ARGS=(--cache "$CACHE" --out "$OUT" --steps 40000 --batch 4 --workers 8 --val-every 2000)
    if [ -f "$OUT/last.pt" ]; then ARGS+=(--resume); fi
    if [ -n "${FULL_PIECE_MINUTES:-}" ]; then ARGS+=(--time-limit "$FULL_PIECE_MINUTES"); fi
    "$PYTHON" -m mrfseg.learned.train "${ARGS[@]}" $EXTRA
    ;;
  eval)
    "$PYTHON" -m mrfseg.learned.evaluate --cache "$CACHE" --model "$OUT/best.pt" --basic --cases 20 --out "$OUT/eval"
    ;;
  *)
    echo "unknown mode $MODE: sanity | full | eval" >&2
    exit 2
    ;;
esac
