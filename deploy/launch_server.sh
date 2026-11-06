#!/bin/bash
# Launch the UMI real-robot inference server.
#
# Usage:
#   ./deploy/launch_server.sh CHECKPOINT_DIR
#
#   CHECKPOINT_DIR   A saved checkpoint, e.g. checkpoints/<run>/checkpoint-10000
#                    (contains config.json, the weights and experiment_cfg/).
#
# Optional env overrides:
#   PORT             WebSocket port (default 18000)
#   HOST             Bind host (default 0.0.0.0)
#   TOKENIZER_PATH   Tokenizer override (default "google/umt5-xxl"; pass a local dir to stay offline)
#   IMAGE_HEIGHT / IMAGE_WIDTH   Frame size override (default: read from checkpoint)
#   WARMUP_ITERS     Dummy inference rounds before serving (default 12; 0 to skip)
#   SINGLE_FRAME     1 = condition every chunk on the latest frame only
#   PYTHON           Python interpreter (default: `python` on PATH)
#
# Example:
#   PORT=9000 ./deploy/launch_server.sh checkpoints/umi/checkpoint-10000

set -e
export HYDRA_FULL_ERROR=1

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 CHECKPOINT_DIR"
    exit 1
fi
MODEL_PATH="$(cd "$1" && pwd)"
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: $MODEL_PATH is not a saved checkpoint (missing config.json)."
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PORT=${PORT:-18000}
HOST=${HOST:-0.0.0.0}
TOKENIZER_PATH=${TOKENIZER_PATH:-google/umt5-xxl}
PY=${PYTHON:-python}

ARGS=(--model_path "$MODEL_PATH" --tokenizer_path "$TOKENIZER_PATH" --port "$PORT" --host "$HOST")
if [ -n "${IMAGE_HEIGHT:-}" ] && [ -n "${IMAGE_WIDTH:-}" ]; then
    ARGS+=(--image_height "$IMAGE_HEIGHT" --image_width "$IMAGE_WIDTH")
fi
if [ -n "${WARMUP_ITERS:-}" ]; then
    ARGS+=(--warmup_iters "$WARMUP_ITERS")
fi
if [ "${SINGLE_FRAME:-0}" = "1" ]; then
    ARGS+=(--single-frame)
fi

echo "Checkpoint: $MODEL_PATH"
echo "Serving on: $HOST:$PORT  (single_frame=${SINGLE_FRAME:-0})"
echo "Python:     $PY"

cd "$REPO_ROOT"
exec "$PY" deploy/server.py "${ARGS[@]}"
