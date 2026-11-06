#!/bin/bash
# Force-compute meta/relative_stats.json for a UMI LeRobot dataset.
#
# Instantiates the training dataset (data=umi, the config the UMI training script
# uses) with relative actions on; the dataset computes the SE(3)-relative action
# stats and writes the cached relative_stats.json next to meta/stats.json.
# Safe to ctrl-c at any time: the file is only written at the very end.
#
# Usage:
#   UMI_DATA_ROOT=/path/to/umi_lerobot bash scripts/data/compute_umi_relative_stats.sh
#
# Env overrides:
#   REPO_ROOT       repo root            (default: the checkout containing this script)
#   PY              python interpreter   (default: python)
#   TOKENIZER_DIR   umt5-xxl tokenizer   (default: $REPO_ROOT/checkpoints/umt5-xxl if present,
#                                         else google/umt5-xxl)

set -e

: "${UMI_DATA_ROOT:?Set UMI_DATA_ROOT to the LeRobot dataset path}"
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PY:-python}
if [ -z "${TOKENIZER_DIR:-}" ]; then
    TOKENIZER_DIR=$REPO_ROOT/checkpoints/umt5-xxl
    [ -d "$TOKENIZER_DIR" ] || TOKENIZER_DIR=google/umt5-xxl
fi

REL_STATS="$UMI_DATA_ROOT/meta/relative_stats.json"

echo "Dataset:        $UMI_DATA_ROOT"
echo "Output file:    $REL_STATS"
echo "REPO_ROOT:      $REPO_ROOT"
echo

cd "$REPO_ROOT"
# Move any existing file aside (the dataset only recomputes when it is absent) and put it back
# if this run does not produce a new one.
BACKUP=""
if [ -f "$REL_STATS" ]; then
    BACKUP="$REL_STATS.bak.$(date +%Y%m%d_%H%M%S)"
    mv "$REL_STATS" "$BACKUP"
    echo "Existing stats moved to $BACKUP"
fi
restore_backup() {
    if [ -n "$BACKUP" ] && [ ! -f "$REL_STATS" ] && [ -f "$BACKUP" ]; then
        mv "$BACKUP" "$REL_STATS"
        echo "Restored the previous $REL_STATS"
    fi
}
trap restore_backup EXIT

UMI_DATA_ROOT="$UMI_DATA_ROOT" TOKENIZER_DIR="$TOKENIZER_DIR" "$PY" - <<'PY'
import os

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

with initialize_config_dir(
    version_base=None,
    config_dir=os.path.abspath("spatial_umi_dream/configs"),
    job_name="compute_rel_stats",
):
    cfg = compose(config_name="train", overrides=[
        "data=umi",
        f"umi_data_roots=[\"{os.environ['UMI_DATA_ROOT']}\"]",
        f"tokenizer_path={os.environ['TOKENIZER_DIR']}",
    ])
instantiate(cfg.train_dataset)
print("DONE")
PY

echo
if [ -f "$REL_STATS" ]; then
    echo "SUCCESS: wrote $REL_STATS"
    ls -la "$REL_STATS"
else
    echo "FAILED: $REL_STATS not produced"
    exit 1
fi
