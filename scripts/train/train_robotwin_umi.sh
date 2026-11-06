#!/bin/bash
# RoboTwin-UMI training, from Wan2.2 or from a checkpoint.
#
# Input: four streams per sample [robot0 RGB, robot1 RGB, robot0 depth, robot1 depth] assembled
# into a 320x320 canvas, 2x2 panels [RGB | depth] per wrist.
#
#   # from Wan2.2 on one task
#   ROBOTWIN_UMI_DATA_ROOT=/data/robotwin_umi/shake_bottle bash scripts/train/train_robotwin_umi.sh
#   # finetune a checkpoint on a multitask mixture
#   CKPT_DIR=checkpoints/robotwin_umi/checkpoint-14000 \
#       ROBOTWIN_UMI_DATA_ROOTS=/data/robotwin_umi/a,/data/robotwin_umi/b bash scripts/train/train_robotwin_umi.sh
#   # multi-node: run on every node with WORLD_SIZE (nodes) RANK MASTER_ADDR MASTER_PORT set
#
# ROBOTWIN_UMI_DATA_ROOT trains a single dataset; ROBOTWIN_UMI_DATA_ROOTS=a,b (globs allowed) a multitask
# mixture weighted by episode count.
#
# Knobs (defaults; the first value is from-Wan2.2, the second with CKPT_DIR set):
# PER_DEVICE_BS=16|8, GRADIENT_ACCUMULATION_STEPS=2|4, MAX_STEPS=14000|3000, SAVE_STEPS=2000|500,
# GLOBAL_BATCH_SIZE=nodes*nproc*PER_DEVICE_BS*GRADIENT_ACCUMULATION_STEPS, SAVE_TOTAL_LIMIT=20,
# USE_STATE=true, RELATIVE_STATE=true, MAX_STATE_DIM=32, STATE_HORIZON=1, DATALOADER_NUM_WORKERS=4,
# SAVE_ONLY_MODEL=true, WANDB_DATALOADER_LOG_EVERY_STEPS=0, RUN_TAG, OUTPUT_DIR. See common.sh for
# the rest (MAX_CHUNK_SIZE, LEARNING_RATE, MAX_TRAJECTORIES, ...). Extra arguments are passed to
# hydra as overrides.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [ -n "$CKPT_DIR" ]; then
    MODE=finetune
    PER_DEVICE_BS=${PER_DEVICE_BS:-8}
    GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
    MAX_STEPS=${MAX_STEPS:-3000}
    SAVE_STEPS=${SAVE_STEPS:-500}
else
    MODE=scratch
    PER_DEVICE_BS=${PER_DEVICE_BS:-16}
    GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
    MAX_STEPS=${MAX_STEPS:-14000}
    SAVE_STEPS=${SAVE_STEPS:-2000}
fi
RUN_TAG=${RUN_TAG:-robotwin_umi_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/checkpoints/${MODE}_${RUN_TAG}"}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((TOTAL_PROCESSES * PER_DEVICE_BS * GRADIENT_ACCUMULATION_STEPS))}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-20}
USE_STATE=${USE_STATE:-true}
RELATIVE_STATE=${RELATIVE_STATE:-true}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
DATALOADER_PIN_MEMORY=true
WANDB_DATALOADER_LOG_EVERY_STEPS=${WANDB_DATALOADER_LOG_EVERY_STEPS:-0}

# ---------------------------------------------------------------------------
# Data: dataset roots.
if [ -n "${ROBOTWIN_UMI_DATA_ROOTS:-}" ]; then
    IFS=',' read -r -a ROOTS <<< "$ROBOTWIN_UMI_DATA_ROOTS"
    DATA_DESCRIPTION="multitask roots: $ROBOTWIN_UMI_DATA_ROOTS"
else
    ROBOTWIN_UMI_DATA_ROOT=${ROBOTWIN_UMI_DATA_ROOT:-"$REPO_ROOT/data/robotwin_umi"}
    if [ ! -f "$ROBOTWIN_UMI_DATA_ROOT/meta/modality.json" ]; then
        echo "ERROR: RoboTwin-UMI LeRobot dataset not found under: $ROBOTWIN_UMI_DATA_ROOT" >&2
        exit 1
    fi
    ROOTS=("$ROBOTWIN_UMI_DATA_ROOT")
    DATA_DESCRIPTION="root: $ROBOTWIN_UMI_DATA_ROOT"
fi
echo "RoboTwin-UMI $MODE: $DATA_DESCRIPTION"
echo "State: use_state=$USE_STATE relative_state=$RELATIVE_STATE max_state_dim=${MAX_STATE_DIM:-32} state_horizon=${STATE_HORIZON:-1}"

common_overrides
run_training \
    "${COMMON_OVERRIDES[@]}" \
    data=robotwin_umi \
    "robotwin_data_roots=$(quote_list "${ROOTS[@]}")" \
    "training_args.save_only_model=${SAVE_ONLY_MODEL:-true}" \
    ${MAX_STATE_DIM:+"max_state_dim=$MAX_STATE_DIM"} \
    ${STATE_HORIZON:+"state_horizon=$STATE_HORIZON"} \
    "$@"
