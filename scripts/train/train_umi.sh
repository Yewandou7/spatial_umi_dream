#!/bin/bash
# UMI training (action + video), from Wan2.2 or from a checkpoint.
#
# Input: four synchronized streams per sample [robot0 RGB, robot1 RGB, robot0 depth, robot1 depth]
# assembled into a 320x320 canvas.
#
#   # from Wan2.2, single node, all visible GPUs
#   UMI_DATA_ROOT=/data/umi_lerobot bash scripts/train/train_umi.sh
#   # finetune a checkpoint with state registers
#   CKPT_DIR=checkpoints/umi/checkpoint-10000 USE_STATE=true RELATIVE_STATE=true \
#       UMI_DATA_ROOT=/data/umi_lerobot bash scripts/train/train_umi.sh
#   # multi-dataset mixture (weights optional; default proportional to valid steps)
#   UMI_DATA_ROOTS=/data/a,/data/b UMI_DATA_WEIGHTS=1,2 bash scripts/train/train_umi.sh
#   # multi-node: run on every node with WORLD_SIZE (nodes) RANK MASTER_ADDR MASTER_PORT set
#
# Knobs (defaults): PER_DEVICE_BS=8, GLOBAL_BATCH_SIZE=nodes*nproc*PER_DEVICE_BS, MAX_STEPS=10000,
# SAVE_STEPS=2000, SAVE_TOTAL_LIMIT=15, DATASET_SHARD_SAMPLING_RATE=1.0 with CKPT_DIR else 0.1, USE_STATE=false,
# RELATIVE_STATE=false, DATALOADER_NUM_WORKERS=2, WANDB_DATALOADER_LOG_EVERY_STEPS=1000,
# RUN_TAG, OUTPUT_DIR. See common.sh for the rest (MAX_CHUNK_SIZE, LEARNING_RATE, MAX_TRAJECTORIES, ...).
# Extra arguments are passed to hydra as overrides.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

RUN_TAG=${RUN_TAG:-umi_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-"$REPO_ROOT/checkpoints/${RUN_TAG}"}
PER_DEVICE_BS=${PER_DEVICE_BS:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((TOTAL_PROCESSES * PER_DEVICE_BS))}
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_STEPS=${SAVE_STEPS:-2000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-15}
# Finetuning a checkpoint on a small dataset reads every shard in full; large from-Wan2.2 runs
# sample 10% of each shard before moving on (the rates the released checkpoints were trained with).
if [ -n "$CKPT_DIR" ]; then
    DATASET_SHARD_SAMPLING_RATE=${DATASET_SHARD_SAMPLING_RATE:-1.0}
else
    DATASET_SHARD_SAMPLING_RATE=${DATASET_SHARD_SAMPLING_RATE:-0.1}
fi
USE_STATE=${USE_STATE:-false}
RELATIVE_STATE=${RELATIVE_STATE:-false}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
DATALOADER_PIN_MEMORY=false
WANDB_DATALOADER_LOG_EVERY_STEPS=${WANDB_DATALOADER_LOG_EVERY_STEPS:-1000}

# ---------------------------------------------------------------------------
# Dataset roots: UMI_DATA_ROOT (single) or UMI_DATA_ROOTS=a,b[,c] (+ UMI_DATA_WEIGHTS=1,2[,0.5]).
check_umi_root() {
    local root=$1
    if [ ! -d "$root" ]; then
        echo "ERROR: UMI dataset not found at $root" >&2
        exit 1
    fi
    # relative_stats.json is computed and cached by the dataset when missing. With more
    # than one process every rank would compute and write it concurrently, so precompute it.
    local required=(meta/info.json meta/modality.json meta/stats.json meta/depth_encoding.json)
    if [ "$TOTAL_PROCESSES" -gt 1 ] && [ ! -f "$root/meta/relative_stats.json" ]; then
        echo "ERROR: $root/meta/relative_stats.json is missing; compute it once first:" >&2
        echo "       UMI_DATA_ROOT=$root bash scripts/data/compute_umi_relative_stats.sh" >&2
        exit 1
    fi
    for f in "${required[@]}"; do
        if [ ! -f "$root/$f" ]; then
            echo "ERROR: UMI dataset root is missing $f: $root" >&2
            exit 1
        fi
    done
}

DATA_OVERRIDES=()
if [ -n "${UMI_DATA_ROOTS:-}" ]; then
    IFS=',' read -r -a ROOTS <<< "$UMI_DATA_ROOTS"
    if [ "${#ROOTS[@]}" -lt 2 ]; then
        echo "ERROR: UMI_DATA_ROOTS must contain at least two comma-separated dataset roots." >&2
        exit 1
    fi
    WEIGHTS=()
    if [ -n "${UMI_DATA_WEIGHTS:-}" ]; then
        IFS=',' read -r -a WEIGHTS <<< "$UMI_DATA_WEIGHTS"
        if [ "${#WEIGHTS[@]}" -ne "${#ROOTS[@]}" ]; then
            echo "ERROR: UMI_DATA_WEIGHTS must contain one comma-separated weight per UMI_DATA_ROOTS entry." >&2
            exit 1
        fi
    fi
    MIXTURE_SPEC=""
    for i in "${!ROOTS[@]}"; do
        root="${ROOTS[$i]}"; root="${root#"${root%%[![:space:]]*}"}"; root="${root%"${root##*[![:space:]]}"}"
        ROOTS[$i]="$root"
        check_umi_root "$root"
        if [ "${#WEIGHTS[@]}" -gt 0 ]; then
            w="${WEIGHTS[$i]}"; w="${w#"${w%%[![:space:]]*}"}"; w="${w%"${w##*[![:space:]]}"}"
            if ! awk -v weight="$w" 'BEGIN { exit !(weight > 0) }'; then
                echo "ERROR: UMI_DATA_WEIGHTS entries must be positive numbers: $w" >&2
                exit 1
            fi
            WEIGHTS[$i]="$w"
            MIXTURE_SPEC+="${MIXTURE_SPEC:+,}{dataset_path:{umi:[\"${root}\"]},dataset_weight:${w},distribute_weights:false}"
        fi
    done
    DATA_OVERRIDES+=("umi_data_roots=$(quote_list "${ROOTS[@]}")")
    if [ "${#WEIGHTS[@]}" -gt 0 ]; then
        # One mixture entry per root with an explicit sampling weight (replaces the default
        # single entry over umi_data_roots, whose weights are distributed by valid step count).
        DATA_OVERRIDES+=("train_dataset.mixture_spec=[$MIXTURE_SPEC]")
    fi
else
    if [ -n "${UMI_DATA_WEIGHTS:-}" ]; then
        echo "ERROR: UMI_DATA_WEIGHTS requires UMI_DATA_ROOTS." >&2
        exit 1
    fi
    UMI_DATA_ROOT=${UMI_DATA_ROOT:-"$REPO_ROOT/data/umi_lerobot"}
    check_umi_root "$UMI_DATA_ROOT"
    ROOTS=("$UMI_DATA_ROOT"); WEIGHTS=()
    DATA_OVERRIDES+=("umi_data_roots=$(quote_list "$UMI_DATA_ROOT")")
fi

echo "UMI: shard_sampling_rate=$DATASET_SHARD_SAMPLING_RATE use_state=$USE_STATE relative_state=$RELATIVE_STATE"
echo "UMI datasets (${#ROOTS[@]}; ${WEIGHTS:+explicit weights}${WEIGHTS:-weights proportional to valid steps}):"
for i in "${!ROOTS[@]}"; do echo "  ${WEIGHTS[$i]:-}  ${ROOTS[$i]}"; done

common_overrides
run_training \
    "${COMMON_OVERRIDES[@]}" \
    data=umi \
    "dataset_shard_sampling_rate=$DATASET_SHARD_SAMPLING_RATE" \
    "${DATA_OVERRIDES[@]}" \
    "$@"
