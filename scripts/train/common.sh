#!/bin/bash
# Shared pieces of the training launchers (sourced by train_umi.sh / train_robotwin_umi.sh):
# repo root, Wan2.2 weights (auto-downloaded), optional warm-start checkpoint, python,
# torchrun single-node / multi-node launch and the hydra overrides common to both recipes.
# Recipe constants (320x320 canvas, 2 latent frames + 24 actions per block, bf16/tf32, ZeRO-2,
# AdamW 1e-5 with 5% warmup, seed 42, ...) live in spatial_umi_dream/configs/; the launchers
# pass only the knobs below.
#
# Environment knobs read here:
#   REPO_ROOT           repo root (default: the checkout containing this script)
#   CKPT_DIR            checkpoint to warm-start from (unset: train from Wan2.2)
#   WAN22_CKPT_DIR, TOKENIZER_DIR, IMAGE_ENCODER_DIR
#                       pretrained weights; default $REPO_ROOT/checkpoints/<name>,
#                       downloaded from HuggingFace when missing
#   PYTHON              interpreter with torch (default: first of python / python3 that has it)
#   NPROC_PER_NODE      GPU processes per node (default: visible GPU count, else 8)
#   GPU_IDS             comma list -> CUDA_VISIBLE_DEVICES and NPROC_PER_NODE
#   WORLD_SIZE RANK MASTER_ADDR MASTER_PORT
#                       set all four for multi-node (WORLD_SIZE = number of nodes,
#                       RANK = this node's rank); unset -> torchrun --standalone
#   MAX_CHUNK_SIZE      latent chunks per sample (default 4); the loader emits 8*chunks+1 frames
#   LEARNING_RATE       peak learning rate (default 1e-5)
#   MAX_TRAJECTORIES    keep only the first N episodes of each dataset (default: all)
#   WANDB_PROJECT       W&B project (default spatial-umi-dream)
#   WANDB_DATALOADER_VIDEO_FPS  fps of the logged dataloader video (default 10)
#   DRY_RUN=1           print the launch command instead of running it

set -euo pipefail
export HYDRA_FULL_ERROR=1

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-"$(cd "$_COMMON_DIR/../.." && pwd)"}
if [ ! -d "$REPO_ROOT/spatial_umi_dream" ]; then
    echo "ERROR: no spatial_umi_dream/ package under REPO_ROOT=$REPO_ROOT" >&2
    exit 1
fi
EXPERIMENT_PY="$REPO_ROOT/spatial_umi_dream/train/experiment.py"

MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE:-4}

# ---------------------------------------------------------------------------
# Pretrained weights.
WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"$REPO_ROOT/checkpoints/Wan2.2-TI2V-5B"}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"$REPO_ROOT/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"$REPO_ROOT/checkpoints/umt5-xxl"}
CLIP_FILE=models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth

download_weights() {
    if [ ! -d "$WAN22_CKPT_DIR" ] || [ -z "$(ls -A "$WAN22_CKPT_DIR" 2>/dev/null)" ]; then
        echo "Wan2.2-TI2V-5B not found at $WAN22_CKPT_DIR. Downloading from HuggingFace..."
        huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN22_CKPT_DIR"
    fi
    if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
        echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
        huggingface-cli download google/umt5-xxl --include "*.json" "spiece.model" --local-dir "$TOKENIZER_DIR"
    fi
    if [ ! -f "$IMAGE_ENCODER_DIR/$CLIP_FILE" ]; then
        echo "Image encoder not found. Downloading Wan2.1-I2V-14B-480P (for CLIP only)..."
        huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$IMAGE_ENCODER_DIR"
    fi
}

# Optional checkpoint to warm-start from (cfg.pretrained_model_path, strict=False).
# Unset: the DiT is initialized from the Wan2.2 weights above.
CKPT_DIR=${CKPT_DIR:-}
PRETRAINED_OVERRIDES=()
if [ -n "$CKPT_DIR" ]; then
    if [ ! -f "$CKPT_DIR/model.safetensors" ] && [ ! -f "$CKPT_DIR/model.safetensors.index.json" ]; then
        echo "ERROR: no weights under CKPT_DIR=$CKPT_DIR" >&2
        echo "       (expected model.safetensors or model.safetensors.index.json)" >&2
        exit 1
    fi
    # The checkpoint overrides every DiT weight, so skip loading the Wan2.2 DiT release first.
    PRETRAINED_OVERRIDES+=("pretrained_model_path=$CKPT_DIR" model.config.load_wan_dit=false)
fi

# ---------------------------------------------------------------------------
# Python with torch.
if [ -n "${PYTHON:-}" ] && [ -x "$PYTHON" ]; then
    PY="$PYTHON"
elif command -v python >/dev/null 2>&1 && python -c "import torch" >/dev/null 2>&1; then
    PY="$(command -v python)"
elif command -v python3 >/dev/null 2>&1 && python3 -c "import torch" >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    echo "ERROR: No Python with torch installed. Activate the training environment or set PYTHON=<path>." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Distributed layout.
if [ -n "${GPU_IDS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    IFS=',' read -r -a _GPUS <<< "$GPU_IDS"
    NPROC_PER_NODE=${NPROC_PER_NODE:-${#_GPUS[@]}}
fi
if [ -z "${NPROC_PER_NODE:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
    NPROC_PER_NODE=${NPROC_PER_NODE:-0}
    [ "$NPROC_PER_NODE" -gt 0 ] || NPROC_PER_NODE=8
fi
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
if [ "$NNODES" -gt 1 ]; then
    for v in RANK MASTER_ADDR MASTER_PORT; do
        if [ -z "${!v:-}" ]; then
            echo "ERROR: WORLD_SIZE=$NNODES nodes requested; set RANK (node rank), MASTER_ADDR and MASTER_PORT." >&2
            exit 1
        fi
    done
    TORCHRUN_ARGS=(--nnodes "$NNODES" --node_rank "$NODE_RANK"
                   --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")
else
    TORCHRUN_ARGS=(--standalone)
fi
TOTAL_PROCESSES=$((NNODES * NPROC_PER_NODE))

# ---------------------------------------------------------------------------
# quote_list a b c -> ["a","b","c"] (a hydra list override of paths / globs).
quote_list() {
    local out="" item
    for item in "$@"; do out+="${out:+,}\"${item}\""; done
    printf '[%s]' "$out"
}

# Hydra overrides shared by the UMI and RoboTwin recipes. Recipe-specific values are appended
# by the caller; everything not listed here keeps its spatial_umi_dream/configs default.
common_overrides() {
    COMMON_OVERRIDES=(
        "training_args.output_dir=$OUTPUT_DIR"
        "${PRETRAINED_OVERRIDES[@]}"
        # Wan2.2 weights
        "wan_dir=$WAN22_CKPT_DIR"
        "text_encoder_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
        "image_encoder_path=$IMAGE_ENCODER_DIR/$CLIP_FILE"
        "vae_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth"
        "tokenizer_path=$TOKENIZER_DIR"
        # Shape
        "max_chunk_size=$MAX_CHUNK_SIZE"
        "use_state=$USE_STATE"
        "relative_state=$RELATIVE_STATE"
        # Optimization / checkpointing / data loading
        "training_args.per_device_train_batch_size=$PER_DEVICE_BS"
        "global_batch_size=$GLOBAL_BATCH_SIZE"
        "training_args.max_steps=$MAX_STEPS"
        "training_args.save_steps=$SAVE_STEPS"
        "training_args.save_total_limit=$SAVE_TOTAL_LIMIT"
        "training_args.dataloader_num_workers=$DATALOADER_NUM_WORKERS"
        "training_args.dataloader_pin_memory=$DATALOADER_PIN_MEMORY"
        "trainer.wandb_dataloader_log_every_steps=$WANDB_DATALOADER_LOG_EVERY_STEPS"
        # Optional knobs: unset keeps the config default.
        ${LEARNING_RATE:+"training_args.learning_rate=$LEARNING_RATE"}
        ${MAX_TRAJECTORIES:+"max_trajectories=$MAX_TRAJECTORIES"}
        ${WANDB_PROJECT:+"wandb_project=$WANDB_PROJECT"}
        ${WANDB_DATALOADER_VIDEO_FPS:+"trainer.wandb_dataloader_video_fps=$WANDB_DATALOADER_VIDEO_FPS"}
    )
}

# run_training <hydra overrides...>
run_training() {
    local log_dir="$OUTPUT_DIR/logs/node${NODE_RANK}"
    local cmd=(
        "$PY" -m torch.distributed.run
        --nproc_per_node "$NPROC_PER_NODE" "${TORCHRUN_ARGS[@]}"
        --redirects 3 --tee 1 --log_dir "$log_dir"
        "$EXPERIMENT_PY" "$@"
    )
    echo "Python:       $PY"
    echo "Launch:       nodes=$NNODES node_rank=$NODE_RANK nproc_per_node=$NPROC_PER_NODE total_processes=$TOTAL_PROCESSES"
    echo "Init:         ${CKPT_DIR:-Wan2.2 weights ($WAN22_CKPT_DIR)}"
    echo "Batch:        per_device=$PER_DEVICE_BS global=$GLOBAL_BATCH_SIZE max_steps=$MAX_STEPS"
    echo "Chunking:     max_chunk_size=$MAX_CHUNK_SIZE"
    echo "Output:       $OUTPUT_DIR"
    printf '%q ' "${cmd[@]}"; printf '\n'
    if [ -n "${DRY_RUN:-}" ]; then
        return 0
    fi
    download_weights
    mkdir -p "$OUTPUT_DIR" "$log_dir"
    cd "$REPO_ROOT"
    exec "${cmd[@]}"
}
