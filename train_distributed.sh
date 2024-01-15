#!/bin/bash

NUM_GPUS=8

EPOCHS=400
BATCH_SIZE=512

INPUT_SIZE=128
PATCH_SIZE=16

MODEL=$1
ATTENTION=$2
LOSS=$3

function usage {
    echo "Usage: $0 <model> <attention> <loss> [additional flags]"
    echo "Example: $0 mae_vit_base scaled_dot_product mse --use_xformers --wandb_project satmae --device cuda:0"
    echo "See additional flags in main_pretrain.py"
    exit 1
}

# if any of the above are empty, then usage
if [ -z "$MODEL" ] || [ -z "$ATTENTION" ] || [ -z "$LOSS" ]; then
    usage
fi

shift 3

IN_PATH_BASE="../fmow-rgb-preproc"
IN_PATH="$IN_PATH_BASE/train_${INPUT_SIZE}.csv"

OUT_DIR_BASE="weights"

torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port=1234 main_pretrain.py \
    --train_path "$IN_PATH" \
    --output_dir_base "$OUT_DIR_BASE" \
    --model "$MODEL" \
    --loss "$LOSS" \
    --attn_name "$ATTENTION" \
    --input_size "$INPUT_SIZE" \
    --patch_size "$PATCH_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --use_xformers $@
