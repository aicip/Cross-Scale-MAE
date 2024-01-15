#!/bin/bash
#* "To finetune from scratch, simply omit the --finetune argument.
#*  To resume a finetuning job, you can replace --finetune path/to/pretrained_checkpoint.pth
#*  with --resume path/to/finetune_checkpoint.pth in the command above."

BATCH_SIZE=512
INPUT_SIZE=64
PATCH_SIZE=8

MODEL=$1

function usage {
    echo "Usage: $0 <model> [additional flags]"
    echo "Example: $0 mae_vit_base --wandb_project satmae --device cuda:0"
    echo "See additional flags in main_finetune.py"
    exit 1
}

# if any of the above are empty, then usage
if [ -z "$MODEL" ]; then
    usage
fi

shift 1

#! Change this
# fmow rgb
IN_PATH_BASE="../fmow-rgb-preproc"
IN_TRAIN_PATH="$IN_PATH_BASE/train_${INPUT_SIZE}.csv"
IN_VAL_PATH="$IN_PATH_BASE/val_${INPUT_SIZE}.csv"
# resisc45
# IN_PATH_BASE="../NWPU_RESISC45_processed"
# IN_TRAIN_PATH="$IN_PATH_BASE/train_${INPUT_SIZE}.csv"
# IN_VAL_PATH="$IN_PATH_BASE/val_${INPUT_SIZE}.csv"
# spacenet
# IN_PATH_BASE="../spacenetv1"
# IN_TRAIN_PATH="$IN_PATH_BASE/data_${INPUT_SIZE}/input/train"
# IN_VAL_PATH="$IN_PATH_BASE/data_${INPUT_SIZE}/input/val"

#! Change this
OUT_DIR_BASE="weights/finetune_fmowrgb"

MODEL_TYPE=""

#! Change this
DATASET_TYPE="rgb"
# DATASET_TYPE="resisc45"
# DATASET_TYPE="spacenetv1"

#! Change this
# -- BASE MODELS
# PRE_CHECKPOINT_DIR="weights/cross_scale_mae_base_pretrain.pth"
# -- LARGE MODELS
PRE_CHECKPOINT_DIR="weights/cross_scale_mae_large_pretrain.pth"

set -x
python3 main_finetune.py \
    --train_path "$IN_TRAIN_PATH" \
    --test_path "$IN_VAL_PATH" \
    --output_dir_base "$OUT_DIR_BASE" \
    --model "$MODEL" \
    --model_type "$MODEL_TYPE" \
    --input_size "$INPUT_SIZE" \
    --patch_size "$PATCH_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --finetune "$PRE_CHECKPOINT_DIR" \
    --dataset_type "$DATASET_TYPE" \
    --wandb_project satmae_finetune $@
