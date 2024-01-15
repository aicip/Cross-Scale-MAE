#!/bin/bash
#* "To linprobe from scratch, simply omit the --finetune argument.
#*  To resume a linprobe job, you can replace --finetune path/to/pretrained_checkpoint.pth
#*  with --resume path/to/linprobe_checkpoint.pth in the command above."

EPOCHS=50
BATCH_SIZE=1024
INPUT_SIZE=128
PATCH_SIZE=16

MODEL=$1 # mae_vit_base
LOSS=$2  # mse

function usage {
    echo "Usage: $0 <model> <loss> [additional flags]"
    echo "Example: $0 mae_vit_base classification_cross --wandb_project satmae --device cuda:0"
    echo "See additional flags in main_linprobe.py"
    exit 1
}

# if any of the above are empty, then usage
if [ -z "$MODEL" ] || [ -z "$LOSS" ]; then
    usage
fi

shift 2

#! Change this
# fmow rgb
IN_PATH_BASE="../fmow-rgb-preproc"
IN_TRAIN_PATH="$IN_PATH_BASE/train_${INPUT_SIZE}.csv"
IN_VAL_PATH="$IN_PATH_BASE/val_${INPUT_SIZE}.csv"
# spacenet
# IN_PATH_BASE="../spacenetv1"
# IN_TRAIN_PATH="$IN_PATH_BASE/data_${INPUT_SIZE}/input/train"
# IN_VAL_PATH="$IN_PATH_BASE/data_${INPUT_SIZE}/input/val"

#! Change this
OUT_DIR_BASE="weights/linprobe_fmowrgb"

MODEL_TYPE=""

#! Change this
DATASET_TYPE="rgb"
# DATASET_TYPE="spacenetv1"
# DATASET_TYPE="smart"

#! Change this
# -- BASE MODELS
# PRE_CHECKPOINT_DIR="weights/cross_scale_mae_base_pretrain.pth"
# -- LARGE MODELS
PRE_CHECKPOINT_DIR="weights/cross_scale_mae_large_pretrain.pth"

set -x
python3 main_linprobe.py \
    --train_path "${IN_TRAIN_PATH}" \
    --test_path "${IN_VAL_PATH}" \
    --output_dir_base "${OUT_DIR_BASE}" \
    --model "${MODEL}" \
    --loss "${LOSS}" \
    --model_type "${MODEL_TYPE}" \
    --input_size "${INPUT_SIZE}" \
    --patch_size "${PATCH_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --finetune "${PRE_CHECKPOINT_DIR}" \
    --dataset_type "${DATASET_TYPE}" \
    --wandb_project satmae_linprobe $@
