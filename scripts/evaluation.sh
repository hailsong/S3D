#!/bin/bash

cuda_num=7
model_name="sketch2mask_face"

REAL_IMAGES="./data/celebamask/test/image/"
GEN_IMAGES="./sketch2mask/inference_results/model_output/${model_name}/gtview"
SG_IMAGES="./sketch2mask/inference_results/model_output/${model_name}/sgdv"  # 1000 subfolders × 6 images each
FVV_IMAGES="./sketch2mask/inference_results/model_output/${model_name}/fvv" # 1000 subfolders × 15 views each
OUTDIR="evaluation_results"

mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES="${cuda_num}" python sketch2mask/evaluation.py \
    --real_images_folder $REAL_IMAGES \
    --gen_images_folder $GEN_IMAGES \
    --sg_images_folder $SG_IMAGES \
    --fvv_images_folder $FVV_IMAGES \
    --device cuda \
    --output_dir $OUTDIR
