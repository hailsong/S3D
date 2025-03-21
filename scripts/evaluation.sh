#!/bin/bash

cuda_num=0
REAL_IMAGES="data_eval/test/image"
GEN_IMAGES="inference_results/celebamask/_GTViews/"
SG_IMAGES="dataset/generated_images_sg"  # 1000 subfolders × 6 images each
FVV_IMAGES="dataset/generated_images_fvv" # 1000 subfolders × 15 views each
OUTDIR="evaluation_results"
mkdir -p "$OUTDIR"

CUDA_VISIBLE_DEVICES="${cuda_num}" python sketch2mask/evaluation.py \
    --real_images_folder $REAL_IMAGES \
    --gen_images_folder $GEN_IMAGES \
    --sg_images_folder $SG_IMAGES \
    --fvv_images_folder $FVV_IMAGES \
    --device cuda \
    --output_dir $OUTDIR
