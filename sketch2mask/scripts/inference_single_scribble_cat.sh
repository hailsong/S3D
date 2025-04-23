gpu_idx=7
models=("sketch2mask_cat_distill" "sketch2mask_cat_distill_aug")

for model in "${models[@]}"
do
    for (( i=1; i<=4; i++ ))
    do
        CUDA_VISIBLE_DEVICES=${gpu_idx} python inference_single.py \
                                        --input_image ../data/cat/scribble/cat_scribble${i}_resize.png \
                                        --output_dir inference_results/cat_scribble/${model}/ \
                                        --model_path results/${model}/best_unet_model.pth \
                                        --data_type cat
    done
done