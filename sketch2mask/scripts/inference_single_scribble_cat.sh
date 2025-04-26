gpu_idx=6
models=("sketch2mask_cat_distill_aug")

# for model in "${models[@]}"
# do
#     for (( i=1; i<=6; i++ ))
#     do
#         CUDA_VISIBLE_DEVICES=${gpu_idx} python inference_single.py \
#                                         --input_image ../data/cat/scribble/scribble_0${i}_resize.png \
#                                         --output_dir inference_results/cat_scribble/${model}/ \
#                                         --model_path results/${model}/best_unet_model.pth \
#                                         --data_type cat
#     done
# done

for model in "${models[@]}"
do
    CUDA_VISIBLE_DEVICES=${gpu_idx} python inference_single.py \
                                    --input_image ../data/cat/scribble/cat_scribble_resize_refine.jpeg \
                                    --output_dir inference_results/cat_scribble/${model}/ \
                                    --model_path results/${model}/best_unet_model.pth \
                                    --data_type cat
done