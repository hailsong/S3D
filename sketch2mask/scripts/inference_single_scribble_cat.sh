CUDA_VISIBLE_DEVICES='7' python inference_single.py \
                                --input_image ../data/cat/scribble/cat_scribble_resize.jpeg \
                                --output_dir inference_results/cat_scribble/sketch2mask_cat_aug/ \
                                --model_path results/sketch2mask_cat_aug/best_unet_model.pth \
                                --data_type cat \
