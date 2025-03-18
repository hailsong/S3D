CUDA_VISIBLE_DEVICES='7' python inference_single.py \
                                --input_image ../data/celebamask/test/sketch/02743.png \
                                --output_dir inference_results/celebamask/02743/ \
                                --model_path results/sketch2mask_face_aug/best_unet_model.pth \
                                --data_type celeba \
