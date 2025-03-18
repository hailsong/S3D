CUDA_VISIBLE_DEVICES='7' python inference_single.py \
                                --input_image ../data/celebamask/scribble/face_scribble_4_resize.jpeg \
                                --output_dir inference_results/face_scribble/1742152629/ \
                                --model_path results/1742152629/best_unet_model.pth \
                                --data_type celeba \
