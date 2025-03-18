CUDA_VISIBLE_DEVICES='4' python ./sketch2mask/inference_single.py \
                                --input_image ../data/celebamask/scribble/face_scribble_2_resize.jpeg \
                                --output_dir inference_results/face_scribble/ \
                                --model_path ./sketch2mask/results/1742112314/best_unet_model.pth \
                                --data_type celeba \

cd pix2pix3D
CUDA_VISIBLE_DEVICES='4' python applications/generate_video.py \
                                --network checkpoints/pix2pix3d_seg2face.pkl \
                                --outdir ../inference_results/face_scribble/ \
                                --random_seed 1 \
                                --cfg seg2face \
                                --input ../sketch2mask/inference_results/face_scribble/face_scribble_2_resize_mask_gray.png
