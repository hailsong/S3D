
CUDA_VISIBLE_DEVICES='7' python evaluation.py \
                            --real_images_folder ../data/celebamask/test/image \
                            --gen_images_folder ../inference_results/model_output/pix2pix3d_edge2face/gtview \
                            --sg_images_folder ../inference_results/model_output/pix2pix3d_edge2face/sgdv \
                            --fvv_images_folder ../inference_results/model_output/pix2pix3d_edge2face/fvv \
                            --device cuda \
                            --output_dir ../inference_results/model_output