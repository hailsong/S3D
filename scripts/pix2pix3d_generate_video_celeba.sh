cuda_num=7

cd pix2pix3D

model_name="sketch2mask_face_distill"
CUDA_VISIBLE_DEVICES="${cuda_num}" python applications/generate_multiple_video.py \
                            --network checkpoints/pix2pix3d_seg2face.pkl \
                            --outdir ../sketch2mask/results/${model_name}/inference/pix2pix3d/ \
                            --input_dir ../sketch2mask/results/${model_name}/inference/pred_mask/ \
                            --cfg seg2face \
