cuda_num=7

cd pix2pix3D

models=("sketch2mask_face" "sketch2mask_face_distill")
n=6

for model in "${models[@]}"
do
    for (( i=1; i<=n; i++ ))
    do
        CUDA_VISIBLE_DEVICES="${cuda_num}" python applications/generate_multiple_video.py \
                                    --network checkpoints/pix2pix3d_seg2face.pkl \
                                    --outdir ../sketch2mask/results/${model}/inference/pix2pix3d_sgdv/ \
                                    --input_dir ../sketch2mask/results/${model}/inference/pred_mask/ \
                                    --cfg seg2face
    done
done