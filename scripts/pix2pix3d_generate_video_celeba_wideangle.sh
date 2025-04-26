cuda_num=6

cd pix2pix3D

# model="sketch2mask_face_distill_aug"
# id=469
# input="../sketch2mask/results/${model}/inference/pred_mask/mask_${id}.png"
# outdir="../sketch2mask/results/${model}/inference/pix2pix3d_wideangle/mask_${id}"

id="875"
input="../sketch2mask/results/sketch2mask_face_distill_aug/inference/pred_mask/mask_${id}.png"
outdir="../sketch2mask/results/sketch2mask_face_distill_aug/inference/pix2pix3d_wideangle/mask_${id}"

mkdir -p ${outdir}

CUDA_VISIBLE_DEVICES="${cuda_num}" python applications/generate_video_wide_angle.py \
                            --network checkpoints/pix2pix3d_seg2face.pkl \
                            --outdir ${outdir} \
                            --random_seed $RANDOM \
                            --input ${input} \
                            --cfg seg2face
