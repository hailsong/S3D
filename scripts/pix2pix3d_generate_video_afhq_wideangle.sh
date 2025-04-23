cuda_num=7

cd pix2pix3D

model="sketch2mask_cat_distill_aug"
id=88
input="../sketch2mask/results/${model}/inference/pred_mask/mask_${id}.png"
outdir="../sketch2mask/results/${model}/inference/pix2pix3d_wideangle/mask_${id}"

mkdir -p ${outdir}

CUDA_VISIBLE_DEVICES="${cuda_num}" python applications/generate_video_wide_angle.py \
                            --network checkpoints/pix2pix3d_seg2cat.pkl \
                            --outdir ${outdir} \
                            --random_seed 322457 \
                            --input ${input} \
                            --cfg seg2cat
