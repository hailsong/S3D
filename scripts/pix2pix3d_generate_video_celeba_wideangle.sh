cuda_num=7

cd pix2pix3D

# model="sketch2mask_face_distill_aug"
# id=469
# input="../sketch2mask/results/${model}/inference/pred_mask/mask_${id}.png"
# outdir="../sketch2mask/results/${model}/inference/pix2pix3d_wideangle/mask_${id}"

id="00468"
input="../data/celebamask/test/mask/${id}.png"
outdir="../inference_results/pix2pix3d_wideangle/celebamask/${id}"

mkdir -p ${outdir}

CUDA_VISIBLE_DEVICES="${cuda_num}" python applications/generate_video_wide_angle.py \
                            --network checkpoints/pix2pix3d_seg2face.pkl \
                            --outdir ${outdir} \
                            --random_seed 235543 \
                            --input ${input} \
                            --cfg seg2face
