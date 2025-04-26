cuda_num=7

cd pix2pix3D

ids=("01" "02" "03" "04" "05" "06")

for id in "${ids[@]}"
do
    for (( j=1; j<=5; j++ ))
    do
        input="../sketch2mask/inference_results/cat_scribble/sketch2mask_cat_distill_aug/scribble_${id}_resize_mask_gray.png"
        outdir="../inference_results/s3d_wideangle/scribble2cat/scribble_${id}"

        mkdir -p ${outdir}

        CUDA_VISIBLE_DEVICES="${cuda_num}" python applications/generate_video_wide_angle.py \
                                    --network checkpoints/pix2pix3d_seg2cat.pkl \
                                    --outdir ${outdir} \
                                    --random_seed $RANDOM \
                                    --input ${input} \
                                    --cfg seg2cat
    done
done
