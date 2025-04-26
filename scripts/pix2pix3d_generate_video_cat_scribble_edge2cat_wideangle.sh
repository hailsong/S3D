cuda_num=7

cd pix2pix3D

for (( i=1; i<=6; i++ ))
do
    id="scribble_0${i}"
    input="../data/cat/scribble/${id}_resize.png"
    outdir="../inference_results/pix2pix3d_wideangle/scribble2cat/${id}"

    mkdir -p ${outdir}

    for (( j=1; j<=5; j++ ))
    do
        CUDA_VISIBLE_DEVICES="${cuda_num}" python applications/generate_video_wide_angle.py \
                                    --network checkpoints/pix2pix3d_edge2cat.pkl \
                                    --outdir ${outdir} \
                                    --random_seed $RANDOM \
                                    --input ${input} \
                                    --cfg edge2cat
    done
done
