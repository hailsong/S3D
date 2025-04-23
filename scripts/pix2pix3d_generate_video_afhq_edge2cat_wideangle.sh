cuda_num=7

cd pix2pix3D

id="00001558"
input="../data/cat/test/afhqcat_edge_pidinet/img${id}.png"
outdir="../inference_results/pix2pix3d_wideangle/edge2cat/${id}"

mkdir -p ${outdir}

CUDA_VISIBLE_DEVICES="${cuda_num}" python applications/generate_video_wide_angle.py \
                            --network checkpoints/pix2pix3d_edge2cat.pkl \
                            --outdir ${outdir} \
                            --random_seed 876454 \
                            --input ${input} \
                            --cfg edge2cat
