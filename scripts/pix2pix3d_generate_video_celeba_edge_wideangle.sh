cuda_num=7

cd pix2pix3D

id="00469"
input="../data/celebamask/test/sketch/${id}.png"
outdir="../inference_results/pix2pix3d_wideangle/celebamask_edge_00008/${id}"

mkdir -p ${outdir}

CUDA_VISIBLE_DEVICES="${cuda_num}" python applications/generate_video_wide_angle.py \
                            --network logs/00008-celeba-celebamask_train-gpus1-batch16-gamma0.3/network-snapshot-000280.pkl \
                            --outdir ${outdir} \
                            --random_seed 1 \
                            --input ${input} \
                            --cfg edge2face
