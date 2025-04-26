cuda_num=6

cd pix2pix3D

n=5

for (( i=1; i<=n; i++ ))
do
    CUDA_VISIBLE_DEVICES="${cuda_num}" python applications/generate_multiple_video.py \
                                --network logs/00008-celeba-celebamask_train-gpus1-batch16-gamma0.3/network-snapshot-000280.pkl \
                                --outdir ../results/pix2pix3d_edge2face_sgdv/ \
                                --input_dir ../data/celebamask/test/mask/ \
                                --cfg edge2face
done
