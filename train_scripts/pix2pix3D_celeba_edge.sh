cd pix2pix3D

NCCL_P2P_DISABLE=1 python train.py --outdir=../logs \
                --cfg=celeba --seg_weight=2 --data=../data/celebamask_train.zip \
                --mask_data=../data/256_celebasketch_train.zip \
                --data_resolution=256 \
                --render_mask=True --dis_mask=True \
                --data_type=edge \
                --gpus=1 --batch=4 --mbstd-group=1 \
                --gamma=1 --gen_pose_cond=True \
                --random_c_prob=0.5 \
                --lambda_cross_view=1e-4 \
                --lambda_d_semantic=0.1 \
                --lambda_lpips=1 \
                --edge_weight=10 --geometry_layer=9 \
                --wandb_log=False
