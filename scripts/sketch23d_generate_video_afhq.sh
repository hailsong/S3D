cuda_num=7
_id=$(printf "%08d" "5442")

CUDA_VISIBLE_DEVICES="${cuda_num}" python sketch2mask/inference_single.py \
                                    --input_image ./data/cat/test/afhqcat_edge_pidinet/img${_id}.png \
                                    --output_dir sketch2mask/inference_results/afhq/${_id}/ \
                                    --model_path sketch2mask/results/sketch2mask_cat_aug/best_unet_model.pth \
                                    --data_type cat

cd pix2pix3D
OUTDIR=../inference_results/afhq/${_id}
if [ ! -d "$OUTDIR" ]; then
    mkdir -p "$OUTDIR"
fi
CUDA_VISIBLE_DEVICES="${cuda_num}" python applications/generate_video.py \
                                    --network checkpoints/pix2pix3d_seg2cat.pkl \
                                    --outdir "$OUTDIR" \
                                    --random_seed 1 \
                                    --cfg seg2cat \
                                    --input ../sketch2mask/inference_results/afhq/${_id}/img${_id}_mask_gray.png
