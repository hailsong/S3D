cuda_num=0
_id=46
celebamask_test_id=$(printf "%05d" "${_id}")


# CUDA_VISIBLE_DEVICES="${cuda_num}" python sketch2mask/inference_single.py \
#                                     --input_image ./data/celebamask/test/sketch/${celebamask_test_id}.png \
#                                     --output_dir sketch2mask/inference_results/celebamask/${celebamask_test_id}/ \
#                                     --model_path sketch2mask/results/sketch2mask_face_aug/best_unet_model.pth \
#                                     --data_type celeba
CUDA_VISIBLE_DEVICES="${cuda_num}" python sketch2mask/inference_single.py \
                                    --input_image ./data_eval/test/sketch/${celebamask_test_id}.png \
                                    --output_dir sketch2mask/inference_results/celebamask/${celebamask_test_id}/ \
                                    --model_path ckpts/sketch2mask_face_aug/best_unet_model.pth \
                                    --data_type celeba

cd pix2pix3D
OUTDIR=../inference_results/celebamask/${celebamask_test_id}
if [ ! -d "$OUTDIR" ]; then
    mkdir -p "$OUTDIR"
fi
CUDA_VISIBLE_DEVICES="${cuda_num}" python applications/generate_video.py \
                                    --network checkpoints/pix2pix3d_seg2face.pkl \
                                    --outdir "$OUTDIR" \
                                    --random_seed 1 \
                                    --cfg seg2face \
                                    --input ../sketch2mask/inference_results/celebamask/${celebamask_test_id}/${celebamask_test_id}_mask_gray.png
