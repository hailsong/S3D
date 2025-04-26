model_name='sketch2mask_face_aug'

CUDA_VISIBLE_DEVICES='7' python inference.py \
                            --model_path results/${model_name}/best_unet_model.pth \
                            --sketch_path ../data/celebamask/test/sketch/ \
                            --mask_path ../data/celebamask/test/mask/ \
                            --output_path results/${model_name}/inference/ \
                            --dataset_type celeba
