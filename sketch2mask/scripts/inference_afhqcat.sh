model_name='sketch2mask_cat_aug'

CUDA_VISIBLE_DEVICES='7' python inference.py \
                            --model_path results/${model_name}/best_unet_model.pth \
                            --sketch_path ../data/cat/test/afhqcat_edge_pidinet/ \
                            --mask_path ../data/cat/test/afhqcat_seg_6c_no_nose/ \
                            --output_path results/${model_name}/inference/ \
                            --dataset_type afhq
