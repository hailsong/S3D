python ./utils/preprocess_sketch.py --input_image ./demo_scripts/example_image.png --output_image ./demo_scripts/example_sketch.png

python ./sketch2mask/inference_single.py --input_image ./demo_scripts/example_sketch.png --output_dir ./demo_scripts/ --model_path ./sketch2mask/best_unet_model.pth

# cd pix2pix3D
# python applications/generate_video.py --network checkpoints/pix2pix3d_seg2face.pkl --outdir examples --random_seed 1 --cfg seg2face --input examples/00004_seg.png

cd pix2pix3D
python applications/generate_video.py --network checkpoints/pix2pix3d_seg2face.pkl --outdir ../demo_scripts --random_seed 1 --cfg seg2face --input ../demo_scripts/example_sketch_mask_gray.png