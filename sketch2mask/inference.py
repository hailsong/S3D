import os
import torch
import numpy as np
from network import UNet, UNetMod, UNetStyleDistil
from dataset import SketchSegmentationDataset
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import argparse

def convert_mask_to_rgb(mask):
    # Color mapping
    color_map = [
        [255, 255, 255],  # White
        [204, 0, 0],      # Dark Red
        [76, 153, 0],     # Olive Green
        [204, 204, 0],    # Mustard Yellow
        [51, 51, 255],    # Bright Blue
        [204, 0, 204],    # Purple
        [0, 255, 255],    # Cyan
        [255, 204, 204],  # Light Pink
        [102, 51, 0],     # Brown
        [255, 0, 0],      # Red
        [102, 204, 0],    # Lime Green
        [255, 255, 0],    # Yellow
        [0, 0, 153],      # Navy Blue
        [0, 0, 204],      # Royal Blue
        [255, 51, 153],   # Hot Pink
        [0, 204, 204],    # Teal
        [0, 51, 0],       # Dark Green
        [255, 153, 51],   # Orange
        [0, 204, 0]       # Green
    ]
    height, width = mask.shape
    rgb_tensor = np.zeros((height, width, 3), dtype=np.uint8)
    for value, color in enumerate(color_map):
        rgb_tensor[mask == value] = color
    return rgb_tensor

def main(model_path, sketch_path, mask_path, output_path, dataset_type):
    # 모델 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset_type == 'afhq':
        num_classes = 6
    elif dataset_type == 'celeba':
        num_classes = 19
    else:
        raise NotImplementedError

    # 모델 생성 및 로드
    model = UNet(in_channels=1, out_channels=num_classes, init_features=64, bottleneck_features=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 데이터 변환
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 이미지 크기 설정 (트레이닝 시 설정한 크기와 동일하게)
        transforms.ToTensor(),
    ])

    # 테스트 데이터셋 및 데이터로더 생성
    test_dataset = SketchSegmentationDataset(
        sketch_dir=sketch_path,
        mask_dir=mask_path,
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Inference
    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc="Inference", ncols=100)
        for idx, (sketches, masks) in enumerate(test_loader_tqdm):
            sketches = sketches.to(device)
            masks = masks.to(device)

            outputs, style_embed = model(sketches)
            preds = torch.argmax(outputs, dim=1).unsqueeze(1).float()

            # Save Output
            images_to_save = torch.stack(
                [
                    sketches.cpu()[0].repeat(3, 1, 1) * 255, 
                    convert_mask_to_rgb(masks.unsqueeze(1).cpu()[0]), 
                    convert_mask_to_rgb(preds.cpu()[0])
                ]
                , dim=0)
            grid = utils.make_grid(images_to_save, nrow=3, normalize=True)
            utils.save_image(grid, os.path.join(os.path.join(output_path, 'rgb_mask'), f'result_{idx}.png'))

            # Save Original Mask
            Image.fromarray(preds.cpu()[0].squeeze().numpy().astype(np.uint8), mode='L').save(os.path.join(os.path.join(output_path, 'pred_mask'), f'mask_{idx}.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a single sketch into segmentation masks")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--sketch_path", type=str, required=True, help="Input sketch directory")
    parser.add_argument("--mask_path", type=str, required=True, help="Input mask directory")
    parser.add_argument("--output_path", type=str, required=True, help="output_file_path")
    parser.add_argument("--dataset_type", type=str, required=True, help="dataset type (afhq/celeba)")
    args = parser.parse_args()

    main(args.model_path, args.sketch_path, args.mask_path, args.output_path, args.dataset_type)
