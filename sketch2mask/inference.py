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

import torch

def convert_mask_to_rgb(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a 2D (or 1×H×W) integer mask to a 3×H×W RGB image, entirely on the same device as `mask`.
    
    Args:
        mask: torch.Tensor of shape (H, W) or (1, H, W), with integer values in [0, N−1],
              located on CPU or CUDA.
    Returns:
        torch.Tensor of shape (3, H, W), dtype=torch.uint8, on mask.device.
    """
    # 1) Squeeze away a leading channel dim if present
    if mask.dim() == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    
    # 2) Make sure it’s integer for indexing
    mask = mask.long()
    
    # 3) Build our (N×3) color map on the same device
    color_map = torch.tensor([
        [255, 255, 255],  # White
        [204,   0,   0],  # Dark Red
        [ 76, 153,   0],  # Olive Green
        [204, 204,   0],  # Mustard Yellow
        [ 51,  51, 255],  # Bright Blue
        [204,   0, 204],  # Purple
        [  0, 255, 255],  # Cyan
        [255, 204, 204],  # Light Pink
        [102,  51,   0],  # Brown
        [255,   0,   0],  # Red
        [102, 204,   0],  # Lime Green
        [255, 255,   0],  # Yellow
        [  0,   0, 153],  # Navy Blue
        [  0,   0, 204],  # Royal Blue
        [255,  51, 153],  # Hot Pink
        [  0, 204, 204],  # Teal
        [  0,  51,   0],  # Dark Green
        [255, 153,  51],  # Orange
        [  0, 204,   0],  # Green
    ], dtype=torch.uint8, device=mask.device)
    
    # 4) Index into the color map: gives H×W×3 uint8 tensor on GPU if mask is on GPU
    rgb_hw3 = color_map[mask]  
    
    # 5) Permute to 3×H×W
    return rgb_hw3.permute(2, 0, 1)


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
    model = UNetStyleDistil(in_channels=1, out_channels=num_classes, init_features=64, bottleneck_features=512).to(device)
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
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Inference
    # import time
    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc="Inference", ncols=100)
        for idx, (sketches, masks) in enumerate(test_loader_tqdm):
            # tt = time.time()
            sketches = sketches.to(device)
            masks = masks.to(device)
            # print(f'timestamp1: {time.time()-tt}'); tt = time.time()
            outputs, style_embed = model(sketches)
            # print(f'timestamp2: {time.time()-tt}'); tt = time.time()
            preds = torch.argmax(outputs, dim=1).unsqueeze(1).float()
            # print(f'timestamp3: {time.time()-tt}'); tt = time.time()

            # Save Output
            if idx < 10:
                images_to_save = torch.stack(
                    [
                        sketches.cpu()[0].repeat(3, 1, 1) * 255, 
                        convert_mask_to_rgb(masks.unsqueeze(1)[0]).cpu(), 
                        convert_mask_to_rgb(preds[0]).cpu()
                    ]
                    , dim=0)
                # print(f'timestamp4: {time.time()-tt}'); tt = time.time()
                grid = utils.make_grid(images_to_save, nrow=3, normalize=True)
                # print(f'timestamp5: {time.time()-tt}'); tt = time.time()
                utils.save_image(grid, os.path.join(os.path.join(output_path, 'rgb_mask'), f'result_{idx}.png'))
                # print(f'timestamp6: {time.time()-tt}'); tt = time.time()

            # Save Original Mask
            Image.fromarray(preds.cpu()[0].squeeze().numpy().astype(np.uint8), mode='L').save(os.path.join(os.path.join(output_path, 'pred_mask'), f'mask_{idx}.png'))
            # print(f'timestamp7: {time.time()-tt}'); tt = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a single sketch into segmentation masks")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--sketch_path", type=str, required=True, help="Input sketch directory")
    parser.add_argument("--mask_path", type=str, required=True, help="Input mask directory")
    parser.add_argument("--output_path", type=str, required=True, help="output_file_path")
    parser.add_argument("--dataset_type", type=str, required=True, help="dataset type (afhq/celeba)")
    args = parser.parse_args()

    main(args.model_path, args.sketch_path, args.mask_path, args.output_path, args.dataset_type)
