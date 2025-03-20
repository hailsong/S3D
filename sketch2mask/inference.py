import os
import torch
import numpy as np
from network import UNet, UNetMod, UNetStyleDistil
from dataset import SketchSegmentationDataset
from torchvision import transforms
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
    batch = 64
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

    gray_mask_path = os.path.join(output_path, 'gray')
    rgb_mask_path = os.path.join(output_path, 'rgb')

    # 각 샘플에 대해 예측 수행
    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc="Inference", ncols=100)
        for idx, (sketch, mask) in enumerate(test_loader_tqdm):
            sketch = sketch.to(device)
            mask = mask.to(device)

            outputs = model(sketch)
            preds = torch.argmax(outputs, dim=1)  # (B, H, W)

            # CPU로 이동 및 numpy 변환
            preds_list = list(preds.cpu().numpy())    # [(H, W), ...]

            for j in range(len(preds_list)):
                p = preds_list[j]
                img_index = (idx * batch) + j
                if dataset_type == 'afhq':
                    img_fname = 'img'+str(img_index).zfill(8)+'.png'
                elif dataset_type == 'celeba':
                    img_fname = str(img_index).zfill(5)+'.png'

                # save gray path
                save_path_gray = os.path.join(gray_mask_path, img_fname)
                Image.fromarray(p).save(save_path_gray)

                # save rgb path
                save_path_rgb = os.path.join(rgb_mask_path, img_fname)
                Image.fromarray(convert_mask_to_rgb(p)).save(save_path_rgb)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a single sketch into segmentation masks")
    parser.add_argument("--model_path", type=str, required=True, help="Model path")
    parser.add_argument("--sketch_path", type=str, required=True, help="Input sketch directory")
    parser.add_argument("--mask_path", type=str, required=True, help="Input mask directory")
    parser.add_argument("--output_path", type=str, required=True, help="output_file_path")
    parser.add_argument("--dataset_type", type=str, required=True, help="dataset type (afhq/celeba)")
    args = parser.parse_args()

    main(args.model_path, args.sketch_path, args.mask_path, args.output_path, args.dataset_type)
