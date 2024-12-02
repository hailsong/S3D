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

# 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7

# 모델 생성 및 로드
model = UNet(in_channels=1, out_channels=num_classes, init_features=64, bottleneck_features=512).to(device)
model.load_state_dict(torch.load('best_unet_model.pth', map_location=device))
model.eval()

# 데이터 변환
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 이미지 크기 설정 (트레이닝 시 설정한 크기와 동일하게)
    transforms.ToTensor(),
])

# 테스트 데이터셋 및 데이터로더 생성
test_dataset = SketchSegmentationDataset(
    sketch_dir="../data/celebamask_test_sketch/",
    mask_dir="../data/celebamask_test_label/",
    transform=transform,
)

# 테스트 세트에서 5개 샘플 선택
num_samples = 5
indices = np.random.choice(len(test_dataset), num_samples, replace=False)
test_samples = [test_dataset[i] for i in indices]

# 시각화를 위한 컬러 맵 정의
def decode_segmap(image, num_classes):
    label_colors = np.array([
        (0, 0, 0),        # 0: Background
        (128, 0, 0),      # 1: Skin
        (0, 128, 0),      # 2: Nose
        (128, 128, 0),    # 3: Eye Glasses
        (0, 0, 128),      # 4: Left Eye
        (128, 0, 128),    # 5: Right Eye
        (0, 128, 128),    # 6: Left Brow
        (128, 128, 128),  # 7: Right Brow
        (64, 0, 0),       # 8: Left Ear
        (192, 0, 0),      # 9: Right Ear
        (64, 128, 0),     # 10: Mouth
        (192, 128, 0),    # 11: Upper Lip
        (64, 0, 128),     # 12: Lower Lip
        (192, 0, 128),    # 13: Hair
        (64, 128, 128),   # 14: Hat
        (192, 128, 128),  # 15: Ear Ring
        (0, 64, 0),       # 16: Necklace
        (128, 64, 0),     # 17: Neck
        (0, 192, 0),      # 18: Cloth
        # 필요한 경우 더 많은 클래스 색상 추가
    ])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, num_classes):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

# 이미지 리스트 초기화
sketches_list = []
gt_masks_list = []
pred_masks_list = []

# 각 샘플에 대해 예측 수행
with torch.no_grad():
    for sketch, mask in test_samples:
        sketch = sketch.to(device).unsqueeze(0)  # 배치 차원 추가
        mask = mask.to(device).unsqueeze(0)      # 배치 차원 추가

        outputs = model(sketch)
        preds = torch.argmax(outputs, dim=1)  # (1, H, W)

        # CPU로 이동 및 numpy 변환
        sketch_np = sketch.cpu().squeeze(0).numpy()  # (1, H, W)
        mask_np = mask.cpu().squeeze(0).numpy()      # (H, W)
        preds_np = preds.cpu().squeeze(0).numpy()    # (H, W)

        # 스케치 이미지는 그레이스케일로 변환
        sketch_img = np.transpose(sketch_np, (1, 2, 0)) * 255.0  # (H, W, 1)
        sketch_img = sketch_img.astype(np.uint8).squeeze(2)      # (H, W)

        # 마스크와 예측 결과에 컬러 맵 적용
        gt_mask_img = decode_segmap(mask_np, num_classes)
        pred_mask_img = decode_segmap(preds_np, num_classes)

        # 리스트에 추가
        sketches_list.append(sketch_img)
        gt_masks_list.append(gt_mask_img)
        pred_masks_list.append(pred_mask_img)

# 그리드 이미지 생성
fig, axs = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))

for i in range(num_samples):
    # 스케치 이미지
    axs[i, 0].imshow(sketches_list[i], cmap='gray')
    axs[i, 0].set_title('Sketch')
    axs[i, 0].axis('off')

    # 실제 마스크
    axs[i, 1].imshow(gt_masks_list[i])
    axs[i, 1].set_title('Ground Truth Mask')
    axs[i, 1].axis('off')

    # 예측된 마스크
    axs[i, 2].imshow(pred_masks_list[i])
    axs[i, 2].set_title('Predicted Mask')
    axs[i, 2].axis('off')

plt.tight_layout()
plt.savefig('test_results_grid.png')
plt.close()
