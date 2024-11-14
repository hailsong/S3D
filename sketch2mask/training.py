import os
import torch.optim as optim
from network import UNet
from dataset import SketchSegmentationDataset
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 하이퍼파라미터 설정
learning_rate = 5e-5
batch_size = 8
num_epochs = 10

# 결과 저장을 위한 디렉토리 생성
if not os.path.exists('sample_images'):
    os.makedirs('sample_images')

if not os.path.exists('inference_results'):
    os.makedirs('inference_results')

# 데이터 변환
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# 데이터셋 및 데이터로더 생성
train_dataset = SketchSegmentationDataset(
    sketch_dir="../data/celebamask_train_sketch/",
    mask_dir="../data/celebamask_train_label/",
    transform=transform,
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 모델, 손실 함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=1, init_features=64, bottleneck_features=512).to(device)

criterion = nn.MSELoss()  # 손실 함수를 MSELoss로 변경

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델의 마지막 활성화 함수 제거
# network.py 파일에서 UNet 클래스의 forward 함수 수정 필요

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", ncols=100)
    for batch_idx, (sketches, masks) in enumerate(train_loader_tqdm):
        sketches = sketches.to(device)
        masks = masks.to(device)

        # 순전파
        outputs = model(sketches)
        loss = criterion(outputs, masks)

        # 역전파 및 옵티마이저 스텝
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # tqdm 진행률 바 업데이트
        train_loader_tqdm.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # 샘플 이미지 저장
    model.eval()
    with torch.no_grad():
        sample_sketches, sample_masks = next(iter(train_loader))
        sample_sketches = sample_sketches.to(device)
        outputs = model(sample_sketches)
        outputs = outputs.cpu()

        # 원본 스케치, 실제 마스크, 예측 마스크를 저장
        images_to_save = torch.stack([sample_sketches.cpu()[0], sample_masks[0], outputs[0]], dim=0)
        grid = utils.make_grid(images_to_save, nrow=3, normalize=True)
        utils.save_image(grid, f'sample_images/epoch_{epoch+1}.png')

    model.train()

# 모델 저장
torch.save(model.state_dict(), 'unet_model.pth')

# 테스트 데이터셋 및 데이터로더 생성
test_dataset = SketchSegmentationDataset(
    sketch_dir="../data/celebamask_test_sketch/",
    mask_dir="../data/celebamask_test_label/",
    transform=transform,
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 평가 모드
model.eval()
with torch.no_grad():
    test_loader_tqdm = tqdm(test_loader, desc="Inference", ncols=100)
    for idx, (sketches, masks) in enumerate(test_loader_tqdm):
        sketches = sketches.to(device)
        masks = masks.to(device)

        outputs = model(sketches)
        outputs = outputs.cpu()

        # 결과 시각화 및 저장
        sketch_img = transforms.ToPILImage()(sketches.cpu().squeeze(0))
        mask_img = transforms.ToPILImage()(masks.cpu().squeeze(0))
        output_img = transforms.ToPILImage()(outputs.squeeze(0))

        # 시각화된 이미지를 저장
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(sketch_img, cmap='gray')
        axs[0].set_title('Sketch')
        axs[0].axis('off')

        axs[1].imshow(mask_img, cmap='gray')
        axs[1].set_title('Ground Truth')
        axs[1].axis('off')

        axs[2].imshow(output_img, cmap='gray')
        axs[2].set_title('Prediction')
        axs[2].axis('off')

        plt.savefig(f'inference_results/result_{idx+1}.png')
        plt.close()
