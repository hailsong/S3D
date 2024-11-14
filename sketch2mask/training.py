import os
import torch.optim as optim
from network import UNet, UNetMod
from dataset import SketchSegmentationDataset
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb  # wandb 임포트

# wandb 초기화
wandb.init(project='Sketch-to-Segmentation', config={
    'learning_rate': 5e-5,
    'batch_size': 2,
    'num_epochs': 10,
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss',  # 변경됨
    'image_size': 512,
    'val_split': 0.1,
    'patience': 5,
    'min_delta': 0.0001
})

# 하이퍼파라미터 설정
config = wandb.config
learning_rate = config.learning_rate
batch_size = config.batch_size
num_epochs = config.num_epochs
val_split = config.val_split
patience = config.patience
min_delta = config.min_delta

# 결과 저장을 위한 디렉토리 생성
if not os.path.exists('sample_images'):
    os.makedirs('sample_images')

if not os.path.exists('inference_results'):
    os.makedirs('inference_results')

# 데이터 변환
transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
])

# 전체 데이터셋 생성
full_dataset = SketchSegmentationDataset(
    sketch_dir="../data/celebamask_train_sketch/",
    mask_dir="../data/celebamask_train_label/",
    transform=transform,
)

# 데이터셋 분할
total_size = len(full_dataset)
val_size = int(total_size * val_split)
train_size = total_size - val_size

# 시드 고정 (재현성을 위해)
torch.manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 데이터로더 생성
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 모델, 손실 함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2  # 클래스 수 설정 (배경과 객체)

# 모델 생성 시 out_channels를 클래스 수로 변경
model = UNetMod(in_channels=1, out_channels=num_classes, init_features=64, bottleneck_features=512).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 감시 설정
wandb.watch(model, log='all', log_freq=10)

# Early Stopping을 위한 변수 초기화
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

# 학습 루프
for epoch in range(num_epochs):
    if early_stop:
        print("Early stopping triggered. Training stopped.")
        break

    model.train()
    epoch_loss = 0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train", ncols=100)
    for batch_idx, (sketches, masks) in enumerate(train_loader_tqdm):
        sketches = sketches.to(device)
        masks = masks.to(device)

        # 마스크를 LongTensor로 변환하고 차원 조정
        masks = masks.to(torch.long).squeeze(1)  # (N, 1, H, W) -> (N, H, W)

        # 순전파
        outputs = model(sketches)  # (N, num_classes, H, W)
        loss = criterion(outputs, masks)

        # 역전파 및 옵티마이저 스텝
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # tqdm 진행률 바 업데이트
        train_loader_tqdm.set_postfix(loss=loss.item())

        # 손실 값 로깅
        wandb.log({'Train Loss': loss.item(), 'Epoch': epoch + 1})

    avg_train_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Train Loss: {avg_train_loss:.4f}")

    # 검증 루프
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", ncols=100)
        for batch_idx, (sketches, masks) in enumerate(val_loader_tqdm):
            sketches = sketches.to(device)
            masks = masks.to(device)

            masks = masks.to(torch.long).squeeze(1)  # (N, H, W)

            outputs = model(sketches)
            loss = criterion(outputs, masks)

            val_loss += loss.item()

            # tqdm 진행률 바 업데이트
            val_loader_tqdm.set_postfix(loss=loss.item())

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}")

    # 검증 손실 로깅
    wandb.log({'Validation Loss': avg_val_loss, 'Epoch': epoch + 1})

    # Early Stopping 조건 체크
    if avg_val_loss + min_delta < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0

        # 최적의 모델 저장
        torch.save(model.state_dict(), 'best_unet_model.pth')
        wandb.save('best_unet_model.pth')
        print(f"Validation loss improved. Model saved at epoch {epoch+1}.")
    else:
        epochs_no_improve += 1
        print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= patience:
        print(f"Validation loss did not improve for {patience} consecutive epochs. Early stopping.")
        early_stop = True

    # 샘플 이미지 저장 및 로깅 (검증 세트에서)
    with torch.no_grad():
        sample_sketches, sample_masks = next(iter(val_loader))
        sample_sketches = sample_sketches.to(device)
        sample_masks = sample_masks.to(device)

        outputs = model(sample_sketches)
        preds = torch.argmax(outputs, dim=1).unsqueeze(1).float()

        # 원본 스케치, 실제 마스크, 예측 마스크를 저장
        images_to_save = torch.stack([sample_sketches.cpu()[0], sample_masks[0], preds.cpu()[0]], dim=0)
        grid = utils.make_grid(images_to_save, nrow=3, normalize=True)
        utils.save_image(grid, f'sample_images/epoch_{epoch+1}.png')

        # 샘플 이미지 로깅
        wandb.log({
            'Sample Images': [wandb.Image(grid, caption=f'Epoch {epoch+1} (Validation)')]
        })

# 최종 모델 저장 및 업로드 (필요한 경우)
torch.save(model.state_dict(), 'final_unet_model.pth')
wandb.save('final_unet_model.pth')

# 테스트 데이터셋 및 데이터로더 생성
test_dataset = SketchSegmentationDataset(
    sketch_dir="../data/celebamask_test_sketch/",
    mask_dir="../data/celebamask_test_label/",
    transform=transform,
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 최적의 모델 로드
model.load_state_dict(torch.load('best_unet_model.pth'))

# 평가 모드
model.eval()
with torch.no_grad():
    test_loader_tqdm = tqdm(test_loader, desc="Inference", ncols=100)
    for idx, (sketches, masks) in enumerate(test_loader_tqdm):
        sketches = sketches.to(device)
        masks = masks.to(device)

        outputs = model(sketches)
        preds = torch.argmax(outputs, dim=1).unsqueeze(1).float()

        # 결과 시각화 및 저장
        sketch_img = transforms.ToPILImage()(sketches.cpu().squeeze(0))
        mask_img = transforms.ToPILImage()(masks.cpu().squeeze(0))
        output_img = transforms.ToPILImage()(preds.cpu().squeeze(0))

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

        # 테스트 결과 로깅 (처음 10개만)
        if idx < 10:
            wandb.log({
                f'Test Image {idx+1}': [
                    wandb.Image(sketch_img, caption='Sketch'),
                    wandb.Image(mask_img, caption='Ground Truth'),
                    wandb.Image(output_img, caption='Prediction')
                ]
            })

# wandb 종료
wandb.finish()
