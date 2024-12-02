import torch
import argparse
from PIL import Image
from torchvision import transforms
import os
import numpy as np
from network import UNetStyleDistil

def load_model(model_path, device, num_classes=19):
    # 모델 생성 및 로드
    model = UNetStyleDistil(in_channels=1, out_channels=num_classes, init_features=64, bottleneck_features=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

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

def process_single_image(model, input_image_path, output_dir, transform, device):
    # 스케치 이미지 로드
    sketch_image = Image.open(input_image_path).convert('L')  # Grayscale
    sketch_tensor = transform(sketch_image).unsqueeze(0).to(device)  # Add batch dimension

    # 모델 추론
    with torch.no_grad():
        output, _ = model(sketch_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 파일명 추출
    filename = os.path.basename(input_image_path).split('.')[0]

    # 원본 마스크 저장 (그레이스케일)
    save_path_gray = os.path.join(output_dir, f"{filename}_mask_gray.png")
    pred_mask_gray = Image.fromarray(pred_mask.astype(np.uint8))
    pred_mask_gray.save(save_path_gray)
    print(f"Grayscale mask saved at {save_path_gray}")

    # RGB 마스크 저장
    save_path_rgb = os.path.join(output_dir, f"{filename}_mask_rgb.png")
    pred_mask_rgb = convert_mask_to_rgb(pred_mask)
    pred_mask_img = Image.fromarray(pred_mask_rgb)
    pred_mask_img.save(save_path_rgb)
    print(f"RGB mask saved at {save_path_rgb}")

if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description="Process a single sketch into segmentation masks")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input sketch image")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save masks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    args = parser.parse_args()

    # 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # 모델 로드
    model = load_model(args.model_path, device)

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 단일 이미지 처리
    process_single_image(model, args.input_image, args.output_dir, transform, device)
