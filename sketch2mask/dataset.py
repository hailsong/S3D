import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class SketchSegmentationDataset(Dataset):
    def __init__(self, sketch_dir, mask_dir, transform=None):
        self.sketch_dir = sketch_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.sketch_images = sorted(os.listdir(sketch_dir))
        self.mask_images = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.sketch_images)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketch_dir, self.sketch_images[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_images[idx])

        sketch = Image.open(sketch_path).convert("L")  # 그레이스케일 스케치 이미지
        mask = Image.open(mask_path).convert("L")      # 그레이스케일 마스크 이미지

        if self.transform:
            sketch = self.transform(sketch)
            # mask = self.transform(mask)
        
        mask = torch.tensor(np.array(mask))

        return sketch, mask
