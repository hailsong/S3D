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


class SketchSegmentationDistilDataset(Dataset):
    def __init__(self, sketch_dir, mask_dir, style_dir=None, transform=None):
        self.sketch_dir = sketch_dir
        self.mask_dir = mask_dir
        self.style_dir = style_dir
        self.transform = transform

        self.sketch_images = sorted(os.listdir(sketch_dir))
        self.mask_images = sorted(os.listdir(mask_dir))
        if style_dir is not None:
            self.style_vectors = sorted(os.listdir(style_dir))

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

        if self.style_dir:
            style_path = os.path.join(self.style_dir, self.style_vectors[idx])
            style_vector = torch.load(style_path)
            return sketch, mask, style_vector
        else:
            return sketch, mask
    

class SketchSegmentationDatasetBackup(Dataset):
    def __init__(self, sketch_dir, mask_dir, style_dir=None, transform=None):
        self.sketch_dir = sketch_dir
        self.mask_dir = mask_dir
        self.style_dir = style_dir
        self.transform = transform

        self.sketch_images = sorted(os.listdir(sketch_dir))
        self.mask_images = sorted(os.listdir(mask_dir))
        if style_dir is not None:
            self.style_embeds = sorted(os.listdir(style_dir))

    def __len__(self):
        return len(self.sketch_images)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.sketch_dir, self.sketch_images[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_images[idx])

        sketch = Image.open(sketch_path).convert("L")  # 그레이스케일 스케치 이미지
        mask = Image.open(mask_path).convert("L")      # 그레이스케일 마스크 이미지
        
        if self.style_dir:
            style_path = os.path.join(self.style_dir, self.style_embeds[idx])
            style_embed = torch.load(style_path)
            style_embed = style_embed.view(512, 7)

        if self.transform:
            sketch = self.transform(sketch)
            # mask = self.transform(mask)
        
        mask = torch.tensor(np.array(mask))

        if not self.style_dir:
            return sketch, mask
        else:
            return sketch, mask, style_embed
