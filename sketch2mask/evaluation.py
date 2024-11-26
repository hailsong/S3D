import torch
import numpy as np
import requests
from dataset import SketchSegmentationDataset
from network import UNetStyleDistil
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pickle
import wandb
from metrics import compute_fid, compute_kid, compute_ap, compute_fvv


#-------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

test_dataset = SketchSegmentationDataset(
    sketch_dir="../data/celebamask/test/sketch/",
    mask_dir="../data/celebamask/test/mask/",
    transform=transform,
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#-------------------------------------------------

response = requests.get('https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl', stream=True)
with open('/home/s2/naeunlee/sketch2face3D/sketch2mask/metrics', 'wb') as f:
    f.write(response.content)

inception = pickle.load(f)

def get_inception_features(images):
    images_resized = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear')
    images_resized = images_resized.clamp(0, 1)
    images_np = images_resized.permute(0, 2, 3, 1).cpu().numpy()
    features = inception.run(images_np, num_gpus=1, assume_frozen=True)

    return features

#-------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 19  # Number of Segmentation Classes

model = UNetStyleDistil(in_channels=1, out_channels=num_classes, init_features=64, bottleneck_features=512).to(device)

model.load_state_dict(torch.load('/home/s2/naeunlee/sketch2face3D/sketch2mask/final_unet_model.pth'))

#-------------------------------------------------

model.eval()
with torch.no_grad():
    real_features = []
    gen_features = []
    all_real_masks = []
    all_pred_probs = []
    fvv_image_pairs = []

    test_loader_tqdm = tqdm(test_loader, desc="Inference", ncols=100)
    for idx, (sketches, masks) in enumerate(test_loader_tqdm):
        sketches = sketches.to(device)
        masks = masks.to(device)

        outputs, style_embed = model(sketches)
        preds = torch.argmax(outputs, dim=1).unsqueeze(1).float()
        pred_probs = torch.nn.functional.softmax(outputs, dim=1)

        real_features.append(get_inception_features(masks, inception))
        gen_features.append(get_inception_features(preds, inception))

        all_real_masks.append(masks)
        all_pred_probs.append(pred_probs)

        for mask, pred in zip(masks, preds):
            real_image = transforms.ToPILImage()(mask.squeeze(0).cpu())
            pred_image = transforms.ToPILImage()(pred.squeeze(0).cpu())
            fvv_image_pairs.append((real_image, pred_image))


    real_features = np.concatenate(real_features, axis=0)
    gen_features = np.concatenate(gen_features, axis=0)

    fid = compute_fid(real_features, gen_features)
    kid = compute_kid(real_features, gen_features)

    all_real_masks = torch.cat(all_real_masks, dim=0)
    all_pred_probs = torch.cat(all_pred_probs, dim=0)
    ap = compute_ap(all_real_masks, all_pred_probs)

    fvv = compute_fvv(fvv_image_pairs)

#-------------------------------------------------
