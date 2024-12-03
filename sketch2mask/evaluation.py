import torch
import numpy as np
from dataset import SketchSegmentationDataset
from network import UNetStyleDistil
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pickle
from metrics import compute_fid, compute_kid, compute_ap, compute_fvv


#-------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

test_dataset = SketchSegmentationDataset(
    sketch_dir="../data/celebamask_test_sketch",
    mask_dir="../data/celebamask_test_label",
    transform=transform,
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#-------------------------------------------------

file_path = '../sketch2mask/metrics_model/inception-2015-12-05.pkl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(file_path, 'rb') as f:
    inception = pickle.load(f)
    inception.to(device)

def get_inception_features(images):
    if images.dim() == 3:
        images = images.unsqueeze(1)  
    images = images.to(torch.float32) / 255.0
    images_resized = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear')

    images_resized = images_resized.clamp(0, 1)

    if images_resized.shape[1] == 1:
        images_resized = images_resized.repeat(1, 3, 1, 1)

    images_resized = images_resized.to(device)  

    inception.eval()

    with torch.no_grad():
        features = inception(images_resized)

    return features

#-------------------------------------------------

num_classes = 19

model = UNetStyleDistil(in_channels=1, out_channels=num_classes, init_features=64, bottleneck_features=512).to(device)
model.load_state_dict(torch.load('../sketch2mask/best_unet_model.pth', map_location=device))

#-------------------------------------------------
torch.cuda.empty_cache()
model.eval()
with torch.no_grad():
    real_features = []
    gen_features = []
    all_real_masks = []
    all_pred_probs = []

    test_loader_tqdm = tqdm(test_loader, desc="Inference", ncols=100)
    for idx, (sketches, masks) in enumerate(test_loader_tqdm):
        sketches = sketches.to(device)
        masks = masks.to(device)

        outputs, style_embed = model(sketches)
        preds = torch.argmax(outputs, dim=1).float()
        pred_probs = torch.nn.functional.softmax(outputs, dim=1)

        real_feature = get_inception_features(masks).detach().cpu().numpy()
        real_features.append(real_feature)
        gen_feature = get_inception_features(preds).detach().cpu().numpy()
        gen_features.append(gen_feature)

        real_mask = masks.detach().cpu()  
        all_real_masks.append(real_mask)
        pred_prob = pred_probs.detach().cpu()  
        all_pred_probs.append(pred_prob)

        if idx % 50 == 0:
            torch.cuda.empty_cache()
            

    real_features = np.concatenate(real_features, axis=0)
    gen_features = np.concatenate(gen_features, axis=0)

#------------------------------------------------------------------
# Metric 계산
print("Computing metrics...")
fid = compute_fid(real_features, gen_features)
print(f"FID Score: {fid}")

kid = compute_kid(real_features, gen_features)
print(f"KID Score: {kid}")

ap = compute_ap(all_real_masks, all_pred_probs, 500, device)
print(f"Average Precision (AP): {ap}")
print("Finished.")
#-------------------------------------------------
