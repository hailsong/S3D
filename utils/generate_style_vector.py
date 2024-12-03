import os
new_working_dir = "../pix2pix3D/applications"
os.chdir(new_working_dir)

import sys
sys.path.append('../')

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import dnnlib
import legacy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# load model
with dnnlib.util.open_url('../checkpoints/pix2pix3d_seg2face.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'].eval().to('cuda')

# print hyperparameters
print(f'w_dim: {G.backbone.mapping.w_dim}')
print(f'in_resolution: {G.backbone.mapping.in_resolution}')
print(f'in_channels: {G.backbone.mapping.in_channels}')
print(f'one_hot: {G.backbone.mapping.one_hot}')

# init dataset path
_root = '../../data'
mask_root = os.path.join(_root, 'celebamask_train_label')
w_plus_root = os.path.join(_root, 'w_plus')

if not os.path.exists(w_plus_root):
    os.makedirs(w_plus_root)

n_mask = len(os.listdir(mask_root))

# generate style vector
for idx in tqdm(range(n_mask)):
    w_plus_path = os.path.join(w_plus_root, f'{str(idx).zfill(5)}.pth')
    
    # only if there's no file
    if not os.path.isfile(w_plus_path):
        # load mask
        input_label = Image.open(os.path.join(mask_root, f'{str(idx).zfill(5)}.png'))
        input_label = np.array(input_label).astype(np.uint8)
        input_label = torch.from_numpy(input_label).unsqueeze(0).unsqueeze(0).to('cuda')

        # preprocess mask
        batch = {'mask': input_label}
        in_channels = G.backbone.mapping.in_channels
        mask_one_hot = torch.nn.functional.one_hot(batch['mask'].squeeze(1).long(), in_channels).permute(0,3,1,2)

        # generate style vector
        w_plus = G.backbone.mapping.embed_mask(mask_one_hot.to(torch.float32))['ws']
        
        # save
        torch.save(w_plus, w_plus_path)
