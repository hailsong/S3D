import os
new_working_dir = "./pix2pix3D/applications"
os.chdir(new_working_dir)

import sys
sys.path.append('../')

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import dnnlib
import legacy
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main(model_path, data_root, mask_fname, output_fname):
    # load model
    with dnnlib.util.open_url(model_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].eval().to('cuda')

    # print hyperparameters
    print(f'w_dim: {G.backbone.mapping.w_dim}')
    print(f'in_resolution: {G.backbone.mapping.in_resolution}')
    print(f'in_channels: {G.backbone.mapping.in_channels}')
    print(f'one_hot: {G.backbone.mapping.one_hot}')

    # init dataset path
    _root = data_root
    mask_root = os.path.join(_root, mask_fname)
    w_plus_root = os.path.join(_root, output_fname)

    if not os.path.exists(w_plus_root):
        os.makedirs(w_plus_root)

    masks = sorted(os.listdir(mask_root))

    # generate style vector
    for m in tqdm(masks):
        w_plus_path = os.path.join(w_plus_root, f'{m.split(".")[0]}.pth')
        
        # only if there's no file
        if not os.path.isfile(w_plus_path):
            # load mask
            input_label = Image.open(os.path.join(mask_root, m))
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


if __name__ == '__main__':
    main(model_path='../../checkpoints/pix2pix3d_seg2cat.pkl',
         data_root='../../data/cat',
         mask_fname='afhqcat_seg_6c_no_nose',
         output_fname='afhqcat_seg_w_plus',
         )
