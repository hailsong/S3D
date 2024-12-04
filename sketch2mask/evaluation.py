import gc
from dataset import SketchSegmentationDataset
from network import UNetStyleDistil
import numpy as np
from metrics import compute_fid, compute_kid, compute_ap, compute_fvv, compute_miou
import pickle
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


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

load_model = False # True if you want to load model and inference 
output_dir = "../sketch2mask"
num_classes = 19
model = UNetStyleDistil(in_channels=1, out_channels=num_classes, init_features=64, bottleneck_features=512).to(device)
#for model_name in ['best_unet_model','best_unet_model_dist']:
for model_name in ['best_unet_model_dist']:
        
    #model_name = 'best_unet_model_dist'
    print(f'====================={model_name} result=====================')

    if load_model:
            
        model.load_state_dict(torch.load(f'../sketch2mask/{model_name}.pth', map_location=device))
        #-------------------------------------------------
        torch.cuda.empty_cache()
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

                for mask, pred in zip(masks.cpu(), preds.cpu()):
                    real_image = transforms.ToPILImage()(mask.squeeze(0))
                    pred_image = transforms.ToPILImage()(pred.squeeze(0))
                    fvv_image_pairs.append((real_image, pred_image))

                if idx % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    

            real_features = np.concatenate(real_features, axis=0)
            gen_features = np.concatenate(gen_features, axis=0)

        
        # #------------------------------------------------------------------
        if 'dist' in model_name:
            np.save(f"{output_dir}/real_features_dist.npy", real_features)
            np.save(f"{output_dir}/gen_features_dist.npy", gen_features)

            #torch.save(all_real_masks, f"{output_dir}/all_real_masks_dist.pt") # 얘는 어차피 똑같으니 굳이 저장필요 x 
            torch.save(all_pred_probs, f"{output_dir}/all_pred_probs_dist.pt")

            with open(f"{output_dir}/fvv_image_pairs.pkl", "wb") as f:
                pickle.dump(fvv_image_pairs, f)
        else:
            np.save(f"{output_dir}/real_features.npy", real_features)
            np.save(f"{output_dir}/gen_features.npy", gen_features)

            torch.save(all_real_masks, f"{output_dir}/all_real_masks.pt")
            torch.save(all_pred_probs, f"{output_dir}/all_pred_probs.pt")

            with open(f"{output_dir}/fvv_image_pairs.pkl", "wb") as f:
                pickle.dump(fvv_image_pairs, f)        
    else:       
        # Loading 
        real_features = np.load(f"{output_dir}/real_features.npy")
        if 'dist' in model_name:
            gen_features = np.load(f"{output_dir}/gen_features_dist.npy")
            all_real_masks = torch.load(f"{output_dir}/all_real_masks.pt")
            all_pred_probs = torch.load(f"{output_dir}/all_pred_probs_dist.pt")  
            #with open(f"{output_dir}/fvv_image_pairs_dist.pkl", "rb") as f:
            #    fvv_image_pairs = pickle.load(f)              
        else:
            gen_features = np.load(f"{output_dir}/gen_features.npy")
            all_real_masks = torch.load(f"{output_dir}/all_real_masks.pt")
            all_pred_probs = torch.load(f"{output_dir}/all_pred_probs.pt")    
            #with open(f"{output_dir}/fvv_image_pairs.pkl", "rb") as f:
            #    fvv_image_pairs = pickle.load(f)                          


    # ----------------Calculate metrics----------------
    # print("fid")
    # fid = compute_fid(real_features, gen_features)
    # print("kid")
    # kid = compute_kid(real_features, gen_features)

    # all_real_masks = torch.cat(all_real_masks, dim=0)
    # all_pred_probs = torch.cat(all_pred_probs, dim=0)
    # print("ap")
    # ap = compute_ap(all_real_masks, all_pred_probs)

    # fvv = compute_fvv(fvv_image_pairs)

    #------------------------------------------------------------------
    # # Metric 계산
    # fid = compute_fid(real_features, gen_features)
    # print(f"FID Score: {fid}")

    # kid = compute_kid(real_features, gen_features)
    # print(f"KID Score: {kid}")

    ap = compute_ap(all_real_masks, all_pred_probs, device='cuda')
    print(f"Average Precision (AP): {ap}")

    if isinstance(all_pred_probs, list):
        all_pred_probs = torch.cat(all_pred_probs, dim=0)  # Concatenate along the batch dimension
    if isinstance(all_real_masks, list):
        all_real_masks = torch.cat(all_real_masks, dim=0)  # Concatenate along the batch dimension
    # Compute predicted masks using argmax over the class dimension (dim=1)
    predicted_masks = torch.argmax(all_pred_probs, dim=1)  # Shape: (B, 512, 512)
    true_masks = all_real_masks.squeeze(1)  # Remove the singleton channel dimension, Shape: (B, 512, 512)
    # Compute mIoU
    num_classes = 19  # Update according to your dataset
    miou = compute_miou(predicted_masks, true_masks, num_classes)
    print(f"Mean IoU (mIoU): {miou}")
    print('\n')

    #fvv = compute_fvv(fvv_image_pairs)
    #print(f"Face Verification Value (FVV): {fvv}")

    #-------------------------------------------------

        # np.save(f"{output_dir}/real_features.npy", real_features)
        # np.save(f"{output_dir}/gen_features.npy", gen_features)

        # torch.save(all_real_masks, f"{output_dir}/all_real_masks.pt")
        # torch.save(all_pred_probs, f"{output_dir}/all_pred_probs.pt")

        # with open(f"{output_dir}/fvv_image_pairs.pkl", "wb") as f:
        #     pickle.dump(fvv_image_pairs, f)

        # fid = compute_fid(real_features, gen_features)
        # kid = compute_kid(real_features, gen_features)

        # all_real_masks = torch.cat(all_real_masks, dim=0)
        # all_pred_probs = torch.cat(all_pred_probs, dim=0)
        # ap = compute_ap(all_real_masks, all_pred_probs)

        # fvv = compute_fvv(fvv_image_pairs)

    #-------------------------------------------------

    # print(f"FID: {fid}, KID: {kid}, AP: {ap}, FVV: {fvv}")
