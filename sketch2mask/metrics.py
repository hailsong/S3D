import dlib
import face_recognition
import lpips
import numpy as np
import os
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from tqdm import tqdm
from scipy.linalg import sqrtm
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

# Function to load images
def load_images(image_folder, transform, device='cuda', recursive=False):
    images = []
    image_files = []

    if recursive:
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    image_files.append(os.path.join(root, file))
    else:
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                       if f.lower().endswith(('png', 'jpg', 'jpeg'))]

    if len(image_files) == 0:
        raise ValueError(f"Error: No image files found in {image_folder}.")

    print(f"Loading {len(image_files)} images from {image_folder}...")

    for img_path in tqdm(image_files, desc="Loading Images", unit="image"):
        try:
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            images.append(img)
        except Exception as e:
            print(f"Warning: Error processing {img_path} - {e}")

    if len(images) == 0:
        raise ValueError(f"Error: Unable to load images from {image_folder}. Check for corrupted images.")

    return images, image_files

# Function to load images and extract features
def extract_features(image_folder, inception, transform, device='cuda', recursive=False):
    images, image_files = load_images(image_folder, transform, device, recursive)

    features = []
    inception.eval()

    with torch.no_grad():
        for img in tqdm(images, desc="Extracting Features", unit="image"):
            feat = inception(img)
            features.append(feat.cpu().numpy())

    if len(features) == 0:
        raise ValueError(f"Error: Unable to extract features from {image_folder}. Check for corrupted images.")

    return np.concatenate(features, axis=0)  # Feature matrix of shape (N, feature_dim)

# Frechet Inception Distance (FID)
def calculate_fid(real_images_folder, gen_images_folder, device='cuda'):
    """
    real_images_folder: Root directory containing real images
    gen_images_folder: Root directory containing generated images 
    device: 'cuda' or 'cpu'
    """    
    # Load InceptionV3 model
    inception = inception_v3(weights="IMAGENET1K_V1", transform_input=False).to(device)
    inception.fc = torch.nn.Identity()  # Remove fully connected layer (feature extraction)
    inception.eval()

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Input size for Inception model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # Extract feature vectors from real and generated images
    real_features = extract_features(real_images_folder, inception, transform, device, recursive=True)  
    gen_features = extract_features(gen_images_folder, inception, transform, device, recursive=False) 

    # Compute mean and covariance
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    # Compute FID Score
    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real @ sigma_gen, disp=False)  # Compute square root of covariance matrix product

    # If result contains complex numbers, take only the real part
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)

    return fid_score

# Kernel Inception Distance (KID)
def calculate_kid(real_images_folder, gen_images_folder, device='cuda', num_subsets=100, max_subset_size=1000):
    """
    real_images_folder: Root directory containing real images
    gen_images_folder: Root directory containing generated images 
    device: 'cuda' or 'cpu'
    num_subsets: Number of subsets randomly sampled for KID calculation
    max_subset_size: Maximum number of samples per subset
    """
    # Load InceptionV3 model (used for feature extraction instead of classification)
    inception = inception_v3(weights="IMAGENET1K_V1", transform_input=False).to(device)
    inception.fc = torch.nn.Identity()  # Remove the last fully connected layer
    inception.eval()

    # Define transform suitable for Inception model input
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Extract features from real and generated images
    real_features = extract_features(real_images_folder, inception, transform, device, recursive=True)  
    gen_features = extract_features(gen_images_folder, inception, transform, device, recursive=False) 

    # Determine subset size for KID calculation
    num_real = real_features.shape[0]
    num_gen  = gen_features.shape[0]
    m = min(num_real, num_gen, max_subset_size)
    if m < 2:
        raise ValueError("Not enough samples to compute KID.")

    n = real_features.shape[1]  # Feature vector dimension (typically 2048)

    # KID uses a polynomial kernel k(x, y) = (xᵀy/n + 1)³
    # Below is an unbiased estimator calculation approach based on the given code snippet.
    t = 0.0
    for _ in tqdm(range(num_subsets), desc="Computing KID subsets"):
        # Randomly sample m elements from each set without replacement
        idx_real = np.random.choice(num_real, m, replace=False)
        idx_gen  = np.random.choice(num_gen, m, replace=False)
        x = gen_features[idx_gen]   # (m, n)
        y = real_features[idx_real] # (m, n)

        # a = (K_xx + K_yy), b = K_xy
        a = (np.dot(x, x.T) / n + 1) ** 3 + (np.dot(y, y.T) / n + 1) ** 3
        b = (np.dot(x, y.T) / n + 1) ** 3

        # Exclude diagonal elements (self-kernel values)
        subset_estimate = (a.sum() - np.trace(a)) / (m - 1) - 2 * b.sum() / m
        t += subset_estimate

    # Final KID is the average of all subset estimates, divided by m
    kid_score = t / (num_subsets * m)
    return kid_score

# SG Diversity (using LPIPS)
# (https://arxiv.org/pdf/2007.03780)
def calculate_sg_diversity(gen_images_root, net='vgg', device='cuda', n_of_pairs=10):
    """
    gen_images_root: Root directory containing subfolders per same instance,
                     each subfolder contains 6 generated images.
    device: 'cuda' or 'cpu'
    """
    lpips_model = lpips.LPIPS(net=net).to(device) # AlexNet, VGG or SqueezeNet
    lpips_model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    subfolders = [os.path.join(gen_images_root, d) for d in os.listdir(gen_images_root)
                  if os.path.isdir(os.path.join(gen_images_root, d))]

    #if len(subfolders) != 1000:
    #    raise ValueError(f"Expected 1000 subfolders, found {len(subfolders)}")

    all_lpips_means = []

    for folder in tqdm(subfolders, desc="Processing subfolders"):
        images = load_images(folder, transform, device, recursive=False)
        
        n_of_images = len(images)
        #if len(images) != 6:
        #    raise ValueError(f"Each subfolder must contain exactly 6 images, found {len(images)} in {folder}")

        pairs = [(i, j) for i in range(n_of_images) for j in range(i+1, n_of_images)]
        sampled_pairs = random.sample(pairs, 10)

        lpips_values = []
        with torch.no_grad():
            for (i, j) in sampled_pairs:
                dist = lpips_model(images[i], images[j]).item()
                lpips_values.append(dist)

        folder_mean_lpips = np.mean(lpips_values)
        all_lpips_means.append(folder_mean_lpips)

    sg_diversity = np.mean(all_lpips_means)

    return sg_diversity

# FVV Identity (Face Verification Value)
# (https://arxiv.org/pdf/2007.03780)
def calculate_fvv_identity(gen_images_root, n_of_pairs=10):
    """
    gen_images_root: Root directory containing subfolders per same instance,
                     each subfolder contains 15 generated views of the same person.
    """
    subfolders = [os.path.join(gen_images_root, d) for d in os.listdir(gen_images_root)
                  if os.path.isdir(os.path.join(gen_images_root, d))]

    #if len(subfolders) != 1000:
    #    raise ValueError(f"Expected 1000 subfolders, found {len(subfolders)}")

    all_identity_means = []

    for folder in tqdm(subfolders, desc="Processing subfolders"):
        image_files = sorted([os.path.join(folder, f) for f in os.listdir(folder)
                              if f.lower().endswith(('png', 'jpg', 'jpeg'))])

        n_of_images = len(image_files)
        #if len(image_files) != 15:
        #    raise ValueError(f"Each subfolder must contain 15 images, found {len(image_files)} in {folder}")

        embeddings_list = []  
        for img_path in image_files:
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            if len(encodings) == 0:
                raise ValueError(f"No face detected in image {img_path}")
            embedding = encodings[0]
            embedding = np.array(embedding)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings_list.append(embedding)

        pairs = [(i, j) for i in range(n_of_images) for j in range(i+1, n_of_images)]
        sampled_pairs = random.sample(pairs, n_of_pairs)

        distances = []
        for i, j in sampled_pairs:
            dist = np.linalg.norm(embeddings_list[i] - embeddings_list[j])
            distances.append(dist)

        folder_mean_distance = np.mean(distances)
        all_identity_means.append(folder_mean_distance)

    fvv_identity = np.mean(all_identity_means)

    return fvv_identity


# Compute Average Precision (AP)
def compute_ap(real_masks, pred_probs, device='cpu'):
    """
    Computes the Average Precision (AP) score using PyTorch for GPU compatibility.

    Args:
        real_masks (list of torch.Tensor): List of ground truth masks (B, H, W).
        pred_probs (list of torch.Tensor): List of predicted probability masks (B, C, H, W).
        device (str): Device to use for computation (e.g., 'cpu' or 'cuda').

    Returns:
        float: Average Precision (AP) score.
    """
    ap_score = 0.0
    total_samples = 0

    # Use tqdm to display progress
    for real, pred in tqdm(zip(real_masks, pred_probs), desc="Computing AP", total=len(real_masks), ncols=100):
        # Move data to the specified device
        real = real.to(device)
        pred = pred.to(device)

        # Flatten masks for processing
        real_flat = real.view(-1).long()  # Flatten ground truth
        pred_flat = pred.permute(0, 2, 3, 1).reshape(-1, pred.size(1))  # Flatten predictions

        # Convert ground truth to one-hot encoding
        real_one_hot = torch.nn.functional.one_hot(real_flat, num_classes=pred.size(1)).float()

        # Compute precision-recall curve using Torch
        precision, recall, _ = precision_recall_curve_torch(real_one_hot, pred_flat, device)

        # Compute AP for the current sample
        sample_ap = torch.trapz(precision, recall).item()  # Compute area under the curve
        ap_score += sample_ap
        total_samples += 1

    # Calculate mean AP
    if total_samples > 0:
        ap_score /= total_samples

    return ap_score


def precision_recall_curve_torch(real_one_hot, pred_probs, device):
    """
    Computes precision-recall curve for each class using Torch.

    Args:
        real_one_hot (torch.Tensor): Ground truth one-hot encoding (N, C).
        pred_probs (torch.Tensor): Predicted probabilities (N, C).
        device (str): Device to use for computation (e.g., 'cpu' or 'cuda').

    Returns:
        torch.Tensor: Precision values (C,).
        torch.Tensor: Recall values (C,).
        torch.Tensor: Thresholds (C,).
    """
    precision_list = []
    recall_list = []
    thresholds_list = []

    for class_idx in range(real_one_hot.size(1)):
        real = real_one_hot[:, class_idx]
        pred = pred_probs[:, class_idx]

        # Sort by predicted probabilities
        sorted_indices = torch.argsort(pred, descending=True)
        real = real[sorted_indices]
        pred = pred[sorted_indices]

        # Compute true positives and false positives
        tp = torch.cumsum(real, dim=0)
        fp = torch.cumsum(1 - real, dim=0)

        # Precision and recall
        precision = tp / (tp + fp + 1e-8)  # Add epsilon for numerical stability
        recall = tp / (real.sum() + 1e-8)

        thresholds = pred

        precision_list.append(precision)
        recall_list.append(recall)
        thresholds_list.append(thresholds)

    # Stack results for all classes
    precision = torch.stack(precision_list, dim=0).mean(dim=0)
    recall = torch.stack(recall_list, dim=0).mean(dim=0)
    thresholds = torch.stack(thresholds_list, dim=0).mean(dim=0)

    return precision, recall, thresholds


# Compute Face Verification Value (FVV)
def compute_fvv(image_pairs):
    distances = []

    for img1, img2 in image_pairs:
        # Convert PIL.Image to numpy.ndarray if necessary
        if not isinstance(img1, np.ndarray):
            img1 = np.array(img1)
        if not isinstance(img2, np.ndarray):
            img2 = np.array(img2)

        img2 = np.stack((img2,) * 3, axis=-1).astype(np.uint8)

        encodings1 = face_recognition.face_encodings(img1)
        encodings2 = face_recognition.face_encodings(img2)

        if len(encodings1) > 0 and len(encodings2) > 0:
            embedding1 = encodings1[0]
            embedding2 = encodings2[0]

            distance = np.linalg.norm(embedding1 - embedding2)
            distances.append(distance)

    if len(distances) == 0:
        raise ValueError("No valid face pairs for comparison.")
    
    return np.mean(distances)

# Compute Mean IoU
def compute_miou(pred_masks, true_masks, num_classes):
    """
    Computes the mean Intersection over Union (mIoU) across all classes.

    Args:
        pred_masks (torch.Tensor): Predicted segmentation masks (B, H, W).
        true_masks (torch.Tensor): Ground truth segmentation masks (B, H, W).
        num_classes (int): Number of classes.

    Returns:
        miou (float): Mean IoU across all classes.
    """
    iou_per_class = []

    for cls in range(num_classes):
        #print(f"Computing IoU of class {num_classes}")
        
        # Per-class masks
        pred_cls = (pred_masks == cls)
        true_cls = (true_masks == cls)

        # Intersection and Union
        intersection = (pred_cls & true_cls).sum().item()
        union = (pred_cls | true_cls).sum().item()

        if union == 0:
            iou = float('nan')  # Ignore this class if no pixels are present
        else:
            iou = intersection / union
        iou_per_class.append(iou)

    # Compute mean IoU, ignoring NaNs
    miou = np.nanmean(iou_per_class)
    return miou
