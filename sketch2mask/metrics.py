import numpy as np
import torch
from scipy.linalg import sqrtm
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import face_recognition
from tqdm import tqdm


# Frechet Inception Distance (FID)
def compute_fid(real_features, generated_features):
    mu_real, mu_gen = np.mean(real_features, axis=0), np.mean(generated_features, axis=0)
    cov_real = np.cov(real_features, rowvar=False)
    cov_gen = np.cov(generated_features, rowvar=False)
    mean_diff = np.sum((mu_real - mu_gen) ** 2)
    cov_sqrt = sqrtm(cov_real @ cov_gen)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    fid_score = mean_diff + np.trace(cov_real + cov_gen - 2 * cov_sqrt)
    return fid_score

# Kernel Inception Distance (KID)
def compute_kid(real_features, generated_features):
    num_subsets = 100
    max_subset_size = 1000
    feature_dim = real_features.shape[1]

    m = min(min(real_features.shape[0], generated_features.shape[0]), max_subset_size)

    kid_score = 0

    for _ in range(num_subsets):
        x = generated_features[np.random.choice(generated_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]

        a = (x @ x.T / feature_dim + 1) ** 3 + (y @ y.T / feature_dim + 1) ** 3
        b = (x @ y.T / feature_dim + 1) ** 3
        kid_score += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m

    kid_score /= num_subsets * m

    return kid_score


# Compute Average Precision (AP)
# def compute_ap(real_masks, pred_probs, device='cpu'):
#     """
#     Computes the Average Precision (AP) score.

#     Args:
#         real_masks (list of torch.Tensor): List of ground truth masks (B, H, W).
#         pred_probs (list of torch.Tensor): List of predicted probability masks (B, C, H, W).
#         device (str): Device to use for computation (e.g., 'cpu' or 'cuda').

#     Returns:
#         float: Average Precision (AP) score.
#     """
#     ap_score = 0.0
#     total_samples = 0

#     # Use tqdm to display progress
#     for real, pred in tqdm(zip(real_masks, pred_probs), desc="Computing AP", total=len(real_masks), ncols=100):
#         real = real.to(device)
#         pred = pred.to(device)

#         # Flatten masks and prepare for sklearn metrics
#         real_np = real.cpu().numpy().astype(int).flatten() 
#         pred_np = pred.cpu().numpy().squeeze()
#         pred_np = pred_np.transpose(1, 2, 0).reshape(-1, pred_np.shape[0])  

#         # Convert ground truth to one-hot for multi-class AP calculation
#         real_np = label_binarize(real_np, classes=list(range(pred_np.shape[1])))

#         # Compute AP for the current sample
#         sample_ap = average_precision_score(real_np, pred_np, average="micro")
#         ap_score += sample_ap 
#         total_samples += 1 

#     # Calculate mean AP
#     if total_samples > 0:
#         ap_score /= total_samples

#     return ap_score

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
        print(f"Computing IoU of class {num_class}")
        
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
