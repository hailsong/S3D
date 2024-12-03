import numpy as np
import torch
from scipy.linalg import sqrtm
from sklearn.metrics import average_precision_score
import face_recognition


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
def compute_ap(real_masks, pred_probs, batch_size=500, device='cpu'):
    ap_score = 0.0
    total_samples = 0

    current_batch_real = []
    current_batch_pred = []
    current_batch_size = 0

    for real, pred in zip(real_masks, pred_probs):
        current_batch_real.append(real)
        current_batch_pred.append(pred)
        current_batch_size += real.shape[0]

        if current_batch_size >= batch_size:
            batch_real = torch.cat(current_batch_real, dim=0).to(device)
            batch_pred = torch.cat(current_batch_pred, dim=0).to(device)

            batch_real = batch_real.cpu().numpy().astype(int).flatten()
            batch_pred = batch_pred.cpu().numpy()
            batch_pred = batch_pred.reshape(-1, batch_pred.shape[1])

            ap_score += average_precision_score(batch_real, batch_pred, average="micro") * len(batch_real)
            total_samples += len(batch_real)

            current_batch_real = []
            current_batch_pred = []
            current_batch_size = 0

    # Process remaining data
    if current_batch_real:
        batch_real = torch.cat(current_batch_real, dim=0).to(device)
        batch_pred = torch.cat(current_batch_pred, dim=0).to(device)

        batch_real = batch_real.cpu().numpy().astype(int).flatten()
        batch_pred = batch_pred.cpu().numpy()
        batch_pred = batch_pred.reshape(-1, batch_pred.shape[1])

        ap_score += average_precision_score(batch_real, batch_pred, average="micro") * len(batch_real)
        total_samples += len(batch_real)

    ap_score /= total_samples
    return ap_score


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
