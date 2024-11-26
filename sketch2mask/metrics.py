import numpy as np
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
    mu_real, mu_gen = np.mean(real_features, axis=0), np.mean(generated_features, axis=0)
    cov_real = np.cov(real_features, rowvar=False)
    cov_gen = np.cov(generated_features, rowvar=False)
    mean_diff = np.sum((mu_real - mu_gen) ** 2)
    cov_sqrt = sqrtm(cov_real @ cov_gen)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real
    kid_score = mean_diff + np.trace(cov_real + cov_gen - 2 * cov_sqrt)
    return kid_score

# Compute Average Precision (AP)
def compute_ap(real_masks, pred_probs):
    real_masks = real_masks.cpu().numpy().flatten()
    pred_probs = pred_probs.cpu().numpy().flatten()
    ap_score = average_precision_score(real_masks, pred_probs)
    return ap_score

def compute_fvv(image_pairs):
    distances = []

    for img1_path, img2_path in image_pairs:
        img1 = face_recognition.load_image_file(img1_path)
        img2 = face_recognition.load_image_file(img2_path)

        encodings1 = face_recognition.face_encodings(img1)
        encodings2 = face_recognition.face_encodings(img2)

        if len(encodings1) > 0 and len(encodings2) > 0:
            embedding1 = encodings1[0] 
            embedding2 = encodings2[0]

            distance = np.linalg.norm(embedding1 - embedding2)
            distances.append(distance)
        else:
            print(f"Warning: No face detected in one of the images ({img1_path}, {img2_path})")

    if len(distances) == 0:
        raise ValueError("No valid face pairs for comparison.")
    
    return np.mean(distances)
