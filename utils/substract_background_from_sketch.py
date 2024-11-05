import os
import cv2
import numpy as np
from glob import glob
import shutil

"""
This script performs background subtraction on sketch images obtained from the 
MM-CelebA-HQ-Dataset (https://github.com/IIGROUP/MM-CelebA-HQ-Dataset) by utilizing 
mask images from CelebAMask-HQ (https://github.com/switchablenorms/CelebAMask-HQ?tab=readme-ov-file).
It combines and dilates all mask images per ID to create a background mask, then 
removes the background from the corresponding sketch images and saves the final result.
"""

# Define paths for the mask and sketch data
mask_base_path = '/Users/soomin/Desktop/LSMA/Project/dataset/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
sketch_base_path = '/Users/soomin/Desktop/LSMA/Project/dataset/sketch'

# Setting parameters for dilation
dilation_iterations = 3
kernel_size = 5
param_info = f'kernel_size{str(kernel_size)}_dilation_iterations{str(dilation_iterations)}'
output_path = f'/Users/soomin/Desktop/LSMA/Project/dataset/{param_info}'
os.makedirs(output_path, exist_ok=True)

# Function to combine masks and apply dilation
def get_combined_dilated_mask(mask_paths, dilation_iterations=3,kernel_size=5):
    """
    Combines multiple mask images into one and applies dilation to expand 
    the mask region, making it slightly larger to ensure all background is covered.
    
    Parameters:
    - mask_paths (list): List of paths to individual mask images.
    - dilation_iterations (int): Number of iterations for the dilation process.
    - kernel_size (int): Size of the kernel used for dilation.

    Returns:
    - np.ndarray: Dilated and combined mask image.
    """    
    combined_mask = None
    
    # Combine all masks for the same ID
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not load mask image {mask_path}")
            continue
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Return empty mask if no valid mask images were found
    if combined_mask is None:
        return np.zeros((512, 512), dtype=np.uint8)
    
    # Apply dilation to expand the mask region
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=dilation_iterations)
    return dilated_mask

# Process each subfolder and mask image
for subfolder_name in os.listdir(mask_base_path):
    subfolder_path = os.path.join(mask_base_path, subfolder_name)
    if os.path.isdir(subfolder_path):
        # Get all mask files in this subfolder
        mask_files = glob(os.path.join(subfolder_path, "*.png"))
        
        # Process each unique ID within this subfolder
        unique_ids = set(os.path.basename(f).split('_')[0] for f in mask_files)
        
        for unique_id in unique_ids:
            # Format the ID to match the sketch filename (e.g., "00000" -> "0")
            sketch_id = str(int(unique_id))  # Remove leading zeros
            
            # Get all mask paths for this ID
            mask_paths = [f for f in mask_files if f.startswith(os.path.join(subfolder_path, unique_id))]
            
            # Combine masks and apply dilation
            combined_dilated_mask = get_combined_dilated_mask(mask_paths,dilation_iterations=dilation_iterations,kernel_size=kernel_size)
            
            # Invert the mask to use white as background
            inverted_mask = cv2.bitwise_not(combined_dilated_mask)
            
            # Load corresponding sketch image
            sketch_path = os.path.join(sketch_base_path, f"{sketch_id}.jpg")
            if os.path.exists(sketch_path):
                sketch_image = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
                
                # Resize mask to match sketch size if needed
                if combined_dilated_mask.shape != sketch_image.shape:
                    combined_dilated_mask = cv2.resize(combined_dilated_mask, (sketch_image.shape[1], sketch_image.shape[0]))
                    inverted_mask = cv2.resize(inverted_mask, (sketch_image.shape[1], sketch_image.shape[0]))
                
                # Ensure both mask and sketch are of type CV_8U
                combined_dilated_mask = combined_dilated_mask.astype(np.uint8)
                inverted_mask = inverted_mask.astype(np.uint8)
                sketch_image = sketch_image.astype(np.uint8)
                
                # Create a white background
                white_background = np.full_like(sketch_image, 255, dtype=np.uint8)
                
                # Combine the sketch with the white background using the inverted mask
                foreground = cv2.bitwise_and(sketch_image, sketch_image, mask=combined_dilated_mask)
                background = cv2.bitwise_and(white_background, white_background, mask=inverted_mask)
                final_output = cv2.bitwise_or(foreground, background)
                
                # Resize the final output to 512x512
                final_output = cv2.resize(final_output, (512, 512))
                
                # Save the processed image
                output_file_path = os.path.join(output_path, f"{sketch_id}.png")
                cv2.imwrite(output_file_path, final_output)
                #print(f"Processed and saved: {output_file_path}")
            else:
                print(f"Sketch image not found for ID: {sketch_id}")
