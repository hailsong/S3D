import argparse
from PIL import Image
import os

def preprocess_image(input_image_path, output_image_path):
    """
    Preprocess an image for inference:
    - Convert to grayscale
    - Resize to 512x512
    - Save as a PNG
    """
    # Load the image
    image = Image.open(input_image_path).convert('L')  # Convert to grayscale

    # Resize the image to 512x512
    resized_image = image.resize((512, 512))

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_image_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save the processed image
    resized_image.save(output_image_path, format="PNG")
    print(f"Preprocessed image saved at: {output_image_path}")

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Preprocess an image for inference")
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output_image", type=str, required=True, help="Path to save the processed image (including filename)")
    args = parser.parse_args()

    # Preprocess the image
    preprocess_image(args.input_image, args.output_image)
