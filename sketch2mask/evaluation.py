import argparse
import metrics
import os
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image Generation Metrics")
    parser.add_argument('--real_images_folder', type=str, required=True, help='Path to real images for FID/KID.')
    parser.add_argument('--gen_images_folder', type=str, required=True, help='Path to generated images for FID/KID.')
    parser.add_argument('--sg_images_folder', type=str, required=True, help='Path for SG diversity images (1000 subfolders × 6 styles).')
    parser.add_argument('--fvv_images_folder', type=str, required=True, help='Path for FVV images (1000 subfolders × 15 viewpoints).')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu.')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for saving results.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Calculating FID...")
    fid_score = metrics.calculate_fid(args.real_images_folder, args.gen_images_folder, device=args.device)
    print(f"FID Score: {fid_score:.4f}")

    print("Calculating KID...")
    kid_score = metrics.calculate_kid(args.real_images_folder, args.gen_images_folder, device=args.device)
    print(f"KID Score: {kid_score:.4f}")

    print("Calculating SG Diversity...")
    sg_diversity = metrics.calculate_sg_diversity(args.sg_images_folder, device=args.device)
    print(f"SG Diversity (LPIPS): {sg_diversity:.4f}")

    print("Calculating FVV Identity...")
    fvv_identity = metrics.calculate_fvv_identity(args.fvv_images_folder)
    print(f"FVV Identity: {fvv_identity:.4f}")

    # Save results to file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    result_path = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(result_path, "w") as f:
        f.write(f"FID Score: {fid_score:.4f}\n")
        f.write(f"KID Score: {kid_score:.4f}\n")
        f.write(f"SG Diversity (LPIPS): {sg_diversity:.4f}\n")
        f.write(f"FVV Identity: {fvv_identity:.4f}\n")
    result_path = os.path.join(args.output_dir, "evaluation_results.pkl")
    results = {
        'FID Score': fid_score,
        'KID Score': kid_score,
        'SG Diversity (LPIPS)': sg_diversity,
        'FVV Identity': fvv_identity
    }
    with open(result_path, "wb") as f:
        pickle.dump(results, f)


    print(f"Results saved to {result_path}")