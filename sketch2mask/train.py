import os
import torch.optim as optim
from network import UNet, UNetMod, UNetStyleDistil
from dataset import SketchSegmentationDataset, SketchSegmentationDistilDataset
from loss import MultiClassDiceLoss
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import time

def convert_mask_to_rgb(mask):
    # color mapping
    color_map = [
        [255, 255, 255],  # White
        [204, 0, 0],      # Dark Red
        [76, 153, 0],     # Olive Green
        [204, 204, 0],    # Mustard Yellow
        [51, 51, 255],    # Bright Blue
        [204, 0, 204],    # Purple
        [0, 255, 255],    # Cyan
        [255, 204, 204],  # Light Pink
        [102, 51, 0],     # Brown
        [255, 0, 0],      # Red
        [102, 204, 0],    # Lime Green
        [255, 255, 0],    # Yellow
        [0, 0, 153],      # Navy Blue
        [0, 0, 204],      # Royal Blue
        [255, 51, 153],   # Hot Pink
        [0, 204, 204],    # Teal
        [0, 51, 0],       # Dark Green
        [255, 153, 51],   # Orange
        [0, 204, 0]       # Green
    ]

    # Array to Image
    _, height, width = mask.shape
    rgb_tensor = torch.zeros((height, width, 3), dtype=torch.uint8)

    for value, color in enumerate(color_map):
        rgb_tensor[(mask == value).squeeze()] = torch.tensor(color).to(torch.uint8)

    return rgb_tensor.permute(2, 0, 1)


def main(data_root, sketch_fname, mask_fname, num_classes, style_fname=None):
    # set seed
    torch.manual_seed(42)

    # Initialize Wandb
    wandb.init(project='Sketch-to-Segmentation', config={
        'learning_rate': 5e-5,
        'batch_size': 8,
        'num_epochs': 2,
        'optimizer': 'Adam',
        'loss_function': 'CrossEntropyLoss+DICELoss',
        'image_size': 512,
        'val_split': 0.1,
        'patience': 5,
        'min_delta': 0.0001
    })

    # Initialize Hyperparameter
    config = wandb.config
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    val_split = config.val_split
    patience = config.patience
    min_delta = config.min_delta

    # Create Output Directories
    output_root = os.path.join('./results', str(int(time.time())))

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    sample_image_path = os.path.join(output_root, 'sample_images')
    if not os.path.exists(sample_image_path):
        os.makedirs(sample_image_path)

    inf_rgb_mask_path = os.path.join(output_root, 'inference', 'rgb_mask')
    if not os.path.exists(inf_rgb_mask_path):
        os.makedirs(inf_rgb_mask_path)

    inf_pred_mask_path = os.path.join(output_root, 'inference', 'pred_mask')
    if not os.path.exists(inf_pred_mask_path):
        os.makedirs(inf_pred_mask_path)

    # Data Transform
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
    ])

    # Set variables
    use_distill = True if style_fname is not None else False

    # Datasets
    train_data_root = os.path.join(data_root, 'train')
    if not use_distill:
        full_dataset = SketchSegmentationDataset(
            sketch_dir=os.path.join(train_data_root, sketch_fname),
            mask_dir=os.path.join(train_data_root, mask_fname),
            transform=transform,
        )
    else:
        full_dataset = SketchSegmentationDistilDataset(
            sketch_dir=os.path.join(train_data_root, sketch_fname),
            mask_dir=os.path.join(train_data_root, mask_fname),
            style_dir=os.path.join(train_data_root, style_fname),
            transform=transform,
        )
        

    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetStyleDistil(in_channels=1, out_channels=num_classes, init_features=64, bottleneck_features=512).to(device)

    CELoss = nn.CrossEntropyLoss()
    DiceLoss = MultiClassDiceLoss()
    dice_weight = 1
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if use_distill:
        MSELoss = nn.MSELoss()
        mse_weight = 1

    # Wandb setting
    wandb.watch(model, log='all', log_freq=10)

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    # Train!
    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered. Training stopped.")

        model.train()
        epoch_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train", ncols=100)
        for batch_idx, batch in enumerate(train_loader_tqdm):
            if not use_distill:
                sketches, masks = batch
            else:
                sketches, masks, style_vectors = batch
                style_vectors = style_vectors.to(device)
                style_vectors = style_vectors.squeeze().permute(0, 2, 1)

            sketches = sketches.to(device)
            masks = masks.to(device)

            # Make mask into LongTensor and modify dimentions
            masks = masks.to(torch.long).squeeze(1)  # (N, 1, H, W) -> (N, H, W)

            # Forward
            if not sketches.is_contiguous():
                print(sketches)
            outputs, style_embed = model(sketches)  # (N, num_classes, H, W)
            ce_loss = CELoss(outputs, masks)
            dice_loss = DiceLoss(outputs, masks)
            if use_distill:
                mse_loss = MSELoss(style_embed, style_vectors)
                loss = ce_loss + (dice_weight * dice_loss) + (mse_weight * mse_loss)
            else:
                loss = ce_loss + dice_weight * dice_loss

            # Backward & Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # tqdm
            train_loader_tqdm.set_postfix(loss=loss.item())

            # Wandb logging
            wandb.log({'Train Loss': loss.item(), 'Epoch': epoch + 1})

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Train Loss: {avg_train_loss:.4f}")

        # Evaluation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", ncols=100)
            for batch_idx, batch in enumerate(val_loader_tqdm):
                if not use_distill:
                    sketches, masks = batch
                else:
                    sketches, masks, style_vectors = batch
                    style_vectors = style_vectors.to(device)
                    style_vectors = style_vectors.squeeze().permute(0, 2, 1)

                sketches = sketches.to(device)
                masks = masks.to(device)

                masks = masks.to(torch.long).squeeze(1)  # (N, H, W)

                outputs, style_embed = model(sketches)
                ce_loss = CELoss(outputs, masks)
                dice_loss = DiceLoss(outputs, masks)
                if use_distill:
                    mse_loss = MSELoss(style_embed, style_vectors)
                    loss = ce_loss + (dice_weight * dice_loss) + (mse_weight * mse_loss)
                else:
                    loss = ce_loss + dice_weight * dice_loss

                val_loss += loss.item()

                # tqdm
                val_loader_tqdm.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}")

        # Wandb Logging
        wandb.log({'Validation Loss': avg_val_loss, 'Epoch': epoch + 1})

        # Check for Early Stopping
        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            # Save best model
            torch.save(model.state_dict(), os.path.join(output_root, 'best_unet_model.pth'))
            # wandb.save('best_unet_model.pth')
            print(f"Validation loss improved. Model saved at epoch {epoch+1}.")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"Validation loss did not improve for {patience} consecutive epochs. Early stopping.")
            early_stop = True

        # Save sample image (Eval set)
        with torch.no_grad():
            sample_batch = next(iter(val_loader))
            if not use_distill:
                sample_sketches, sample_masks = sample_batch
            else:
                sample_sketches, sample_masks, sample_style_vectors = sample_batch
                sample_style_vectors = sample_style_vectors.to(device)
                sample_style_vectors = sample_style_vectors.squeeze().permute(0, 2, 1)

            sample_sketches = sample_sketches.to(device)
            sample_masks = sample_masks.to(device)

            outputs, style_embed = model(sample_sketches)
            preds = torch.argmax(outputs, dim=1).unsqueeze(1).float()

            # Save Sketch, GT Mask, Pred Mask
            for i in range(batch_size):
                images_to_save = torch.stack(
                    [
                        sample_sketches.cpu()[i].repeat(3, 1, 1) * 255, 
                        convert_mask_to_rgb(sample_masks.unsqueeze(1).cpu()[i]), 
                        convert_mask_to_rgb(preds.cpu()[i])
                    ]
                    , dim=0)
                grid = utils.make_grid(images_to_save, nrow=3, normalize=True)
                utils.save_image(grid, os.path.join(output_root, 'sample_images', f'epoch_{epoch+1}_batch_{i}.png'))

                # Wandb logging
                wandb.log({
                    'Sample Images': [wandb.Image(grid, caption=f'sample_images/epoch_{epoch+1}_batch_{i} (Validation)')]
                })    

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_root, 'final_unet_model.pth'))
    # wandb.save('final_unet_model.pth')  # upload model to Wandb

    # Test dataset & Dataloader
    test_data_root = os.path.join(data_root, 'test')
    if use_distill:
        test_dataset = SketchSegmentationDataset(
            sketch_dir=os.path.join(test_data_root, sketch_fname),
            mask_dir=os.path.join(test_data_root, mask_fname),
            transform=transform,
        )
    else:
        test_dataset = SketchSegmentationDataset(
            sketch_dir=os.path.join(test_data_root, sketch_fname),
            mask_dir=os.path.join(test_data_root, mask_fname),
            transform=transform,
        )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load Best Model
    model.load_state_dict(os.path.join(output_root, 'final_unet_model.pth'))

    # Inference
    model.eval()
    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc="Inference", ncols=100)
        for idx, (sketches, masks) in enumerate(test_loader_tqdm):
            sketches = sketches.to(device)
            masks = masks.to(device)

            outputs, style_embed = model(sketches)
            preds = torch.argmax(outputs, dim=1).unsqueeze(1).float()

            # Save Output
            images_to_save = torch.stack(
                [
                    sketches.cpu()[0].repeat(3, 1, 1) * 255, 
                    convert_mask_to_rgb(masks.unsqueeze(1).cpu()[0]), 
                    convert_mask_to_rgb(preds.cpu()[0])
                ]
                , dim=0)
            grid = utils.make_grid(images_to_save, nrow=3, normalize=True)
            utils.save_image(grid, os.path.join(inf_rgb_mask_path, f'result_{idx}.png'))

            # Save Original Mask
            Image.fromarray(preds.cpu()[0].squeeze().numpy().astype(np.uint8), mode='L').save(os.path.join(inf_pred_mask_path, f'mask_{idx}.png'))

            # Log test results (first 10 results)
            if idx < 10:
                sketch_img = transforms.ToPILImage()(sketches.cpu().squeeze(0))
                mask_img = transforms.ToPILImage()(masks.cpu().squeeze(0))
                output_img = transforms.ToPILImage()(preds.cpu().squeeze(0))
            
                wandb.log({
                    f'Test Image {idx+1}': [
                        wandb.Image(sketch_img, caption='Sketch'),
                        wandb.Image(mask_img, caption='Ground Truth'),
                        wandb.Image(output_img, caption='Prediction')
                    ]
                })

    # Finish Wandb
    wandb.finish()


if __name__ == '__main__':
    main(data_root='../data/cat',
         sketch_fname='afhqcat_edge_pidinet',
         mask_fname='afhqcat_seg_6c_no_nose',
         num_classes=6,
         style_fname='afhqcat_seg_w_plus'
         )
