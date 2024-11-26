import torch
import torch.nn as nn

class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        preds: Model predictions (B, C, H, W) - logits or probabilities
        targets: Target labels (B, H, W) or (B, C, H, W)
        """
        # Apply Softmax to convert logits into probabilities
        preds = torch.softmax(preds, dim=1)

        if targets.dim() == 3:  # Shape is (B, H, W)
            # Convert target labels to one-hot encoding
            targets = nn.functional.one_hot(targets, num_classes=preds.size(1))
            targets = targets.permute(0, 3, 1, 2).float()  # Shape becomes (B, C, H, W)

        # Flatten the tensors: (B, C, H, W) -> (B, C, H*W)
        preds = preds.view(preds.size(0), preds.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        # Calculate intersection and union for each class
        intersection = (preds * targets).sum(dim=2)
        union = preds.sum(dim=2) + targets.sum(dim=2)

        # Compute Dice score for each class
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Compute the average Dice Loss across all classes
        dice_loss = 1 - dice_score.mean()

        return dice_loss

# Example usage
if __name__ == "__main__":
    # Model output: (B, C, H, W) - C is the number of classes
    preds = torch.randn(8, 3, 256, 256)  # Batch=8, Classes=3, Size=256x256
    # Target labels: (B, H, W) or (B, C, H, W)
    targets = torch.randint(0, 3, (8, 256, 256))  # Class labels (0, 1, 2)

    criterion = MultiClassDiceLoss()
    loss = criterion(preds, targets)

    print(f"Dice Loss: {loss.item():.4f}")