import torch
import torch.nn.functional as F

# Calculates overlap between input and target images 
def IoULoss(inputs, targets, smooth=1e-6): 
    # Flatten the tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    # Calculate IoU
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    # Return IoU loss (1 - IoU)
    return 1 - iou

# Balances class imbalance
def DiceLoss(inputs, targets, smooth=1e-6):
    """
    Dice Loss is effective in handling class imbalance, common in edge segmentation where edge pixels are sparse compared to non-edge pixels.
    """
    dice_loss = 1 - (2 * (inputs * targets).sum() + 1e-6) / ((inputs**2).sum() + (targets**2).sum() + 1e-6)
    
    # Return Dice loss (1 - Dice score)
    return dice_loss

# Focuses on edge pixels to improve boundary detection
def WeightedCELoss(inputs, targets, lambda_boundary): 
    """
    This loss focuses on boundary pixels. You can assign higher weights to boundary regions to handle sparsity. 
    
    The gradients are amplified near boundaries due to higher weights, improving the model's ability to predict edge structures.
    """
    # Weighted Cross-Entropy Loss
    weights = 1 + lambda_boundary * targets  # Dynamic weighting for edges
    ce_loss = -torch.mean(weights * (targets * torch.log(inputs + 1e-6) + (1 - targets) * torch.log(1 - inputs + 1e-6)))

    return ce_loss

# Guides intermediate feature maps for multi-resolution consistency
def DeepSupervisionLoss(targets, intermediate_preds=None):
    """
    Deep supervision encourages the network to produce consistent boundary predictions across multiple resolutions.
    """
    # Deep Supervision Loss
    ds_loss = 0.0
    if intermediate_preds is not None:
        for pred in intermediate_preds:
            ds_loss += 1 - (2 * (pred * targets).sum() + 1e-6) / ((pred**2).sum() + (targets**2).sum() + 1e-6)
        
        ds_loss /= len(intermediate_preds)
    
    return ds_loss

def CompositeLoss(inputs, targets, alpha, beta, gamma, lambda_boundary, intermediate_preds=None): 
    # Composite Loss
    total_loss = alpha * DiceLoss(inputs, targets, smooth=1e-6) + beta * WeightedCELoss(inputs, targets, lambda_boundary) + gamma * DeepSupervisionLoss(targets, intermediate_preds)

    return total_loss