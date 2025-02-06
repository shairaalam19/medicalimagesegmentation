import torch
import torch.nn.functional as F
from torch import nn, optim

from models.LossFunctions import IoULoss

# IoU Loss
class EdgeSegmentationLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2, lambda_boundary=2.0, bce=True, composite=False, iou=False, dice=False):
        super(EdgeSegmentationLoss, self).__init__()
        # Composite Loss Parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_boundary = lambda_boundary
        # Loss functions 
        self.bce = bce
        self.composite = composite
        self.iou = iou
        self.dice = dice

        # Initialize loss functions
        self.bce_loss = nn.BCELoss()

    def forward(self, inputs, targets, smooth=1e-6):
        if self.bce: 
            return self.bce_loss(inputs, targets)
        elif self.composite: 
            return CompositeLoss(inputs, targets, self.alpha, self.beta, self.gamma, self.lambda_boundary)
        elif self.iou:
            return IoULoss(inputs, targets, smooth=1e-6)
        elif self.dice:
            return DiceLoss(inputs, targets, smooth=1e-6)
        else: 
            print("No loss function provided. Using BCE Loss.")
            return nn.BCELoss()